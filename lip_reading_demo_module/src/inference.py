from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from .task_stage2 import Zero_AVSRConfig, Zero_AVSR_TrainingTask
from . import model_stage2

class Compose(object):

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class CenterCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round(w - tw) / 2.0)
        delta_h = int(round(h - th) / 2.0)
        frames = frames[:, delta_h:delta_h + th, delta_w:delta_w + tw]
        return frames

class MyVideoLoader:

    def __init__(
        self,
        llm_path: str,
        image_mean: float = 0.421,
        image_std: float = 0.165,
        image_crop_size: int = 88,
    ):
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id
        self.llm_tokenizer.sep_token_id = self.llm_tokenizer.unk_token_id
        self.transform = Compose([
            Normalize(0.0, 255.0),
            CenterCrop((image_crop_size, image_crop_size)),
            Normalize(image_mean, image_std),
        ])

    def load_video(self, videoPath):
        cap = cv2.VideoCapture(videoPath)
        frames = []
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
            else:
                break
        frames = np.stack(frames)
        return frames

    def getMyVideo(self, videoPath):
        video_feats = self.load_video(videoPath)
        video_feats = self.transform(video_feats)
        video_feats = np.expand_dims(video_feats, axis=-1)
        video_feats = torch.from_numpy(video_feats.astype(np.float32))
        txt_feats = self.llm_tokenizer(
            f'Given romanized transcriptions extracted from audio-visual materials, back-transliterate them into the original script of Romanian. Input:',
            return_tensors='pt',
        ).input_ids[0]
        text_instructions = [txt_feats]
        padding_mask = torch.BoolTensor(1, len(video_feats)).fill_(False)
        video_feats = video_feats.unsqueeze(0)
        video_feats = video_feats.permute((0, 4, 1, 2, 3)).contiguous()
        video_feats = video_feats.to(torch.half)
        source = {
            'audio': None,
            'video': video_feats,
            'instruction': text_instructions,
        }
        net_input = {
            'source': source,
            'padding_mask': padding_mask,
        }
        net_input = my_move_to_cuda(net_input)
        return net_input

def my_apply_to_sample(f, sample):
    if hasattr(sample, '__len__') and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple((_apply(x) for x in x))
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x
    return _apply(sample)

def my_move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        return tensor.to(device=device, non_blocking=True)
    return my_apply_to_sample(_move_to_cuda, sample)

def get_rng_state():
    state = {'torch_rng_state': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda_rng_state'] = torch.cuda.get_rng_state()
    return state

def set_rng_state(state):
    torch.set_rng_state(state['torch_rng_state'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda_rng_state'])

class set_torch_seed(object):

    def __init__(self, seed):
        assert isinstance(seed, int)
        self.rng_state = get_rng_state()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        set_rng_state(self.rng_state)

def my_setup_task(cfg, **kwargs):
    dc = Zero_AVSRConfig()
    merged_cfg = OmegaConf.merge(dc, cfg)
    merged_cfg.__dict__['_parent'] = cfg.__dict__['_parent']
    OmegaConf.set_struct(merged_cfg, True)
    task = Zero_AVSR_TrainingTask
    return task.setup_task(merged_cfg, **kwargs)

def load_model_ensemble_and_task(
    filename,
    arg_overrides: Optional[Dict[str, Any]] = None,
    strict=True,
):
    with open(filename, 'rb') as f:
        state = torch.load(f, map_location=torch.device('cpu'))
    from omegaconf import _utils
    old_primitive = _utils.is_primitive_type
    _utils.is_primitive_type = lambda _: True
    state['cfg'] = OmegaConf.create(state['cfg'])
    _utils.is_primitive_type = old_primitive
    OmegaConf.set_struct(state['cfg'], True)
    cfg = state['cfg']
    cfg['model']['llm_path'] = arg_overrides['model']['llm_path']
    cfg['model']['av_romanizer_path'] = arg_overrides['model']['av_romanizer_path']
    task = my_setup_task(cfg.task)
    task.load_state_dict(state['task_state'])
    model = task.build_model(cfg.model)
    model.load_state_dict(
        state['model'], strict=strict, model_cfg=cfg.model
    )
    return (model, cfg, task)

def predictOneVideo(videoPath, myVideoLoader, model, tokenizer):
    mySample = myVideoLoader.getMyVideo(videoPath)
    with torch.no_grad():
        best_hypo = model.generate(
            num_beams=2,
            temperature=0.3,
            use_speech_embs=True,
            use_roman_toks=False,
            **mySample,
        )
    best_hypo = tokenizer.batch_decode(
        best_hypo,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    predictie = best_hypo[0]
    return predictie
