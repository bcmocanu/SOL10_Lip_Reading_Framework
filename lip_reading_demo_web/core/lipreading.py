from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

class LipReadingError(RuntimeError):
    """Raised when lip-reading inference cannot be completed."""

CORE_DIR = Path(__file__).resolve().parent
DEFAULT_MODULE_DIR = CORE_DIR / "lip_reading_demo_module"

def _resolve_module_dir() -> Path:
    """Return the folder containing avhubert, fairseq, src and Models."""
    env_path = os.environ.get("SOL10_LIPREADING_MODULE_DIR")
    module_dir = Path(env_path).expanduser().resolve() if env_path else DEFAULT_MODULE_DIR
    return module_dir

def _insert_dependency_paths(module_dir: Path) -> None:
    """Use the bundled packages and register checkpoint classes before loading."""
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
    import src  # Registers model/task classes and their checkpoint aliases.


class Compose:
    def __init__(self, preprocess: list[Any]) -> None:
        self.preprocess = preprocess

    def __call__(self, sample: Any) -> Any:
        for transform in self.preprocess:
            sample = transform(sample)
        return sample

class Normalize:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        return (frames - self.mean) / self.std

class CenterCrop:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        _, height, width = frames.shape
        target_height, target_width = self.size
        if height < target_height or width < target_width:
            raise LipReadingError(
                f"Video frames are too small for center crop: "
                f"got {width}x{height}, need at least {target_width}x{target_height}."
            )

        delta_w = int(round((width - target_width) / 2.0))
        delta_h = int(round((height - target_height) / 2.0))
        return frames[:, delta_h : delta_h + target_height, delta_w : delta_w + target_width]

class VideoLoader:
    """Load the preprocessed mouth video and prepare model input tensors."""

    def __init__(
        self,
        llm_path: Path,
        device: torch.device,
        image_mean: float = 0.421,
        image_std: float = 0.165,
        image_crop_size: int = 88,
    ) -> None:
        self.device = device
        self.llm_tokenizer = AutoTokenizer.from_pretrained(str(llm_path), local_files_only=True)
        self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id
        self.llm_tokenizer.sep_token_id = self.llm_tokenizer.unk_token_id

        self.transform = Compose(
            [
                Normalize(0.0, 255.0),
                CenterCrop((image_crop_size, image_crop_size)),
                Normalize(image_mean, image_std),
            ]
        )

    @staticmethod
    def load_video(video_path: Path) -> np.ndarray:
        capture = cv2.VideoCapture(str(video_path))
        frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
        finally:
            capture.release()

        if not frames:
            raise LipReadingError(f"No frames could be read from video: {video_path}")

        return np.stack(frames)

    def get_video_sample(self, video_path: Path) -> dict[str, Any]:
        video_feats = self.load_video(video_path)
        video_feats = self.transform(video_feats)
        video_feats = np.expand_dims(video_feats, axis=-1)
        video_feats = torch.from_numpy(video_feats.astype(np.float32))

        prompt = (
            "Given romanized transcriptions extracted from audio-visual materials, "
            "back-transliterate them into the original script of Romanian. Input:"
        )
        text_tokens = self.llm_tokenizer(prompt, return_tensors="pt").input_ids[0]

        padding_mask = torch.BoolTensor(1, len(video_feats)).fill_(False)

        video_feats = video_feats.unsqueeze(0)
        video_feats = video_feats.permute((0, 4, 1, 2, 3)).contiguous()
        video_feats = video_feats.to(torch.half)

        source = {
            "audio": None,
            "video": video_feats,
            "instruction": [text_tokens],
        }
        net_input = {"source": source, "padding_mask": padding_mask}
        return move_to_device(net_input, self.device)

def apply_to_sample(function: Any, sample: Any) -> Any:
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(value: Any) -> Any:
        if torch.is_tensor(value):
            return function(value)
        if isinstance(value, dict):
            return {key: _apply(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_apply(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_apply(item) for item in value)
        if isinstance(value, set):
            return {_apply(item) for item in value}
        return value

    return _apply(sample)

def move_to_device(sample: Any, device: torch.device) -> Any:
    def _move_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_tensor, sample)

def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def setup_task(cfg: Any, module_dir: Path, **kwargs: Any) -> Any:
    _insert_dependency_paths(module_dir)
    from src import Zero_AVSRConfig, Zero_AVSR_TrainingTask

    default_config = Zero_AVSRConfig()
    merged_cfg = OmegaConf.merge(default_config, cfg)
    merged_cfg.__dict__["_parent"] = cfg.__dict__["_parent"]
    OmegaConf.set_struct(merged_cfg, True)
    return Zero_AVSR_TrainingTask.setup_task(merged_cfg, **kwargs)

def _torch_load_checkpoint(filename: Path) -> dict[str, Any]:
    """Load a PyTorch checkpoint while remaining compatible with multiple PyTorch versions."""
    try:
        return torch.load(str(filename), map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        return torch.load(str(filename), map_location=torch.device("cpu"))

def load_model_ensemble_and_task(
    filename: Path,
    module_dir: Path,
    arg_overrides: Optional[Dict[str, Any]] = None,
    strict: bool = True,
) -> tuple[Any, Any, Any]:
    if arg_overrides is None:
        arg_overrides = {}

    _insert_dependency_paths(module_dir)
    state = _torch_load_checkpoint(filename)

    from omegaconf import _utils

    old_primitive = _utils.is_primitive_type
    _utils.is_primitive_type = lambda _: True
    try:
        state["cfg"] = OmegaConf.create(state["cfg"])
    finally:
        _utils.is_primitive_type = old_primitive

    OmegaConf.set_struct(state["cfg"], True)
    cfg = state["cfg"]

    cfg["model"]["llm_path"] = arg_overrides["model"]["llm_path"]
    cfg["model"]["av_romanizer_path"] = arg_overrides["model"]["av_romanizer_path"]

    task = setup_task(cfg.task, module_dir)
    task.load_state_dict(state["task_state"])

    model = task.build_model(cfg.model)
    model.load_state_dict(state["model"], strict=strict, model_cfg=cfg.model)

    return model, cfg, task

class LipReadingEngine:
    """GPU-only lip-reading inference engine for the SOL10 web backend."""

    def __init__(
        self,
        module_dir: Path | None = None,
        avsr_checkpoint_name: str = "checkpoint_best.pt",
        romanizer_checkpoint_name: str = "checkpoint_best.pt",
    ) -> None:
        self.module_dir = module_dir or _resolve_module_dir()
        self.models_dir = self.module_dir / "Models"
        self.llm_path = self.models_dir / "llama"
        self.av_romanizer_path = self.models_dir / "romanizer" / romanizer_checkpoint_name
        self.avsr_path = self.models_dir / "avsr" / avsr_checkpoint_name
        self.device = torch.device("cuda:0")

        self._validate_environment()
        _insert_dependency_paths(self.module_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.llm_path), local_files_only=True)
        model_override_cfg = {
            "model": {
                "llm_path": str(self.llm_path),
                "av_romanizer_path": str(self.av_romanizer_path),
            }
        }

        self.model, self.saved_cfg, self.task = load_model_ensemble_and_task(
            self.avsr_path,
            self.module_dir,
            model_override_cfg,
            strict=False,
        )

        set_torch_seed(1)
        self.model.eval()
        self.model.cuda()
        self.model.half()
        self.video_loader = VideoLoader(self.llm_path, self.device)

    def _validate_environment(self) -> None:
        if not torch.cuda.is_available():
            raise LipReadingError("CUDA is not available. This lip-reading engine is configured for GPU inference.")

        required_paths = [
            self.module_dir,
            self.module_dir / "avhubert",
            self.module_dir / "fairseq",
            self.module_dir / "src" / "__init__.py",
            self.module_dir / "src" / "model_stage1.py",
            self.module_dir / "src" / "model_stage2.py",
            self.models_dir / "avhubert" / "large_vox_iter5.pt",
            self.models_dir,
            self.llm_path,
            self.av_romanizer_path,
            self.avsr_path,
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            formatted = "\n".join(f"- {path}" for path in missing)
            raise LipReadingError(
                "The lip-reading module is not fully installed. Missing paths:\n"
                f"{formatted}\n\n"
                "Expected structure: core/lip_reading_demo_module/{avhubert,fairseq,src,Models}."
            )

    @torch.no_grad()
    def predict(self, video_path: Path) -> str:
        if not video_path.exists():
            raise LipReadingError(f"Mouth video does not exist: {video_path}")

        sample = self.video_loader.get_video_sample(video_path)
        best_hypothesis = self.model.generate(
            num_beams=2,
            temperature=0.3,
            use_speech_embs=True,
            use_roman_toks=False,
            **sample,
        )

        decoded = self.tokenizer.batch_decode(
            best_hypothesis,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not decoded:
            raise LipReadingError("The lip-reading model returned no decoded output.")

        return decoded[0].strip()

_ENGINE: LipReadingEngine | None = None
_ENGINE_LOCK = threading.Lock()

def get_lipreading_engine() -> LipReadingEngine:
    """Load the model once and reuse it across lip-reading requests."""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = LipReadingEngine()
    return _ENGINE

def run_lipreading(mouth_video_path: Path) -> str:
    """Run lip-reading inference over a preprocessed mouth-region video."""
    engine = get_lipreading_engine()
    return engine.predict(Path(mouth_video_path))
