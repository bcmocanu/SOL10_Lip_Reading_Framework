import faulthandler
faulthandler.enable()

import os
import sys
import argparse

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from transformers import AutoTokenizer

from .inference import (
    Compose,
    Normalize,
    CenterCrop,
    MyVideoLoader,
    my_apply_to_sample,
    my_move_to_cuda,
    get_rng_state,
    set_rng_state,
    set_torch_seed,
    my_setup_task,
    load_model_ensemble_and_task,
)

def processOneVideos(videoPath):
    llm_path = './Models/llama'
    av_romanizer_path = './Models/romanizer/checkpoint_best.pt'
    avsr_path = './Models/avsr/checkpoint_best.pt'
    tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
    model_override_cfg = {
        'model': {
            'llm_path': llm_path,
            'av_romanizer_path': av_romanizer_path,
        }
    }
    model, saved_cfg, task = load_model_ensemble_and_task(
        avsr_path, model_override_cfg, strict=False
    )
    np.random.seed(1)
    set_torch_seed(1)
    model.eval()
    model.cuda()
    model.half()
    myVideoLoader = MyVideoLoader(llm_path)
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
    print(f'\nVideo: {videoPath}\nPredictie   : {predictie}\n')
    os._exit(0)

def main():
    parser = argparse.ArgumentParser(description='Lip reading - demo!')
    parser.add_argument(
        'video_path',
        type=str,
        help='Calea catre fisierul video (ex: /path/to/video.mp4)',
    )
    args = parser.parse_args()
    print('RUN lip_reading_demo_module - V03 14.11.2026 - Test pe video')
    processOneVideos(args.video_path)
    return

