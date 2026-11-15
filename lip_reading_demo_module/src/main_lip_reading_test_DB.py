import faulthandler
faulthandler.enable()

import os
import sys
import argparse

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
import json
from datetime import datetime

import numpy as np
from rouge_score import rouge_scorer
from bert_score import score
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
    predictOneVideo,
)
from .metrics import (
    levenshtein_distance,
    wer,
    cer,
    evaluate_metrics,
    compute_scores,
    normalizeaza_text,
)

def citeste_text_din_json(video_path):
    json_path = Path(video_path).with_name(
        Path(video_path).stem.replace('_NoAudio', '_text') + '.json'
    )
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('text')

def processVideosFromDB(dbPath):
    llm_path = './Models/llama'
    av_romanizer_path = './Models/romanizer/checkpoint_best.pt'
    avsr_path = './Models/avsr/checkpoint_best.pt'
    bert_score_path = './Models/bert_score'
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
    videosPath = [str(p) for p in Path(dbPath).rglob('*_NoAudio.mp4')]
    references = []
    predictions = []
    lipResults = {}
    for videoPath in videosPath:
        PR_text = predictOneVideo(videoPath, myVideoLoader, model, tokenizer)
        GT_text = citeste_text_din_json(videoPath)
        print(f'\nVideo: {videoPath}\nGroundTruth: {GT_text}\nPredictie   : {PR_text}\n')
        PR_text_normalizat = normalizeaza_text(PR_text)
        GT_text_normalizat = normalizeaza_text(GT_text)
        predictions.append(PR_text_normalizat)
        references.append(GT_text_normalizat)
        lipResults[videoPath] = {
            'Groundtruth': GT_text,
            'Predictie': PR_text,
        }
    globalScores = compute_scores(references, predictions, bert_score_path)
    print('Rezultate globale:')
    print(json.dumps(globalScores, indent=4, ensure_ascii=False))
    lipResults = dict(sorted(lipResults.items()))
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / ('Rezultate_LipReading_Model_' + datetime.now().strftime('%d_%m_%Y_%H_%M_%S') + '.json')
    with open(result_path, 'w', encoding='utf-8') as outfile:
        json.dump(
            {'videos': lipResults, 'globalScores': globalScores},
            outfile,
            indent=4,
            ensure_ascii=False,
        )
    os._exit(0)

def main():
    parser = argparse.ArgumentParser(description='Lip reading - test on DB!')
    parser.add_argument(
        '--db_path',
        type=str,
        default='/media/user/S4A/SOL10/DB_V2/VALIDATION/',
        help='Calea catre folderul cu fisiere video (ex: /path/to/DB/)',
    )
    args = parser.parse_args()
    print('RUN lip_reading_demo_module - V03 14.11.2026 - Evaluare BD')
    processVideosFromDB(args.db_path)
    return

