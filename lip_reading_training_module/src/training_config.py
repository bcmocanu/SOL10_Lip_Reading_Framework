from __future__ import annotations

import copy
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "src" / "configs"
DICTIONARY_PATH = CONFIG_DIR / "dict.ltr.txt"
LANGUAGE_CODE = "ron"
LANGUAGE_NAME = "Romanian"
logger = logging.getLogger(__name__)

_PROFILES = {1: {'common': {'fp16': True,
                'seed': 1337,
                'log_format': 'json',
                'log_interval': 200,
                'empty_cache_freq': 10000,
                'cpu': False,
                'user_dir': None,
                'tensorboard_logdir': None,
                'wandb_project': None,
                'azureml_logging': False},
     'distributed_training': {'device_id': 0},
     'task': {'_name': 'av-roman-task',
              'is_s2s': False,
              'labels': ['ltr'],
              'sample_rate': 25,
              'max_sample_size': 600,
              'pad_audio': False,
              'random_crop': True,
              'normalize': True,
              'stack_order_audio': 4,
              'modalities': ['video', 'audio'],
              'image_aug': True,
              'max_trim_sample_size': '${task.max_sample_size}',
              'noise_prob': 0.0,
              'noise_wav': None,
              'time_mask': False,
              'random_erase': False},
     'model': {'_name': 'av_romanizer',
               'skip_masked': False,
               'skip_nomask': False,
               'modality_dropout': 0.5,
               'audio_dropout': 0.5,
               'modality_fuse': 'concat',
               'selection_type': 'same_seq',
               'masking_type': 'input',
               'mask_prob_image': 0.3,
               'mask_length_image': 5,
               'mask_prob_audio': 0.8,
               'mask_length_audio': 10,
               'encoder_layerdrop': 0.05,
               'dropout_input': 0.1,
               'dropout_features': 0.1,
               'dropout': 0.1,
               'attention_dropout': 0.1,
               'feature_grad_mult': 1.0,
               'activation_dropout': 0.0,
               'layer_norm_first': True,
               'encoder_embed_dim': 1024},
     'criterion': {'_name': 'ctc',
                   'zero_infinity': True,
                   'sentence_avg': False,
                   'post_process': 'letter',
                   'wer_kenlm_model': None,
                   'wer_lexicon': None,
                   'wer_lm_weight': 2.0,
                   'wer_word_score': -1.0,
                   'wer_args': None},
     'dataset': {'num_workers': 4,
                 'skip_invalid_size_inputs_valid_test': True,
                 'max_tokens': 1000,
                 'batch_size': None,
                 'required_batch_size_multiple': 8,
                 'train_subset': 'train',
                 'valid_subset': 'validation'},
     'optimization': {'max_epoch': 50,
                      'max_update': 0,
                      'clip_norm': 10.0,
                      'sentence_avg': False,
                      'update_freq': [8],
                      'lr': [0.0001]},
     'optimizer': {'_name': 'adam',
                   'adam_betas': '(0.9,0.98)',
                   'adam_eps': 1e-06,
                   'weight_decay': 0.01,
                   'use_old_adam': False},
     'lr_scheduler': {'_name': 'tri_stage',
                      'warmup_steps': 0,
                      'hold_steps': 0,
                      'decay_steps': 0,
                      'phase_ratio': [0.05, 0.15, 0.80],
                      'init_lr_scale': 0.01,
                      'final_lr_scale': 0.01},
     'checkpoint': {'restore_file': 'checkpoint_last.pt',
                    'finetune_from_model': None,
                    'reset_optimizer': False,
                    'reset_lr_scheduler': False,
                    'reset_meters': False,
                    'reset_dataloader': False,
                    'save_interval_updates': 5000,
                    'keep_interval_updates': 1,
                    'no_epoch_checkpoints': True,
                    'best_checkpoint_metric': 'uer'}},
 2: {'common': {'fp16': True,
                'seed': 1,
                'log_format': 'json',
                'log_interval': 100,
                'empty_cache_freq': 500,
                'cpu': False,
                'user_dir': None,
                'tensorboard_logdir': None,
                'wandb_project': None,
                'azureml_logging': False},
     'distributed_training': {'device_id': 0},
     'task': {'_name': 'zero-avsr-task',
              'is_s2s': True,
              'labels': ['wrd'],
              'normalize': True,
              'single_target': True,
              'fine_tuning': True,
              'stack_order_audio': 4,
              'tokenizer_bpe_name': 'sentencepiece',
              'max_sample_size': 500,
              'modalities': ['video'],
              'image_aug': True,
              'pad_audio': True,
              'random_crop': False,
              'zero_shot': True,
              'zero_shot_lang': 'no',
              'max_trim_sample_size': '${task.max_sample_size}',
              'noise_prob': 0.0,
              'noise_wav': None,
              'time_mask': False,
              'random_erase': False},
     'model': {'_name': 'zero-avsr',
               'apply_mask': False,
               'mask_selection': 'static',
               'mask_length': 10,
               'mask_other': 0,
               'mask_prob': 0.75,
               'mask_channel_selection': 'static',
               'mask_channel_length': 64,
               'mask_channel_other': 0,
               'mask_channel_prob': 0.5,
               'layerdrop': 0.1,
               'dropout': 0.0,
               'attention_dropout': 0.1,
               'feature_grad_mult': 1.0,
               'activation_dropout': 0.0,
               'target_modules': 'q_proj.k_proj.v_proj.o_proj',
               'llama_embed_dim': 3072,
               'lora_rank': 16,
               'lora_alpha': 32,
               'av_romanizer_embed_dim': 1024,
               'use_roman_tok': False,
               'use_speech_emb': True,
               'back_trans_prob': 0},
     'criterion': {'_name': 'zero-avsr-criterion'},
     'dataset': {'num_workers': 3,
                 'skip_invalid_size_inputs_valid_test': False,
                 'max_tokens': 1000,
                 'batch_size': None,
                 'required_batch_size_multiple': 8,
                 'train_subset': 'train',
                 'valid_subset': 'validation'},
     'optimization': {'max_epoch': 50,
                      'max_update': 130000,
                      'clip_norm': 1.0,
                      'sentence_avg': True,
                      'update_freq': [9],
                      'lr': [5e-05]},
     'optimizer': {'_name': 'adam',
                   'adam_betas': '(0.9,0.98)',
                   'adam_eps': 1e-08,
                   'weight_decay': 0.0,
                   'use_old_adam': False},
     'lr_scheduler': {'_name': 'cosine',
                      'warmup_updates': 500,
                      'warmup_init_lr': -1.0,
                      'min_lr': 1e-05,
                      't_mult': 1.0,
                      'lr_period_updates': -1.0,
                      'lr_shrink': 0.1},
     'checkpoint': {'restore_file': 'checkpoint_last.pt',
                    'finetune_from_model': None,
                    'reset_optimizer': False,
                    'reset_lr_scheduler': False,
                    'reset_meters': False,
                    'reset_dataloader': False,
                    'save_interval_updates': 30000,
                    'keep_interval_updates': 1,
                    'no_epoch_checkpoints': True,
                    'best_checkpoint_metric': 'wer'}}}


def local_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def read_settings(stage: int) -> dict[str, Any]:
    if stage not in (1, 2):
        raise ValueError("Etapa trebuie sa fie 1 sau 2.")
    filename = CONFIG_DIR / f"stage{stage}.yaml"
    with filename.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict) or not isinstance(settings.get("paths"), dict):
        raise ValueError(f"Configuratie invalida: {filename}")
    for key in ("db_root", "models_dir", "manifest_dir", "output_dir"):
        if not isinstance(settings["paths"].get(key), str) or not settings["paths"][key].strip():
            raise ValueError(f"{filename}: lipseste paths.{key}")
    if "initial_checkpoint" not in settings["paths"]:
        raise ValueError(f"{filename}: specificati paths.initial_checkpoint (cale sau null).")
    if stage == 2 and not settings["paths"].get("romanizer_checkpoint"):
        raise ValueError(f"{filename}: lipseste paths.romanizer_checkpoint")
    for section, values in settings.items():
        if section in ("paths", "data"):
            continue
        if section not in _PROFILES[stage] or not isinstance(values, dict):
            raise ValueError(f"{filename}: sectiune necunoscuta sau invalida: {section}")
        unknown = set(values) - set(_PROFILES[stage][section])
        if unknown:
            raise ValueError(f"{filename}: optiuni necunoscute in {section}: {sorted(unknown)}")
    return settings


def register_components(stage: int) -> None:
    import fairseq
    if not Path(fairseq.__file__).resolve().is_relative_to(PROJECT_ROOT):
        raise RuntimeError(f"Fairseq importat din alt proiect: {fairseq.__file__}")
    importlib.import_module("avhubert")
    for name in ("stage1_task", "stage1_model"):
        importlib.import_module(f"src.{name}")
    if stage == 2:
        for name in ("stage2_task", "stage2_model", "stage2_criterion"):
            importlib.import_module(f"src.{name}")


def build_fairseq_config(settings: dict[str, Any], stage: int):
    from omegaconf import OmegaConf, open_dict
    from fairseq.dataclass.configs import FairseqConfig
    from fairseq.dataclass.initialize import add_defaults

    raw = copy.deepcopy(_PROFILES[stage])
    for key, value in settings.items():
        if key not in ("paths", "data"):
            raw[key].update(value)
    cfg = OmegaConf.merge(OmegaConf.structured(FairseqConfig), OmegaConf.create(raw))
    add_defaults(cfg)
    paths = settings["paths"]
    models, manifests, output, db_root = (local_path(paths[key]) for key in
                                         ("models_dir", "manifest_dir", "output_dir", "db_root"))
    for protected in (models, db_root):
        if output == protected or output.is_relative_to(protected):
            raise ValueError("Directorul de rezultate trebuie separat de Models si de baza de date.")
    if manifests == models or manifests.is_relative_to(models) or manifests == output or manifests.is_relative_to(output):
        raise ValueError("Manifeste, modele de intrare si rezultate: folositi directoare separate.")
    expected_task, expected_model = (("av-roman-task", "av_romanizer") if stage == 1
                                    else ("zero-avsr-task", "zero-avsr"))
    if cfg.task._name != expected_task or cfg.model._name != expected_model:
        raise ValueError(f"Configuratia nu corespunde etapei {stage}.")
    with open_dict(cfg):
        cfg.task.data = cfg.task.label_dir = str(manifests)
        cfg.checkpoint.save_dir = str(output)
        cfg.model.w2v_path = str(models / "avhubert" / "large_vox_iter5.pt")
        cfg.common.user_dir = None
        cfg.common.log_file = str(output / "train.log")
        cfg.common.tensorboard_logdir = None
        cfg.common.wandb_project = None
        cfg.common.azureml_logging = False
        cfg.task.noise_prob, cfg.task.noise_wav = 0.0, None
        cfg.distributed_training.distributed_world_size = 1
        cfg.distributed_training.distributed_num_procs = 1
        cfg.distributed_training.nprocs_per_node = 1
        cfg.distributed_training.ddp_backend = "no_c10d"
        if stage == 2:
            cfg.model.llm_path = cfg.task.llm_path = str(models / "llama")
            cfg.model.av_romanizer_path = str(local_path(paths["romanizer_checkpoint"]))
        cfg.checkpoint.restore_file = "checkpoint_last.pt"
        # Fairseq resets optimization only for the first fine-tuning launch.
        start = paths.get("initial_checkpoint")
        last = output / "checkpoint_last.pt"
        cfg.checkpoint.finetune_from_model = str(local_path(start)) if start and not last.is_file() else None
        for key in ("reset_optimizer", "reset_lr_scheduler", "reset_meters", "reset_dataloader"):
            cfg.checkpoint[key] = False

    if cfg.common.tpu or cfg.common.cpu or cfg.common.bf16 or cfg.common.amp:
        raise ValueError("Acest modul foloseste o GPU CUDA si precizia FP16 a profilului de antrenare.")
    if not cfg.common.fp16 or cfg.optimization.use_bmuf or cfg.distributed_training.zero_sharding != "none":
        raise ValueError("Configuratie de optimizare nesuportata de profilul single-GPU.")
    if cfg.common.model_parallel_size != 1 or cfg.common.quantization_config_path:
        raise ValueError("Model parallelism si Fairseq quantizer nu sunt disponibile in acest modul.")
    if cfg.task.time_mask or cfg.task.random_erase:
        raise ValueError("time_mask si random_erase nu fac parte din profilul de antrenare.")
    if stage == 1 and list(cfg.task.modalities) != ["video", "audio"]:
        raise ValueError("Stage 1 foloseste video si audio.")
    if stage == 2 and (list(cfg.task.modalities) != ["video"] or cfg.task.zero_shot_lang != "no"
                       or cfg.model.back_trans_prob != 0):
        raise ValueError('Stage 2 foloseste video, zero_shot_lang="no", back_trans_prob=0.')
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        raise ValueError("Este necesara o limita pentru minibatchuri.")
    if not cfg.optimization.update_freq or any(int(n) < 1 for n in cfg.optimization.update_freq):
        raise ValueError("optimization.update_freq trebuie sa contina valori pozitive.")
    if int(cfg.dataset.num_workers) < 0 or int(cfg.distributed_training.device_id) < 0:
        raise ValueError("num_workers si device_id nu pot fi negative.")
    if cfg.task.max_sample_size is not None and cfg.task.max_sample_size < 2:
        raise ValueError("task.max_sample_size trebuie sa fie cel putin 2.")
    from .training_schedule import validate_schedule_config
    validate_schedule_config(cfg)
    return cfg


def check_model_files(cfg, stage: int, preparing: bool = False) -> bool:
    """Check local files without deserializing model weights. Return stage readiness."""
    required = [Path(cfg.model.w2v_path)]
    if cfg.checkpoint.finetune_from_model:
        required.append(Path(cfg.checkpoint.finetune_from_model))
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Checkpoint local lipsa sau gol: {path}")
    if not DICTIONARY_PATH.is_file():
        raise FileNotFoundError(DICTIONARY_PATH)
    if stage != 2:
        return True
    ready = Path(cfg.model.av_romanizer_path).is_file()
    if not ready and not preparing:
        raise FileNotFoundError(
            f"Romanizerul antrenat nu exista: {cfg.model.av_romanizer_path}\n"
            "Rulati mai intai: python train.py --stage 1"
        )
    if ready and Path(cfg.model.av_romanizer_path).stat().st_size == 0:
        raise ValueError(f"Checkpoint romanizer gol: {cfg.model.av_romanizer_path}")
    folder = Path(cfg.model.llm_path)
    for name in ("config.json", "tokenizer_config.json"):
        path = folder / name
        if not path.is_file():
            raise FileNotFoundError(f"Llama local incomplet: lipseste {path}")
        json.loads(path.read_text(encoding="utf-8"))
    if not any((folder / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise FileNotFoundError(f"Lipseste tokenizer.json sau tokenizer.model din {folder}")
    indices = [folder / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
               if (folder / name).is_file()]
    if indices:
        for index in indices:
            weights = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
            if not isinstance(weights, dict) or not weights:
                raise ValueError(f"Index Llama invalid: {index}")
            for shard in set(weights.values()):
                path = (folder / shard).resolve()
                if not path.is_relative_to(folder.resolve()) or not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(f"Fragment Llama lipsa sau invalid: {path}")
    elif not any((folder / name).is_file() and (folder / name).stat().st_size > 0
                 for name in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError(f"Lipsesc ponderile Llama din {folder}")
    return ready


def romanizer_identity(filename: str) -> dict[str, str]:
    """Stored inside stage 2 checkpoints to guard against swapping its frozen encoder."""
    path = Path(filename)
    logger.info("Verificare identitate romanizer: %s (citire checkpoint pentru SHA-256)...", path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Romanizerul s-a modificat in timpul citirii; opriti antrenarea stage 1 inainte de stage 2.")
    logger.info("Identitate romanizer verificata.")
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}
