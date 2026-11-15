#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import ExitStack
import logging
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, required=True, choices=(1, 2))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("train")
    try:
        from src.training_config import (read_settings, register_components, build_fairseq_config,
                                         check_model_files, local_path)
        from src.training_manifests import check_manifests, locked_directory
        from src.training_setup import check_python, check_cuda, ensure_native_batching

        check_python()
        settings = read_settings(args.stage)
        output = local_path(settings["paths"]["output_dir"])
        manifests = local_path(settings["paths"]["manifest_dir"])
        with ExitStack() as locks:
            locks.enter_context(locked_directory(output))
            locks.enter_context(locked_directory(manifests, exclusive=False))
            ensure_native_batching(build=False)
            register_components(args.stage)
            cfg = build_fairseq_config(settings, args.stage)
            check_model_files(cfg, args.stage)
            if args.stage == 2:
                romanizer_dir = Path(cfg.model.av_romanizer_path).parent
                if romanizer_dir == output:
                    raise ValueError("Romanizerul si rezultatele stage 2 trebuie sa fie in directoare distincte.")
                locks.enter_context(locked_directory(romanizer_dir, exclusive=False))
            check_cuda(cfg.distributed_training.device_id)
            logger.info("Stage %d: verificare manifeste pregatite...", args.stage)
            check_manifests(settings)
            last = output / "checkpoint_last.pt"
            if last.is_file():
                logger.info("Reluare completa: %s", last)
            elif cfg.checkpoint.finetune_from_model:
                logger.info("Fine-tuning nou: %s; optimizerul, schedulerul si contoarele pornesc de la zero.",
                            cfg.checkpoint.finetune_from_model)
            else:
                logger.info("Etapa noua din componentele preantrenate locale, fara checkpoint al etapei.")
            if args.stage == 2:
                logger.info("Romanizer ales: %s", cfg.model.av_romanizer_path)
            logger.info("Modalitati=%s; pad_audio=%s; acumulare=%s minibatchuri.",
                        list(cfg.task.modalities), cfg.task.pad_audio, list(cfg.optimization.update_freq))
            from src.training_runner import main as run_training
            run_training(cfg)
        return 0
    except KeyboardInterrupt:
        logger.warning("Antrenare intrerupta; urmatoarea rulare restaureaza ultimul checkpoint salvat.")
        return 130
    except ImportError as exc:
        logger.error("Dependenta indisponibila: %s. Rulati python setupFramework.py in mediul EnvZeroAVSR.", exc)
        return 2
    except Exception:
        logger.exception("Antrenarea s-a oprit.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
