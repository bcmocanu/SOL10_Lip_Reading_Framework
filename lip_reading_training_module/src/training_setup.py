from __future__ import annotations

import importlib
import importlib.machinery
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .training_config import (PROJECT_ROOT, read_settings, register_components,
                              build_fairseq_config, check_model_files, local_path)

logger = logging.getLogger(__name__)


def check_python() -> None:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Folositi Python 3.10 in mediul EnvZeroAVSR (compatibilitatea Fairseq/Hydra).")
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Acest proiect este configurat pentru Linux si o GPU NVIDIA CUDA.")


def check_dependencies() -> None:
    check_python()
    logger.info("Verificare biblioteci Python...")
    modules = ("numpy", "torch", "cv2", "soundfile", "scipy", "python_speech_features",
               "omegaconf", "hydra", "transformers", "peft", "bitsandbytes", "accelerate",
               "editdistance", "sentencepiece", "Cython", "setuptools")
    errors = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise RuntimeError("Biblioteci lipsa/incompatibile:\n" + "\n".join(errors)
                           + "\nActivati mediul de antrenare si verificati requirements.txt / README.md.")
    import torch
    logger.info("Biblioteci disponibile; Python %s, PyTorch %s.", sys.version.split()[0], torch.__version__)


def check_cuda(device_id: int):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA nu este disponibil. Activati mediul PyTorch CUDA si verificati driverul NVIDIA.")
    if not 0 <= device_id < torch.cuda.device_count():
        raise ValueError(f"device_id={device_id}: GPU inexistenta in lista vizibila CUDA.")
    torch.cuda.set_device(device_id)
    logger.info("GPU %d: %s", device_id, torch.cuda.get_device_name(device_id))
    return torch


def _extension_file() -> Path | None:
    base = PROJECT_ROOT / "fairseq" / "data" / "data_utils_fast"
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = Path(str(base) + suffix)
        if path.is_file():
            return path
    return None


def _probe_extension(path: Path) -> tuple[bool, str]:
    # A subprocess can safely reject a broken extension without loading it into setup.
    script = """
import importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location('data_utils_fast', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
a = np.arange(4, dtype=np.int64)
sizes = np.array([3, 4, 5, 8], dtype=np.int64)
batches = module.batch_by_size_vec(a, sizes, 16, -1, 1)
assert np.concatenate(batches).tolist() == [0, 1, 2, 3]
assert callable(module.batch_by_size_fn)
"""
    try:
        result = subprocess.run([sys.executable, "-c", script, str(path)], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False, "verificarea extensiei a depasit 60 de secunde"
    return result.returncode == 0, (result.stderr or result.stdout).strip()


def ensure_native_batching(build: bool = True) -> None:
    source = PROJECT_ROOT / "fairseq/data/data_utils_fast.pyx"
    extension = _extension_file()
    valid, detail = _probe_extension(extension) if extension else (False, "extensie absenta")
    if valid and extension.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        logger.info("Cython: OK (extensie deja compilata si verificata).")
        return
    if not build:
        raise RuntimeError("Extensia Cython lipseste, este incompatibila sau necesita recompilare. "
                           "Rulati: python setupFramework.py")
    logger.info("Cython: compilare necesara%s...", " (sursa este mai noua)" if valid else "")
    if not valid and extension:
        logger.info("Extensia existenta nu poate fi folosita: %s", detail[-1000:])
    if not any(shutil.which(c) for c in ("c++", "g++")) and not os.environ.get("CXX"):
        raise RuntimeError("Lipseste compilatorul C++. Instalati build-essential pentru compilarea Cython.")
    import numpy
    from Cython.Build import cythonize
    from setuptools import Distribution, Extension

    build_dir = PROJECT_ROOT / "src" / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        modules = cythonize(
            [Extension("fairseq.data.data_utils_fast", ["fairseq/data/data_utils_fast.pyx"],
                       include_dirs=[numpy.get_include()], language="c++",
                       extra_compile_args=["-O3", "-std=c++11"])],
            build_dir=str(build_dir / "cython"),
            compiler_directives={"language_level": "3"}, force=True,
        )
        distribution = Distribution({"name": "lip-reading-batching", "packages": [], "ext_modules": modules})
        command = distribution.get_command_obj("build_ext")
        command.build_lib = str(PROJECT_ROOT)
        command.build_temp = str(build_dir / "temp")
        command.inplace = False
        command.force = True
        command.ensure_finalized()
        command.run()
    finally:
        os.chdir(previous)
    extension = _extension_file()
    valid, detail = _probe_extension(extension) if extension else (False, "nu s-a generat extensia")
    if not valid:
        raise RuntimeError(f"Verificarea extensiei Cython a esuat: {detail}")
    importlib.invalidate_caches()
    logger.info("Cython: compilare si verificare reusite.")


def setup_framework() -> None:
    from .training_manifests import prepare_manifests

    check_dependencies()
    logger.info("Verificare extensie Cython...")
    ensure_native_batching()
    logger.info("Verificare configuratii pentru ambele etape...")
    settings = {stage: read_settings(stage) for stage in (1, 2)}
    outputs = [local_path(settings[s]["paths"]["output_dir"]) for s in (1, 2)]
    if outputs[0] == outputs[1] or any(a.is_relative_to(b) for a, b in (outputs, outputs[::-1])):
        raise ValueError("Etapele trebuie sa aiba directoare de rezultate distincte, nu incluse unul in celalalt.")
    configs = {}
    for stage in (1, 2):
        register_components(stage)
        cfg = build_fairseq_config(settings[stage], stage)
        configs[stage] = cfg
        check_cuda(cfg.distributed_training.device_id)
        logger.info("Stage %d: verificare fisiere modele locale...", stage)
        ready = check_model_files(cfg, stage, preparing=True)
        if not ready:
            logger.info("Stage 2: romanizerul antrenat va fi disponibil dupa stage 1: %s", cfg.model.av_romanizer_path)
    logger.info("Verificare tokenizer Llama local (fara incarcarea ponderilor)...")
    from transformers import AutoTokenizer
    folder = Path(configs[2].model.llm_path)
    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizerul Llama nu defineste eos_token_id.")
    llama_cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if llama_cfg.get("hidden_size") != configs[2].model.llama_embed_dim:
        raise ValueError("Dimensiunea Llama nu corespunde adaptorului (3072 pentru modelul configurat).")
    del tokenizer
    logger.info("Pregatire baza de date pentru stage 1 si stage 2...")
    prepare_manifests(settings)
    for output in outputs:
        output.mkdir(parents=True, exist_ok=True)
    logger.info("Framework pregatit. Urmatorul pas: python train.py --stage 1")
    logger.info("Au fost verificate fisierele si datele; ponderile complete vor fi incarcate la antrenare.")
