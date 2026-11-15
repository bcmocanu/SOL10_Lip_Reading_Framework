"""Verify the installed dependencies and small real CUDA/media operations."""
from pathlib import Path
import importlib.metadata
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check_versions():
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('Este necesar Python 3.10 in LipsWebEnv.')
    if Path(sys.prefix).resolve() != (ROOT / 'LipsWebEnv').resolve() or sys.prefix == sys.base_prefix:
        raise RuntimeError('Verificarea nu ruleaza in mediul propriu LipsWebEnv.')
    for name in ('requirements.lock', 'torch.lock'):
        for line in (ROOT / 'framework' / name).read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            package, expected = line.strip().split('==')
            actual = importlib.metadata.version(package)
            if actual != expected:
                raise RuntimeError(f'{package}: instalat {actual}; necesar {expected}. Ruleaza ./setupFramework.')


def check_gpu():
    import torch
    import torchvision
    import bitsandbytes as bnb
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA nu este accesibil din LipsWebEnv. Driverul nu a fost modificat.')
    if torch.version.cuda != '12.1':
        raise RuntimeError(f'Runtime CUDA neasteptat: {torch.version.cuda}')
    with torch.inference_mode():
        x = torch.ones((16, 16), device='cuda')
        assert float((x @ x).sum()) == 4096.0
        boxes = torch.tensor([[0., 0., 8., 8.], [1., 1., 7., 7.]], device='cuda')
        scores = torch.tensor([0.9, 0.7], device='cuda')
        assert torchvision.ops.nms(boxes, scores, 0.5).numel() == 1
        layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16,
                                 quant_type='nf4', compress_statistics=True).cuda()
        result = layer(torch.ones((2, 64), device='cuda', dtype=torch.bfloat16))
        assert torch.isfinite(result).all()
        torch.cuda.synchronize()
    print(f'CUDA/PyTorch/torchvision/NF4: OK pe {torch.cuda.get_device_name(0)}.', flush=True)


def check_full():
    import cv2
    import numpy as np
    import imageio.v2 as imageio
    import imageio_ffmpeg
    from core.lipreading import _insert_dependency_paths
    _insert_dependency_paths(ROOT / 'core/lip_reading_demo_module')
    from src import AV_RomanizerModel, Zero_AVSR
    from backend.app.main import app
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(str(ROOT / 'core/lip_reading_demo_module/Models/llama'), local_files_only=True)
    with tempfile.TemporaryDirectory(dir=ROOT / '.framework/tmp') as directory:
        first = Path(directory) / 'ffmpeg.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                        '-i', 'color=c=black:s=32x32:r=25', '-t', '0.12', '-an', '-c:v',
                        'libx264', '-pix_fmt', 'yuv420p', '-y', str(first)], check=True, timeout=30)
        second = Path(directory) / 'imageio.mp4'
        with imageio.get_writer(str(second), fps=25, codec='libx264', pixelformat='yuv420p') as writer:
            for _ in range(3):
                writer.append_data(np.zeros((32, 32, 3), dtype=np.uint8))
        for path in (first, second):
            capture = cv2.VideoCapture(str(path))
            try:
                ok, frame = capture.read()
                assert ok and frame.shape[:2] == (32, 32)
            finally:
                capture.release()
    print('Importuri API/AV-HuBERT, tokenizer local si video H.264: OK.', flush=True)
    print('Ponderile complete ale modelului se incarca la prima transcriere, nu in acest test.', flush=True)


if __name__ == '__main__':
    try:
        check_versions()
        if '--versions' not in sys.argv:
            check_gpu()
        if '--full' in sys.argv:
            check_full()
    except Exception as exc:
        print(f'EROARE runtime: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
