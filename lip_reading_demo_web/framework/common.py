"""Shared local paths, validation and isolated subprocess environment."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PATH = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'


class FrameworkError(RuntimeError):
    pass


def state_dir():
    return ROOT / '.framework'


def env_python():
    return ROOT / 'LipsWebEnv/bin/python'


def log(message):
    print(message, flush=True)


def confirm(message, expected='da'):
    if not sys.stdin.isatty():
        raise FrameworkError('Confirmarea necesita un terminal interactiv. Operatia nu a fost aprobata.')
    log(message)
    if input(f'Scrie "{expected}" pentru confirmare: ').strip() != expected:
        raise FrameworkError('Operatie anulata. Nu s-a aprobat modificarea.')


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise FrameworkError(f'Fisier invalid: {path}: {exc}') from exc


def atomic_json(path, value):
    path = Path(path)
    if path.is_symlink():
        raise FrameworkError(f'Nu suprascriu linkul simbolic: {path}')
    fd, name = tempfile.mkstemp(prefix='.write-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)



def local_cuda_library_dirs():
    """CUDA libraries bundled inside this installation's Python environment."""
    site = ROOT / 'LipsWebEnv/lib/python3.10/site-packages'
    candidates = [site / 'torch/lib']
    for name in ('nvjitlink', 'cusparse', 'cublas', 'cusolver', 'cuda_runtime',
                 'cuda_nvrtc', 'cuda_cupti', 'cudnn', 'cufft', 'curand', 'nccl', 'nvtx'):
        candidates.append(site / 'nvidia' / name / 'lib')
    return [directory for directory in candidates
            if directory.is_dir() and directory.resolve().is_relative_to(ROOT)
            and any(directory.glob('*.so*'))]


def local_cuda_library(name):
    return next((directory / name for directory in local_cuda_library_dirs()
                 if (directory / name).is_file()), None)


def child_env(gpu=None):
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(('PYTHON', 'CONDA', 'PIP_', 'CUDA_', 'HF_', 'BNB_', 'npm_', 'NPM_', 'SOL10_', 'LIPREADING_')) or name in (
            'VIRTUAL_ENV', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'NODE_PATH', 'NODE_OPTIONS',
            'NODE_ENV', 'TRANSFORMERS_CACHE', 'TRANSFORMERS_OFFLINE'):
            env.pop(name, None)
    local = state_dir()
    env.update({
        'PATH': f'{ROOT}/LipsWebEnv/bin:{ROOT}/.runtime/node/bin:{SYSTEM_PATH}',
        'PYTHONNOUSERSITE': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'PIP_CONFIG_FILE': os.devnull, 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
        'PIP_REQUIRE_VIRTUALENV': 'true', 'PIP_CACHE_DIR': str(local / 'cache/pip'),
        'npm_config_cache': str(local / 'cache/npm'),
        'npm_config_userconfig': str(local / 'npmrc'), 'npm_config_globalconfig': os.devnull,
        'XDG_CACHE_HOME': str(local / 'cache'), 'HF_HOME': str(local / 'cache/huggingface'),
        'TORCH_HOME': str(local / 'cache/torch'),
        'TORCH_EXTENSIONS_DIR': str(local / 'cache/torch-extensions'),
        'MPLCONFIGDIR': str(local / 'cache/matplotlib'),
        'NUMBA_CACHE_DIR': str(local / 'cache/numba'),
        'TMPDIR': str(local / 'tmp'), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'SOL10_LIPREADING_MODULE_DIR': str(ROOT / 'core/lip_reading_demo_module'),
    })
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        # Only the local CUDA runtime is visible to LipsWeb GPU processes.
        cuda_dirs = local_cuda_library_dirs()
        if cuda_dirs:
            env['LD_LIBRARY_PATH'] = ':'.join(map(str, cuda_dirs))
    return env


def run(args, *, capture=False, check=True, env=None, cwd=None, timeout=None):
    args = [str(x) for x in args]
    try:
        result = subprocess.run(args, cwd=cwd or ROOT, env=env or child_env(), text=True,
                                stdout=subprocess.PIPE if capture else None,
                                stderr=subprocess.PIPE if capture else None, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FrameworkError(f'Nu pot executa {args[0]}: {exc}') from exc
    if check and result.returncode:
        detail = (result.stderr or result.stdout or '').strip()[-3500:]
        raise FrameworkError(f'Comanda a esuat ({result.returncode}): {" ".join(args[:4])}\n{detail}')
    return result


def config():
    cfg = read_json(ROOT / 'framework/config.json')
    if not isinstance(cfg, dict) or set(cfg) != {'gpu', 'bind', 'port'}:
        raise FrameworkError('Configuratia accepta numai gpu, bind si port.')
    if type(cfg['gpu']) is not int or cfg['gpu'] < 0:
        raise FrameworkError('gpu trebuie sa fie un index intreg nenegativ.')
    if type(cfg['port']) is not int or not 1024 <= cfg['port'] <= 65435:
        raise FrameworkError('port trebuie sa fie intre 1024 si 65435.')
    try:
        ipaddress.IPv4Address(cfg['bind'])
    except (ValueError, TypeError) as exc:
        raise FrameworkError('bind trebuie sa fie o adresa IPv4.') from exc
    return cfg


def platform_check(setup=False):
    if sys.platform != 'linux' or os.uname().machine != 'x86_64':
        raise FrameworkError('Este necesar Linux x86_64.')
    if os.geteuid() == 0:
        raise FrameworkError('Ruleaza comanda fara sudo; instalarea componentelor lipsa cere acord separat.')
    if sys.version_info < (3, 10):
        raise FrameworkError('Comenzile necesita Python de sistem >= 3.10 (Ubuntu 22.04 sau mai nou).')
    if setup:
        release = {}
        for line in Path('/etc/os-release').read_text().splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                release[key] = value.strip('"')
        if release.get('ID') != 'ubuntu' or int(release.get('VERSION_ID', '0').split('.')[0]) < 22:
            raise FrameworkError('Instalarea este destinata Ubuntu 22.04/24.04 sau mai nou, x86_64.')
    marker = read_json(ROOT / 'framework/package.json')
    if marker.get('application') != 'lip-reading-module' or marker.get('format') != 1:
        raise FrameworkError('Pachet Lip reading module nerecunoscut.')


def machine_id():
    value = Path('/etc/machine-id').read_text().strip()
    if not value:
        raise FrameworkError('Nu pot identifica masina: /etc/machine-id este gol.')
    return hashlib.sha256(value.encode()).hexdigest()


def local_path(path):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT) or path == ROOT:
        raise FrameworkError(f'Calea nu este un subdirector propriu: {path}')
    current = path
    while current != ROOT:
        if current.is_symlink():
            raise FrameworkError(f'Calea administrata nu poate fi un link simbolic: {current}')
        current = current.parent
    return path


def submounts(path):
    path = Path(path).resolve()
    result = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        raw = line.split()[4]
        mount = Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), raw))
        if mount == path or mount.is_relative_to(path):
            result.append(str(mount))
    return result


def remove_local_tree(path):
    path = local_path(path)
    if submounts(path):
        raise FrameworkError(f'Nu sterg un director cu sisteme de fisiere montate: {path}')
    if path.exists():
        if not shutil.rmtree.avoids_symlink_attacks:
            raise FrameworkError('Sistemul nu permite stergere sigura fara urmarirea linkurilor.')
        shutil.rmtree(path)


@contextlib.contextmanager
def installation_lock():
    local_path(state_dir()).mkdir(mode=0o700, exist_ok=True)
    if state_dir().stat().st_uid != os.getuid():
        raise FrameworkError('Aceasta instalare apartine altui utilizator. Nu ii modific starea.')
    fd = os.open(state_dir() / 'lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FrameworkError('Alta comanda pentru aceasta instalare este deja in executie.') from exc
        for relative in ('cache', 'tmp', 'logs'):
            local_path(state_dir() / relative).mkdir(mode=0o700, exist_ok=True)
        npmrc = state_dir() / 'npmrc'
        if not npmrc.exists():
            npmrc.touch(mode=0o600)
        yield
    finally:
        os.close(fd)


def gpu_check(cfg):
    binary = shutil.which('nvidia-smi', path=SYSTEM_PATH)
    if not binary:
        raise FrameworkError('nvidia-smi lipseste. Driverul NVIDIA trebuie pregatit de administrator; nu il instalez.')
    output = run([binary, '-i', cfg['gpu'], '--query-gpu=name,driver_version,uuid',
                  '--format=csv,noheader'], capture=True, timeout=20).stdout.strip()
    parts = [x.strip() for x in output.split(',')]
    if len(parts) != 3 or not parts[2].startswith('GPU-'):
        raise FrameworkError(f'Raspuns NVIDIA neasteptat: {output}')
    try:
        version = tuple(int(v) for v in parts[1].split('.'))
    except ValueError as exc:
        raise FrameworkError('Versiune de driver NVIDIA nerecunoscuta.') from exc
    if version < (525, 60, 13):
        raise FrameworkError('CUDA 12.1 necesita cel putin driverul 525.60.13. Nu modific driverul.')
    log(f'GPU: {parts[0]}, driver {parts[1]}. Driverul ramane neschimbat.')
    return parts[2]


def model_check():
    base = ROOT / 'core/lip_reading_demo_module/Models'
    llama = base / 'llama'
    required = [ROOT / 'core/models' / n for n in
                ('Resnet50_Final.pth', '2dfan2.pth', 'bfm_lms.npy', '20words_mean_face.npy')]
    required += [base / s / 'checkpoint_best.pt' for s in ('avsr', 'romanizer')]
    required.append(base / 'avhubert/large_vox_iter5.pt')
    required.append(llama / 'config.json')
    if not any((llama / n).is_file() and (llama / n).stat().st_size for n in ('tokenizer.json', 'tokenizer.model')):
        raise FrameworkError('Lipseste tokenizer.json sau tokenizer.model din Models/llama/.')
    indexes = [llama / n for n in ('model.safetensors.index.json', 'pytorch_model.bin.index.json') if (llama / n).is_file()]
    if indexes:
        for index in indexes:
            shards = set(read_json(index).get('weight_map', {}).values())
            if not shards:
                raise FrameworkError(f'Index de ponderi gol: {index}')
            for shard in shards:
                p = llama / shard
                if not p.resolve().is_relative_to(llama.resolve()):
                    raise FrameworkError(f'Index de ponderi cu o cale externa: {shard}')
                required.append(p)
    else:
        weights = [llama / n for n in ('model.safetensors', 'pytorch_model.bin') if (llama / n).is_file()]
        if not weights:
            raise FrameworkError('Lipsesc ponderile Llama din Models/llama/. Copiaza directorul complet.')
        required += weights
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file() or not p.stat().st_size or not os.access(p, os.R_OK)]
    if missing:
        raise FrameworkError('Modele lipsa/incomplete:\n  ' + '\n  '.join(missing))
    for directory in (ROOT / 'core/models', base):
        local_path(directory)
        for p in directory.rglob('*'):
            if p.is_symlink() and not p.resolve().is_relative_to(directory.resolve()):
                raise FrameworkError(f'Modele legate de o cale externa: {p}. Copiaza fisierele efective.')
    local_path(ROOT / 'backend/storage')
    log('Modele: fisierele obligatorii sunt prezente.')


def digest_paths(paths):
    result = hashlib.sha256()
    for p in sorted(paths):
        result.update(str(p.relative_to(ROOT)).encode() + b'\0')
        result.update(p.read_bytes())
    return result.hexdigest()


def dependencies_key():
    return digest_paths([ROOT / 'framework' / n for n in ('requirements.lock', 'torch.lock')])


def frontend_key():
    paths = []
    for p in (ROOT / 'frontend').rglob('*'):
        relative = p.relative_to(ROOT / 'frontend')
        if any(x in {'node_modules', 'dist', '.git'} for x in relative.parts):
            continue
        if p.is_file() and p.suffix not in {'.tsbuildinfo'}:
            paths.append(p)
    return digest_paths(paths)
