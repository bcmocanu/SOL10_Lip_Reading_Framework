"""Provision project-local Python/Node and explicitly approved missing OS libraries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import urllib.request

import common as c

PYTHON_URL = ('https://github.com/astral-sh/python-build-standalone/releases/download/20260211/'
              'cpython-3.10.19%2B20260211-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz')
PYTHON_SHA256 = 'd71df61d1cdb59af4443912da8eeca744a52e782ff5deefa966ead893235a39e'
NODE_URL = 'https://nodejs.org/download/release/v22.16.0/node-v22.16.0-linux-x64.tar.xz'
NODE_SHA256 = 'f4cb75bb036f0d0eddf6b79d9596df1aaab9ddccd6a20bf489be5abe9467e84e'


class AptIndexError(c.FrameworkError):
    pass


def installed_package(name):
    result = c.run(['dpkg-query', '-W', '-f=${Status}', name], capture=True, check=False)
    return result.returncode == 0 and result.stdout.strip() == 'install ok installed'


def apt_plan(packages):
    result = c.run(['apt-get', '-s', '--no-upgrade', '--no-remove', '--no-install-recommends',
                    'install', *packages], capture=True, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout or '').strip()
        if 'Unable to locate package' in detail or 'has no installation candidate' in detail:
            raise AptIndexError(detail)
        raise c.FrameworkError('APT nu poate construi un plan sigur:\n' + detail)
    proposed = []
    configured = []
    for line in result.stdout.splitlines():
        if line.startswith('Remv '):
            raise c.FrameworkError('APT ar elimina pachete existente. Nu continui.')
        if line.startswith('Conf '):
            configured.append(line.split()[1])
        if line.startswith('Inst '):
            match = re.fullmatch(r'Inst ([\w.+:-]+) \(([^\s)]+).*', line)
            if not match or installed_package(match[1]):
                raise c.FrameworkError(f'APT ar modifica un pachet existent sau planul nu este sigur:\n{line}')
            name, version = match.groups()
            if re.search(r'^((lib)?nvidia|(lib)?cuda|docker|containerd|systemd|linux-image|linux-headers|python)', name):
                raise c.FrameworkError(f'Nu permit instalarea acestei dependinte globale: {name}')
            proposed.append(f'{name}={version}')
    approved_names = {item.split('=', 1)[0] for item in proposed}
    if any(name not in approved_names for name in configured):
        raise c.FrameworkError('APT ar configura pachete din afara instalarii aprobate. Nu continui.')
    if not proposed:
        raise c.FrameworkError('Nu am identificat un plan de instalare pentru dependintele lipsa.')
    return proposed, result.stdout


def apt_missing(packages):
    packages = [p for p in sorted(set(packages)) if not installed_package(p)]
    if not packages:
        return
    if not shutil.which('sudo', path=c.SYSTEM_PATH):
        raise c.FrameworkError('Lipsesc pachete de sistem: ' + ', '.join(packages) + '. Este necesar un administrator.')
    try:
        proposed, detail = apt_plan(packages)
    except AptIndexError as exc:
        c.log(str(exc))
        c.confirm('Lipsesc: ' + ', '.join(packages) +
                  '. Permit doar actualizarea indexului APT cu sudo (fara instalari/upgrade-uri)?')
        c.run(['sudo', '--', 'apt-get', 'update'])
        proposed, detail = apt_plan(packages)
    c.log(detail)
    c.confirm('Permit instalarea prin sudo numai a pachetelor noi din planul de mai sus?\n'
              'Nu sunt acceptate inlocuiri, upgrade-uri sau dezinstalari. Docker si driverul NVIDIA nu sunt configurate.')
    checked, _ = apt_plan(proposed)
    if sorted(checked) != sorted(proposed):
        raise c.FrameworkError('Planul APT s-a schimbat. Nu instalez nimic.')
    c.run(['sudo', '--', 'env', 'DEBIAN_FRONTEND=noninteractive', 'NEEDRESTART_MODE=l',
           'apt-get', '-y', '--no-upgrade', '--no-remove', '--no-install-recommends', 'install', *proposed])


def missing_system():
    missing = []
    for program in ('ffmpeg', 'ffprobe'):
        binary = shutil.which(program, path=c.SYSTEM_PATH)
        if not binary:
            missing.append('ffmpeg')
        elif c.run([binary, '-version'], capture=True, check=False, timeout=20).returncode:
            raise c.FrameworkError(f'{program} exista, dar nu functioneaza. Nu il inlocuiesc automat.')
    libraries = {
        'libGL.so.1': 'libgl1', 'libglib-2.0.so.0': 'libglib2.0-0',
        'libSM.so.6': 'libsm6', 'libXext.so.6': 'libxext6',
        'libXrender.so.1': 'libxrender1', 'libsndfile.so.1': 'libsndfile1',
    }
    probe = 'import ctypes,json\nmissing=[]\nfor n in ' + repr(list(libraries)) + ':\n'
    probe += ' try: ctypes.CDLL(n)\n except OSError: missing.append(n)\nprint(json.dumps(missing))\n'
    absent = json.loads(c.run(['/usr/bin/python3', '-I', '-c', probe], capture=True).stdout)
    for library in absent:
        package = libraries[library]
        if package == 'libglib2.0-0':
            policy = c.run(['apt-cache', 'policy', 'libglib2.0-0t64'], capture=True, check=False).stdout
            if re.search(r'Candidate:\s+(?!\(none\))\S+', policy):
                package = 'libglib2.0-0t64'
        if installed_package(package):
            raise c.FrameworkError(f'{package} este instalat, dar {library} nu se incarca. Nu reinstalez automat.')
        missing.append(package)
    return sorted(set(missing))


def system_requirements():
    missing = missing_system()
    if missing:
        apt_missing(missing)
        missing = missing_system()
        if missing:
            raise c.FrameworkError('Dependinte de sistem inca absente: ' + ', '.join(missing))
    c.log('FFmpeg si bibliotecile native sunt disponibile. Nu modific pachetele existente.')


def fetch_archive(url, checksum, destination):
    c.log('Descarc in proiect: ' + url)
    request = urllib.request.Request(url, headers={'User-Agent': 'LipReadingModule-setup'})
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=60) as response, open(destination, 'xb') as output:
            if not response.url.startswith('https://'):
                raise c.FrameworkError('Descarcarea a fost redirectionata catre o conexiune nesecurizata.')
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != checksum:
            raise c.FrameworkError('Checksum SHA256 invalid. Arhiva nu va fi executata.')
    except OSError as exc:
        raise c.FrameworkError(f'Descarcarea nu a reusit: {exc}. Este necesar acces HTTPS la sursa indicata.') from exc


def extract_archive(archive, target):
    """Extract verified archives without allowing escape paths or special files."""
    target = Path(target).resolve()
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        links = []
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or '..' in relative.parts:
                raise c.FrameworkError('Arhiva contine o cale nesigura.')
            path = target / member.name
            if not path.resolve().is_relative_to(target):
                raise c.FrameworkError('Arhiva iese din directorul de instalare.')
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as stream, path.open('xb') as output:
                    shutil.copyfileobj(stream, output)
                path.chmod(member.mode & 0o777)
            elif member.issym() or member.islnk():
                links.append(member)
            else:
                raise c.FrameworkError('Arhiva contine un fisier special neacceptat.')
        for member in links:
            path = target / member.name
            link = Path(member.linkname)
            resolved = ((path.parent if member.issym() else target) / link).resolve()
            if link.is_absolute() or not resolved.is_relative_to(target) or not path.parent.resolve().is_relative_to(target):
                raise c.FrameworkError('Arhiva contine un link extern.')
            path.parent.mkdir(parents=True, exist_ok=True)
            if member.issym():
                path.symlink_to(member.linkname)
            else:
                os.link(resolved, path)


def install_private_runtime(name, url, sha256, subdirectory):
    parent = c.local_path(c.ROOT / '.runtime')
    parent.mkdir(exist_ok=True)
    destination = c.local_path(parent / name)
    if destination.exists():
        raise c.FrameworkError(f'Exista deja {destination}, dar nu poate fi utilizat. Nu il suprascriu automat.')
    c.confirm(f'Permit descarcarea componentei {name} numai in {destination}, fara sudo?')
    with tempfile.TemporaryDirectory(prefix='download-', dir=c.state_dir() / 'tmp') as temporary:
        temp = Path(temporary)
        archive = temp / 'runtime.tar'
        fetch_archive(url, sha256, archive)
        unpack = temp / 'unpack'
        unpack.mkdir()
        extract_archive(archive, unpack)
        source = unpack / subdirectory
        if not source.is_dir():
            raise c.FrameworkError('Structura arhivei runtime nu este cea asteptata.')
        source.rename(destination)
    return destination


def python_compatible(binary, base=True):
    if not Path(binary).is_file():
        return False
    code = ('import sys,ssl,venv,ensurepip,lzma,sqlite3; '
            'assert sys.version_info[:2] == (3,10); ')
    if base:
        code += 'assert sys.prefix == sys.base_prefix; '
    result = c.run([binary, '-I', '-c', code], capture=True, check=False, timeout=20)
    return result.returncode == 0


def verify_venv():
    root = c.local_path(c.ROOT / 'LipsWebEnv')
    code = ('import sys,pathlib; assert sys.version_info[:2]==(3,10); '
            'assert sys.prefix!=sys.base_prefix; '
            f'assert pathlib.Path(sys.prefix).resolve()==pathlib.Path({str(root)!r}).resolve(); '
            'import pip; print(sys.version.split()[0])')
    version = c.run([c.env_python(), '-I', '-c', code], capture=True).stdout.strip()
    c.log(f'Python {version}: mediu izolat {root}')


def create_venv():
    target = c.local_path(c.ROOT / 'LipsWebEnv')
    if target.exists():
        try:
            verify_venv()
            return
        except c.FrameworkError as exc:
            c.log(str(exc))
            c.confirm('Mediul local LipsWebEnv este incomplet sau incompatibil. Permit recrearea lui? Modelele si datele nu se sterg.')
            c.remove_local_tree(target)
    private = c.ROOT / '.runtime/python/bin/python3.10'
    candidates = [private, Path('/usr/bin/python3.10'), Path('/usr/local/bin/python3.10')]
    binary = next((p for p in candidates if python_compatible(p)), None)
    if binary is None:
        install_private_runtime('python', PYTHON_URL, PYTHON_SHA256, 'python')
        binary = private
        if not python_compatible(binary):
            raise c.FrameworkError('Python local nu poate fi utilizat pe acest sistem.')
    c.log(f'Creez LipsWebEnv utilizand {binary}. Python-ul sistemului nu este inlocuit.')
    try:
        c.run([binary, '-I', '-m', 'venv', '--copies', target])
        verify_venv()
    except BaseException:
        if target.exists():
            c.remove_local_tree(target)
        raise


def install_dependencies():
    verify_venv()
    py = c.env_python()
    pip = [py, '-I', '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple']
    c.run([*pip, 'pip==23.3.2', 'setuptools==68.2.2', 'wheel==0.41.3'])
    c.run([py, '-I', '-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cu121',
           '-r', c.ROOT / 'framework/torch.lock'])
    c.run([*pip, '--no-build-isolation', '-r', c.ROOT / 'framework/requirements.lock',
           '-c', c.ROOT / 'framework/torch.lock'])
    c.run([py, '-I', '-m', 'pip', 'check'])



def ensure_local_cuda_libraries():
    """Make native CUDA libraries available to bitsandbytes, never to the host."""
    if c.local_cuda_library('libcusparse.so.12') is not None:
        return
    c.log('Biblioteca cuSPARSE 12 nu exista in mediul local; instalez doar runtime-ul necesar in LipsWebEnv.')
    py = c.env_python()
    c.run([py, '-I', '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple',
           '--no-deps', 'nvidia-cusparse-cu12==12.1.0.106',
           'nvidia-nvjitlink-cu12==12.2.140'])
    if c.local_cuda_library('libcusparse.so.12') is None:
        raise c.FrameworkError('Pachetul CUDA a fost instalat, dar libcusparse.so.12 nu a fost gasit in LipsWebEnv.')
    c.run([py, '-I', '-m', 'pip', 'check'])


def node_compatible(binary):
    if not binary or not Path(binary).is_file():
        return False
    result = c.run([binary, '--version'], capture=True, check=False, timeout=20)
    match = re.fullmatch(r'v(\d+)\.(\d+)\.(\d+)\s*', result.stdout)
    if result.returncode or not match:
        return False
    major, minor, _ = map(int, match.groups())
    return (major == 20 and minor >= 19) or (major == 22 and minor >= 12) or major > 22


def node_command():
    local = c.ROOT / '.runtime/node/bin'
    candidates = [local, Path('/usr/local/bin'), Path('/usr/bin')]
    # A compatible existing Node from nvm can also be used without changing it.
    inherited = shutil.which('node')
    if inherited:
        candidates.append(Path(inherited).parent)
    for directory in candidates:
        node, npm = directory / 'node', directory / 'npm'
        if node_compatible(node) and npm.is_file():
            return node, npm
    install_private_runtime('node', NODE_URL, NODE_SHA256, 'node-v22.16.0-linux-x64')
    return local / 'node', local / 'npm'


def build_frontend():
    node, npm = node_command()
    env = c.child_env()
    env['PATH'] = str(node.parent) + ':' + env['PATH']
    env['VITE_API_BASE_URL'] = ''
    env['NODE_ENV'] = 'development'
    frontend = c.local_path(c.ROOT / 'frontend')
    c.local_path(frontend / 'node_modules')
    c.local_path(frontend / 'dist')
    c.log('Construiesc interfata web; toate pachetele npm sunt locale proiectului.')
    # Node executes npm explicitly, without relying on a system npm interpreter.
    npm_cli = npm.resolve()
    c.run([node, npm_cli, 'ci', '--include=dev', '--no-audit', '--no-fund'], env=env, cwd=frontend)
    env['NODE_ENV'] = 'production'
    c.run([node, npm_cli, 'run', 'build'], env=env, cwd=frontend)
    if not (frontend / 'dist/index.html').is_file():
        raise c.FrameworkError('Compilarea nu a produs frontend/dist/index.html.')
    c.log('Interfata web a fost construita. Node nu este necesar la pornirea aplicatiei.')


def check_runtime(gpu, full=False):
    args = [c.env_python(), '-I', '-B', c.ROOT / 'framework/verify_runtime.py']
    if full:
        args.append('--full')
    c.run(args, env=c.child_env(gpu), timeout=180)
