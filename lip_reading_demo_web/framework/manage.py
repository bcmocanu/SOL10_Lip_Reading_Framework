"""Local setup, start, stop and removal of one Lip reading module installation."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c
import install
import processes


def installation_identity(action):
    path = c.state_dir() / 'installation.json'
    current = {'application': 'lip-reading-module', 'root': str(c.ROOT),
               'machine': c.machine_id(), 'uid': os.getuid(),
               'directory': [c.ROOT.stat().st_dev, c.ROOT.stat().st_ino]}
    previous = c.read_json(path) if path.is_file() else None
    if previous is not None:
        if previous.get('application') != current['application'] or not isinstance(previous.get('id'), str):
            raise c.FrameworkError('Stare de instalare nerecunoscuta; nu o suprascriu.')
        if all(previous.get(key) == value for key, value in current.items()):
            return previous
    if action not in ('setup', 'delete'):
        raise c.FrameworkError('Instalarea este noua, mutata sau copiata. Ruleaza ./setupFramework pe acest PC.')
    if previous and action == 'setup':
        if processes.referenced_by_other_processes(previous):
            raise c.FrameworkError('Directorul este folosit de alte procese. Opreste instalarea inainte de mutare/configurare.')
        folders = [c.ROOT / name for name in ('LipsWebEnv', '.runtime') if (c.ROOT / name).exists()]
        if folders:
            c.confirm('Proiectul a fost mutat/copiat. Mediul Python nu se reutilizeaza intre instalari.\n'
                      'Permit recrearea numai a directoarelor locale de executie? Modelele si datele sunt pastrate.\n'
                      + '\n'.join(str(p) for p in folders))
            for folder in folders:
                c.remove_local_tree(folder)
    elif previous is None and action == 'setup':
        if any((c.ROOT / name).exists() for name in ('LipsWebEnv', '.runtime')):
            raise c.FrameworkError('Exista medii locale fara identitate de instalare. Nu le suprascriu. Foloseste un director nou.')
    current['id'] = uuid.uuid4().hex
    c.atomic_json(path, current)
    for name in ('setup.json', 'process.json'):
        target = c.state_dir() / name
        if target.exists() or target.is_symlink():
            if target.is_symlink():
                raise c.FrameworkError(f'Starea nu poate fi un link simbolic: {target}')
            target.unlink()
    return current


def read_setup(identity):
    path = c.state_dir() / 'setup.json'
    if not path.is_file():
        return {}
    state = c.read_json(path)
    return state if state.get('instance') == identity['id'] else {}


def setup(identity, cfg):
    if processes.owned_processes(identity):
        raise c.FrameworkError('Aplicatia este pornita. Foloseste ./stopFramework inainte de setup.')
    if processes.referenced_by_other_processes(identity):
        raise c.FrameworkError('Alte procese folosesc directorul proiectului. Nu ii modific mediul de executie.')
    c.model_check()
    gpu = c.gpu_check(cfg)
    install.system_requirements()
    previous = read_setup(identity)
    c.atomic_json(c.state_dir() / 'setup.json', {'instance': identity['id'], 'ready': False})
    install.create_venv()
    # Also reuse an intact environment if an earlier setup stopped during GPU verification.
    matches = c.run([c.env_python(), '-I', '-B', c.ROOT / 'framework/verify_runtime.py', '--versions'],
                    capture=True, check=False, timeout=40).returncode == 0
    if matches:
        matches = c.run([c.env_python(), '-I', '-m', 'pip', 'check'], capture=True, check=False, timeout=40).returncode == 0
    if not matches:
        c.log('Instalez dependintele fixate numai in LipsWebEnv. Sunt descarcate biblioteci CUDA/PyTorch, nu driverul NVIDIA.')
        install.install_dependencies()
    else:
        c.log('Dependintele Python sunt deja instalate la versiunile necesare; nu le reinstalez.')
    install.ensure_local_cuda_libraries()
    frontend_signature = c.frontend_key()
    if previous.get('frontend') != frontend_signature or not (c.ROOT / 'frontend/dist/index.html').is_file():
        install.build_frontend()
        frontend_signature = c.frontend_key()
    else:
        c.log('Interfata compilata este deja disponibila; nu reinstalez pachetele npm.')
    install.check_runtime(gpu, full=True)
    c.atomic_json(c.state_dir() / 'setup.json', {
        'instance': identity['id'], 'ready': True, 'dependencies': c.dependencies_key(),
        'frontend': frontend_signature,
    })
    c.log('\nInstalare verificata. Pornire: ./startFramework')


def reserve_socket(bind, preferred):
    for port in range(preferred, preferred + 101):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind, port))
            sock.listen(128)
            if port != preferred:
                c.log(f'Portul {preferred} este ocupat; folosesc {port}, fara sa opresc alta aplicatie.')
            return sock, port
        except OSError as exc:
            sock.close()
            if exc.errno != errno.EADDRINUSE:
                raise c.FrameworkError(f'Nu pot asculta pe {bind}:{port}: {exc}') from exc
    raise c.FrameworkError('Nu exista un port liber in intervalul configurat.')


def health(process):
    host = process['bind'] if process['bind'] != '0.0.0.0' else '127.0.0.1'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f'http://{host}:{process["port"]}/_framework/health', timeout=1) as response:
            data = json.load(response)
        return (data.get('status') == 'ok' and data.get('instance') == process['instance']
                and data.get('run') == process['run'])
    except (OSError, ValueError, urllib.error.URLError):
        return False


def url_for(process):
    host = process['bind'] if process['bind'] != '0.0.0.0' else '127.0.0.1'
    return f'http://{host}:{process["port"]}'


def tail_log():
    path = c.state_dir() / 'logs/application.log'
    if path.is_file():
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 6000))
            c.log(stream.read().decode(errors='replace'))
    c.log(f'Jurnal: {path}')


def start(identity, cfg):
    process_path = c.state_dir() / 'process.json'
    process = c.read_json(process_path) if process_path.is_file() else {}
    existing = processes.owned_processes(identity)
    if existing:
        if (process.get('instance') == identity['id'] and process.get('boot') == processes.boot_id()
                and any(x['pid'] == process.get('pid') and x['start'] == process.get('start') for x in existing)
                and health(process)):
            c.log('Aplicatia ruleaza deja: ' + url_for(process))
            return
        raise c.FrameworkError('Exista procese proprii, dar aplicatia nu raspunde normal. Foloseste ./stopFramework si verifica .framework/logs/application.log.')
    state = read_setup(identity)
    if not state.get('ready'):
        raise c.FrameworkError('Setup-ul nu este complet. Ruleaza ./setupFramework.')
    if state.get('dependencies') != c.dependencies_key() or state.get('frontend') != c.frontend_key():
        raise c.FrameworkError('Configuratia de instalare sau frontend-ul s-au schimbat. Ruleaza ./setupFramework.')
    if not (c.ROOT / 'frontend/dist/index.html').is_file():
        raise c.FrameworkError('Lipseste interfata compilata. Ruleaza ./setupFramework.')
    c.model_check()
    gpu = c.gpu_check(cfg)
    install.verify_venv()
    install.check_runtime(gpu)
    for relative in ('backend/storage/uploads', 'backend/storage/processed'):
        c.local_path(c.ROOT / relative).mkdir(parents=True, exist_ok=True)
    preferred = cfg['port']
    if (process.get('instance') == identity['id'] and process.get('preferred') == preferred and
            process.get('bind') == cfg['bind'] and type(process.get('port')) is int):
        preferred = process['port']
    sock, port = reserve_socket(cfg['bind'], preferred)
    token = secrets.token_hex(24)
    env = c.child_env(gpu)
    env.update({'LIPREADING_INSTANCE': identity['id'], 'LIPREADING_RUN': token,
                'LIPREADING_ROOT': str(c.ROOT), 'FRONTEND_ORIGIN': f'http://127.0.0.1:{port}'})
    log_path = c.state_dir() / 'logs/application.log'
    if log_path.is_symlink():
        sock.close()
        raise c.FrameworkError('Jurnalul aplicatiei nu poate fi un link simbolic.')
    if log_path.is_file():
        old = log_path.with_name('application.previous.log')
        if old.is_symlink():
            sock.close()
            raise c.FrameworkError('Jurnalul anterior nu poate fi un link simbolic.')
        log_path.replace(old)
    child = None
    try:
        with log_path.open('wb') as stream:
            child = subprocess.Popen(
                [str(c.env_python()), '-I', '-B', '-u', str(c.ROOT / 'framework/service.py'),
                 '--fd', str(sock.fileno()), '--instance', identity['id'], '--run', token],
                cwd=c.ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(sock.fileno(),))
        info = None
        for _ in range(100):
            info = processes.process_info(child.pid)
            if processes.belongs(info, identity, token):
                break
            if child.poll() is not None:
                break
            time.sleep(0.02)
        if not processes.belongs(info, identity, token):
            raise c.FrameworkError('Procesul nu a pornit cu identitatea asteptata.')
        process = {'instance': identity['id'], 'run': token, 'pid': child.pid, 'start': info['start'],
                   'boot': processes.boot_id(), 'bind': cfg['bind'], 'port': port, 'preferred': cfg['port']}
        c.atomic_json(process_path, process)
        sock.close()
        for _ in range(120):
            if child.poll() is not None:
                raise c.FrameworkError('Procesul aplicatiei s-a oprit la initializare.')
            if health(process):
                c.log('\nLip reading module ruleaza: ' + url_for(process))
                c.log('Oprire: ./stopFramework')
                return
            time.sleep(0.5)
        raise c.FrameworkError('Aplicatia nu a raspuns la verificarea de pornire.')
    except BaseException:
        sock.close()
        if child is not None:
            processes.stop(identity, grace=2, run_token=token)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        tail_log()
        raise


def delete(identity):
    root = c.ROOT
    forbidden = {Path('/'), Path('/home'), Path.home().resolve(), Path('/tmp'), Path('/var'),
                 Path('/opt'), Path('/usr'), Path('/mnt'), Path('/mnt/data')}
    if root in forbidden or len(root.parts) < 3 or root.is_symlink():
        raise c.FrameworkError('Director nesigur pentru stergere.')
    mounts = c.submounts(root)
    if mounts:
        raise c.FrameworkError('Nu sterg un director care contine volume montate: ' + ', '.join(mounts))
    for script in ('setupFramework', 'startFramework', 'stopFramework', 'deleteFramework'):
        if not (root / script).is_file():
            raise c.FrameworkError('Structura instalarii este incompleta; stergerea automata este refuzata.')
    if not shutil.rmtree.avoids_symlink_attacks:
        raise c.FrameworkError('Sistemul nu permite stergere sigura fara urmarirea linkurilor.')
    c.confirm(f'ATENTIE: se opreste aceasta instalare si se sterge integral:\n{root}\n'
              'Inclusiv codul, MODELELE, fisierele incarcate, rezultatele, mediul local si cache-urile.\n'
              'Linkurile externe se elimina ca linkuri; tintele lor nu sunt sterse.\n'
              'Pachetele de sistem si resursele altor proiecte NU se dezinstaleaza.\n'
              'Directorul nu trebuie sa fie partajat cu alte aplicatii sau montat manual in containere.',
              expected=f'STERGE {identity["id"][:12]}')
    processes.stop(identity)
    others = processes.referenced_by_other_processes(identity)
    if others:
        raise c.FrameworkError(f'Alte procese folosesc directorul (PID {others}). Nu sterg fisierele.')
    if c.submounts(root):
        raise c.FrameworkError('Au aparut volume montate. Stergerea este refuzata.')
    os.chdir(root.parent)
    try:
        shutil.rmtree(root)
    except OSError as exc:
        raise c.FrameworkError(f'Stergerea nu este completa: {exc}. Nu folosesc sudo pentru a forta stergerea.') from exc
    c.log('Instalarea a fost stearsa. Configuratia sistemului si celelalte proiecte nu au fost modificate.')


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('setup', 'start', 'stop', 'delete'):
        print('Foloseste ./setupFramework, ./startFramework, ./stopFramework sau ./deleteFramework.', file=sys.stderr)
        return 2
    action = sys.argv[1]
    os.umask(0o077)
    try:
        c.platform_check(setup=action == 'setup')
        with c.installation_lock():
            identity = installation_identity(action)
            if action == 'setup':
                setup(identity, c.config())
            elif action == 'start':
                start(identity, c.config())
            elif action == 'stop':
                processes.stop(identity)
            else:
                delete(identity)
    except c.FrameworkError as exc:
        print(f'EROARE: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nOperatie intrerupta. Poti relua aceeasi comanda.', file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        print(f'EROARE: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
