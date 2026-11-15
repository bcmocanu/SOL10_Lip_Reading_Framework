"""Identify local processes by installation token, owner and Linux start time."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import time

import common as c


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def process_info(pid):
    try:
        proc = Path('/proc') / str(pid)
        if proc.stat().st_uid != os.getuid():
            return None
        raw = (proc / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        if fields[0] in ('Z', 'X'):
            return None
        env = {}
        for entry in (proc / 'environ').read_bytes().split(b'\0'):
            if b'=' in entry:
                key, value = entry.split(b'=', 1)
                if key.startswith(b'LIPREADING_'):
                    env[key.decode()] = value.decode(errors='replace')
        return {'pid': int(pid), 'start': fields[19], 'ppid': int(fields[1]), 'env': env}
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError:
        return None


def belongs(info, identity, run_token=None):
    if not info:
        return False
    env = info['env']
    return (env.get('LIPREADING_INSTANCE') == identity['id'] and
            env.get('LIPREADING_ROOT') == str(c.ROOT) and
            (run_token is None or env.get('LIPREADING_RUN') == run_token))


def owned_processes(identity, run_token=None):
    processes = []
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit() and int(proc.name) != os.getpid():
            info = process_info(int(proc.name))
            if belongs(info, identity, run_token):
                processes.append(info)
    return processes


def signal_owned(info, identity, sig, run_token=None):
    pid = info['pid']
    fd = None
    try:
        # pidfd pins a particular process even if its PID is reused concurrently.
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise c.FrameworkError('Kernel/Python fara suport pidfd; nu opresc procese prin identificatori nesiguri.')
        fd = os.pidfd_open(pid)
        current = process_info(pid)
        if current and current['start'] == info['start'] and belongs(current, identity, run_token):
            signal.pidfd_send_signal(fd, sig)
    except ProcessLookupError:
        pass
    finally:
        if fd is not None:
            os.close(fd)


def stop(identity, grace=30.0, run_token=None):
    initial = owned_processes(identity, run_token)
    if not initial:
        c.log('Aplicatia nu are procese active pentru aceasta instalare.')
        return
    c.log('Opresc numai procesele acestei instalari...')
    parents = {p['pid'] for p in initial}
    # Let the API finish active work before signalling its helper processes.
    leaders = [p for p in initial if p['ppid'] not in parents]
    for info in leaders or initial:
        signal_owned(info, identity, signal.SIGTERM, run_token)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not owned_processes(identity, run_token):
            break
        time.sleep(0.15)
    pending = owned_processes(identity, run_token)
    if pending:
        c.log('Procesele proprii nu s-au inchis normal; finalizez fortat numai procesele identificate.')
    # Descendants can outlive the service process; rescan the inherited markers.
    deadline = time.monotonic() + 5
    while pending and time.monotonic() < deadline:
        for info in pending:
            signal_owned(info, identity, signal.SIGKILL, run_token)
        time.sleep(0.1)
        pending = owned_processes(identity, run_token)
    if pending:
        raise c.FrameworkError('Unele procese proprii nu pot fi oprite. Stergerea este refuzata.')
    c.log('Aplicatia este oprita.')


def referenced_by_other_processes(identity):
    """Best-effort guard for readable processes sharing this directory."""
    ancestors = {os.getpid()}
    pid = os.getpid()
    while pid > 1:
        try:
            text = (Path('/proc') / str(pid) / 'stat').read_text()
            pid = int(text[text.rfind(')') + 2:].split()[1])
        except (OSError, ValueError):
            break
        ancestors.add(pid)
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in ancestors:
            continue
        info = process_info(int(proc.name))
        if belongs(info, identity):
            continue
        try:
            paths = [proc / 'cwd', proc / 'exe']
            paths += list((proc / 'fd').iterdir())
            for entry in paths:
                try:
                    value = os.readlink(entry).removesuffix(' (deleted)')
                    if value.startswith('/') and Path(value).is_relative_to(c.ROOT):
                        found.append(int(proc.name))
                        break
                except (OSError, ValueError):
                    continue
        except (OSError, PermissionError):
            continue
    return sorted(set(found))
