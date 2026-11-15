"""Run one API worker on the socket reserved by the launcher."""
import argparse
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    if (os.environ.get('LIPREADING_INSTANCE') != args.instance or
            os.environ.get('LIPREADING_RUN') != args.run or
            os.environ.get('LIPREADING_ROOT') != str(ROOT)):
        raise SystemExit('Identitatea procesului nu corespunde instalarii.')
    os.chdir(ROOT)
    sock = socket.socket(fileno=args.fd)
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(
        'framework.webapp:create_app', factory=True, workers=1, log_level='info',
        access_log=False, proxy_headers=False, timeout_graceful_shutdown=25,
    ))
    server.run(sockets=[sock])


if __name__ == '__main__':
    main()
