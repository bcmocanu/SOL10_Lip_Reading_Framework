"""Serve the compiled interface alongside the unchanged inference API."""
from pathlib import Path
import os

from fastapi import HTTPException
from starlette.responses import FileResponse, JSONResponse
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent


class FrontendFiles(StaticFiles):
    async def get_response(self, path, scope):
        if any(part.startswith('.') and part not in ('', '.') for part in path.split('/')):
            raise HTTPException(status_code=404)
        if scope['method'] in ('GET', 'HEAD') and path.rstrip('/') == 'processing':
            response = FileResponse(Path(self.directory) / 'index.html', media_type='text/html')
        else:
            response = await super().get_response(path, scope)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = ('public, max-age=31536000, immutable'
                                             if path.startswith('assets/') else 'no-cache')
        return response


def create_app():
    from backend.app.main import app
    directory = ROOT / 'frontend/dist'
    if not (directory / 'index.html').is_file():
        raise RuntimeError('Interfata lipseste. Ruleaza ./setupFramework.')

    @app.get('/_framework/health', include_in_schema=False)
    def installation_health():
        return JSONResponse({'status': 'ok', 'instance': os.environ['LIPREADING_INSTANCE'],
                             'run': os.environ['LIPREADING_RUN']}, headers={'Cache-Control': 'no-store'})

    app.mount('/', FrontendFiles(directory=directory, html=True, follow_symlink=False))
    return app
