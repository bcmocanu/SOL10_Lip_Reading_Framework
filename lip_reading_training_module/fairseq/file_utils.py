import os
from pathlib import Path


def cached_path(url_or_filename, cache_dir=None):
    value = os.fspath(url_or_filename)
    if "://" in value:
        raise ValueError("Only local files are supported in offline mode: " + value)
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)
