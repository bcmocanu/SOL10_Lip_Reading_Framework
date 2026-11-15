from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"

@dataclass(frozen=True)
class Settings:
    app_name: str = "SOL10 LipReadingWeb API"
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
    max_segment_seconds: float = float(os.getenv("MAX_SEGMENT_SECONDS", "15"))
    upload_dir: Path = BACKEND_ROOT / "storage" / "uploads"
    processed_dir: Path = BACKEND_ROOT / "storage" / "processed"

settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.processed_dir.mkdir(parents=True, exist_ok=True)
