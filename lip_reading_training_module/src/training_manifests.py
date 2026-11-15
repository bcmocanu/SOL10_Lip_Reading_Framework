from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import random
import tempfile
import time
import unicodedata

from .training_config import LANGUAGE_CODE, local_path

logger = logging.getLogger(__name__)
ALLOWED_CHARS = set("\n '0123456789abcdefghijklmnopqrstuvwxyz")
EXTENSIONS = ("tsv", "wrd", "ltr")


def normalize_text(text: str) -> str:
    text = text.replace("-", " ").replace("%", " la suta ")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    return "".join(c for c in text if c in ALLOWED_CHARS or c.isspace())


def to_ltr(text: str) -> str:
    return " | ".join(" ".join(word) for word in normalize_text(text).strip().split()) + " |"


@contextmanager
def locked_directory(folder: Path, exclusive: bool = True):
    """Linux advisory lock; no persistent lock file is created."""
    folder.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(folder), os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Director utilizat de alt proces setup/training: {folder}") from exc
        yield
    finally:
        os.close(fd)


@dataclass(frozen=True)
class Record:
    video: str
    audio: str
    frames: int
    wrd: str
    ltr: str


def split_paths(folder: Path, split: str) -> dict[str, Path]:
    if not split or Path(split).name != split or split in (".", ".."):
        raise ValueError(f"Nume de split invalid: {split!r}")
    return {ext: folder / f"{split}.{ext}" for ext in EXTENSIONS}


def _discover(source: Path) -> list[Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"Lipseste directorul bazei de date: {source}")
    logger.info("%s: cautare fisiere *_text.json in %s...", source.name, source)
    files, directories, last = [], 0, time.monotonic()
    def onerror(error):
        raise error
    for root, _, names in os.walk(source, onerror=onerror):
        directories += 1
        files.extend(Path(root) / name for name in names if name.endswith("_text.json"))
        if time.monotonic() - last >= 5:
            logger.info("%s: cautare in curs — %d directoare, %d JSON-uri gasite...",
                        source.name, directories, len(files))
            last = time.monotonic()
    logger.info("%s: gasite %d exemple; ordonare...", source.name, len(files))
    if not files:
        raise ValueError(f"Nu exista *_text.json in {source}")
    return sorted(files)


def _scan(db_root: Path, split: str, require_audio: bool, seed: int, shuffle: bool) -> list[Record]:
    import cv2
    import soundfile as sf

    files = _discover(db_root / split)
    if shuffle:
        random.Random(seed).shuffle(files)
    logger.info("%s: verificare JSON, metadate video, primul cadru%s...",
                split, " si audio mono/16 kHz" if require_audio else "")
    records, errors, error_count = [], [], 0
    last = time.monotonic()
    for idx, json_path in enumerate(files, 1):
        try:
            base = str(json_path)[:-len("_text.json")]
            video, audio = Path(base + "_NoAudio.mp4"), Path(base + "_Audio.wav")
            obj = json.loads(json_path.read_text(encoding="utf-8"))
            text = obj.get("text") if isinstance(obj, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError("campul 'text' lipseste sau este gol")
            if any(c in text for c in ("\n", "\r", "\v", "\f", "\x85", "\u2028", "\u2029")):
                raise ValueError("transcrierea trebuie sa ocupe un singur rand")
            if not normalize_text(text).strip():
                raise ValueError("transcriere goala dupa normalizare")
            if not video.is_file():
                raise FileNotFoundError(video)
            cap = cv2.VideoCapture(str(video))
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                ok, first = cap.read()
                if not ok or not math.isfinite(length) or length < 2:
                    raise ValueError("video nedecodabil sau prea scurt")
                if not math.isfinite(fps) or abs(fps - 25.0) > 0.05:
                    raise ValueError(f"video cu {fps:g} FPS; sunt necesare 25 FPS")
                if min(first.shape[:2]) < 88:
                    raise ValueError(f"cadru {first.shape[:2]}: prea mic pentru cropul 88×88")
                frames = int(length)
            finally:
                cap.release()
            if require_audio:
                info = sf.info(str(audio))
                if info.samplerate != 16000 or info.channels != 1 or info.frames <= 0:
                    raise ValueError(f"audio trebuie sa fie mono, 16 kHz si nevid: {audio}")
            if any(c in str(video) + str(audio) for c in "\t\r\n"):
                raise ValueError("caile nu pot contine TAB sau newline")
            records.append(Record(str(video.resolve()), str(audio.resolve()), frames, text.lower(), to_ltr(text)))
        except Exception as exc:
            error_count += 1
            if len(errors) < 20:
                errors.append(f"{json_path}: {exc}")
        if time.monotonic() - last >= 5 or idx == len(files):
            logger.info("%s: verificate %d/%d exemple", split, idx, len(files))
            last = time.monotonic()
    if error_count:
        raise ValueError(f"{split}: {error_count} exemple invalide; nu s-au scris manifestele.\n" + "\n".join(errors))
    return records


def _read_existing(folder: Path, split: str) -> list[Record] | None:
    paths = split_paths(folder, split)
    if not all(path.is_file() for path in paths.values()):
        return None
    content = {ext: path.read_text(encoding="utf-8").splitlines() for ext, path in paths.items()}
    if not content["tsv"] or len(content["tsv"]) - 1 != len(content["wrd"]) or len(content["wrd"]) != len(content["ltr"]):
        return None
    root, records = Path(content["tsv"][0]), []
    try:
        for idx, line in enumerate(content["tsv"][1:]):
            parts = line.split("\t")
            if len(parts) not in (4, 5) or parts[0] != LANGUAGE_CODE:
                return None
            video, audio = (Path(name) for name in parts[1:3])
            records.append(Record(str((root / video).resolve()), str((root / audio).resolve()),
                                  int(parts[3]), content["wrd"][idx], content["ltr"][idx]))
    except (ValueError, OSError):
        return None
    return records


def _same_records(existing: list[Record] | None, current: list[Record]) -> bool:
    if existing is None or len(existing) != len(current):
        return False
    old = {r.video: r for r in existing}
    new = {r.video: r for r in current}
    return len(old) == len(existing) and len(new) == len(current) and old == new


def _write(folder: Path, split: str, records: list[Record]) -> None:
    with tempfile.TemporaryDirectory(prefix=f".{split}-", dir=folder) as tmp:
        staging = split_paths(Path(tmp), split)
        with staging["tsv"].open("w", encoding="utf-8", newline="\n") as tsv, \
             staging["wrd"].open("w", encoding="utf-8", newline="\n") as wrd, \
             staging["ltr"].open("w", encoding="utf-8", newline="\n") as ltr:
            tsv.write("/\n")
            for row in records:
                tsv.write(f"{LANGUAGE_CODE}\t{row.video}\t{row.audio}\t{row.frames}\n")
                wrd.write(row.wrd + "\n")
                ltr.write(row.ltr + "\n")
        for ext, destination in split_paths(folder, split).items():
            os.replace(staging[ext], destination)


def prepare_manifests(settings_by_stage: dict[int, dict]) -> None:
    groups: dict[Path, list[tuple[int, dict]]] = {}
    for stage, settings in settings_by_stage.items():
        groups.setdefault(local_path(settings["paths"]["manifest_dir"]), []).append((stage, settings))
    for folder, members in groups.items():
        first = members[0][1]
        root = local_path(first["paths"]["db_root"])
        options = first.get("data", {})
        split_map = options.get("splits", {"train": "TRAIN", "validation": "VALIDATION"})
        seed, shuffle = int(options.get("seed", 42)), bool(options.get("shuffle", True))
        for stage, settings in members:
            if local_path(settings["paths"]["db_root"]) != root or settings.get("data", {}) != options:
                raise ValueError("Configuratiile care folosesc aceleasi manifeste trebuie sa aiba aceleasi paths.db_root si data.")
        if not isinstance(split_map, dict) or not all(s in split_map for s in ("train", "validation")):
            raise ValueError("data.splits trebuie sa defineasca train si validation.")
        require_audio = any(stage == 1 for stage, _ in members)
        with locked_directory(folder):
            pending = []
            for split, source_split in split_map.items():
                split_paths(folder, split)
                if not isinstance(source_split, str) or not source_split:
                    raise ValueError(f"data.splits.{split} trebuie sa indice un director.")
                records = _scan(root, source_split, require_audio, seed, shuffle)
                existing = _read_existing(folder, split)
                if _same_records(existing, records):
                    logger.info("%s: manifeste verificate, reutilizate (%d exemple); ordinea existenta este pastrata.", split, len(records))
                else:
                    resumable = [local_path(s["paths"]["output_dir"]) / "checkpoint_last.pt" for _, s in members]
                    if any(p.is_file() for p in resumable):
                        raise ValueError(
                            f"{split}: manifestele lipsesc sau datele s-au schimbat, dar exista checkpoint_last.pt. "
                            "Pentru un experiment nou, configurati directoare de rezultate goale; "
                            "nu reluati o rulare cu alte date. Manifestele nu au fost rescrise."
                        )
                    pending.append((split, records))
                for stage, settings in members:
                    limit = settings.get("task", {}).get("max_sample_size", 600 if stage == 1 else 500)
                    kept = sum(r.frames <= limit for r in records) if limit is not None else len(records)
                    logger.info("%s / stage %d: %d exemple in limita de %s cadre (%d peste limita).",
                                split, stage, kept, limit, len(records) - kept)
                    if kept == 0:
                        raise ValueError(f"{split}: niciun exemplu utilizabil pentru stage {stage}.")
            for split, records in pending:
                _write(folder, split, records)
                logger.info("%s: TSV/WRD/LTR pregatite (%d exemple).", split, len(records))


def check_manifests(settings: dict) -> None:
    """Cheap startup check; the full media validation belongs to setupFramework.py."""
    folder = local_path(settings["paths"]["manifest_dir"])
    for split in ("train", "validation"):
        paths = split_paths(folder, split)
        counts = {}
        for ext, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Lipseste {path}. Rulati: python setupFramework.py")
            with path.open(encoding="utf-8") as handle:
                counts[ext] = sum(1 for _ in handle) - (1 if ext == "tsv" else 0)
        if counts["tsv"] <= 0 or len(set(counts.values())) != 1:
            raise ValueError(f"{split}: manifeste incomplete sau nealiniate. Rulati: python setupFramework.py")
