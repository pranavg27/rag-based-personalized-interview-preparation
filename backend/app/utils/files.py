"""Secure upload handling: extension allow-list, size cap, magic-byte checks, random filenames."""
from __future__ import annotations

import asyncio
import zipfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.errors import BadRequestError, ValidationFailed
from app.utils.ids import new_id

CHUNK = 1024 * 256


@dataclass
class SavedFile:
    path: Path
    original_name: str
    ext: str
    size: int


def _ext(filename: str) -> str:
    return Path(filename or "").suffix.lower().lstrip(".")


def _check_magic(path: Path, ext: str) -> None:
    head = path.read_bytes()[:8]
    if ext == "pdf" and not head.startswith(b"%PDF"):
        raise ValidationFailed("File is not a valid PDF.")
    if ext == "docx":
        if not head.startswith(b"PK") or not zipfile.is_zipfile(path):
            raise ValidationFailed("File is not a valid DOCX.")
    if ext == "txt":
        data = path.read_bytes()[:4096]
        if b"\x00" in data:
            raise ValidationFailed("File is not a valid text file.")


async def save_upload(
    upload: UploadFile, *, base_dir: Path, user_id: str, allowed: set[str], max_bytes: int
) -> SavedFile:
    original = Path(upload.filename or "upload").name[:200]
    ext = _ext(original)
    if ext not in allowed:
        raise ValidationFailed(f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(allowed))}.")
    target_dir = base_dir / user_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{new_id()}.{ext}"
    size = 0
    try:
        with target.open("wb") as fh:
            while chunk := await upload.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise BadRequestError(f"File too large (limit {max_bytes // (1024 * 1024)} MB).")
                fh.write(chunk)
        if size == 0:
            raise ValidationFailed("Uploaded file is empty.")
        await asyncio.to_thread(_check_magic, target, ext)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return SavedFile(path=target, original_name=original, ext=ext, size=size)


def remove_file(path: str | Path | None) -> None:
    if path:
        Path(path).unlink(missing_ok=True)
