"""Disk cache keyed by sha256 of the input, so benchmark reruns and demos are free and offline."""
import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def get(k: str) -> dict | None:
    p = CACHE_DIR / f"{k}.json"
    return json.loads(p.read_text()) if p.exists() else None


def put(k: str, value: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / f"{k}.json").write_text(json.dumps(value))
