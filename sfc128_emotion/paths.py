"""Consistent local model discovery for CLI/API/verification."""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_model_dir(value: str | Path | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    env = os.environ.get("SFC128_MODEL_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (PROJECT_ROOT / "models" / "SFC128_Emotion").resolve()
