#!/usr/bin/env python3
"""Read-only display of metrics stored in the trainer's exported model.

This DOES NOT rerun the official benchmark. Use the full eval script and the
original run's tokenized test cache for actual benchmark reproducibility.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sfc128_emotion.paths import resolve_model_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=None)
    args = p.parse_args()
    file = resolve_model_dir(args.model_dir) / "metrics.json"
    if not file.is_file():
        raise SystemExit(f"Missing exported metrics file: {file}")
    payload = json.loads(file.read_text(encoding="utf-8"))
    print(json.dumps({"validation": payload.get("validation"), "test": payload.get("test")}, indent=2))
    print("This reads saved metrics; it is not a new benchmark run.")


if __name__ == "__main__":
    main()
