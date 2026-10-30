#!/usr/bin/env python3
"""Install a *locally exported* SFC-128 Emotion v2 model into this repository.

No training and no external downloads. Input files are copied; source is kept.
Only the five/six deployment artifacts are copied, never scripts from export.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sfc128_emotion.verify import verify_export

ARTIFACTS = ("model.pt", "tokenizer_1k.json", "config.json", "labels.json", "metrics.json", "README.txt")


def attach(source: Path, destination: Path, *, replace: bool = False) -> dict:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination:
        raise ValueError("Source and destination are identical")
    if not source.is_dir():
        raise FileNotFoundError(f"Model export directory not found: {source}")
    for name in ARTIFACTS[:4]:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing exported file: {source / name}")
    # Validate BEFORE copying, including strict checkpoint shape compatibility.
    source_report = verify_export(source, require_standard=True, run_inference=False)
    present = [name for name in ARTIFACTS if (source / name).is_file()]
    dest_existing = [name for name in ARTIFACTS if (destination / name).exists()]
    if dest_existing and not replace:
        raise FileExistsError(f"Destination already has model artifacts: {dest_existing}. "
                              "Use --replace only when you deliberately want to overwrite them.")
    destination.mkdir(parents=True, exist_ok=True)
    for name in present:
        shutil.copy2(source / name, destination / name)
    copied_report = verify_export(destination, require_standard=True, run_inference=False)
    if source_report["model_sha256"] != copied_report["model_sha256"] or \
       source_report["tokenizer_sha256"] != copied_report["tokenizer_sha256"]:
        raise RuntimeError("Model or tokenizer checksum differs after copy")
    return copied_report


def main():
    parser = argparse.ArgumentParser(description="Copy trained SFC128_Emotion model into this repository")
    parser.add_argument("--source", required=True, type=Path,
                        help="Original trainer's .../SFC128_Emotion folder")
    parser.add_argument("--destination", type=Path, default=ROOT / "models" / "SFC128_Emotion")
    parser.add_argument("--replace", action="store_true", help="Intentionally replace existing model artifacts")
    args = parser.parse_args()
    report = attach(args.source, args.destination, replace=args.replace)
    print(f"[attached] verified checkpoint: {report['model_dir']}")
    print(f"[model] depth={report['depth']}, parameters={report['parameters']:,}, vocabulary={report['vocab_size']}")
    print(f"[sha256] {report['model_sha256']}")


if __name__ == "__main__":
    main()
