"""Check model-export integrity without changing the checkpoint.

Only load trusted model.pt files (PyTorch checkpoint files are a serialized format).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from semantic_floating_ren_t_128 import SFCConfig
from sfc128_emotion_meanpool_model import SemanticFloatingMeanPoolClassifier
from sfc128_emotion_tokenizer import SFCAsciiTokenizer

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EXPECTED_PARAMETERS = 792454


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_export(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Backward compatibility only; use checkpoints you created/trust.
        return torch.load(path, map_location="cpu")


def verify_export(model_dir: Path | str, *, require_standard: bool = True,
                  run_inference: bool = False) -> dict:
    folder = Path(model_dir).expanduser().resolve()
    needed = ("model.pt", "tokenizer_1k.json", "config.json", "labels.json")
    missing = [name for name in needed if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete exported model at {folder}: missing {', '.join(missing)}. "
                                "Run: python tools/attach_model.py --source <export-directory>")
    payload = load_export(folder / "model.pt")
    if payload.get("architecture") != "SemanticFloatingMeanPoolClassifier":
        raise ValueError("Model architecture is not the Stage 2 v2 mean-pool classifier")
    if payload.get("readout") != "masked_mean":
        raise ValueError("Expected masked_mean readout, not DECIDE token")
    if payload.get("labels") != LABELS:
        raise ValueError("The model's six labels differ from the trained label mapping")
    temperature = float(payload.get("temperature", 1.0))
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError("Invalid calibration temperature")
    cfg = SFCConfig(**payload["config"])
    if cfg.num_classes != 6 or cfg.vocab_size != 1000 or cfg.pad_token_id != 0:
        raise ValueError("Checkpoint classification/vocabulary config mismatch")
    tok = SFCAsciiTokenizer.load(folder / "tokenizer_1k.json")
    if tok.pad_id != cfg.pad_token_id or len(tok.vocab) != cfg.vocab_size:
        raise ValueError("Tokenizer and embedding vocabulary do not match")
    cfg_json = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if cfg_json.get("labels") != LABELS or cfg_json.get("model") != payload["config"]:
        # JSON turns tuples into lists; normalize via canonical JSON string instead.
        from dataclasses import asdict
        if cfg_json.get("labels") != LABELS or cfg_json.get("model") != json.loads(json.dumps(asdict(cfg))):
            raise ValueError("Exported config.json differs from model.pt")
    label_file = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    if label_file != {str(i): label for i, label in enumerate(LABELS)}:
        raise ValueError("Exported labels.json differs from model.pt")
    max_text_tokens = int(cfg_json.get("max_text_tokens", 128))
    if not 1 <= max_text_tokens <= cfg.max_context:
        raise ValueError("Invalid max_text_tokens in metadata")
    for key, value in payload["state_dict"].items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint contains a non-finite or invalid tensor: {key}")
    model = SemanticFloatingMeanPoolClassifier(cfg)
    model.load_state_dict(payload["state_dict"], strict=True)
    count = sum(parameter.numel() for parameter in model.parameters())
    if require_standard and (cfg.depth != 128 or cfg.hidden_size != 96 or count != EXPECTED_PARAMETERS):
        raise ValueError(f"Unexpected trained architecture: depth={cfg.depth}, width={cfg.hidden_size}, params={count}")
    if run_inference:
        model.eval()
        with torch.inference_mode():
            input_ids = torch.tensor([tok.encode("I feel happy!")], dtype=torch.long)
            logits = model(input_ids).logits
            if logits.shape != (1, 6) or not torch.isfinite(logits).all():
                raise ValueError("Inference produced wrong shape or nonfinite logits")
    return {
        "status": "ok",
        "model_dir": str(folder),
        "architecture": payload["architecture"],
        "readout": payload["readout"],
        "depth": cfg.depth,
        "hidden_size": cfg.hidden_size,
        "parameters": count,
        "vocab_size": len(tok.vocab),
        "labels": LABELS,
        "temperature": temperature,
        "max_text_tokens": max_text_tokens,
        "model_sha256": sha256(folder / "model.pt"),
        "tokenizer_sha256": sha256(folder / "tokenizer_1k.json"),
        "inference_verified": bool(run_inference),
    }
