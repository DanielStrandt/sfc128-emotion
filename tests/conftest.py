"""Synthetic fixtures; no actual trained weights or network access required."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from semantic_floating_ren_t_128 import SFCConfig
from sfc128_emotion_meanpool_model import SemanticFloatingMeanPoolClassifier
from sfc128_emotion_tokenizer import SPECIAL, ASCII_CHARS

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def create_synthetic_export(directory: Path, *, depth: int = 2, width: int = 16):
    """Create a structurally valid but RANDOM-WEIGHT export. NOT the trained model."""
    directory.mkdir(parents=True, exist_ok=True)
    vocab = list(SPECIAL) + list(ASCII_CHARS)
    vocab.extend("zz" + chr(ord("a") + i // 676) + chr(ord("a") + (i // 26) % 26) + chr(ord("a") + i % 26)
                 for i in range(895))
    assert len(vocab) == 1000 and len(set(vocab)) == 1000
    vocab[-1] = "ab"  # One valid synthetic merge; tests exercise actual BPE path.
    (directory / "tokenizer_1k.json").write_text(
        json.dumps({"kind": "ascii_constrained_bpe_v1", "vocab": vocab, "merges": [["a", "b"]]}),
        encoding="utf-8")
    cfg = SFCConfig(vocab_size=1000, pad_token_id=0, decision_token_id=3,
                    num_classes=6, depth=depth, hidden_size=width,
                    bottleneck_size=24 if depth == 128 else 8,
                    floating_hidden=32 if depth == 128 else 8,
                    ren_groups=8 if depth == 128 else 4,
                    max_context=128, gradient_checkpointing=False)
    torch.manual_seed(2026)
    model = SemanticFloatingMeanPoolClassifier(cfg).eval()
    torch.save({"architecture": "SemanticFloatingMeanPoolClassifier",
                "config": asdict(cfg), "readout": "masked_mean",
                "state_dict": model.state_dict(), "labels": LABELS,
                "temperature": 1.418506}, directory / "model.pt")
    (directory / "config.json").write_text(json.dumps({"model": asdict(cfg),
        "temperature": 1.418506, "max_text_tokens": 32, "labels": LABELS}), encoding="utf-8")
    (directory / "labels.json").write_text(
        json.dumps({str(i): label for i, label in enumerate(LABELS)}), encoding="utf-8")
    (directory / "metrics.json").write_text(
        json.dumps({"validation": {}, "test": {}}), encoding="utf-8")
    return directory


@pytest.fixture
def synthetic_export(tmp_path):
    return create_synthetic_export(tmp_path / "SFC128_Emotion")
