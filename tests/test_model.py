from __future__ import annotations
import json
import pytest
import torch

from sfc128_emotion.runtime import EmotionEngine
from sfc128_emotion.verify import verify_export
from sfc128_emotion_tokenizer import SFCAsciiTokenizer
from sfc128_emotion_meanpool_model import SemanticFloatingMeanPoolClassifier
from semantic_floating_ren_t_128 import SFCConfig


def test_model_strict_load_and_inference(synthetic_export):
    report = verify_export(synthetic_export, require_standard=False, run_inference=True)
    assert report["status"] == "ok"
    assert report["depth"] == 2
    assert report["vocab_size"] == 1000
    assert report["temperature"] == 1.418506
    with pytest.raises(ValueError, match="Unexpected trained architecture"):
        verify_export(synthetic_export, require_standard=True)


def test_probability_and_batch_equivalence(synthetic_export):
    predictor = EmotionEngine(synthetic_export, device="cpu")
    text = "I am afraid!"
    one = predictor.predict(text, top_k=6)
    both = predictor.predict_many([text, "I am happy!"], top_k=6, batch_size=2)
    from sfc128_emotion_v2_inference import EmotionPredictor
    original = EmotionPredictor(synthetic_export, device="cpu")
    legacy = original.predict_many([text, "I am happy!"], top_k=6)
    assert legacy[0]["label"] == both[0]["label"]
    assert abs(legacy[0]["confidence"] - both[0]["confidence"]) < 1e-7
    assert one["label"] == both[0]["label"]
    assert abs(one["confidence"] - both[0]["confidence"]) < 1e-5
    assert abs(sum(item["probability"] for item in one["top_predictions"]) - 1) < 1e-5
    assert one["token_count"] > 0 and not one["truncated"]


def test_token_truncation_and_rejections(synthetic_export):
    predictor = EmotionEngine(synthetic_export, device="cpu")
    result = predictor.predict("x" * 100)
    assert result["truncated"] is True
    assert result["token_count"] == 32
    with pytest.raises(ValueError, match="empty text"):
        predictor.predict("   ")
    with pytest.raises(ValueError, match="non-ASCII"):
        predictor.predict("hello🙂")
    with pytest.raises(ValueError):
        predictor.predict_many(["hello"], batch_size=0)


def test_model_masked_pool_padding(synthetic_export):
    payload = torch.load(synthetic_export / "model.pt", weights_only=True, map_location="cpu")
    cfg = SFCConfig(**payload["config"])
    model = SemanticFloatingMeanPoolClassifier(cfg)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    tokenizer = SFCAsciiTokenizer.load(synthetic_export / "tokenizer_1k.json")
    ids = tokenizer.encode("happy")
    with torch.inference_mode():
        plain = model(torch.tensor([ids], dtype=torch.long)).logits
        padded = model(torch.tensor([ids + [0] * 12], dtype=torch.long)).logits
    assert torch.allclose(plain, padded, atol=1e-5, rtol=1e-5)
