from __future__ import annotations

import pytest
import torch

from sfc128_emotion.runtime import EmotionEngine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph test requires a CUDA device")
def test_cuda_graph_predictions_match_eager(synthetic_export, monkeypatch):
    monkeypatch.delenv("SFC128_CUDA_GRAPHS", raising=False)
    eager = EmotionEngine(synthetic_export, device="cuda", cuda_graphs=False)
    graphed = EmotionEngine(synthetic_export, device="cuda", cuda_graphs=True)
    texts = ["Today is a wonderful day", "I feel calm and hopeful"]

    expected = eager.predict_many(texts, top_k=6, batch_size=2)
    actual = graphed.predict_many(texts, top_k=6, batch_size=2)

    assert graphed.cuda_graphs_enabled
    assert graphed.cuda_graph_captures == 1
    for reference, optimized in zip(expected, actual):
        assert optimized["label"] == reference["label"]
        assert optimized["token_count"] == reference["token_count"]
        assert optimized["truncated"] == reference["truncated"]
        for ref_top, opt_top in zip(reference["top_predictions"], optimized["top_predictions"]):
            assert ref_top["label"] == opt_top["label"]
            assert opt_top["probability"] == pytest.approx(ref_top["probability"], abs=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph test requires a CUDA device")
def test_cuda_graph_cache_evicts_old_shapes_and_keeps_running(synthetic_export, monkeypatch):
    monkeypatch.delenv("SFC128_CUDA_GRAPHS", raising=False)
    engine = EmotionEngine(synthetic_export, device="cuda", cuda_graphs=True)
    engine._GRAPH_CACHE_LIMIT = 2

    for size in (1, 2, 3):
        assert len(engine.predict_many(["Today feels good"] * size,
                                       top_k=1, batch_size=size)) == size

    assert engine.cuda_graph_captures == 3
    assert engine.cuda_graph_cache_evictions == 1
    assert len(engine.predict_many(["Today feels good"] * 2,
                                   top_k=1, batch_size=2)) == 2
    assert engine.cuda_graphs_enabled
