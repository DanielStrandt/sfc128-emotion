"""Reproducible eager inference latency measurements (no hidden compilation)."""
from __future__ import annotations

import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .runtime import EmotionEngine


def benchmark_inference(model_dir: Path | str, text: str, batch_size: int,
                        device: str, precision: str, warmup: int, iterations: int,
                        cuda_graphs: bool | None = None) -> dict:
    if not 1 <= batch_size <= 256:
        raise ValueError("batch_size must be 1..256")
    if not 0 <= warmup <= 1000 or not 1 <= iterations <= 100000:
        raise ValueError("warmup/iterations out of range")
    engine = EmotionEngine(model_dir=model_dir, device=device, precision=precision,
                           cuda_graphs=cuda_graphs)
    examples = [text] * batch_size

    def sync():
        if engine.device.type == "cuda":
            torch.cuda.synchronize(engine.device)

    for _ in range(warmup):
        engine.predict_many(examples, batch_size=batch_size)
    sync()
    times_ms = []
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)
    for _ in range(iterations):
        sync()
        start = time.perf_counter()
        engine.predict_many(examples, batch_size=batch_size)
        sync()
        times_ms.append((time.perf_counter() - start) * 1000.0)
    p50 = statistics.median(times_ms)
    p95 = float(np.percentile(np.asarray(times_ms), 95))
    return {
        "model": "SFC-128 Emotion v2",
        "platform": platform.system(),
        "torch_version": torch.__version__,
        "device": str(engine.device),
        "precision": precision,
        "cuda_graphs_enabled": engine.cuda_graphs_enabled,
        "cuda_graphs_used": engine.cuda_graph_captures > 0,
        "cuda_graph_captures": engine.cuda_graph_captures,
        "cuda_graph_capture_ms_total": round(engine.cuda_graph_capture_ms_total, 3),
        "cuda_graph_cache_evictions": engine.cuda_graph_cache_evictions,
        "cuda_graph_error": engine.cuda_graph_error,
        "input_tokens": len(engine.tokenizer.encode(text)),
        "batch_size": batch_size,
        "warmup_runs": warmup,
        "timed_runs": iterations,
        "p50_batch_ms": round(p50, 3),
        "p95_batch_ms": round(p95, 3),
        "examples_per_second_at_p50": round(batch_size * 1000.0 / p50, 3) if p50 else None,
        "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated(engine.device)
                                      if engine.device.type == "cuda" else None),
    }
