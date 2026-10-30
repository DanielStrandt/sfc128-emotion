"""Bounded-batch, calibrated inference with the unchanged Stage 2 v2 predictor.

The source architecture, tokenization, pooling and trained weights remain intact.
This wrapper provides batching and optional CUDA autocast for serving.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import logging
import os
from pathlib import Path
from threading import RLock
import time
from typing import Literal

import torch

from sfc128_emotion_v2_inference import EmotionPredictor
from sfc128_emotion_tokenizer import normalize_english_ascii

Precision = Literal["fp32", "bf16", "fp16"]
logger = logging.getLogger(__name__)


class _CapturedBatch:
    """One fixed-shape CUDA graph and the stable buffers it replays against."""

    def __init__(self, graph: torch.cuda.CUDAGraph, input_ids: torch.Tensor,
                 probabilities: torch.Tensor) -> None:
        self.graph = graph
        self.input_ids = input_ids
        self.probabilities = probabilities


class EmotionEngine(EmotionPredictor):
    _GRAPH_TOKEN_BUCKETS = (16, 32, 64, 128)
    _GRAPH_MAX_BATCH = 32
    _GRAPH_CACHE_LIMIT = 8

    def __init__(self, model_dir: str | Path, device: str = "auto", precision: Precision = "fp32",
                 cuda_graphs: bool | None = None):
        if precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("precision must be fp32, bf16, or fp16")
        super().__init__(model_dir=model_dir, device=device)
        if precision != "fp32" and self.device.type != "cuda":
            raise ValueError("bf16/fp16 autocast requires CUDA; use fp32 on CPU")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA device does not report bf16 support")
        self.precision = precision
        if cuda_graphs is None:
            configured = os.environ.get("SFC128_CUDA_GRAPHS", "auto").strip().lower()
            if configured not in {"auto", "1", "true", "yes", "0", "false", "no"}:
                raise ValueError("SFC128_CUDA_GRAPHS must be auto, 1, or 0")
            if configured == "auto":
                cuda_graphs = self.device.type == "cuda" and precision == "fp32"
            else:
                cuda_graphs = configured in {"1", "true", "yes"}
        if cuda_graphs and (self.device.type != "cuda" or precision != "fp32"):
            raise ValueError("CUDA Graph inference currently requires CUDA with fp32 precision")
        self.cuda_graphs_enabled = bool(cuda_graphs)
        self._cuda_graphs: OrderedDict[tuple[int, int], _CapturedBatch] = OrderedDict()
        self._cuda_graph_lock = RLock()
        self.cuda_graph_captures = 0
        self.cuda_graph_capture_ms_total = 0.0
        self.cuda_graph_cache_evictions = 0
        self.cuda_graph_error: str | None = None

    def predict_many(self, texts: list[str], top_k: int = 3, batch_size: int = 16) -> list[dict]:
        if not 1 <= batch_size <= 256:
            raise ValueError("batch_size must be between 1 and 256")
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        if not texts:
            return []
        if not 1 <= top_k <= 6:
            raise ValueError("top_k must be 1..6")
        if self.cuda_graphs_enabled and batch_size <= self._GRAPH_MAX_BATCH:
            try:
                return self._predict_many_cuda_graphs(texts, top_k, batch_size)
            except Exception as exc:
                # The model remains usable through the well-tested eager path if
                # a driver/device cannot capture or replay a particular graph.
                self.cuda_graphs_enabled = False
                self.cuda_graph_error = f"{type(exc).__name__}: {exc}"
                logger.warning("CUDA Graph inference disabled; falling back to eager: %s",
                               self.cuda_graph_error)
                with self._cuda_graph_lock:
                    self._clear_cuda_graphs()
        return self._predict_many_eager(texts, top_k, batch_size)

    def _predict_many_eager(self, texts: list[str], top_k: int, batch_size: int) -> list[dict]:
        result: list[dict] = []
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.precision)
        for offset in range(0, len(texts), batch_size):
            context = (torch.autocast(device_type="cuda", dtype=dtype)
                       if dtype is not None else nullcontext())
            with context:
                result.extend(super().predict_many(texts[offset:offset + batch_size], top_k=top_k))
        return result

    def _predict_many_cuda_graphs(self, texts: list[str], top_k: int,
                                  batch_size: int) -> list[dict]:
        encodings: list[list[int]] = []
        truncated: list[bool] = []
        for text in texts:
            clean = normalize_english_ascii(text).strip()
            if not clean:
                raise ValueError("cannot classify empty text")
            ids = self.tokenizer.encode(clean)
            truncated.append(len(ids) > self.max_text_tokens)
            ids = ids[:self.max_text_tokens]
            if not ids:
                raise ValueError("no valid tokens in text")
            encodings.append(ids)

        result: list[dict] = []
        chunk_size = min(batch_size, self._GRAPH_MAX_BATCH)
        for offset in range(0, len(encodings), chunk_size):
            chunk = encodings[offset:offset + chunk_size]
            bucket = self._token_bucket(max(map(len, chunk)))
            if bucket is None:
                # Unusual exports with longer configured inputs stay supported.
                return self._predict_many_eager(texts, top_k, batch_size)
            cpu_ids = torch.full((len(chunk), bucket), self.cfg.pad_token_id,
                                 dtype=torch.long)
            for row, ids in enumerate(chunk):
                cpu_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)

            key = (len(chunk), bucket)
            with self._cuda_graph_lock, torch.inference_mode():
                entry = self._cuda_graphs.get(key)
                if entry is None:
                    entry = self._capture_batch(key, cpu_ids)
                    self._cuda_graphs[key] = entry
                    self.cuda_graph_captures += 1
                    self._evict_cuda_graphs_if_needed()
                else:
                    self._cuda_graphs.move_to_end(key)
                entry.input_ids.copy_(cpu_ids)
                entry.graph.replay()
                probabilities = entry.probabilities.cpu()

            for local_row, probability in enumerate(probabilities):
                ranked = torch.argsort(probability, descending=True).tolist()
                token_count = len(chunk[local_row])
                original_index = offset + local_row
                result.append({
                    "label": self.labels[ranked[0]],
                    "confidence": float(probability[ranked[0]]),
                    "top_predictions": [
                        {"label": self.labels[index], "probability": float(probability[index])}
                        for index in ranked[:top_k]
                    ],
                    "token_count": token_count,
                    "truncated": truncated[original_index],
                })
        return result

    @classmethod
    def _token_bucket(cls, length: int) -> int | None:
        return next((bucket for bucket in cls._GRAPH_TOKEN_BUCKETS if length <= bucket), None)

    def _captured_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Inputs are checked and guaranteed non-empty before replay. Deriving the
        # mask inside the graph keeps token values and sequence lengths dynamic.
        mask = input_ids.ne(self.cfg.pad_token_id)
        lengths = mask.sum(dim=1).clamp_min(1).float()
        states = self.model._run_core(input_ids, mask)
        pooled = (states.float() * mask.unsqueeze(-1).float()).sum(dim=1)
        pooled = pooled / lengths.unsqueeze(-1)
        logits = self.model.decision_head(pooled).float()
        return torch.softmax(logits / self.temperature, dim=-1)

    def _capture_batch(self, key: tuple[int, int], cpu_ids: torch.Tensor) -> _CapturedBatch:
        capture_started = time.perf_counter()
        static_ids = torch.empty(cpu_ids.shape, dtype=torch.long, device=self.device)
        static_ids.copy_(cpu_ids)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(3):
                self._captured_forward(static_ids)
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            probabilities = self._captured_forward(static_ids)
        torch.cuda.synchronize(self.device)
        self.cuda_graph_capture_ms_total += (time.perf_counter() - capture_started) * 1000.0
        logger.info("Captured SFC-128 CUDA Graph for batch=%d, token_bucket=%d", *key)
        return _CapturedBatch(graph, static_ids, probabilities)

    def _evict_cuda_graphs_if_needed(self) -> None:
        evicted = False
        while len(self._cuda_graphs) > self._GRAPH_CACHE_LIMIT:
            _, old = self._cuda_graphs.popitem(last=False)
            old.graph.reset()
            del old
            self.cuda_graph_cache_evictions += 1
            evicted = True
        if evicted:
            torch.cuda.empty_cache()

    def _clear_cuda_graphs(self) -> None:
        for entry in self._cuda_graphs.values():
            entry.graph.reset()
        self._cuda_graphs.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def predict(self, text: str, top_k: int = 3) -> dict:
        return self.predict_many([text], top_k=top_k, batch_size=1)[0]
