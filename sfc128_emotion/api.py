"""FastAPI REST app for the standalone emotion classifier.

Starts only when the model directory contains the trained export. No downloads.
Routes: GET /health, GET /model-info, POST /predict, POST /predict-batch.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .paths import resolve_model_dir
from .runtime import EmotionEngine


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4096)
    top_k: int = Field(default=3, ge=1, le=6)


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)
    top_k: int = Field(default=3, ge=1, le=6)


class Prediction(BaseModel):
    label: str
    confidence: float
    top_predictions: list[dict]
    token_count: int
    truncated: bool


def create_app(model_dir: str | Path | None = None, *, predictor=None) -> FastAPI:
    """Optional predictor dependency injection is used solely for offline API tests."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = predictor if predictor is not None else EmotionEngine(resolve_model_dir(model_dir))
        app.state.predict_lock = Lock()  # serialize concurrent GPU/CPU model calls
        yield

    app = FastAPI(title="SFC-128 Emotion v2 API", version="1.0.0", lifespan=lifespan,
                  description="Six-class, English ASCII-constrained emotion prediction")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": True}

    @app.get("/model-info")
    def model_info():
        engine = app.state.engine
        return {"architecture": "SFC-128 REN-T masked-mean emotion",
                "labels": engine.labels,
                "depth": engine.cfg.depth,
                "hidden_size": engine.cfg.hidden_size,
                "vocab_size": engine.cfg.vocab_size,
                "temperature": engine.temperature,
                "max_text_tokens": engine.max_text_tokens,
                "device": str(engine.device),
                "cuda_graphs_enabled": getattr(engine, "cuda_graphs_enabled", False)}

    @app.post("/predict", response_model=Prediction)
    def predict(request: PredictRequest):
        try:
            with app.state.predict_lock:
                return app.state.engine.predict(request.text, top_k=request.top_k)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/predict-batch", response_model=list[Prediction])
    def predict_batch(request: BatchRequest):
        if any(not isinstance(s, str) or not s.strip() or len(s) > 4096 for s in request.texts):
            raise HTTPException(status_code=422, detail="Every text must be 1..4096 characters and nonblank")
        try:
            with app.state.predict_lock:
                return app.state.engine.predict_many(request.texts, top_k=request.top_k, batch_size=16)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
