# CPU-first reference server; can be large because PyTorch is included.
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SFC128_MODEL_DIR=/app/models/SFC128_Emotion
WORKDIR /app
COPY pyproject.toml README.md ./
COPY semantic_floating_ren_t_128.py sfc128_emotion_meanpool_model.py sfc128_emotion_tokenizer.py sfc128_emotion_v2_inference.py ./
COPY sfc128_emotion/ ./sfc128_emotion/
RUN python -m pip install --no-cache-dir ".[api]"
COPY models/SFC128_Emotion/ ./models/SFC128_Emotion/
EXPOSE 8000
CMD ["uvicorn", "sfc128_emotion.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
