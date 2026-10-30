# Deployment guide

## 1. Attach your model

The user's trained weights live outside this repository until attached. From the repo root:

```powershell
python tools/attach_model.py --source "D:\path\to\SemanticFloat\sfc_emotion_v2_run\SFC128_Emotion"
```

The destination is `models/SFC128_Emotion/`. The source is preserved. This tool performs strict checks on the checkpoint's architecture, the six label IDs, fixed 1K BPE, parameter count and SHA256 copy integrity. Use `--replace` only when you intentionally want to overwrite existing model artifacts.

You can instead leave the model where it was exported and set a model path:

```powershell
$env:SFC128_MODEL_DIR="D:\path\to\SemanticFloat\sfc_emotion_v2_run\SFC128_Emotion"
```

## 2. Local CLI and API

Install with `python -m pip install -e ".[api]"`. Start the API:

```powershell
python -m uvicorn sfc128_emotion.api:app --host 127.0.0.1 --port 8000
```

API documentation is at `http://127.0.0.1:8000/docs`.

The server loads the model once during startup; a missing model causes startup to fail (there is no silent fallback). CPU is supported. On a CUDA-enabled PyTorch installation, `auto` chooses CUDA and FP32 inference uses CUDA Graph replay by default. Each new batch/token-length bucket is captured on first use, so that first call is slower; later calls replay the captured graph. The REST service serializes inference calls to bound concurrent GPU load and protect the graph's reusable input buffers. The maximum batch request is 32 texts, each at most 4,096 characters. The tokenizer then truncates tokens to the original model's configured limit. Set `SFC128_CUDA_GRAPHS=0` to disable graph replay and use eager PyTorch.

## 3. Docker

First attach weights. Then build/run a simple local CPU-first reference server:

```powershell
docker build -t sfc128-emotion:1.0 .
```

```powershell
docker run --rm -p 127.0.0.1:8000:8000 sfc128-emotion:1.0
```

The image includes your model files if they were in `models/SFC128_Emotion/` when you built it. Images can be large because they include PyTorch. This Dockerfile does not configure NVIDIA Container Toolkit or GPU-enabled PyTorch; treat it as a portable **CPU reference**. For GPU deployment, install a compatible CUDA PyTorch build and configure Docker's NVIDIA runtime separately.

## 4. Performance

Compare eager inference with the default CUDA Graph path on your actual hardware:

```powershell
python -m sfc128_emotion benchmark --device cuda --batch-size 1 --warmup 5 --iterations 30 --eager
```

```powershell
python -m sfc128_emotion benchmark --device cuda --batch-size 16 --warmup 5 --iterations 30
```

Measurements are synchronized on CUDA and include tokenization, batching, model forward, softmax and CPU conversion. Record batch sizes and text length when comparing versions. Graph replay preserves full-precision model calculations and weights; it reduces CPU overhead from launching the many small GPU operations. Warm the common input shapes before recording steady-state latency. BF16 autocast can be tested explicitly with `--precision bf16`, but it was slower than FP32 on the RTX 3060 used for this project's comparison. The graph path falls back to eager if capture is unsupported. No quantization or custom CUDA kernels are used.

## 5. Security and reproducibility

- Do not expose the sample API to the public internet unprotected: **no authentication, rate limiting or TLS termination** is provided. Use a trusted reverse proxy, access control, request-size limits and monitoring.
- Load `model.pt` only from a checkpoint you trust; PyTorch serialized files should not be treated as untrusted user uploads.
- The API processes text in memory; it does not persist inputs by default. Your hosting proxy/logger may have different privacy behavior.
- Keep `model.pt` and `tokenizer_1k.json` together and track their SHA256 hashes.
- The full original FineWeb foundation and Hugging Face dataset downloads are **not needed for inference**.
- For production, pin Python and dependency versions after validating your target OS/GPU stack; the dependency ranges are intended for project portability, not bitwise reproducibility.

## 6. GitHub repo publication

From the extracted folder:

```powershell
git init
```

```powershell
git add .
```

```powershell
git commit -m "Initial SFC-128 Emotion v2 repository"
```

The source and included model export are distributed under Apache-2.0 (see the repository `LICENSE`). The exported model is ~3MB in raw float32 parameters plus metadata and is generally small enough for ordinary Git, but verify the actual exported file size and platform limits; Git LFS is optional. Dataset and pretraining-corpus rights remain separate; review their terms and attribution requirements before redistributing those materials or making downstream rights claims. Do not paste sensitive paths, tokens or private training data into issues or commits.
