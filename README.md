# SFC-128 Emotion v2

**A fully fine-tuned, 128-layer, sub-million-parameter emotion classifier.** This repository contains the exact PyTorch architecture and 1,000-token BPE loader used in Stage 2 v2, plus a CLI, FastAPI service, model validator, benchmark utility, tests, Dockerfile, original Stage 2 training/final-evaluation scripts, and the original Stage 1 pretraining script for research provenance.

> **Included model export:** `models/SFC128_Emotion/` contains the model and matching tokenizer/configuration exported by the completed training run. The source export remains in the original training folder. **Do not retrain the tokenizer.** Before publishing, decide whether and under what terms you may distribute the checkpoint and tokenizer.

## Measured result

| Metric | Result |
|---|---:|
| Model | SFC-128 Emotion v2 (masked mean pooling) |
| Trained parameters | **792,454** |
| Independent REN-T/Floating blocks | **128** |
| Hidden channels | **96** |
| BPE vocabulary | **1,000 fixed IDs** |
| DAIR.AI Emotion held-out test | **1,997 examples after audit** |
| Test accuracy | **86.3295%** (1,724 / 1,997) |
| Test macro-F1 | **0.820221** |
| Test weighted-F1 | **0.8631** |
| Selected checkpoint | Step **24,000** (highest validation macro-F1) |
| Calibration temperature | **1.418506** (validation-fitted) |

These measurements are from the completed training/evaluation run recorded by the owner. The final held-out evaluation completed all 1,997 audited examples and reported these scores; it did not change training or checkpoints. Training used ~367,961 deduplicated training examples from the `dair-ai/emotion` split and unsplit configurations, with balanced sampling and natural-frequency validation/test distributions. The test set was inspected during development, so these results describe an exploratory development run rather than a pristine, never-consulted test protocol. See [MODEL_CARD.md](MODEL_CARD.md) and [Evaluation notes](docs/EVALUATION.md).

## Quick start — Windows PowerShell

From the extracted repository folder:

**1. Install the code** (Python 3.10+; PyTorch may need a CUDA-specific install for your NVIDIA setup):

```powershell
python -m pip install -e ".[api]"
```

**2. (Optional) Refresh the included model export** from your original training output. This checks compatibility and copies only deployment artifacts; the original export remains untouched. The current repo already has these files attached:

```powershell
python tools/attach_model.py --source "D:\path\to\SemanticFloat\sfc_emotion_v2_run\SFC128_Emotion"
```

**3. Verify model weights and vocabulary:**

```powershell
python -m sfc128_emotion verify --run-inference
```

**4. Classify a sentence:**

```powershell
python -m sfc128_emotion predict --text "I can't believe everything worked out so well!"
```

On CUDA with FP32, inference automatically uses CUDA Graph replay after the first call for each batch/token-length bucket. CPU and non-FP32 inference use eager PyTorch. Use `--eager` to force the original eager path. CUDA Graphs preserve the checkpoint and FP32 calculations; they reduce repeated GPU launch overhead.

**5. Start a local REST server:**

```powershell
python -m uvicorn sfc128_emotion.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive request/response documentation. The server runs offline: it does **not** download datasets, models, or tokenizers.

You can also avoid copying your weights by supplying the original export path directly:

```powershell
python -m sfc128_emotion predict --model-dir "D:\path\to\SemanticFloat\sfc_emotion_v2_run\SFC128_Emotion" --text "I am thrilled!"
```

## API example

PowerShell:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{"text":"I feel hopeful today!","top_k":3}'
```

`POST /predict` returns the top emotion, its calibrated softmax score, the top-k label distribution, the actual token count and a truncation indicator. Other endpoints: `POST /predict-batch` (up to 32 texts), `GET /health`, and `GET /model-info`.

Sample structure (numbers below are **illustrative**, not measured output):

```json
{
  "label": "joy",
  "confidence": 0.84,
  "top_predictions": [{"label":"joy","probability":0.84}],
  "token_count": 21,
  "truncated": false
}
```

Confidence scores are temperature-scaled from validation data; they are not guarantees or medical interpretations.

## Other commands

**Interactive CLI**

```powershell
python -m sfc128_emotion interactive
```

**Bulk file** — one English sentence per UTF-8 line, results written as JSON Lines:

```powershell
python -m sfc128_emotion predict --input-file examples.txt --output-file predictions.jsonl --batch-size 16
```

**Compare eager and CUDA Graph latency** (warmup and CUDA synchronization included):

```powershell
python -m sfc128_emotion benchmark --device cuda --batch-size 1 --warmup 5 --iterations 30 --eager
python -m sfc128_emotion benchmark --device cuda --batch-size 1 --warmup 5 --iterations 30
python -m sfc128_emotion benchmark --device cuda --batch-size 16 --warmup 5 --iterations 30
```

The graph benchmark reports capture counts and times only steady-state calls after warmup. The first request for a new batch/token-length bucket has graph-capture startup cost. Set `SFC128_CUDA_GRAPHS=0` or pass `--eager` to force eager inference. Graphs are cached for up to eight recently used shapes; unsupported captures fall back to eager automatically.

**Unit tests without your trained weights** (synthetic 1,000-token fixture and tiny synthetic checkpoint):

```powershell
python -m pip install -e ".[api,dev]"; python -m pytest -q
```

**Read the original trainer's saved metrics** (does not rerun the benchmark):

```powershell
python tools/check_reference_metrics.py
```

**Rerun the full held-out test** against your original run directory and cache (see [Evaluation notes](docs/EVALUATION.md)):

```powershell
python evaluate_sfc128_emotion_v2_final.py --run-dir "D:\path\to\SemanticFloat\sfc_emotion_v2_run" --foundation-dir "D:\path\to\SemanticFloat\sfc_fineweb_run" --overwrite
```

## Repository contents

```text
sfc128-emotion-repo/
  README.md                     # Getting started, usage, results
  MODEL_CARD.md                 # Intended use, limitations and test scores
  pyproject.toml                # Editable package + console command
  requirements*.txt            # Base/API/test dependency sets
  semantic_floating_ren_t_128.py     # ORIGINAL 128-layer PyTorch architecture
  sfc128_emotion_meanpool_model.py   # ORIGINAL v2 masked-mean classifier
  sfc128_emotion_tokenizer.py        # ORIGINAL 1K BPE loader
  sfc128_emotion_v2_inference.py     # ORIGINAL exported predictor
  train_sfc128_emotion_v2.py         # Original trainer (optional)
  evaluate_sfc128_emotion_v2_final.py # Original held-out evaluator (optional)
  sfc128_emotion/               # CLI, API, inference wrapper, verification, benchmark
  tools/                        # Install exported weights, show metrics
  models/SFC128_Emotion/        # Included trained model export and matching tokenizer
  tests/                        # Model-free synthetic tests
  docs/                         # Architecture, evaluation, deployment guidance
  research/pretraining/         # Original FineWeb-Edu Stage 1 script + notes (optional)
  research/                     # Archived Stage 2/final-eval instructions
  Dockerfile                    # Optional CPU reference API image
  .github/workflows/tests.yml   # CI; no private model required
```

## Included model artifacts

The attached Stage 2 v2 export contains:

```text
SFC128_Emotion/
  model.pt                 # Required: trained 128-layer weights + configuration
  tokenizer_1k.json        # Required: exact original BPE IDs and merge order
  config.json              # Required: max input token length and model config
  labels.json              # Required: class-ID mapping
  metrics.json             # Recommended: official exported result/audit
  README.txt               # Optional trainer-export note
```

Keep those files together. The `attach_model.py` tool refuses to overwrite existing model files unless `--replace` is explicitly provided. It verifies the original and copied model's SHA256 hashes; **it does not alter your source model**. SHA256 hashes for the attached files are recorded in `models/SFC128_Emotion/SHA256SUMS.txt`.

## Important design properties

- **One model, one task:** six-way emotion classification. No language-model head, no multitasking, no shared specialist weights.
- **Exact pretrained tokenizer:** no new merges, token IDs, or vocabulary training. All training and inference use the same 1,000-token BPE.
- **Masked-mean readout:** all valid token states contribute to the six-class output. Padding is excluded; no appended `DECIDE` token.
- **128 unique REN-T/Floating blocks:** the original PyTorch architecture is preserved verbatim for state-dict compatibility.
- **Input:** short English text. The tokenizer accepts printable ASCII, tab/newline, and explicitly normalized common Unicode punctuation. Other Unicode text is rejected with a clear error; long inputs are truncated to the original training setting (typically 128 tokenizer IDs).
- **Output:** one of `sadness`, `joy`, `love`, `anger`, `fear`, `surprise` with a temperature-calibrated probability.

See [Inference performance](docs/PERFORMANCE.md) for the matched eager-versus-CUDA-Graph benchmark and full validation prediction-parity results.

## Deployment and licensing

The FastAPI example has **no authentication or rate limiting**. Bind to localhost during development; add authentication, TLS, request limits and monitoring before exposing it publicly. The Dockerfile is a CPU-first reference image; see [Deployment guide](docs/DEPLOYMENT.md).

**No license has been assumed for your research code or weights.** Review [LICENSE_REQUIRED.md](LICENSE_REQUIRED.md) before publishing a public repository. Also verify the dataset and pretrained-data usage terms applicable to distributing the checkpoint. A GitHub repository can remain private while you decide on licensing.

