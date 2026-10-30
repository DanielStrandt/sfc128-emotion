# SFC-128 / FineWeb-Edu 200M-token pretraining

Files in this package:

- `train_sfc128_fineweb_edu.py` — complete streaming, custom BPE tokenizer, preparation, and language-model pretraining pipeline.
- `semantic_floating_ren_t_128.py` — the existing independent-weight, causal, 128-block REN-T model; placed beside the trainer.
- `requirements_sfc128.txt` — Python requirements.

## Install

Use a Python environment with a current CUDA-enabled PyTorch installation for practical full pretraining. Then install the dependencies:

```bash
python -m pip install -r requirements_sfc128.txt
```

## First run: smoke test (no internet needed)

```bash
python train_sfc128_fineweb_edu.py --smoke-test
```

This tests a tiny 2-block stand-in with synthetic data, including an interrupted/resumed training job, and verifies both epochs and successful transfer into the classifier. It does NOT measure real 128-block language-model quality.

## Run the full pipeline

```bash
python train_sfc128_fineweb_edu.py --output-dir ./sfc_fineweb_run
```

Defaults:

| Setting | Value |
| --- | --- |
| Source | `HuggingFaceFW/fineweb-edu`, config `sample-10BT` (streaming) |
| Tokenizer sample | ~5,000,000 reference/source GPT-2 tokens; `token_count` metadata where available |
| Tokenizer vocabulary | Exactly 1,000 IDs: 8 specials, 97 ASCII single characters, 895 alphabetic BPE entries |
| Training corpus | ~200,000,000 **custom 1K-tokenizer tokens** (full documents, slight overshoot possible) |
| Additional validation data | ~2,000,000 held-out custom tokens; sampled evaluation subset for speed |
| Pretraining epochs | Exactly 2 complete passes through fixed training token shards |
| Model | 128 independent causal REN-T/Floating convolution blocks, width 96, rank 24 |
| Pretraining output head | Tied to 1,000 x 96 token embeddings |
| Pretraining sequence | 512 tokens (tunable with `--sequence-length`) |
| Batch size | 8, gradient accumulation 4 |
| Precision | bf16 on compatible CUDA GPUs, otherwise fp16 on CUDA or fp32 on CPU |
| Optimizer | AdamW, peak LR 3e-4, 1,000-step warmup, cosine decay |
| Validation | Held-out documents, fixed random subset every 5,000 steps and each epoch |

The tokenizer **only encodes ASCII letters, digits, punctuation, spaces, tab, and newline**. Learned ordinary tokens never contain digits, punctuation, or whitespace. All numeric characters and punctuation remain individual single-character tokens. Eight internal control strings are exceptions to the ordinary-token restriction. Non-ASCII source typography is mapped to ASCII when possible; remaining unsupported characters are replaced with spaces, or documents with too much unsupported content are rejected. Inference encoding rejects unsupported characters rather than silently changing them.

FineWeb-Edu is an English-focused corpus. The script additionally rejects rows with non-English language metadata or low language score when such metadata are provided; an English-only corpus is not mathematically guaranteed merely by ASCII filtering.

The 5M tokenizer-sample count is measured against FineWeb's *reference* token count, because the new tokenizer does not exist yet. The 200M training budget is counted using the completed custom tokenizer.

## Restart or change execution

Run the identical command in the same output directory again. Token shards, tokenizer, model state, optimizer state, mixed-precision scaler, RNG states, step cursor, and epoch cursor are stored to permit resumption. Corpus preparation resumes at last committed whole-doc shard; a small amount of work since that point may be repeated after interruption. For the most reproducible remote stream, optionally pin a dataset revision via `--dataset-revision`.

To **only build the vocabulary**:

```bash
python train_sfc128_fineweb_edu.py --output-dir ./sfc_fineweb_run --tokenizer-only
```

To **build vocabulary and prepare tokens without pretraining**:

```bash
python train_sfc128_fineweb_edu.py --output-dir ./sfc_fineweb_run --prepare-only
```

To run with **1,024-token sequences** (must be chosen at initial training start):

```bash
python train_sfc128_fineweb_edu.py --output-dir ./sfc_fineweb_1024 --sequence-length 1024 --batch-size 2 --grad-accum 16
```

To **pause after 100 optimizer steps** (and later resume without `--max-steps`):

```bash
python train_sfc128_fineweb_edu.py --output-dir ./sfc_fineweb_run --max-steps 100
```

Note: do not change vocabulary, model dimensions, token shard size, sequence length, batch size, gradient accumulation, seed, or dataset configuration when resuming the same run. Use a fresh output directory for different experiments.

## Files written in the output directory

- `tokenizer_sample.jsonl`, `tokenizer_sample_meta.json` — raw ASCII tokenizer training sample, metadata.
- `tokenizer_1k.json` — constrained 1,000-token vocabulary and ordered merges.
- `tokens/train_*.bin`, `tokens/validation_*.bin` — little-endian uint16 ID arrays.
- `tokens/*_manifest.json` — shard counts and source stream offsets.
- `checkpoint_latest.pt` — model + optimizer + scaler + training progress for resume.
- `checkpoint_best_model.pt` — best sampled held-out loss (for evaluation).
- `pretrained_sfc128_backbone.pt` — all pretrained backbone weights without classifier head.
- `train_log.jsonl` — periodic training records.

Token IDs alone for 200M training tokens occupy about **400 MB** (decimal) on disk, plus ~4 MB validation IDs. Extra room is needed for tokenizer sample text, streamed Parquet/cache files, model checkpoints, and training logs.

## Load the resulting backbone for decisions

```python
import torch
from pathlib import Path
from semantic_floating_ren_t_128 import SFCConfig, SemanticFloatingClassifier
from train_sfc128_fineweb_edu import AsciiConstrainedBPE

root = Path("./sfc_fineweb_run")
artifact = torch.load(root / "pretrained_sfc128_backbone.pt", map_location="cpu", weights_only=False)
model = SemanticFloatingClassifier(SFCConfig(**artifact["config"]))
status = model.load_state_dict(artifact["state_dict"], strict=False)
assert set(status.missing_keys) == {"decision_head.weight", "decision_head.bias"}
assert not status.unexpected_keys
tokenizer = AsciiConstrainedBPE.load(root / "tokenizer_1k.json")

# The newly initialized decision head must now be trained on labeled decisions.
ids = torch.tensor([tokenizer.encode("The answer is yes.")], dtype=torch.long)
model.eval()
with torch.no_grad():
    untrained_decision_logits = model(ids).logits
print(untrained_decision_logits.shape)
```

## Important limitations

- The pretraining corpus uses the FineWeb-Edu English-language subset but can still contain noisy web text.
- A vocabulary of 1,000 may expand the number of tokens required to represent English compared to a conventional large vocabulary.
- Long-context competence is **not guaranteed**: defaults train at sequence length 512, despite the architecture having a theoretical receptive field of 2,653 token positions.
- The 128-layer model executes 128 sequential transformations; its small parameter size does not imply low training cost.
- This trainer is for **single-device** pretraining (one CUDA GPU, or CPU testing), not distributed training.
- The untrained decision head needs downstream supervised data; LM pretraining does not by itself create decision accuracy.
- No full 200M-token download or full 128-layer pretraining run was performed during construction of this script.

Dataset attribution: FineWeb-Edu, Hugging Face, https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu (ODC-By; subject to data source terms).
