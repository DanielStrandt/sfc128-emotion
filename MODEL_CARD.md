# Model Card — SFC-128 Emotion v2

## Model overview

- **Name:** SFC-128 Emotion v2
- **Architecture:** 128 *independent* REN-T + causal depthwise convolution + low-rank channel mixer + bounded Floating-gate blocks; masked mean pooling; 6-class classifier head.
- **Parameters:** 792,454 (includes embeddings and classification head).
- **Width:** 96 hidden channels; 8 REN-T channel groups; RMS blending α = 0.75.
- **Vocabulary:** original custom 1,000-ID ASCII-constrained BPE from FineWeb-Edu pretraining (fixed token IDs; embeddings fully fine-tuned).
- **Task:** classify short English text into **sadness, joy, love, anger, fear, surprise**.
- **Outputs:** categorical class probabilities after temperature scaling (T = 1.418506 for the exported experiment).
- **Weights:** the trained export is included under `models/SFC128_Emotion/` and licensed under Apache-2.0 with this repository; see [`LICENSE`](LICENSE). Third-party dataset and corpus terms remain separate; see [`LICENSE_REQUIRED.md`](LICENSE_REQUIRED.md).

## Training

The foundation was pretrained for two passes over about 200M custom-token FineWeb-Edu samples (~400M token prediction exposures). Emotion v2 then fully fine-tuned all 128 layers on `dair-ai/emotion` using the combined official split and larger unsplit collection, after de-duplication and validation/test overlap filtering (367,961 training examples). Class-balanced sampling with unweighted cross-entropy was used. The best checkpoint was selected by validation macro-F1 at optimizer step **24,000**. Training stopped at step **28,000** after validation macro-F1 stopped improving (eight bad evaluations). At step 28,000, validation macro-F1 was 0.851307 and accuracy was 0.86987; this was not the selected checkpoint.

## Evaluation

The reported held-out test is the official DAIR.AI `split/test` after deleting three rows duplicated with the holdout or internally: **1,997 test examples**. The official split was not resampled to balance labels.

| Metric | Measured result |
|---|---:|
| Accuracy | **86.3295%** (1724/1997) |
| Macro-F1 | **0.820221** |
| Weighted-F1 | 0.8631 |
| Macro-precision | 0.8254 |
| Macro-recall | 0.8178 |
| Cross-entropy (uncalibrated) | 0.56753 |
| Test accuracy bootstrap 95% CI | 84.78%–87.88% |
| Test macro-F1 bootstrap 95% CI | 0.7965–0.8427 |
| Best validation accuracy | 86.9369% |
| Best validation macro-F1 | 0.8533 |

Per-class test scores:

| Label | Precision | Recall | F1 | Test support |
|---|---:|---:|---:|---:|
| sadness | 0.8581 | 0.9157 | 0.8859 | 581 |
| joy | 0.9100 | 0.8759 | 0.8926 | 693 |
| love | 0.7222 | 0.7405 | 0.7312 | 158 |
| anger | 0.8498 | 0.9055 | 0.8768 | 275 |
| fear | 0.9153 | 0.7723 | 0.8378 | 224 |
| surprise | 0.6970 | 0.6970 | 0.6970 | 66 |

The owner-provided final evaluator completed all 1,997 examples and reported these scores. It evaluated the best-validation checkpoint at step 24,000 on CUDA with BF16 and batch size 16; no training or checkpoint changes occurred during this evaluation. These scores are taken from the provided training/evaluation logs; this repository's synthetic unit tests do **not** establish these benchmark scores independently.

### Evaluation-protocol disclosure

The held-out test was inspected *once before training finished* and repeated after completion. The trainer's best checkpoint was still chosen **only by validation macro-F1**. However, the test set was no longer pristine for the broader development process; treat this as an exploratory development result. Do not select hyperparameters or checkpoints based on repeated test scores.

### Comparability

Other DAIR.AI results often train on the standard 16K training split; this experiment used a much larger deduplicated training set. Comparisons require matching training data, evaluation cleaning, computational budget and random seeds. Parameter count alone does not determine inference speed, particularly with 128 sequential layers.

## Intended use

- Research and prototyping of compact six-class English text emotion classifiers.
- Assistive tagging of short, informal English text with optional human review.
- Local experiments on classifier speed and architecture ablations.

## Out-of-scope use

- **Not** a mental-health assessment, medical diagnosis, crisis detection system or authoritative measurement of someone's feelings.
- Not validated for non-English text, long-form documents, multimodal content, dialects or domain shifts.
- Not a substitute for human oversight in consequential decisions about individuals.

## Known limitations

- The 1,000-token ASCII-constrained tokenizer may reject many Unicode characters; some common punctuation is normalized.
- Input text longer than the configured max token count is truncated. Inference returns a `truncated` indicator.
- Some classes are substantially weaker: love (F1 0.7312) and surprise (F1 0.6970); surprise had only 66 held-out examples.
- Training used balanced sampling, so deployment label frequencies may differ from training priors; confidence is temperature-calibrated on validation data, not proven calibrated across domains.
- Error behavior on negation, sarcasm, code-switching, ambiguous emotions, and unseen domains has not been comprehensively established.

## Reproducibility and files

Reference model weights are saved by the original Stage 2 trainer as `SFC128_Emotion/model.pt` alongside `tokenizer_1k.json`, `config.json`, `labels.json`, and `metrics.json`. Use `python tools/attach_model.py --source ...` to copy and verify them. The original trainer and held-out evaluator are retained at repo root; see `docs/EVALUATION.md` for the exact rerun command.

## Provenance and permissions

The repository source and included trained export are licensed under Apache-2.0; see [`LICENSE`](LICENSE). This license does not relicense third-party datasets or pretraining corpora. Review their current terms and attribution requirements before redistributing those materials or making downstream rights claims; see [`LICENSE_REQUIRED.md`](LICENSE_REQUIRED.md).
