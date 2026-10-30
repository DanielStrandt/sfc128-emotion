# SFC-128 Emotion v2 architecture

This repository preserves the original model implementation in the exact four Python files used for Stage 2 v2 export:

1. `semantic_floating_ren_t_128.py`
2. `sfc128_emotion_meanpool_model.py`
3. `sfc128_emotion_tokenizer.py`
4. `sfc128_emotion_v2_inference.py`

These are the source modules the trainer originally copied into the deployment directory. The higher-level `sfc128_emotion/` package imports them without renaming any model parameter. The trained checkpoint is `model.pt`, **not** a language-model head.

## Forward path

```text
ASCII-normalized English text
    -> original 1,000-ID constrained BPE (fixed merges)
    -> trainable pretrained embedding (width 96)
    -> 128 independent, causally padded REN-T/Floating blocks
    -> final REN-T normalization
    -> masked mean of valid final token states
    -> 6-way linear decision head
    -> validation-fitted temperature softmax
    -> sadness / joy / love / anger / fear / surprise
```

In a block, input states first pass through grouped, token-local **Relative Energy Normalization (REN-T)**. Each group calculates non-centered mean absolute magnitude and RMS magnitude. The blended denominator uses 25% L1 and 75% RMS. Temporal mixing uses a causal depthwise convolution, with dilations cycling `(1, 2, 4, 8, 16, 32)`. A low-rank channel mixer and a per-token bounded Floating gain/bias are applied before a learned, depth-scaled residual addition.

All 128 blocks have **independent weights**. There is no weight sharing, no auxiliary LM head, no LM replay and no B-only curvature backward method.

## Classification readout

This is **v2**: the readout is a masked mean of the final hidden states across real input tokens. There is **no appended `DECIDE` token**. Padding is right-aligned and excluded from the pooling numerator and denominator. Causal convolutions prevent right-padding from influencing earlier real tokens.

## Input and vocabulary

The exact original `tokenizer_1k.json` determines all token IDs, the alphabetic BPE merge order and printable ASCII fallbacks. Embeddings were fine-tuned, but vocabulary IDs never changed. Inputs outside the tokenizer's supported character set are rejected after a small, explicit set of Unicode punctuation replacements. The exported `config.json` controls truncation length; no unseen tokenizer is generated.

## Inference implementation

The original `EmotionPredictor` uses `torch.inference_mode()`, no dropout, no training data, and temperature-scaled softmax after the logits. The `EmotionEngine` wrapper adds chunked batches and automatic CUDA Graph replay for FP32 inference on supported CUDA devices; it falls back to eager when capture is unsupported. **Default precision is FP32.** Switching to BF16/FP16 is an explicit opt-in experiment and may change numerical outputs or rare borderline labels.

## What has not been fused

No blocks have been collapsed or pruned. REN-T and Floating gating are input-dependent, so their input-specific calculations cannot be permanently replaced by static constants. CUDA FP32 serving can use graph replay to reduce CPU launch overhead while retaining the same operations, parameter values, and ordering; eager PyTorch remains available with `--eager`. No custom CUDA kernels or quantization are used. A custom fused REN-T/gate kernel remains a separate optimization opportunity and would need numerical and performance validation.
