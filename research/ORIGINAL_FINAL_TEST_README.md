# SFC-128 Emotion v2 — independent held-out final test

This is an **evaluation-only** companion to the existing Stage 2 v2 trainer. It does not train or modify the model, checkpoint, optimizer, tokenizer, training logs, or dataset cache.

**Default checkpoint:** `checkpoint_best_emotion.pt`, which is the checkpoint selected by the **highest validation macro-F1**, rather than the newest saved optimizer state. This is the correct checkpoint for reporting final held-out test performance.

**Default dataset:** the exact original Stage 2 v2 tokenized DAIR.AI `dair-ai/emotion` `split/test` sample, after the same ASCII cleaning, duplicate/validation-overlap removal, and token truncation. In the reported run the test set contains **1,997 examples** after 3 rows were excluded. `prepared_data.npz` and `prepared_data.json` are read from the **existing** run directory. If the NPZ is missing, the evaluator can re-download *only* the official validation/test splits and rebuild the identical test protocol; it checks the expected example count. The original 1,000-token BPE is required and its SHA256 hash is verified against the training dataset metadata.

## Install

The ZIP contains:

- `evaluate_sfc128_emotion_v2_final.py` — complete final evaluation program
- `semantic_floating_ren_t_128.py` — exact SFC-128 architecture
- `sfc128_emotion_meanpool_model.py` — v2 masked-mean classifier
- `sfc128_emotion_tokenizer.py` — original 1K tokenizer loader (no training)
- `SFC128_EMOTION_FINAL_TEST_README.md` — these instructions

Unzip into `D:\path\to\SemanticFloat` alongside your Stage 2 v2 trainer, or into a separate folder. Requires `torch` and `numpy`, already needed by v2. `datasets` is **only** needed if you request original test text or need to reconstruct a missing token cache. `matplotlib` is optional and saves a confusion matrix PNG.

## First: offline smoke test

```powershell
python evaluate_sfc128_emotion_v2_final.py --smoke-test
```

## Run your actual held-out test (one line)

```powershell
python evaluate_sfc128_emotion_v2_final.py --run-dir "D:\path\to\SemanticFloat\sfc_emotion_v2_run" --foundation-dir "D:\path\to\SemanticFloat\sfc_fineweb_run"
```

This command tests the **best validation checkpoint**, and saves the report under:

`D:\path\to\SemanticFloat\sfc_emotion_v2_run\final_test_report\`

Outputs:

- `final_test_summary.txt`: accuracy, macro-F1, precision, recall, weighted-F1, per-class scores, and labeled confusion matrix
- `final_test_metrics.json`: full machine-readable metrics, checkpoint SHA256, checkpoint step, dataset signature, 95% bootstrap confidence intervals, probability calibration diagnostics
- `test_predictions.csv`: prediction, correct label, confidence, probabilities for **all six** emotions and token length for each example
- `test_errors.csv`: only misclassified examples, with the same fields
- `confusion_matrix.csv`: actual-label rows and predicted-label columns
- `confusion_matrix.png`: same matrix visualized, when matplotlib is installed

To attach the **original text** to the two CSV files, use `--include-text`. This downloads the official validation/test texts from Hugging Face, and the evaluator **verifies that their token IDs and labels exactly match your original cached test set** before using them:

```powershell
python evaluate_sfc128_emotion_v2_final.py --run-dir "D:\path\to\SemanticFloat\sfc_emotion_v2_run" --foundation-dir "D:\path\to\SemanticFloat\sfc_fineweb_run" --include-text
```

## Other options

- `--checkpoint latest`: evaluate the *latest optimizer checkpoint* rather than best validation checkpoint. This is **not recommended for the final score** and must use a separate `--report-dir` if an earlier test report exists.
- `--checkpoint export`: evaluate the deployed `SFC128_Emotion/model.pt` after the trainer completes; applies its saved calibration temperature when calculating confidence, without changing classification accuracy.
- `--device cuda` or `--device cpu`: choose device explicitly. `auto` uses CUDA when available.
- `--precision bf16`: match your RTX 3060 training/evaluation precision (auto selects this if supported).
- `--batch-size 8`: lower evaluation batch size if GPU memory is insufficient (default 16).
- `--bootstrap 0`: skip confidence intervals (default 1,000 resamples).
- `--require-complete`: refuse testing unless the Stage 2 training/export is finished.
- `--report-dir <path>`: put outputs somewhere else.
- `--overwrite`: replace an existing final test report intentionally. By default the program refuses to overwrite previous test results.

**Research caution:** The original v2 training program already evaluates `split/test` at its end and writes summary test metrics into the exported `SFC128_Emotion/metrics.json`. This independent module adds detailed reports and individual predictions. Ideally run it **once**, after the final checkpoint has been selected using validation data. If you test during epoch 3, the script warns that the result is provisional. Repeatedly choosing checkpoints or training strategies based on the test score would compromise its status as an unbiased final held-out test.

## What we verified

The evaluator passed an offline end-to-end synthetic checkpoint and cached-test smoke test, producing all output files, plus strict state-loading tests for `best`, `latest`, and `export` checkpoint formats. It has **not** run against your actual 128-layer checkpoint (which remains on your Windows machine). The final accuracy will be reported by your own GPU after you run the command above.

