# Evaluation and reproducibility

## Reported DAIR.AI Emotion benchmark

Official dataset: `dair-ai/emotion`, configuration `split`, test partition after the user's original duplicate/overlap audit: **1,997 rows**. Training combined the official training split and the `unsplit` larger training collection, retaining 367,961 deduplicated training examples and excluding 4,453 training rows overlapping official held-out examples. Unlike balanced *training* sampling, the validation/test partitions retained their natural label distributions.

- Best checkpoint: `checkpoint_best_emotion.pt`, optimizer step **24,000** (highest validation macro-F1).
- Best validation accuracy: **86.9369%**; validation macro-F1: **0.853272**.
- Test accuracy: **86.3295%** (1,724 correct out of 1,997).
- Test macro-F1: **0.820221**; weighted-F1: **0.8631**.
- Uncalibrated cross-entropy: **0.56753**.
- Validation-fitted temperature exported: **1.418506**.
- Accuracy bootstrap 95% CI: **84.78–87.88%**.
- Macro-F1 bootstrap 95% CI: **0.7965–0.8427**.

The same held-out result was reproduced after training finished. This establishes repeatability of the same model and examples, not a new independent test. The held-out test was initially inspected before training finished, and should therefore be presented as an exploratory development-set experiment when making broader research claims.

The owner-provided completion log records training stopping at step **28,000** after eight evaluations without validation macro-F1 improvement. The final step-28,000 validation score was macro-F1 **0.851307** and accuracy **0.86987**; the selected best-validation checkpoint remained step **24,000** (validation macro-F1 **0.853272**). The final evaluator ran the best checkpoint on CUDA with BF16, batch size 16, and completed all 1,997 examples. The attached `metrics.json` preserves the exported train/evaluation metrics and audit metadata; `SHA256SUMS.txt` records the copied artifact hashes.

## Complete test rerun

The export `models/SFC128_Emotion/model.pt` is enough for predictions, **but not enough to reconstruct the exact deduplicated test set** without the original trainer run's cached data or source audit. The original evaluator is included at repository root.

From the repo root, with the original run and foundation directories still present:

```powershell
python evaluate_sfc128_emotion_v2_final.py --run-dir "D:\path\to\SemanticFloat\sfc_emotion_v2_run" --foundation-dir "D:\path\to\SemanticFloat\sfc_fineweb_run" --overwrite
```

This loads the best-validation checkpoint and the `prepared_data.npz` + `prepared_data.json` cached official test from the original run. The evaluator emits a text summary, machine-readable metrics, predictions, errors and confusion matrix.

If you want to verify the standalone exported checkpoint rather than the best checkpoint, add `--checkpoint export`, but use a separate `--report-dir` to keep independent reports and remember probabilities may differ due to deployment temperature calibration:

```powershell
python evaluate_sfc128_emotion_v2_final.py --run-dir "D:\path\to\SemanticFloat\sfc_emotion_v2_run" --foundation-dir "D:\path\to\SemanticFloat\sfc_fineweb_run" --checkpoint export --report-dir "D:\path\to\SemanticFloat\sfc_emotion_v2_run\export_test_report"
```

The original evaluator does not alter trained weights. If the cached test data is unavailable, it can re-download only the official validation/test rows and verify alignment with the original run metadata. Re-downloading a newer revision of a dataset could change results unless you pin the original dataset revision.

## Saved metrics without rerunning the dataset

```powershell
python tools/check_reference_metrics.py
```

This merely reads `metrics.json` from the exported model. It does **not** perform classification or verify accuracy on test examples.

## Limits of score comparisons

The benchmark should not be directly equated with public fine-tunings of DistilBERT or BERT-tiny that used only the standard 16K DAIR.AI training split. We used the much larger `unsplit` collection (cleaned), potentially different text normalization/deduplication rules, and a distinct vocabulary. A fair architecture comparison needs matched training data, seeds, evaluation and computing budget. Perplexity values from custom tokenizers are not directly comparable.

