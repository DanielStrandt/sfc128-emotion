# SFC-128 Emotion v2 — masked-mean and six-way balanced fine-tuning

This bundle replaces the **Stage 2 training code only**. It uses the **same original FineWeb-Edu checkpoint and the exact same 1,000-token BPE vocabulary**. No tokenizer training, no LM head, no language replay, and **no parameter sharing**. All 128 REN-T blocks, their Floating gates, the pretrained embedding, the final norm and a new six-class emotion head are fully trainable.

## Why v2

The first emotion training run used an appended `DECIDE` token as its only classification readout. In v2, the 128-layer network produces one representation per input token, and the decision head reads the **mask-weighted mean of the real token representations**. All padding is excluded, and **no decision token is appended**.

The first trainer also trained on naturally imbalanced examples and applied frequency-derived class weights. The v2 default uses **stratified six-way training samples** and ordinary **unweighted cross-entropy** (avoiding double-balancing).

For full-data training (about 368K deduplicated examples), every optimizer update draws approximately equal numbers of sadness, joy, love, anger, fear and surprise. The sampled frequencies across one epoch differ by **at most one example**. Per-class shuffled pools are cycled independently: rare examples are repeated when their pool is exhausted; frequent examples are not all necessarily seen within one epoch. Repeated minority examples can overfit, so validation macro-F1 and per-class F1 remain the decision metrics. The original validation and test sets are **never resampled or rebalanced**.

## Files

- `train_sfc128_emotion_v2.py` — complete end-to-end Stage 2 v2 trainer, evaluation, resume, export and smoke tests.
- `sfc128_emotion_meanpool_model.py` — readout-only subclass of the original architecture; no parameter-shape changes.
- `semantic_floating_ren_t_128.py` — exact original pretrained model block definitions.
- `sfc128_emotion_tokenizer.py` — inference-only copy of original 1K ASCII BPE, no retraining.
- `sfc128_emotion_v2_inference.py` — six-class final model inference using masked-mean readout.
- `requirements_sfc128_emotion.txt` — dependencies.

## Windows setup

Unzip everything into `D:\path\to\SemanticFloat` (or another folder). Keep the original FineWeb foundation folder unchanged.

Install dependencies if they are not already installed:

```powershell
python -m pip install -r requirements_sfc128_emotion.txt
```

Run local smoke tests (does not download data):

```powershell
python train_sfc128_emotion_v2.py --smoke-test
```

### Recommended short pilot before another full run

This reuses the existing FineWeb foundation and loads the DAIR.AI Emotion dataset. It pauses **after 3,000 optimizer steps** so you can inspect validation macro-F1 before committing to another entire 3-epoch run. It saves checkpoints at validation intervals. The new output directory ensures the old emotion run and checkpoint are untouched.

```powershell
python train_sfc128_emotion_v2.py --foundation-dir "D:\path\to\SemanticFloat\sfc_fineweb_run" --output-dir "D:\path\to\SemanticFloat\sfc_emotion_v2_run" --eval-every 500 --max-steps 3000
```

If validation metrics improve sufficiently, resume the *same* run (with the same `--eval-every` setting) by omitting `--max-steps`:

```powershell
python train_sfc128_emotion_v2.py --foundation-dir "D:\path\to\SemanticFloat\sfc_fineweb_run" --output-dir "D:\path\to\SemanticFloat\sfc_emotion_v2_run" --eval-every 500
```

If you prefer to train without pausing, use the second command directly.

### Optional experiment: small-data pilot

```powershell
python train_sfc128_emotion_v2.py --foundation-dir "D:\path\to\SemanticFloat\sfc_fineweb_run" --output-dir "D:\path\to\SemanticFloat\sfc_emotion_v2_pilot" --data-mode pilot --epochs 8 --eval-every 100
```

### Optional alternatives for future ablations

`--sampling natural` uses the dataset's original frequencies. `--sampling balanced` (the default) uses six-way stratification. `--class-weighting none` is the default and required for balanced sampling to prevent double-correcting frequencies. `--epoch-samples 100000` (balanced mode only) reduces sample draws and minority repeat exposure per epoch, but also exposes the model to fewer majority examples. These settings change training behavior: use a **new output directory** rather than resume with incompatible settings.

## Model selection, logs and deployment

Validation runs on the original official DAIR.AI validation texts. Logged `macro_f1`, `accuracy`, `per_class_f1`, `predicted_counts`, and `loss` make class collapse visible. The best validation macro-F1 checkpoint wins; test data is evaluated only after training completes. Calibration uses validation data only.

Outputs appear in `D:\path\to\SemanticFloat\sfc_emotion_v2_run`:

- `prepared_data.npz`, `prepared_data.json` — cached tokenized dataset and leakage audit.
- `checkpoint_latest.pt` — resumable model, optimizer and RNG state.
- `checkpoint_best_emotion.pt` — best validation checkpoint.
- `train_log.jsonl` — training/evaluation history.
- `SFC128_Emotion/` — standalone emotion-only deployment directory after completion.

The deployment folder contains the original `tokenizer_1k.json`, exported model, label mapping, model Python modules, calibration metadata and evaluation metrics.

After the run exports successfully, inference is a single command:

```powershell
python sfc128_emotion_v2_inference.py --model-dir "D:\path\to\SemanticFloat\sfc_emotion_v2_run\SFC128_Emotion" --text "I feel so relieved and happy today!"
```

The six labels are sadness, joy, love, anger, fear and surprise.

## Compatibility and limitations

- **Use a new output directory:** v2 does not resume the previous `DECIDE`-token emotion checkpoint, because it changes the training readout and class distribution. It starts from the **finished FineWeb-pretrained backbone**.
- **Same vocabulary:** The exact `tokenizer_1k.json` and all 128 pretrained block weights are used. No vocabulary changes.
- **Checkpoints:** Trained checkpoints are self-contained for this architecture, but the foundation and training data are not modified.
- **Balanced sampling is not guaranteed better:** It can overfit rare examples and shift probability priors. Always compare validation macro-F1, accuracy and per-label performance against the earlier run. The original test distribution remains untouched.
- **Validation:** Local synthetic smoke tests, sampled-class quota tests and a 128-layer CPU forward/backward test have passed. A full GPU run on DAIR.AI has **not** been executed here.

