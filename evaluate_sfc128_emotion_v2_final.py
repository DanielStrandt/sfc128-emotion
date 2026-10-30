#!/usr/bin/env python3
"""Independent, READ-ONLY held-out evaluation for SFC-128 Emotion v2.

Uses the Stage-2 v2 run's *original cached, deduplicated* official DAIR.AI
Emotion test examples (1,997 examples in the reported run). It does not train,
modify, or save weights. The selected checkpoint is based on validation only.

Windows, ONE LINE:
  python evaluate_sfc128_emotion_v2_final.py --run-dir "F:\\SemanticFloat\\sfc_emotion_v2_run" --foundation-dir "F:\\SemanticFloat\\sfc_fineweb_run"

Default: checkpoint_best_emotion.pt (BEST validation macro-F1, not last step).
Optional: --checkpoint latest or --checkpoint export.

Outputs: JSON metrics, text report, per-example predictions, mistakes,
confusion matrix CSV and (if matplotlib installed) confusion matrix PNG.

Recommended scientific protocol: run this ONCE after training/checkpoint
selection finishes. Testing a partial run is supported but consumes the
untouched-test-set status for subsequent model development decisions.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from semantic_floating_ren_t_128 import SFCConfig
from sfc128_emotion_meanpool_model import SemanticFloatingMeanPoolClassifier
from sfc128_emotion_tokenizer import SFCAsciiTokenizer, normalize_english_ascii

LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for part in iter(lambda: src.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    target = path.with_name(path.name + ".part")
    with target.open("w", encoding="utf-8") as dst:
        json.dump(obj, dst, indent=2, ensure_ascii=True, allow_nan=False)
        dst.write("\n")
    os.replace(target, path)


def load_weights(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older PyTorch.
        return torch.load(path, map_location="cpu")


def training_signature(settings: dict) -> str:
    # Exactly the serializer used by the original Stage-2 v2 trainer.
    raw = json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


@dataclass
class TestData:
    flat: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray
    texts: Optional[list[str]] = None

    def __post_init__(self):
        n = len(self.labels)
        if n == 0 or len(self.offsets) != n + 1:
            raise ValueError("empty test data or invalid offset array")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != len(self.flat):
            raise ValueError("test token cache has inconsistent token offsets")
        if np.any(np.diff(self.offsets) <= 0) or np.any(np.diff(self.offsets) > 1024):
            raise ValueError("test data has invalid token lengths")
        if np.any(self.labels >= len(LABELS)):
            raise ValueError("test data contains unknown class labels")
        if self.texts is not None and len(self.texts) != n:
            raise ValueError("test text count does not match cached labels")

    def __len__(self):
        return len(self.labels)

    def batch(self, start: int, end: int, pad_id: int):
        # Precisely match the trainer's right-padding convention.
        sizes = self.offsets[start + 1:end + 1] - self.offsets[start:end]
        longest = int(sizes.max())
        tokens = torch.full((end - start, longest), pad_id, dtype=torch.long)
        for row, idx in enumerate(range(start, end)):
            a, b = int(self.offsets[idx]), int(self.offsets[idx + 1])
            tokens[row, :b - a] = torch.from_numpy(self.flat[a:b].astype(np.int64))
        return tokens


def cache_test(run_dir: Path) -> tuple[TestData | None, dict | None]:
    meta_file = run_dir / "prepared_data.json"
    data_file = run_dir / "prepared_data.npz"
    if not meta_file.is_file():
        raise FileNotFoundError(f"Missing Stage-2 data audit: {meta_file}")
    meta = read_json(meta_file)
    if meta.get("signature", {}).get("source") != "dair-ai/emotion":
        raise ValueError("Training data source is not the official dair-ai/emotion dataset")
    if not data_file.is_file():
        return None, meta
    with np.load(data_file, allow_pickle=False) as data:
        packed = TestData(flat=data["test_flat"].copy(),
                          offsets=data["test_offsets"].copy(),
                          labels=data["test_labels"].copy())
    expected = meta.get("encoded", {}).get("test", {}).get("examples")
    if expected is not None and len(packed) != int(expected):
        raise ValueError("test cache size disagrees with trainer audit")
    return packed, meta


def canonical(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def clean_holdout_rows(rows):
    cleaned = []
    for row in rows:
        try:
            text = normalize_english_ascii(row["text"]).strip()
            label = int(row["label"])
        except (KeyError, TypeError, ValueError):
            continue
        if text and re.search(r"[A-Za-z]", text) and label in range(6):
            cleaned.append((text, label, canonical(text)))
    return cleaned


def dedupe_rows(rows, forbidden=None):
    seen = set() if forbidden is None else set(forbidden)
    unique = []
    for text, label, key in rows:
        if key in seen:
            continue
        unique.append((text, label))
        seen.add(key)
    return unique


def download_official_test(tokenizer: SFCAsciiTokenizer, max_tokens: int,
                           meta: dict) -> TestData:
    """Rebuild test exactly as v2 if cache is missing, or recover original texts.

    Only official validation and test subsets are downloaded, NEVER training.
    The deduped ordering matches the original v2 trainer's test cache.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to recover test examples: python -m pip install datasets") from exc
    rev = meta["signature"].get("revision", "main")
    print("[download] official dair-ai/emotion split validation/test only", flush=True)
    kwargs = {"revision": rev} if rev and rev != "main" else {}
    validation = load_dataset("dair-ai/emotion", "split", split="validation", **kwargs)
    test = load_dataset("dair-ai/emotion", "split", split="test", **kwargs)
    # Match Stage 2 v2: dedupe validation internally, then remove test rows
    # overlapping validation and repeated within test.
    val_rows = clean_holdout_rows(validation)
    val_unique = dedupe_rows(val_rows)
    forbidden = {canonical(text) for text, _ in val_unique}
    test_unique = dedupe_rows(clean_holdout_rows(test), forbidden)
    flat, offsets, labels, texts = [], [0], [], []
    for text, label in test_unique:
        ids = tokenizer.encode(text)[:max_tokens]
        if not ids:
            continue
        flat.extend(ids)
        offsets.append(len(flat))
        labels.append(label)
        texts.append(text)
    return TestData(flat=np.asarray(flat, dtype=np.uint16),
                    offsets=np.asarray(offsets, dtype=np.int64),
                    labels=np.asarray(labels, dtype=np.uint8), texts=texts)


def resolve_tokenizer(run_dir: Path, foundation_dir: Optional[Path], explicit: Optional[str]) -> Path:
    candidates = ([Path(explicit)] if explicit else [])
    if foundation_dir is not None:
        candidates.append(foundation_dir / "tokenizer_1k.json")
    candidates.extend([run_dir.parent / "sfc_fineweb_run" / "tokenizer_1k.json",
                       run_dir / "SFC128_Emotion" / "tokenizer_1k.json"])
    for file in candidates:
        if file.is_file():
            return file
    raise FileNotFoundError("Cannot locate original tokenizer_1k.json. Supply --foundation-dir or --tokenizer.")


def choose_checkpoint(run_dir: Path, selection: str, settings: dict):
    if selection == "best":
        file = run_dir / "checkpoint_best_emotion.pt"
    elif selection == "latest":
        file = run_dir / "checkpoint_latest.pt"
    elif selection == "export":
        file = run_dir / "SFC128_Emotion" / "model.pt"
    else:
        file = Path(selection).expanduser()
    if not file.is_file():
        raise FileNotFoundError(f"Requested checkpoint not found: {file}")
    payload = load_weights(file)
    if selection == "export" or "state_dict" in payload:
        if payload.get("architecture") != "SemanticFloatingMeanPoolClassifier":
            raise ValueError("Model architecture is not SFC-128 Emotion v2")
        if payload.get("readout") != "masked_mean":
            raise ValueError("Exported model has wrong readout")
        state = payload["state_dict"]
    else:
        if "model" not in payload:
            raise ValueError("Checkpoint does not contain a model state")
        if selection != "latest":
            if payload.get("architecture") != "SemanticFloatingMeanPoolClassifier":
                raise ValueError("Checkpoint is not SFC-128 Emotion v2")
            if payload.get("readout") != "masked_mean":
                raise ValueError("Checkpoint has wrong readout type")
        state = payload["model"]
    config = payload.get("config") or settings.get("model_config")
    if config is None:
        raise ValueError("No model config found. Requires the original training_settings.json")
    sig = payload.get("signature")
    if sig is not None and settings:
        if sig != training_signature(settings):
            raise ValueError("Checkpoint signature does not match the run's training settings")
    best_step = payload.get("progress", {}).get("step")
    return file, payload, config, state, best_step


def evaluate(model, data: TestData, batch_size: int, device: torch.device,
             precision: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    preds, logits = [], []
    loss_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(data), batch_size):
            end = min(i + batch_size, len(data))
            x = data.batch(i, end, model.cfg.pad_token_id).to(device)
            if device.type == "cuda" and precision != "fp32":
                with torch.autocast(device_type="cuda", dtype=dtype):
                    batch_logits = model(x).logits.float()
            else:
                batch_logits = model(x).logits.float()
            target = torch.as_tensor(data.labels[i:end].astype(np.int64), device=device)
            loss_sum += float(F.cross_entropy(batch_logits, target, reduction="sum"))
            logits.append(batch_logits.cpu().numpy())
            preds.append(batch_logits.argmax(dim=-1).cpu().numpy())
            if (i // batch_size + 1) % 25 == 0 or end == len(data):
                print(f"[test] {end:,}/{len(data):,} examples evaluated", flush=True)
    return (np.concatenate(preds), np.concatenate(logits),
            data.labels.astype(np.int64), loss_sum / len(data))


def confusion(actual, predicted):
    matrix = np.zeros((6, 6), dtype=np.int64)
    np.add.at(matrix, (actual, predicted), 1)
    return matrix


def score_matrix(matrix):
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("test evaluation is empty")
    support = matrix.sum(axis=1)
    predcounts = matrix.sum(axis=0)
    diag = np.diag(matrix)
    precision = np.divide(diag, predcounts, out=np.zeros(6, dtype=float), where=predcounts > 0)
    recall = np.divide(diag, support, out=np.zeros(6, dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(6, dtype=float), where=(precision + recall) > 0)
    accuracy = float(diag.sum() / total)
    return {
        "num_examples": total,
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / total),
        "micro_f1": accuracy,  # Exactly equal for single-label multi-class classification.
        "per_class": {label: {
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i]),
            "predicted_count": int(predcounts[i])}
            for i, label in enumerate(LABELS)},
        "confusion_matrix": matrix.tolist(),
        "matrix_rows_actual_columns_predicted": True,
    }


def bootstrap_intervals(truth, pred, n_bootstraps: int, seed: int):
    if n_bootstraps == 0:
        return None
    rng = np.random.default_rng(seed)
    values = np.zeros((n_bootstraps, 2), dtype=np.float64)
    for j in range(n_bootstraps):
        ids = rng.integers(0, len(truth), size=len(truth))
        matrix = confusion(truth[ids], pred[ids])
        result = score_matrix(matrix)
        values[j] = result["accuracy"], result["macro_f1"]
    bounds = np.percentile(values, [2.5, 97.5], axis=0)
    return {"method": "nonparametric_iid_bootstrap", "replicates": n_bootstraps,
            "confidence_level": 0.95, "seed": seed,
            "accuracy": [float(bounds[0, 0]), float(bounds[1, 0])],
            "macro_f1": [float(bounds[0, 1]), float(bounds[1, 1])]}


def softmax_numpy(logits, temperature: float):
    z = logits.astype(np.float64) / temperature
    z -= z.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def probability_metrics(probs, truth, n_bins=10):
    idx = np.arange(len(truth))
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = ((confidence >= lo) & (confidence <= hi)) if i == n_bins - 1 else \
               ((confidence >= lo) & (confidence < hi))
        if mask.any():
            ece += float(mask.mean() * abs(np.mean(predicted[mask] == truth[mask]) - confidence[mask].mean()))
    return {"negative_log_likelihood": float(-np.log(np.maximum(probs[idx, truth], 1e-300)).mean()),
            "mean_top1_confidence": float(confidence.mean()),
            "expected_calibration_error_10bins": ece,
            "brier_score_6class": float(np.mean(np.sum((probs - np.eye(6)[truth]) ** 2, axis=1)))}


def write_predictions(output: Path, data: TestData, truth, pred, probs):
    fields = ["test_index_after_audit", "actual", "predicted", "correct", "confidence",
              "token_count"] + ["prob_" + label for label in LABELS]
    if data.texts is not None:
        fields.append("text")
    wrong = 0
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as full, \
         (output / "test_errors.csv").open("w", newline="", encoding="utf-8-sig") as mistakes:
        writer = csv.DictWriter(full, fieldnames=fields)
        errors = csv.DictWriter(mistakes, fieldnames=fields)
        writer.writeheader()
        errors.writeheader()
        for i in range(len(truth)):
            p = int(pred[i]); t = int(truth[i])
            record = {"test_index_after_audit": i, "actual": LABELS[t],
                      "predicted": LABELS[p], "correct": int(p == t),
                      "confidence": round(float(probs[i, p]), 8),
                      "token_count": int(data.offsets[i + 1] - data.offsets[i])}
            record.update({"prob_" + label: round(float(probs[i, j]), 8)
                           for j, label in enumerate(LABELS)})
            if data.texts is not None:
                record["text"] = data.texts[i]
            writer.writerow(record)
            if t != p:
                wrong += 1
                errors.writerow(record)
    return wrong


def write_confusion_csv(output: Path, matrix):
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as dst:
        writer = csv.writer(dst)
        writer.writerow(["actual \\ predicted", *LABELS])
        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *map(int, row)])


def maybe_draw_confusion_png(output: Path, matrix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; confusion_matrix.csv is available", flush=True)
        return
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    im = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(6), yticks=np.arange(6), xticklabels=LABELS,
           yticklabels=LABELS, xlabel="Predicted emotion", ylabel="True emotion",
           title="SFC-128 Emotion v2 — held-out DAIR.AI test")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() * 0.55 else "black")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def summary_text(metrics: dict, best_meta: dict) -> str:
    m = metrics["metrics"]
    lines = ["SFC-128 Emotion v2 — official DAIR.AI held-out test",
             "=" * 64,
             f"Checkpoint:  {metrics['checkpoint']['path']}",
             f"Checkpoint step: {metrics['checkpoint'].get('step')}",
             f"Test examples: {m['num_examples']:,}",
             f"Test accuracy: {m['accuracy']:.4%}",
             f"Test macro-F1: {m['macro_f1']:.4f}",
             f"Test weighted-F1: {m['weighted_f1']:.4f}",
             f"Test macro-precision: {m['macro_precision']:.4f}",
             f"Test macro-recall: {m['macro_recall']:.4f}",
             f"Test cross-entropy (uncalibrated): {metrics['raw_test_loss']:.5f}"]
    if metrics.get("bootstrap_95ci"):
        ci = metrics["bootstrap_95ci"]
        lines.extend([f"Accuracy bootstrap 95% CI: {ci['accuracy'][0]:.2%}–{ci['accuracy'][1]:.2%}",
                      f"Macro-F1 bootstrap 95% CI: {ci['macro_f1'][0]:.4f}–{ci['macro_f1'][1]:.4f}"])
    val = best_meta.get("validation_metrics") or {}
    if val:
        lines.append(f"Best checkpoint validation: accuracy {val['accuracy']:.4%}; macro-F1 {val['macro_f1']:.4f}")
    lines.extend(["", "PER-CLASS PERFORMANCE", "Emotion       Precision    Recall       F1       Support    Predicted"])
    for label, item in m["per_class"].items():
        lines.append(f"{label:11s} {item['precision']:9.4f}   {item['recall']:8.4f}   "
                     f"{item['f1']:8.4f}    {item['support']:6d}     {item['predicted_count']:6d}")
    lines.extend(["", "CONFUSION MATRIX (rows = actual, columns = predicted)",
                  "                " + "  ".join(f"{s[:5]:>6}" for s in LABELS)])
    for label, row in zip(LABELS, m["confusion_matrix"]):
        lines.append(f"{label:<12} " + "  ".join(f"{int(v):6d}" for v in row))
    if metrics["training_finished"] is False:
        lines.extend(["", "WARNING: TRAINING IS NOT CONFIRMED FINISHED.",
                      "This evaluation is provisional. Do not tune future runs to test results."])
    lines.append("")
    return "\n".join(lines)


def run(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    output = Path(args.report_dir).expanduser().resolve() if args.report_dir else run_dir / "final_test_report"
    if (output / "final_test_metrics.json").exists() and not args.overwrite:
        raise FileExistsError(f"Test report already exists: {output}. Use --overwrite only if intentional.")

    settings_file = run_dir / "training_settings.json"
    settings = read_json(settings_file) if settings_file.is_file() else {}
    file, payload, cfg_values, state, ckpt_step = choose_checkpoint(run_dir, args.checkpoint, settings)
    if args.checkpoint == "latest":
        print("[warning] Evaluating LATEST training state, not best validation model.", flush=True)
    if args.checkpoint == "best":
        print(f"[checkpoint] BEST validation checkpoint: {file}", flush=True)
    cfg = SFCConfig(**cfg_values)
    if not args._smoke_test and (cfg.depth != 128 or cfg.vocab_size != 1000 or cfg.num_classes != 6):
        raise ValueError("Checkpoint must be the original SFC-128, 1000-token, six-class model")
    if cfg.num_classes != 6:
        raise ValueError("Model has wrong number of output classes")
    model = SemanticFloatingMeanPoolClassifier(cfg)
    model.load_state_dict(state, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)

    # Model stays read-only even if the trainer is concurrently still writing checkpoints.
    training_finished = (run_dir / "SFC128_Emotion" / "model.pt").is_file()
    if not training_finished:
        latest_file = run_dir / "checkpoint_latest.pt"
        if latest_file.is_file():
            latest = load_weights(latest_file)
            current = latest.get("progress", {})
            training_finished = bool(current.get("stopped_early", False) or
                                     (settings and current.get("epoch", 0) >= settings.get("epochs", math.inf)))
    if not training_finished:
        print("[warning] Stage 2 appears to still be training. Testing now makes this an",
              "exploratory test; do not use test scores for checkpoint selection or tuning.", flush=True)
        if args.require_complete:
            raise RuntimeError("Training not complete; omit --require-complete only if you knowingly test early")

    cached, meta = cache_test(run_dir)
    if meta is None:
        raise RuntimeError("No original data audit found")
    sig = meta["signature"]
    if settings and settings.get("data_signature") != sig:
        raise ValueError("Cached data signature disagrees with training settings")
    foundation = Path(args.foundation_dir).expanduser().resolve() if args.foundation_dir else None
    tokenizer_file = resolve_tokenizer(run_dir, foundation, args.tokenizer)
    if sha256(tokenizer_file) != sig.get("tokenizer_sha256"):
        raise ValueError("Tokenizer SHA256 mismatch: refusing to test with a different vocabulary")
    tokenizer = SFCAsciiTokenizer.load(tokenizer_file)
    if len(tokenizer.vocab) != cfg.vocab_size:
        raise ValueError("Tokenizer vocabulary size disagrees with trained model")
    max_tokens = int(sig["max_text_tokens"])
    if max_tokens > cfg.max_context:
        raise ValueError("Test token cap exceeds trained model max_context")

    if cached is None or args.include_text:
        downloaded = download_official_test(tokenizer, max_tokens, meta)
        if cached is not None:
            if (not np.array_equal(cached.flat, downloaded.flat) or
                not np.array_equal(cached.offsets, downloaded.offsets) or
                not np.array_equal(cached.labels, downloaded.labels)):
                raise ValueError("Downloaded official test differs from the original Stage 2 cached test")
            cached.texts = downloaded.texts
        else:
            cached = downloaded
    assert cached is not None
    expected = meta.get("encoded", {}).get("test", {}).get("examples")
    if expected is not None and len(cached) != expected:
        raise ValueError("Loaded test has a different number of samples from original training audit")
    print(f"[dataset] official dair-ai/emotion split/test; {len(cached):,} examples after original audit", flush=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA GPU is available")
    precision = args.precision
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "fp32"
    if precision in ("bf16", "fp16") and device.type != "cuda":
        raise ValueError("BF16/FP16 are only enabled for CUDA in this evaluator")
    model = model.to(device).eval()
    print(f"[model] masked-mean v2; depth={cfg.depth}; parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"[device] {device}, precision={precision}, batch_size={args.batch_size}", flush=True)
    predicted, logits, actual, raw_loss = evaluate(model, cached, args.batch_size, device, precision)

    matrix = confusion(actual, predicted)
    metrics = score_matrix(matrix)
    temperature = float(payload.get("temperature", 1.0)) if args.checkpoint == "export" else 1.0
    if temperature <= 0:
        raise ValueError("Invalid temperature in exported checkpoint")
    probabilities = softmax_numpy(logits, temperature)
    if not np.array_equal(probabilities.argmax(axis=1), predicted):
        raise RuntimeError("Temperature unexpectedly changed the top-1 predictions")
    probability_scores = probability_metrics(probabilities, actual)
    intervals = bootstrap_intervals(actual, predicted, args.bootstrap, args.seed)
    report = {
        "dataset": "dair-ai/emotion", "config": "split", "split": "test",
        "test_set_protocol": "same v2 cleaned/deduplicated holdout as training cache",
        "checkpoint": {"selection": args.checkpoint, "path": str(file),
                       "sha256": sha256(file), "step": ckpt_step,
                       "validation_metrics": payload.get("validation_metrics")},
        "training_finished": bool(training_finished),
        "model": {"architecture": "SemanticFloatingMeanPoolClassifier",
                  "readout": "masked_mean", "depth": cfg.depth,
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "vocabulary_size": cfg.vocab_size,
                  "tokenizer_sha256": sha256(tokenizer_file),
                  "max_text_tokens": max_tokens},
        "precision": precision, "batch_size": args.batch_size,
        "metrics": metrics, "raw_test_loss": raw_loss,
        "probability_temperature": temperature,
        "probability_metrics": probability_scores,
        "bootstrap_95ci": intervals,
        "data_signature": sig,
        "note": "Model selection must be based on validation; do not tune to held-out test results.",
    }
    output.mkdir(parents=True, exist_ok=True)
    wrong = write_predictions(output, cached, actual, predicted, probabilities)
    write_confusion_csv(output, matrix)
    maybe_draw_confusion_png(output, matrix)
    summary = summary_text(report, payload)
    (output / "final_test_summary.txt").write_text(summary, encoding="utf-8")
    # Write the completion marker only after all the other results succeed.
    write_json(output / "final_test_metrics.json", report)
    print("\n" + summary, flush=True)
    print(f"[saved] {output} (predictions={len(cached)}, mistakes={wrong})", flush=True)
    print("[note] No training, gradient steps, checkpoint writes, or optimizer changes occurred.", flush=True)
    return report


def smoke_test():
    """No network or GPU. Build 3-block checkpoint and matching cached test, run evaluator."""
    import string
    from dataclasses import replace
    with tempfile.TemporaryDirectory(prefix="sfc_final_eval_smoke_") as tmp:
        root = Path(tmp)
        run_dir = root / "run"
        foundation = root / "sfc_fineweb_run"
        run_dir.mkdir()
        foundation.mkdir()
        from sfc128_emotion_tokenizer import SPECIAL, ASCII_CHARS
        def token_suffix(i):
            # Unique valid alphabetic dummy tokens.
            s = ""
            while True:
                s = chr(97 + i % 26) + s
                i //= 26
                if i == 0:
                    return s
        vocab = list(SPECIAL) + list(ASCII_CHARS)
        occupied = set(vocab)
        i = 0
        while len(vocab) < 1000:
            t = "zz" + token_suffix(i)
            i += 1
            if t not in occupied:
                vocab.append(t)
                occupied.add(t)
        tokenizer_file = foundation / "tokenizer_1k.json"
        write_json(tokenizer_file, {"kind": "ascii_constrained_bpe_v1", "vocab": vocab, "merges": []})
        cfg = SFCConfig(depth=3, hidden_size=24, ren_groups=4, bottleneck_size=8,
                        floating_hidden=8, vocab_size=1000, pad_token_id=0,
                        decision_token_id=3, num_classes=6, max_context=64,
                        gradient_checkpointing=False)
        model = SemanticFloatingMeanPoolClassifier(cfg)
        settings = {"model_config": asdict(cfg),
                    "data_signature": {"source": "dair-ai/emotion", "version": 3,
                                       "tokenizer_sha256": sha256(tokenizer_file),
                                       "max_text_tokens": 32}}
        write_json(run_dir / "training_settings.json", settings)
        torch.save({"architecture": "SemanticFloatingMeanPoolClassifier", "readout": "masked_mean",
                    "config": asdict(cfg), "model": model.state_dict(),
                    "signature": training_signature(settings),
                    "progress": {"step": 25},
                    "validation_metrics": {"accuracy": 0.5, "macro_f1": 0.4}},
                   run_dir / "checkpoint_best_emotion.pt")
        rng = np.random.default_rng(4)
        labels = np.asarray([i % 6 for i in range(24)], dtype=np.uint8)
        lengths = rng.integers(2, 15, size=len(labels))
        flat = np.concatenate([rng.integers(8, 100, size=l, dtype=np.uint16) for l in lengths])
        offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
        with (run_dir / "prepared_data.npz").open("wb") as stream:
            np.savez_compressed(stream, test_flat=flat, test_offsets=offsets, test_labels=labels)
        write_json(run_dir / "prepared_data.json", {
            "signature": settings["data_signature"], "encoded": {"test": {"examples": len(labels)}}})
        args = argparse.Namespace(run_dir=str(run_dir), foundation_dir=str(foundation),
                                  checkpoint="best", report_dir=None, tokenizer=None,
                                  batch_size=7, device="cpu", precision="fp32",
                                  bootstrap=50, seed=1337, include_text=False,
                                  require_complete=False, overwrite=False, _smoke_test=True)
        report = run(args)
        assert report["metrics"]["num_examples"] == 24
        assert len(report["metrics"]["confusion_matrix"]) == 6
        assert abs(sum(item["support"] for item in report["metrics"]["per_class"].values()) - 24) < 1e-6
        assert (run_dir / "final_test_report" / "test_predictions.csv").is_file()
        assert (run_dir / "final_test_report" / "test_errors.csv").is_file()
        assert (run_dir / "final_test_report" / "confusion_matrix.csv").is_file()
        # Test the other two checkpoint formats WITHOUT an additional held-out evaluation.
        torch.save({"signature": training_signature(settings), "model": model.state_dict(),
                    "progress": {"step": 30}}, run_dir / "checkpoint_latest.pt")
        exported = run_dir / "SFC128_Emotion"
        exported.mkdir()
        torch.save({"architecture": "SemanticFloatingMeanPoolClassifier", "readout": "masked_mean",
                    "config": asdict(cfg), "state_dict": model.state_dict(),
                    "labels": list(LABELS), "temperature": 1.15}, exported / "model.pt")
        for choice in ("latest", "export"):
            _, _, config, weights, _ = choose_checkpoint(run_dir, choice, settings)
            restored = SemanticFloatingMeanPoolClassifier(SFCConfig(**config))
            restored.load_state_dict(weights, strict=True)
        print("[smoke] PASS: best/latest/export checkpoint compatibility; cached holdout; "
              "six-class metrics, bootstrap, CSV, report", flush=True)


def main():
    p = argparse.ArgumentParser(description="Final held-out DAIR.AI Emotion six-class test for SFC-128 v2")
    p.add_argument("--run-dir", default="./sfc_emotion_v2_run",
                   help="Directory containing checkpoint_best_emotion.pt and prepared_data.npz")
    p.add_argument("--foundation-dir", default=None,
                   help="Original FineWeb foundation directory containing tokenizer_1k.json")
    p.add_argument("--tokenizer", default=None,
                   help="Explicit path to the original tokenizer_1k.json")
    p.add_argument("--checkpoint", default="best",
                   help="best (default), latest, export, or explicit checkpoint path")
    p.add_argument("--report-dir", default=None,
                   help="Default is <run-dir>/final_test_report")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Number of bootstrap replicates for 95%% CI; 0 disables")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--include-text", action="store_true",
                   help="Download original official held-out texts, verify cached token IDs, add text to CSV")
    p.add_argument("--require-complete", action="store_true",
                   help="Refuse evaluation until finished training/export is confirmed")
    p.add_argument("--overwrite", action="store_true",
                   help="Intentionally replace an existing final test report")
    p.add_argument("--smoke-test", action="store_true", help="Run small offline end-to-end validation")
    args = p.parse_args()
    if args.batch_size < 1 or args.bootstrap < 0:
        p.error("batch-size must be positive; bootstrap nonnegative")
    args._smoke_test = False
    if args.smoke_test:
        smoke_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
