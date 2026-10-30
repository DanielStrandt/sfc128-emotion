"""Stage 2 v2: balanced, fully fine-tuned SFC-128 Emotion specialist.

Changes relative to Stage 2 v1:
- Classify a masked mean of ALL real token states, never an appended DECIDE token.
- Default six-way balanced sampling (approximately equal labels per optimizer step).
- Default ordinary, UNWEIGHTED cross-entropy: no double-balancing.
- Preserve the identical 1,000-ID Stage 1 vocabulary and all 128 pretrained blocks.
- Save checkpoints with a distinct signature and export a separate v2-only model.

The original FineWeb checkpoint is never edited or overwritten.
The original failed/partial emotion run is not modified; use a NEW output dir.
All parameters, including embeddings and every REN-T layer, receive gradients.
The language-modeling head is absent; there is no language replay.

Quick start (Windows, single terminal line):
  python train_sfc128_emotion_v2.py --foundation-dir ./sfc_fineweb_run --output-dir ./sfc_emotion_v2_run

Offline tests:
  python train_sfc128_emotion_v2.py --smoke-test
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from semantic_floating_ren_t_128 import SFCConfig, SemanticFloatingClassifier
from sfc128_emotion_meanpool_model import SemanticFloatingMeanPoolClassifier
from sfc128_emotion_tokenizer import SFCAsciiTokenizer, normalize_english_ascii

LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")
DATA_VERSION = 3


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def save_torch(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    torch.save(data, tmp)
    os.replace(tmp, path)


def load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: dict) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(content.encode("ascii")).hexdigest()


def write_log(out: Path, payload: dict) -> None:
    with (out / "train_log.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=True) + "\n")
    kind = payload.get("event", "log")
    print(f"[{kind}] " + json.dumps({k: v for k, v in payload.items() if k != "event"}), flush=True)


def canonical_key(text: str) -> str:
    # Conservative data-leak check: insensitive to case and whitespace, and
    # applied only AFTER the very same ASCII typography normalization.
    return re.sub(r"\s+", " ", text).strip().casefold()


def prepare_source_rows(source: Iterable[dict], label: str) -> tuple[list[tuple[str, int, str]], Counter]:
    stats: Counter = Counter()
    result = []
    for row in source:
        stats["source_rows"] += 1
        try:
            text = normalize_english_ascii(row["text"]).strip()
            cls = int(row["label"])
        except (ValueError, TypeError, KeyError):
            stats["invalid_or_nonascii"] += 1
            continue
        if not text or not re.search(r"[A-Za-z]", text) or cls not in range(len(LABELS)):
            stats["empty_or_invalid_label"] += 1
            continue
        result.append((text, cls, canonical_key(text)))
    stats["accepted_rows"] = len(result)
    print(f"[data/{label}] accepted {len(result):,}/{stats['source_rows']:,} source rows", flush=True)
    return result, stats


def dedupe_holdout(rows: list[tuple[str, int, str]], exclude: set[str] | None = None):
    exclude = exclude or set()
    seen = set(exclude)
    kept = []
    removed = 0
    for text, label, key in rows:
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append((text, label, key))
    return kept, removed


def select_training_rows(
    split_train: list[tuple[str, int, str]],
    unsplit_train: list[tuple[str, int, str]],
    validation: list[tuple[str, int, str]],
    test: list[tuple[str, int, str]],
    data_mode: str,
) -> tuple[list[tuple[str, int, str]], dict]:
    forbidden = {row[2] for row in validation} | {row[2] for row in test}
    combined = split_train + (unsplit_train if data_mode == "full" else [])
    seen: dict[str, tuple[str, int, str]] = {}
    conflicts: set[str] = set()
    stats: Counter = Counter()
    for row in combined:
        text, cls, key = row
        if key in forbidden:
            stats["heldout_overlap_excluded"] += 1
            continue
        if key in conflicts:
            stats["conflicting_label_excluded"] += 1
            continue
        old = seen.get(key)
        if old is not None:
            if old[1] != cls:
                del seen[key]
                conflicts.add(key)
                stats["conflicting_label_excluded"] += 2
            else:
                stats["duplicate_train_excluded"] += 1
            continue
        seen[key] = row
    stats["training_rows"] = len(seen)
    stats["unique_conflicting_texts"] = len(conflicts)
    return list(seen.values()), dict(stats)


@dataclass
class PackedSplit:
    flat: np.ndarray  # uint16, concatenated token ID arrays
    offsets: np.ndarray  # int64, N+1
    labels: np.ndarray  # uint8, N

    def __len__(self) -> int:
        return int(len(self.labels))

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def batch(self, indices: np.ndarray, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not len(indices):
            raise ValueError("empty batch")
        length = max(int(self.offsets[i + 1] - self.offsets[i]) for i in indices)
        tokens = np.full((len(indices), length), pad_id, dtype=np.int64)
        targets = np.empty(len(indices), dtype=np.int64)
        for k, item in enumerate(indices):
            item = int(item)
            a, b = int(self.offsets[item]), int(self.offsets[item + 1])
            tokens[k, : b - a] = self.flat[a:b]
            targets[k] = self.labels[item]
        return torch.from_numpy(tokens), torch.from_numpy(targets)


def pack_rows(rows: list[tuple[str, int, str]], tokenizer: SFCAsciiTokenizer, max_tokens: int):
    flat = array("H")
    offsets = [0]
    labels = array("B")
    truncations = 0
    empty = 0
    for text, cls, _ in rows:
        encoded = tokenizer.encode(text)
        if len(encoded) > max_tokens:
            encoded = encoded[:max_tokens]
            truncations += 1
        if not encoded:
            empty += 1
            continue
        flat.extend(encoded)
        offsets.append(len(flat))
        labels.append(cls)
    return (
        PackedSplit(np.asarray(flat, dtype=np.uint16), np.asarray(offsets, dtype=np.int64),
                    np.asarray(labels, dtype=np.uint8)),
        {"examples": len(labels), "truncated": truncations, "zero_tokens_skipped": empty,
         "total_tokens": len(flat), "mean_tokens": round(len(flat) / max(len(labels), 1), 2)},
    )


def load_dataset_sources(args) -> dict[str, list[dict]]:
    if args.local_data:
        with Path(args.local_data).open("r", encoding="utf-8") as source:
            data = json.load(source)
        for key in ("split_train", "validation", "test"):
            if key not in data or not isinstance(data[key], list):
                raise ValueError(f"local dataset JSON must contain list {key!r}")
        if args.data_mode == "full" and "unsplit_train" not in data:
            raise ValueError("local full mode requires 'unsplit_train' list")
        data.setdefault("unsplit_train", [])
        return data
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install Hugging Face datasets: python -m pip install datasets") from exc
    kw = {"revision": args.dataset_revision} if args.dataset_revision else {}
    if args.dataset_cache:
        kw["cache_dir"] = args.dataset_cache
    print("[download] fetching official DAIR.AI Emotion train/validation/test", flush=True)
    official = load_dataset("dair-ai/emotion", "split", **kw)
    assert tuple(official["train"].features["label"].names) == LABELS, "dataset label mapping changed"
    data = {"split_train": list(official["train"]),
            "validation": list(official["validation"]),
            "test": list(official["test"])}
    if args.data_mode == "full":
        print("[download] fetching DAIR.AI Emotion unsplit train", flush=True)
        unsplit = load_dataset("dair-ai/emotion", "unsplit", split="train", **kw)
        assert tuple(unsplit.features["label"].names) == LABELS, "unsplit label mapping changed"
        data["unsplit_train"] = list(unsplit)
    else:
        data["unsplit_train"] = []
    return data


def prepare_dataset(args, out: Path, tokenizer: SFCAsciiTokenizer, foundation_hash: str):
    tokenizer_hash = file_sha256(Path(args.foundation_dir) / "tokenizer_1k.json")
    signature = {"version": DATA_VERSION, "source": "dair-ai/emotion" if not args.local_data else
                 "local:" + file_sha256(Path(args.local_data)),
                 "revision": args.dataset_revision or "main", "data_mode": args.data_mode,
                 "max_text_tokens": args.max_text_tokens, "tokenizer_sha256": tokenizer_hash}
    meta_file = out / "prepared_data.json"
    data_file = out / "prepared_data.npz"
    if meta_file.exists() and data_file.exists():
        with meta_file.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("signature") != signature:
            raise ValueError("prepared data settings differ; choose another output directory")
        with np.load(data_file, allow_pickle=False) as cache:
            result = {split: PackedSplit(cache[f"{split}_flat"].copy(),
                                         cache[f"{split}_offsets"].copy(),
                                         cache[f"{split}_labels"].copy())
                      for split in ("train", "validation", "test")}
        print("[data] using prepared/verified DAIR.AI token cache", flush=True)
        return result, metadata

    source = load_dataset_sources(args)
    cleaned = {}
    source_stats = {}
    for split in ("split_train", "unsplit_train", "validation", "test"):
        cleaned[split], stats = prepare_source_rows(source[split], split)
        source_stats[split] = dict(stats)
    val, val_duplicate_count = dedupe_holdout(cleaned["validation"])
    test, test_duplicate_count = dedupe_holdout(cleaned["test"], {r[2] for r in val})
    train, train_stats = select_training_rows(cleaned["split_train"], cleaned["unsplit_train"],
                                             val, test, args.data_mode)
    if not train or not val or not test:
        raise RuntimeError("empty train/validation/test after clean ASCII and strict deduplication")
    rng = random.Random(args.seed)
    rng.shuffle(train)
    packs = {}
    pack_stats = {}
    for label, records in (("train", train), ("validation", val), ("test", test)):
        packs[label], pack_stats[label] = pack_rows(records, tokenizer, args.max_text_tokens)
        print(f"[data/{label}] {pack_stats[label]['examples']:,} sequences, "
              f"{pack_stats[label]['total_tokens']:,} tokens, "
              f"{pack_stats[label]['truncated']:,} truncated", flush=True)
    metadata = {"signature": signature, "foundation_sha256": foundation_hash,
                "source_stats": source_stats, "training_audit": train_stats,
                "validation_duplicates_excluded": val_duplicate_count,
                "test_overlaps_or_duplicates_excluded": test_duplicate_count,
                "encoded": pack_stats,
                "label_histograms": {name: {LABELS[i]: int((p.labels == i).sum())
                                            for i in range(len(LABELS))} for name, p in packs.items()}}
    if metadata["training_audit"]["heldout_overlap_excluded"]:
        print(f"[audit] excluded {metadata['training_audit']['heldout_overlap_excluded']:,} "
              "train rows overlapping official validation/test", flush=True)
    if test_duplicate_count:
        print(f"[audit] excluded {test_duplicate_count:,} test rows duplicated in held-out "
              "validation or internally", flush=True)
    tmp = data_file.with_name(data_file.name + ".part")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **{f"{split}_{key}": getattr(packs[split], key)
                                       for split in ("train", "validation", "test")
                                       for key in ("flat", "offsets", "labels")})
    os.replace(tmp, data_file)
    atomic_json(meta_file, metadata)
    return packs, metadata


def make_batches(lengths: np.ndarray, batch_size: int, bucket_size: int, seed: int) -> list[np.ndarray]:
    """Randomized length buckets reduce padding without changing sample frequency."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(lengths))
    batches = []
    for start in range(0, len(perm), bucket_size):
        ids = perm[start:start + bucket_size]
        ids = ids[np.argsort(lengths[ids], kind="stable")]
        for i in range(0, len(ids), batch_size):
            batches.append(ids[i:i + batch_size])
    rng.shuffle(batches)
    return batches


def make_balanced_batches(
    labels: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    grad_accum: int,
    seed: int,
    epoch_size: int = 0,
) -> list[np.ndarray]:
    """Deterministic six-way stratification per optimizer step.

    Samples are drawn from independently shuffled per-class pools. On pool
    exhaustion, that class is reshuffled and reused. No examples are thrown
    away permanently; for overrepresented classes, each epoch samples only a
    portion and subsequent epochs use new seeded permutations.

    Returns contiguous micro-batches, so consecutive `grad_accum` batches
    form a nearly uniform optimizer update. Each class appears equally often
    per epoch within one example, provided epoch_size is nonzero.
    """
    labels = np.asarray(labels, dtype=np.int64)
    lengths = np.asarray(lengths)
    if len(labels) != len(lengths) or not len(labels):
        raise ValueError("labels and lengths must be equal, nonempty arrays")
    if batch_size < 1 or grad_accum < 1:
        raise ValueError("batch_size and grad_accum must be >= 1")
    if epoch_size == 0:
        epoch_size = len(labels)
    if epoch_size < 1:
        raise ValueError("epoch_size must be positive")

    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(labels == c) for c in range(len(LABELS))]
    if any(len(pool) == 0 for pool in pools):
        raise ValueError("balanced sampler requires examples for every emotion")
    for pool in pools:
        rng.shuffle(pool)
    positions = [0 for _ in LABELS]

    def draw(cls: int, count: int) -> np.ndarray:
        picks = []
        while count:
            pool = pools[cls]
            pos = positions[cls]
            available = len(pool) - pos
            if available == 0:
                rng.shuffle(pool)
                pos = 0
                available = len(pool)
            take = min(count, available)
            picks.append(pool[pos:pos + take].copy())
            positions[cls] = pos + take
            count -= take
        return np.concatenate(picks) if picks else np.empty(0, dtype=np.int64)

    effective_size = batch_size * grad_accum
    batches: list[np.ndarray] = []
    for start in range(0, epoch_size, effective_size):
        size = min(effective_size, epoch_size - start)
        assignments = (start + np.arange(size)) % len(LABELS)
        rng.shuffle(assignments)
        selected = np.empty(size, dtype=np.int64)
        for cls in range(len(LABELS)):
            slots = np.flatnonzero(assignments == cls)
            if len(slots):
                selected[slots] = draw(cls, len(slots))
        # Only sort WITHIN this optimizer update, reducing padding without
        # sacrificing its six-way balance. Shuffle the resulting microbatches.
        selected = selected[np.argsort(lengths[selected], kind="stable")]
        local_batches = [selected[i:i + batch_size] for i in range(0, size, batch_size)]
        rng.shuffle(local_batches)
        batches.extend(local_batches)
    return batches


def training_batches(train: PackedSplit, args, epoch: int) -> list[np.ndarray]:
    seed = args.seed + epoch
    if args.sampling == "balanced":
        return make_balanced_batches(train.labels, train.lengths, args.batch_size,
                                     args.grad_accum, seed, args.epoch_samples)
    return make_batches(train.lengths, args.batch_size, args.bucket_size, seed)


def metric_report(truth: np.ndarray, predictions: np.ndarray) -> dict:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(matrix, (truth, predictions), 1)
    total = int(matrix.sum())
    per_class = {}
    f1_list = []
    weighted = 0.0
    for i, name in enumerate(LABELS):
        tp = int(matrix[i, i])
        support = int(matrix[i, :].sum())
        predicted = int(matrix[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_list.append(f1)
        weighted += f1 * support
        per_class[name] = {"precision": round(precision, 6), "recall": round(recall, 6),
                           "f1": round(f1, 6), "support": support}
    return {"accuracy": round(float(np.trace(matrix) / max(1, total)), 6),
            "macro_f1": round(float(np.mean(f1_list)), 6),
            "weighted_f1": round(float(weighted / max(1, total)), 6),
            "per_class": per_class, "confusion_matrix": matrix.tolist(), "num_examples": total,
            "matrix_rows_actual_columns_predicted": True}


def device_precision(args):
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    precision = args.precision
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "fp32"
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("bf16/fp16 require CUDA")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    print(f"[device] {device}, precision={precision}", flush=True)
    return device, precision, dtype


def autocast(device: torch.device, precision: str, dtype: torch.dtype):
    return torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" and precision != "fp32" else nullcontext()


def load_pretrained(args, device):
    ckpt_path = Path(args.foundation_dir) / "pretrained_sfc128_backbone.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("missing Stage 1 backbone export: " + str(ckpt_path))
    artifact = load_torch(ckpt_path)
    if artifact.get("architecture") != "SemanticFloatingClassifier":
        raise ValueError("wrong pretraining artifact architecture")
    if "state_dict" not in artifact or "config" not in artifact:
        raise ValueError("pretraining artifact requires state_dict and config")
    original = SFCConfig(**artifact["config"])
    if original.vocab_size != 1000 or original.pad_token_id != 0 or original.decision_token_id != 3:
        raise ValueError("pretrained config mismatches 1K tokenizer IDs")
    if not args.smoke_test and original.depth != 128:
        raise ValueError(f"expected real SFC-128 depth=128, found {original.depth}")
    cfg = replace(original, num_classes=6, max_context=max(original.max_context, args.max_text_tokens),
                  gradient_checkpointing=not args.no_gradient_checkpointing,
                  label_smoothing=args.label_smoothing)
    model = SemanticFloatingMeanPoolClassifier(cfg)
    outcome = model.load_state_dict(artifact["state_dict"], strict=False)
    if set(outcome.missing_keys) != {"decision_head.weight", "decision_head.bias"} or outcome.unexpected_keys:
        raise RuntimeError(f"incompatible pretraining checkpoint: {outcome}")
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError("all parameters must be trainable")
    print(f"[model] loaded pretrained backbone; MASKED MEAN readout; {cfg.depth} unique layers; "
          f"{sum(p.numel() for p in model.parameters()):,} trainable parameters", flush=True)
    return model.to(device), cfg


def make_optimizer(model: SemanticFloatingMeanPoolClassifier, args):
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {
        ("backbone", True): [], ("backbone", False): [],
        ("head", True): [], ("head", False): []}
    for name, param in model.named_parameters():
        is_head = name.startswith("decision_head.")
        decay = param.ndim >= 2 and not name.endswith(".bias")
        groups[("head" if is_head else "backbone", decay)].append(param)
    options = []
    for (component, decay), params in groups.items():
        if params:
            base = args.head_lr if component == "head" else args.backbone_lr
            options.append({"params": params, "lr": base, "base_lr": base,
                            "weight_decay": args.weight_decay if decay else 0.0})
    return torch.optim.AdamW(options, betas=(0.9, 0.999), eps=1e-8)


def learning_rate_multiplier(step: int, total_steps: int, warmup: int, min_lr_ratio: float) -> float:
    if step <= warmup:
        return step / max(1, warmup)
    frac = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * frac))


def evaluate(model: SemanticFloatingMeanPoolClassifier, data: PackedSplit, args, device, precision, dtype,
             return_logits: bool = False):
    was_training = model.training
    model.eval()
    ids = np.arange(len(data))
    predictions = []
    truth = []
    all_logits = []
    loss_sum = 0.0
    try:
        with torch.inference_mode():
            for offset in range(0, len(ids), args.eval_batch_size):
                x, y = data.batch(ids[offset:offset + args.eval_batch_size], model.cfg.pad_token_id)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast(device, precision, dtype):
                    output = model(x)
                scores = output.logits.float()
                loss_sum += F.cross_entropy(scores, y.long(), reduction="sum").item()
                predictions.append(scores.argmax(dim=-1).cpu().numpy())
                truth.append(y.cpu().numpy())
                if return_logits:
                    all_logits.append(scores.cpu().numpy())
    finally:
        model.train(was_training)
    pred = np.concatenate(predictions)
    target = np.concatenate(truth)
    stats = metric_report(target, pred)
    stats["loss"] = round(loss_sum / len(data), 6)
    if return_logits:
        return stats, np.concatenate(all_logits), target
    return stats


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Simple validation-only post-hoc calibration; NEVER trained on test labels."""
    x = torch.from_numpy(logits.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.int64))
    log_temp = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.2, max_iter=80, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        temperature = log_temp.clamp(math.log(0.25), math.log(5)).exp()
        loss = F.cross_entropy(x / temperature, y)
        loss.backward()
        return loss
    optimizer.step(closure)
    return round(float(log_temp.detach().clamp(math.log(0.25), math.log(5)).exp()), 6)


def get_rng_state(device) -> dict:
    # Convert NumPy's uint32 RNG array into ordinary integers so that the
    # entire checkpoint can be loaded with torch.load(weights_only=True).
    np_state = np.random.get_state()
    return {"torch_cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "python": random.getstate(),
            "numpy": (np_state[0], np_state[1].tolist(), np_state[2],
                      np_state[3], np_state[4])}


def set_rng_state(state: dict, device):
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    saved_np = state["numpy"]
    np.random.set_state((saved_np[0], np.asarray(saved_np[1], dtype=np.uint32),
                         saved_np[2], saved_np[3], saved_np[4]))


def save_latest(out, model, optimizer, scaler, progress, signature, device):
    save_torch(out / "checkpoint_latest.pt", {
        "signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "progress": dict(progress), "rng": get_rng_state(device)})


def save_best(out, model, progress, metrics, cfg, signature):
    save_torch(out / "checkpoint_best_emotion.pt", {
        "architecture": "SemanticFloatingMeanPoolClassifier", "config": asdict(cfg),
        "readout": "masked_mean", "model": model.state_dict(), "progress": dict(progress),
        "validation_metrics": metrics, "signature": signature})


def export_emotion(out, model, cfg, args, meta, validation_stats, test_stats, temperature, signature):
    dest = out / "SFC128_Emotion"
    dest.mkdir(parents=True, exist_ok=True)
    deploy_cfg = replace(cfg, gradient_checkpointing=False, dropout=0.0, embedding_dropout=0.0)
    save_torch(dest / "model.pt", {
        "architecture": "SemanticFloatingMeanPoolClassifier", "config": asdict(deploy_cfg),
        "readout": "masked_mean", "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "labels": list(LABELS), "temperature": temperature,
    })
    shutil.copy2(Path(args.foundation_dir) / "tokenizer_1k.json", dest / "tokenizer_1k.json")
    for module in ("semantic_floating_ren_t_128.py", "sfc128_emotion_meanpool_model.py",
                   "sfc128_emotion_tokenizer.py", "sfc128_emotion_v2_inference.py"):
        shutil.copy2(Path(__file__).resolve().parent / module, dest / module)
    atomic_json(dest / "config.json", {"model": asdict(deploy_cfg), "temperature": temperature,
                                       "max_text_tokens": args.max_text_tokens,
                                       "input": "ASCII English text; masked mean across all non-pad tokens, no appended decision token",
                                       "labels": list(LABELS)})
    atomic_json(dest / "labels.json", {str(i): label for i, label in enumerate(LABELS)})
    atomic_json(dest / "metrics.json", {"validation": validation_stats, "test": test_stats,
                                        "data_audit": meta, "train_signature": signature})
    (dest / "README.txt").write_text(
        "SFC128 Emotion v2: 128 REN-T layers, masked-mean readout, one 6-class head.\n"
        "Run: python sfc128_emotion_v2_inference.py --model-dir . --text \"I feel wonderful today\"\n"
        "Dependencies: torch and numpy. This is a text classifier, not a diagnosis.\n",
        encoding="utf-8")
    print(f"[export] standalone emotion model saved: {dest}", flush=True)


def run_training(args, out: Path, packs: dict[str, PackedSplit], meta: dict, foundation_hash: str):
    device, precision, dtype = device_precision(args)
    model, cfg = load_pretrained(args, device)
    optimizer = make_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))
    batches_per_epoch = len(training_batches(packs["train"], args, 0))
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    counts = np.bincount(packs["train"].labels.astype(np.int64), minlength=len(LABELS)).astype(np.float64)
    if (counts == 0).any():
        raise ValueError("training set must contain at least one example for each emotion label")
    if args.sampling == "balanced" and args.class_weighting != "none":
        raise ValueError("Do not combine balanced sampling with class-weighted loss. "
                         "Use --class-weighting none for balanced runs.")
    exponent = {"none": 0.0, "sqrt": 0.5, "inverse": 1.0}[args.class_weighting]
    weights = (counts.sum() / (len(LABELS) * counts)) ** exponent
    weights /= np.average(weights, weights=counts)
    class_weights = (None if args.class_weighting == "none" else
                     torch.tensor(weights, device=device, dtype=torch.float32))
    print("[train] raw label counts: " + ", ".join(f"{LABELS[i]}={int(counts[i]):,}" for i in range(6)),
          flush=True)
    print(f"[train] sampling={args.sampling}, loss_weights={args.class_weighting}; "
          "validation/test distributions remain natural", flush=True)
    print("[train] label weights: " + ", ".join(f"{LABELS[i]}={weights[i]:.3f}" for i in range(6)),
          flush=True)
    if args.sampling == "balanced":
        reference = training_batches(packs["train"], args, 0)
        sampled = np.bincount(packs["train"].labels[np.concatenate(reference)].astype(np.int64),
                              minlength=len(LABELS))
        print("[train] expected sampled labels per epoch: " + ", ".join(
              f"{LABELS[i]}={int(sampled[i]):,}" for i in range(len(LABELS))), flush=True)
    args_signature = {"trainer_version": "emotion_v2_meanpool_balanced", "readout": "masked_mean",
        "sampling": args.sampling, "epoch_samples": args.epoch_samples,
        "foundation_sha256": foundation_hash,
        "tokenizer_sha256": meta["signature"]["tokenizer_sha256"],
        "data_signature": meta["signature"], "model_config": asdict(cfg),
        "data_examples": len(packs["train"]), "epochs": args.epochs,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "bucket_size": args.bucket_size, "seed": args.seed,
        "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
        "warmup_steps": args.warmup_steps, "min_lr_ratio": args.min_lr_ratio,
        "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
        "eval_every": args.eval_every, "patience": args.patience,
        "min_delta": args.min_delta, "max_text_tokens": args.max_text_tokens,
        "label_smoothing": args.label_smoothing, "precision": args.precision,
        "class_weighting": args.class_weighting}
    signature = stable_hash(args_signature)
    progress = {"epoch": 0, "next_batch": 0, "step": 0, "best_f1": -1.0,
                "best_val_loss": float("inf"), "bad_evals": 0,
                "last_val_step": -1, "stopped_early": False}
    last = out / "checkpoint_latest.pt"
    if last.exists():
        checkpoint = load_torch(last)
        if checkpoint["signature"] != signature:
            raise ValueError("checkpoint training settings differ; choose a fresh output directory")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        progress = checkpoint["progress"]
        set_rng_state(checkpoint["rng"], device)
        print(f"[resume] epoch={progress['epoch']}, step={progress['step']}, "
              f"next_batch={progress['next_batch']}", flush=True)
    else:
        atomic_json(out / "training_settings.json", args_signature)

    if (progress["epoch"] >= args.epochs or progress["stopped_early"]) and \
            (out / "SFC128_Emotion" / "model.pt").exists():
        print("[complete] final emotion model already exported; no training repeated", flush=True)
        return
    print(f"[train] total_optimizer_steps={total_steps:,}, batch={args.batch_size}, "
          f"accum={args.grad_accum}, epochs={args.epochs}; "
          f"{len(packs['train']):,} training examples", flush=True)
    start_time = time.monotonic()
    if not progress["stopped_early"]:
        for epoch in range(progress["epoch"], args.epochs):
            batches = training_batches(packs["train"], args, epoch)
            cursor = int(progress["next_batch"]) if epoch == progress["epoch"] else 0
            if cursor < 0 or cursor > len(batches):
                raise RuntimeError("bad resumed batch cursor")
            model.train()
            while cursor < len(batches):
                micro_batches = batches[cursor:cursor + args.grad_accum]
                effective_size = sum(len(ids) for ids in micro_batches)
                optimizer.zero_grad(set_to_none=True)
                train_loss = 0.0
                for indices in micro_batches:
                    input_ids, labels = packs["train"].batch(indices, model.cfg.pad_token_id)
                    input_ids = input_ids.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with autocast(device, precision, dtype):
                        output = model(input_ids)
                        classification_loss = F.cross_entropy(
                            output.logits.float(), labels.long(), weight=class_weights,
                            label_smoothing=cfg.label_smoothing)
                        weighted = classification_loss * (len(indices) / effective_size)
                    scaler.scale(weighted).backward()
                    train_loss += float(weighted.detach())
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
                next_step = progress["step"] + 1
                multiplier = learning_rate_multiplier(next_step, total_steps,
                                                      args.warmup_steps, args.min_lr_ratio)
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * multiplier
                scaler.step(optimizer)
                scaler.update()
                cursor += len(micro_batches)
                progress.update(epoch=epoch, next_batch=cursor, step=next_step)

                if next_step % args.log_every == 0 or next_step == 1:
                    write_log(out, {"event": "train", "epoch": epoch + 1,
                        "step": next_step, "epoch_fraction": round(cursor / len(batches), 5),
                        "loss": round(train_loss, 6), "grad_norm": round(grad_norm, 5),
                        "backbone_lr": optimizer.param_groups[0]["lr"],
                        "elapsed_s": round(time.monotonic() - start_time, 1)})
                due_eval = next_step % args.eval_every == 0 or cursor == len(batches)
                if due_eval:
                    val_stats = evaluate(model, packs["validation"], args, device, precision, dtype)
                    f1, val_loss = val_stats["macro_f1"], val_stats["loss"]
                    improved = (f1 > progress["best_f1"] + args.min_delta or
                                (abs(f1 - progress["best_f1"]) <= args.min_delta and
                                 val_loss < progress["best_val_loss"] - 1e-6))
                    if improved:
                        progress["best_f1"] = f1
                        progress["best_val_loss"] = val_loss
                        progress["bad_evals"] = 0
                        save_best(out, model, progress, val_stats, cfg, signature)
                    else:
                        progress["bad_evals"] += 1
                    progress["last_val_step"] = next_step
                    predicted_counts = np.asarray(val_stats["confusion_matrix"]).sum(axis=0)
                    write_log(out, {"event": "eval", "step": next_step,
                                    "epoch": epoch + 1, "macro_f1": f1,
                                    "accuracy": val_stats["accuracy"], "loss": val_loss,
                                    "per_class_f1": {name: val_stats["per_class"][name]["f1"]
                                                     for name in LABELS},
                                    "predicted_counts": {name: int(predicted_counts[i])
                                                         for i, name in enumerate(LABELS)},
                                    "best": improved, "bad_evals": progress["bad_evals"]})
                if next_step % args.save_every == 0 or due_eval:
                    save_latest(out, model, optimizer, scaler, progress, signature, device)
                if args.max_steps and progress["step"] >= args.max_steps:
                    save_latest(out, model, optimizer, scaler, progress, signature, device)
                    print("[pause] max-steps reached. Rerun with a larger --max-steps or "
                          "without it to resume.", flush=True)
                    return
                if due_eval and args.patience and progress["bad_evals"] >= args.patience:
                    progress["stopped_early"] = True
                    save_latest(out, model, optimizer, scaler, progress, signature, device)
                    print("[early-stop] validation macro-F1 stopped improving", flush=True)
                    break
            if progress["stopped_early"]:
                break
            progress["epoch"] = epoch + 1
            progress["next_batch"] = 0
            save_latest(out, model, optimizer, scaler, progress, signature, device)
        print("[train] training finished; evaluating BEST validation checkpoint", flush=True)
    if not (out / "checkpoint_best_emotion.pt").exists():
        raise RuntimeError("no validation checkpoint was saved")
    best = load_torch(out / "checkpoint_best_emotion.pt")
    if best["signature"] != signature:
        raise ValueError("best checkpoint and training signature differ")
    model.load_state_dict(best["model"])
    validation, val_logits, val_truth = evaluate(model, packs["validation"], args, device,
                                                precision, dtype, return_logits=True)
    temperature = fit_temperature(val_logits, val_truth) if not args.no_calibration else 1.0
    test = evaluate(model, packs["test"], args, device, precision, dtype)
    write_log(out, {"event": "complete", "best_step": best["progress"]["step"],
                    "validation_macro_f1": validation["macro_f1"],
                    "test_macro_f1": test["macro_f1"], "test_accuracy": test["accuracy"],
                    "temperature": temperature})
    export_emotion(out, model, cfg, args, meta, validation, test, temperature, signature)


def smoke_test() -> None:
    """No network. Test transfer, leakage exclusion, all-parameter fine-tune,
    checkpoint resume, evaluation, and independent standalone inference."""
    print("[smoke] preparing miniature synthetic emotion training test", flush=True)
    with tempfile.TemporaryDirectory(prefix="sfc_emotion_smoke_") as tmp:
        root = Path(tmp)
        foundation = root / "foundation"
        foundation.mkdir()
        from sfc128_emotion_tokenizer import SPECIAL, ASCII_CHARS
        from sfc128_emotion_meanpool_model import test_mean_pool
        test_mean_pool()
        demo_labels = np.asarray([0] * 17 + [1] * 4 + [2] * 3 + [3] * 6 + [4] * 2 + [5] * 1)
        demo_lengths = np.arange(len(demo_labels)) % 11 + 1
        sampling_batches = make_balanced_batches(demo_labels, demo_lengths, 3, 2, 123, 120)
        drawn = demo_labels[np.concatenate(sampling_batches)]
        assert np.array_equal(np.bincount(drawn, minlength=6), np.full(6, 20))
        # Every six draws are spread roughly evenly even when some source
        # categories contain far fewer items.
        assert len(set(np.concatenate(sampling_batches).tolist())) <= len(demo_labels)
        vocab = list(SPECIAL) + list(ASCII_CHARS)
        while len(vocab) < 1000:
            # Unused legal multi-character placeholders, just for a smoke test.
            i = len(vocab)
            n = i
            chars = []
            while n:
                chars.append(chr(ord("a") + n % 26))
                n //= 26
            vocab.append("zzq" + "".join(chars) + "qzz")
        atomic_json(foundation / "tokenizer_1k.json", {
            "kind": "ascii_constrained_bpe_v1", "vocab": vocab, "merges": []})
        test_tokenizer = SFCAsciiTokenizer.load(foundation / "tokenizer_1k.json")
        assert test_tokenizer.encode("Hello! 2026") == [test_tokenizer.ids[ch] for ch in "Hello! 2026"]
        orig_cfg = SFCConfig(vocab_size=1000, pad_token_id=0, decision_token_id=3,
                             num_classes=32, depth=2, hidden_size=16, bottleneck_size=8,
                             ren_groups=4, floating_hidden=8, max_context=64,
                             gradient_checkpointing=False)
        dummy = SemanticFloatingClassifier(orig_cfg)
        original_embedding = dummy.token_embeddings.weight.detach().clone()
        save_torch(foundation / "pretrained_sfc128_backbone.pt", {
            "architecture": "SemanticFloatingClassifier", "config": asdict(orig_cfg),
            "state_dict": {k: t for k, t in dummy.state_dict().items()
                           if not k.startswith("decision_head.")}})
        rows = {"split_train": [], "unsplit_train": [], "validation": [], "test": []}
        syllables = ("rainy", "bright", "loving", "furious", "nervous", "astonished")
        for label, word in enumerate(syllables):
            for i in range(8):
                rows["split_train"].append({"text": f"i am feeling {word} about news number {i}", "label": label})
            for i in range(9, 16):
                rows["unsplit_train"].append({"text": f"today i was {word} about event number {i}", "label": label})
            for i in range(2):
                rows["validation"].append({"text": f"a very {word} response for {i}", "label": label})
                rows["test"].append({"text": f"that was a {word} surprise for {i}", "label": label})
        # Exact held-out duplicate in unsplit MUST NOT enter train.
        rows["unsplit_train"].append(rows["validation"][0])
        rows["unsplit_train"].append(rows["test"][0])
        local = root / "emotion.json"
        atomic_json(local, rows)
        args = parse_args(["--smoke-test", "--foundation-dir", str(foundation),
                           "--output-dir", str(root / "run"), "--local-data", str(local),
                           "--max-text-tokens", "48", "--epochs", "2", "--batch-size", "4",
                           "--grad-accum", "1", "--bucket-size", "32", "--eval-batch-size", "4",
                           "--eval-every", "2", "--save-every", "2", "--log-every", "2",
                           "--max-steps", "2", "--device", "cpu", "--precision", "fp32",
                           "--backbone-lr", "0.005", "--head-lr", "0.01",
                           "--warmup-steps", "1", "--no-gradient-checkpointing"])
        run(args)
        assert not (Path(args.output_dir) / "SFC128_Emotion" / "model.pt").exists()
        args.max_steps = 0
        run(args)
        output = Path(args.output_dir) / "SFC128_Emotion"
        assert (output / "model.pt").exists()
        report = json.loads((Path(args.output_dir) / "prepared_data.json").read_text())
        assert report["training_audit"]["heldout_overlap_excluded"] >= 2
        loaded = load_torch(output / "model.pt")
        assert len(loaded["labels"]) == 6
        assert loaded["config"]["num_classes"] == 6
        assert loaded["config"]["depth"] == 2
        assert loaded["architecture"] == "SemanticFloatingMeanPoolClassifier"
        assert loaded["readout"] == "masked_mean"
        assert not torch.equal(loaded["state_dict"]["token_embeddings.weight"], original_embedding)
        from sfc128_emotion_v2_inference import EmotionPredictor
        predictor = EmotionPredictor(output, device="cpu")
        result = predictor.predict("I feel bright today!", top_k=6)
        assert result["label"] in LABELS and len(result["top_predictions"]) == 6
        assert abs(sum(item["probability"] for item in result["top_predictions"]) - 1.0) < 1e-5
        print("[smoke] PASS: original checkpoint transfer; 6-way sampler; "
              "masked mean; strict dedup; training/resume; standalone inference", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage 2 v2: six-way balanced, masked-mean SFC-128 emotion fine-tuning")
    p.add_argument("--foundation-dir", default="./sfc_fineweb_run",
                   help="Stage 1 dir with pretrained_sfc128_backbone.pt and tokenizer_1k.json")
    p.add_argument("--output-dir", default="./sfc_emotion_v2_run")
    p.add_argument("--data-mode", choices=("full", "pilot"), default="full",
                   help="full: deduplicated unsplit+split/train; pilot: official split/train only")
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--dataset-cache", default=None)
    p.add_argument("--local-data", default=None, help="offline tests: JSON split_train/unsplit_train/validation/test")
    p.add_argument("--max-text-tokens", type=int, default=128,
                   help="max tokenizer tokens per example (no decision token in v2)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--bucket-size", type=int, default=1024,
                   help="number of shuffled examples sorted by length at a time")
    p.add_argument("--backbone-lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--sampling", choices=("balanced", "natural"), default="balanced",
                   help="balanced: uniform six-way samples per optimizer update; natural: original distribution")
    p.add_argument("--epoch-samples", type=int, default=0,
                   help="balanced-only number of sample draws per epoch; 0 = full deduplicated train size")
    p.add_argument("--class-weighting", choices=("none", "sqrt", "inverse"), default="none",
                   help="default none; do NOT use with balanced sampling")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--patience", type=int, default=8, help="validation checks without improvement; 0 disables")
    p.add_argument("--min-delta", type=float, default=0.0001)
    p.add_argument("--max-steps", type=int, default=0,
                   help="pause after this global optimizer step; 0 runs to completion")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--no-calibration", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args(argv)
    for name in ("epochs", "batch_size", "grad_accum", "eval_batch_size", "bucket_size",
                 "max_text_tokens", "eval_every", "save_every", "log_every"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be >= 1")
    for name in ("backbone_lr", "head_lr", "grad_clip"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.epoch_samples < 0:
        p.error("epoch-samples must be nonnegative")
    if args.sampling == "balanced" and args.class_weighting != "none":
        p.error("balanced sampling must use --class-weighting none to avoid double-balancing")
    if args.sampling == "natural" and args.epoch_samples:
        p.error("epoch-samples can only be set for balanced sampling")
    if args.max_steps < 0 or args.patience < 0 or args.warmup_steps < 0:
        p.error("max-steps, patience and warmup-steps must be nonnegative")
    if not 0 < args.min_lr_ratio <= 1 or not 0 <= args.label_smoothing < 1:
        p.error("invalid min-lr-ratio or label-smoothing")
    if args.bucket_size < args.batch_size:
        p.error("bucket-size must be >= batch-size")
    return args


def run(args):
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    foundation = Path(args.foundation_dir).expanduser().resolve()
    args.foundation_dir = str(foundation)
    tokenizer_path = foundation / "tokenizer_1k.json"
    model_path = foundation / "pretrained_sfc128_backbone.pt"
    if not tokenizer_path.exists() or not model_path.exists():
        raise FileNotFoundError("Stage 1 foundation must contain tokenizer_1k.json "
                                "and pretrained_sfc128_backbone.pt")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = SFCAsciiTokenizer.load(tokenizer_path)
    foundation_hash = file_sha256(model_path)
    packs, metadata = prepare_dataset(args, out, tokenizer, foundation_hash)
    run_training(args, out, packs, metadata, foundation_hash)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoke_test:
        smoke_test()
    else:
        run(arguments)
