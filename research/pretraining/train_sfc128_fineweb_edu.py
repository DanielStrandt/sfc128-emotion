#!/usr/bin/env python3
"""SFC-128: constrained 1K ASCII tokenizer + FineWeb-Edu pretraining.

The DEFAULT full run:
  1. Stream FineWeb-Edu sample-10BT and collect ~5M *source-reference* tokens
     to learn a 1,000-ID alphabetic-only BPE tokenizer.
  2. Stream again, encode 200M tokens of training data plus 2M held-out tokens.
     Persist the exact token IDs locally in restartable uint16 shards.
  3. Pretrain the original 128-INDEPENDENT-layer causal REN-T model with a
     tied-weight next-token language head for TWO FULL epochs.
  4. Export the pretrained backbone for subsequent decision fine-tuning.

Rules: ASCII only; multi-character ordinary tokens contain [A-Za-z] only;
ASCII digits, punctuation, space, newline and tab are single-character tokens.
Eight special CONTROL tokens are exempt from the ordinary-token rule.

Setup:
  pip install torch numpy datasets huggingface_hub tqdm
  # Put semantic_floating_ren_t_128.py in the same directory as this script.

Full run (one terminal command):
  python train_sfc128_fineweb_edu.py --output-dir ./sfc_fineweb_run

Offline end-to-end test (requires only torch and numpy):
  python train_sfc128_fineweb_edu.py --smoke-test

Progress, data and optimizer checkpoints persist in --output-dir. Re-running
with the same arguments automatically resumes the corpus and training.

Data: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
FineWeb-Edu is ODC-By licensed; see the dataset card for attribution and terms.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import lru_cache
import heapq
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import string
import sys
import tempfile
import time
from typing import Iterator, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from semantic_floating_ren_t_128 import SFCConfig, SemanticFloatingClassifier

# The eight control strings are IDs for the model, never merges in ordinary text.
SPECIAL = ("<PAD>", "<BOS>", "<EOS>", "<DECIDE>", "<SEP>", "<UNK>", "<MASK>", "<RESERVED>")
ASCII_CHARS = tuple(chr(c) for c in range(32, 127)) + ("\n", "\t")
assert len(SPECIAL) == 8 and len(ASCII_CHARS) == 97
WORD_RE = re.compile(r"[A-Za-z]+")
SEGMENT_RE = re.compile(r"[A-Za-z]+|[^A-Za-z]")
ALPHA_RE = re.compile(r"[A-Za-z]+\Z")
PUNCT_REMAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
    "\u2212": "-", "\u00a0": " ", "\u2026": "...", "\u2022": "*", "\u200b": "", "\ufeff": "",
})
ALLOWED = set(ASCII_CHARS)


def write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_english_ascii(raw: str, min_chars: int, max_nonascii_fraction: float) -> Optional[str]:
    """Normalize English typography, never leave non-ASCII codepoints.

    No translation of non-English words is attempted. Reject documents with
    substantial remaining non-ASCII text; replace sporadic unsupported chars
    by a space instead of silently concatenating neighboring words.
    """
    if not isinstance(raw, str):
        return None
    raw = raw.translate(PUNCT_REMAP).replace("\r\n", "\n").replace("\r", "\n")
    if not raw:
        return None
    bad = sum(ch not in ALLOWED for ch in raw)
    if bad / len(raw) > max_nonascii_fraction:
        return None
    cleaned = "".join(ch if ch in ALLOWED else " " for ch in raw)
    return cleaned if len(cleaned.strip()) >= min_chars and WORD_RE.search(cleaned) else None


def _base26(n: int) -> str:
    out = ""
    while True:
        out = chr(97 + n % 26) + out
        n //= 26
        if n == 0:
            return out


class AsciiConstrainedBPE:
    """Fully deterministic, alphabetic-only BPE; zero tokenizer dependencies.

    The algorithm learns pair merges *only within individual ASCII words*.
    Single ASCII characters provide an exact reversible fallback for every
    allowed input. Regex isolates every non-letter separately, so numerals,
    spaces, and punctuation can never participate in a learned merge.
    """

    def __init__(self, vocab: list[str], merges: list[tuple[str, str]]) -> None:
        self.vocab = list(vocab)
        self.ids = {s: i for i, s in enumerate(vocab)}
        if len(self.ids) != len(vocab):
            raise ValueError("vocabulary contains duplicate tokens")
        self.merges = [tuple(m) for m in merges]
        self.ranks = {p: i for i, p in enumerate(self.merges)}
        self._validate()

    def _validate(self) -> None:
        if len(self.vocab) != 1000 or self.vocab[:8] != list(SPECIAL):
            raise ValueError("tokenizer must have precisely 1,000 IDs and eight fixed specials")
        if not all(ch in self.ids for ch in ASCII_CHARS):
            raise ValueError("ASCII fallback alphabet incomplete")
        for token in self.vocab[len(SPECIAL):]:
            if len(token) > 1 and not ALPHA_RE.fullmatch(token):
                raise ValueError(f"forbidden multi-character token: {token!r}")
            if len(token) == 1 and token not in ALLOWED:
                raise ValueError(f"forbidden character: {token!r}")
        if any(not ALPHA_RE.fullmatch(a + b) for a, b in self.merges):
            raise ValueError("non-alphabetic BPE merge found")

    @classmethod
    def train(
        cls, corpus_jsonl: Path, vocab_size: int = 1000,
        min_pair_frequency: int = 2, max_piece_chars: int = 28,
        max_distinct_words: int = 200_000,
    ) -> "AsciiConstrainedBPE":
        if vocab_size != 1000:
            raise ValueError("this model uses exactly 1000 tokens")
        print("[tokenizer] counting English alphabetic word frequencies...", flush=True)
        word_counts: Counter[str] = Counter()
        with corpus_jsonl.open("r", encoding="ascii") as f:
            for line in f:
                row = json.loads(line)
                word_counts.update(WORD_RE.findall(row["text"]))
        if not word_counts:
            raise RuntimeError("no English words in tokenizer sample")
        all_distinct = len(word_counts)
        if len(word_counts) > max_distinct_words:
            word_counts = Counter(dict(word_counts.most_common(max_distinct_words)))
        print(f"[tokenizer] inspected {all_distinct:,} distinct words; "
              f"using {len(word_counts):,} most frequent for BPE merges", flush=True)

        words = list(word_counts)
        freqs = [word_counts[w] for w in words]
        symbols: list[tuple[str, ...]] = [tuple(w) for w in words]
        pair_count: Counter[tuple[str, str]] = Counter()
        inverted: dict[tuple[str, str], set[int]] = defaultdict(set)
        for wid, syms in enumerate(symbols):
            counts = Counter(zip(syms, syms[1:]))
            for p, occurrence in counts.items():
                pair_count[p] += occurrence * freqs[wid]
                inverted[p].add(wid)

        heap = [(-count, pair) for pair, count in pair_count.items()]
        heapq.heapify(heap)
        vocab = list(SPECIAL) + list(ASCII_CHARS)
        existing = set(vocab)
        merges: list[tuple[str, str]] = []
        max_iterations = vocab_size * 4
        while len(vocab) < vocab_size and heap and len(merges) < max_iterations:
            negfreq, pair = heapq.heappop(heap)
            frequency = pair_count.get(pair, 0)
            if frequency != -negfreq or frequency == 0:
                continue  # stale heap entry
            if frequency < min_pair_frequency:
                break
            combined = pair[0] + pair[1]
            if len(combined) > max_piece_chars:
                pair_count.pop(pair, None)
                continue
            if not ALPHA_RE.fullmatch(combined):
                raise AssertionError("invalid alphabetic merge")
            impacted = list(inverted.get(pair, ()))
            touched: set[tuple[str, str]] = set()
            for wid in impacted:
                old = symbols[wid]
                if len(old) < 2:
                    continue
                old_pairs = Counter(zip(old, old[1:]))
                if pair not in old_pairs:
                    raise AssertionError("stale inverted word index")
                newer = []
                pos = 0
                while pos < len(old):
                    if pos + 1 < len(old) and (old[pos], old[pos + 1]) == pair:
                        newer.append(combined)
                        pos += 2
                    else:
                        newer.append(old[pos])
                        pos += 1
                new = tuple(newer)
                symbols[wid] = new
                new_pairs = Counter(zip(new, new[1:]))
                for p in old_pairs.keys() | new_pairs.keys():
                    oc, nc = old_pairs[p], new_pairs[p]
                    if oc != nc:
                        pair_count[p] += (nc - oc) * freqs[wid]
                        touched.add(p)
                        if nc:
                            inverted[p].add(wid)
                        else:
                            ids = inverted.get(p)
                            if ids is not None:
                                ids.discard(wid)
                                if not ids:
                                    inverted.pop(p, None)
            merges.append(pair)
            if combined not in existing:
                vocab.append(combined)
                existing.add(combined)
            for p in touched:
                count = pair_count[p]
                if count > 0:
                    heapq.heappush(heap, (-count, p))
                else:
                    pair_count.pop(p, None)
            if len(heap) > max(500_000, 6 * len(pair_count)):
                heap = [(-n, p) for p, n in pair_count.items() if n > 0]
                heapq.heapify(heap)
            if len(merges) % 100 == 0:
                print(f"[tokenizer] {len(vocab):,}/1,000 IDs, {len(merges):,} BPE merges", flush=True)

        # Very small smoke corpora may not support 895 frequent distinct pieces.
        # Fill the remaining IDs with legal, intentionally unused alphabetic
        # sentinels; a genuine 5M reference-token sample should not need this.
        fill_i = 0
        while len(vocab) < vocab_size:
            candidate = "zzq" + _base26(fill_i) + "qzz"
            fill_i += 1
            if candidate not in existing:
                vocab.append(candidate)
                existing.add(candidate)
        print(f"[tokenizer] done: {len(vocab)} IDs, {len(merges)} merges, {fill_i} fillers", flush=True)
        return cls(vocab, merges)

    def save(self, path: Path) -> None:
        write_json_atomic(path, {"kind": "ascii_constrained_bpe_v1", "vocab": self.vocab, "merges": self.merges})

    @classmethod
    def load(cls, path: Path) -> "AsciiConstrainedBPE":
        state = read_json(path)
        if state.get("kind") != "ascii_constrained_bpe_v1":
            raise ValueError("unexpected tokenizer format")
        return cls(state["vocab"], state["merges"])

    @lru_cache(maxsize=200_000)
    def _encode_word(self, word: str) -> tuple[int, ...]:
        pieces = list(word)
        while len(pieces) > 1:
            best_rank: Optional[int] = None
            best_pair: Optional[tuple[str, str]] = None
            for p in zip(pieces, pieces[1:]):
                r = self.ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, p
            if best_pair is None:
                break
            updated: list[str] = []
            k = 0
            while k < len(pieces):
                if k + 1 < len(pieces) and (pieces[k], pieces[k + 1]) == best_pair:
                    updated.append(pieces[k] + pieces[k + 1])
                    k += 2
                else:
                    updated.append(pieces[k])
                    k += 1
            pieces = updated
        return tuple(self.ids[p] for p in pieces)

    def encode(self, text: str, append_eos: bool = False) -> list[int]:
        if any(ch not in ALLOWED for ch in text):
            raise ValueError("tokenizer accepts only printable ASCII, tab, newline")
        result: list[int] = []
        for segment in SEGMENT_RE.findall(text):
            if ALPHA_RE.fullmatch(segment):
                result.extend(self._encode_word(segment))
            else:
                result.append(self.ids[segment])
        if append_eos:
            result.append(self.ids["<EOS>"])
        return result

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return "".join(self.vocab[i] for i in ids if not (skip_special and i < len(SPECIAL)))

    @property
    def eos_id(self) -> int:
        return self.ids["<EOS>"]


def rows_from_source(args, start: int = 0) -> Iterator[tuple[int, dict]]:
    if args.local_jsonl:
        with Path(args.local_jsonl).open("r", encoding="utf-8") as f:
            for i, line in enumerate(itertools.islice(f, start, None), start=start):
                row = json.loads(line)
                yield i, row
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install data streaming dependency: pip install datasets") from exc
    options = {"name": args.dataset_config, "split": "train", "streaming": True}
    if args.dataset_revision:
        options["revision"] = args.dataset_revision
    stream = load_dataset(args.dataset, **options)
    if args.shuffle_buffer > 1:
        stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    if start:
        stream = stream.skip(start)
    for i, row in enumerate(stream, start=start):
        yield i, row


def accept_row(row: dict, args) -> Optional[str]:
    lang = row.get("language")
    if isinstance(lang, str) and lang.lower() not in {"en", "eng", "english"}:
        return None
    lang_score = row.get("language_score")
    if lang_score is not None and float(lang_score) < args.min_language_score:
        return None
    return clean_english_ascii(row.get("text", ""), args.min_doc_chars, args.max_nonascii_fraction)


def estimated_reference_tokens(row: dict, cleaned: str) -> int:
    # FineWeb-Edu's `token_count` is a source-side estimate (GPT-2 tokenization),
    # NOT the custom 1K tokenizer count. For vocab sampling only.
    n = row.get("token_count")
    if n is not None:
        try:
            n = int(n)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    return max(1, math.ceil(len(cleaned) / 4))


def build_tokenizer_sample(args, out: Path) -> Path:
    sample = out / "tokenizer_sample.jsonl"
    meta_file = out / "tokenizer_sample_meta.json"
    if sample.exists() and meta_file.exists():
        meta = read_json(meta_file)
        if meta["target_reference_tokens"] != args.tokenizer_sample_tokens:
            raise ValueError("existing tokenizer sample has different target; use a new output directory")
        return sample
    temporary = sample.with_name(sample.name + ".part")
    count = 0
    docs = 0
    start = time.time()
    with temporary.open("w", encoding="ascii") as f:
        for idx, row in rows_from_source(args):
            text = accept_row(row, args)
            if text is None:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")
            docs += 1
            count += estimated_reference_tokens(row, text)
            if docs % 1000 == 0:
                print(f"[sample] ~{count:,}/{args.tokenizer_sample_tokens:,} reference tokens; {docs:,} docs", flush=True)
            if count >= args.tokenizer_sample_tokens:
                break
        else:
            raise RuntimeError(f"stream exhausted at ~{count:,} sample reference tokens")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, sample)
    write_json_atomic(meta_file, {
        "reference_token_count": count,
        "target_reference_tokens": args.tokenizer_sample_tokens,
        "docs": docs,
        "reference_counting": "FineWeb-Edu token_count metadata or len(text)/4 fallback",
    })
    print(f"[sample] ready: ~{count:,} reference tokens, {docs:,} docs in {time.time()-start:.1f}s", flush=True)
    return sample


def build_tokenizer(args, out: Path) -> AsciiConstrainedBPE:
    path = out / "tokenizer_1k.json"
    if path.exists():
        print("[tokenizer] loading existing vocabulary", flush=True)
        return AsciiConstrainedBPE.load(path)
    sample = build_tokenizer_sample(args, out)
    tokenizer = AsciiConstrainedBPE.train(
        sample, vocab_size=1000, max_distinct_words=args.tokenizer_max_words)
    tokenizer.save(path)
    for example in ("Hello, 2026!\nNext line.", "Don't change US to us.", "x=1.23; yes\tno"):
        assert tokenizer.decode(tokenizer.encode(example)) == example
    print("[tokenizer] ASCII reversibility checks passed", flush=True)
    return tokenizer


def _corpus_signature(args) -> dict:
    return {"dataset": args.dataset if not args.local_jsonl else str(Path(args.local_jsonl).resolve()),
            "config": args.dataset_config, "revision": args.dataset_revision,
            "shuffle_buffer": args.shuffle_buffer, "seed": args.seed,
            "min_language_score": args.min_language_score,
            "min_doc_chars": args.min_doc_chars,
            "max_nonascii_fraction": args.max_nonascii_fraction,
            "tokenizer_sample_tokens": args.tokenizer_sample_tokens,
            "shard_tokens": args.shard_tokens}


def prepare_split(
    args, out: Path, tokenizer: AsciiConstrainedBPE, name: str,
    target: int, source_start: int, tokenizer_digest: str,
) -> dict:
    directory = out / "tokens"
    directory.mkdir(parents=True, exist_ok=True)
    state_file = directory / f"{name}_manifest.json"
    sig = _corpus_signature(args)
    if state_file.exists():
        state = read_json(state_file)
        if (state["signature"] != sig or state["target_tokens"] != target or
                state["source_start"] != source_start or state["tokenizer_sha256"] != tokenizer_digest):
            raise ValueError(f"{name} manifest disagrees with requested data configuration")
    else:
        state = {"format": "uint16_le", "name": name, "target_tokens": target,
                 "source_start": source_start, "source_next": source_start,
                 "tokens": 0, "documents": 0, "shards": [], "signature": sig,
                 "tokenizer_sha256": tokenizer_digest}
        write_json_atomic(state_file, state)

    for entry in state["shards"]:
        p = directory / entry["file"]
        if not p.exists() or p.stat().st_size != 2 * entry["tokens"]:
            raise RuntimeError(f"missing/corrupted prepared shard: {p}")
    # On a mid-write crash, an orphan newer shard is safe to delete, because
    # source_next is only committed after a complete shard and manifest update.
    for p in directory.glob(f"{name}_*.bin"):
        if p.name not in {e["file"] for e in state["shards"]}:
            p.unlink()
    if state["tokens"] >= target:
        return state

    next_shard = len(state["shards"])
    buffer: list[int] = []
    accepted_docs_since_commit = 0
    now_source_next = state["source_next"]
    start = time.time()

    def flush() -> None:
        nonlocal next_shard, accepted_docs_since_commit
        if not buffer:
            return
        filename = f"{name}_{next_shard:05d}.bin"
        path = directory / filename
        temp = directory / (filename + ".part")
        data = np.asarray(buffer, dtype="<u2")
        with temp.open("wb") as f:
            data.tofile(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        state["shards"].append({"file": filename, "tokens": len(buffer)})
        state["tokens"] += len(buffer)
        state["documents"] += accepted_docs_since_commit
        state["source_next"] = now_source_next
        write_json_atomic(state_file, state)
        next_shard += 1
        accepted_docs_since_commit = 0
        buffer.clear()
        print(f"[data/{name}] {state['tokens']:,}/{target:,} tokens; {len(state['shards'])} shards; "
              f"last shard {len(data):,} tokens; {time.time()-start:.1f}s", flush=True)

    for idx, row in rows_from_source(args, start=state["source_next"]):
        cleaned = accept_row(row, args)
        if cleaned is None:
            now_source_next = idx + 1
            continue
        ids = tokenizer.encode(cleaned, append_eos=True)
        if len(ids) < 2:
            now_source_next = idx + 1
            continue
        if len(buffer) + len(ids) > args.shard_tokens and buffer:
            # Commit previous docs. Current doc has not yet been consumed.
            now_source_next = idx
            flush()
        buffer.extend(ids)
        accepted_docs_since_commit += 1
        now_source_next = idx + 1
        if state["tokens"] + len(buffer) >= target:
            flush()
            break
    else:
        flush()
        raise RuntimeError(f"FineWeb-Edu stream exhausted at {state['tokens']:,}/{target:,} {name} tokens")
    return state


def prepare_corpus(args, out: Path, tokenizer: AsciiConstrainedBPE) -> tuple[dict, dict]:
    import hashlib
    sha = hashlib.sha256((out / "tokenizer_1k.json").read_bytes()).hexdigest()
    train = prepare_split(args, out, tokenizer, "train", args.train_tokens, 0, sha)
    valid = prepare_split(args, out, tokenizer, "validation", args.validation_tokens, train["source_next"], sha)
    return train, valid


class TokenShardBlocks:
    """Memory-mapped fixed-stride causal windows, each target seen once/epoch.

    For a shard s, sample j uses x=s[j*L:j*L+L], labels=x shifted
    by one token. Shard boundaries discard at most L trailing tokens.
    """

    def __init__(self, manifest: dict, directory: Path, sequence_length: int) -> None:
        self.seq = int(sequence_length)
        self.shards: list[np.memmap] = []
        self.cumulative: list[int] = [0]
        for shard in manifest["shards"]:
            path = directory / shard["file"]
            mm = np.memmap(path, dtype="<u2", mode="r", shape=(shard["tokens"],))
            self.shards.append(mm)
            count = max(0, (len(mm) - 1) // self.seq)
            self.cumulative.append(self.cumulative[-1] + count)
        self.total = self.cumulative[-1]
        if self.total == 0:
            raise ValueError("not enough corpus tokens for a single training window")

    def batch(self, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        b = len(indices)
        pairs = np.empty((b, self.seq + 1), dtype=np.int64)
        for i, global_index in enumerate(indices):
            idx = int(global_index)
            s = bisect.bisect_right(self.cumulative, idx) - 1
            local = idx - self.cumulative[s]
            start = local * self.seq
            pairs[i] = self.shards[s][start:start + self.seq + 1]
        inputs = torch.from_numpy(pairs[:, :-1].copy())
        targets = torch.from_numpy(pairs[:, 1:].copy())
        return inputs, targets


class SemanticFloatingLanguageModel(SemanticFloatingClassifier):
    """Same pretrained 128-layer classifier backbone; tied next-token head.

    Inherit from the classifier to keep identical state_dict keys for transfer.
    The untrained class head is deleted and NOT saved with the pretraining model.
    """

    def __init__(self, cfg: SFCConfig):
        super().__init__(cfg)
        del self.decision_head

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        if input_ids.ndim != 2 or input_ids.shape[1] > self.cfg.max_context:
            raise ValueError(f"expected (B,T) and T <= {self.cfg.max_context}")
        states = self.encode(input_ids)
        logits = F.linear(states, self.token_embeddings.weight)  # tied vocabulary head
        # Only EOS is a valid corpus control target; the other seven controls
        # are never emitted by the tokenizer on ordinary text.
        forbidden = (0, 1, 3, 4, 5, 6, 7)
        logits[..., list(forbidden)] = -1e4
        if labels is None:
            return logits, None
        if labels.shape != input_ids.shape:
            raise ValueError("labels must have same shape as input_ids")
        loss = F.cross_entropy(logits.float().reshape(-1, self.cfg.vocab_size), labels.reshape(-1))
        return logits, loss


def create_config(args) -> SFCConfig:
    return SFCConfig(
        vocab_size=1000, pad_token_id=0, decision_token_id=3,
        num_classes=args.num_decisions, max_context=args.max_context,
        depth=args.depth, hidden_size=args.width, bottleneck_size=args.rank,
        kernel_size=args.kernel_size, dilation_cycle=tuple(args.dilations),
        ren_groups=args.ren_groups, ren_alpha=0.75,
        ren_affine=True, use_floating_gate=(not args.no_floating),
        floating_hidden=args.gate_hidden,
        dropout=args.dropout,
        embedding_dropout=args.embedding_dropout,
        residual_gain_init=args.residual_gain_init,
        residual_scale_mode="sqrt",
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )


def device_and_precision(args):
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    precision = args.precision
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "fp32"
    if precision in ("bf16", "fp16") and device.type != "cuda":
        raise ValueError("bf16/fp16 autocast is currently supported only on CUDA by this trainer")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    return device, dtype, precision


def autocast_context(device: torch.device, precision: str, dtype: torch.dtype):
    return torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" and precision != "fp32" else nullcontext()


def cosine_lr(step: int, total: int, warmup: int, peak: float, min_ratio: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    torch.save(state, tmp)
    os.replace(tmp, path)


def export_backbone(out: Path, model: SemanticFloatingLanguageModel, cfg: SFCConfig, step: int) -> None:
    # Decision training: load this into SemanticFloatingClassifier(cfg) with
    # strict=False; only decision_head.weight/bias should be missing.
    state = {name: param.detach().cpu() for name, param in model.state_dict().items()}
    save_checkpoint(out / "pretrained_sfc128_backbone.pt", {
        "architecture": "SemanticFloatingClassifier", "config": asdict(cfg),
        "state_dict": state, "global_step": step, "tokenizer": "tokenizer_1k.json",
        "expected_missing_keys": ["decision_head.weight", "decision_head.bias"],
    })


def evaluate(model: nn.Module, data: TokenShardBlocks, args, device, dtype, precision) -> float:
    model.eval()
    n = min(data.total, args.validation_batches * args.batch_size)
    rng = np.random.default_rng(args.seed + 387)
    indices = rng.choice(data.total, size=n, replace=False)
    weighted_loss = 0.0
    seen = 0
    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            x, target = data.batch(indices[start:start+args.batch_size])
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with autocast_context(device, precision, dtype):
                _, loss = model(x, labels=target)
            size = x.shape[0]
            weighted_loss += loss.float().item() * size
            seen += size
    model.train()
    return weighted_loss / seen


def train(args, out: Path, train_manifest: dict, validation_manifest: dict) -> None:
    cfg = create_config(args)
    if args.sequence_length > cfg.max_context:
        raise ValueError("sequence-length cannot exceed max-context")
    if args.sequence_length < 2:
        raise ValueError("sequence-length must be >= 2")
    device, dtype, precision = device_and_precision(args)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    else:
        torch.set_float32_matmul_precision("high")
    data_dir = out / "tokens"
    train_blocks = TokenShardBlocks(train_manifest, data_dir, args.sequence_length)
    val_blocks = TokenShardBlocks(validation_manifest, data_dir, args.sequence_length)
    microbatches = math.ceil(train_blocks.total / args.batch_size)
    optim_steps_per_epoch = math.ceil(microbatches / args.grad_accum)
    total_steps = optim_steps_per_epoch * args.epochs
    effective = train_blocks.total * args.sequence_length
    print(f"[train] {train_blocks.total:,} sequences/epoch, {effective:,} predicted tokens/epoch; "
          f"{total_steps:,} optimizer steps over {args.epochs} passes", flush=True)
    print(f"[train] device={device}, precision={precision}, batch={args.batch_size}, grad-accum={args.grad_accum}", flush=True)

    model = SemanticFloatingLanguageModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] {cfg.depth} unique blocks, {n_params:,} trainable weights (tied vocabulary head)", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))

    checkpoint = out / "checkpoint_latest.pt"
    progress = {"epoch": 0, "next_cursor": 0, "global_step": 0,
                "best_validation_loss": float("inf"), "training_token_positions": 0}
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["config"] != asdict(cfg):
            raise ValueError("checkpoint architecture differs; use original model flags or a new output directory")
        if saved["training_size"] != train_blocks.total or saved["epochs"] != args.epochs:
            raise ValueError("checkpoint data or number of passes differs; use a new output directory")
        for field, value in (("sequence_length", args.sequence_length), ("batch_size", args.batch_size),
                             ("grad_accum", args.grad_accum), ("seed", args.seed)):
            if saved[field] != value:
                raise ValueError(f"checkpoint {field} differs; use original flags or a new output directory")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        progress = saved["progress"]
        if "rng" in saved:
            torch.set_rng_state(saved["rng"]["torch_cpu"])
            if device.type == "cuda" and saved["rng"].get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(saved["rng"]["torch_cuda"])
            random.setstate(saved["rng"]["python"])
            np.random.set_state(saved["rng"]["numpy"])
        print(f"[train] resumed: epoch {progress['epoch']+1}, step {progress['global_step']:,}", flush=True)

    if args.max_steps and progress["global_step"] >= args.max_steps:
        print(f"[train] already at step {progress['global_step']:,}, satisfying --max-steps", flush=True)
        export_backbone(out, model, cfg, progress["global_step"])
        return

    def checkpoint_now() -> None:
        save_checkpoint(checkpoint, {
            "config": asdict(cfg), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "progress": progress, "training_size": train_blocks.total,
            "epochs": args.epochs, "sequence_length": args.sequence_length,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "seed": args.seed,
            "rng": {"torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "python": random.getstate(), "numpy": np.random.get_state()},
        })

    log_file = out / "train_log.jsonl"
    start_time = time.time()
    for epoch in range(progress["epoch"], args.epochs):
        # Fully independent ordering per epoch; exact resumability without
        # storing the large permutation in a checkpoint.
        order = np.random.default_rng(args.seed + 1009 * epoch).permutation(train_blocks.total)
        cursor = progress["next_cursor"] if epoch == progress["epoch"] else 0
        if not 0 <= cursor <= train_blocks.total:
            raise RuntimeError("invalid saved batch cursor")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        accumulated_loss = 0.0
        while cursor < train_blocks.total:
            nxt = min(cursor + args.batch_size, train_blocks.total)
            batch_indices = order[cursor:nxt]
            x, labels = train_blocks.batch(batch_indices)
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_context(device, precision, dtype):
                _, loss = model(x, labels=labels)
                scaled = loss / args.grad_accum
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"non-finite training loss at step {progress['global_step']}")
            scaler.scale(scaled).backward()
            accumulated += 1
            accumulated_loss += loss.detach().float().item()
            cursor = nxt
            progress["training_token_positions"] += x.numel()
            if accumulated == args.grad_accum or cursor == train_blocks.total:
                scaler.unscale_(optimizer)
                if accumulated < args.grad_accum:
                    # Correct weighting for final partial accumulation group.
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.mul_(args.grad_accum / accumulated)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite gradient norm")
                lr = cosine_lr(progress["global_step"], total_steps, args.warmup_steps, args.lr, args.min_lr_ratio)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                progress["global_step"] += 1
                progress["epoch"] = epoch
                progress["next_cursor"] = cursor
                gstep = progress["global_step"]
                if gstep % args.log_every == 0 or gstep == 1:
                    metrics = {"epoch": epoch + 1, "step": gstep,
                               "epoch_fraction": cursor / train_blocks.total,
                               "loss": accumulated_loss / accumulated,
                               "lr": lr, "grad_norm": float(grad_norm),
                               "elapsed_s": round(time.time() - start_time, 1)}
                    print("[train] " + json.dumps(metrics), flush=True)
                    with log_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(metrics) + "\n")
                accumulated = 0
                accumulated_loss = 0.0
                if gstep % args.eval_every == 0:
                    vloss = evaluate(model, val_blocks, args, device, dtype, precision)
                    print(f"[eval] step={gstep:,}, loss={vloss:.5f}, ppl={math.exp(min(20, vloss)):.3f}", flush=True)
                    if vloss < progress["best_validation_loss"]:
                        progress["best_validation_loss"] = vloss
                        save_checkpoint(out / "checkpoint_best_model.pt", {
                            "config": asdict(cfg), "model": model.state_dict(),
                            "global_step": gstep, "validation_loss": vloss,
                        })
                if gstep % args.save_every == 0:
                    checkpoint_now()
                    export_backbone(out, model, cfg, gstep)
                if args.max_steps and gstep >= args.max_steps:
                    checkpoint_now()
                    export_backbone(out, model, cfg, gstep)
                    print("[train] stopped at --max-steps; rerun with a higher or zero cap to resume", flush=True)
                    return
        progress["epoch"] = epoch + 1
        progress["next_cursor"] = 0
        vloss = evaluate(model, val_blocks, args, device, dtype, precision)
        print(f"[epoch] {epoch+1}/{args.epochs}, validation loss={vloss:.5f}, "
              f"ppl={math.exp(min(20, vloss)):.3f}", flush=True)
        if vloss < progress["best_validation_loss"]:
            progress["best_validation_loss"] = vloss
            save_checkpoint(out / "checkpoint_best_model.pt", {
                "config": asdict(cfg), "model": model.state_dict(),
                "global_step": progress["global_step"], "validation_loss": vloss,
            })
        checkpoint_now()
        export_backbone(out, model, cfg, progress["global_step"])

    print(f"[complete] {args.epochs} passes finished; exported pretrained backbone to "
          f"{out / 'pretrained_sfc128_backbone.pt'}", flush=True)


def smoke_test() -> None:
    print("[smoke] end-to-end synthetic English training test; no web access", flush=True)
    with tempfile.TemporaryDirectory(prefix="sfc_smoke_") as work:
        root = Path(work)
        source = root / "mock.jsonl"
        syllables = ["calm", "reason", "logic", "context", "decision", "river", "bird", "number",
                     "never", "allow", "except", "under", "standing", "conditional", "however",
                     "correct", "simple", "stable", "learn", "memory", "choice", "system"]
        rng = random.Random(92)
        with source.open("w", encoding="utf-8") as f:
            for doc in range(190):
                words = [rng.choice(syllables) + (rng.choice(["", "s", "ing", "ly"])) for _ in range(28)]
                text = " ".join(words) + f". The year is {2020 + doc % 8}!\n"
                f.write(json.dumps({"text": text, "language": "en", "token_count": 80}) + "\n")
        cli = parse_args(["--output-dir", str(root / "run"), "--local-jsonl", str(source),
                          "--tokenizer-sample-tokens", "800", "--train-tokens", "4500",
                          "--validation-tokens", "850", "--shard-tokens", "1900",
                          "--min-doc-chars", "20", "--depth", "2", "--width", "24",
                          "--rank", "8", "--ren-groups", "4", "--gate-hidden", "8",
                          "--sequence-length", "16", "--max-context", "64",
                          "--batch-size", "2", "--grad-accum", "1", "--epochs", "2",
                          "--max-steps", "2", "--device", "cpu", "--precision", "fp32",
                          "--log-every", "1", "--eval-every", "1", "--validation-batches", "1",
                          "--save-every", "1"])
        cli.no_gradient_checkpointing = True
        run_pipeline(cli)
        t = AsciiConstrainedBPE.load(Path(cli.output_dir) / "tokenizer_1k.json")
        sample = "Hello, 2026! Next word.\nA=3.9"
        assert t.decode(t.encode(sample)) == sample
        assert all(len(v) == 1 or ALPHA_RE.fullmatch(v) for v in t.vocab[len(SPECIAL):])
        assert all(len(v) == 1 for v in t.vocab if v.isdigit() or v in ("!", "?", ".", " "))
        assert (Path(cli.output_dir) / "pretrained_sfc128_backbone.pt").exists()
        # Confirm that the pretrained backbone loads into the classifier.
        ckpt = torch.load(Path(cli.output_dir) / "pretrained_sfc128_backbone.pt",
                          map_location="cpu", weights_only=False)
        cls_model = SemanticFloatingClassifier(SFCConfig(**ckpt["config"]))
        incompatible = cls_model.load_state_dict(ckpt["state_dict"], strict=False)
        assert set(incompatible.missing_keys) == {"decision_head.weight", "decision_head.bias"}
        assert not incompatible.unexpected_keys
        # The LM head remains causal, not merely the classifier readout.
        pretrained_lm = SemanticFloatingLanguageModel(SFCConfig(**ckpt["config"]))
        pretrained_lm.load_state_dict(ckpt["state_dict"]); pretrained_lm.eval()
        prefix = torch.randint(8, 1000, (1, 12))
        changed = prefix.clone(); changed[:, 7:] = torch.randint(8, 1000, (1, 5))
        with torch.no_grad():
            a, _ = pretrained_lm(prefix)
            b, _ = pretrained_lm(changed)
        torch.testing.assert_close(a[:, :7], b[:, :7], rtol=2e-6, atol=2e-6)
        # Fresh invocation finishes both passes without repeating corpus construction.
        cli.max_steps = 0
        run_pipeline(cli)
        final = torch.load(Path(cli.output_dir) / "checkpoint_latest.pt",
                           map_location="cpu", weights_only=False)
        assert final["progress"]["epoch"] == 2
        print("[smoke] PASS: tokenizer invariants, corpus, two full epochs, "
              "checkpoints, classifier transfer, resume")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train 1K-tokenizer SFC-128 on ~200M FineWeb-Edu tokens (two epochs)")
    p.add_argument("--output-dir", type=str, default="./sfc_fineweb_run")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--dataset-revision", default=None,
                   help="Optional pinned Hugging Face dataset revision for repeatability")
    p.add_argument("--local-jsonl", default=None, help="Offline input: one {text,token_count} JSON object per line")
    p.add_argument("--shuffle-buffer", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--tokenizer-max-words", type=int, default=200_000)
    p.add_argument("--tokenizer-only", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--tokenizer-sample-tokens", type=int, default=5_000_000,
                   help="Reference FineWeb token_count; 5M is approximate, before custom BPE")
    p.add_argument("--train-tokens", type=int, default=200_000_000,
                   help="Actual custom-BPE train token target; whole documents can overshoot slightly")
    p.add_argument("--validation-tokens", type=int, default=2_000_000,
                   help="Additional held-out custom-BPE tokens (no training leakage)")
    p.add_argument("--shard-tokens", type=int, default=5_000_000)
    p.add_argument("--min-doc-chars", type=int, default=80)
    p.add_argument("--min-language-score", type=float, default=0.7)
    p.add_argument("--max-nonascii-fraction", type=float, default=0.05)
    p.add_argument("--depth", type=int, default=128)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--rank", type=int, default=24)
    p.add_argument("--ren-groups", type=int, default=8)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--dilations", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--gate-hidden", type=int, default=32)
    p.add_argument("--no-floating", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--embedding-dropout", type=float, default=0.0)
    p.add_argument("--residual-gain-init", type=float, default=0.5)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--num-decisions", type=int, default=32,
                   help="Reserved classifier-head class count for downstream transfer")
    p.add_argument("--max-context", type=int, default=2048)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--validation-batches", type=int, default=32)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=0,
                   help="Optional pause after this optimizer step; set 0 for full run")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    a = p.parse_args(argv)
    for key in ("tokenizer_sample_tokens", "train_tokens", "validation_tokens", "shard_tokens",
                "sequence_length", "batch_size", "grad_accum", "epochs", "save_every", "eval_every",
                "log_every", "validation_batches"):
        if getattr(a, key) < 1:
            p.error(f"--{key.replace('_','-')} must be positive")
    if a.train_tokens < a.sequence_length + 1 or a.validation_tokens < a.sequence_length + 1:
        p.error("dataset targets must exceed one sequence length")
    if a.tokenizer_max_words < 100:
        p.error("tokenizer-max-words must be >= 100")
    if a.max_steps < 0:
        p.error("max-steps cannot be negative")
    return a


def run_pipeline(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(args, out)
    if args.tokenizer_only:
        return
    train_manifest, validation_manifest = prepare_corpus(args, out, tokenizer)
    if args.prepare_only:
        return
    train(args, out, train_manifest, validation_manifest)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoke_test:
        smoke_test()
    else:
        run_pipeline(arguments)
