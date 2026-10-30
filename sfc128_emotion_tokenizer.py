#!/usr/bin/env python3
"""Inference-only implementation of the original 1K ASCII-constrained BPE.

This deliberately reuses the *identical* merge order, segment regex, and
per-word BPE semantics of train_sfc128_fineweb_edu.AsciiConstrainedBPE.
It NEVER retrains or changes the vocabulary.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re

SPECIAL = ("<PAD>", "<BOS>", "<EOS>", "<DECIDE>", "<SEP>", "<UNK>", "<MASK>", "<RESERVED>")
ASCII_CHARS = tuple(chr(i) for i in range(32, 127)) + ("\n", "\t")
ALLOWED = set(ASCII_CHARS)
SEGMENT_RE = re.compile(r"[A-Za-z]+|[^A-Za-z]")
ALPHA_RE = re.compile(r"[A-Za-z]+\Z")
PUNCT_REMAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ",
    "\u2026": "...", "\u2022": "*", "\u200b": "", "\ufeff": "",
})


def normalize_english_ascii(text: str) -> str:
    """Map conventional Unicode typography, but reject all remaining non-ASCII.

    Training and deployment use the SAME normalization. No unsupported
    character is silently dropped or replaced (except the explicit map).
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = text.translate(PUNCT_REMAP).replace("\r\n", "\n").replace("\r", "\n")
    unexpected = sorted(set(text) - ALLOWED)
    if unexpected:
        codes = ", ".join("U+%04X" % ord(ch) for ch in unexpected[:6])
        raise ValueError("unsupported non-ASCII or control characters: " + codes)
    return text


class SFCAsciiTokenizer:
    def __init__(self, vocab: list[str], merges: list[list[str]] | list[tuple[str, str]]) -> None:
        self.vocab = list(vocab)
        self.ids = {piece: idx for idx, piece in enumerate(self.vocab)}
        self.merges = [tuple(pair) for pair in merges]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        if len(self.vocab) != 1000 or len(self.ids) != 1000 or tuple(self.vocab[:8]) != SPECIAL:
            raise ValueError("expected exactly 1,000 original IDs with 8 fixed control tokens")
        if not all(ch in self.ids for ch in ASCII_CHARS):
            raise ValueError("missing single-character ASCII fallback tokens")
        if any(len(tok) > 1 and not ALPHA_RE.fullmatch(tok) for tok in self.vocab[8:]):
            raise ValueError("forbidden non-alphabetic multi-character token")
        if any(len(tok) == 1 and tok not in ALLOWED for tok in self.vocab[8:]):
            raise ValueError("forbidden single-character token")
        if any(len(pair) != 2 or not ALPHA_RE.fullmatch("".join(pair)) for pair in self.merges):
            raise ValueError("invalid BPE merge")

    @classmethod
    def load(cls, path: str | Path) -> "SFCAsciiTokenizer":
        with Path(path).open("r", encoding="utf-8") as file:
            state = json.load(file)
        if state.get("kind") != "ascii_constrained_bpe_v1":
            raise ValueError("not an original SFC ASCII BPE tokenizer")
        return cls(state["vocab"], state["merges"])

    @lru_cache(maxsize=200_000)
    def _encode_word(self, word: str) -> tuple[int, ...]:
        pieces = list(word)
        while len(pieces) > 1:
            rank, chosen = None, None
            for pair in zip(pieces, pieces[1:]):
                candidate_rank = self.ranks.get(pair)
                if candidate_rank is not None and (rank is None or candidate_rank < rank):
                    rank, chosen = candidate_rank, pair
            if chosen is None:
                break
            merged = []
            i = 0
            while i < len(pieces):
                if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == chosen:
                    merged.append(pieces[i] + pieces[i + 1])
                    i += 2
                else:
                    merged.append(pieces[i])
                    i += 1
            pieces = merged
        return tuple(self.ids[part] for part in pieces)

    def encode(self, text: str, append_eos: bool = False) -> list[int]:
        if any(ch not in ALLOWED for ch in text):
            raise ValueError("only printable ASCII, tab, and newline are supported")
        output: list[int] = []
        for segment in SEGMENT_RE.findall(text):
            if ALPHA_RE.fullmatch(segment):
                output.extend(self._encode_word(segment))
            else:
                output.append(self.ids[segment])
        if append_eos:
            output.append(self.ids["<EOS>"])
        return output

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return "".join(self.vocab[i] for i in ids if not (skip_special and i < 8))

    @property
    def pad_id(self) -> int:
        return self.ids["<PAD>"]

    @property
    def decision_id(self) -> int:
        return self.ids["<DECIDE>"]
