from __future__ import annotations
import pytest

from sfc128_emotion_tokenizer import SFCAsciiTokenizer, normalize_english_ascii


def test_existing_tokenizer_keeps_whitespace_and_case(synthetic_export):
    tokenizer = SFCAsciiTokenizer.load(synthetic_export / "tokenizer_1k.json")
    phrase = "I feel HAPPY!\n"
    ids = tokenizer.encode(phrase)
    assert tokenizer.decode(ids) == phrase
    assert len(tokenizer.vocab) == 1000
    assert tokenizer.pad_id == 0
    assert tokenizer.decision_id == 3
    assert len(tokenizer.encode("ab")) == 1  # Synthetic ranked BPE merge.
    assert tokenizer.decode(tokenizer.encode("ab")) == "ab"


def test_normalization_and_ascii_contract():
    assert normalize_english_ascii("I’m glad—really!") == "I'm glad-really!"
    with pytest.raises(ValueError, match="non-ASCII"):
        normalize_english_ascii("I love 🐈")
