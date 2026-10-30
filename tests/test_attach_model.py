from __future__ import annotations
import pytest

from .conftest import create_synthetic_export
from tools.attach_model import attach


def test_attach_real_architecture_and_hashes(tmp_path):
    # Random weights but EXACT real 128x96 geometry and parameter count.
    source = create_synthetic_export(tmp_path / "original", depth=128, width=96)
    dest = tmp_path / "repo_model"
    result = attach(source, dest)
    assert result["depth"] == 128
    assert result["parameters"] == 792454
    assert (dest / "model.pt").is_file()
    assert (source / "model.pt").is_file()  # Never moves/deletes source.
    with pytest.raises(FileExistsError):
        attach(source, dest)
