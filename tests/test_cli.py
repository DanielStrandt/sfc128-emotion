from __future__ import annotations
import json

from sfc128_emotion.cli import main


def test_cli_predict_and_verify(synthetic_export, capsys):
    result = main(["verify", "--model-dir", str(synthetic_export),
                   "--allow-nonstandard", "--run-inference"])
    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    result = main(["predict", "--model-dir", str(synthetic_export),
                   "--text", "Today is a wonderful day"])
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["text"].startswith("Today")
    assert payload["label"] in report["labels"]
