#!/usr/bin/env python3
"""Load the exported SFC-128 Emotion specialist and predict six-class emotion.

  python sfc128_emotion_inference.py --model-dir ./sfc_emotion_v2_run/SFC128_Emotion --text "I feel wonderful!"
  python sfc128_emotion_inference.py --model-dir ./sfc_emotion_v2_run/SFC128_Emotion --interactive

No Hugging Face download, LM head, or Stage 1 trainer is used at inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from semantic_floating_ren_t_128 import SFCConfig
from sfc128_emotion_meanpool_model import SemanticFloatingMeanPoolClassifier
from sfc128_emotion_tokenizer import SFCAsciiTokenizer, normalize_english_ascii


class EmotionPredictor:
    def __init__(self, model_dir: str | Path, device: str = "auto") -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        model_file = self.model_dir / "model.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"no SFC emotion checkpoint at {model_file}")
        try:
            artifact = torch.load(model_file, map_location="cpu", weights_only=True)
        except TypeError:
            artifact = torch.load(model_file, map_location="cpu")
        if artifact.get("architecture") != "SemanticFloatingMeanPoolClassifier":
            raise ValueError("unsupported architecture")
        if artifact.get("readout") != "masked_mean":
            raise ValueError("this inference script requires a masked-mean v2 checkpoint")
        self.cfg = SFCConfig(**artifact["config"])
        self.labels = artifact["labels"]
        if len(self.labels) != self.cfg.num_classes or self.cfg.num_classes != 6:
            raise ValueError("the supplied model is not a six-class Emotion specialist")
        self.tokenizer = SFCAsciiTokenizer.load(self.model_dir / "tokenizer_1k.json")
        if self.cfg.vocab_size != len(self.tokenizer.vocab):
            raise ValueError("tokenizer/model vocabulary mismatch")
        metadata = self.model_dir / "config.json"
        if metadata.exists():
            self.max_text_tokens = int(json.loads(metadata.read_text(encoding="utf-8"))
                                       .get("max_text_tokens", min(128, self.cfg.max_context - 1)))
        else:
            self.max_text_tokens = min(128, self.cfg.max_context - 1)
        self.temperature = float(artifact.get("temperature", 1.0))
        if self.temperature <= 0:
            raise ValueError("invalid calibration temperature")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = SemanticFloatingMeanPoolClassifier(self.cfg)
        self.model.load_state_dict(artifact["state_dict"], strict=True)
        self.model.to(self.device).eval()

    def predict_many(self, texts: list[str], top_k: int = 3) -> list[dict]:
        if not texts:
            return []
        if not 1 <= top_k <= len(self.labels):
            raise ValueError("top_k must be from 1 to 6")
        encodings = []
        truncated = []
        for text in texts:
            clean = normalize_english_ascii(text).strip()
            if not clean:
                raise ValueError("cannot classify empty text")
            ids = self.tokenizer.encode(clean)
            truncated.append(len(ids) > self.max_text_tokens)
            ids = ids[:self.max_text_tokens]
            if not ids:
                raise ValueError("no valid tokens in text")
            encodings.append(ids)
        length = max(map(len, encodings))
        batch = torch.full((len(encodings), length), self.cfg.pad_token_id, dtype=torch.long)
        for row, ids in enumerate(encodings):
            batch[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(batch.to(self.device)).logits.float()
            probabilities = torch.softmax(logits / self.temperature, dim=-1).cpu()
        predictions = []
        for index, prob in enumerate(probabilities):
            top = torch.argsort(prob, descending=True).tolist()
            predictions.append({
                "label": self.labels[top[0]],
                "confidence": float(prob[top[0]]),
                "top_predictions": [{"label": self.labels[j], "probability": float(prob[j])}
                                    for j in top[:top_k]],
                "token_count": len(encodings[index]), "truncated": truncated[index],
            })
        return predictions

    def predict(self, text: str, top_k: int = 3) -> dict:
        return self.predict_many([text], top_k=top_k)[0]


def main():
    parser = argparse.ArgumentParser(description="SFC-128 Emotion v2: masked-mean standalone inference")
    parser.add_argument("--model-dir", default="./sfc_emotion_v2_run/SFC128_Emotion")
    parser.add_argument("--text", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    predictor = EmotionPredictor(args.model_dir, device=args.device)
    if args.text is not None:
        print(json.dumps(predictor.predict(args.text, top_k=args.top_k), indent=2))
    elif args.interactive:
        print("SFC-128 Emotion (type 'exit' to quit)")
        while True:
            try:
                sentence = input("text> ")
            except (EOFError, KeyboardInterrupt):
                break
            if sentence.strip().lower() == "exit":
                break
            try:
                print(json.dumps(predictor.predict(sentence, top_k=args.top_k), indent=2))
            except (ValueError, TypeError) as ex:
                print("Input rejected:", ex)
    else:
        parser.error("provide --text or --interactive")


if __name__ == "__main__":
    main()
