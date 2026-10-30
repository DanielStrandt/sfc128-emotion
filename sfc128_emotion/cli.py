"""Command-line inference, verification, and latency benchmarking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .paths import resolve_model_dir


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", default=None, help="Exported SFC128_Emotion folder")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--eager", action="store_true",
                        help="Disable automatic CUDA Graph inference and use eager PyTorch")


def create_parser():
    parser = argparse.ArgumentParser(prog="sfc128-emotion", description="SFC-128 Emotion v2 deployment")
    sub = parser.add_subparsers(dest="command", required=True)
    prediction = sub.add_parser("predict", help="Predict emotions from one text or UTF-8 line file")
    add_runtime_args(prediction)
    group = prediction.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str)
    group.add_argument("--input-file", type=Path, help="One text per line (UTF-8)")
    prediction.add_argument("--output-file", type=Path, help="Write JSONL rather than stdout")
    prediction.add_argument("--top-k", type=int, default=3)
    prediction.add_argument("--batch-size", type=int, default=16)

    interactive = sub.add_parser("interactive", help="Type text interactively")
    add_runtime_args(interactive)
    interactive.add_argument("--top-k", type=int, default=3)

    check = sub.add_parser("verify", help="Verify exported checkpoint, vocabulary and configuration")
    check.add_argument("--model-dir", default=None)
    check.add_argument("--allow-nonstandard", action="store_true", help="For synthetic tests only")
    check.add_argument("--run-inference", action="store_true", help="Also execute a real forward pass")

    benchmark = sub.add_parser("benchmark", help="Benchmark inference (CUDA Graphs auto-enable on CUDA FP32)")
    add_runtime_args(benchmark)
    benchmark.add_argument("--warmup", type=int, default=5)
    benchmark.add_argument("--iterations", type=int, default=30)
    benchmark.add_argument("--batch-size", type=int, default=1)
    benchmark.add_argument("--text", default="I feel grateful and happy that everything worked out today.")
    benchmark.add_argument("--output-file", type=Path, default=None)
    return parser


def _write_records(output: Path | None, records: list[dict]) -> None:
    if output is None:
        for record in records:
            print(json.dumps(record, ensure_ascii=False))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as sink:
            for record in records:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[saved] {output}", file=sys.stderr)


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    model_dir = resolve_model_dir(args.model_dir)
    if args.command == "verify":
        from .verify import verify_export
        result = verify_export(model_dir, require_standard=not args.allow_nonstandard,
                               run_inference=args.run_inference)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "benchmark":
        from .benchmark import benchmark_inference
        report = benchmark_inference(model_dir, args.text, args.batch_size,
                                     args.device, args.precision, args.warmup, args.iterations,
                                     cuda_graphs=False if args.eager else None)
        print(json.dumps(report, indent=2))
        if args.output_file:
            args.output_file.parent.mkdir(parents=True, exist_ok=True)
            args.output_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 0

    from .runtime import EmotionEngine
    engine = EmotionEngine(model_dir, device=args.device, precision=args.precision,
                           cuda_graphs=False if args.eager else None)
    if args.command == "interactive":
        print("SFC-128 Emotion v2. Enter a sentence, or 'exit' to quit.")
        while True:
            try:
                line = input("emotion> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line.strip().lower() in ("exit", "quit"):
                break
            try:
                print(json.dumps(engine.predict(line, top_k=args.top_k), indent=2))
            except (ValueError, TypeError) as error:
                print(f"Invalid text: {error}", file=sys.stderr)
        return 0
    if args.input_file:
        texts = args.input_file.read_text(encoding="utf-8").splitlines()
        if not texts or any(not line.strip() for line in texts):
            raise ValueError("input file cannot be empty or contain blank lines")
    else:
        texts = [args.text]
    predictions = engine.predict_many(texts, top_k=args.top_k, batch_size=args.batch_size)
    _write_records(args.output_file, [dict(text=text, **prediction)
                                     for text, prediction in zip(texts, predictions)])
    return 0
