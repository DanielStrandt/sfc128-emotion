# Research reproducibility

The original complete Stage 2 v2 trainer and held-out evaluator live at the repository root because their original import/export paths assume the architecture and tokenizer modules are alongside them. They have not been rewritten or optimized:

- `train_sfc128_emotion_v2.py`: download and fine-tune the foundation, select best validation checkpoint and export model.
- `evaluate_sfc128_emotion_v2_final.py`: original held-out DAIR.AI benchmark evaluator, with the source run's deduplicated test cache.

Neither script is required for CLI/API inference. Training requires the **original FineWeb foundation checkpoint and tokenizer**; the final benchmark requires the original Stage 2 run's prepared test cache (or the documented reconstruction path). These original local files are *not* included in the ZIP.

The final reported held-out results came from best checkpoint step 24,000. See `MODEL_CARD.md` and `docs/EVALUATION.md`.

`research/pretraining/train_sfc128_fineweb_edu.py` and its original README document the earlier FineWeb-Edu foundation training pipeline. It is provided for research provenance, **not** used by the deployed emotion API. The original pretraining corpus, tokenizer samples and checkpoints are not included. To run this historical script, install the research dependencies and ensure the repository package is installed (`python -m pip install -e .`), so imports resolve.
