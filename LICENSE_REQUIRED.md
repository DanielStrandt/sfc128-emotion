# Third-party data and provenance review

The project source code and included SFC-128 Emotion v2 export are licensed under Apache-2.0; see [`LICENSE`](LICENSE). That license applies to this repository's contributions and does not grant rights to third-party datasets, source corpora, or other separately licensed material.

The model was fine-tuned on `dair-ai/emotion` and its foundation was pretrained using FineWeb-Edu. Review the current terms, attribution requirements, and any applicable restrictions for those materials before redistributing datasets or making claims about downstream rights. The datasets themselves are not included in this repository. The inference export is included.

The reported test result is an exploratory development result because the held-out test was inspected before training finished and repeated after completion. See the evaluation-protocol disclosure in [`MODEL_CARD.md`](MODEL_CARD.md). Nothing in the sample Docker/API deployment authorizes high-stakes use.
