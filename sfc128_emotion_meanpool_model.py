#!/usr/bin/env python3
"""SFC-128 Emotion v2: full-token masked-mean classification readout.

The 128 REN-T blocks and all pretrained parameter names are unchanged. The
only architectural change is the readout: instead of appending an untrained
DECIDE token and classifying its state, the model pools all *real* final token
states. This module is compatible with the original Stage 1 backbone export.

A six-class emotion head is initialized separately, and all weights are
fine-tuned normally. Padding is excluded from pooling; padded tokens cannot
leak into real-token states because the temporal convolutions are causal.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from semantic_floating_ren_t_128 import (
    SFCConfig,
    SemanticDecisionOutput,
    SemanticFloatingClassifier,
)


class SemanticFloatingMeanPoolClassifier(SemanticFloatingClassifier):
    """Parameter-compatible SFC-128 with an all-token masked mean readout."""

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        *,
        return_decision_states: bool = False,
    ) -> SemanticDecisionOutput:
        mask, lengths = self._check_ids_and_mask(input_ids, attention_mask)
        if torch.any(lengths == 0):
            raise ValueError("masked-mean classification requires nonempty text")
        if input_ids.size(1) > self.cfg.max_context:
            raise ValueError("input sequence exceeds max_context")

        states = self._run_core(input_ids, mask)
        weights = mask.unsqueeze(-1).to(dtype=states.dtype)
        # Accumulate in float32 for stability with bf16/fp16 mixed precision.
        summed = (states.float() * weights.float()).sum(dim=1)
        pooled = (summed / lengths.unsqueeze(-1).float()).to(dtype=states.dtype)
        logits = self.decision_head(pooled)

        loss = None
        if labels is not None:
            if labels.ndim != 1 or labels.shape[0] != input_ids.shape[0]:
                raise ValueError("labels must have shape (batch,)")
            loss = F.cross_entropy(logits.float(), labels.long(),
                                   label_smoothing=self.cfg.label_smoothing)
        return SemanticDecisionOutput(
            logits=logits,
            loss=loss,
            decision_states=pooled if return_decision_states else None,
        )


def test_mean_pool() -> None:
    torch.manual_seed(43)
    cfg = SFCConfig(depth=3, hidden_size=24, ren_groups=4,
                    bottleneck_size=8, floating_hidden=8,
                    vocab_size=1000, pad_token_id=0, decision_token_id=3,
                    num_classes=6, max_context=64,
                    gradient_checkpointing=False)
    model = SemanticFloatingMeanPoolClassifier(cfg).eval()
    x = torch.tensor([[11, 12, 13, 0, 0], [18, 19, 20, 21, 22]], dtype=torch.long)
    with torch.no_grad():
        batched = model(x).logits
        single = model(torch.tensor([[11, 12, 13]])).logits
        assert torch.allclose(batched[:1], single, atol=1e-5), "pooling used padding"
        # Replacing right padding must not change the first prediction.
        suffix = torch.tensor([[11, 12, 13, 0, 0, 0, 0]])
        assert torch.allclose(model(suffix).logits, single, atol=1e-5)
        assert model(x, return_decision_states=True).decision_states.shape == (2, 24)
    model.train()
    loss = model(x, labels=torch.tensor([2, 4])).loss
    loss.backward()
    assert all(p.grad is not None for p in model.parameters()), "a parameter received no gradient"
    print("[model-test] masked-mean readout, padding isolation and full autograd PASS")


if __name__ == "__main__":
    test_mean_pool()
