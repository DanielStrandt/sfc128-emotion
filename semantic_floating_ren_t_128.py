"""Semantic Floating Core 128 (SFC-128): a compact causal decision model.

A complete PyTorch classifier for *tokenized* text, not a text generator.

Default architecture
--------------------
* 128 INDEPENDENT learned blocks (no weight sharing).
* Width 96, REN-T with 8 channel groups and 75% RMS / 25% L1 scaling.
* Per-token, sample-local REN-T statistics; no batch/time reductions.
* Causal depthwise convolutions with dilations (1, 2, 4, 8, 16, 32) repeated.
* Low-rank channel mixing, bottleneck 24, with SiLU nonlinearity.
* Optional per-token, per-group bounded Floating modulation of the branch.
* Standard PyTorch autograd; no B-only / surrogate backward operation.
* Pre-norm identity-residual pathway with 1/sqrt(depth) scaling.
* An automatically inserted final decision token and classification head.
* Right-padded input IDs and optional activation checkpointing.

This module does NOT include a tokenizer or pretrained weights. To obtain meaningful
semantic decisions, train it on labeled examples using your own tokenizer whose
vocabulary size agrees with SFCConfig.vocab_size.

Causality note: REN-T and the Floating gate reduce across channels *at each token*
only. Causal convolutions only access the present and earlier tokens. The final
readout may use all preceding tokens via the causal stack, within its theoretical
receptive field; this does not guarantee effective long-context recall.

Default theoretical receptive field (128 blocks, kernel=3, cycle 1..32): 2653 tokens.
Default maximum input length INCLUDING the appended decision token: 2048 tokens.

Usage
-----
    from semantic_floating_ren_t_128 import SFCConfig, SemanticFloatingClassifier
    cfg = SFCConfig()
    model = SemanticFloatingClassifier(cfg)
    input_ids = torch.randint(1, cfg.vocab_size - 1, (2, 80))
    labels = torch.tensor([3, 7])
    out = model(input_ids, labels=labels)
    print(out.logits.shape, out.loss)

Quick verification from a terminal:
    python semantic_floating_ren_t_128.py

Optional 128-layer forward check:
    python semantic_floating_ren_t_128.py --full-forward
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class SFCConfig:
    # Input and classification
    vocab_size: int = 4096
    pad_token_id: int = 0
    decision_token_id: int = 4095
    num_classes: int = 32
    max_context: int = 2048    # Includes the final decision token.

    # Independent-layer semantic core
    depth: int = 128
    hidden_size: int = 96
    bottleneck_size: int = 24
    kernel_size: int = 3
    dilation_cycle: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)

    # REN-T: alpha is the RMS contribution, as in the REN-T manuscript.
    ren_groups: int = 8
    ren_alpha: float = 0.75
    ren_eps: float = 1e-6
    ren_affine: bool = True

    # Per-token groupwise Floating gate
    use_floating_gate: bool = True
    floating_hidden: int = 32
    floating_gain_deviation: float = 0.03  # Gains in [0.97, 1.03]
    floating_bias_bound: float = 0.01

    # Stability and training
    residual_gain_init: float = 0.5  # Total init = 0.5 / sqrt(depth)
    residual_scale_mode: str = "sqrt"  # "sqrt", "depth", "none"
    dropout: float = 0.0
    embedding_dropout: float = 0.0
    gradient_checkpointing: bool = False
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size < 3:
            raise ValueError("vocab_size must be >= 3")
        if not (0 <= self.pad_token_id < self.vocab_size):
            raise ValueError("pad_token_id must be in the vocabulary")
        if not (0 <= self.decision_token_id < self.vocab_size):
            raise ValueError("decision_token_id must be in the vocabulary")
        if self.pad_token_id == self.decision_token_id:
            raise ValueError("pad_token_id and decision_token_id must differ")
        if self.num_classes < 2 or self.max_context < 2:
            raise ValueError("num_classes and max_context must be >= 2")
        if self.depth < 1 or self.hidden_size < 1 or self.bottleneck_size < 1:
            raise ValueError("depth, hidden_size and bottleneck_size must be positive")
        if self.kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        if not self.dilation_cycle or any(d < 1 for d in self.dilation_cycle):
            raise ValueError("dilation_cycle must contain positive integers")
        if self.ren_groups < 1 or self.hidden_size % self.ren_groups:
            raise ValueError("hidden_size must be divisible by ren_groups")
        if not (0.0 <= self.ren_alpha <= 1.0):
            raise ValueError("ren_alpha must be in [0, 1]")
        if not (math.isfinite(self.ren_eps) and self.ren_eps > 0.0):
            raise ValueError("ren_eps must be positive and finite")
        if self.floating_hidden < 1:
            raise ValueError("floating_hidden must be positive")
        if not (0.0 <= self.floating_gain_deviation < 1.0):
            raise ValueError("floating_gain_deviation must be in [0, 1)")
        if self.floating_bias_bound < 0.0:
            raise ValueError("floating_bias_bound must be nonnegative")
        if self.residual_scale_mode not in {"sqrt", "depth", "none"}:
            raise ValueError("residual_scale_mode must be 'sqrt', 'depth', or 'none'")
        if not math.isfinite(self.residual_gain_init):
            raise ValueError("residual_gain_init must be finite")
        if not (0.0 <= self.dropout < 1.0 and 0.0 <= self.embedding_dropout < 1.0):
            raise ValueError("dropout values must be in [0, 1)")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")


class RelativeEnergyNormT(nn.Module):
    """Token-local non-centering REN-T for (B, T, C).

    Computes mean(abs(x)) and RMS(x) over C/G channels at each (B,T,G),
    not across batch or time. Scale gradients are NOT detached.
    Statistics are accumulated in float32 for mixed-precision stability.
    """

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        alpha: float = 0.75,
        eps: float = 1e-6,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if channels < 1 or groups < 1 or channels % groups:
            raise ValueError("channels must be positive and divisible by groups")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]; it weights RMS")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.channels = int(channels)
        self.groups = int(groups)
        self.alpha = float(alpha)
        self.eps = float(eps)
        if affine:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(f"REN-T requires (B,T,{self.channels}), got {tuple(x.shape)}")
        b, t, c = x.shape
        xg = x.reshape(b, t, self.groups, c // self.groups).float()
        mean_abs = xg.abs().mean(dim=-1, keepdim=True)
        rms = (xg.square().mean(dim=-1, keepdim=True) + self.eps).sqrt()
        scale = (1.0 - self.alpha) * mean_abs + self.alpha * rms + self.eps
        y = (xg / scale).reshape(b, t, c).to(dtype=x.dtype)
        if self.weight is not None:
            y = y * self.weight.to(dtype=y.dtype) + self.bias.to(dtype=y.dtype)
        return y


class TokenFloatingGate(nn.Module):
    """Sample-local gate conditioned on REN-T coordinates.

    Every (batch, token, group) independently gets a bounded gain and bias.
    Importantly, descriptors NEVER reduce over tokens or batch items.

    Input `coordinates` are REN-T-normalized; `signal` is the unnormalized
    output of the convolution-plus-mixer branch (REN-C-like separation).
    """

    def __init__(
        self,
        channels: int,
        groups: int,
        hidden: int = 32,
        gain_deviation: float = 0.03,
        bias_bound: float = 0.01,
    ) -> None:
        super().__init__()
        if channels < 1 or groups < 1 or channels % groups:
            raise ValueError("channels must be divisible by positive groups")
        if hidden < 1 or not (0.0 <= gain_deviation < 1.0) or bias_bound < 0.0:
            raise ValueError("invalid gate bounds or hidden size")
        self.channels = channels
        self.groups = groups
        self.gain_deviation = gain_deviation
        self.bias_bound = bias_bound

        # Descriptors: signed mean, L1, RMS, bounded uncentered third moment.
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)  # Exactly identity (gain=1, bias=0).

    def forward(self, coordinates: Tensor, signal: Tensor) -> Tensor:
        if coordinates.shape != signal.shape:
            raise ValueError("coordinates and signal must have identical shapes")
        b, t, c = signal.shape
        if c != self.channels:
            raise ValueError(f"expected C={self.channels}, got {c}")
        cg = c // self.groups
        xg = coordinates.reshape(b, t, self.groups, cg).float()
        descriptors = torch.stack(
            (
                xg.mean(dim=-1),
                xg.abs().mean(dim=-1),
                (xg.square().mean(dim=-1) + 1e-6).sqrt(),
                xg.pow(3).mean(dim=-1).tanh(),
            ),
            dim=-1,
        )
        output = self.net(descriptors)
        gain = 1.0 + self.gain_deviation * output[..., 0].tanh()
        bias = self.bias_bound * output[..., 1].tanh()
        signal_groups = signal.reshape(b, t, self.groups, cg)
        return (
            signal_groups * gain.to(signal.dtype).unsqueeze(-1)
            + bias.to(signal.dtype).unsqueeze(-1)
        ).reshape(b, t, c)


class CausalDepthwiseConv1d(nn.Module):
    """Causal depthwise temporal convolution; input/output (B,T,C)."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        if channels < 1 or kernel_size < 1 or dilation < 1:
            raise ValueError("channels, kernel_size, dilation must be positive")
        self.left_pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=False,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        if self.left_pad:
            x = F.pad(x, (self.left_pad, 0))
        return self.conv(x).transpose(1, 2)


class LowRankChannelMixer(nn.Module):
    """Tokenwise cross-channel mixing; rank controls per-block parameter cost."""

    def __init__(self, channels: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(channels, rank)
        self.activation = nn.SiLU()
        self.up = nn.Linear(rank, channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.activation(self.down(x)))


class SemanticFloatingBlock(nn.Module):
    """Unique-weight REN-T / causal-conv / low-rank / Floating residual block."""

    def __init__(self, cfg: SFCConfig, dilation: int) -> None:
        super().__init__()
        self.ren = RelativeEnergyNormT(
            cfg.hidden_size, cfg.ren_groups, cfg.ren_alpha, cfg.ren_eps, cfg.ren_affine
        )
        self.temporal = CausalDepthwiseConv1d(cfg.hidden_size, cfg.kernel_size, dilation)
        self.channel_mixer = LowRankChannelMixer(cfg.hidden_size, cfg.bottleneck_size)
        self.gate: Optional[TokenFloatingGate] = (
            TokenFloatingGate(
                cfg.hidden_size,
                cfg.ren_groups,
                cfg.floating_hidden,
                cfg.floating_gain_deviation,
                cfg.floating_bias_bound,
            )
            if cfg.use_floating_gate
            else None
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.residual_gain = nn.Parameter(torch.tensor(cfg.residual_gain_init, dtype=torch.float32))
        self.depth_scale = {
            "sqrt": cfg.depth ** -0.5,
            "depth": 1.0 / cfg.depth,
            "none": 1.0,
        }[cfg.residual_scale_mode]

    def forward(self, x: Tensor) -> Tensor:
        coordinates = self.ren(x)
        branch = self.channel_mixer(self.temporal(coordinates))
        if self.gate is not None:
            branch = self.gate(coordinates, branch)
        return x + self.depth_scale * self.residual_gain * self.dropout(branch)


@dataclass
class SemanticDecisionOutput:
    logits: Tensor                    # (B, num_classes)
    loss: Optional[Tensor] = None     # Cross-entropy when labels are provided.
    decision_states: Optional[Tensor] = None  # (B, hidden_size), if requested.


class SemanticFloatingClassifier(nn.Module):
    """128-layer, narrow, causal semantic decision classifier.

    Vocabulary is supplied externally by a tokenizer. By default, forward()
    inserts a decision token immediately after each example's last real token,
    even when examples have different lengths in a right-padded batch.

    `attention_mask` is a (B,T) tensor of 1s for tokens and 0s for RIGHT
    padding. Do not use left padding or interior masked positions. `labels`
    are integer class indices [0, num_classes). `return_decision_states=True`
    exposes the final representation for analysis.
    """

    def __init__(self, cfg: Optional[SFCConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else SFCConfig()
        cfg = self.cfg
        self.token_embeddings = nn.Embedding(
            cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id
        )
        self.embedding_dropout = nn.Dropout(cfg.embedding_dropout)
        self.layers = nn.ModuleList(
            [
                SemanticFloatingBlock(cfg, cfg.dilation_cycle[i % len(cfg.dilation_cycle)])
                for i in range(cfg.depth)
            ]
        )
        self.final_norm = RelativeEnergyNormT(
            cfg.hidden_size, cfg.ren_groups, cfg.ren_alpha, cfg.ren_eps, affine=True
        )
        self.decision_head = nn.Linear(cfg.hidden_size, cfg.num_classes)

        # Give the embedding a modest initial scale, not a huge raw variance.
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embeddings.weight[cfg.pad_token_id].zero_()

    @property
    def receptive_field(self) -> int:
        """The maximum number of input tokens visible to a single output token."""
        cfg = self.cfg
        return 1 + (cfg.kernel_size - 1) * sum(
            cfg.dilation_cycle[i % len(cfg.dilation_cycle)] for i in range(cfg.depth)
        )

    def count_parameters(self, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in parameters if p.requires_grad)
        return sum(p.numel() for p in parameters)

    def parameter_breakdown(self) -> dict[str, int]:
        return {
            "embedding": sum(p.numel() for p in self.token_embeddings.parameters()),
            "micro_blocks": sum(p.numel() for p in self.layers.parameters()),
            "final_norm": sum(p.numel() for p in self.final_norm.parameters()),
            "decision_head": sum(p.numel() for p in self.decision_head.parameters()),
        }

    def _check_ids_and_mask(
        self, input_ids: Tensor, attention_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (B,T)")
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise TypeError("input_ids must contain integer token IDs")
        b, t = input_ids.shape
        if b < 1 or t < 1:
            raise ValueError("input_ids batch and sequence dimensions must be nonempty")
        if attention_mask is None:
            mask = input_ids.ne(self.cfg.pad_token_id)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids shape")
            mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

        # Right-padding correctness is needed to position appended decision
        # tokens. Detect invalid masks instead of silently making bad readouts.
        lengths = mask.long().sum(dim=1)
        positions = torch.arange(t, device=input_ids.device).unsqueeze(0)
        right_padded = positions < lengths.unsqueeze(1)
        if not torch.equal(mask, right_padded):
            raise ValueError("attention_mask must be contiguous 1s followed by right-padding 0s")
        return mask, lengths

    def _prepare_tokens(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
        append_decision_token: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        mask, lengths = self._check_ids_and_mask(input_ids, attention_mask)
        b, t = input_ids.shape
        if t + int(append_decision_token) > self.cfg.max_context:
            raise ValueError(
                f"sequence length {t} + appended token exceeds max_context={self.cfg.max_context}"
            )
        if append_decision_token:
            # Place the decision token at the first pad position *for each row*.
            # Appending at the far right of a padded batch would incorrectly add
            # spurious padding time steps between the text and the decision token.
            padded_ids = F.pad(input_ids, (0, 1), value=self.cfg.pad_token_id).clone()
            padded_mask = F.pad(mask, (0, 1), value=False).clone()
            row = torch.arange(b, device=input_ids.device)
            padded_ids[row, lengths] = self.cfg.decision_token_id
            padded_mask[row, lengths] = True
            readout_idx = lengths
            return padded_ids, padded_mask, readout_idx
        if (lengths == 0).any():
            raise ValueError("cannot read out empty examples without appending a decision token")
        return input_ids, mask, lengths - 1

    def encode(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Return causal token states (B,T,C) WITHOUT adding a decision token.

        Useful for causal-prefix invariance checks, probing, and feature use.
        """
        token_mask, _ = self._check_ids_and_mask(input_ids, attention_mask)
        if input_ids.size(1) > self.cfg.max_context:
            raise ValueError("input exceeds max_context")
        return self._run_core(input_ids, token_mask)

    def _run_core(self, ids: Tensor, mask: Tensor) -> Tensor:
        # Mask padded embeddings; their states can evolve, but because every
        # operation is causal, they cannot affect earlier real/decision tokens.
        x = self.embedding_dropout(self.token_embeddings(ids))
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        for layer in self.layers:
            if self.training and self.cfg.gradient_checkpointing:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.final_norm(x)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        *,
        append_decision_token: bool = True,
        return_decision_states: bool = False,
    ) -> SemanticDecisionOutput:
        ids, mask, readout_positions = self._prepare_tokens(
            input_ids, attention_mask, append_decision_token
        )
        states = self._run_core(ids, mask)
        row = torch.arange(ids.size(0), device=ids.device)
        decision_states = states[row, readout_positions, :]
        logits = self.decision_head(decision_states)
        loss = None
        if labels is not None:
            if labels.ndim != 1 or labels.shape[0] != ids.shape[0]:
                raise ValueError("labels must be (B,) integer class indices")
            loss = F.cross_entropy(logits.float(), labels.long(),
                                   label_smoothing=self.cfg.label_smoothing)
        return SemanticDecisionOutput(
            logits=logits,
            loss=loss,
            decision_states=decision_states if return_decision_states else None,
        )

    @torch.no_grad()
    def predict(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        abstain_below: Optional[float] = None,
        abstain_class_id: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return (class IDs, max-softmax scores); scores are NOT calibrated.

        If abstain_below is specified, predictions below the threshold are
        replaced by `abstain_class_id`, a class whose meaning you must define
        in your own task. This is a decision policy, not learned uncertainty.
        """
        if abstain_below is not None and not (0.0 <= abstain_below <= 1.0):
            raise ValueError("abstain_below must be in [0,1]")
        if abstain_below is not None and abstain_class_id is None:
            raise ValueError("abstain_class_id required when abstain_below is used")
        if abstain_class_id is not None and not (0 <= abstain_class_id < self.cfg.num_classes):
            raise ValueError("abstain_class_id out of range")
        was_training = self.training
        try:
            self.eval()
            probs = self(input_ids, attention_mask).logits.float().softmax(dim=-1)
            score, class_id = probs.max(dim=-1)
            if abstain_below is not None:
                class_id = torch.where(score >= abstain_below, class_id,
                                       torch.full_like(class_id, abstain_class_id))
            return class_id, score
        finally:
            self.train(was_training)


def test_semantic_floating_model() -> None:
    """Fast small-model checks: causality, isolation, right padding, gradients."""
    torch.manual_seed(7)
    cfg = SFCConfig(
        vocab_size=128,
        decision_token_id=127,
        pad_token_id=0,
        num_classes=5,
        max_context=64,
        depth=6,
        hidden_size=24,
        bottleneck_size=8,
        ren_groups=4,
        floating_hidden=8,
        dropout=0.0,
        embedding_dropout=0.0,
    )
    model = SemanticFloatingClassifier(cfg).eval()
    ids = torch.randint(1, 127, (2, 23), dtype=torch.long)
    modified = ids.clone()
    modified[:, 13:] = torch.randint(1, 127, (2, 10), dtype=torch.long)
    with torch.no_grad():
        states_a = model.encode(ids)
        states_b = model.encode(modified)
    torch.testing.assert_close(states_a[:, :13], states_b[:, :13], atol=2e-6, rtol=2e-6)

    # Changing another example must not affect sample 0.
    changed_batch = ids.clone()
    changed_batch[1] = torch.randint(1, 127, (23,), dtype=torch.long)
    with torch.no_grad():
        alone = model(ids[:1]).logits
        in_batch = model(changed_batch).logits[:1]
    torch.testing.assert_close(alone, in_batch, atol=2e-6, rtol=2e-6)

    # Sample 0 is right-padded, sample 1 is longer. Its decision token must
    # be inserted immediately after its last real token, not at physical end.
    batch = ids.clone()
    batch[0, 9:] = 0
    with torch.no_grad():
        solo = model(batch[:1, :9]).logits
        padded = model(batch).logits[:1]
    torch.testing.assert_close(solo, padded, atol=2e-6, rtol=2e-6)

    # No synthetic backward/autograd tricks; check every main component learns.
    model.train()
    output = model(ids, labels=torch.tensor([1, 2]))
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    for name in ["token_embeddings.weight", "layers.0.temporal.conv.weight",
                 "layers.0.channel_mixer.up.weight", "layers.5.temporal.conv.weight",
                 "decision_head.weight"]:
        param = dict(model.named_parameters())[name]
        if param.grad is None or not torch.isfinite(param.grad).all():
            raise AssertionError(f"missing/nonfinite gradient: {name}")
    print("PASS: causal prefixes, batch independence, padding, gradients")

    # Checkpointing should produce consistent output and finite gradients.
    chk_cfg = SFCConfig(
        vocab_size=128, decision_token_id=127, num_classes=5, max_context=64,
        depth=2, hidden_size=24, bottleneck_size=8, ren_groups=4,
        floating_hidden=8, gradient_checkpointing=True,
    )
    checkpointed = SemanticFloatingClassifier(chk_cfg)
    out = checkpointed(ids, labels=torch.tensor([1, 2]))
    assert out.loss is not None
    out.loss.backward()
    assert checkpointed.layers[0].temporal.conv.weight.grad is not None
    print("PASS: gradient checkpointing")


def main() -> None:
    parser = argparse.ArgumentParser(description="SFC-128 architecture and smoke tests")
    parser.add_argument("--full-forward", action="store_true",
                        help="run a tiny full-depth eval forward pass as well")
    args = parser.parse_args()
    # Keep tiny CPU tests responsive on large machines with many CPU threads.
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, torch.get_num_threads()))
    test_semantic_floating_model()
    default_cfg = SFCConfig()
    model = SemanticFloatingClassifier(default_cfg)
    print(f"SFC-{default_cfg.depth}: {model.count_parameters():,} trainable parameters")
    print(f"  effective receptive field: {model.receptive_field:,} tokens")
    for name, size in model.parameter_breakdown().items():
        print(f"  {name:14s} {size:,}")
    if args.full_forward:
        model.eval()
        ids = torch.randint(1, default_cfg.vocab_size - 1, (1, 12))
        with torch.no_grad():
            result = model(ids)
        print("PASS: full-depth forward", tuple(result.logits.shape),
              "finite=", bool(torch.isfinite(result.logits).all()))


if __name__ == "__main__":
    main()
