"""The GEODE fingerprint embedder (v24 section 4, M168 build).

f(task) = normalise( sum_k emb(attribute_k) + mlp(descriptor) )

Design invariants:
- I2 additive composition: the attribute-embedding sum provides the
  traversable structure (the word2vec/GloVe mechanism); the tiny MLP
  captures axis interactions the sum misses. Both are trained once
  (M169) and then frozen.
- I1 freeze-on-ship / G1 determinism: with fixed weights and seeds,
  inference is a pure function — no dropout, no randomness, no LLM.

The continuity channel (sweep parameter -> learned 1-D embedding) is a
registered M169 training feature; this build reserves the interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from geode.core.descriptor import AXES, FALLBACK_TOKEN, NormalisedDescriptor

# The shipped trained weights (M224, frozen 2026-08-20). token_emb +
# mlp only; the M224 evidence and the byte-hash manifest live beside
# the file. The training gates G1-G3 passed on the sealed run
# (logs/results/v25/m224_fingerprint_v1_train/evidence.json).
SHIPPED_V1_PATH = Path(__file__).resolve().parent / "assets" / "fingerprint_v1.pt"


class FingerprintEncoder(nn.Module):
    """Additive attribute embeddings + a tiny MLP, F-dimensional output."""

    def __init__(self, f_dim: int = 16, mlp_hidden: int = 32,
                 mlp_on: bool = True, seed: int = 11,
                 continuity: Optional[int] = None,
                 axes: Optional[dict[str, list[str]]] = None):
        """The default schema is the frozen AXES; `axes` lets prototype
        runners train an EXTENDED schema (M225) without touching the
        frozen product schema (registered: product call sites pass
        nothing and are unaffected)."""
        super().__init__()
        self.f_dim = f_dim
        schema = AXES if axes is None else axes
        self.axis_order = list(schema)
        self.token_index: dict[tuple[str, str], int] = {}
        offset = 0
        for axis, vocab in schema.items():
            for token in vocab:
                self.token_index[(axis, token)] = offset
                offset += 1
        self.total_tokens = offset
        self.oov_index = offset
        gen = torch.Generator().manual_seed(seed)
        self.token_emb = nn.Embedding(self.total_tokens + 1, f_dim)
        nn.init.normal_(self.token_emb.weight, std=0.1, generator=gen)
        self.mlp_on = mlp_on
        if mlp_on:
            self.mlp = nn.Sequential(
                nn.Linear(self.total_tokens + 1, mlp_hidden),
                nn.Tanh(),
                nn.Linear(mlp_hidden, f_dim))
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.05, generator=gen)
                    nn.init.zeros_(layer.bias)
        else:
            self.mlp = None
        # the continuity channel interface (trained in M169): a learned
        # 1-D embedding of a sweep parameter, added to the sum.
        self.continuity = continuity
        if continuity is not None:
            self.sweep_emb = nn.Embedding(continuity, f_dim)
            nn.init.normal_(self.sweep_emb.weight, std=0.1, generator=gen)

    def _indices(self, desc: NormalisedDescriptor) -> torch.Tensor:
        indices = []
        for axis in self.axis_order:
            token = desc.axes.get(axis, FALLBACK_TOKEN)
            indices.append(self.token_index.get((axis, token),
                                                self.oov_index))
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, desc: NormalisedDescriptor,
                sweep_bin: Optional[int] = None) -> torch.Tensor:
        """Deterministic, unit-normalised fingerprint (F dims)."""
        idx = self._indices(desc)
        emb_sum = self.token_emb(idx).sum(dim=0)
        if self.mlp is not None:
            onehot = F.one_hot(idx, num_classes=self.total_tokens + 1
                               ).float().sum(dim=0)
            emb_sum = emb_sum + self.mlp(onehot.unsqueeze(0)).squeeze(0)
        if self.continuity is not None and sweep_bin is not None:
            emb_sum = emb_sum + self.sweep_emb(
                torch.tensor(sweep_bin, dtype=torch.long))
        return F.normalize(emb_sum, dim=-1)

    def fingerprint(self, desc: NormalisedDescriptor,
                    sweep_bin: Optional[int] = None) -> torch.Tensor:
        """The frozen inference path: no-grad, eval mode, deterministic."""
        with torch.no_grad():
            self.eval()
            return self.forward(desc, sweep_bin)

    @classmethod
    def pretrained_v1(cls, f_dim: int = 16, mlp_hidden: int = 32
                      ) -> "FingerprintEncoder":
        """The shipped M224-trained encoder (v1 weights, strict load).

        Loads token_emb + mlp from the shipped asset; any missing or
        unexpected key raises (strict). The loaded encoder is a pure
        function exactly like the random-init constructor.
        """
        enc = cls(f_dim=f_dim, mlp_hidden=mlp_hidden, mlp_on=True,
                  seed=11)
        state = torch.load(SHIPPED_V1_PATH, map_location="cpu",
                           weights_only=True)
        enc.load_state_dict(state, strict=True)
        enc.eval()
        return enc


class EmpiricalFingerprintEncoder(nn.Module):
    """The F..N segment (M227: two SEPARATE fingerprints).

    The empirical fingerprint is MEASURED-not-DECLARED: it is trained
    ONLY on measured task x arm outcomes (registry-owned probes +
    routing history - the M227 amendment 2 contributor policy) and is
    ABSENT until trained. While untrained, encode() returns None and
    routing falls back to the task fingerprint (the registered v0
    combination rule: task fingerprint gates admission, empirical
    fingerprint ranks selection).

    Training is GATED on the registered data-volume trigger (the
    measured-label inventory is currently too thin - M227 invariant 4),
    so train_on_measured() raises with that note until the trigger is
    met.
    """

    def __init__(self, n_dim: int = 32, seed: int = 11):
        super().__init__()
        self.n_dim = n_dim
        self.seed = seed
        self._trained = False
        self.weight = nn.Parameter(torch.zeros(1, n_dim))

    @property
    def trained(self) -> bool:
        return self._trained

    def encode(self, desc: NormalisedDescriptor
               ) -> Optional[torch.Tensor]:
        """None while untrained - an absent segment does not exist in
        the metric (registered: it is simply not used)."""
        if not self._trained:
            return None
        with torch.no_grad():
            self.eval()
            return F.normalize(self.weight, dim=-1).squeeze(0)

    def train_on_measured(self, records: Any) -> None:
        """The registered future hook (the F..N training)."""
        raise RuntimeError(
            "the empirical segment is GATED on the registered "
            "data-volume trigger: the measured task x arm outcome "
            "inventory is too thin to train on (M227 invariant 4, "
            "20 Aug 2026)")


class DriftGate:
    """M242: the empirical-profile validity gate (deterministic).

    An arm's empirical profile is INVALID for ranking when (a) the
    cosine distance to the registered measured profile exceeds the
    drift bound, or (b) the measurement is stale in LEDGER INDEX
    space (no wall clocks — the reproducibility rule). The gate
    consumes only quorum-admitted measurements (M245); arm
    self-reports never enter (the Byzantine contract).
    """

    def __init__(self, drift_bound: float = 0.2,
                 staleness_window: int = 100):
        if drift_bound < 0.0 or drift_bound > 1.0:
            raise ValueError("drift_bound must lie in [0, 1] "
                             "(cosine-distance space)")
        if staleness_window < 0:
            raise ValueError("staleness_window must be non-negative")
        self.drift_bound = float(drift_bound)
        self.staleness_window = int(staleness_window)

    def drift(self, profile: Sequence[float],
              arm_profile: Sequence[float]) -> float:
        """1 - cosine between two equal-length profiles."""
        if len(profile) != len(arm_profile):
            raise ValueError("profile lengths differ "
                             f"({len(profile)} vs {len(arm_profile)})")
        dot = sum(a * b for a, b in zip(profile, arm_profile))
        na = sum(a * a for a in profile) ** 0.5
        nb = sum(b * b for b in arm_profile) ** 0.5
        if na == 0.0 or nb == 0.0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - dot / (na * nb)))

    def fresh(self, measured_index: int,
              as_of_index: int) -> bool:
        """Staleness in ledger-index space (deterministic)."""
        return as_of_index - measured_index <= self.staleness_window

    def admits(self, profile: Sequence[float],
               arm_profile: Sequence[float],
               measured_index: int, as_of_index: int,
               bound: Optional[float] = None
               ) -> tuple[bool, str]:
        """(admitted, reason) — the single validity decision."""
        if not self.fresh(measured_index, as_of_index):
            return False, "stale_measurement"
        b = self.drift_bound if bound is None else float(bound)
        d = self.drift(profile, arm_profile)
        if d > b:
            return False, f"drift_exceeded ({d:.6f} > {b:.6f})"
        return True, "admitted"
