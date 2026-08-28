"""Top-k sparse autoencoder over frozen features — the M80 concept dictionary.

Registered hypothesis H80: frozen DINOv2 embeddings admit an overcomplete sparse
decomposition whose atoms are substantially monosemantic, at a reconstruction
fidelity sufficient to preserve downstream accuracy.

**On the M78 sample-adequacy floor.** That floor (ten samples per fitted
dimension) is a statement about estimating a *per-class* basis, where the fitted
parameters and the samples that determine them are in one-to-one correspondence.
A dictionary is fitted jointly over the whole corpus and is deliberately
overcomplete, so the ratio is not defined the same way and the floor does not
transfer. The guard here is instead a **held-out split**: every reported operand
is measured on rows the dictionary never saw. That substitution is registered
rather than assumed, so it can be argued with.

**Dead atoms are reported, never resampled.** Resampling dead atoms is standard
practice in the sparse-autoencoder literature, and it is deliberately omitted
here because the dead fraction is a registered operand. Repairing an operand
before measuring it is how M77's probe objective went four milestones without
anyone noticing it had no gradient.

Codes are non-negative by construction (ReLU before the top-k selection) and
exactly ``active_atoms``-sparse per row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class SparseCodes:
    """Row-sparse non-negative codes, stored as values and atom indices."""

    indices: np.ndarray  # (rows, active_atoms) int64
    values: np.ndarray  # (rows, active_atoms) float32
    dictionary_size: int

    @property
    def rows(self) -> int:
        return int(self.indices.shape[0])

    def active_atom_count(self) -> np.ndarray:
        """Atoms with a strictly positive coefficient, per row."""
        return np.count_nonzero(self.values > 0.0, axis=1)

    def atom_usage(self) -> np.ndarray:
        """How many rows each atom is active on."""
        usage = np.zeros(self.dictionary_size, dtype=np.int64)
        live = self.indices[self.values > 0.0]
        if live.size:
            counts = np.bincount(live.ravel(), minlength=self.dictionary_size)
            usage[: len(counts)] = counts
        return usage


class SparseDictionary:
    """A top-k sparse autoencoder with unit-norm decoder atoms."""

    def __init__(
        self,
        *,
        encoder_weight: Tensor,
        encoder_bias: Tensor,
        decoder_weight: Tensor,
        pre_bias: Tensor,
        active_atoms: int,
    ) -> None:
        self.encoder_weight = encoder_weight.detach().clone()
        self.encoder_bias = encoder_bias.detach().clone()
        self.decoder_weight = decoder_weight.detach().clone()
        self.pre_bias = pre_bias.detach().clone()
        self.active_atoms = int(active_atoms)

    @property
    def dictionary_size(self) -> int:
        return int(self.decoder_weight.shape[1])

    def encode(self, features: Tensor) -> tuple[Tensor, Tensor]:
        centered = features - self.pre_bias
        activations = torch.relu(centered @ self.encoder_weight.T + self.encoder_bias)
        values, indices = torch.topk(activations, self.active_atoms, dim=1)
        return values, indices

    def decode(self, values: Tensor, indices: Tensor) -> Tensor:
        atoms = self.decoder_weight.T[indices]  # (rows, active_atoms, dimension)
        return (atoms * values.unsqueeze(-1)).sum(dim=1) + self.pre_bias

    def codes(self, features: np.ndarray, *, batch_size: int = 4096) -> SparseCodes:
        index_blocks: list[np.ndarray] = []
        value_blocks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                block = torch.from_numpy(features[start : start + batch_size])
                values, indices = self.encode(block)
                value_blocks.append(values.numpy().astype(np.float32))
                index_blocks.append(indices.numpy().astype(np.int64))
        return SparseCodes(
            indices=np.concatenate(index_blocks),
            values=np.concatenate(value_blocks),
            dictionary_size=self.dictionary_size,
        )

    def reconstruct(
        self, features: np.ndarray, *, batch_size: int = 4096
    ) -> np.ndarray:
        blocks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                block = torch.from_numpy(features[start : start + batch_size])
                values, indices = self.encode(block)
                blocks.append(self.decode(values, indices).numpy())
        return np.concatenate(blocks)


def _initialize(
    features: Tensor, *, dictionary_size: int, generator: torch.Generator
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dimension = int(features.shape[1])
    pre_bias = features.mean(dim=0)
    decoder = torch.randn(
        dimension, dictionary_size, generator=generator, dtype=features.dtype
    )
    decoder /= decoder.norm(dim=0, keepdim=True)
    encoder_weight = decoder.T.clone()
    encoder_bias = torch.zeros(dictionary_size, dtype=features.dtype)
    return encoder_weight, encoder_bias, decoder, pre_bias


def random_dictionary(
    features: np.ndarray, *, dictionary_size: int, active_atoms: int, seed: int
) -> SparseDictionary:
    """Untrained dictionary of random unit-norm atoms.

    The positive control for every dictionary operand. A trained dictionary that
    does not clearly beat this has learned nothing, and its reconstruction R^2 or
    probe accuracy would be a property of sparse random projection rather than of
    the corpus.
    """
    generator = torch.Generator().manual_seed(int(seed))
    tensor = torch.from_numpy(features)
    encoder_weight, encoder_bias, decoder, pre_bias = _initialize(
        tensor, dictionary_size=dictionary_size, generator=generator
    )
    return SparseDictionary(
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        decoder_weight=decoder,
        pre_bias=pre_bias,
        active_atoms=active_atoms,
    )


def fit_sparse_dictionary(
    features: np.ndarray,
    *,
    dictionary_size: int,
    active_atoms: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[SparseDictionary, dict[str, Any]]:
    """Fit the dictionary by reconstruction loss under a fixed top-k budget."""
    generator = torch.Generator().manual_seed(int(seed))
    tensor = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))

    encoder_weight, encoder_bias, decoder, pre_bias = _initialize(
        tensor, dictionary_size=dictionary_size, generator=generator
    )
    encoder_weight = encoder_weight.requires_grad_(True)
    encoder_bias = encoder_bias.requires_grad_(True)
    decoder = decoder.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [encoder_weight, encoder_bias, decoder], lr=float(learning_rate)
    )
    rows = int(tensor.shape[0])
    trace: list[float] = []

    for _ in range(int(epochs)):
        order = torch.randperm(rows, generator=generator)
        total = 0.0
        batches = 0
        for start in range(0, rows, batch_size):
            block = tensor[order[start : start + batch_size]]
            centered = block - pre_bias
            activations = torch.relu(centered @ encoder_weight.T + encoder_bias)
            values, indices = torch.topk(activations, active_atoms, dim=1)
            atoms = decoder.T[indices]
            reconstruction = (atoms * values.unsqueeze(-1)).sum(dim=1) + pre_bias
            loss = torch.mean((reconstruction - block) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                decoder /= decoder.norm(dim=0, keepdim=True).clamp_min(1e-8)

            total += float(loss.detach())
            batches += 1
        trace.append(total / max(batches, 1))

    dictionary = SparseDictionary(
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        decoder_weight=decoder,
        pre_bias=pre_bias,
        active_atoms=active_atoms,
    )
    diagnostics = {
        "epoch_loss_trace": [float(value) for value in trace],
        "final_train_loss": float(trace[-1]) if trace else float("nan"),
        "loss_decreased": bool(len(trace) > 1 and trace[-1] < trace[0]),
    }
    return dictionary, diagnostics


def reconstruction_r2(features: np.ndarray, reconstruction: np.ndarray) -> float:
    """Fraction of held-out feature variance explained, against the corpus mean."""
    residual = float(np.sum((features - reconstruction) ** 2))
    total = float(np.sum((features - features.mean(axis=0)) ** 2))
    return 1.0 - residual / total


def atom_label_entropy(
    codes: SparseCodes, labels: np.ndarray, *, class_count: int
) -> dict[str, float]:
    """Mean label entropy over live atoms, in bits. Lower means more monosemantic.

    Reported, not gating. H80 asserts monosemanticity but the registered M80
    operands do not measure it, so this closes that gap without moving the gate.
    """
    counts = np.zeros((codes.dictionary_size, class_count), dtype=np.float64)
    active = codes.values > 0.0
    np.add.at(counts, (codes.indices[active], labels.repeat(
        codes.indices.shape[1]
    ).reshape(codes.indices.shape)[active]), 1.0)

    totals = counts.sum(axis=1)
    live = totals > 0
    if not np.any(live):
        return {"mean_bits": float("nan"), "live_atoms": 0}
    distribution = counts[live] / totals[live][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(distribution > 0.0, np.log2(distribution), 0.0)
    entropy = -np.sum(distribution * logs, axis=1)
    return {
        "mean_bits": float(entropy.mean()),
        "live_atoms": int(np.count_nonzero(live)),
    }
