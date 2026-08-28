"""Heads and controls for M81, the decisive I5 measurement.

Three arms are fitted over the M80 sparse atoms (m=8192, k=32, the only cell
admissible under N80.2) and are compared against controls fitted on the raw
frozen features.

Two registered substitutions, both forced and both recorded rather than
silently taken:

* **RBF.** v12 used `sklearn.svm.SVC`. At 128 classes over 65,536 fit rows a
  one-vs-one SVC is 8,128 binary problems on a dense kernel and is not
  tractable here. A Nystroem feature map followed by a linear head is used
  instead. It is a genuine RBF kernel machine and its explanation retains the
  same form -- similarity to stored landmarks -- so the control keeps its
  meaning.
* **Metric field.** v12 trained an anisotropic quadratic. Over atoms the
  closed-form diagonal analogue is used: per-class mean and inverse variance
  per atom. This is the natural refit of that head onto a sparse basis and it
  removes a training-schedule confound from the comparison. It is not the v12
  head and is not reported as one.

Scores for every atom head are computed without densifying the codes. At 8,192
atoms and 65,536 rows a dense matrix is 2.1 GB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from experiments.common.v13_sparse_dictionary import SparseCodes

_EPSILON = 1e-6


def _scatter_dense(codes: SparseCodes, rows: np.ndarray) -> torch.Tensor:
    """Densify a *subset* of rows only. Callers must keep the subset small."""
    dense = torch.zeros(
        (len(rows), codes.dictionary_size), dtype=torch.float32
    )
    indices = torch.from_numpy(codes.indices[rows])
    values = torch.from_numpy(codes.values[rows])
    dense.scatter_(1, indices, values)
    return dense


# --------------------------------------------------------------------------
# Arm 1 -- sparse linear head, L1 regularised
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseLinearHead:
    weight: torch.Tensor  # (classes, atoms)
    bias: torch.Tensor  # (classes,)

    def scores(self, codes: SparseCodes, *, batch_size: int = 4096) -> np.ndarray:
        out = np.empty((codes.rows, len(self.bias)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, codes.rows, batch_size):
                stop = min(start + batch_size, codes.rows)
                indices = torch.from_numpy(codes.indices[start:stop])
                values = torch.from_numpy(codes.values[start:stop])
                gathered = self.weight[:, indices.reshape(-1)].reshape(
                    len(self.bias), stop - start, indices.shape[1]
                )
                block = (gathered * values.unsqueeze(0)).sum(dim=2).T
                out[start:stop] = (block + self.bias).numpy()
        return out

    def contributions(
        self, codes: SparseCodes, predicted: np.ndarray
    ) -> np.ndarray:
        """Per-active-atom contribution to the predicted class score."""
        indices = torch.from_numpy(codes.indices)
        values = torch.from_numpy(codes.values)
        chosen = self.weight[torch.from_numpy(predicted)]
        gathered = torch.gather(chosen, 1, indices)
        return (gathered * values).numpy()

    def active_parameter_count(self, *, tolerance: float = 1e-8) -> int:
        return int((self.weight.abs() > tolerance).sum().item()) + len(self.bias)


def fit_sparse_linear(
    codes: SparseCodes,
    labels: np.ndarray,
    *,
    class_count: int,
    l1_penalty: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    atom_budget: int | None = None,
) -> SparseLinearHead:
    """Fit a linear head over atoms with a genuinely sparse coefficient matrix.

    The L1 term is applied as a proximal soft-threshold after each optimiser
    step rather than added to the loss. Adding ``l1 * |w|`` to the objective and
    differentiating it gives a subgradient that Adam's per-coordinate rescaling
    never drives to exactly zero: coefficients shrink uniformly, so accuracy
    collapses while the number of atoms cited per decision barely moves. This
    was measured before the correction and is recorded as note N81.4. The
    proximal operator yields exact zeros, which is what an explanation-length
    budget has to be counted against.

    ``atom_budget`` optionally projects each class onto its ``atom_budget``
    largest coefficients and retrains on that fixed support, giving a hard
    guarantee on explanation length rather than the soft one L1 provides.
    """
    generator = torch.Generator().manual_seed(seed)
    weight = torch.zeros(
        (class_count, codes.dictionary_size), dtype=torch.float32,
        requires_grad=True,
    )
    bias = torch.zeros(class_count, dtype=torch.float32, requires_grad=True)
    objective = torch.nn.CrossEntropyLoss()
    target = torch.from_numpy(labels.astype(np.int64))

    def train(support: torch.Tensor | None) -> None:
        optimizer = torch.optim.Adam([weight, bias], lr=learning_rate)
        for _ in range(epochs):
            order = torch.randperm(codes.rows, generator=generator)
            for start in range(0, codes.rows, batch_size):
                rows = order[start : start + batch_size]
                indices = torch.from_numpy(codes.indices[rows.numpy()])
                values = torch.from_numpy(codes.values[rows.numpy()])
                gathered = weight[:, indices.reshape(-1)].reshape(
                    class_count, len(rows), indices.shape[1]
                )
                logits = (gathered * values.unsqueeze(0)).sum(dim=2).T + bias
                loss = objective(logits, target[rows])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    if support is None:
                        threshold = learning_rate * l1_penalty
                        weight.copy_(
                            torch.sign(weight)
                            * torch.clamp(weight.abs() - threshold, min=0.0)
                        )
                    else:
                        weight.mul_(support)

    train(None)
    if atom_budget is not None and atom_budget < codes.dictionary_size:
        with torch.no_grad():
            # Rank by expected contribution mass, not by coefficient magnitude.
            # Magnitude alone selects rare atoms, which carry large weights
            # precisely because they seldom fire; the resulting head cites
            # almost nothing and collapses to chance. This was measured before
            # the correction and is recorded as note N81.5.
            activation = torch.zeros(codes.dictionary_size, dtype=torch.float32)
            activation.index_add_(
                0,
                torch.from_numpy(codes.indices.reshape(-1)),
                torch.from_numpy(np.abs(codes.values).reshape(-1)),
            )
            activation /= codes.rows
            keep = (weight.abs() * activation).topk(atom_budget, dim=1).indices
            support = torch.zeros_like(weight)
            support.scatter_(1, keep, 1.0)
            weight.mul_(support)
        train(support)
    return SparseLinearHead(weight.detach(), bias.detach())


# --------------------------------------------------------------------------
# Arm 2 -- short decision list over atoms
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionListHead:
    atoms: np.ndarray  # (rules,) atom index tested by each rule
    classes: np.ndarray  # (rules,) class predicted when the rule fires
    default_class: int

    def _fired(self, codes: SparseCodes) -> np.ndarray:
        """Index of the first rule whose atom is active; -1 for none."""
        fired = np.full(codes.rows, -1, dtype=np.int64)
        for position in range(len(self.atoms) - 1, -1, -1):
            active = np.any(codes.indices == self.atoms[position], axis=1)
            fired[active] = position
        return fired

    def predict(self, codes: SparseCodes) -> np.ndarray:
        fired = self._fired(codes)
        out = np.full(codes.rows, self.default_class, dtype=np.int64)
        matched = fired >= 0
        out[matched] = self.classes[fired[matched]]
        return out

    def contributions(self, codes: SparseCodes) -> np.ndarray:
        """One column per rule: the atom's code value where the rule fires."""
        fired = self._fired(codes)
        out = np.zeros((codes.rows, len(self.atoms)), dtype=np.float32)
        for position, atom in enumerate(self.atoms):
            rows = np.flatnonzero(fired == position)
            if len(rows) == 0:
                continue
            hit = codes.indices[rows] == atom
            out[rows, position] = (codes.values[rows] * hit).sum(axis=1)
        return out

    def active_parameter_count(self) -> int:
        return 2 * len(self.atoms) + 1


def fit_decision_list(
    codes: SparseCodes,
    labels: np.ndarray,
    *,
    class_count: int,
    max_rules: int,
) -> DecisionListHead:
    """Greedy rule list.

    Rules are selected on **advantage** -- rows the rule gets right minus rows
    it gets wrong -- not on raw coverage. Selecting on coverage alone picks the
    most frequently active atoms, which are the least class-specific ones, and
    builds a long list of impure rules.
    """
    remaining = np.ones(codes.rows, dtype=bool)
    atoms: list[int] = []
    classes: list[int] = []

    for _ in range(max_rules):
        rows = np.flatnonzero(remaining)
        if len(rows) == 0:
            break
        flat_atoms = codes.indices[rows].reshape(-1)
        flat_labels = np.repeat(labels[rows], codes.indices.shape[1])
        pair_counts = np.zeros(
            (codes.dictionary_size, class_count), dtype=np.int64
        )
        np.add.at(pair_counts, (flat_atoms, flat_labels), 1)
        atom_totals = pair_counts.sum(axis=1, keepdims=True)
        advantage = 2 * pair_counts - atom_totals
        best_flat = int(np.argmax(advantage))
        atom, klass = divmod(best_flat, class_count)
        if advantage[atom, klass] <= 0:
            break
        atoms.append(int(atom))
        classes.append(int(klass))
        fires = np.any(codes.indices == atom, axis=1)
        remaining &= ~fires

    leftover = labels[remaining] if remaining.any() else labels
    default_class = int(np.bincount(leftover, minlength=class_count).argmax())
    return DecisionListHead(
        np.asarray(atoms, dtype=np.int64),
        np.asarray(classes, dtype=np.int64),
        default_class,
    )


# --------------------------------------------------------------------------
# Arm 3 -- metric field over atoms (diagonal, closed form)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricFieldHead:
    centers: torch.Tensor  # (classes, atoms)
    precisions: torch.Tensor  # (classes, atoms)

    def scores(self, codes: SparseCodes, *, batch_size: int = 2048) -> np.ndarray:
        """Negative weighted squared distance, computed on sparse codes.

        For inactive atoms the term is `precision * center**2`, which is a
        per-class constant. Only the active atoms need per-row work.
        """
        constant = (self.precisions * self.centers.pow(2)).sum(dim=1)
        out = np.empty((codes.rows, len(constant)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, codes.rows, batch_size):
                stop = min(start + batch_size, codes.rows)
                indices = torch.from_numpy(codes.indices[start:stop])
                values = torch.from_numpy(codes.values[start:stop])
                flat = indices.reshape(-1)
                centers = self.centers[:, flat].reshape(
                    len(constant), stop - start, indices.shape[1]
                )
                precisions = self.precisions[:, flat].reshape(
                    len(constant), stop - start, indices.shape[1]
                )
                active = precisions * (values.unsqueeze(0) - centers).pow(2)
                inactive = precisions * centers.pow(2)
                delta = (active - inactive).sum(dim=2).T
                out[start:stop] = (-(constant + delta)).numpy()
        return out

    def contributions(
        self, codes: SparseCodes, predicted: np.ndarray
    ) -> np.ndarray:
        indices = torch.from_numpy(codes.indices)
        values = torch.from_numpy(codes.values)
        rows = torch.from_numpy(predicted)
        centers = torch.gather(self.centers[rows], 1, indices)
        precisions = torch.gather(self.precisions[rows], 1, indices)
        return (-precisions * (values - centers).pow(2)).numpy()

    def active_parameter_count(self) -> int:
        return int(self.centers.numel() + self.precisions.numel())


def fit_metric_field(
    codes: SparseCodes,
    labels: np.ndarray,
    *,
    class_count: int,
    variance_shrinkage: float = 0.1,
) -> MetricFieldHead:
    """Moment-matched diagonal field over atoms.

    The variance floor is **relative, not absolute**. An absolute floor of 1e-6
    gives precision 1e6 to every atom a class never activates, so a single
    unexpected atom outvotes the entire rest of the explanation and the head
    collapses to chance. The floor is instead a fraction of the pooled
    per-atom variance, which is the standard shrinkage prior and keeps every
    precision on one scale.
    """
    atoms = codes.dictionary_size
    totals = np.zeros((class_count, atoms), dtype=np.float64)
    squares = np.zeros((class_count, atoms), dtype=np.float64)
    row_labels = np.repeat(labels, codes.indices.shape[1])
    flat_atoms = codes.indices.reshape(-1)
    flat_values = codes.values.reshape(-1).astype(np.float64)
    np.add.at(totals, (row_labels, flat_atoms), flat_values)
    np.add.at(squares, (row_labels, flat_atoms), flat_values**2)

    counts = np.bincount(labels, minlength=class_count).astype(np.float64)
    means = totals / counts[:, None]
    variances = np.maximum(squares / counts[:, None] - means**2, 0.0)

    pooled = float(variances.mean())
    prior = max(variance_shrinkage * pooled, _EPSILON)
    precisions = 1.0 / (variances + prior)
    return MetricFieldHead(
        torch.from_numpy(means.astype(np.float32)),
        torch.from_numpy(precisions.astype(np.float32)),
    )


# --------------------------------------------------------------------------
# Control -- MLP, in torch so gradient attributions are available
# --------------------------------------------------------------------------


class MultilayerPerceptron(torch.nn.Module):
    """``hidden=0`` gives a single linear layer, which is what the Nystroem RBF
    control needs: a kernel machine is linear in its feature map, so adding a
    hidden layer there would make it something other than an RBF model."""

    def __init__(self, dimension: int, hidden: int, class_count: int) -> None:
        super().__init__()
        if hidden <= 0:
            self.stack = torch.nn.Sequential(torch.nn.Linear(dimension, class_count))
        else:
            self.stack = torch.nn.Sequential(
                torch.nn.Linear(dimension, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, class_count),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.stack(features)


def fit_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> MultilayerPerceptron:
    torch.manual_seed(seed)
    model = MultilayerPerceptron(features.shape[1], hidden, class_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    objective = torch.nn.CrossEntropyLoss()
    tensor = torch.from_numpy(features)
    target = torch.from_numpy(labels.astype(np.int64))
    generator = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(features), generator=generator)
        for start in range(0, len(features), batch_size):
            rows = order[start : start + batch_size]
            loss = objective(model(tensor[rows]), target[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def integrated_gradients(
    model: MultilayerPerceptron,
    features: np.ndarray,
    predicted: np.ndarray,
    *,
    baseline: np.ndarray,
    steps: int,
    batch_size: int = 512,
) -> np.ndarray:
    """Sundararajan et al. 2017, single fixed baseline."""
    base = torch.from_numpy(baseline.astype(np.float32)).unsqueeze(0)
    out = np.empty_like(features)
    alphas = torch.linspace(1.0 / steps, 1.0, steps).view(steps, 1, 1)
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        block = torch.from_numpy(features[start:stop])
        target = torch.from_numpy(predicted[start:stop].astype(np.int64))
        difference = block - base
        path = base.unsqueeze(0) + alphas * difference.unsqueeze(0)
        path = path.reshape(-1, features.shape[1]).requires_grad_(True)
        logits = model(path)
        selected = logits.gather(
            1, target.repeat(steps).unsqueeze(1)
        ).sum()
        (gradient,) = torch.autograd.grad(selected, path)
        averaged = gradient.reshape(steps, stop - start, -1).mean(dim=0)
        out[start:stop] = (difference * averaged).detach().numpy()
    return out


def expected_gradients(
    model: MultilayerPerceptron,
    features: np.ndarray,
    predicted: np.ndarray,
    *,
    reference: np.ndarray,
    samples: int,
    seed: int,
    batch_size: int = 512,
) -> np.ndarray:
    """GradientExplainer-style SHAP estimate (Erion et al.).

    The `shap` package is unavailable and cannot be installed: `.venv` is the
    frozen replay environment for the sealed M73/M77 hashes. KernelSHAP is in
    any case infeasible at 128 classes over 384 features. Expected gradients is
    a SHAP-family estimator and is named as such -- never as KernelSHAP.
    """
    generator = torch.Generator().manual_seed(seed)
    reference_tensor = torch.from_numpy(reference.astype(np.float32))
    out = np.zeros_like(features)
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        block = torch.from_numpy(features[start:stop])
        target = torch.from_numpy(predicted[start:stop].astype(np.int64))
        total = torch.zeros_like(block)
        for _ in range(samples):
            picks = torch.randint(
                len(reference_tensor), (stop - start,), generator=generator
            )
            base = reference_tensor[picks]
            alpha = torch.rand((stop - start, 1), generator=generator)
            point = (base + alpha * (block - base)).requires_grad_(True)
            selected = model(point).gather(1, target.unsqueeze(1)).sum()
            (gradient,) = torch.autograd.grad(selected, point)
            total += gradient * (block - base)
        out[start:stop] = (total / samples).detach().numpy()
    return out


# --------------------------------------------------------------------------
# Explanation vectors -- the v12 identity-withheld form
# --------------------------------------------------------------------------


def withheld_explanation(
    contributions: np.ndarray, *, top_count: int
) -> np.ndarray:
    """Exactly the v12 construction: sorted top contribution magnitudes plus
    three summary statistics, with component identity withheld.

    Identity is withheld because the v12 record this milestone is compared
    against was measured this way. See registration note N81.2 for why that
    choice bounds what M81 can conclude about nameability.
    """
    ordered = np.sort(contributions, axis=1)[:, ::-1]
    count = min(top_count, ordered.shape[1])
    selected = ordered[:, :count]
    if selected.shape[1] < top_count:
        selected = np.pad(
            selected, ((0, 0), (0, top_count - selected.shape[1]))
        )
    if contributions.shape[1] == 0:
        # A head that cites nothing -- an empty decision list, for instance --
        # is degenerate rather than erroneous. It is given an all-zero
        # explanation so that I5 records it at the null instead of crashing,
        # which is the honest reading: no explanation, no simulatability.
        summary = np.zeros((len(contributions), 3), dtype=np.float32)
    else:
        summary = np.column_stack(
            [
                np.sum(contributions, axis=1),
                np.max(contributions, axis=1),
                np.mean(contributions, axis=1),
            ]
        )
    return np.column_stack([selected, summary]).astype(np.float32)


def active_atoms_per_decision(
    contributions: np.ndarray, *, tolerance: float = 1e-8
) -> float:
    return float(np.mean(np.sum(np.abs(contributions) > tolerance, axis=1)))


def cited_atoms_per_decision(
    contributions: np.ndarray, *, budget: int, tolerance: float = 1e-8
) -> dict[str, Any]:
    """How much of the decision the reader's budget actually covers.

    M79 registered a 30-second read at 10 active atoms. A head that needs 40
    atoms to reach its score does not meet that budget just because the top 10
    are shown.
    """
    magnitudes = np.abs(contributions)
    ordered = np.sort(magnitudes, axis=1)[:, ::-1]
    total = ordered.sum(axis=1)
    covered = ordered[:, :budget].sum(axis=1)
    safe = np.where(total > tolerance, total, 1.0)
    return {
        "budget": int(budget),
        "mean_active_atoms": active_atoms_per_decision(
            contributions, tolerance=tolerance
        ),
        "mean_mass_within_budget": float(np.mean(covered / safe)),
        "fraction_of_decisions_within_budget": float(
            np.mean(np.sum(magnitudes > tolerance, axis=1) <= budget)
        ),
    }
