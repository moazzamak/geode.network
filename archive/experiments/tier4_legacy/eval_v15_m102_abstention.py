"""M102 — abstention as the objective. Tier A: gate quality, not compute.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v15.md`` section 7.7,
testing **H110**: a sparse model whose fitting objective is *deferral quality*
recovers substantially more of the oracle cascade headroom than an
accuracy-fitted model's confidence margin.

**What Tier A can and cannot show.** Every arm here consumes the sealed 384-d
DINOv2 feature, so every arm pays the full 6,065,759,232-MAC trunk on every
input, including the ones it "declines" to send to stage-2. **No arm saves any
compute.** Tier A measures whether the *deferral decision* is any good. Plan
section 11.2 item 20 forbids reporting any number from this runner in the
language of compute saving. Tier B — refitting stage-1 on a cheap input so a
non-deferred row genuinely never pays the trunk — is conditional on H110 and is
not run here.

The registered quantity is the **recovered fraction**::

    (arm_accuracy - random_null_accuracy) / (oracle_accuracy - random_null_accuracy)

that is, how much of the gain an oracle deferral rule could have delivered the
arm's gate actually captures. Plan section 8.9 D5 records that this framing was
searched for and not located in the literature; under section 8.6 item 15 that
is recorded as a search limit and **not** as a claim that it is absent.

Registration notes carried by this runner:

* **N102.1 — the planning probe's headline arm is inadmissible here.** Plan
  section 2.8 read 44.4% recovery from a 64-dimensional stage-1. At 448 fit rows
  per class a 64-vector class mean has 7 fit samples per fitted dimension, below
  the floor of 10 that section 5.3 never waives. The d=64 arm is therefore run
  and reported ``void``, not negative (M83.1 / N83.8). The primary operating
  point is d=32 (ratio 14), whose probe reading was 45.4%. H110's 0.60 bar sits
  above both, so the verdict does not depend on this.
* **N102.2 — thresholds and temperature are fitted on a calibration split**
  disjoint from both fit and evaluation. Choosing a deferral threshold on the
  evaluation rows would make every reported deferral rate optimistic, which is
  the selective-prediction analogue of the defect that voided M80's gate.
* **N102.3 — arm (b') exists to try to make H110 unnecessary.** Guo et al.
  (2017) remove most of a deep network's calibration error with one scalar. If
  temperature scaling alone clears the bar, H110 is refuted *by sufficiency of
  the baseline* and that is the reported finding.
* **N102.4 — arms (b) and (c) differ only in the objective.** Same trunk, same
  width, same L1 penalty, same epochs, same optimiser, same seed. The
  comparison therefore isolates the objective and not the capacity.
* **N102.5 — joint gate training is established prior art.** SelectiveNet
  (arXiv:1901.09192), Madras et al. (arXiv:1711.06664) and Mozannar & Sontag
  (arXiv:2006.01862) all train the gate with the task. Plan section 11.2 item 22
  forbids presenting arm (c) as new.
* **N102.6 — the oracle is a ranking, not a policy.** At deferral rate p the
  oracle defers the stage-1 errors first and fills any remaining budget with
  correct rows chosen by a fixed seeded permutation. It is an upper bound on
  what *any* gate reading this stage-1 could achieve at that rate, not an
  achievable model.

Reproduce with::

    .\\.venv\\Scripts\\python.exe -m experiments.tier4.eval_v15_m102_abstention
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.tier4.eval_v13_m80_sparse_dictionary import (
    _load_corpus,
    _verify_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v15" / "m102_abstention.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v15" / "m102_abstention"


# --------------------------------------------------------------------------
# partition
# --------------------------------------------------------------------------
def _seeded_partition(
    labels: np.ndarray,
    *,
    fit_per_class: int,
    calibration_per_class: int,
    evaluation_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class-stratified split under a seeded within-class permutation.

    The seed varies split membership, so the three seeds give an error bar on
    the split and not only on the fit. M81's partition is deterministic because
    its seed varied the learned dictionary; M102 learns no dictionary, so
    without this the seed axis would be nearly inert for arm (a).
    """
    need = fit_per_class + calibration_per_class + evaluation_per_class
    generator = np.random.default_rng(seed)
    fit_rows: list[np.ndarray] = []
    calibration_rows: list[np.ndarray] = []
    evaluation_rows: list[np.ndarray] = []
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        if len(rows) < need:
            raise ValueError(
                f"M102 partition needs {need} rows for class {label}, "
                f"found {len(rows)}"
            )
        rows = rows[generator.permutation(len(rows))]
        fit_rows.append(np.sort(rows[:fit_per_class]))
        calibration_rows.append(
            np.sort(rows[fit_per_class : fit_per_class + calibration_per_class])
        )
        evaluation_rows.append(
            np.sort(
                rows[
                    fit_per_class
                    + calibration_per_class : fit_per_class
                    + calibration_per_class
                    + evaluation_per_class
                ]
            )
        )
    return (
        np.concatenate(fit_rows),
        np.concatenate(calibration_rows),
        np.concatenate(evaluation_rows),
    )


def _balanced_accuracy(predicted: np.ndarray, truth: np.ndarray) -> float:
    total = 0.0
    classes = np.unique(truth)
    for label in classes:
        mask = truth == label
        total += float(np.mean(predicted[mask] == label))
    return total / len(classes)


# --------------------------------------------------------------------------
# stage 2 — the expensive model
# --------------------------------------------------------------------------
def _weighted_knn(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    query: np.ndarray,
    *,
    neighbors: int,
    class_count: int,
    chunk: int = 512,
) -> np.ndarray:
    """Inverse-distance weighted kNN, carried unchanged from the M81 control."""
    fit_norms = np.sum(fit_features * fit_features, axis=1)
    predictions = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), chunk):
        block = query[start : start + chunk]
        distances = (
            fit_norms[None, :]
            - 2.0 * block @ fit_features.T
            + np.sum(block * block, axis=1)[:, None]
        )
        np.maximum(distances, 0.0, out=distances)
        nearest = np.argpartition(distances, neighbors, axis=1)[:, :neighbors]
        rows = np.arange(len(block))[:, None]
        near_distance = np.sqrt(distances[rows, nearest])
        weights = 1.0 / (near_distance + 1e-8)
        votes = np.zeros((len(block), class_count), dtype=np.float64)
        np.add.at(votes, (rows, fit_labels[nearest]), weights)
        predictions[start : start + chunk] = np.argmax(votes, axis=1)
    return predictions


# --------------------------------------------------------------------------
# stage 1 — arm (a), nearest class mean on a PCA subspace
# --------------------------------------------------------------------------
def _fit_pca(fit_features: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    mean = fit_features.mean(axis=0)
    centred = fit_features - mean
    # Economy SVD on the fit split only; calibration and evaluation rows are
    # never seen by the basis.
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    return mean, right[:dimensions].T


def _ncm(
    projected_fit: np.ndarray,
    fit_labels: np.ndarray,
    projected_query: np.ndarray,
    *,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, gate score). Gate = top-2 distance margin."""
    means = np.stack(
        [projected_fit[fit_labels == label].mean(axis=0) for label in range(class_count)]
    )
    distances = (
        np.sum(means * means, axis=1)[None, :]
        - 2.0 * projected_query @ means.T
        + np.sum(projected_query * projected_query, axis=1)[:, None]
    )
    order = np.argsort(distances, axis=1)
    best = order[:, 0]
    rows = np.arange(len(projected_query))
    margin = distances[rows, order[:, 1]] - distances[rows, best]
    return best, margin


# --------------------------------------------------------------------------
# stage 1 — arms (b), (b'), (c): a sparse linear trunk
# --------------------------------------------------------------------------
class _SparseSelective(torch.nn.Module):
    """Shared sparse linear trunk with a classifier head and a selector head.

    Arm (b) uses ``classifier`` only and reads its gate from the logit margin.
    Arm (c) trains ``classifier`` and ``selector`` jointly under the
    SelectiveNet objective. The two arms instantiate the identical module with
    the identical seed, so the only difference between them is the loss.
    """

    def __init__(self, dimensions: int, class_count: int) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(dimensions, class_count)
        self.selector = torch.nn.Sequential(
            torch.nn.Linear(dimensions, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.classifier(x), torch.sigmoid(self.selector(x)).squeeze(-1)


def _soft_threshold(weight: torch.Tensor, amount: float) -> None:
    """Proximal L1 step, applied as in N81.4 rather than added to the loss."""
    with torch.no_grad():
        weight.copy_(torch.sign(weight) * torch.clamp(weight.abs() - amount, min=0.0))


def _fit_sparse(
    projected_fit: np.ndarray,
    fit_labels: np.ndarray,
    *,
    class_count: int,
    settings: dict[str, Any],
    seed: int,
    selective: bool,
    target_coverage: float | None = None,
    coverage_penalty: float = 32.0,
) -> _SparseSelective:
    torch.manual_seed(seed)
    model = _SparseSelective(projected_fit.shape[1], class_count)
    optimiser = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    features = torch.from_numpy(projected_fit.astype(np.float32))
    targets = torch.from_numpy(fit_labels.astype(np.int64))
    batch = settings["batch_size"]
    penalty = settings["l1_penalty"] * settings["learning_rate"]
    generator = torch.Generator().manual_seed(seed)
    for _ in range(settings["epochs"]):
        order = torch.randperm(len(features), generator=generator)
        for start in range(0, len(order), batch):
            rows = order[start : start + batch]
            logits, selection = model(features[rows])
            per_sample = torch.nn.functional.cross_entropy(
                logits, targets[rows], reduction="none"
            )
            if selective:
                coverage = selection.mean()
                # Selective risk: the loss the model is accountable for is the
                # loss on the rows it CHOSE to answer, normalised by coverage.
                selective_risk = (selection * per_sample).sum() / (
                    selection.sum() + 1e-8
                )
                shortfall = torch.clamp(target_coverage - coverage, min=0.0)
                loss = selective_risk + coverage_penalty * shortfall * shortfall
            else:
                loss = per_sample.mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            _soft_threshold(model.classifier.weight, penalty)
            if selective:
                _soft_threshold(model.selector[0].weight, penalty)
    return model


def _fit_dense_gate(
    projected_calibration: np.ndarray,
    correct: np.ndarray,
    *,
    settings: dict[str, Any],
    seed: int,
) -> torch.nn.Module:
    """Arm (d): a budget-matched DENSE gate, supervised directly on 'was
    stage-1 right?'. This is the R5 null for the sparsity claim."""
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(projected_calibration.shape[1], settings["hidden"]),
        torch.nn.ReLU(),
        torch.nn.Linear(settings["hidden"], 1),
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    features = torch.from_numpy(projected_calibration.astype(np.float32))
    targets = torch.from_numpy(correct.astype(np.float32))
    batch = settings["batch_size"]
    generator = torch.Generator().manual_seed(seed)
    for _ in range(settings["epochs"]):
        order = torch.randperm(len(features), generator=generator)
        for start in range(0, len(order), batch):
            rows = order[start : start + batch]
            logit = model(features[rows]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, targets[rows]
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return model


def _fit_temperature(
    logits: np.ndarray, truth: np.ndarray, settings: dict[str, Any]
) -> float:
    """Guo et al. (2017) temperature scaling, fitted on the calibration split."""
    grid = np.linspace(
        settings["temperature_grid_min"],
        settings["temperature_grid_max"],
        settings["temperature_grid_points"],
    )
    tensor = torch.from_numpy(logits.astype(np.float32))
    target = torch.from_numpy(truth.astype(np.int64))
    best_temperature, best_loss = float(grid[0]), float("inf")
    for temperature in grid:
        loss = float(
            torch.nn.functional.cross_entropy(tensor / float(temperature), target)
        )
        if loss < best_loss:
            best_loss, best_temperature = loss, float(temperature)
    return best_temperature


# --------------------------------------------------------------------------
# the cascade measurement
# --------------------------------------------------------------------------
def _cascade(
    stage_one_predictions: np.ndarray,
    stage_two_predictions: np.ndarray,
    truth: np.ndarray,
    gate_score: np.ndarray,
    *,
    deferral_rate: float,
) -> float:
    """Defer the ``deferral_rate`` lowest-gate-score rows to stage 2.

    Ties are broken by a stable sort on the score, so an arm that emits a
    constant score degenerates to "defer the first p rows" rather than to a
    silently favourable ordering.
    """
    count = int(round(deferral_rate * len(truth)))
    order = np.argsort(gate_score, kind="stable")
    deferred = order[:count]
    predictions = stage_one_predictions.copy()
    predictions[deferred] = stage_two_predictions[deferred]
    return _balanced_accuracy(predictions, truth)


def _oracle(
    stage_one_predictions: np.ndarray,
    stage_two_predictions: np.ndarray,
    truth: np.ndarray,
    *,
    deferral_rate: float,
    seed: int,
) -> float:
    """Upper bound: defer the rows where deferring actually helps.

    **N102.6, corrected.** The first implementation deferred stage-1's errors
    first, chosen at random among them. That is an upper bound only when the
    error count fits inside the deferral budget. When stage-1 is weak its errors
    outnumber the budget, and a real gate can beat that rule by preferring the
    errors stage-2 can actually fix — which is exactly what was observed: arm
    (c) at 8 PCA dimensions read **141.4%** of a quantity that was supposed to
    cap it, and every anomalous cell coincided with error rate > deferral rate.

    The corrected oracle ranks rows by the benefit of deferring them:

    * ``+1`` — stage-1 wrong, stage-2 right. Deferring gains a row.
    * `` 0`` — both right or both wrong. Deferring changes nothing.
    * ``-1`` — stage-1 right, stage-2 wrong. Deferring loses a row.

    and spends the budget on the highest benefit first. Because the evaluation
    split is exactly balanced (equal rows per class), balanced accuracy equals
    plain accuracy here and this greedy rule is exactly optimal for the reported
    metric, not merely a heuristic. It is therefore a true upper bound on what
    *any* deferral rule at that rate could achieve against this stage-1 and this
    stage-2, and the recovered fraction is bounded above by 1.
    """
    count = int(round(deferral_rate * len(truth)))
    benefit = (stage_two_predictions == truth).astype(np.int8) - (
        stage_one_predictions == truth
    ).astype(np.int8)
    # Seeded permutation first, so ties inside a benefit tier are broken
    # reproducibly rather than by the corpus's own row order.
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(len(truth))
    order = shuffled[np.argsort(-benefit[shuffled], kind="stable")]
    deferred = order[:count]
    predictions = stage_one_predictions.copy()
    predictions[deferred] = stage_two_predictions[deferred]
    return _balanced_accuracy(predictions, truth)


def _random_null(
    stage_one_predictions: np.ndarray,
    stage_two_predictions: np.ndarray,
    truth: np.ndarray,
    *,
    deferral_rate: float,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    """R5 null: defer the SAME NUMBER of rows, chosen at random."""
    count = int(round(deferral_rate * len(truth)))
    generator = np.random.default_rng(seed)
    scores = []
    for _ in range(draws):
        deferred = generator.choice(len(truth), size=count, replace=False)
        predictions = stage_one_predictions.copy()
        predictions[deferred] = stage_two_predictions[deferred]
        scores.append(_balanced_accuracy(predictions, truth))
    return float(np.mean(scores)), float(np.std(scores))


def _recovered_fraction(arm: float, null: float, oracle: float) -> float | None:
    """Undefined when the oracle offers nothing to recover; emit null, not 0."""
    if oracle - null <= 1e-9:
        return None
    return (arm - null) / (oracle - null)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _gate_diagnostics(score: np.ndarray) -> dict[str, Any]:
    """Detect a degenerate gate before its number is read as a result.

    A gate that emits a (near-)constant score carries no information, and its
    cascade reading then measures the tie-breaking order rather than the gate.
    Such an arm must be reported as degenerate rather than as a low recovered
    fraction, on the same principle as M81's degeneracy floor (N81.8).
    """
    finite = np.isfinite(score)
    unique = int(len(np.unique(score[finite]))) if finite.any() else 0
    spread = float(np.std(score[finite])) if finite.any() else 0.0
    return {
        "distinct_values": unique,
        "standard_deviation": spread,
        "degenerate": bool(unique <= 1 or spread < 1e-9),
        "all_finite": bool(finite.all()),
    }


def _measure_gate(
    label: str,
    *,
    stage_one_predictions: np.ndarray,
    stage_two_predictions: np.ndarray,
    truth: np.ndarray,
    gate_score: np.ndarray,
    rates: list[float],
    draws: int,
    seed: int,
    active_parameters: int,
    gate_parameters: int,
    per_rate_score: dict[float, np.ndarray] | None = None,
) -> dict[str, Any]:
    points = []
    for rate in rates:
        score = gate_score if per_rate_score is None else per_rate_score[rate]
        arm = _cascade(
            stage_one_predictions,
            stage_two_predictions,
            truth,
            score,
            deferral_rate=rate,
        )
        oracle = _oracle(
            stage_one_predictions,
            stage_two_predictions,
            truth,
            deferral_rate=rate,
            seed=seed,
        )
        null, null_spread = _random_null(
            stage_one_predictions,
            stage_two_predictions,
            truth,
            deferral_rate=rate,
            draws=draws,
            seed=seed,
        )
        points.append(
            {
                "deferral_rate": rate,
                "cascade_balanced_accuracy": arm,
                "oracle_balanced_accuracy": oracle,
                "random_null_balanced_accuracy": null,
                "random_null_spread": null_spread,
                "recovered_fraction": _recovered_fraction(arm, null, oracle),
                "oracle_exceeds_stage_two_alone": None,
                "gate": _gate_diagnostics(score),
                # The oracle must dominate every real rule at the same rate, or
                # it is not an upper bound and the recovered fraction has no
                # denominator. The first implementation failed exactly here.
                "oracle_is_upper_bound": bool(oracle >= arm - 1e-9),
            }
        )
    return {
        "arm": label,
        "stage_one_balanced_accuracy": _balanced_accuracy(
            stage_one_predictions, truth
        ),
        "active_parameters": active_parameters,
        "gate_parameters": gate_parameters,
        "any_degenerate_gate": bool(
            any(p["gate"]["degenerate"] for p in points)
        ),
        "points": points,
    }


def _active(model: torch.nn.Module) -> int:
    with torch.no_grad():
        return int(
            sum(int((p != 0).sum()) for p in model.parameters() if p.requires_grad)
        )


def run_m102(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    torch.set_num_threads(config["threading"]["torch_threads"])

    index_path = _verify_corpus(config["corpus_index"])
    features, labels = _load_corpus(index_path)
    class_count = int(labels.max()) + 1
    partition = config["partition"]
    rates = config["deferral_rates"]
    draws = config["nulls"]["random_deferral_draws"]
    floor = config["sample_adequacy"]["fit_samples_per_fitted_dimension_floor"]

    seeds_payload = []
    for seed in config["seeds"]:
        started = time.time()
        fit_rows, calibration_rows, evaluation_rows = _seeded_partition(
            labels,
            fit_per_class=partition["fit_per_class"],
            calibration_per_class=partition["calibration_per_class"],
            evaluation_per_class=partition["evaluation_per_class"],
            seed=seed,
        )
        fit_features = features[fit_rows]
        fit_labels = labels[fit_rows]
        calibration_features = features[calibration_rows]
        calibration_labels = labels[calibration_rows]
        evaluation_features = features[evaluation_rows]
        evaluation_labels = labels[evaluation_rows]

        stage_two_predictions = _weighted_knn(
            fit_features,
            fit_labels,
            evaluation_features,
            neighbors=config["stage_two"]["neighbors"],
            class_count=class_count,
        )
        stage_two_accuracy = _balanced_accuracy(
            stage_two_predictions, evaluation_labels
        )

        widths = []
        for dimensions in config["stage_one"]["pca_dimensions"]:
            ratio = partition["fit_per_class"] / dimensions
            adequate = ratio >= floor
            mean, basis = _fit_pca(fit_features, dimensions)
            projected_fit = (fit_features - mean) @ basis
            projected_calibration = (calibration_features - mean) @ basis
            projected_evaluation = (evaluation_features - mean) @ basis

            arms = []

            # ---- arm (a): nearest class mean, gate = distance margin --------
            ncm_predictions, ncm_margin = _ncm(
                projected_fit,
                fit_labels,
                projected_evaluation,
                class_count=class_count,
            )
            arms.append(
                _measure_gate(
                    "a_ncm_margin",
                    stage_one_predictions=ncm_predictions,
                    stage_two_predictions=stage_two_predictions,
                    truth=evaluation_labels,
                    gate_score=ncm_margin,
                    rates=rates,
                    draws=draws,
                    seed=seed,
                    active_parameters=class_count * dimensions,
                    gate_parameters=0,
                )
            )

            # ---- arm (b): sparse linear fitted for accuracy -----------------
            settings_b = config["arms"]["b_sparse_margin"]
            model_b = _fit_sparse(
                projected_fit,
                fit_labels,
                class_count=class_count,
                settings=settings_b,
                seed=seed,
                selective=False,
            )
            with torch.no_grad():
                logits_evaluation = (
                    model_b.classifier(
                        torch.from_numpy(projected_evaluation.astype(np.float32))
                    )
                    .numpy()
                    .astype(np.float64)
                )
                logits_calibration = (
                    model_b.classifier(
                        torch.from_numpy(projected_calibration.astype(np.float32))
                    )
                    .numpy()
                    .astype(np.float64)
                )
            sorted_logits = np.sort(logits_evaluation, axis=1)
            b_predictions = np.argmax(logits_evaluation, axis=1)
            b_margin = sorted_logits[:, -1] - sorted_logits[:, -2]
            active_b = int((model_b.classifier.weight != 0).sum()) + class_count
            arms.append(
                _measure_gate(
                    "b_sparse_margin",
                    stage_one_predictions=b_predictions,
                    stage_two_predictions=stage_two_predictions,
                    truth=evaluation_labels,
                    gate_score=b_margin,
                    rates=rates,
                    draws=draws,
                    seed=seed,
                    active_parameters=active_b,
                    gate_parameters=0,
                )
            )

            # ---- arm (b'): the same model with a temperature-scaled gate ----
            settings_bp = config["arms"]["b_prime_sparse_temperature"]
            temperature = _fit_temperature(
                logits_calibration, calibration_labels, settings_bp
            )
            scaled = logits_evaluation / temperature
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            probabilities = np.exp(scaled)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            bp_score = probabilities.max(axis=1)
            record_bp = _measure_gate(
                "b_prime_sparse_temperature",
                stage_one_predictions=b_predictions,
                stage_two_predictions=stage_two_predictions,
                truth=evaluation_labels,
                gate_score=bp_score,
                rates=rates,
                draws=draws,
                seed=seed,
                active_parameters=active_b,
                gate_parameters=1,
            )
            record_bp["fitted_temperature"] = temperature
            arms.append(record_bp)

            # ---- arm (c): sparse selective, gate trained WITH the task ------
            settings_c = config["arms"]["c_sparse_selective"]
            per_rate_score: dict[float, np.ndarray] = {}
            c_predictions = None
            active_c = 0
            gate_parameters_c = 0
            for rate in rates:
                model_c = _fit_sparse(
                    projected_fit,
                    fit_labels,
                    class_count=class_count,
                    settings=settings_c,
                    seed=seed,
                    selective=True,
                    target_coverage=1.0 - rate,
                    coverage_penalty=settings_c["coverage_penalty"],
                )
                with torch.no_grad():
                    logits_c, selection_c = model_c(
                        torch.from_numpy(projected_evaluation.astype(np.float32))
                    )
                per_rate_score[rate] = selection_c.numpy().astype(np.float64)
                if c_predictions is None:
                    c_predictions = np.argmax(logits_c.numpy(), axis=1)
                    active_c = _active(model_c)
                    gate_parameters_c = int(
                        (model_c.selector[0].weight != 0).sum()
                    ) + 1
            # The classifier reported is the one trained at the PRIMARY rate,
            # so stage-1 accuracy is quoted from a single registered model.
            model_primary = _fit_sparse(
                projected_fit,
                fit_labels,
                class_count=class_count,
                settings=settings_c,
                seed=seed,
                selective=True,
                target_coverage=1.0 - config["primary_deferral_rate"],
                coverage_penalty=settings_c["coverage_penalty"],
            )
            with torch.no_grad():
                logits_primary, _ = model_primary(
                    torch.from_numpy(projected_evaluation.astype(np.float32))
                )
            c_predictions = np.argmax(logits_primary.numpy(), axis=1)
            arms.append(
                _measure_gate(
                    "c_sparse_selective",
                    stage_one_predictions=c_predictions,
                    stage_two_predictions=stage_two_predictions,
                    truth=evaluation_labels,
                    gate_score=np.zeros(len(evaluation_labels)),
                    rates=rates,
                    draws=draws,
                    seed=seed,
                    active_parameters=active_c,
                    gate_parameters=gate_parameters_c,
                    per_rate_score=per_rate_score,
                )
            )

            # ---- arm (d): budget-matched DENSE gate over arm (b) ------------
            settings_d = config["arms"]["d_dense_gate"]
            calibration_predictions = np.argmax(logits_calibration, axis=1)
            model_d = _fit_dense_gate(
                projected_calibration,
                (calibration_predictions == calibration_labels).astype(np.float64),
                settings=settings_d,
                seed=seed,
            )
            with torch.no_grad():
                d_score = (
                    model_d(torch.from_numpy(projected_evaluation.astype(np.float32)))
                    .squeeze(-1)
                    .numpy()
                    .astype(np.float64)
                )
            arms.append(
                _measure_gate(
                    "d_dense_gate",
                    stage_one_predictions=b_predictions,
                    stage_two_predictions=stage_two_predictions,
                    truth=evaluation_labels,
                    gate_score=d_score,
                    rates=rates,
                    draws=draws,
                    seed=seed,
                    active_parameters=active_b,
                    gate_parameters=_active(model_d),
                )
            )

            for arm in arms:
                for point in arm["points"]:
                    point["oracle_exceeds_stage_two_alone"] = bool(
                        point["oracle_balanced_accuracy"] > stage_two_accuracy
                    )

            widths.append(
                {
                    "pca_dimensions": dimensions,
                    "fit_samples_per_fitted_dimension": ratio,
                    "sample_adequate": bool(adequate),
                    "status": "measured" if adequate else "void_below_sample_floor",
                    "arms": arms,
                }
            )

        seeds_payload.append(
            {
                "seed": seed,
                "stage_two_balanced_accuracy": stage_two_accuracy,
                "widths": widths,
                "elapsed_seconds": time.time() - started,
            }
        )

    evidence: dict[str, Any] = {
        "milestone": "M102",
        "tier": "A",
        "hypothesis": "H110",
        "registered_in": config["registered_in"],
        "tier_a_saves_no_compute": True,
        "trunk_macs_per_input_all_arms": config["compute_ledger"][
            "trunk_macs_per_input"
        ],
        "corpus_index_sha256": config["corpus_index"]["sha256"],
        "configuration_hash": sha256_file(Path(config_path)),
        "seeds": seeds_payload,
    }
    evidence["gate"] = _build_gate(evidence, config)

    # An oracle that any real arm beats is not an upper bound, and every
    # recovered fraction computed against it is meaningless. This is checked as
    # a hard instrument failure rather than reported as a caveat, because the
    # first implementation of _oracle failed it and produced a 141% reading.
    violations = [
        (seed_row["seed"], width["pca_dimensions"], arm["arm"], point["deferral_rate"])
        for seed_row in evidence["seeds"]
        for width in seed_row["widths"]
        for arm in width["arms"]
        for point in arm["points"]
        if not point["oracle_is_upper_bound"]
    ]
    if violations:
        raise ValueError(
            "M102 instrument failure: the oracle was beaten by a real arm at "
            f"{len(violations)} point(s), so it is not an upper bound. "
            f"First: {violations[0]}"
        )

    evidence["payload_hash"] = payload_hash(evidence["seeds"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    return evidence


def _point(evidence: dict[str, Any], seed_row: dict[str, Any], arm: str,
           dimensions: int, rate: float) -> dict[str, Any] | None:
    width = next(
        (w for w in seed_row["widths"] if w["pca_dimensions"] == dimensions), None
    )
    if width is None:
        return None
    record = next((a for a in width["arms"] if a["arm"] == arm), None)
    if record is None:
        return None
    return next((p for p in record["points"] if p["deferral_rate"] == rate), None)


def _build_gate(evidence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Compute H110's verdict from the artifact, never from prose."""
    dimensions = config["stage_one"]["primary_dimensions"]
    rate = config["primary_deferral_rate"]
    bar = config["gate"]["recovered_fraction_bar"]

    def series(arm: str) -> list[float]:
        values = []
        for seed_row in evidence["seeds"]:
            point = _point(evidence, seed_row, arm, dimensions, rate)
            if point is not None and point["recovered_fraction"] is not None:
                values.append(point["recovered_fraction"])
        return values

    recovered = {arm: series(arm) for arm in
                 ("a_ncm_margin", "b_sparse_margin", "b_prime_sparse_temperature",
                  "c_sparse_selective", "d_dense_gate")}
    summary = {
        arm: {
            "mean": float(np.mean(v)) if v else None,
            "spread": float(np.max(v) - np.min(v)) if v else None,
            "per_seed": v,
        }
        for arm, v in recovered.items()
    }

    c_mean = summary["c_sparse_selective"]["mean"]
    c_spread = summary["c_sparse_selective"]["spread"] or 0.0
    b_mean = summary["b_sparse_margin"]["mean"]
    bp_mean = summary["b_prime_sparse_temperature"]["mean"]
    d_mean = summary["d_dense_gate"]["mean"]

    # A degenerate gate carries no information, and its cascade reading then
    # measures tie-breaking order rather than the gate. Such an arm is reported
    # as degenerate rather than as a low number.
    degenerate = {}
    for arm in recovered:
        flags = []
        for seed_row in evidence["seeds"]:
            point = _point(evidence, seed_row, arm, dimensions, rate)
            if point is not None:
                flags.append(point["gate"]["degenerate"])
        degenerate[arm] = bool(flags and all(flags))

    baseline_sufficient = bp_mean is not None and bp_mean > bar
    clears_bar = c_mean is not None and (c_mean - c_spread) > bar
    beats_accuracy_arm = (
        c_mean is not None and b_mean is not None and (c_mean - c_spread) > b_mean
    )

    if degenerate.get("c_sparse_selective"):
        verdict = "void_degenerate_gate"
    elif baseline_sufficient:
        verdict = "refuted_by_sufficiency_of_baseline"
    elif clears_bar and beats_accuracy_arm:
        verdict = "confirmed"
    else:
        verdict = "refuted"

    return {
        "hypothesis": "H110",
        "primary_pca_dimensions": dimensions,
        "primary_deferral_rate": rate,
        "recovered_fraction_bar": bar,
        "recovered_fraction": summary,
        "degenerate_gate": degenerate,
        "clears_bar": bool(clears_bar),
        "beats_accuracy_fitted_arm": bool(beats_accuracy_arm),
        "baseline_temperature_arm_sufficient": bool(baseline_sufficient),
        "sparse_gate_beats_dense_gate": (
            bool(c_mean > d_mean) if c_mean is not None and d_mean is not None else None
        ),
        "verdict": verdict,
        "tier_b_opens": bool(verdict == "confirmed"),
        "_note": (
            "Tier A measures gate quality only. Every arm consumed the full "
            "384-dimensional feature and therefore paid the full trunk on every "
            "input; no arm saved any compute. Plan section 11.2 item 20."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    evidence = run_m102(arguments.config, arguments.output)

    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    rate = config["primary_deferral_rate"]
    print(
        f"\nM102 Tier A — recovered fraction of oracle gain at "
        f"deferral rate {rate:.0%}\n"
        f"(gate quality only; every arm paid the full "
        f"{evidence['trunk_macs_per_input_all_arms']:,}-MAC trunk on every input)"
    )
    for seed_row in evidence["seeds"]:
        print(f"\n  seed {seed_row['seed']}  "
              f"stage-2 alone = {seed_row['stage_two_balanced_accuracy']:.4f}")
        for width in seed_row["widths"]:
            flag = "" if width["sample_adequate"] else "   [VOID: below sample floor]"
            print(f"    PCA {width['pca_dimensions']:>3}d "
                  f"({width['fit_samples_per_fitted_dimension']:.0f} fit/dim){flag}")
            for arm in width["arms"]:
                point = next(
                    p for p in arm["points"] if p["deferral_rate"] == rate
                )
                fraction = point["recovered_fraction"]
                print(
                    f"      {arm['arm']:>28}  "
                    f"stage1 {arm['stage_one_balanced_accuracy']:.4f}  "
                    f"cascade {point['cascade_balanced_accuracy']:.4f}  "
                    f"oracle {point['oracle_balanced_accuracy']:.4f}  "
                    f"null {point['random_null_balanced_accuracy']:.4f}  "
                    f"recovered "
                    f"{'n/a' if fraction is None else format(fraction, '.1%')}"
                )
    print()
    print(json.dumps(evidence["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
