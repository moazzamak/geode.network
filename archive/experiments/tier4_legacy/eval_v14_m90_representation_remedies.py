"""M90: the three corrections that do not require training anything.

M89 located v13's rejection failure in the coordinates rather than in the
boundary: a known row is interior to a mean of 78.74 of 128 class regions, class
clouds are 2.34x wider than the gaps between them, and the gap between a class's
own domain cells (32.30) exceeds the gap to the nearest foreign class (19.15).

Three corrections follow from that and none of them touches the trunk --
normalise the coordinates, remove the domain mean, or stop pooling six domains
into one region. This run measures all three against v13's sealed values on
identical rows, each beside a null that holds its structure and budget constant.

Registered before execution (N90.1--N90.9), including the reading if every arm
ties or loses, which is a stronger statement of v13's finding than v13 could
make and is not a disappointment (N90.2).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common.v5_artifacts import payload_hash, write_canonical_json
from experiments.common.v13_boundary import (
    Geometry,
    boundary_scores,
    domain_auroc,
    domain_matched_partition,
    domain_stratified_halves,
    far_field_points,
    minimum_scores_numpy,
    rejection_recall,
    score_auroc,
)
from experiments.tier4.eval_v13_m80_sparse_dictionary import (
    _load_corpus,
    _verify_corpus,
)
from experiments.tier4.eval_v13_m84_exposure_ladder import (
    _corpus_domains,
    _load_arrays,
    _resolve,
    _verify_sealed,
)
from src.subspace_primitive import fit_subspace_primitive

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v14" / "m90_representation_remedies.json"
)

_EPSILON = 1e-12


@dataclass(frozen=True)
class Assignment:
    """Which fit rows each component is fitted on, and which class owns it.

    Components may overlap: the mixture arm gives every class one component over
    all its rows plus one per sufficiently supported domain cell. ``owner`` is
    non-decreasing so that a per-class minimum is a single ``reduceat``.
    """

    memberships: tuple[np.ndarray, ...]
    owner: np.ndarray

    @property
    def count(self) -> int:
        return len(self.memberships)

    @property
    def starts(self) -> np.ndarray:
        return np.searchsorted(self.owner, np.arange(self.owner[-1] + 1), side="left")


def class_assignment(labels: np.ndarray, class_count: int) -> Assignment:
    memberships = tuple(np.flatnonzero(labels == k) for k in range(class_count))
    return Assignment(memberships, np.arange(class_count, dtype=np.int64))


def domain_cell_assignment(
    labels: np.ndarray,
    domains: np.ndarray,
    *,
    class_count: int,
    domain_count: int,
    minimum_support: int,
) -> tuple[Assignment, list[list[int]]]:
    """One component per class, plus one per domain cell that clears support.

    The class-level component is what keeps the partition well-formed (N90.5):
    every row is covered by it, so the cells are a refinement rather than a
    partition that would strand rows in components too small to fit.
    """
    memberships: list[np.ndarray] = []
    owner: list[int] = []
    cell_sizes: list[list[int]] = []
    for k in range(class_count):
        rows = np.flatnonzero(labels == k)
        memberships.append(rows)
        owner.append(k)
        sizes: list[int] = []
        for d in range(domain_count):
            cell = rows[domains[rows] == d]
            if len(cell) >= minimum_support:
                memberships.append(cell)
                owner.append(k)
                sizes.append(int(len(cell)))
        cell_sizes.append(sizes)
    return Assignment(tuple(memberships), np.array(owner, dtype=np.int64)), cell_sizes


def random_cell_assignment(
    labels: np.ndarray,
    cell_sizes: list[list[int]],
    *,
    class_count: int,
    seed: int,
) -> Assignment:
    """N90.3's null: the same components over disjoint random subsets."""
    rng = np.random.default_rng(seed)
    memberships: list[np.ndarray] = []
    owner: list[int] = []
    for k in range(class_count):
        rows = np.flatnonzero(labels == k)
        memberships.append(rows)
        owner.append(k)
        shuffled = rng.permutation(rows)
        cursor = 0
        for size in cell_sizes[k]:
            memberships.append(np.sort(shuffled[cursor : cursor + size]))
            owner.append(k)
            cursor += size
    return Assignment(tuple(memberships), np.array(owner, dtype=np.int64))


def fit_components(
    features: np.ndarray, assignment: Assignment, *, rank: int
) -> Geometry:
    """Phase A, closed form, once per component instead of once per class."""
    dimension = features.shape[1]
    centers = np.zeros((assignment.count, dimension), dtype=np.float64)
    bases = np.zeros((assignment.count, dimension, rank), dtype=np.float64)
    tangent = np.zeros((assignment.count, rank), dtype=np.float64)
    residual = np.zeros(assignment.count, dtype=np.float64)
    for index, rows in enumerate(assignment.memberships):
        if len(rows) < rank + 2:
            raise ValueError(
                f"component {index} has {len(rows)} rows; rank {rank} needs {rank + 2}"
            )
        primitive = fit_subspace_primitive(features[rows], rank)
        centers[index] = primitive.center
        bases[index] = primitive.basis
        tangent[index] = np.sqrt(primitive.tangent_variances)
        residual[index] = float(np.sqrt(primitive.residual_variance))
    return Geometry(centers, bases, tangent, residual)


def class_scores(
    features: np.ndarray,
    geometry: Geometry,
    log_beta: np.ndarray,
    assignment: Assignment,
    *,
    chunk_rows: int = 512,
) -> np.ndarray:
    """Per-row, per-class score: the minimum over that class's components."""
    starts = assignment.starts
    beta = torch.as_tensor(log_beta, dtype=torch.float64)
    out = np.empty((len(features), len(starts)), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(features), chunk_rows):
            stop = min(start + chunk_rows, len(features))
            block = torch.as_tensor(features[start:stop], dtype=torch.float64)
            scores = boundary_scores(block, geometry, beta).cpu().numpy()
            out[start:stop] = np.minimum.reduceat(scores, starts, axis=1)
    return out


def union_coverage_offsets(
    features: np.ndarray,
    labels: np.ndarray,
    geometry: Geometry,
    log_beta: np.ndarray,
    assignment: Assignment,
    *,
    coverage: float,
    class_count: int,
) -> np.ndarray:
    """N90.7. One offset per class, matched over the union of its components.

    The conformal order statistic is N83.3's, unchanged. What changes is the
    score it is read on: a row counts as covered if *any* of its class's
    components accepts it, which is the only matching rule under which adding
    components can shrink the accepted volume instead of only growing it.
    """
    scores = class_scores(features, geometry, log_beta, assignment)
    offsets = np.zeros(class_count, dtype=np.float64)
    for label in range(class_count):
        rows = np.flatnonzero(labels == label)
        if len(rows) == 0:
            continue
        owned = np.sort(scores[rows, label])
        rank = int(np.ceil((len(owned) + 1) * coverage))
        threshold = float(owned[min(rank, len(owned)) - 1])
        offsets[label] = float(np.log(max(threshold, _EPSILON)))
    return offsets


# ---------------------------------------------------------------------------
# Transforms


def _l2_normalise(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, _EPSILON)


def _group_means(
    features: np.ndarray, groups: np.ndarray, *, group_count: int
) -> np.ndarray:
    means = np.zeros((group_count, features.shape[1]), dtype=np.float64)
    for g in range(group_count):
        rows = np.flatnonzero(groups == g)
        if len(rows):
            means[g] = features[rows].mean(axis=0)
    return means


def _random_groups(
    row_count: int, sizes: np.ndarray, *, seed: int
) -> np.ndarray:
    """A random partition with the same group sizes as the domain partition."""
    rng = np.random.default_rng(seed)
    scaled = np.floor(sizes / sizes.sum() * row_count).astype(np.int64)
    scaled[0] += row_count - int(scaled.sum())
    groups = np.repeat(np.arange(len(sizes), dtype=np.int64), scaled)
    return rng.permutation(groups)


# ---------------------------------------------------------------------------


def _controls(
    known: np.ndarray, far: np.ndarray, *, floor: float, tolerance: float
) -> dict[str, Any]:
    positive = score_auroc(known, far)
    midpoint = len(known) // 2
    negative = score_auroc(known[:midpoint], known[midpoint:])
    return {
        "positive_control": positive,
        "positive_floor": floor,
        "positive_passes": bool(positive >= floor),
        "negative_control": negative,
        "negative_tolerance": tolerance,
        "negative_passes": bool(abs(negative - 0.5) <= tolerance),
        "valid": bool(positive >= floor and abs(negative - 0.5) <= tolerance),
    }


def _clears(arm: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    if not arm["controls"]["valid"]:
        return {"clears": False, "reason": "instrument_invalid"}
    multiplicity = arm["acceptance_multiplicity"]["mean"]
    recall = arm["rejection_recall"]["rejection_recall"]
    auroc = arm["auroc"]["auroc"]
    checks = {
        "multiplicity_below_bar": bool(multiplicity < float(gate["multiplicity_bar"])),
        "recall_above_v13": bool(recall > float(gate["v13_rejection_recall"])),
        "auroc_above_v13_by_margin": bool(
            auroc > float(gate["v13_geometry_auroc"]) + float(gate["auroc_margin"])
        ),
    }
    return {"clears": all(checks.values()), **checks}


def _verdict(arms: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    """N90.8. An arm whose instrument failed refutes nothing.

    M83.1 is the precedent: a run that fails its own validity check is void, not
    negative. Reading ``clears == False`` as a refutation would let an invalid
    instrument close a hypothesis, which is the same error in a new place.
    """
    margin = float(gate["auroc_margin"])

    def valid(name: str) -> bool:
        return bool(arms[name]["controls"]["valid"])

    def beats(name: str, null: str) -> bool:
        if not valid(name) or not valid(null):
            return False
        return bool(
            arms[name]["rejection_recall"]["rejection_recall"]
            > arms[null]["rejection_recall"]["rejection_recall"]
            and arms[name]["auroc"]["auroc"] > arms[null]["auroc"]["auroc"] + margin
        )

    def decide(survivors: list[bool], deciding: list[str]) -> str:
        if any(survivors):
            return "survives"
        if all(valid(name) for name in deciding):
            return "refuted"
        return "undetermined"

    h90 = decide(
        [
            arms["domain_mixture"]["gate"]["clears"]
            and beats("domain_mixture", "domain_mixture_null")
        ],
        ["domain_mixture", "domain_mixture_null"],
    )
    h91 = decide(
        [
            arms["cosine"]["gate"]["clears"],
            arms["domain_centred"]["gate"]["clears"]
            and beats("domain_centred", "domain_centred_null"),
        ],
        ["cosine", "domain_centred", "domain_centred_null"],
    )
    both_refuted = h90 == "refuted" and h91 == "refuted"
    return {
        "h90_misspecification": {
            "verdict": h90,
            "basis": "domain_mixture must clear all three bars and beat its null.",
        },
        "h91_coordinates": {
            "verdict": h91,
            "basis": (
                "cosine or domain_centred must clear all three bars, and "
                "domain_centred must beat its null."
            ),
        },
        "m91_opens": both_refuted,
        "reading": (
            "Every arm was validly measured and every arm ties or loses. The "
            "rejection failure survives the three cheapest corrections "
            "available without training anything, which narrows the cause to "
            "the representation itself and opens M91 (N90.2)."
            if both_refuted
            else (
                "At least one hypothesis is undetermined because the arm that "
                "would decide it failed its own instrument validation. An "
                "invalid instrument is void, not negative (N90.8), so M91 does "
                "not open on it."
                if "undetermined" in (h90, h91)
                else "At least one correction moved the operands; see the arm table."
            )
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    started = time.time()

    torch.set_num_threads(1)
    config = json.loads(args.config.read_text(encoding="utf-8"))

    corpus_index = _verify_corpus(config["corpus"])
    features, labels = _load_corpus(corpus_index)
    domains = _corpus_domains(corpus_index)
    partition = config["partition"]
    class_count = int(config["corpus"]["class_count"])
    domain_count = int(config["corpus"]["domain_count"])

    fit_rows, evaluation_rows, partition_report = domain_matched_partition(
        labels,
        domains,
        quota=tuple(int(v) for v in partition["evaluation_domain_quota"]),
        fit_per_class=int(partition["fit_per_class"]),
        domain_count=domain_count,
    )
    calibration_rows, report_rows = domain_stratified_halves(
        labels, domains, evaluation_rows
    )

    openset_index = _verify_sealed(config["openset"], "open-set")
    open_features, open_labels, open_domains = _load_arrays(
        openset_index, stratified_only=bool(config["openset"]["stratified_only"])
    )
    keep = open_labels >= int(config["openset"]["evaluation_first_label"])
    unseen_raw = open_features[keep]
    unseen_domains = open_domains[keep]

    fit_raw = features[fit_rows]
    fit_domains = domains[fit_rows]
    calibration_raw = features[calibration_rows]
    calibration_labels = labels[calibration_rows]
    report_raw = features[report_rows]
    report_labels = labels[report_rows]
    report_domains = domains[report_rows]

    domain_sizes = np.array(
        [int((fit_domains == d).sum()) for d in range(domain_count)], dtype=np.int64
    )
    seeds = config["seeds"]
    null_groups = {
        "fit": _random_groups(
            len(fit_raw), domain_sizes, seed=int(seeds["random_group"])
        ),
        "calibration": _random_groups(
            len(calibration_raw), domain_sizes, seed=int(seeds["random_group"]) + 1
        ),
        "report": _random_groups(
            len(report_raw), domain_sizes, seed=int(seeds["random_group"]) + 2
        ),
        "unseen": _random_groups(
            len(unseen_raw), domain_sizes, seed=int(seeds["random_group"]) + 3
        ),
    }

    def transformed(name: str) -> dict[str, np.ndarray]:
        if name == "identity":
            return {
                "fit": fit_raw,
                "calibration": calibration_raw,
                "report": report_raw,
                "unseen": unseen_raw,
            }
        if name == "l2_normalise":
            return {
                "fit": _l2_normalise(fit_raw),
                "calibration": _l2_normalise(calibration_raw),
                "report": _l2_normalise(report_raw),
                "unseen": _l2_normalise(unseen_raw),
            }
        if name == "domain_centre":
            means = _group_means(fit_raw, fit_domains, group_count=domain_count)
            return {
                "fit": fit_raw - means[fit_domains],
                "calibration": calibration_raw - means[domains[calibration_rows]],
                "report": report_raw - means[report_domains],
                "unseen": unseen_raw - means[unseen_domains],
            }
        if name == "random_group_centre":
            means = _group_means(fit_raw, null_groups["fit"], group_count=domain_count)
            return {
                "fit": fit_raw - means[null_groups["fit"]],
                "calibration": calibration_raw - means[null_groups["calibration"]],
                "report": report_raw - means[null_groups["report"]],
                "unseen": unseen_raw - means[null_groups["unseen"]],
            }
        raise ValueError(f"unknown transform {name}")

    mixture_config = config["mixture"]
    cell_assignment, cell_sizes = domain_cell_assignment(
        labels[fit_rows],
        fit_domains,
        class_count=class_count,
        domain_count=domain_count,
        minimum_support=int(mixture_config["minimum_cell_support"]),
    )

    def assignment_for(kind: str) -> Assignment:
        if kind == "class":
            return class_assignment(labels[fit_rows], class_count)
        if kind == "domain_cell":
            return cell_assignment
        if kind == "random_cell":
            return random_cell_assignment(
                labels[fit_rows],
                cell_sizes,
                class_count=class_count,
                seed=int(seeds["random_cell"]),
            )
        raise ValueError(f"unknown component kind {kind}")

    controls_config = config["controls"]
    gate_config = config["gate"]
    arms: dict[str, Any] = {}

    for spec in config["arms"]:
        name = spec["name"]
        space = transformed(spec["transform"])
        assignment = assignment_for(spec["components"])
        rank = int(spec["rank"])
        geometry = fit_components(space["fit"], assignment, rank=rank)
        log_beta = np.concatenate(
            [
                np.log(geometry.tangent_scales),
                np.log(geometry.residual_scales)[:, None],
            ],
            axis=1,
        )
        offsets = union_coverage_offsets(
            space["calibration"],
            calibration_labels,
            geometry,
            log_beta,
            assignment,
            coverage=float(config["coverage"]["known_coverage"]),
            class_count=class_count,
        )
        matched = log_beta + offsets[assignment.owner][:, None]

        per_class = class_scores(space["report"], geometry, matched, assignment)
        counts = (per_class <= 1.0).sum(axis=1)
        known_scores = minimum_scores_numpy(space["report"], geometry, matched)[1]
        unseen_scores = minimum_scores_numpy(space["unseen"], geometry, matched)[1]
        far = far_field_points(
            space["fit"],
            count=int(controls_config["far_field_count"]),
            multiplier=float(controls_config["far_field_multiplier"]),
            seed=int(controls_config["far_field_seed"]),
        )
        far_scores = minimum_scores_numpy(far, geometry, matched)[1]

        arm: dict[str, Any] = {
            "transform": spec["transform"],
            "components": spec["components"],
            "component_count": assignment.count,
            "rank": rank,
            "rationale": spec["rationale"],
            "controls": _controls(
                known_scores,
                far_scores,
                floor=float(controls_config["far_field_floor"]),
                tolerance=float(controls_config["known_split_tolerance"]),
            ),
            "acceptance_multiplicity": {
                "mean": float(counts.mean()),
                "median": float(np.median(counts)),
                "accepted_by_none": float(np.mean(counts == 0)),
                "accepted_by_exactly_one": float(np.mean(counts == 1)),
                "accepted_by_all": float(np.mean(counts == class_count)),
            },
            "known_accuracy": float(np.mean(per_class.argmin(axis=1) == report_labels)),
            "rejection_recall": rejection_recall(
                space["unseen"],
                geometry,
                matched,
                domains=unseen_domains,
                domain_count=domain_count,
            ),
            "auroc": domain_auroc(
                known_scores,
                report_domains,
                unseen_scores,
                unseen_domains,
                domain_count=domain_count,
            ),
        }
        arm["gate"] = _clears(arm, gate_config)
        arms[name] = arm
        print(
            f"  {name:<22} comp {assignment.count:>4} rank {rank:>3}"
            f"  mult {arm['acceptance_multiplicity']['mean']:>6.2f}"
            f"  recall {arm['rejection_recall']['rejection_recall']:.5f}"
            f"  AUROC {arm['auroc']['auroc']:.4f}"
            f"  acc {arm['known_accuracy']:.4f}"
            f"  {'ok' if arm['controls']['valid'] else 'INSTRUMENT INVALID'}"
        )

    evidence: dict[str, Any] = {
        "milestone": "M90",
        "component": "representation_remedies",
        "generated_at": datetime.now(UTC).isoformat(),
        "registered_question": config["registered_question"],
        "registration_notes": config["registration_notes"],
        "corpus": config["corpus"],
        "openset": config["openset"],
        "partition": partition_report,
        "coverage": config["coverage"],
        "mixture": {
            **mixture_config,
            "cells_retained": sum(len(sizes) for sizes in cell_sizes),
            "smallest_retained_cell": min(
                (size for sizes in cell_sizes for size in sizes), default=0
            ),
        },
        "controls": controls_config,
        "seeds": seeds,
        "gate": gate_config,
        "arms": arms,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "runtime_seconds": None,
    }
    evidence["verdict"] = _verdict(arms, gate_config)
    evidence["control_defect"] = {
        "found_at": "execution, not registration",
        "control": "known_split negative control, inherited from M85 via N90.8",
        "finding": (
            "The control halves the report rows by position, and those rows are "
            "sorted by class, so the two halves are classes 0-64 and 64-127 "
            "with exactly one class in common. It is a class-block split, not a "
            "random split of comparable rows, and what it measures is whether a "
            "scorer assigns systematically different scores to the lower and "
            "upper halves of the class index. M85's docstring states the "
            "control 'measures the scorer rather than the split'; the domain "
            "mixture of the two halves is indeed matched, but the class "
            "membership is disjoint."
        ),
        "evidence": (
            "All six arms sit below 0.5 on it, from 0.4751 to 0.4860, which is "
            "a one-sided bias rather than the noise a genuine null split would "
            "produce."
        ),
        "consequence": (
            "The control is left exactly as registered and its verdict stands "
            "for this run: domain_mixture is instrument_invalid and its "
            "operands are suppressed, so H90 is undetermined rather than "
            "refuted. Repairing a control after seeing it fail an arm is the "
            "error this program exists to avoid, so the repair is registered "
            "as M90.1 and applied to every arm including the baseline."
        ),
        "does_not_void": (
            "M85's figures stand. Its arms passed this control, and passing a "
            "mis-aimed control does not invalidate a measurement; only the "
            "docstring's claim about what the control demonstrates is wrong."
        ),
    }
    evidence["runtime_seconds"] = round(time.time() - started, 2)
    evidence["evidence_hash"] = payload_hash(
        {
            key: value
            for key, value in evidence.items()
            if key not in ("generated_at", "runtime_seconds")
        }
    )

    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)

    verdict = evidence["verdict"]
    print(f"\nH90 misspecification  {verdict['h90_misspecification']['verdict']}")
    print(f"H91 coordinates       {verdict['h91_coordinates']['verdict']}")
    print(f"M91 opens             {verdict['m91_opens']}")
    print(f"control defect        {evidence['control_defect']['control']}")
    print(f"evidence_hash         {evidence['evidence_hash']}")
    print(f"runtime               {evidence['runtime_seconds'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
