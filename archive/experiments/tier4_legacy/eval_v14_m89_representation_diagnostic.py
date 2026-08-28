"""M89: what the frozen representation actually looks like under v13's boundary.

v13 established that rejection fails. It never measured why. This run describes
the object M83, M84 and M85 thresholded -- the acceptance multiplicity at their
coverage, the spread of each class against the gaps between classes, and how
much of that spread is visual domain rather than semantic class -- so that
Plan v14's remedies are aimed at a measured cause.

Non-gating by construction (N89.1). It registers no hypothesis, trains nothing,
and re-fits nothing beyond closed-form Phase A on the sealed fit rows. Its only
correctness requirement is object identity (N89.3): it must reproduce M84's
zero-rung recall and M85a's geometry AUROC from its own recomputed geometry, or
it is describing something else and says so.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common.v5_artifacts import payload_hash, write_canonical_json
from experiments.common.v13_boundary import (
    apply_offsets,
    boundary_scores,
    domain_auroc,
    domain_matched_partition,
    domain_stratified_halves,
    fit_geometry,
    matched_coverage_offsets,
    minimum_scores_numpy,
    rejection_recall,
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v14" / "m89_representation_diagnostic.json"
)

DOMAIN_NAMES = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")

#: Percentiles reported for every distribution, so that a median is never the
#: only thing on record about a quantity whose tail is the interesting part.
PERCENTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)


def _distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "percentiles": {
            f"p{int(p)}": float(np.percentile(array, p)) for p in PERCENTILES
        },
    }


def acceptance_multiplicity(
    features: np.ndarray,
    geometry: Any,
    log_beta: np.ndarray,
    *,
    class_count: int,
    labels: np.ndarray,
    chunk_rows: int = 512,
) -> dict[str, Any]:
    """How many of the class regions accept each row, at matched coverage.

    Rejection is a conjunction over all classes, so this count is the direct
    determinant of whether anything can be rejected at all: a row interior to
    two or more regions cannot be rejected no matter what either region does.
    """
    beta = torch.as_tensor(log_beta, dtype=torch.float64)
    counts = np.empty(len(features), dtype=np.int64)
    argmin_correct = np.empty(len(features), dtype=bool)
    with torch.no_grad():
        for start in range(0, len(features), chunk_rows):
            stop = min(start + chunk_rows, len(features))
            block = torch.as_tensor(features[start:stop], dtype=torch.float64)
            scores = boundary_scores(block, geometry, beta).cpu().numpy()
            counts[start:stop] = (scores <= 1.0).sum(axis=1)
            argmin_correct[start:stop] = scores.argmin(axis=1) == labels[start:stop]
    return {
        "distribution": _distribution(counts),
        "class_count": int(class_count),
        "accepted_by_none": float(np.mean(counts == 0)),
        "accepted_by_exactly_one": float(np.mean(counts == 1)),
        "accepted_by_more_than_one": float(np.mean(counts > 1)),
        "accepted_by_all": float(np.mean(counts == class_count)),
        "argmin_score_accuracy": float(argmin_correct.mean()),
        "chance_accuracy": 1.0 / float(class_count),
    }


def _nearest_centroid_accuracy(
    features: np.ndarray, centers: np.ndarray, labels: np.ndarray, *, chunk: int = 512
) -> float:
    correct = 0
    for start in range(0, len(features), chunk):
        block = features[start : start + chunk]
        distances = np.linalg.norm(block[:, None, :] - centers[None, :, :], axis=2)
        correct += int(np.sum(distances.argmin(axis=1) == labels[start : start + chunk]))
    return float(correct) / float(len(features))


def separation_report(
    features: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    *,
    class_count: int,
    domain_count: int,
    minimum_cell_support: int,
    floor_samples_per_dimension: int,
) -> dict[str, Any]:
    """Spread within a group against the gaps between groups, at two granularities.

    The class-level ratio is what v13's one-region-per-class model assumes it can
    work with. The cell-level ratio is what it would have had if the model were
    not pooling six visual domains into one region.
    """
    class_centers = np.stack(
        [features[labels == k].mean(axis=0) for k in range(class_count)]
    )
    own_class = np.linalg.norm(features - class_centers[labels], axis=1)
    class_gaps = np.linalg.norm(
        class_centers[:, None, :] - class_centers[None, :, :], axis=2
    )
    np.fill_diagonal(class_gaps, np.inf)
    nearest_class_gap = class_gaps.min(axis=1)

    cell_keys: list[tuple[int, int]] = []
    cell_centers: list[np.ndarray] = []
    cell_support: list[int] = []
    own_cell = np.full(len(features), np.nan, dtype=np.float64)
    for k in range(class_count):
        for d in range(domain_count):
            mask = (labels == k) & (domains == d)
            support = int(mask.sum())
            if support < 2:
                continue
            center = features[mask].mean(axis=0)
            own_cell[mask] = np.linalg.norm(features[mask] - center, axis=1)
            cell_keys.append((k, d))
            cell_centers.append(center)
            cell_support.append(support)

    centers_array = np.stack(cell_centers)
    cell_class = np.array([k for k, _ in cell_keys])
    cell_gaps = np.linalg.norm(
        centers_array[:, None, :] - centers_array[None, :, :], axis=2
    )
    same_class = cell_class[:, None] == cell_class[None, :]
    foreign = cell_gaps.copy()
    foreign[same_class] = np.inf
    nearest_foreign_cell = foreign.min(axis=1)

    sibling_gaps: list[np.ndarray] = []
    for k in range(class_count):
        indices = np.flatnonzero(cell_class == k)
        if len(indices) > 1:
            block = cell_gaps[np.ix_(indices, indices)]
            sibling_gaps.append(block[np.triu_indices(len(indices), k=1)])
    # Undefined rather than zero when no class spans two domains: with one cell
    # per class the cell and class granularities are the same measurement.
    sibling_median = (
        float(np.median(np.concatenate(sibling_gaps))) if sibling_gaps else None
    )

    measured = ~np.isnan(own_cell)
    class_spread = float(np.median(own_class))
    class_gap = float(np.median(nearest_class_gap))
    cell_spread = float(np.median(own_cell[measured]))
    cell_gap = float(np.median(nearest_foreign_cell))

    retained = [support for support in cell_support if support >= minimum_cell_support]
    permitted_rank = (
        int(min(retained) // floor_samples_per_dimension) if retained else 0
    )

    return {
        "class_level": {
            "within_spread": _distribution(own_class),
            "nearest_gap": _distribution(nearest_class_gap),
            "spread_over_separation": class_spread / class_gap,
        },
        "cell_level": {
            "cells_populated": len(cell_keys),
            "within_spread": _distribution(own_cell[measured]),
            "nearest_foreign_class_gap": _distribution(nearest_foreign_cell),
            "spread_over_separation": cell_spread / cell_gap,
        },
        "domain_dominates_class": {
            "own_class_sibling_cell_gap_median": sibling_median,
            "nearest_foreign_class_cell_gap_median": cell_gap,
            "sibling_exceeds_foreign": (
                bool(sibling_median > cell_gap) if sibling_median is not None else None
            ),
            "reading": (
                "When the median gap between a class's own domain cells exceeds "
                "the median gap to the nearest foreign class cell, visual domain "
                "is a larger axis of variation than semantic label, and a single "
                "region per class is fitted across that larger axis."
            ),
        },
        "per_domain_spread": {
            DOMAIN_NAMES[d]: {
                "row_count": int((domains == d).sum()),
                "median_distance_to_class_centroid": (
                    float(np.median(own_class[domains == d]))
                    if int((domains == d).sum()) > 0
                    else None
                ),
            }
            for d in range(domain_count)
        },
        "mixture_probe": {
            "minimum_cell_support": int(minimum_cell_support),
            "cells_retained": len(retained),
            "smallest_retained_support": int(min(retained)) if retained else 0,
            "permitted_common_rank": permitted_rank,
            "v13_rank": 51,
            "rationale": (
                "The rank M90's mixture arm may use if every retained component "
                "is to keep the standing floor of "
                f"{floor_samples_per_dimension} samples per fitted dimension. "
                "More components cost rank; the trade is registered before "
                "M90's operands are seen, and its null holds both constant."
            ),
        },
    }


def corpus_composition(
    domains: np.ndarray, *, domain_count: int, label: str
) -> dict[str, Any]:
    total = int(len(domains))
    return {
        "split": label,
        "row_count": total,
        "by_domain": {
            DOMAIN_NAMES[d]: {
                "row_count": int((domains == d).sum()),
                "fraction": float((domains == d).sum()) / float(total) if total else 0.0,
            }
            for d in range(domain_count)
        },
    }


def _identity(evidence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """N89.3. Object identity is the only thing this run can get wrong."""
    spec = config["identity"]
    tolerance = float(spec["tolerance"])
    recall_delta = abs(
        evidence["rejection_recall_at_matched_coverage"]["rejection_recall"]
        - float(spec["m84_zero_rung_recall"])
    )
    auroc_delta = abs(
        evidence["geometry_auroc"]["auroc"] - float(spec["m85a_geometry_auroc"])
    )
    passes = bool(recall_delta <= tolerance and auroc_delta <= tolerance)
    return {
        "m84_zero_rung_recall_delta": recall_delta,
        "m85a_geometry_auroc_delta": auroc_delta,
        "tolerance": tolerance,
        "verdict": "v13_geometry" if passes else "not_v13_geometry",
        "passes": passes,
        "reason": (
            "Both sealed quantities reproduced from recomputed geometry."
            if passes
            else (
                "The recomputed geometry does not reproduce v13's sealed "
                "quantities, so nothing in this file describes the object M83, "
                "M84 and M85 measured."
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
        quota=tuple(int(value) for value in partition["evaluation_domain_quota"]),
        fit_per_class=int(partition["fit_per_class"]),
        domain_count=domain_count,
    )
    calibration_rows, report_rows = domain_stratified_halves(
        labels, domains, evaluation_rows
    )
    geometry = fit_geometry(
        features[fit_rows],
        labels[fit_rows],
        rank=int(config["geometry"]["rank"]),
        class_count=class_count,
    )

    openset_index = _verify_sealed(config["openset"], "open-set")
    open_features, open_labels, open_domains = _load_arrays(
        openset_index, stratified_only=bool(config["openset"]["stratified_only"])
    )
    keep = open_labels >= int(config["openset"]["evaluation_first_label"])
    unseen_features = open_features[keep]
    unseen_domains = open_domains[keep]

    log_beta = np.concatenate(
        [
            np.log(geometry.tangent_scales),
            np.log(geometry.residual_scales)[:, None],
        ],
        axis=1,
    )
    offsets = matched_coverage_offsets(
        features[calibration_rows],
        labels[calibration_rows],
        geometry,
        log_beta,
        coverage=float(config["coverage"]["known_coverage"]),
        class_count=class_count,
    )
    matched = apply_offsets(log_beta, offsets)

    known_features = features[report_rows]
    known_labels = labels[report_rows]
    known_domains = domains[report_rows]

    multiplicity = acceptance_multiplicity(
        known_features,
        geometry,
        matched,
        class_count=class_count,
        labels=known_labels,
    )
    multiplicity["nearest_centroid_accuracy"] = _nearest_centroid_accuracy(
        known_features, geometry.centers, known_labels
    )

    recall = rejection_recall(
        unseen_features,
        geometry,
        matched,
        domains=unseen_domains,
        domain_count=domain_count,
    )
    known_scores = minimum_scores_numpy(known_features, geometry, matched)[1]
    unseen_scores = minimum_scores_numpy(unseen_features, geometry, matched)[1]
    auroc = domain_auroc(
        known_scores,
        known_domains,
        unseen_scores,
        unseen_domains,
        domain_count=domain_count,
    )

    probe = config["mixture_probe"]
    separation = separation_report(
        features[fit_rows],
        labels[fit_rows],
        domains[fit_rows],
        class_count=class_count,
        domain_count=domain_count,
        minimum_cell_support=int(probe["minimum_cell_support"]),
        floor_samples_per_dimension=int(probe["floor_samples_per_dimension"]),
    )

    norms = np.linalg.norm(features[fit_rows], axis=1)

    evidence: dict[str, Any] = {
        "milestone": "M89",
        "component": "representation_diagnostic",
        "generated_at": datetime.now(UTC).isoformat(),
        "registered_question": config["registered_question"],
        "registration_notes": config["registration_notes"],
        "gating": False,
        "corpus": config["corpus"],
        "openset": config["openset"],
        "partition": partition_report,
        "geometry": config["geometry"],
        "coverage": config["coverage"],
        "acceptance_multiplicity": multiplicity,
        "separation": separation,
        "feature_norm": _distribution(norms),
        "composition": {
            "known": corpus_composition(
                domains, domain_count=domain_count, label="known_corpus"
            ),
            "known_fit": corpus_composition(
                domains[fit_rows], domain_count=domain_count, label="known_fit_rows"
            ),
            "open_set": corpus_composition(
                unseen_domains, domain_count=domain_count, label="open_set_evaluation"
            ),
            "note": config["registration_notes"][4],
        },
        "rejection_recall_at_matched_coverage": recall,
        "geometry_auroc": auroc,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "runtime_seconds": None,
    }
    evidence["identity"] = _identity(evidence, config)
    evidence["runtime_seconds"] = round(time.time() - started, 2)
    # Excludes the two wall-clock fields so the hash is itself the replay
    # check, rather than needing a comparator that knows what to ignore.
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

    identity = evidence["identity"]
    print(f"identity            {identity['verdict']}")
    if not identity["passes"]:
        print(f"  {identity['reason']}")
        print(f"evidence_hash       {evidence['evidence_hash']}")
        return 1

    dist = multiplicity["distribution"]
    print(
        f"acceptance          mean {dist['mean']:.2f} of {class_count}"
        f"  median {dist['percentiles']['p50']:.0f}"
    )
    print(
        f"  none {multiplicity['accepted_by_none']:.4f}"
        f"  exactly one {multiplicity['accepted_by_exactly_one']:.4f}"
        f"  all {multiplicity['accepted_by_all']:.4f}"
    )
    print(
        f"spread/separation   class {separation['class_level']['spread_over_separation']:.3f}"
        f"  cell {separation['cell_level']['spread_over_separation']:.3f}"
    )
    dominance = separation["domain_dominates_class"]
    print(
        f"domain dominance    sibling cell gap "
        f"{dominance['own_class_sibling_cell_gap_median']:.3f}"
        f"  vs foreign class cell {dominance['nearest_foreign_class_cell_gap_median']:.3f}"
        f"  {'DOMAIN DOMINATES' if dominance['sibling_exceeds_foreign'] else 'class dominates'}"
    )
    mixture = separation["mixture_probe"]
    print(
        f"mixture probe       {mixture['cells_retained']} cells at support "
        f">= {mixture['minimum_cell_support']}"
        f"  permitted rank {mixture['permitted_common_rank']} (v13 used {mixture['v13_rank']})"
    )
    print(
        f"reproduced          recall "
        f"{recall['rejection_recall']:.5f}  AUROC {auroc['auroc']:.6f}"
    )
    print(f"evidence_hash       {evidence['evidence_hash']}")
    print(f"runtime             {evidence['runtime_seconds'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
