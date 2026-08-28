"""M85: the threshold-free operand the open-set leg registered and never reported.

`ACCEPTANCE_CRITERIA_v13.md` makes AUROC a **gating** bar for L2 and states that
a bare recall percentage is not admissible. No v13 evidence file contains one.
M83 and M84 both reported rejection recall at matched coverage and stopped
there, which is the very habit the criterion calls out in as many words.

This run completes their reporting without re-opening their verdicts. It trains
nothing. Phase A is closed-form on the same fit rows at the same rank under the
same domain-quota partition, so the ranked object is the object those milestones
thresholded, and the recall printed here reproduces M84's zero rung exactly.

Both ends of the instrument are validated before any figure is read (N85.4d),
and every arm is reported beside free baselines on identical rows (N85.4c).
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
    domain_auroc,
    far_field_points,
    fit_geometry,
    domain_matched_partition,
    domain_stratified_halves,
    matched_coverage_offsets,
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v13" / "m85_open_set_auroc.json"
)


def _nearest_center_scores(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Free baseline: distance to the nearest class centre.

    This uses the geometry's own centres and none of what Phase A or Phase B
    fitted on top of them. It is the cheapest thing that could possibly work,
    and it is the number the ellipsoids have to beat to have earned their rank.
    """
    scores = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), 512):
        block = features[start : start + 512]
        deltas = block[:, None, :] - centers[None, :, :]
        scores[start : start + 512] = np.linalg.norm(deltas, axis=2).min(axis=1)
    return scores


def _knn_scores(
    features: np.ndarray, reference: np.ndarray, *, k: int, chunk_rows: int
) -> np.ndarray:
    """Free baseline: distance to the k-th nearest frozen fit row.

    Squared distances via the inner-product expansion, which is what makes a
    5760x65536 comparison affordable; the square root is taken once at the end
    so the reported unit is a distance.
    """
    reference32 = np.ascontiguousarray(reference, dtype=np.float32)
    reference_norms = (reference32**2).sum(axis=1)
    scores = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), chunk_rows):
        block = np.ascontiguousarray(
            features[start : start + chunk_rows], dtype=np.float32
        )
        squared = (
            (block**2).sum(axis=1)[:, None]
            - 2.0 * (block @ reference32.T)
            + reference_norms[None, :]
        )
        kth = np.partition(squared, k - 1, axis=1)[:, k - 1]
        scores[start : start + chunk_rows] = np.sqrt(np.maximum(kth, 0.0))
    return scores


def _controls(
    known: np.ndarray,
    far_field: np.ndarray,
    *,
    floor: float,
    tolerance: float,
) -> dict[str, Any]:
    """Validate the instrument at both ends before reading anything from it.

    The negative control halves the knowns by position after the rows have
    already been drawn in a domain-stratified order, so the two halves share a
    domain mixture and the control measures the scorer rather than the split.
    """
    positive = score_auroc(known, far_field)
    midpoint = len(known) // 2
    negative = score_auroc(known[:midpoint], known[midpoint:])
    return {
        "positive_control": positive,
        "positive_floor": floor,
        "positive_passes": bool(positive >= floor),
        "negative_control": negative,
        "negative_tolerance": tolerance,
        "negative_passes": bool(abs(negative - 0.5) <= tolerance),
    }


def _gate(evidence: dict[str, Any]) -> dict[str, Any]:
    """Object identity first, then instrument validity, then the free bars.

    The identity clause runs first and is not a formality. The first draft of
    this runner initialised ``log_beta`` at zeros instead of at the log of the
    fitted scales, which coverage-matching then turned into a plausible-looking
    boundary that was a unit sphere per class rather than M84's ellipsoid. It
    produced a full set of AUROCs with both controls passing. The only thing
    that caught it was the recall failing to reproduce M84's registered zero
    rung, so that reproduction is gating here rather than printed.
    """
    reproduction = evidence["m84_reproduction"]
    if not reproduction["passes"]:
        return {
            "verdict": "not_m84_boundary",
            "reason": (
                "rejection recall at matched coverage is "
                f"{reproduction['measured']} against M84's registered "
                f"{reproduction['registered']}. The object being ranked is not "
                "the object M84 thresholded, so no figure below is about the "
                "open-set leg."
            ),
            "supports_threshold_free_bar": False,
        }

    arms = evidence["arms"]
    broken = [
        name
        for name, arm in arms.items()
        if not (arm["controls"]["positive_passes"] and arm["controls"]["negative_passes"])
    ]
    if broken:
        return {
            "verdict": "instrument_invalid",
            "reason": (
                "these scorers failed a control and every figure below them is "
                f"suppressed: {sorted(broken)}"
            ),
            "supports_threshold_free_bar": False,
        }

    geometry = arms["geometry"]["auroc"]
    baselines = {
        name: arm["auroc"] for name, arm in arms.items() if name != "geometry"
    }
    strongest = max(baselines, key=lambda name: baselines[name])
    margin = geometry - baselines[strongest]
    decisive = float(evidence["reporting"]["decisive_margin"])

    if margin > decisive:
        verdict = "geometry_ranks_above_free_baselines"
    elif margin < -decisive:
        verdict = "free_baseline_ranks_above_geometry"
    else:
        verdict = "geometry_ties_free_baselines"
    return {
        "verdict": verdict,
        "geometry_auroc": geometry,
        "strongest_baseline": strongest,
        "strongest_baseline_auroc": baselines[strongest],
        "margin": margin,
        "decisive_margin": decisive,
        "supports_threshold_free_bar": bool(
            verdict == "geometry_ranks_above_free_baselines"
        ),
        "reason": (
            "AUROC is threshold-free, so it cannot overturn a threshold result "
            "(N85.4a). It completes M83's and M84's reporting; it does not "
            "reopen their verdicts."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    started = time.time()
    torch.set_num_threads(1)

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
    known_domains = domains[report_rows]
    controls = config["controls"]
    far_field = far_field_points(
        features[fit_rows],
        count=int(controls["far_field_count"]),
        multiplier=float(controls["far_field_multiplier"]),
        seed=int(controls["far_field_seed"]),
    )

    baselines = config["baselines"]
    scorers: dict[str, Any] = {
        "geometry": lambda rows: minimum_scores_numpy(rows, geometry, matched)[1],
        "nearest_center": lambda rows: _nearest_center_scores(rows, geometry.centers),
        "knn": lambda rows: _knn_scores(
            rows,
            features[fit_rows],
            k=int(baselines["knn_k"]),
            chunk_rows=int(baselines["chunk_rows"]),
        ),
    }

    arms: dict[str, Any] = {}
    for name, scorer in scorers.items():
        known_scores = scorer(known_features)
        unseen_scores = scorer(unseen_features)
        report = domain_auroc(
            known_scores,
            known_domains,
            unseen_scores,
            unseen_domains,
            domain_count=domain_count,
        )
        report["controls"] = _controls(
            known_scores,
            scorer(far_field),
            floor=float(controls["far_field_floor"]),
            tolerance=float(controls["known_split_tolerance"]),
        )
        report["known_score_median"] = float(np.median(known_scores))
        report["unseen_score_median"] = float(np.median(unseen_scores))
        arms[name] = report
        print(
            f"  {name:<15} AUROC {report['auroc']:.4f}"
            f"  far-field {report['controls']['positive_control']:.4f}"
            f"  known-split {report['controls']['negative_control']:.4f}"
        )

    recall = rejection_recall(
        unseen_features,
        geometry,
        matched,
        domains=unseen_domains,
        domain_count=domain_count,
    )
    registered = float(config["reporting"]["m84_zero_rung_recall"])
    measured = float(recall["rejection_recall"])
    reproduction = {
        "registered": registered,
        "measured": measured,
        "tolerance": 1e-9,
        "passes": bool(abs(measured - registered) <= 1e-9),
        "rationale": (
            "N84.4's zero rung, recomputed from the sealed corpus. This pins "
            "the ranked object to M84's thresholded object."
        ),
    }

    evidence: dict[str, Any] = {
        "milestone": "M85",
        "component": "open_set_auroc",
        "generated_at": datetime.now(UTC).isoformat(),
        "registered_question": config["registered_question"],
        "registration_notes": config["registration_notes"],
        "corpus": config["corpus"],
        "openset": config["openset"],
        "partition": partition_report,
        "geometry": config["geometry"],
        "coverage": config["coverage"],
        "baselines": baselines,
        "controls": controls,
        "reporting": config["reporting"],
        "arms": arms,
        "rejection_recall_at_matched_coverage": recall,
        "m84_reproduction": reproduction,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "runtime_seconds": None,
    }
    evidence["gate"] = _gate(evidence)
    evidence["runtime_seconds"] = round(time.time() - started, 2)
    evidence["evidence_hash"] = payload_hash(
        {key: value for key, value in evidence.items() if key != "generated_at"}
    )

    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)

    gate = evidence["gate"]
    print(f"\nverdict         {gate['verdict']}")
    print(
        f"M84 zero rung   {measured:.5f} vs registered {registered:.5f}"
        f"  {'reproduced' if reproduction['passes'] else 'MISMATCH'}"
    )
    print(f"evidence_hash   {evidence['evidence_hash']}")
    print(f"runtime         {evidence['runtime_seconds'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
