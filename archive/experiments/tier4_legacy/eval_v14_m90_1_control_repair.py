"""M90.1: repair of the known-split negative control, and the retrial of H90.

M90's mixture arm failed its own negative control at 0.47506 against a 0.48
floor, so its operands were suppressed and H90 was left undetermined rather than
refuted. The control turned out to be defective, and the defect was inherited
from M85: it halves the report rows by position, but those rows are class-sorted,
so the halves are classes 0-64 and 64-127 with exactly one class in common. It is
a class-block split, and what it measures is whether a scorer scores the low and
high halves of the class index differently. All six M90 arms sat between 0.4751
and 0.4860 on it -- one-sided bias, not null noise.

This run repairs the split: a random partition of the report rows stratified
jointly by class and by domain, so the halves are exchangeable in both factors.
The repair is applied to every arm including the baseline (N90.1.2), the old
control is still reported beside it (N90.1.6), and every arm operand is checked
against M90's sealed evidence, because only the control changed and anything
that moves is a defect rather than a finding (N90.1.3).

Nothing here is re-tuned. The arms, ranks, seeds, partition, coverage and gate
are byte-for-byte the ones M90 registered.
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
from experiments.tier4.eval_v14_m90_representation_remedies import (
    Assignment,
    _clears,
    _group_means,
    _l2_normalise,
    _random_groups,
    _verdict,
    class_assignment,
    class_scores,
    domain_cell_assignment,
    fit_components,
    random_cell_assignment,
    union_coverage_offsets,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v14" / "m90_1_control_repair.json"
)

_OPERANDS = (
    ("acceptance_multiplicity", "mean"),
    ("rejection_recall", "rejection_recall"),
    ("auroc", "auroc"),
)


def stratified_halves(
    labels: np.ndarray, domains: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """A random split of the rows that is balanced within every (class, domain) cell.

    N90.1.1. Splitting inside each cell rather than over the whole set is what
    makes the two halves exchangeable in both factors at once; a split stratified
    on class alone would still be free to skew the domain mixture, and the M90
    defect is precisely a split that balanced one factor while destroying the
    other.
    """
    rng = np.random.default_rng(seed)
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    keys = labels.astype(np.int64) * (int(domains.max()) + 1) + domains.astype(np.int64)
    for key in np.unique(keys):
        rows = rng.permutation(np.flatnonzero(keys == key))
        cut = len(rows) // 2
        # The odd row alternates sides by cell so neither half is favoured.
        if len(rows) % 2 and rng.random() < 0.5:
            cut += 1
        left.append(rows[:cut])
        right.append(rows[cut:])
    return np.sort(np.concatenate(left)), np.sort(np.concatenate(right))


def split_profile(
    labels: np.ndarray, domains: np.ndarray, rows: np.ndarray, *, domain_count: int
) -> dict[str, Any]:
    """What a split actually partitions, recorded so it can be checked."""
    return {
        "rows": int(len(rows)),
        "distinct_classes": int(len(np.unique(labels[rows]))),
        "domain_mixture": [
            int((domains[rows] == d).sum()) for d in range(domain_count)
        ],
    }


def controls_with_repair(
    known: np.ndarray,
    far: np.ndarray,
    *,
    floor: float,
    tolerance: float,
    class_block: tuple[np.ndarray, np.ndarray],
    stratified: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """N90.1.1 and N90.1.6. Both splits are reported; only the repaired one decides."""
    positive = score_auroc(known, far)
    block = score_auroc(known[class_block[0]], known[class_block[1]])
    repaired = score_auroc(known[stratified[0]], known[stratified[1]])
    return {
        "positive_control": positive,
        "positive_floor": floor,
        "positive_passes": bool(positive >= floor),
        "negative_control_class_block": block,
        "negative_control_class_block_passes": bool(abs(block - 0.5) <= tolerance),
        "negative_control_stratified": repaired,
        "negative_tolerance": tolerance,
        "negative_passes": bool(abs(repaired - 0.5) <= tolerance),
        "valid": bool(positive >= floor and abs(repaired - 0.5) <= tolerance),
        "decided_on": "negative_control_stratified",
    }


def replication_report(
    arms: dict[str, Any], reference: dict[str, Any], *, tolerance: float
) -> dict[str, Any]:
    """N90.1.3. Only the control changed, so every operand must be M90's."""
    deltas: dict[str, dict[str, float]] = {}
    for name, arm in arms.items():
        previous = reference["arms"][name]
        deltas[name] = {
            f"{block}.{key}": float(arm[block][key] - previous[block][key])
            for block, key in _OPERANDS
        }
        deltas[name]["known_accuracy"] = float(
            arm["known_accuracy"] - previous["known_accuracy"]
        )
        deltas[name]["positive_control"] = float(
            arm["controls"]["positive_control"]
            - previous["controls"]["positive_control"]
        )
        deltas[name]["negative_control_class_block"] = float(
            arm["controls"]["negative_control_class_block"]
            - previous["controls"]["negative_control"]
        )
    worst = max(
        (abs(v) for arm in deltas.values() for v in arm.values()), default=0.0
    )
    return {
        "reference_hash": reference["evidence_hash"],
        "tolerance": tolerance,
        "largest_absolute_delta": worst,
        "operands_reproduce": bool(worst <= tolerance),
        "deltas": deltas,
        "reading": (
            "Every arm operand reproduces M90 exactly, so the only difference "
            "between the two milestones is the control."
            if worst <= tolerance
            else "An operand moved when only the control should have changed. "
            "This is a defect to diagnose, not a finding (N90.1.3)."
        ),
    }


def _h90_reading(arms: dict[str, Any]) -> dict[str, Any]:
    """N90.1.4. The two ways this milestone can end, distinguished in advance."""
    mixture = arms["domain_mixture"]
    if not mixture["controls"]["valid"]:
        return {
            "outcome": "mixture_unmeasurable",
            "detail": (
                "The mixture arm fails the repaired control as well, so the "
                "design is unmeasurable at this budget. H90 stays undetermined "
                "and is not converted into a negative."
            ),
        }
    return {
        "outcome": "mixture_measurable",
        "detail": (
            "The mixture arm passes the repaired control, so its operands are "
            "readable and H90 is decided on them."
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

    controls_config = config["controls"]
    midpoint = len(report_rows) // 2
    class_block_split = (
        np.arange(midpoint, dtype=np.int64),
        np.arange(midpoint, len(report_rows), dtype=np.int64),
    )
    stratified_split = stratified_halves(
        report_labels, report_domains, seed=int(controls_config["known_split_seed"])
    )
    split_report = {
        "class_block": {
            "definition": "known[:midpoint] against known[midpoint:], M85's split",
            "left": split_profile(
                report_labels, report_domains, class_block_split[0], domain_count=domain_count
            ),
            "right": split_profile(
                report_labels, report_domains, class_block_split[1], domain_count=domain_count
            ),
            "classes_in_common": int(
                len(
                    np.intersect1d(
                        report_labels[class_block_split[0]],
                        report_labels[class_block_split[1]],
                    )
                )
            ),
        },
        "stratified": {
            "definition": "random halves drawn inside every (class, domain) cell",
            "seed": int(controls_config["known_split_seed"]),
            "left": split_profile(
                report_labels, report_domains, stratified_split[0], domain_count=domain_count
            ),
            "right": split_profile(
                report_labels, report_domains, stratified_split[1], domain_count=domain_count
            ),
            "classes_in_common": int(
                len(
                    np.intersect1d(
                        report_labels[stratified_split[0]],
                        report_labels[stratified_split[1]],
                    )
                )
            ),
        },
    }

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
            "controls": controls_with_repair(
                known_scores,
                far_scores,
                floor=float(controls_config["far_field_floor"]),
                tolerance=float(controls_config["known_split_tolerance"]),
                class_block=class_block_split,
                stratified=stratified_split,
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
        controls = arm["controls"]
        print(
            f"  {name:<22} comp {assignment.count:>4} rank {rank:>3}"
            f"  mult {arm['acceptance_multiplicity']['mean']:>6.2f}"
            f"  recall {arm['rejection_recall']['rejection_recall']:.5f}"
            f"  AUROC {arm['auroc']['auroc']:.4f}"
            f"  block {controls['negative_control_class_block']:.5f}"
            f"  strat {controls['negative_control_stratified']:.5f}"
            f"  {'ok' if controls['valid'] else 'INSTRUMENT INVALID'}"
        )

    replication_config = config["replication"]
    reference = json.loads(
        _resolve(replication_config["reference_evidence"]).read_text(encoding="utf-8")
    )
    if reference["evidence_hash"] != replication_config["reference_hash"]:
        raise SystemExit(
            "M90 reference evidence does not match its registered hash: "
            f"{reference['evidence_hash']}"
        )

    evidence: dict[str, Any] = {
        "milestone": "M90.1",
        "component": "control_repair",
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
        "known_splits": split_report,
        "seeds": seeds,
        "gate": gate_config,
        "arms": arms,
        "replication": replication_report(
            arms, reference, tolerance=float(replication_config["operand_tolerance"])
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "runtime_seconds": None,
    }
    evidence["verdict"] = _verdict(arms, gate_config)
    evidence["h90"] = _h90_reading(arms)
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

    block = split_report["class_block"]
    strat = split_report["stratified"]
    print(
        f"\nclass-block split     {block['left']['distinct_classes']} vs "
        f"{block['right']['distinct_classes']} classes, "
        f"{block['classes_in_common']} in common"
    )
    print(
        f"stratified split      {strat['left']['distinct_classes']} vs "
        f"{strat['right']['distinct_classes']} classes, "
        f"{strat['classes_in_common']} in common"
    )
    print(f"operands reproduce    {evidence['replication']['operands_reproduce']}")
    verdict = evidence["verdict"]
    print(f"H90 misspecification  {verdict['h90_misspecification']['verdict']}")
    print(f"H91 coordinates       {verdict['h91_coordinates']['verdict']}")
    print(f"M91 opens             {verdict['m91_opens']}")
    print(f"evidence_hash         {evidence['evidence_hash']}")
    print(f"runtime               {evidence['runtime_seconds'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
