"""V14-M91: does a larger frozen backbone rescue v13's rejection failure?

Plan v14 sections 7 and 7.1. H92 says the acceptance overlap is a capacity limit
of dinov2-small. Three arms -- dinov2-small, -base and -large -- are measured on
byte-identical rows under an identical partition, coverage and rank, with the
backbone as the only factor that changes (N91.1, N91.2, checked upstream by
``prepare_v14_m91_backbones.py``).

Two things are worth stating about how this milestone can fail.

**The reference arm pins the code path.** dinov2-small is re-extracted rather than
copied, and its recomputed geometry must reproduce M84's zero-rung recall and
M85a's AUROC exactly (N91.8). If it does not, this run is describing a different
object and reports nothing -- the M89 N89.3 pattern.

**An arm that shows no added capacity is void, not negative** (N91.5). H92 is a
hypothesis about capacity, so a backbone that fails to beat the reference on
known-class accuracy has not demonstrated any, and whatever it did to rejection
is not evidence about capacity. Reading such an arm as a refutation would let a
quantisation artefact close a hypothesis.

Rank stays 51 at every ambient dimension because the ten-samples-per-dimension
floor binds on fit rows per class, not on the dimension (N91.4).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common.v13_boundary import (
    domain_auroc,
    domain_matched_partition,
    domain_stratified_halves,
    far_field_points,
    minimum_scores_numpy,
    rejection_recall,
)
from experiments.common.v5_artifacts import payload_hash, write_canonical_json
from experiments.tier4.eval_v14_m89_representation_diagnostic import separation_report
from experiments.tier4.eval_v14_m90_1_control_repair import (
    controls_with_repair,
    stratified_halves,
)
from experiments.tier4.eval_v14_m90_2_domain_erasure import (
    domain_probe,
    dominance_ratio,
    erasure_certificate,
    leace_eraser,
)
from experiments.tier4.eval_v14_m90_representation_remedies import (
    _clears,
    class_assignment,
    class_scores,
    fit_components,
    union_coverage_offsets,
)

DEFAULT_CONFIG = Path("experiments/configs/v14/m91_backbone_capacity.json")


def _resolve(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"required artifact is missing: {resolved}")
    return resolved


def load_backbone_split(directory: Path, split: str) -> dict[str, Any]:
    """Load one arm's arrays together with the builder's own identity report.

    The builder discharged N91.2. This refuses to read an arm that did not
    reproduce v13's selection, because an arm on different rows is not
    comparable to any other arm or to any sealed number.
    """
    root = directory / split
    evidence = json.loads(_resolve(root / "evidence.json").read_text(encoding="utf-8"))
    arrays = root / "arrays"
    payload = {
        "features": np.load(_resolve(arrays / "features.npy")).astype(np.float32),
        "labels": np.load(_resolve(arrays / "labels.npy")).astype(np.int64),
        "domains": np.load(_resolve(arrays / "domains.npy")).astype(np.int64),
        "rows_identical": bool(evidence["manifest_identity"]["passes"]),
        "evidence": evidence,
    }
    if split == "openset":
        payload["stratified"] = np.load(_resolve(arrays / "stratified.npy"))
    return payload


def rank_variance_retained(
    fit: np.ndarray, assignment: Assignment, *, rank: int
) -> dict[str, Any]:
    """N91.12. How much within-class variance the fixed rank survives per arm.

    Diagnostic only. Rank is pinned by sample adequacy, not by ambient
    dimension, so a wider representation is necessarily fitted more lossily.
    """
    retained: list[float] = []
    for rows in assignment.memberships:
        block = fit[rows].astype(np.float64)
        if block.shape[0] <= 1:
            continue
        block = block - block.mean(axis=0, keepdims=True)
        spectrum = np.linalg.svd(block, compute_uv=False) ** 2
        total = float(spectrum.sum())
        if total <= 0.0:
            continue
        retained.append(float(spectrum[:rank].sum() / total))
    values = np.asarray(retained, dtype=np.float64)
    return {
        "rank": rank,
        "classes_measured": int(values.size),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "gated": False,
    }


def capacity_check(
    arm: dict[str, Any], reference: dict[str, Any], *, config: dict[str, Any]
) -> dict[str, Any]:
    """N91.5. Has this arm demonstrated that it has more capacity at all?

    The operand is known-class accuracy on the identical evaluation rows. This is
    not a bar on H92 -- clearing it is the precondition for the arm's rejection
    operands being readable as evidence about capacity, not a partial pass.
    """
    accuracy = float(arm["known_accuracy"])
    reference_accuracy = float(reference["known_accuracy"])
    demonstrated = accuracy > reference_accuracy
    return {
        "operand": config["operand"],
        "known_accuracy": accuracy,
        "reference_known_accuracy": reference_accuracy,
        "delta": accuracy - reference_accuracy,
        "capacity_demonstrated": bool(demonstrated),
        "verdict": "capacity_demonstrated"
        if demonstrated
        else "capacity_not_demonstrated",
    }


def _identity(arm: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    """N91.8. Does the re-extracted reference arm reproduce v13 exactly?"""
    tolerance = float(identity["tolerance"])
    recall = float(arm["rejection_recall"]["rejection_recall"])
    auroc = float(arm["auroc"]["auroc"])
    recall_matches = abs(recall - float(identity["m84_zero_rung_recall"])) <= tolerance
    auroc_matches = abs(auroc - float(identity["m85a_geometry_auroc"])) <= tolerance
    return {
        "recomputed_rejection_recall": recall,
        "registered_rejection_recall": float(identity["m84_zero_rung_recall"]),
        "reproduces_m84": bool(recall_matches),
        "recomputed_auroc": auroc,
        "registered_auroc": float(identity["m85a_geometry_auroc"]),
        "reproduces_m85a": bool(auroc_matches),
        "tolerance": tolerance,
        "is_v13_geometry": bool(recall_matches and auroc_matches),
    }


def _verdict(
    arms: dict[str, Any],
    *,
    reference_arm: str,
    gate: dict[str, Any],
    identity: dict[str, Any],
    probe_config: dict[str, Any],
) -> dict[str, Any]:
    """H92's verdict, with every way of not earning one kept distinct.

    Four outcomes are possible and only one of them refutes H92 cleanly:
    the reference arm fails to reproduce v13 (nothing is readable); every
    capacity arm fails its instrument (void); no capacity arm demonstrated
    capacity (untestable); or capacity was demonstrated and the bars were not
    cleared (refuted).
    """
    reference = arms[reference_arm]
    reference_identity = reference["identity"]
    if not reference_identity["is_v13_geometry"]:
        return {
            "reference_reproduces_v13": False,
            "h92_capacity": "not_v13_geometry",
            "reading": (
                "The re-extracted dinov2-small arm does not reproduce M84's "
                "zero-rung recall and M85a's AUROC, so this run is measuring a "
                "different object than v13 thresholded. Every figure below is "
                "suppressed (N91.8)."
            ),
        }

    capacity_arms = {
        name: arm for name, arm in arms.items() if name != reference_arm
    }
    valid = {
        name: arm
        for name, arm in capacity_arms.items()
        if arm["controls"]["valid"]
    }
    demonstrated = {
        name: arm
        for name, arm in valid.items()
        if arm["capacity"]["capacity_demonstrated"]
    }
    cleared = [name for name, arm in demonstrated.items() if arm["gate"]["clears"]]

    probe_valid = all(arm["probe"]["converged"] for arm in arms.values()) and (
        arms[reference_arm]["probe"]["balanced_accuracy"]
        >= float(probe_config["probe_positive_floor"])
    )

    if not valid:
        h92 = "void_instrument"
        reading = (
            "Every capacity arm failed its own control, so none of them refutes "
            "anything. An arm failing its instrument is void, not negative."
        )
    elif not demonstrated:
        h92 = "untestable"
        reading = (
            "No larger backbone beat dinov2-small on known-class accuracy, so no "
            "arm demonstrated added capacity and H92's antecedent was never "
            "satisfied. The rejection operands are recorded and excluded from the "
            "verdict (N91.5). The quantisation-divergence control is the place to "
            "look for a cause."
        )
    elif cleared:
        h92 = "survives"
        reading = (
            "A backbone that demonstrably has more capacity also cleared all "
            f"three bars: {', '.join(sorted(cleared))}. H92 survives."
        )
    else:
        h92 = "refuted"
        reading = (
            "Capacity was demonstrated and rejection did not follow. "
            f"{', '.join(sorted(demonstrated))} beat dinov2-small on known-class "
            "accuracy and still failed the three bars, so the acceptance overlap "
            "is not a capacity limit of dinov2-small."
        )

    return {
        "reference_reproduces_v13": True,
        "arms_with_valid_instrument": sorted(valid),
        "arms_demonstrating_capacity": sorted(demonstrated),
        "arms_clearing_all_bars": sorted(cleared),
        "probe_instrument_valid": bool(probe_valid),
        "h92_capacity": h92,
        "reading": reading,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    started = time.time()

    torch.set_num_threads(1)
    config = json.loads(args.config.read_text(encoding="utf-8"))

    class_count = int(config["corpus"]["class_count"])
    domain_count = int(config["corpus"]["domain_count"])
    partition = config["partition"]
    controls_config = config["controls"]
    separation_config = config["separation"]
    probe_config = config["probe"]
    erasure_config = config["erasure"]
    gate_config = config["gate"]
    rank = int(config["rank"])
    coverage = float(config["coverage"]["known_coverage"])

    arms: dict[str, Any] = {}
    for spec in config["arms"]:
        directory = Path(spec["directory"])
        if not directory.is_dir():
            print(f"  {spec['name']:<14} not extracted; skipped", flush=True)
            continue
        known = load_backbone_split(directory, "known")
        openset = load_backbone_split(directory, "openset")
        if not (known["rows_identical"] and openset["rows_identical"]):
            raise SystemExit(
                f"{spec['name']} was not extracted on v13's rows (N91.2); "
                "no arm on different rows is comparable"
            )

        features = known["features"]
        labels = known["labels"]
        domains = known["domains"]
        if features.shape[1] != int(spec["output_dimension"]):
            raise SystemExit(
                f"{spec['name']} has dimension {features.shape[1]}, "
                f"expected {spec['output_dimension']}"
            )

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

        stratified = openset["stratified"]
        open_features = openset["features"][stratified]
        open_labels = openset["labels"][stratified]
        open_domains = openset["domains"][stratified]
        keep = open_labels >= int(config["openset"]["evaluation_first_label"])
        unseen = open_features[keep]
        unseen_domains = open_domains[keep]

        fit = features[fit_rows]
        fit_labels = labels[fit_rows]
        fit_domains = domains[fit_rows]
        calibration = features[calibration_rows]
        calibration_labels = labels[calibration_rows]
        report = features[report_rows]
        report_labels = labels[report_rows]
        report_domains = domains[report_rows]

        assignment = class_assignment(fit_labels, class_count)
        geometry = fit_components(fit, assignment, rank=rank)
        bottleneck = rank_variance_retained(fit, assignment, rank=rank)
        log_beta = np.concatenate(
            [
                np.log(geometry.tangent_scales),
                np.log(geometry.residual_scales)[:, None],
            ],
            axis=1,
        )
        offsets = union_coverage_offsets(
            calibration,
            calibration_labels,
            geometry,
            log_beta,
            assignment,
            coverage=coverage,
            class_count=class_count,
        )
        matched = log_beta + offsets[assignment.owner][:, None]

        per_class = class_scores(report, geometry, matched, assignment)
        counts = (per_class <= 1.0).sum(axis=1)
        known_scores = minimum_scores_numpy(report, geometry, matched)[1]
        unseen_scores = minimum_scores_numpy(unseen, geometry, matched)[1]
        far = far_field_points(
            fit,
            count=int(controls_config["far_field_count"]),
            multiplier=float(controls_config["far_field_multiplier"]),
            seed=int(controls_config["far_field_seed"]),
        )
        far_scores = minimum_scores_numpy(far, geometry, matched)[1]

        separation = separation_report(
            report,
            report_labels,
            report_domains,
            class_count=class_count,
            domain_count=domain_count,
            minimum_cell_support=int(separation_config["minimum_cell_support"]),
            floor_samples_per_dimension=int(
                separation_config["floor_samples_per_dimension"]
            ),
        )

        # N91.10. Fit LEACE on this arm's own fit rows and re-read the geometry
        # afterwards, so that "domain dominance that survives complete linear
        # erasure" is measured per backbone rather than assumed to transfer.
        eraser, removed_rank = leace_eraser(
            fit,
            fit_domains,
            group_count=domain_count,
            floor=float(erasure_config["eigenvalue_floor"]),
            singular_tolerance=float(erasure_config["singular_value_tolerance"]),
        )
        erased_report = eraser(report)
        certificate = erasure_certificate(
            fit, eraser(fit), fit_domains, group_count=domain_count
        )
        erased_separation = separation_report(
            erased_report,
            report_labels,
            report_domains,
            class_count=class_count,
            domain_count=domain_count,
            minimum_cell_support=int(separation_config["minimum_cell_support"]),
            floor_samples_per_dimension=int(
                separation_config["floor_samples_per_dimension"]
            ),
        )

        midpoint = len(report_rows) // 2
        arm: dict[str, Any] = {
            "backbone": spec["backbone"],
            "role": spec["role"],
            "output_dimension": int(features.shape[1]),
            "rank": rank,
            "rank_variance_retained": bottleneck,
            "partition": partition_report,
            "known_features_sha256": known["evidence"]["features_sha256"],
            "openset_feature_hash": openset["evidence"]["feature_hash"],
            "reproduces_v13_features": bool(
                known["evidence"]["reproduces_v13_features"]
            ),
            "controls": controls_with_repair(
                known_scores,
                far_scores,
                floor=float(controls_config["far_field_floor"]),
                tolerance=float(controls_config["known_split_tolerance"]),
                class_block=(
                    np.arange(midpoint, dtype=np.int64),
                    np.arange(midpoint, len(report_rows), dtype=np.int64),
                ),
                stratified=stratified_halves(
                    report_labels,
                    report_domains,
                    seed=int(controls_config["known_split_seed"]),
                ),
            ),
            "probe": domain_probe(
                fit,
                fit_domains,
                report,
                report_domains,
                max_iterations=int(probe_config["max_iterations"]),
                regularisation=float(probe_config["regularisation_C"]),
                seed=int(probe_config["seed"]),
            ),
            "separation": separation,
            "dominance_ratio": dominance_ratio(separation),
            "leace_diagnostic": {
                "removed_rank": removed_rank,
                "erasure_certificate": certificate,
                "dominance_ratio_after_erasure": dominance_ratio(erased_separation),
                "note": (
                    "N91.10. Diagnostic only; it carries no bar and cannot pass "
                    "or fail an arm."
                ),
            },
            "acceptance_multiplicity": {
                "mean": float(counts.mean()),
                "median": float(np.median(counts)),
                "accepted_by_none": float(np.mean(counts == 0)),
                "accepted_by_exactly_one": float(np.mean(counts == 1)),
                "accepted_by_all": float(np.mean(counts == class_count)),
            },
            "known_accuracy": float(np.mean(per_class.argmin(axis=1) == report_labels)),
            "rejection_recall": rejection_recall(
                unseen,
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
        arms[spec["name"]] = arm

    reference_arm = config["reference_arm"]
    if reference_arm not in arms:
        raise SystemExit(
            f"the reference arm {reference_arm} is not available; without it no "
            "arm can be read against a sealed number (N91.8)"
        )
    arms[reference_arm]["identity"] = _identity(
        arms[reference_arm], config["identity"]
    )
    for name, arm in arms.items():
        arm["capacity"] = capacity_check(
            arm, arms[reference_arm], config=config["capacity"]
        )

    for name, arm in arms.items():
        ratio = arm["dominance_ratio"]
        after = arm["leace_diagnostic"]["dominance_ratio_after_erasure"]
        print(
            f"  {name:<14} dim {arm['output_dimension']:>5}"
            f"  acc {arm['known_accuracy']:.4f}"
            f"  recall {arm['rejection_recall']['rejection_recall']:.5f}"
            f"  AUROC {arm['auroc']['auroc']:.4f}"
            f"  mult {arm['acceptance_multiplicity']['mean']:>6.2f}"
            f"  probe {arm['probe']['balanced_accuracy']:.4f}"
            f"  retained {arm['rank_variance_retained']['mean']:.4f}"
            f"  dominance {'n/a' if ratio is None else f'{ratio:.3f}'}"
            f"->{'n/a' if after is None else f'{after:.3f}'}"
            f"  {arm['capacity']['verdict']}"
            f"  {'ok' if arm['controls']['valid'] else 'INVALID'}"
        )

    verdict = _verdict(
        arms,
        reference_arm=reference_arm,
        gate=gate_config,
        identity=config["identity"],
        probe_config=probe_config,
    )

    evidence: dict[str, Any] = {
        "milestone": "M91",
        "component": "backbone_capacity",
        "generated_at": datetime.now(UTC).isoformat(),
        "registered_question": config["registered_question"],
        "registration_notes": config["registration_notes"],
        "corpus": config["corpus"],
        "openset": config["openset"],
        "coverage": config["coverage"],
        "rank": rank,
        "gate": gate_config,
        "capacity_rule": config["capacity"],
        "controls": controls_config,
        "probe_config": probe_config,
        "separation_config": separation_config,
        "erasure": erasure_config,
        "reference_arm": reference_arm,
        "arms": arms,
        "verdict": verdict,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "runtime_seconds": round(time.time() - started, 1),
    }
    evidence["evidence_hash"] = payload_hash(
        {
            key: value
            for key, value in evidence.items()
            if key not in {"generated_at", "runtime_seconds"}
        }
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)

    print(f"\nreference reproduces v13  {verdict['reference_reproduces_v13']}")
    print(f"H92 capacity              {verdict['h92_capacity']}")
    print(f"evidence_hash             {evidence['evidence_hash']}")
    print(f"runtime                   {evidence['runtime_seconds'] / 60:.1f} min")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
