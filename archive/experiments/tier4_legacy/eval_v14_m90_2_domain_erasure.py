"""M90.2: is domain linearly encoded, can it be erased, and does erasing it help?

M90 and M90.1 answered "does rejection improve" and never answered "did domain
overlap improve". Every figure they read was a downstream operand. M89's
geometric measure of domain dominance -- the median gap between a class's own
domain cells (32.295) against the gap to the nearest foreign class cell (19.153)
-- was never recomputed under any transform, so it is on the record that domain
centring did not help rejection and it is not on the record whether it reduced
domain dominance at all.

This run measures the geometry directly, per arm, and attacks it with methods
taken from the literature rather than invented here: closed-form concept erasure
(LEACE, arXiv:2306.03819), top-direction removal (All-but-the-Top,
arXiv:1702.01417) and pooled whitening as a bound (RobustNet's caution,
arXiv:2103.15597). Each carries a null that removes the same number of
directions with no domain information, because removing any five directions from
a 384-dimensional space perturbs every figure a little.

H94 (domain is linearly encoded and erasable) and H95 (erasing it helps
rejection) are judged separately and are permitted to dissociate. That
dissociation is registered in advance as the interesting outcome: it would mean
domain dominance is real, removable, and not the cause of the rejection failure.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from experiments.common.v5_artifacts import payload_hash, write_canonical_json
from experiments.common.v13_boundary import (
    domain_auroc,
    domain_matched_partition,
    domain_stratified_halves,
    far_field_points,
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
from experiments.tier4.eval_v14_m89_representation_diagnostic import separation_report
from experiments.tier4.eval_v14_m90_representation_remedies import (
    _clears,
    _random_groups,
    class_assignment,
    class_scores,
    fit_components,
    union_coverage_offsets,
)
from experiments.tier4.eval_v14_m90_1_control_repair import (
    controls_with_repair,
    stratified_halves,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v14" / "m90_2_domain_erasure.json"
)


class AffineMap:
    """A fixed map fitted on the fit rows and applied unchanged everywhere else.

    Every arm here must be deployable without a domain label at evaluation time,
    which is the objection N90.6 raised against M90's domain-centred arm. Holding
    all six arms to one affine form is what makes that guarantee structural
    rather than something to be checked per arm.

    The map is held in float64 and applied in float64 (N90.2.16). The corpus is
    float32, and a 384-square inverse square root taken in float32 degrades
    LEACE's exact closed-form guarantee by eight orders of magnitude. The result
    is returned in the caller's dtype so that downstream geometry sees the same
    type the untransformed baseline does.
    """

    def __init__(self, matrix: np.ndarray, offset: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.offset = np.asarray(offset, dtype=np.float64)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        promoted = np.asarray(features, dtype=np.float64)
        mapped = promoted @ self.matrix.T + self.offset
        return mapped.astype(features.dtype, copy=False)


def _symmetric_powers(
    covariance: np.ndarray, *, floor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse and forward square roots of a PSD matrix, sharing one eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    clipped = np.clip(eigenvalues, floor, None)
    inverse_root = (eigenvectors * (clipped ** -0.5)) @ eigenvectors.T
    forward_root = (eigenvectors * (clipped**0.5)) @ eigenvectors.T
    return inverse_root, forward_root


def _one_hot(groups: np.ndarray, group_count: int) -> np.ndarray:
    encoded = np.zeros((len(groups), group_count), dtype=np.float64)
    encoded[np.arange(len(groups)), groups] = 1.0
    return encoded


def leace_eraser(
    features: np.ndarray,
    groups: np.ndarray,
    *,
    group_count: int,
    floor: float,
    singular_tolerance: float,
) -> tuple[AffineMap, int]:
    """LEACE, arXiv:2306.03819, in closed form.

    Returns the eraser and the rank actually removed. A centred one-hot over g
    groups has rank g-1, so the removed rank is the check that the erasure did
    what it claims rather than a free parameter.
    """
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    centred = features - mean
    concept = _one_hot(groups, group_count)
    concept -= concept.mean(axis=0)
    n = len(features)
    covariance = centred.T @ centred / n
    cross = centred.T @ concept / n

    inverse_root, forward_root = _symmetric_powers(covariance, floor=floor)
    whitened_cross = inverse_root @ cross
    basis, singular_values, _ = np.linalg.svd(whitened_cross, full_matrices=False)
    # A centred one-hot over g groups has rank exactly g-1, so the budget is
    # known in advance rather than discovered from the data (N90.2.17). Without
    # this cap the random-group null, whose singular values are all noise, keeps
    # a sixth direction and stops being budget-matched to the arm it controls.
    limit = max(group_count - 1, 0)
    above = int(np.count_nonzero(singular_values > singular_values[0] * singular_tolerance))
    retained = basis[:, : min(limit, above)]
    projector = retained @ retained.T

    eraser = forward_root @ projector @ inverse_root
    matrix = np.eye(features.shape[1]) - eraser
    return AffineMap(matrix, eraser @ mean), int(retained.shape[1])


def top_direction_remover(
    features: np.ndarray, *, count: int
) -> tuple[AffineMap, int]:
    """All-but-the-Top, arXiv:1702.01417: drop the common mean and the top directions."""
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    centred = features - mean
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    directions = right[:count].T
    matrix = np.eye(features.shape[1]) - directions @ directions.T
    return AffineMap(matrix, -matrix @ mean), count


def random_direction_remover(
    features: np.ndarray, *, count: int, seed: int
) -> tuple[AffineMap, int]:
    """N90.2.6. The same removal, at directions chosen without looking at the data."""
    features = np.asarray(features, dtype=np.float64)
    rng = np.random.default_rng(seed)
    dimension = features.shape[1]
    directions, _ = np.linalg.qr(rng.standard_normal((dimension, count)))
    mean = features.mean(axis=0)
    matrix = np.eye(dimension) - directions @ directions.T
    return AffineMap(matrix, -matrix @ mean), count


def zca_whitener(features: np.ndarray, *, floor: float) -> tuple[AffineMap, int]:
    """Pooled whitening, read as a bound rather than a remedy (arXiv:2103.15597)."""
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    centred = features - mean
    covariance = centred.T @ centred / len(features)
    inverse_root, _ = _symmetric_powers(covariance, floor=floor)
    return AffineMap(inverse_root, -inverse_root @ mean), 0


def domain_probe(
    fit_features: np.ndarray,
    fit_domains: np.ndarray,
    report_features: np.ndarray,
    report_domains: np.ndarray,
    *,
    max_iterations: int,
    regularisation: float,
    seed: int,
) -> dict[str, Any]:
    """N90.2.5. Fitted on fit rows, scored on held-out report rows.

    Balanced over domains, because quickdraw is 61 percent of the corpus and an
    unbalanced score would read as high domain predictability for a probe that
    had learned only to say quickdraw. Standardised first: lbfgs does not
    converge on these raw features inside any tolerable iteration budget, and an
    unconverged probe understates domain predictability, which would read as
    successful erasure (N90.2.12).
    """
    centre = fit_features.mean(axis=0)
    scale = np.maximum(fit_features.std(axis=0), 1e-12)
    model = LogisticRegression(
        max_iter=max_iterations,
        C=regularisation,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit((fit_features - centre) / scale, fit_domains)
    predicted = model.predict((report_features - centre) / scale)
    iterations = int(np.max(model.n_iter_))
    present = int(len(np.unique(report_domains)))
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(report_domains, predicted)
        ),
        "iterations": iterations,
        "converged": bool(iterations < max_iterations),
        "domains_present": present,
        "chance_balanced_accuracy": 1.0 / present,
    }


def dominance_ratio(separation: dict[str, Any]) -> float | None:
    """N90.2.1. Scale-invariant, so it is the only cross-arm geometric comparison."""
    dominance = separation["domain_dominates_class"]
    sibling = dominance["own_class_sibling_cell_gap_median"]
    foreign = dominance["nearest_foreign_class_cell_gap_median"]
    if sibling is None or not foreign:
        return None
    return float(sibling / foreign)


def _h94(
    arm: dict[str, Any],
    null: dict[str, Any],
    gate: dict[str, Any],
    *,
    probe_bar: float,
) -> dict[str, Any]:
    """N90.2.2. Both bars, and both beaten against the matched null."""
    ratio = arm["dominance_ratio"]
    null_ratio = null["dominance_ratio"]
    accuracy = arm["probe"]["balanced_accuracy"]
    null_accuracy = null["probe"]["balanced_accuracy"]
    checks = {
        "dominance_ratio_below_bar": bool(
            ratio is not None and ratio < float(gate["dominance_ratio_bar"])
        ),
        "probe_below_chance_bar": bool(accuracy < probe_bar),
        "beats_null_on_ratio": bool(
            ratio is not None and null_ratio is not None and ratio < null_ratio
        ),
        "beats_null_on_probe": bool(accuracy < null_accuracy),
    }
    return {"removes_domain_dominance": all(checks.values()), **checks}


def _verdict(
    arms: dict[str, Any], gate: dict[str, Any], probe_config: dict[str, Any]
) -> dict[str, Any]:
    baseline_probe = arms["baseline"]["probe"]["balanced_accuracy"]
    all_converged = all(arm["probe"]["converged"] for arm in arms.values())
    probe_valid = (
        baseline_probe >= float(probe_config["probe_positive_floor"]) and all_converged
    )
    probe_bar = float(probe_config["probe_chance_bar"])

    first_moment_verified = bool(
        arms["leace_domain"]["erasure_certificate"]["first_moment_erased"]
    )

    h94_arms = {
        "leace_domain": _h94(
            arms["leace_domain"], arms["leace_null"], gate, probe_bar=probe_bar
        ),
        "abtt": _h94(arms["abtt"], arms["abtt_null"], gate, probe_bar=probe_bar),
    }
    erased = [name for name, result in h94_arms.items() if result["removes_domain_dominance"]]

    if not probe_valid:
        h94 = "undetermined"
    elif erased:
        h94 = "survives"
    else:
        h94 = "refuted"

    cleared = [
        name
        for name in ("leace_domain", "abtt", "whiten")
        if arms[name]["gate"]["clears"]
    ]
    if h94 == "undetermined":
        h95 = "undetermined"
    elif not erased:
        h95 = "untestable"
    elif cleared:
        h95 = "survives"
    else:
        h95 = "refuted"

    return {
        "probe_instrument": {
            "baseline_balanced_accuracy": baseline_probe,
            "positive_floor": float(probe_config["probe_positive_floor"]),
            "all_arms_converged": bool(all_converged),
            "chance_balanced_accuracy": arms["baseline"]["probe"][
                "chance_balanced_accuracy"
            ],
            "valid": bool(probe_valid),
            "basis": (
                "If a linear probe cannot read domain off the untouched "
                "features, the probe is broken and no erasure figure can be "
                "read from it (N90.2.4). An unconverged probe understates "
                "domain predictability and would read as successful erasure, "
                "so convergence on every arm is part of validity (N90.2.12)."
            ),
        },
        "h94_domain_erasable": {
            "verdict": h94,
            "arms": h94_arms,
            "first_moment_erasure_verified": first_moment_verified,
            "basis": (
                "An arm must drive the dominance ratio below the bar and the "
                "held-out probe below the chance bar, beating its matched null "
                "on both (N90.2.2). The certificate is reported beside the "
                "verdict because a refuted H94 is otherwise ambiguous between "
                "a domain that is not linearly encoded and an eraser that was "
                "misapplied, which are opposite findings (N90.2.15)."
            ),
        },
        "h95_erasure_helps_rejection": {
            "verdict": h95,
            "cleared": cleared,
            "basis": "M90's three bars, unchanged (N90.2.3).",
        },
        "dissociation": bool(h94 == "survives" and h95 == "refuted"),
        "reading": (
            "Domain dominance is real and removable, and removing it does not "
            "recover rejection. Domain overlap is therefore not the cause of "
            "v13's rejection failure, which is a stronger statement than either "
            "hypothesis alone and was registered before the run (N90.2.0)."
            if h94 == "survives" and h95 == "refuted"
            else (
                "The first moment of domain was erased exactly and domain "
                "dominance survived it, so domain is not a mean offset in these "
                "features. A translation-based remedy cannot reach it, and no "
                "arm moved rejection either way."
                if first_moment_verified and h94 == "refuted"
                else "See the per-arm table; the registered dissociation did not occur."
            )
        ),
    }


def erasure_certificate(
    before: np.ndarray,
    after: np.ndarray,
    groups: np.ndarray,
    *,
    group_count: int,
) -> dict[str, Any]:
    """N90.2.15. Evidence that the transform did what its paper promises.

    Without this, a refuted H94 is ambiguous between "domain is not linearly
    encoded" and "the eraser was misapplied", which are opposite findings. The
    first two quantities are what LEACE guarantees. The third is what it does
    not touch, and is reported because a first-moment erasure that leaves the
    second moment intact is the whole reason RobustNet and Deep CORAL exist.
    """
    present = [d for d in range(group_count) if bool((groups == d).any())]

    def moments(x: np.ndarray) -> tuple[float, float]:
        x = np.asarray(x, dtype=np.float64)
        means = np.stack([x[groups == d].mean(axis=0) for d in present])
        gaps = np.linalg.norm(means[:, None, :] - means[None, :, :], axis=2)
        concept = _one_hot(groups, group_count)
        concept -= concept.mean(axis=0)
        centred = x - x.mean(axis=0)
        cross = np.abs(centred.T @ concept / len(x)).max()
        return float(gaps.max()), float(cross)

    gap_before, cross_before = moments(before)
    gap_after, cross_after = moments(after)
    variances = [
        float(np.trace(np.cov(np.asarray(after, dtype=np.float64)[groups == d].T, bias=True)))
        for d in present
    ]
    smallest = min(variances)
    # Relative, because the features are float32 and a float64 map applied to
    # them and cast back cannot leave a residual smaller than the rounding of
    # the corpus itself (N90.2.16). An absolute bar would measure the dtype.
    return {
        "max_pairwise_domain_mean_gap_before": gap_before,
        "max_pairwise_domain_mean_gap_after": gap_after,
        "max_abs_cross_covariance_before": cross_before,
        "max_abs_cross_covariance_after": cross_after,
        "first_moment_residual_fraction": (
            float(gap_after / gap_before) if gap_before > 0.0 else None
        ),
        "first_moment_erased": bool(
            gap_before > 0.0
            and gap_after / gap_before < 1e-5
            and cross_after / max(cross_before, 1e-300) < 1e-5
        ),
        "per_domain_total_variance_after": dict(
            zip((str(d) for d in present), variances)
        ),
        "second_moment_variance_ratio": (
            float(max(variances) / smallest) if smallest > 0.0 else None
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
    fit_labels = labels[fit_rows]
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

    erasure = config["erasure"]
    floor = float(erasure["eigenvalue_floor"])
    singular_tolerance = float(erasure["singular_value_tolerance"])
    seeds = config["seeds"]
    domain_sizes = np.array(
        [int((fit_domains == d).sum()) for d in range(domain_count)], dtype=np.int64
    )
    null_groups = _random_groups(
        len(fit_raw), domain_sizes, seed=int(seeds["random_group"])
    )

    def build(name: str) -> tuple[AffineMap | None, int]:
        if name == "identity":
            return None, 0
        if name == "leace_domain":
            return leace_eraser(
                fit_raw,
                fit_domains,
                group_count=domain_count,
                floor=floor,
                singular_tolerance=singular_tolerance,
            )
        if name == "leace_random":
            return leace_eraser(
                fit_raw,
                null_groups,
                group_count=domain_count,
                floor=floor,
                singular_tolerance=singular_tolerance,
            )
        if name == "abtt":
            return top_direction_remover(
                fit_raw, count=int(erasure["abtt_directions"])
            )
        if name == "abtt_random":
            return random_direction_remover(
                fit_raw,
                count=int(erasure["abtt_directions"]),
                seed=int(seeds["abtt_null"]),
            )
        if name == "zca_whiten":
            return zca_whitener(fit_raw, floor=floor)
        raise ValueError(f"unknown transform {name}")

    probe_config = config["probe"]
    separation_config = config["separation"]
    gate_config = config["gate"]
    rank = int(config["rank"])
    assignment = class_assignment(fit_labels, class_count)
    arms: dict[str, Any] = {}

    for spec in config["arms"]:
        name = spec["name"]
        mapping, removed_rank = build(spec["transform"])
        if mapping is None:
            space = {
                "fit": fit_raw,
                "calibration": calibration_raw,
                "report": report_raw,
                "unseen": unseen_raw,
            }
        else:
            space = {
                "fit": mapping(fit_raw),
                "calibration": mapping(calibration_raw),
                "report": mapping(report_raw),
                "unseen": mapping(unseen_raw),
            }

        certificate = erasure_certificate(
            fit_raw, space["fit"], fit_domains, group_count=domain_count
        )

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

        separation = separation_report(
            space["report"],
            report_labels,
            report_domains,
            class_count=class_count,
            domain_count=domain_count,
            minimum_cell_support=int(separation_config["minimum_cell_support"]),
            floor_samples_per_dimension=int(
                separation_config["floor_samples_per_dimension"]
            ),
        )

        arm: dict[str, Any] = {
            "transform": spec["transform"],
            "rationale": spec["rationale"],
            "removed_rank": removed_rank,
            "erasure_certificate": certificate,
            "controls": controls_with_repair(
                known_scores,
                far_scores,
                floor=float(controls_config["far_field_floor"]),
                tolerance=float(controls_config["known_split_tolerance"]),
                class_block=class_block_split,
                stratified=stratified_split,
            ),
            "probe": domain_probe(
                space["fit"],
                fit_domains,
                space["report"],
                report_domains,
                max_iterations=int(probe_config["max_iterations"]),
                regularisation=float(probe_config["regularisation_C"]),
                seed=int(probe_config["seed"]),
            ),
            "separation": separation,
            "dominance_ratio": dominance_ratio(separation),
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
        ratio = arm["dominance_ratio"]
        shown = "n/a" if ratio is None else f"{ratio:.3f}"
        cert = arm["erasure_certificate"]
        print(
            f"  {name:<14} removed-{removed_rank}"
            f"  probe {arm['probe']['balanced_accuracy']:.4f}"
            f"  dominance {shown:>6}"
            f"  meangap {cert['max_pairwise_domain_mean_gap_after']:>9.2e}"
            f"  recall {arm['rejection_recall']['rejection_recall']:.5f}"
            f"  AUROC {arm['auroc']['auroc']:.4f}"
            f"  mult {arm['acceptance_multiplicity']['mean']:>6.2f}"
            f"  acc {arm['known_accuracy']:.4f}"
            f"  {'ok' if arm['controls']['valid'] else 'INVALID'}"
        )

    replication_config = config["replication"]
    reference = json.loads(
        _resolve(replication_config["reference_evidence"]).read_text(encoding="utf-8")
    )
    if reference["evidence_hash"] != replication_config["reference_hash"]:
        raise SystemExit(
            "M90.1 reference evidence does not match its registered hash: "
            f"{reference['evidence_hash']}"
        )
    reference_arm = reference["arms"][replication_config["reference_arm"]]
    identity = arms["baseline"]
    replication_deltas = {
        "acceptance_multiplicity.mean": float(
            identity["acceptance_multiplicity"]["mean"]
            - reference_arm["acceptance_multiplicity"]["mean"]
        ),
        "rejection_recall": float(
            identity["rejection_recall"]["rejection_recall"]
            - reference_arm["rejection_recall"]["rejection_recall"]
        ),
        "auroc": float(identity["auroc"]["auroc"] - reference_arm["auroc"]["auroc"]),
        "known_accuracy": float(
            identity["known_accuracy"] - reference_arm["known_accuracy"]
        ),
    }
    worst = max(abs(v) for v in replication_deltas.values())

    evidence: dict[str, Any] = {
        "milestone": "M90.2",
        "component": "domain_erasure",
        "generated_at": datetime.now(UTC).isoformat(),
        "registered_question": config["registered_question"],
        "registration_notes": config["registration_notes"],
        "corpus": config["corpus"],
        "openset": config["openset"],
        "partition": partition_report,
        "coverage": config["coverage"],
        "erasure": erasure,
        "probe_config": probe_config,
        "separation_config": separation_config,
        "controls": controls_config,
        "seeds": seeds,
        "gate": gate_config,
        "arms": arms,
        "replication": {
            "reference_hash": reference["evidence_hash"],
            "tolerance": float(replication_config["operand_tolerance"]),
            "largest_absolute_delta": worst,
            "operands_reproduce": bool(
                worst <= float(replication_config["operand_tolerance"])
            ),
            "deltas": replication_deltas,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "runtime_seconds": None,
    }
    evidence["verdict"] = _verdict(arms, gate_config, probe_config)
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
    print(f"\nprobe instrument      valid={verdict['probe_instrument']['valid']}")
    print(f"H94 domain erasable   {verdict['h94_domain_erasable']['verdict']}")
    print(f"H95 erasure helps     {verdict['h95_erasure_helps_rejection']['verdict']}")
    print(f"dissociation          {verdict['dissociation']}")
    print(f"baseline reproduces   {evidence['replication']['operands_reproduce']}")
    print(f"evidence_hash         {evidence['evidence_hash']}")
    print(f"runtime               {evidence['runtime_seconds'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
