"""M83: does boundary supervision in an absolute length train a real boundary?

v12 placed its synthetic negatives at a multiple of each class's own fitted
extent and normalised the resulting score by that same extent, so the two
cancelled and the score was the multiplier — a constant, independent of the
quantity being supervised. M77 measured the resulting gradient at 1e-12. This
milestone replaces the placement rule with one global length and asks whether
the boundary that falls out rejects genuinely unseen classes.

The instrument comes before the result. Arm ``v12_relative`` is retained as the
negative end of the degeneracy contract: it must read as scale-blind while the
absolute arm reads as live. Under N83.1, if it does not, M83 is void and no
number here may be recorded as evidence.

Three further arms guard the positive claim. ``shuffled_null`` trains on the
same probes with their owners permuted, so it shares geometry, probe count,
distances, optimiser and step budget and differs only in the correspondence
between a negative and the class it is a negative for. ``untrained`` is the
fitted ellipsoid at the same 90 percent coverage, which is what the trained
boundary must beat rather than chance. And every acceptance figure is read on a
report half that the coverage match never saw.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
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
from experiments.common.v13_boundary import (
    Geometry,
    absolute_unit,
    acceptance_rate,
    apply_offsets,
    boundary_displacement,
    build_probe_spec,
    data_scale_unit,
    degeneracy_report,
    domain_matched_partition,
    domain_stratified_halves,
    fit_geometry,
    global_scale_unit,
    matched_coverage_offsets,
    probe_rejection,
    probe_validity,
    rejection_recall,
    shuffled_owners,
    train_boundary,
)
from experiments.tier4.eval_v13_m80_sparse_dictionary import (
    _load_corpus,
    _verify_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v13" / "m83_boundary.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v13" / "m83_boundary"

_WORKER_STATE: dict[str, Any] = {}


def _resolve(path: str) -> Path:
    return (REPO_ROOT / path).resolve()


def _verify_openset(specification: dict[str, Any]) -> Path:
    """The out-of-set rows are a sealed input, extracted on the GPU venv.

    Verifying every artifact against the index, and the index's own feature
    hash against the config, is the only thing standing between this milestone
    and a silently regenerated evaluation set.
    """
    index_path = _resolve(specification["path"])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for artifact in index["artifacts"]:
        artifact_path = index_path.parent / artifact["path"]
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError(f"M83 open-set artifact hash mismatch: {artifact_path}")
    evidence = json.loads(
        (index_path.parent / "evidence.json").read_text(encoding="utf-8")
    )
    if evidence["feature_hash"] != specification["feature_hash"]:
        raise ValueError("M83 open-set features are not the sealed ones")
    if not evidence["corpus_disjointness_control"]["passes"]:
        raise ValueError("M83 open-set rows failed their own disjointness control")
    if not evidence["shard_invariance_control"]["passes"]:
        raise ValueError("M83 open-set features failed their shard-invariance control")
    return index_path


def _load_openset(
    index_path: Path, *, stratified_only: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features, labels and domains for the unseen classes.

    N83.2 restricts the evaluation to the stratified mask: the rows whose
    per-class domain quota was filled exactly, which match the corpus domain
    profile to within 0.0057. The 11 underfilled classes are dropped rather
    than represented at a different mixture from the rest.
    """
    arrays = index_path.parent / "arrays"
    features = np.load(arrays / "features.npy").astype(np.float32)
    labels = np.load(arrays / "labels.npy").astype(np.int64)
    domains = np.load(arrays / "domains.npy").astype(np.int64)
    if stratified_only:
        mask = np.load(arrays / "stratified.npy")
        features, labels, domains = features[mask], labels[mask], domains[mask]
    return features, labels, domains


def _partition_evidence(config: dict[str, Any]) -> dict[str, Any]:
    """What the domain quota actually achieved, recorded beside the result.

    The manifest alone is enough for this, so it costs nothing to state the
    achieved mixture rather than assert it. N83.7 exists because the previous
    split's mixture was never written down and therefore never checked.
    """
    manifest = json.loads(
        (_resolve(config["corpus"]["path"]).parent / "selection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    labels = np.asarray(
        [int(row["class_label"]) for row in manifest["selection"]], dtype=np.int64
    )
    domains = np.asarray(
        [int(row["domain"]) for row in manifest["selection"]], dtype=np.int64
    )
    partition = config["partition"]
    fit_rows, evaluation_rows, report = domain_matched_partition(
        labels,
        domains,
        quota=tuple(int(value) for value in partition["evaluation_domain_quota"]),
        fit_per_class=int(partition["fit_per_class"]),
        domain_count=int(config["corpus"]["domain_count"]),
    )
    calibration_rows, report_rows = domain_stratified_halves(
        labels, domains, evaluation_rows
    )
    domain_count = int(config["corpus"]["domain_count"])

    def profile(rows: np.ndarray) -> list[float]:
        return [
            float(np.count_nonzero(domains[rows] == domain) / max(len(rows), 1))
            for domain in range(domain_count)
        ]

    return {
        "achieved": report,
        "fit_row_count": int(len(fit_rows)),
        "evaluation_row_count": int(len(evaluation_rows)),
        "calibration_row_count": int(len(calibration_rows)),
        "report_row_count": int(len(report_rows)),
        "fit_domain_profile": profile(fit_rows),
        "calibration_domain_profile": profile(calibration_rows),
        "report_domain_profile": profile(report_rows),
        "corpus_domain_profile": [
            float(np.count_nonzero(domains == domain) / len(domains))
            for domain in range(domain_count)
        ],
    }


def _corpus_domains(index_path: Path) -> np.ndarray:
    """Domain labels for the corpus rows, read from the sealed manifest.

    The corpus arrays carry features and labels only; the domain each row came
    from lives in the selection manifest, which is hashed in the same artifact
    index and so is verified before this is called.
    """
    manifest = json.loads(
        (index_path.parent / "selection_manifest.json").read_text(encoding="utf-8")
    )
    return np.asarray(
        [int(row["domain"]) for row in manifest["selection"]], dtype=np.int64
    )


def _initial_log_beta(geometry: Geometry) -> np.ndarray:
    return np.concatenate(
        [
            np.log(geometry.tangent_scales),
            np.log(geometry.residual_scales)[:, None],
        ],
        axis=1,
    )


def _measure(
    log_beta: np.ndarray,
    *,
    state: dict[str, Any],
    coverage: float,
    class_count: int,
) -> dict[str, Any]:
    """Match coverage, then read every acceptance figure on the report half.

    The offsets are pure radius, so this cancels exactly the component of any
    boundary that a coverage match would have erased anyway. What survives to
    be compared across arms is shape.
    """
    geometry = state["geometry"]
    features = state["features"]
    labels = state["labels"]
    offsets = matched_coverage_offsets(
        features[state["calibration_rows"]],
        labels[state["calibration_rows"]],
        geometry,
        log_beta,
        coverage=coverage,
        class_count=class_count,
    )
    matched = apply_offsets(log_beta, offsets)
    known_novel = acceptance_rate(
        features[state["report_rows"]],
        labels[state["report_rows"]],
        geometry,
        matched,
    )
    rejection = rejection_recall(
        state["openset_features"],
        geometry,
        matched,
        domains=state["openset_domains"],
        domain_count=int(state["domain_count"]),
    )
    held_out = probe_rejection(
        geometry,
        matched,
        state["held_out_spec"],
        placement="absolute",
        unit=state["unit"],
    )
    # The same statistic on known rows. A rejection recall is only interpretable
    # against the rate at which the same boundary rejects rows it should keep,
    # and reading both per domain is what makes a domain confound visible
    # instead of leaving it to be inferred from an aggregate.
    known_rejection = rejection_recall(
        features[state["report_rows"]],
        geometry,
        matched,
        domains=state["corpus_domains"][state["report_rows"]],
        domain_count=int(state["domain_count"]),
    )
    return {
        "known_novel_acceptance": known_novel,
        "rejection": rejection,
        "known_false_rejection": known_rejection,
        "held_out_family_rejection": held_out,
        "coverage_offset_mean": float(np.mean(offsets)),
    }


def _worker_init(config: dict[str, Any]) -> None:
    """Fit Phase A once. It is closed-form, so every seed shares it.

    That is worth stating plainly: M83's seeds vary the probe directions, the
    batch order and the owner permutation, not the geometry. The spread across
    seeds is therefore a statement about the training, which is the thing under
    test, and not about the fit, which N83.4 records as exactly converged.
    """
    torch.set_num_threads(int(config["threading"]["torch_threads_per_worker"]))
    corpus_index = _verify_corpus(config["corpus"])
    features, labels = _load_corpus(corpus_index)
    domains = _corpus_domains(corpus_index)
    partition = config["partition"]
    fit_rows, evaluation_rows, partition_report = domain_matched_partition(
        labels,
        domains,
        quota=tuple(int(value) for value in partition["evaluation_domain_quota"]),
        fit_per_class=int(partition["fit_per_class"]),
        domain_count=int(config["corpus"]["domain_count"]),
    )
    calibration_rows, report_rows = domain_stratified_halves(
        labels, domains, evaluation_rows
    )
    class_count = int(config["corpus"]["class_count"])
    geometry = fit_geometry(
        features[fit_rows],
        labels[fit_rows],
        rank=int(config["geometry"]["rank"]),
        class_count=class_count,
    )
    openset_index = _verify_openset(config["openset"])
    openset_features, _, openset_domains = _load_openset(
        openset_index, stratified_only=bool(config["openset"]["stratified_only"])
    )
    # N83.8. The placement unit is registered, not inferred, because M83.1's
    # was wrong in a way no downstream figure could reveal.
    unit_rule = str(config["probes"].get("placement_unit", "tangent_median"))
    if unit_rule == "tangent_median":
        unit = global_scale_unit(geometry)
    elif unit_rule == "data_extent":
        unit = data_scale_unit(features[fit_rows], labels[fit_rows], geometry)
    else:
        raise ValueError(f"unknown placement unit rule: {unit_rule}")
    held_out_spec = build_probe_spec(
        geometry,
        families=tuple(config["probes"]["held_out_families"]),
        multipliers=tuple(float(value) for value in config["probes"]["multipliers"]),
        seed=int(config["probes"]["seed_offset"]),
    )
    _WORKER_STATE.update(
        {
            "config": config,
            "geometry": geometry,
            "features": features,
            "labels": labels,
            "fit_rows": fit_rows,
            "calibration_rows": calibration_rows,
            "report_rows": report_rows,
            "partition_report": partition_report,
            "corpus_domains": domains,
            "class_count": class_count,
            "unit": unit,
            "unit_rule": unit_rule,
            "centroid_unit": absolute_unit(geometry.centers),
            "openset_features": openset_features,
            "openset_domains": openset_domains,
            "domain_count": int(config["corpus"]["domain_count"]),
            "held_out_spec": held_out_spec,
        }
    )


def _worker_run(seed: int) -> dict[str, Any]:
    state = _WORKER_STATE
    config = state["config"]
    geometry = state["geometry"]
    probes = config["probes"]
    training = config["training"]
    coverage = float(config["coverage"]["known_coverage"])
    class_count = int(state["class_count"])
    unit = float(state["unit"])
    margin = float(training["margin"])

    spec = build_probe_spec(
        geometry,
        families=tuple(probes["train_families"]),
        multipliers=tuple(float(value) for value in probes["multipliers"]),
        seed=int(probes["seed_offset"]) + seed,
    )
    initial = _initial_log_beta(geometry)

    arms: dict[str, Any] = {}
    for name, arm in config["arms"].items():
        placement = str(arm["placement"])
        arm_spec = spec
        if arm.get("shuffle_owners"):
            arm_spec = spec.with_owners(
                shuffled_owners(
                    spec.owners,
                    class_count=class_count,
                    seed=int(arm["shuffle_seed_offset"]) + seed,
                )
            )
        epochs = int(arm.get("epochs", training["epochs"]))
        if epochs == 0:
            final, history = initial.copy(), []
        else:
            final, history = train_boundary(
                state["features"][state["fit_rows"]],
                state["labels"][state["fit_rows"]],
                geometry,
                arm_spec,
                placement=placement,
                unit=unit,
                epochs=epochs,
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                margin=margin,
                probe_weight=float(training["probe_weight"]),
                seed=seed,
            )
        arms[name] = {
            "placement": placement,
            "epochs": epochs,
            "degeneracy_at_initialisation": degeneracy_report(
                geometry,
                initial,
                arm_spec,
                placement=placement,
                unit=unit,
                margin=margin,
            ),
            "degeneracy_at_final": degeneracy_report(
                geometry, final, arm_spec, placement=placement, unit=unit, margin=margin
            ),
            "displacement": boundary_displacement(initial, final),
            "history_first": history[0] if history else None,
            "history_last": history[-1] if history else None,
            "measurements": _measure(
                final, state=state, coverage=coverage, class_count=class_count
            ),
        }
    return {
        "seed": seed,
        "placement_unit": unit,
        "placement_unit_rule": str(state["unit_rule"]),
        "centroid_unit": float(state["centroid_unit"]),
        "unit_ratio": unit / float(state["centroid_unit"]),
        "probe_count": len(spec),
        "probe_validity": probe_validity(
            state["features"][state["fit_rows"]],
            state["labels"][state["fit_rows"]],
            geometry,
            initial,
            spec,
            unit=unit,
        ),
        "openset_row_count": int(len(state["openset_features"])),
        "report_row_count": int(len(state["report_rows"])),
        "arms": arms,
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _spread(values: list[float]) -> float | None:
    return float(max(values) - min(values)) if values else None


def _arm_recall(results: list[dict[str, Any]], name: str) -> list[float]:
    return [
        float(entry["arms"][name]["measurements"]["rejection"]["rejection_recall"])
        for entry in results
    ]


def _gate(results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """The degeneracy contract first; everything else is conditional on it.

    Plan section 15 makes this retroactive across v13: an objective that fails
    here is void, and its numbers may not be recorded as evidence at all. So
    the verdict short-circuits — if the instrument does not separate the two
    placement rules, the milestone reports that and nothing else.
    """
    gate_config = config["gate"]
    maximum_degenerate = float(gate_config["maximum_gradient_norm_for_degeneracy"])
    maximum_spread = float(gate_config["maximum_rescale_spread_for_degeneracy"])
    minimum_live = float(gate_config["minimum_gradient_norm_for_live"])

    # N83.8, and it runs before the degeneracy contract because it is prior to
    # it: a scale-sensitive objective supervised by probes that lie inside the
    # data it must accept is still measuring nothing. M83.1 passed every other
    # clause in this gate and was void for exactly this reason.
    validity = [entry["probe_validity"] for entry in results]
    ladder = {
        "probe_distance_maximum": min(
            float(entry["probe_distance_maximum"]) for entry in validity
        ),
        "known_distance_tenth_percentile": max(
            float(entry["known_distance_tenth_percentile"]) for entry in validity
        ),
        "known_distance_median": max(
            float(entry["known_distance_median"]) for entry in validity
        ),
        "fraction_beyond_known_median": min(
            float(entry["fraction_beyond_known_median"]) for entry in validity
        ),
        "passes": all(bool(entry["reaches_past_known_cloud"]) for entry in validity),
    }
    if not ladder["passes"]:
        return {
            "probe_ladder": ladder,
            "verdict": "probe_ladder_interior",
            "verdict_note": (
                "N83.8. Every synthetic negative lies inside the known cloud "
                "the boundary is required to accept, so no rejection figure "
                "below this is an operand. The milestone is void, not negative."
            ),
        }

    relative = [entry["arms"]["v12_relative"] for entry in results]
    absolute = [entry["arms"]["m83_absolute"] for entry in results]
    relative_gradients = [
        float(arm["degeneracy_at_initialisation"]["gradient_norm_log_beta"])
        for arm in relative
    ]
    relative_spreads = [
        float(arm["degeneracy_at_initialisation"]["rescale_spread"]) for arm in relative
    ]
    absolute_gradients = [
        float(arm["degeneracy_at_initialisation"]["gradient_norm_log_beta"])
        for arm in absolute
    ]
    absolute_spreads = [
        float(arm["degeneracy_at_initialisation"]["rescale_spread"]) for arm in absolute
    ]

    degeneracy = {
        "relative_gradient_max": max(relative_gradients),
        "relative_rescale_spread_max": max(relative_spreads),
        "absolute_gradient_min": min(absolute_gradients),
        "absolute_rescale_spread_min": min(absolute_spreads),
        "relative_reads_degenerate": max(relative_gradients) < maximum_degenerate
        and max(relative_spreads) < maximum_spread,
        "absolute_reads_live": min(absolute_gradients) > minimum_live,
    }
    degeneracy["passes"] = (
        degeneracy["relative_reads_degenerate"] and degeneracy["absolute_reads_live"]
    )

    if not degeneracy["passes"]:
        return {
            "probe_ladder": ladder,
            "degeneracy": degeneracy,
            "verdict": "instrument_failed",
            "verdict_note": (
                "N83.1 voids the milestone when the degeneracy report does not "
                "separate the two placement rules. No operand below it is "
                "readable, so none is reported."
            ),
        }

    recalls = {
        name: {
            "mean": _mean(_arm_recall(results, name)),
            "spread": _spread(_arm_recall(results, name)),
            "per_seed": _arm_recall(results, name),
        }
        for name in config["arms"]
    }
    absolute_mean = recalls["m83_absolute"]["mean"] or 0.0
    untrained_mean = recalls["untrained"]["mean"] or 0.0
    null_mean = recalls["shuffled_null"]["mean"] or 0.0
    untrained_margin = absolute_mean - untrained_mean
    null_margin = absolute_mean - null_mean
    recall_spread = recalls["m83_absolute"]["spread"] or 0.0

    acceptances = [
        float(entry["arms"]["m83_absolute"]["measurements"]["known_novel_acceptance"])
        for entry in results
    ]
    coverage = float(config["coverage"]["known_coverage"])
    known_control = {
        "mean": _mean(acceptances),
        "per_seed": acceptances,
        "floor": coverage - 0.05,
        "passes": min(acceptances) >= coverage - 0.05,
    }

    shape = {
        name: _mean(
            [float(entry["arms"][name]["displacement"]["shape"]) for entry in results]
        )
        for name in config["arms"]
    }
    held_out = {
        name: _mean(
            [
                float(
                    entry["arms"][name]["measurements"]["held_out_family_rejection"]
                )
                for entry in results
            ]
        )
        for name in config["arms"]
    }

    beats_untrained = untrained_margin > recall_spread
    beats_null = null_margin > recall_spread
    passes = beats_untrained and beats_null and known_control["passes"]

    if passes:
        verdict = "absolute_boundary_trains"
        note = (
            "The absolute arm rejects unseen classes above both its own "
            "initialisation and the shuffled-owner null by more than the seed "
            "spread, at matched known coverage."
        )
    elif not known_control["passes"]:
        verdict = "known_coverage_not_held"
        note = (
            "N83.3 requires the known-class novel-image control to be accepted "
            "at the matched operating point. Any rejection figure read beside a "
            "failed control is a statement about the coverage match, not about "
            "the boundary."
        )
    elif not beats_null:
        verdict = "not_separable_from_null"
        note = (
            "N83.1. The boundary moved, but a boundary trained against probes "
            "whose owners were permuted rejects unseen classes just as well, so "
            "the movement is not evidence that the supervision found anything."
        )
    else:
        verdict = "no_gain_over_initialisation"
        note = (
            "The trained boundary does not beat the fitted ellipsoid at the same "
            "coverage. Phase B is then an expensive way to arrive where Phase A "
            "already was."
        )

    return {
        "probe_ladder": ladder,
        "degeneracy": degeneracy,
        "rejection_recall": recalls,
        "untrained_margin": untrained_margin,
        "null_margin": null_margin,
        "recall_spread": recall_spread,
        "beats_untrained": beats_untrained,
        "beats_null": beats_null,
        "known_novel_control": known_control,
        "shape_displacement_mean": shape,
        "held_out_family_rejection_mean": held_out,
        "verdict": verdict,
        "verdict_note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M83 absolute boundary supervision")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Override the registered seeds. For smoke runs only; a subset "
        "does not seal.",
    )
    arguments = parser.parse_args()

    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    output_dir = arguments.output or _resolve(config["output_dir"])
    registered_seeds = [int(seed) for seed in config["seeds"]]
    seeds = arguments.seeds if arguments.seeds is not None else registered_seeds
    sealed = seeds == registered_seeds

    started = time.time()
    workers = min(int(config["threading"]["workers"]), len(seeds))
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(config,)
    ) as pool:
        results = list(pool.map(_worker_run, seeds))
    elapsed = time.time() - started

    gate = _gate(results, config)
    evidence = {
        "schema_version": 1,
        "milestone": "M83",
        "program": "v13",
        "hypothesis": config["registered_hypothesis"],
        "registration_notes": config["registration_notes"],
        "seeds": seeds,
        "sealed": sealed,
        "corpus_sha256": config["corpus"]["sha256"],
        "openset_feature_hash": config["openset"]["feature_hash"],
        "geometry": config["geometry"],
        "partition": config["partition"],
        "partition_achieved": _partition_evidence(config),
        "probes": config["probes"],
        "training": config["training"],
        "coverage": config["coverage"],
        "per_seed": results,
        "gate": gate,
        "final_labels_opened": False,
        "elapsed_seconds": round(elapsed, 1),
    }
    evidence["configuration_hash"] = payload_hash(config)
    evidence["evidence_hash"] = payload_hash(evidence)

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "evidence.json"
    write_canonical_json(evidence_path, evidence)
    build_artifact_index(output_dir)

    print(f"verdict: {gate['verdict']}  sealed: {sealed}")
    ladder = gate["probe_ladder"]
    print(
        "probe ladder: farthest probe "
        f"{ladder['probe_distance_maximum']:.4f} | known distance p10 "
        f"{ladder['known_distance_tenth_percentile']:.4f} p50 "
        f"{ladder['known_distance_median']:.4f} | beyond known median "
        f"{ladder['fraction_beyond_known_median']:.4f} | passes "
        f"{ladder['passes']}"
    )
    if "degeneracy" not in gate:
        print(gate["verdict_note"])
        return
    degeneracy = gate["degeneracy"]
    print(
        "degeneracy: relative gradient "
        f"{degeneracy['relative_gradient_max']:.3e} spread "
        f"{degeneracy['relative_rescale_spread_max']:.3e} | absolute gradient "
        f"{degeneracy['absolute_gradient_min']:.3e} spread "
        f"{degeneracy['absolute_rescale_spread_min']:.3e} | passes "
        f"{degeneracy['passes']}"
    )
    if "rejection_recall" in gate:
        for name, entry in gate["rejection_recall"].items():
            print(f"  {name}: rejection recall {entry['mean']} spread {entry['spread']}")
        print(
            f"untrained margin {gate['untrained_margin']} null margin "
            f"{gate['null_margin']} seed spread {gate['recall_spread']}"
        )
        print(
            "known novel control "
            f"{gate['known_novel_control']['mean']} passes "
            f"{gate['known_novel_control']['passes']}"
        )
        absolute = results[0]["arms"]["m83_absolute"]["measurements"]
        print(
            "  m83_absolute known false rejection "
            f"{absolute['known_false_rejection']['rejection_recall']:.5f}"
        )
        for domain, entry in absolute["rejection"]["per_domain"].items():
            known = absolute["known_false_rejection"]["per_domain"].get(domain)
            if entry is None or entry["rejection_recall"] is None:
                continue
            print(
                f"    domain {domain}: unseen n={entry['row_count']} "
                f"recall {entry['rejection_recall']:.5f} | known "
                + (
                    f"n={known['row_count']} recall {known['rejection_recall']:.5f}"
                    if known and known["rejection_recall"] is not None
                    else "n=0"
                )
            )
    print(gate["verdict_note"])
    print(f"wrote {evidence_path} in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
