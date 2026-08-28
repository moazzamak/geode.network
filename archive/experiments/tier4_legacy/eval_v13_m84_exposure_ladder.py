"""M84: how much real out-group exposure does open-set competence need?

M83 closed the synthetic route. Probes placed along the in-group's own fitted
axes did not merely fail to help — the trained boundary ended up **worse** at
rejecting probes than its own initialisation, and rejected 0.0000 of the real
out-of-set. The reason generalises: within each domain, known and unseen rows
sit at near-identical distance from the global mean, so a radially placed
negative supervises a direction along which novelty does not live.

This milestone replaces the synthetic negatives with real out-group images and
sweeps how many of them are needed and how much class diversity they must
carry. The literature establishes that outlier exposure works; it does not
report where the knee is, which is the contribution here.

Two premises are checked before any operand is read, in this order, and each
suppresses everything below it:

1. **N84.6.** The exposure hinge must have gradient at the initialisation
   actually used. At the raw fitted radii every out-group row already sits
   nineteen times outside every boundary, the term is identically inert, and a
   ladder trained there would be flat for a reason that looks exactly like a
   real negative. Training therefore starts at matched coverage.
2. **N84.4.** The zero rung must reproduce the untrained boundary's rejection,
   measured at 0.11875 on this evaluation set before the milestone was written.

N84.1 keeps the exposure classes and the evaluation classes disjoint, and this
runner asserts that on load rather than trusting two configurations to agree.
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
    acceptance_rate,
    apply_offsets,
    boundary_displacement,
    domain_matched_partition,
    domain_stratified_halves,
    exposure_owners,
    exposure_validity,
    fit_geometry,
    matched_coverage_offsets,
    moment_matched_negatives,
    owner_agreement,
    rejection_recall,
    sample_exposure,
    tangent_anisotropy,
    train_exposure_boundary,
)
from experiments.tier4.eval_v13_m80_sparse_dictionary import (
    _load_corpus,
    _verify_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v13" / "m84_exposure_ladder.json"
)

_WORKER_STATE: dict[str, Any] = {}


def _resolve(path: str) -> Path:
    return (REPO_ROOT / path).resolve()


def _verify_sealed(specification: dict[str, Any], label: str) -> Path:
    """Hash every array against the index, and the index against the config.

    Both feature artifacts are sealed inputs. Verifying them here is the only
    thing standing between this milestone and a silently regenerated exposure
    pool or evaluation set.
    """
    index_path = _resolve(specification["path"])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for artifact in index["artifacts"]:
        artifact_path = index_path.parent / artifact["path"]
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError(f"M84 {label} artifact hash mismatch: {artifact_path}")
    evidence = json.loads(
        (index_path.parent / "evidence.json").read_text(encoding="utf-8")
    )
    if evidence["feature_hash"] != specification["feature_hash"]:
        raise ValueError(f"M84 {label} features are not the sealed ones")
    if not evidence["corpus_disjointness_control"]["passes"]:
        raise ValueError(f"M84 {label} rows failed their own disjointness control")
    if not evidence["shard_invariance_control"]["passes"]:
        raise ValueError(f"M84 {label} features failed their shard-invariance control")
    return index_path


def _load_arrays(
    index_path: Path, *, stratified_only: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = index_path.parent / "arrays"
    features = np.load(arrays / "features.npy").astype(np.float32)
    labels = np.load(arrays / "labels.npy").astype(np.int64)
    domains = np.load(arrays / "domains.npy").astype(np.int64)
    if stratified_only:
        mask = np.load(arrays / "stratified.npy")
        features, labels, domains = features[mask], labels[mask], domains[mask]
    return features, labels, domains


def _corpus_domains(index_path: Path) -> np.ndarray:
    manifest = json.loads(
        (index_path.parent / "selection_manifest.json").read_text(encoding="utf-8")
    )
    return np.asarray(
        [int(row["domain"]) for row in manifest["selection"]], dtype=np.int64
    )


def _measure(
    log_beta: np.ndarray,
    *,
    state: dict[str, Any],
    coverage: float,
    class_count: int,
) -> dict[str, Any]:
    """Match coverage, then read every acceptance figure on the report half.

    The coverage offsets are pure radius, so this cancels exactly the component
    of any boundary a coverage match would have erased anyway. What survives to
    be compared across rungs is shape — which is why N84.5 predicts anisotropy
    rather than radius, and why anisotropy is reported here.
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
    return {
        "known_novel_acceptance": acceptance_rate(
            features[state["report_rows"]],
            labels[state["report_rows"]],
            geometry,
            matched,
        ),
        "rejection": rejection_recall(
            state["evaluation_features"],
            geometry,
            matched,
            domains=state["evaluation_domains"],
            domain_count=int(state["domain_count"]),
        ),
        "known_false_rejection": rejection_recall(
            features[state["report_rows"]],
            geometry,
            matched,
            domains=state["corpus_domains"][state["report_rows"]],
            domain_count=int(state["domain_count"]),
        ),
        "coverage_offset_mean": float(np.mean(offsets)),
        "tangent_anisotropy": tangent_anisotropy(matched),
    }


def _worker_init(config: dict[str, Any]) -> None:
    """Phase A once, then the matched-coverage initialisation N84.6 requires.

    Phase A is closed-form, so every seed and every rung shares one geometry.
    The spread across seeds is therefore a statement about the exposure sample
    and the training, not about the fit.
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

    openset_index = _verify_sealed(config["openset"], "open-set")
    open_features, open_labels, open_domains = _load_arrays(
        openset_index, stratified_only=bool(config["openset"]["stratified_only"])
    )
    first_evaluation = int(config["openset"]["evaluation_first_label"])
    keep = open_labels >= first_evaluation
    evaluation_features = open_features[keep]
    evaluation_domains = open_domains[keep]
    evaluation_labels = open_labels[keep]

    exposure_index = _verify_sealed(config["exposure"], "exposure")
    pool_features, pool_labels, _ = _load_arrays(
        exposure_index, stratified_only=bool(config["exposure"]["stratified_only"])
    )
    keep_pool = pool_labels <= int(config["exposure"]["last_label"])
    pool_features = pool_features[keep_pool]
    pool_labels = pool_labels[keep_pool]

    # N84.1, asserted rather than assumed. Two configurations agreeing on a
    # label range is not the same as the arrays being disjoint, and training on
    # a class the boundary is scored against would not show up in any figure
    # below — it would simply make the ladder look like it worked.
    shared = np.intersect1d(np.unique(pool_labels), np.unique(evaluation_labels))
    if len(shared) > 0:
        raise ValueError(
            f"N84.1 violated: {len(shared)} classes appear in both the exposure "
            f"pool and the evaluation set, starting at {int(shared[0])}"
        )

    raw = np.concatenate(
        [
            np.log(geometry.tangent_scales),
            np.log(geometry.residual_scales)[:, None],
        ],
        axis=1,
    )
    coverage = float(config["coverage"]["known_coverage"])
    # N84.6. Train from the operating point the boundary is read at, because at
    # the fitted radii the exposure hinge has identically zero gradient.
    initial = apply_offsets(
        raw,
        matched_coverage_offsets(
            features[calibration_rows],
            labels[calibration_rows],
            geometry,
            raw,
            coverage=coverage,
            class_count=class_count,
        ),
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
            "raw_log_beta": raw,
            "initial_log_beta": initial,
            "evaluation_features": evaluation_features,
            "evaluation_domains": evaluation_domains,
            "evaluation_class_count": int(len(np.unique(evaluation_labels))),
            "pool_features": pool_features,
            "pool_labels": pool_labels,
            "domain_count": int(config["corpus"]["domain_count"]),
        }
    )


def _fit(
    negatives: np.ndarray, *, state: dict[str, Any], seed: int, epochs: int
) -> dict[str, Any]:
    """One arm of one cell: assign owners, train, then measure at coverage."""
    config = state["config"]
    training = config["training"]
    geometry = state["geometry"]
    initial = state["initial_log_beta"]
    margin = float(training["margin"])

    owners = (
        exposure_owners(negatives, geometry, initial)
        if len(negatives)
        else np.empty(0, dtype=np.int64)
    )
    if epochs == 0:
        final, history = initial.copy(), []
    else:
        final, history = train_exposure_boundary(
            state["features"][state["fit_rows"]],
            state["labels"][state["fit_rows"]],
            geometry,
            initial,
            negatives=negatives,
            owners=owners,
            epochs=epochs,
            batch_size=int(training["batch_size"]),
            exposure_batch_size=int(training["exposure_batch_size"]),
            learning_rate=float(training["learning_rate"]),
            margin=margin,
            exposure_weight=float(training["exposure_weight"]),
            seed=seed,
        )
    return {
        "negative_count": int(len(negatives)),
        "epochs": epochs,
        "exposure_validity": exposure_validity(
            negatives, geometry, initial, margin=margin
        ),
        "owner_agreement": owner_agreement(negatives, geometry, final, owners)
        if len(negatives)
        else None,
        "displacement": boundary_displacement(initial, final),
        "history_first": history[0] if history else None,
        "history_last": history[-1] if history else None,
        "measurements": _measure(
            final,
            state=state,
            coverage=float(config["coverage"]["known_coverage"]),
            class_count=int(state["class_count"]),
        ),
    }


def _worker_run(task: tuple[int, int]) -> dict[str, Any]:
    seed, cell_index = task
    state = _WORKER_STATE
    config = state["config"]
    cell = config["ladder"]["cells"][cell_index]
    count = int(cell["count"])

    arms: dict[str, Any] = {}
    if count == 0:
        for name, arm in config["zero_rung_arms"].items():
            arms[name] = _fit(
                np.empty((0, state["pool_features"].shape[1]), dtype=np.float32),
                state=state,
                seed=seed,
                epochs=int(arm["epochs"]),
            )
        selected = np.empty(0, dtype=np.int64)
    else:
        selected = sample_exposure(
            state["pool_labels"],
            count=count,
            diversity=int(cell["diversity"]),
            seed=int(config["ladder"]["sample_seed_offset"]) + seed,
        )
        real = state["pool_features"][selected]
        negatives = {
            "exposure": real,
            "moment_null": moment_matched_negatives(
                real,
                count=count,
                seed=int(config["ladder"]["null_seed_offset"]) + seed,
            ),
        }
        for name, arm in config["arms"].items():
            arms[name] = _fit(
                negatives[name],
                state=state,
                seed=seed,
                epochs=int(arm["epochs"]),
            )
    return {
        "seed": seed,
        "cell": cell["name"],
        "count": count,
        "diversity": int(cell["diversity"]),
        "exposure_class_count": int(
            len(np.unique(state["pool_labels"][selected])) if len(selected) else 0
        ),
        "evaluation_row_count": int(len(state["evaluation_features"])),
        "evaluation_class_count": int(state["evaluation_class_count"]),
        "arms": arms,
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _spread(values: list[float]) -> float | None:
    return float(max(values) - min(values)) if values else None


def _recalls(entries: list[dict[str, Any]], arm: str) -> list[float]:
    return [
        float(entry["arms"][arm]["measurements"]["rejection"]["rejection_recall"])
        for entry in entries
    ]


def _gate(results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """N84.6 first, then N84.4, then N83.3, then the ladder itself.

    Each clause suppresses everything below it. M83.1 passed every clause its
    gate contained and was still meaningless, which is the argument for putting
    the premises above the operands rather than beside them.
    """
    gate_config = config["gate"]
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for entry in results:
        by_cell.setdefault(str(entry["cell"]), []).append(entry)

    # 1. N84.6 — can the negatives move the objective at all?
    active = [
        float(arm["exposure_validity"]["active_fraction"])
        for entry in results
        for arm in entry["arms"].values()
        if int(arm["negative_count"]) > 0
    ]
    floor = float(gate_config["minimum_exposure_active_fraction"])
    validity = {
        "minimum_active_fraction": min(active) if active else None,
        "floor": floor,
        "passes": bool(active) and min(active) >= floor,
    }
    if not validity["passes"]:
        return {
            "exposure_validity": validity,
            "verdict": "exposure_term_inert",
            "verdict_note": (
                "N84.6. The negatives already satisfy the hinge at the "
                "initialisation used, so the exposure term has no gradient and "
                "the ladder measures the optimiser rather than the exposure. No "
                "figure below this clause may be read."
            ),
        }

    # 2. N84.4 — the zero rung is a known value.
    zero = by_cell.get("zero", [])
    untrained = _recalls(zero, "untrained") if zero else []
    expected = float(gate_config["untrained_rejection_expected"])
    tolerance = float(gate_config["untrained_rejection_tolerance"])
    zero_rung = {
        "untrained_mean": _mean(untrained),
        "expected": expected,
        "tolerance": tolerance,
        "known_only_mean": _mean(_recalls(zero, "known_only")) if zero else None,
        "passes": bool(untrained) and abs(_mean(untrained) - expected) <= tolerance,
    }
    if not zero_rung["passes"]:
        return {
            "exposure_validity": validity,
            "zero_rung": zero_rung,
            "verdict": "zero_rung_disagrees",
            "verdict_note": (
                "N84.4. The untrained boundary does not reproduce the rejection "
                "measured on this evaluation set before the milestone was "
                "written, so the partition, the geometry or the evaluation set "
                "is not the one that value belongs to."
            ),
        }

    # 3. N83.3 — a boundary that rejects everything is not detecting novelty.
    acceptances = [
        float(arm["measurements"]["known_novel_acceptance"])
        for entry in results
        for arm in entry["arms"].values()
    ]
    minimum_acceptance = float(gate_config["minimum_known_novel_acceptance"])
    known_control = {
        "minimum": min(acceptances),
        "mean": _mean(acceptances),
        "floor": minimum_acceptance,
        "passes": min(acceptances) >= minimum_acceptance,
    }
    if not known_control["passes"]:
        return {
            "exposure_validity": validity,
            "zero_rung": zero_rung,
            "known_novel_control": known_control,
            "verdict": "known_coverage_not_held",
            "verdict_note": (
                "N83.3. A rejection figure read beside a failed known-class "
                "control is a statement about the coverage match, not about the "
                "boundary."
            ),
        }

    # 4. The ladder.
    ladder: dict[str, Any] = {}
    for name, entries in by_cell.items():
        arms = sorted({arm for entry in entries for arm in entry["arms"]})
        ladder[name] = {
            "count": int(entries[0]["count"]),
            "diversity": int(entries[0]["diversity"]),
            "arms": {
                arm: {
                    "mean": _mean(_recalls(entries, arm)),
                    "spread": _spread(_recalls(entries, arm)),
                    "anisotropy": _mean(
                        [
                            float(
                                entry["arms"][arm]["measurements"]["tangent_anisotropy"]
                            )
                            for entry in entries
                        ]
                    ),
                    "shape_displacement": _mean(
                        [
                            float(entry["arms"][arm]["displacement"]["shape"])
                            for entry in entries
                        ]
                    ),
                }
                for arm in arms
            },
        }
        if "exposure" in arms and "moment_null" in arms:
            ladder[name]["null_margin"] = (
                ladder[name]["arms"]["exposure"]["mean"]
                - ladder[name]["arms"]["moment_null"]["mean"]
            )

    baseline = float(zero_rung["untrained_mean"])
    graded = [
        (entry["count"], entry["diversity"], name, entry["arms"]["exposure"]["mean"])
        for name, entry in ladder.items()
        if "exposure" in entry["arms"]
    ]
    best = max(graded, key=lambda item: item[3]) if graded else None
    spreads = [
        entry["arms"]["exposure"]["spread"]
        for entry in ladder.values()
        if "exposure" in entry["arms"] and entry["arms"]["exposure"]["spread"] is not None
    ]
    seed_spread = max(spreads) if spreads else 0.0

    beats_baseline = bool(best) and best[3] > baseline + seed_spread
    beats_null = bool(best) and (
        ladder[best[2]].get("null_margin") or 0.0
    ) > seed_spread

    if beats_baseline and beats_null:
        verdict = "exposure_governs_rejection"
        note = (
            f"The best rung ({best[2]}) rejects {best[3]:.5f} against an "
            f"untrained {baseline:.5f} and beats its own moment-matched null by "
            f"more than the seed spread of {seed_spread:.5f}."
        )
    elif beats_baseline:
        verdict = "not_separable_from_null"
        note = (
            "N84.3. Exposure raises rejection above the untrained boundary, but "
            "so does the same number of Gaussian negatives carrying only the "
            "exposure sample's mean and covariance. What the boundary used was "
            "not the content of the images."
        )
    else:
        verdict = "ladder_flat"
        note = (
            "Rejection does not rise above the untrained boundary at any "
            "feasible rung. With M83's synthetic route already closed, this "
            "closes the open-set leg for this parameterisation: the limitation "
            "is not how many negatives were available."
        )

    return {
        "exposure_validity": validity,
        "zero_rung": zero_rung,
        "known_novel_control": known_control,
        "ladder": ladder,
        "infeasible_cells": config["ladder"]["infeasible_cells"],
        "infeasible_cells_note": config["ladder"]["infeasible_cells_note"],
        "baseline": baseline,
        "seed_spread": seed_spread,
        "best_cell": best[2] if best else None,
        "best_mean": best[3] if best else None,
        "beats_baseline": beats_baseline,
        "beats_null": beats_null,
        "verdict": verdict,
        "verdict_note": note,
    }


def _check_ladder(config: dict[str, Any]) -> dict[str, Any]:
    """Draw every registered cell before any fitting starts.

    A cell the pool cannot supply is a configuration error, not a result, and
    it must surface in seconds rather than after the workers have spent an hour
    on the cells that happened to come first. The check calls the real sampler
    on the real pool labels, so there is no second definition of feasibility
    that could drift away from the one training uses.
    """
    index_path = _resolve(config["exposure"]["path"])
    labels = np.load(index_path.parent / "arrays" / "labels.npy").astype(np.int64)
    if bool(config["exposure"]["stratified_only"]):
        labels = labels[np.load(index_path.parent / "arrays" / "stratified.npy")]
    labels = labels[labels <= int(config["exposure"]["last_label"])]

    offset = int(config["ladder"]["sample_seed_offset"])
    seed = int(config["seeds"][0])
    for cell in config["ladder"]["cells"]:
        try:
            sample_exposure(
                labels,
                count=int(cell["count"]),
                diversity=int(cell["diversity"]),
                seed=offset + seed,
            )
        except ValueError as error:
            raise ValueError(
                f"M84 ladder cell {cell['name']} cannot be drawn: {error}. "
                "Either the pool is smaller than registration assumed or the "
                "cell was never feasible; declare it in infeasible_cells with "
                "its reason rather than letting the run discover it."
            ) from error
    return {
        "pool_rows": int(len(labels)),
        "pool_classes": int(len(np.unique(labels))),
        "cells_drawn": len(config["ladder"]["cells"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M84 out-group exposure ladder")
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
    parser.add_argument(
        "--cells",
        type=str,
        nargs="*",
        default=None,
        help="Override the registered cells by name. For smoke runs only; a "
        "subset does not seal.",
    )
    arguments = parser.parse_args()

    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    output_dir = arguments.output or _resolve(config["output_dir"])
    registered_seeds = [int(seed) for seed in config["seeds"]]
    seeds = arguments.seeds if arguments.seeds is not None else registered_seeds
    cell_names = [str(cell["name"]) for cell in config["ladder"]["cells"]]
    chosen = arguments.cells if arguments.cells is not None else cell_names
    indices = [cell_names.index(name) for name in chosen]
    sealed = seeds == registered_seeds and chosen == cell_names

    tasks = [(seed, index) for index in indices for seed in seeds]
    feasibility = _check_ladder(config)
    print(
        f"M84 exposure ladder: {len(chosen)} cells x {len(seeds)} seeds "
        f"= {len(tasks)} fits' worth of work"
    )
    print(
        f"  pool {feasibility['pool_rows']} rows in "
        f"{feasibility['pool_classes']} classes; all "
        f"{feasibility['cells_drawn']} registered cells draw"
    )

    started = time.time()
    workers = min(int(config["threading"]["worker_count"]), len(tasks))
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(config,)
    ) as pool:
        results = list(pool.map(_worker_run, tasks))
    elapsed = time.time() - started

    gate = _gate(results, config)
    evidence = {
        "schema_version": 1,
        "milestone": "M84",
        "program": "v13",
        "hypothesis": config["registered_hypothesis"],
        "registration_notes": config["registration_notes"],
        "seeds": seeds,
        "cells": chosen,
        "sealed": sealed,
        "corpus_sha256": config["corpus"]["sha256"],
        "openset_feature_hash": config["openset"]["feature_hash"],
        "exposure_feature_hash": config["exposure"]["feature_hash"],
        "geometry": config["geometry"],
        "partition": config["partition"],
        "ladder": config["ladder"],
        "training": config["training"],
        "coverage": config["coverage"],
        "per_cell": results,
        "gate": gate,
        "final_labels_opened": False,
        "elapsed_seconds": round(elapsed, 1),
    }
    evidence["configuration_hash"] = payload_hash(config)
    evidence["evidence_hash"] = payload_hash(evidence)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)

    print(f"verdict: {gate['verdict']}  sealed: {sealed}")
    validity = gate["exposure_validity"]
    print(
        f"exposure validity: minimum active fraction "
        f"{validity['minimum_active_fraction']} floor {validity['floor']} "
        f"passes {validity['passes']}"
    )
    if "zero_rung" not in gate:
        print(gate["verdict_note"])
        return
    zero = gate["zero_rung"]
    print(
        f"zero rung: untrained {zero['untrained_mean']} expected "
        f"{zero['expected']} known_only {zero['known_only_mean']} passes "
        f"{zero['passes']}"
    )
    if "ladder" not in gate:
        print(gate["verdict_note"])
        return
    print(
        f"known novel control {gate['known_novel_control']['mean']} passes "
        f"{gate['known_novel_control']['passes']}"
    )
    for name in [str(cell["name"]) for cell in config["ladder"]["cells"]]:
        if name not in gate["ladder"]:
            continue
        entry = gate["ladder"][name]
        parts = " | ".join(
            f"{arm} {values['mean']:.5f} aniso {values['anisotropy']:.4f}"
            for arm, values in sorted(entry["arms"].items())
        )
        print(f"  {name:12s} N={entry['count']:5d} d={entry['diversity']:3d}  {parts}")
    print(gate["verdict_note"])
    print(f"wrote {output_dir / 'evidence.json'} in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
