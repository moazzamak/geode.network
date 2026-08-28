"""M77 — probe-degeneracy forensics (GEODE v13).

Registered hypothesis H77: the v12 probe objective cannot move the open-space
boundary, and the M72 "unchanged from initialization" result together with the
M74 probe-ablation null are consequences of that defect rather than evidence
about probe training.

Operands
--------
``O77.1``
    Probe hinge decomposition per epoch: own-class probe score, minimising
    class, hinge magnitude, and the adaptive target that drives it.
``O77.2``
    Analytic scale invariance of own-class probe scores.
``O77.3``
    Gradient norm of the probe term with respect to the fitted extents.
``O77.4``
    Attribution of the observed probe-loss decrease to the moving target versus
    to probes actually being pushed outward.

The milestone reads only the sealed M73 configuration and its source data. No
final-confirmation labels are opened.

Reproduce with::

    .\\.venv\\Scripts\\python.exe -m experiments.tier4.eval_v13_m77_probe_degeneracy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v12_metric_fields import (
    initialize_projected_metric_fields,
    train_projected_metric_fields,
)
from experiments.common.v13_probe_forensics import (
    SCALE_RELATIVE_FAMILIES,
    probe_scale_invariance,
    train_projected_metric_fields_instrumented,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v9_m51_surface_diagnostics import _partition_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v13" / "m77_probe_degeneracy.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v13" / "m77_probe_degeneracy"


def _resolve(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("M77 paths must remain inside the repository")
    return resolved


def _verify(specification: dict[str, str]) -> Path:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"M77 immutable artifact hash mismatch: {path}")
    return path


def _history_delta(
    left: list[dict[str, float]], right: list[dict[str, float]]
) -> float:
    if len(left) != len(right):
        raise ValueError("history lengths differ; instrumentation is not faithful")
    worst = 0.0
    for first, second in zip(left, right):
        if set(first) != set(second):
            raise ValueError("history keys differ; instrumentation is not faithful")
        for key in first:
            worst = max(worst, abs(float(first[key]) - float(second[key])))
    return worst


def run_m77(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    m73_config_path = _verify(config["m73_config"])
    _verify(config["m73_index"])
    m73 = json.loads(m73_config_path.read_text(encoding="utf-8"))
    source = json.loads(
        _resolve(m73["source_config"]["path"]).read_text(encoding="utf-8")
    )

    seed = int(m73["seed"])
    loaded = _load_seed_data(source["seed_inputs"][str(seed)])
    train_x, train_y = loaded["datasets"]["train"]
    dev_x, dev_y = loaded["datasets"]["dev"]
    partitions = _partition_seed(
        train_y,
        dev_y,
        seed=seed,
        known_classes=np.asarray(m73["known_classes"], dtype=np.int64),
        unknown_classes=np.asarray(m73["proxy_unknown_classes"], dtype=np.int64),
        geometry_fraction=float(m73["geometry_fraction"]),
    )
    geometry_x = train_x[partitions["geometry_fit"]]
    geometry_y = train_y[partitions["geometry_fit"]]

    initial = initialize_projected_metric_fields(
        geometry_x,
        geometry_y,
        output_dimension=int(m73["projection_dimension"]),
        rank=int(m73["rank"]),
    )
    training = m73["training"]
    probe_families = tuple(training["trained_probe_families"])
    arguments = {
        "epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "classification_temperature": float(training["classification_temperature"]),
        "target_score": float(training["target_score"]),
        "separation_margin": float(training["separation_margin"]),
        "probe_margin_multiplier": float(training["probe_margin_multiplier"]),
        "loss_weights": {
            name: float(value) for name, value in training["loss_weights"].items()
        },
        "collapse_weight": float(training["collapse_weight"]),
        "probe_families": probe_families,
        "seed": seed,
    }

    # O77.2 — analytic invariance at the initial state.
    invariance = probe_scale_invariance(
        initial.fields,
        families=probe_families,
        scale_factors=config["scale_invariance"]["scale_factors"],
        seed=seed,
    )

    # Faithfulness check: the instrumented loop must reproduce v12 exactly.
    reference_state, reference_history = train_projected_metric_fields(
        initial, geometry_x, geometry_y, **arguments
    )
    state, history, diagnostics = train_projected_metric_fields_instrumented(
        initial, geometry_x, geometry_y, **arguments
    )
    history_delta = _history_delta(reference_history, history)
    state_match = payload_hash(reference_state.to_dict()) == payload_hash(
        state.to_dict()
    )

    # O77.2 at the trained state as well.
    trained_invariance = probe_scale_invariance(
        state.fields,
        families=probe_families,
        scale_factors=config["scale_invariance"]["scale_factors"],
        seed=seed,
    )

    first, last = diagnostics[0], diagnostics[-1]
    probe_loss_drop = float(
        reference_history[0]["probe"] - reference_history[-1]["probe"]
    )
    target_drop = float(first["probe_target"] - last["probe_target"])
    score_rise = float(last["mean_minimum_score"] - first["mean_minimum_score"])
    explained = 0.0 if probe_loss_drop == 0.0 else target_drop / probe_loss_drop

    gate_config = config["gate"]
    tolerance = float(config["scale_invariance"]["invariance_tolerance"])
    max_probe_grad = max(
        float(entry["probe_grad_norm_log_tangent"]) for entry in diagnostics
    )
    invariant_families = sorted(
        set(invariance["invariant_families"])
        & set(trained_invariance["invariant_families"])
    )
    gate = {
        "instrumentation_faithful": bool(
            history_delta <= float(gate_config["maximum_history_reproduction_delta"])
            and state_match
        ),
        "history_reproduction_delta": history_delta,
        "trained_state_hash_match": bool(state_match),
        "invariant_families": invariant_families,
        "expected_scale_relative_families": sorted(
            set(SCALE_RELATIVE_FAMILIES) & set(probe_families)
        ),
        "invariance_confirmed": bool(
            set(invariant_families)
            >= (set(SCALE_RELATIVE_FAMILIES) & set(probe_families))
        ),
        "invariance_tolerance": tolerance,
        "maximum_probe_grad_norm_log_tangent": max_probe_grad,
        "probe_gradient_degenerate": bool(
            max_probe_grad
            <= float(
                gate_config["maximum_probe_grad_norm_log_tangent_for_degeneracy"]
            )
        ),
        "probe_loss_drop": probe_loss_drop,
        "adaptive_target_drop": target_drop,
        "minimum_probe_score_rise": score_rise,
        "loss_drop_explained_by_target_fraction": explained,
        "own_class_is_minimiser_fraction_first_epoch": float(
            first["own_class_is_minimiser_fraction"]
        ),
        "own_class_is_minimiser_fraction_last_epoch": float(
            last["own_class_is_minimiser_fraction"]
        ),
        "final_labels_opened": False,
    }
    gate["h77_confirmed"] = bool(
        gate["instrumentation_faithful"]
        and gate["invariance_confirmed"]
        and explained >= 1.0
    )

    evidence = {
        "schema_version": 1,
        "milestone": "M77",
        "program": "v13",
        "configuration_hash": sha256_file(config_path),
        "m73_configuration_hash": sha256_file(m73_config_path),
        "seed": seed,
        "trained_probe_families": list(probe_families),
        "partition_hashes": {
            name: payload_hash(indices.tolist())
            for name, indices in partitions.items()
        },
        "initial_state_hash": payload_hash(initial.to_dict()),
        "trained_state_hash": payload_hash(state.to_dict()),
        "scale_invariance_initial": invariance,
        "scale_invariance_trained": trained_invariance,
        "reference_history": reference_history,
        "instrumented_history": history,
        "probe_diagnostics": diagnostics,
        "gate": gate,
        "final_labels_opened": False,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    evidence = run_m77(arguments.config, arguments.output)
    print(json.dumps(evidence["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
