"""M225d - the wider authored inverse set (prototype).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (M225d
REGISTERED, 20 Aug, dispatched this session). Same joint objective as
M225c (f_dim 32, frozen-swap term, reference + inverse-pair hinge) but
trained on FIVE inverse pairs and tested on the TWO registered
excluded pairs (convolution/deconvolution, sort/unsort) with the same
true-held-out protocol.

This runner reuses the sealed M225 training/gate machinery and only
changes (a) the transform-relations artifact (v2) and (b) the G6
split: the verdict is the min cosine over the EXCLUDED pairs; the
trained pairs are reported but do not decide. G3's reading stays
TRAINED-not-held-out (the M225c joint term), per the registration.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from experiments.common.data_cache import configure_external_cache_environment
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v25_m225_transform_analogies import (
    FROZEN_G3_TASKS,
    M225_DISSIMILAR,
    M225_SIMILAR,
    TRANSFORM_TASKS,
    _frozen_swap_loss,
    _gates,
    _rel_loss,
    _token_vec,
    _train,
)
from experiments.tier4.eval_v25_m224_fingerprint_v1_train import (
    TASK_DESCRIPTORS,
    _axis_key,
)
from geode.core.descriptor import AXES, normalise
from geode.core.fingerprint import FingerprintEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m225d_wider_inverse.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m225d_wider_inverse")
RELATIONS_PATH = REPO_ROOT / "analysis" / "fingerprint_relations_v0.json"
TRANSFORM_V2_PATH = (REPO_ROOT / "analysis"
                     / "fingerprint_relations_v2_transform.json")


def _gates_v2(enc, descs, relations, trel, tau, rho, inv_tau,
              exclude: list[list[str]] | None = None,
              tau_swap: float | None = None):
    """The M225c gate set with the M225d G6 split: the verdict reads
    ONLY the excluded (held-out) pairs; trained pairs are reported."""
    gates = _gates(enc, descs, relations, trel, tau, rho, inv_tau,
                   exclude, tau_swap)
    g6 = gates["g6_inverse_analogy"]
    excluded_keys = {f"{a}-{b}" for a, b in (exclude or [])}
    held_out_cos = {k: v for k, v in g6["analogy_cos"].items()
                    if k in excluded_keys}
    trained_cos = {k: v for k, v in g6["analogy_cos"].items()
                   if k not in excluded_keys}
    ref = trel["reference_inverse_pair"]
    # drop the reference from the trained report (it defines the
    # direction, not an analogy target)
    trained_cos = {k: v for k, v in trained_cos.items()
                   if k != f"{ref[0]}-{ref[1]}"}
    g6_held_out_pass = bool(min(held_out_cos.values()) >= inv_tau)
    gates["g6_inverse_analogy"]["held_out_cos"] = held_out_cos
    gates["g6_inverse_analogy"]["trained_cos"] = trained_cos
    gates["g6_inverse_analogy"]["held_out_min"] = (
        min(held_out_cos.values()) if held_out_cos else None)
    gates["g6_inverse_analogy"]["held_out_passed"] = g6_held_out_pass
    return gates


def run_m225d(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()

    trel = json.loads(TRANSFORM_V2_PATH.read_text(encoding="utf-8"))
    relations = json.loads(RELATIONS_PATH.read_text(encoding="utf-8"))
    schema = {**AXES, trel["axis"]: trel["vocabulary"]}

    f_dim = int(config["model"]["f_dim"])
    mlp_hidden = int(config["model"]["mlp_hidden"])
    seed = int(config["model"]["seed"])
    lr = float(config["training"]["lr"])
    steps = int(config["training"]["steps"])
    tau = float(config["training"]["rel_ordered_triple_tau"])
    rho = float(config["training"]["rel_polar_rho"])
    margin = float(config["training"]["margin_dissimilar"])
    inv_tau = float(config["training"]["inverse_analogy_tau"])
    tau_swap = config["training"].get("frozen_swap_tau")
    exclude = [list(p) for p in config.get("train_exclude", [])]

    # the v0 tasks get task.transform = identity (registered)
    descs = {}
    for name, d in TASK_DESCRIPTORS.items():
        nd = normalise({**d, "task.transform": "identity"})
        nd.axes[trel["axis"]] = "identity"  # normalise drops unknown axes
        descs[name] = nd
    for name, d in TRANSFORM_TASKS.items():
        nd = normalise(d)
        nd.axes[trel["axis"]] = d["task.transform"]
        descs[name] = nd

    enc = FingerprintEncoder(f_dim=f_dim, mlp_hidden=mlp_hidden,
                             seed=seed, axes=schema)
    enc.attr_heads = torch.nn.ModuleDict({
        _axis_key(axis): torch.nn.Linear(f_dim, len(vocab))
        for axis, vocab in schema.items()})

    print(f"training {steps} steps (v2 inverse set)", flush=True)
    history = _train(enc, descs, M225_SIMILAR, M225_DISSIMILAR,
                     relations, trel, steps, lr, tau, rho, margin,
                     inv_tau, schema, exclude, tau_swap)
    print(f"  final loss {history[-1]:.4f}", flush=True)

    print("gates", flush=True)
    gates = _gates_v2(enc, descs, relations, trel, tau, rho, inv_tau,
                      exclude, tau_swap)
    print(json.dumps(gates, indent=1)[:2400], flush=True)

    void = not gates["g1_deterministic"]
    evidence: dict[str, Any] = {
        "milestone": "M225d",
        "cell": "wider authored inverse set (prototype)",
        "admissible_as_evidence": True,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
        "interpretation_registered_before_running":
            config["interpretation_registered_before_running"],
        "schema_axes": schema,
        "transform_relations_used": trel,
        "gates": gates,
        "training": {"steps": steps, "final_loss": history[-1]},
        "not_shipped": "prototype only; the product ontology migration "
                       "is M226 (registered) and runs only if G1-G3+G6 "
                       "pass",
        "void": void,
        "void_reason": "G1 determinism failed" if void else "",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM225d complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m225d(args.config, args.output)


if __name__ == "__main__":
    main()
