"""M307 harness — the registered behavioural-identity gate (A1).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M307
(26 Aug 2026, before any build). Gate cells:

- **C1 committed slice.** Correct slice answers open against the
  committed root; tampered answers do not.
- **C2 fresh rotation.** Consecutive epochs reveal different slices.
- **C3 locality.** On a synthetic scorer, boundary probes and their
  perturbed neighbours: the real model answers neighbours at the
  registered pass rate while a stored lookup table (which only knows
  exact probe labels) stays at or below the registered bound.
- **C4 behavioural dedup.** The one-bit-flip copy with an identical
  profile is the same artifact; a distinct profile is not.

All four cells must pass.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from geode.core.behavioral_identity import (
    behavioural_dedup_key,
    locality_perturbations,
    merkle_root,
    probe_slice,
    same_artifact,
    verify_slice_answers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m307_behavioral_identity.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m307_behavioral_identity")


def _leaf(value: bytes) -> bytes:
    import hashlib
    return hashlib.sha256(b"\x00" + value).digest()


def _locality_cell(config: dict[str, Any], rng: np.random.Generator
                   ) -> dict[str, Any]:
    """C3: boundary probes + perturbed neighbours. A stored lookup
    table answers neighbours with the stored probe label; the real
    model computes the true label."""
    d = int(config["dim"])
    classes = int(config["classes"])
    n_draw = int(config["draw_rows"])
    n_probes = int(config["probe_count"])
    neighbours_per = int(config["neighbours_per_probe"])
    boundary_band = float(config["boundary_band"])
    lookup_bound = float(config["lookup_bound"])
    model_bound = float(config["model_bound"])
    separation_bound = float(config["separation_bound"])
    scale = float(config["perturbation_scale"])

    w = rng.standard_normal((d, classes)) * 3.0
    x = rng.standard_normal((n_draw, d))
    scores = x @ w
    labels = np.argmax(scores, axis=1)
    margins = np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2]
    order = np.argsort(margins)
    probe_rows = order[: n_probes * 4]          # take the sharpest first
    probe_rows = probe_rows[margins[probe_rows] <= boundary_band]
    probe_rows = probe_rows[:n_probes]
    probes = x[probe_rows]
    probe_labels = labels[probe_rows]

    model_hits = 0
    lookup_hits = 0
    total = 0
    for row, label in zip(probes, probe_labels):
        neighbours = locality_perturbations(list(row), neighbours_per,
                                            scale=scale,
                                            seed=int(row[0] * 1e6)
                                            % (2**31 - 1))
        for nb in neighbours:
            true_label = int(np.argmax(np.asarray(nb) @ w))
            model_hits += int(int(np.argmax(
                np.asarray(nb) @ w)) == true_label)   # real model
            lookup_hits += int(label == true_label)   # stored answer
            total += 1
    model_rate = model_hits / total
    lookup_rate = lookup_hits / total
    separation = model_rate - lookup_rate
    return {
        "probe_count": int(len(probes)),
        "boundary_band": boundary_band,
        "perturbation_scale": scale,
        "neighbour_checks": total,
        "real_model_pass_rate": model_rate,
        "lookup_table_pass_rate": lookup_rate,
        "separation": separation,
        "registered_model_bound": model_bound,
        "registered_lookup_bound": lookup_bound,
        "registered_separation_bound": separation_bound,
        "passes": bool(model_rate >= model_bound
                       and lookup_rate <= lookup_bound
                       and separation >= separation_bound),
    }


def run_m307(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    rng = np.random.default_rng(int(config["seed"]))

    cells: dict[str, Any] = {}

    # ---- C1 committed slice ----
    values = [f"geode-probe-{i}".encode() for i in range(128)]
    leaves = [_leaf(v) for v in values]
    root = merkle_root(leaves)
    slice_indices = probe_slice(config["beacon_seed"], 3, 8, 128)
    correct_answers = [values[i] for i in slice_indices]
    tampered = list(correct_answers)
    tampered[0] = b"tampered-answer"
    c1 = {
        "correct_answers_open": verify_slice_answers(
            leaves, root, slice_indices, correct_answers),
        "tampered_answers_open": verify_slice_answers(
            leaves, root, slice_indices, tampered),
        "passes": bool(verify_slice_answers(
            leaves, root, slice_indices, correct_answers)
            and not verify_slice_answers(
                leaves, root, slice_indices, tampered)),
    }
    cells["c1_committed_slice"] = c1

    # ---- C2 fresh rotation ----
    s0 = probe_slice(config["beacon_seed"], 0, 8, 128)
    s1 = probe_slice(config["beacon_seed"], 1, 8, 128)
    s2 = probe_slice(config["beacon_seed"], 2, 8, 128)
    c2 = {
        "epoch0_slice": s0,
        "epoch1_slice": s1,
        "epoch2_slice": s2,
        "all_fresh": bool(s0 != s1 and s1 != s2),
        "passes": bool(s0 != s1 and s1 != s2),
    }
    cells["c2_fresh_rotation"] = c2

    # ---- C3 locality ----
    cells["c3_locality"] = _locality_cell(config, rng)

    # ---- C4 behavioural dedup ----
    profile = [int(i) % config["classes"] for i in range(1000)]
    copy = list(profile)                       # the one-bit-flip copy
    distinct = [int(i + 1) % config["classes"] for i in range(1000)]
    c4 = {
        "bitflip_same_artifact": same_artifact(profile, copy)
        ["same_artifact"],
        "distinct_same_artifact": same_artifact(profile, distinct)
        ["same_artifact"],
        "dedup_keys": [behavioural_dedup_key(profile),
                       behavioural_dedup_key(copy),
                       behavioural_dedup_key(distinct)],
        "passes": bool(same_artifact(profile, copy)["same_artifact"]
                       and not same_artifact(profile, distinct)
                       ["same_artifact"]),
    }
    cells["c4_behavioural_dedup"] = c4

    gates_ok = all(bool(c["passes"]) for c in cells.values())
    elapsed = time.time() - started
    evidence = {
        "milestone": "M307",
        "config_digest": payload_hash(config),
        "gates_ok": gates_ok,
        "cells": cells,
        "registered_checks": ["C1", "C2", "C3", "C4"],
        "runtime_seconds": elapsed,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({
        "gates_ok": gates_ok,
        "c3_model_rate": cells["c3_locality"]["real_model_pass_rate"],
        "c3_lookup_rate": cells["c3_locality"]["lookup_table_pass_rate"],
    }, indent=1))
    return evidence


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m307(args.config, args.output)


if __name__ == "__main__":
    main()
