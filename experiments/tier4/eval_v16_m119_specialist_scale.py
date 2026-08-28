"""M119 — Per-domain specialists x data scaling (A5 x M116).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v18.md`` section 5.5 and
``experiments/configs/v16/m119_specialist_scale.json``.

Question. A5 showed per-domain routed specialists beat dense at matched-or-
lower cost (5/6 domains, 1.4-7.2x fewer MACs); M116 showed the global frozen
family's Q(n) is steep (overtakes dense r42). M119 composes the two: do
per-domain specialists' per-domain Q_d(n) curves rise at least as steeply as
the global arm's per-domain curves? Lower domain entropy may make specialists
scale faster with their own data.

Arms. For each of the 6 domains: the A5-exact per-domain specialist (random
whitened-patch candidates from that domain, seeded permutation pick, 512
atoms, triangle encode, closed-form ridge penalty 1.0) fitted on the first n
rows of the domain's train rows for n in {0.1, 0.2, 0.4, 0.7, 1.0} x n_d, and
scored on the domain's FULL test rows. The baseline steepness per domain is
M116's sealed global-arm per-domain accuracy over its global ladder (the
scale-fair quantity: each arm's gain over its own ladder).

Gate: KS fired if ANY domain's specialist gain Q_d(n_full) - Q_d(n_first) is
< 0.5 x the global arm's per-domain gain over M116's ladder for that domain.

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m119_specialist_scale
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v15_m107_dense import _score, _verify_pixel_identity
from experiments.tier4.eval_v16_a5_routed import (
    _build_whitener,
    _domain_candidates,
)
from experiments.tier4.eval_v16_m108_dictionary import (
    _encode_block_device,
    _verify_device,
)
from experiments.tier4.eval_v16_m109_trunk import (
    _load_corpus,
    _parity_guard,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m119_specialist_scale.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m119_specialist_scale"
M116_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m116_scale" / "evidence.json"

KS_GRAIN = 0.5
PATCH_DIM = 108


def _specialist_at_n(corpus, domain, atoms, whitener, pool_grid, classes,
                     device, n) -> float:
    """Fit the domain-d specialist on the first n of its train rows; score its
    full test rows. Returns accuracy."""
    rows_d = np.where(corpus["train_domains"] == domain)[0]
    cand = _domain_candidates(corpus, domain, whitener)
    order = np.random.default_rng([11, 100]).permutation(len(cand))[:atoms]
    dictionary = cand[order]
    table = torch.from_numpy(np.ascontiguousarray(dictionary)).to(torch.float32)
    table = table.to(device)

    acc = RidgeAccumulator(atoms * pool_grid * pool_grid, classes)
    for start in range(0, n, 64):
        take = rows_d[start:start + 64]
        block = _encode_block_device(corpus["train_images"][take], table,
                                     whitener, pool_grid)
        acc.add(block, corpus["train_labels"][take])
    solutions = acc.solve_many([1.0])
    standardise = acc.standardiser()

    test_rows_d = np.where(corpus["test_domains"] == domain)[0]
    correct = 0
    for start in range(0, len(test_rows_d), 64):
        take = test_rows_d[start:start + 64]
        block = _encode_block_device(corpus["test_images"][take], table,
                                     whitener, pool_grid)
        correct += int(_score(solutions[1.0], standardise(block),
                              corpus["test_labels"][take]).sum())
    return correct / len(test_rows_d)


def run_m119(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    torch.set_num_threads(config["numerics"]["torch_threads"])
    torch.manual_seed(config["numerics"]["seed"])
    configure_external_cache_environment()
    device_report = _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("parity guard at startup", flush=True)
    parity = _parity_guard(torch, config, device)

    print("loading corpus", flush=True)
    corpus, train_index, test_index = _load_corpus(config)
    classes = int(corpus["train_labels"].max()) + 1
    size = config["corpus"]["image_size"]
    for split, idx in (("train", train_index), ("test", test_index)):
        _verify_pixel_identity(split, idx, corpus[f"{split}_images"], size,
                               config["corpus"]["pixel_identity_rows"])

    rep = config["sparse"]
    atoms = int(rep["atoms_per_domain"])
    pool_grid = int(rep["pool_grid"])
    fractions = [float(f) for f in config["scaling"]["n_ladder_fractions"]]
    smoke_skip = bool(config.get("_smoke_skip_gates", False))

    print("building global whitener (M108 exact)", flush=True)
    whitener = _build_whitener(config, corpus)

    m116 = json.loads(M116_EVIDENCE.read_text(encoding="utf-8"))
    global_per_domain = {}   # domain -> accuracy at M116's n_min and n_max
    gcurve = m116["sparse"]["curve"]
    for d in range(6):
        global_per_domain[d] = {
            "n_min": gcurve[0]["per_domain"][d],
            "n_max": gcurve[-1]["per_domain"][d],
        }

    specialist_curves: dict[str, Any] = {}
    for d in range(6):
        rows_d = np.where(corpus["train_domains"] == d)[0]
        n_d = len(rows_d)
        ladder = [min(n_d, max(1, int(round(f * n_d)))) for f in fractions]
        ladder = list(dict.fromkeys(ladder))
        curve = {}
        for n in ladder:
            acc = _specialist_at_n(corpus, d, atoms, whitener, pool_grid,
                                   classes, device, n)
            curve[str(n)] = float(acc)
            print(f"    domain {d} n={n}/{n_d}: {acc:.4f}", flush=True)
        specialist_curves[str(d)] = {
            "n_domain_rows": int(n_d),
            "ladder": ladder,
            "accuracy": curve,
        }

    # ---- gate -------------------------------------------------------------
    ks = {"registered": "specialist per-domain gain >= 0.5 x global arm's "
                        "per-domain gain (M116 ladder), per domain",
          "domains": {}}
    fired_any = False
    for d in range(6):
        curve = specialist_curves[str(d)]
        ladder = curve["ladder"]
        gain_spec = curve["accuracy"][str(ladder[-1])] - curve["accuracy"][
            str(ladder[0])]
        gain_glob = global_per_domain[d]["n_max"] - global_per_domain[d]["n_min"]
        ratio = (gain_spec / gain_glob) if gain_glob > 0 else None
        fired = gain_spec < KS_GRAIN * gain_glob
        fired_any = fired_any or fired
        ks["domains"][str(d)] = {
            "specialist_gain": float(gain_spec),
            "global_gain": float(gain_glob),
            "ratio": ratio,
            "fired": bool(fired),
            "specialist_full": float(curve["accuracy"][str(ladder[-1])]),
            "global_full": float(global_per_domain[d]["n_max"]),
        }
    ks["fired"] = bool(fired_any)
    gates = {"kill_switch_specialist_steepness": ks,
             "_smoke_skip": smoke_skip}

    evidence = {
        "milestone": "M119",
        "admissible_as_evidence": not inadmissible,
        "registered_in": config.get("registered_in"),
        "question": ("do per-domain specialists' Q_d(n) curves rise at least "
                     "as steeply as the global arm's per-domain curves?"),
        "config_file": Path(config_path).name,
        "config": config,
        "device": device_report,
        "parity_guard": parity,
        "specialist_curves": specialist_curves,
        "global_per_domain_m116": global_per_domain,
        "gates": gates,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM119 complete -> {output_dir / 'evidence.json'}", flush=True)
    for d in range(6):
        c = specialist_curves[str(d)]
        print(f"  domain {d}: spec full {c['accuracy'][str(c['ladder'][-1])]:.4f} "
              f"gain {ks['domains'][str(d)]['specialist_gain']:+.4f} "
              f"ratio {ks['domains'][str(d)]['ratio']}", flush=True)
    print(f"  KS fired: {ks['fired']}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m119(args.config, args.output)


if __name__ == "__main__":
    main()
