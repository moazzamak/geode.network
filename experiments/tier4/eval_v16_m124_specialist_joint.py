"""M124 — Specialist joint surface (per-domain atoms x data).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v19.md`` section 5.3 and
``experiments/configs/v16/m124_specialist_joint.json``.

Question. M119 showed per-domain specialists inherit per-domain data scaling at
512 atoms. M117 showed the GLOBAL surface is super-additive (atoms and data
interact positively). M124 asks: is each domain's OWN atoms x data surface
super-additive too? If yes, the specialist route is the primary buy-back
candidate (cheap + scales + per-domain).

Cells per domain d: atoms {256, 512} x n {round(0.4*n_d), n_d}, A5-exact
specialist construction (nested dictionaries: 256 is a prefix of 512, mirroring
M117). t1: the (512, 0.4*n_d) and (512, n_d) cells reproduce M119's sealed
accuracies within 0.002 (two anchors per domain).

Gate: per-domain super-additivity at cell (256, 0.4*n_d) with the M117 margin
+0.005; KS fired if super-additivity holds on fewer than 4 of 6 domains.

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m124_specialist_joint
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
from experiments.tier4.eval_v15_m107_dense import _verify_pixel_identity
from experiments.tier4.eval_v16_a5_routed import _build_whitener
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v16_m109_trunk import (
    _load_corpus,
    _parity_guard,
)
from experiments.tier4.eval_v16_m119_specialist_scale import (
    _specialist_at_n,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m124_specialist_joint.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m124_specialist_joint"
M119_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m119_specialist_scale" / "evidence.json"

T1_TOLERANCE = 0.002
KS_MARGIN = 0.005
KS_MIN_DOMAINS = 4


def run_m124(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    atoms_ladder = [int(a) for a in rep["atoms_ladder"]]
    fractions = [float(f) for f in config["scaling"]["n_fractions"]]
    pool_grid = int(rep["pool_grid"])
    smoke_skip = bool(config.get("_smoke_skip_gates", False))
    gates: dict[str, Any] = {}

    print("building global whitener (M108 exact)", flush=True)
    whitener = _build_whitener(config, corpus)

    # M119 sealed anchors per domain: {d: {0.4: acc, 1.0: acc}} at 512 atoms
    m119 = json.loads(M119_EVIDENCE.read_text(encoding="utf-8"))
    anchors: dict[int, dict[int, float]] = {}
    for d in range(6):
        c = m119["specialist_curves"][str(d)]
        anchors[d] = {}
        for n in c["ladder"]:
            frac = n / c["n_domain_rows"]
            if any(abs(frac - f) < 0.02 for f in fractions):
                anchors[d][n] = float(c["accuracy"][str(n)])

    surface: dict[str, Any] = {}
    for d in range(6):
        rows_d = np.where(corpus["train_domains"] == d)[0]
        n_d = len(rows_d)
        ns = sorted({min(n_d, max(1, int(round(f * n_d)))) for f in fractions})
        surface[str(d)] = {"n_domain_rows": int(n_d), "ns": ns, "atoms": {}}
        for atoms in atoms_ladder:
            surface[str(d)]["atoms"][str(atoms)] = {}
            for n in ns:
                acc = _specialist_at_n(corpus, d, atoms, whitener, pool_grid,
                                       classes, device, n)
                surface[str(d)]["atoms"][str(atoms)][str(n)] = float(acc)
                print(f"    domain {d} atoms {atoms} n={n}/{n_d}: {acc:.4f}",
                      flush=True)

    def Q(d: int, atoms: int, n: int) -> float:
        return float(surface[str(d)]["atoms"][str(atoms)][str(n)])

    # ---- t1 (two anchors per domain at 512 atoms) --------------------------
    if not smoke_skip:
        for d in range(6):
            for n, ref in anchors[d].items():
                measured = Q(d, atoms_ladder[-1], n)
                delta = measured - ref
                key = f"t1_delta_d{d}_n{n}"
                gates[key] = delta
                if abs(delta) > T1_TOLERANCE:
                    print(f"  t1 FAILED (d={d}, n={n}): {measured:.4f} vs "
                          f"M119 {ref:.4f} (delta {delta:+.5f})", flush=True)
                    write_canonical_json(output_dir / "evidence.json", {
                        "milestone": "M124", "admissible_as_evidence": False,
                        "void": True,
                        "void_reason": "t1 specialist reproduction failed",
                        "domain": d, "n": n, "measured": measured,
                        "reference": ref, "t1_delta": delta,
                    })
                    return {"admissible_as_evidence": False, "void": True}
                print(f"  t1 (d={d}, n={n}) delta {delta:+.5f}", flush=True)
        gates["t1_registered"] = ("(512, 0.4*n_d) and (512, n_d) reproduce "
                                  "M119 sealed accuracies within 0.002")

    # ---- KS: per-domain super-additivity at (256, 0.4*n_d) -----------------
    a0, a1 = atoms_ladder
    ks: dict[str, Any] = {
        "registered": ("per-domain super-additivity at cell (256, 0.4*n_d): "
                       "joint Q(512,n_d)-Q(256,0.4*n_d) > [Q(512,0.4*n_d)-"
                       "Q(256,0.4*n_d)] + [Q(256,n_d)-Q(256,0.4*n_d)] + 0.005; "
                       "KS fired if fewer than 4/6 domains super-additive"),
        "domains": {}}
    n_super = 0
    for d in range(6):
        ns = surface[str(d)]["ns"]
        n_lo, n_hi = ns[0], ns[-1]
        base = Q(d, a0, n_lo)
        joint = Q(d, a1, n_hi) - base
        axes = (Q(d, a1, n_lo) - base) + (Q(d, a0, n_hi) - base)
        excess = joint - axes
        super_add = excess > KS_MARGIN
        n_super += int(super_add)
        ks["domains"][str(d)] = {
            "base_256_04": float(base),
            "atoms_axis_512_04": float(Q(d, a1, n_lo)),
            "data_axis_256_10": float(Q(d, a0, n_hi)),
            "joint_512_10": float(Q(d, a1, n_hi)),
            "joint_gain": float(joint),
            "axes_sum": float(axes),
            "excess": float(excess),
            "margin": KS_MARGIN,
            "super_additive": bool(super_add),
        }
    ks["n_super_additive_domains"] = n_super
    ks["min_domains"] = KS_MIN_DOMAINS
    ks["fired"] = n_super < KS_MIN_DOMAINS
    gates["kill_switch_specialist_superadditive"] = ks
    gates["_smoke_skip"] = smoke_skip

    evidence = {
        "milestone": "M124",
        "admissible_as_evidence": not inadmissible,
        "registered_in": config.get("registered_in"),
        "question": ("is each domain's OWN atoms x data surface super-additive, "
                     "mirroring the global M117 surface?"),
        "config_file": Path(config_path).name,
        "config": config,
        "device": device_report,
        "parity_guard": parity,
        "atoms_ladder": atoms_ladder,
        "n_fractions": fractions,
        "surface": surface,
        "m119_anchors": {str(d): {str(n): v for n, v in anchors[d].items()}
                         for d in anchors},
        "gates": gates,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM124 complete -> {output_dir / 'evidence.json'}", flush=True)
    for d in range(6):
        k = ks["domains"][str(d)]
        print(f"  domain {d}: joint {k['joint_gain']:+.4f} axes "
              f"{k['axes_sum']:+.4f} excess {k['excess']:+.4f} "
              f"super {k['super_additive']}", flush=True)
    print(f"  KS fired: {ks['fired']} "
          f"({ks['n_super_additive_domains']}/6 super-additive)", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m124(args.config, args.output)


if __name__ == "__main__":
    main()
