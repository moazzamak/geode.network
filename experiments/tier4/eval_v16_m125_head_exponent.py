"""M125 — Head-width exponent (data-elasticity design rule).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v19.md`` section 5.4 and
``experiments/configs/v16/m125_head_exponent.json``.

Pure analysis of M117's sealed surface (atoms {1536, 3072, 6144} x
n {34500, 69000, 138000}). M117/M120 established that Q(n) steepness rises
with head width (0.094 / 0.115 / 0.134). M125 quantifies the dependence:
per-atoms n-axis gain exponents beta, then log(beta) vs log(width) -> gamma,
with the few-point disclosure. No new experiment; no GPU.

Reproduce with::

    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m125_head_exponent
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m125_head_exponent.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m125_head_exponent"
M117_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m117_scale" / "evidence.json"


def _per_atoms_exponent(ns: list[int], accs: list[float]) -> dict[str, float]:
    """log(gain) vs log(n) fit over the points where gain > 0."""
    n0 = ns[0]
    a0 = accs[0]
    xs, ys = [], []
    for n, a in zip(ns[1:], accs[1:]):
        g = a - a0
        if g > 0:
            xs.append(np.log(n / n0))
            ys.append(np.log(g))
    if len(xs) < 2:
        return {"n_points": len(xs), "beta": None}
    beta = float(np.polyfit(xs, ys, 1)[0])
    return {"n_points": len(xs), "beta": beta}


def run_m125(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    m117 = json.loads(M117_EVIDENCE.read_text(encoding="utf-8"))

    atoms_list = [int(a) for a in m117["atoms_ladder"]]
    n_list = [int(n) for n in m117["n_ladder"]]
    surface = m117["surface"]

    per_atoms = {}
    for a in atoms_list:
        cells = surface[str(a)]["cells"]
        # canonical JSON sorts dict keys; restore the numeric n order
        ns = sorted(int(k) for k in cells)
        accs = [float(cells[str(n)]["accuracy"]) for n in ns]
        width = int(surface[str(a)]["width"])
        exp = _per_atoms_exponent(ns, accs)
        exp["atoms"] = a
        exp["width"] = width
        exp["steepness"] = float(accs[-1] - accs[0])
        per_atoms[str(a)] = exp
        print(f"  atoms {a} (width {width}): beta {exp['beta']}, "
              f"steepness {exp['steepness']:.4f}", flush=True)

    # log(beta) vs log(width) regression over atoms with a fitted beta
    xs, ys = [], []
    for a in atoms_list:
        exp = per_atoms[str(a)]
        if exp["beta"] is not None:
            xs.append(np.log(exp["width"]))
            ys.append(np.log(exp["beta"]))
    gamma = None
    if len(xs) >= 2:
        gamma = float(np.polyfit(xs, ys, 1)[0])

    widths = [per_atoms[str(a)]["width"] for a in atoms_list]
    betas = [per_atoms[str(a)]["beta"] for a in atoms_list]
    betas_monotone = all(
        betas[i] is not None and betas[i + 1] is not None
        and betas[i + 1] > betas[i]
        for i in range(len(betas) - 1)
    )

    evidence = {
        "milestone": "M125",
        "admissible_as_evidence": True,
        "registered_in": config.get("registered_in"),
        "question": ("what is the head-width exponent gamma in "
                     "Q(n)-steepness ~ width^gamma (the data-elasticity "
                     "design rule)?"),
        "config_file": Path(config_path).name,
        "config": config,
        "source": "M117 sealed surface",
        "per_atoms": per_atoms,
        "width_regression": {
            "log_width": xs,
            "log_beta": ys,
            "gamma": gamma,
            "n_points": len(xs),
            "disclosure": "3 n-points per atom count and 3 widths: the "
                          "exponents and gamma are disclosure-level "
                          "quantifications of an already-sealed monotone "
                          "effect, not laws.",
        },
        "monotonicity": {
            "betas_increasing_in_width": bool(betas_monotone),
            "betas": betas,
            "widths": widths,
            "note": "if betas are not monotone in width, 'data-elastic "
                    "compute' is a 2-3 point trend, not a rule.",
        },
        "gates": {"no_kill_switch": True},
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM125 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(f"  gamma (beta ~ width^gamma): {gamma} "
          f"(betas monotone in width: {betas_monotone})", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m125(args.config, args.output)


if __name__ == "__main__":
    main()
