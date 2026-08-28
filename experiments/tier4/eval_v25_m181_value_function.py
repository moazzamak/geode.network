"""M181 — value function v1 numbers from the sealed M180 game, with
the registered H4 (no-bloat) and H5 (coverage-bonus) sensitivity
gates.

Registered in ``analysis/v25_m181_value_function_spec.md``. All
operands come from the SEALED M180 game; the only non-measured
fixtures are the H4/H5 synthetic gate arms, which are disclosed as
synthetic in the config (the H-gate instrument discipline).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m181_value_function.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m181_value_function")

CLASSES = 345


def _marginal(game: dict[str, float], arm: str) -> float:
    """LOO marginal from the grand coalition."""
    others = {
        "spm": game["V_ms_pool"],
        "ms": game["V_spm_pool"],
        "pool": game["V_spm_ms"],
    }
    return game["V_all"] - others[arm]


def _efficiency(width: int, widths: dict[str, int],
                accuracies: dict[str, float], accuracy: float,
                gamma: float) -> float:
    """cost_ref = lowest cost among singleton arms at-or-above c's
    accuracy; efficiency = (cost_ref / cost)^gamma."""
    cost = width * CLASSES
    qualified = [w * CLASSES for w_name, w in widths.items()
                 if accuracies[w_name] >= accuracy]
    cost_ref = min(qualified) if qualified else cost
    return (cost_ref / cost) ** gamma


def run_m181(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    game = config["game"]
    widths = config["cost_model"]["widths"]
    gamma = float(config["cost_model"]["gamma"])
    accuracies = {"spm": game["V_spm"], "ms": game["V_ms"],
                  "pool": game["V_pool"]}

    values: dict[str, Any] = {}
    for arm in ("spm", "ms", "pool"):
        marginal = _marginal(game, arm)
        eff = _efficiency(widths[arm], widths, accuracies,
                          accuracies[arm], gamma)
        values[arm] = {"loo_marginal": marginal, "efficiency": eff,
                       "value": marginal * eff}
        print(f"V({arm}) = {marginal:.6f} x {eff:.3f} "
              f"= {marginal * eff:.6f}", flush=True)

    # ---- H4: no bloat incentive -------------------------------------------
    h4 = config["h4_pair"]
    bloat_marginal = float(h4["bloat"]["marginal"])
    bloat_eff = _efficiency(int(h4["bloat"]["width"]), widths, accuracies,
                            accuracies["pool"], gamma)
    v_bloat = bloat_marginal * bloat_eff
    lean = h4["lean"]
    lean_marginal = float(lean["marginal"])
    lean_eff = _efficiency(int(lean["width"]), widths, accuracies,
                           accuracies["pool"], gamma)
    v_lean = lean_marginal * lean_eff
    h4_result = {"V_bloat": v_bloat, "V_lean": v_lean,
                 "passes": v_bloat < v_lean,
                 "note": "synthetic pair; if it failed, gamma rises"}
    print(f"H4: V(bloat)={v_bloat:.6f} vs V(lean)={v_lean:.6f} "
          f"passes={h4_result['passes']}", flush=True)

    # ---- H5: coverage bonus != accuracy bonus ------------------------------
    h5 = config["h5_pair"]
    bonus = float(h5["coverage_bonus"])
    opener_marginal = float(h5["axis_opener"]["marginal"])
    opener_eff = _efficiency(int(h5["axis_opener"]["width"]), widths,
                             accuracies, accuracies["pool"], gamma)
    v_opener = opener_marginal * opener_eff + bonus
    redundant_marginal = float(h5["redundant"]["marginal"])
    redundant_eff = _efficiency(int(h5["redundant"]["width"]), widths,
                                accuracies, accuracies["spm"], gamma)
    v_redundant = redundant_marginal * redundant_eff
    h5_result = {"V_axis_opener": v_opener, "V_redundant": v_redundant,
                 "passes": v_opener > v_redundant,
                 "note": "synthetic pair; if it failed, the bonus rises"}
    print(f"H5: V(opener)={v_opener:.6f} vs V(redundant)="
          f"{v_redundant:.6f} passes={h5_result['passes']}", flush=True)

    gates_pass = h4_result["passes"] and h5_result["passes"]
    evidence: dict[str, Any] = {
        "milestone": "M181",
        "cell": "value function v1 numbers + H4/H5 sensitivity gates",
        "admissible_as_evidence": not inadmissible,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "values": values,
        "ranking": sorted(values, key=lambda a: -values[a]["value"]),
        "h4": h4_result,
        "h5": h5_result,
        "verdict": {
            "gates_pass": gates_pass,
            "reading": ("V = LOO marginal x efficiency (gamma 1) ranks "
                        "spm > ms > pool; the H4 bloat pair and the H5 "
                        "coverage pair pass at the registered gamma and "
                        "bonus") if gates_pass
            else "a sensitivity gate failed; the registered consequence "
                 "applies (gamma rises / bonus rises)",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"values": values, "ranking": evidence["ranking"],
                      "gates_pass": gates_pass}, indent=1), flush=True)
    print(f"M181 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m181(args.config, args.output)


if __name__ == "__main__":
    main()
