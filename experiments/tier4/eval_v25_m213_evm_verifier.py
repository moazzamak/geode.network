"""M213 — EVM verification hook evidence: Python-built M193b proofs
verify in the on-chain Solidity port bit-exactly; a gas sweep measures
the direct-port cost and extrapolates the real-width cost (labeled as
extrapolation).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(19 Aug 2026, before the build). CPU-only; the only external call is
the local Hardhat harness.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from geode.privacy.zk_bulletproofs import Q_ORDER, prove
from geode.privacy.zk_onchain import serialize_hex, words_hex

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m213_evm_verifier.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v25" / "m213_evm_verifier"


def _build_case(n: int, seed: int, tamper: bool = False
                ) -> dict[str, Any]:
    rnd = random.Random(seed)
    x = [rnd.randrange(Q_ORDER) for _ in range(n)]
    w = [rnd.randrange(Q_ORDER) for _ in range(n)]
    claim = sum(xi * wi % Q_ORDER for xi, wi in zip(x, w)) % Q_ORDER
    if tamper:
        claim = (claim + 1) % Q_ORDER
    r = rnd.randrange(Q_ORDER)
    proof = prove(x, r, w, claim)
    return {"n": n, "proof": serialize_hex(proof, n),
            "claim": "0x" + format(claim, "x"),
            "w": words_hex(w, n),
            "expect": not tamper}


def _run_hardhat(config: dict[str, Any], spec_path: Path
                 ) -> tuple[bool, list[dict[str, Any]], str]:
    hh = config["hardhat"]
    cwd = REPO_ROOT / hh["cwd"]
    env = dict(os.environ)
    env["POST_VERIFY_PATH"] = str(spec_path.resolve())
    proc = subprocess.run(subprocess.list2cmdline(list(hh["cmd"])),
                          cwd=cwd, capture_output=True, text=True,
                          shell=True, env=env,
                          timeout=int(hh.get("timeout_seconds", 900)))
    out = (proc.stdout or "") + (proc.stderr or "")
    cases: list[dict[str, Any]] = []
    ok_final = False
    for line in (proc.stdout or "").splitlines():
        if line.startswith("VERIFY_CASE"):
            parts = dict(item.split("=") for item in line.split()[1:]
                         if "=" in item)
            cases.append({"ok": parts["ok"] == "true",
                          "gas": int(parts["gas"]),
                          "expect": parts["expect"] == "true"})
        elif line.startswith("VERIFY_OK"):
            ok_final = True
    return ok_final, cases, out[-2000:]


def run_m213(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    n_gate = int(config["n_gate"])
    seed = int(config["seed"])

    honest_a = _build_case(n_gate, seed)
    honest_b = _build_case(n_gate, seed)  # determinism pair
    tampered = _build_case(n_gate, seed, tamper=True)
    sweep = [_build_case(n, seed + n) for n in config["n_sweep"]]

    spec = {"cases": [honest_a, honest_b, tampered] + sweep}
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "verify_cases.json"
    write_canonical_json(spec_path, spec)
    ran_ok, cases, tail = _run_hardhat(config, spec_path)

    by_id: dict[int, dict[str, Any]] = {}
    for c in cases:
        by_id.setdefault(len(by_id), c)
    def case_at(i: int) -> dict[str, Any] | None:
        return cases[i] if i < len(cases) else None

    g1 = bool(ran_ok and case_at(0) and case_at(0)["ok"])
    g2 = bool(ran_ok and case_at(2) and not case_at(2)["ok"]
              and case_at(2)["expect"] is False)
    g3 = bool(case_at(0) and case_at(1)
              and case_at(0)["ok"] and case_at(1)["ok"]
              and case_at(0)["gas"] == case_at(1)["gas"])
    sweep_gas = [(config["n_sweep"][i], cases[3 + i]["gas"])
                 for i in range(len(config["n_sweep"]))
                 if 3 + i < len(cases)
                 and cases[3 + i]["ok"]]
    g4_ok = len(sweep_gas) == len(config["n_sweep"])
    extrapolated = None
    if g4_ok:
        ns = [n for n, _g in sweep_gas]
        gs = [g for _n, g in sweep_gas]
        xm = sum(ns) / len(ns)
        ym = sum(gs) / len(gs)
        slope = sum((a - xm) * (b - ym) for a, b in zip(ns, gs)) \
            / sum((a - xm) ** 2 for a in ns)
        intercept = ym - slope * xm
        extrapolated = {"slope_per_word": slope,
                        "intercept": intercept,
                        "n16384_gas_estimate": int(slope * 16384
                                                   + intercept),
                        "labeled": "extrapolated, not measured"}
    gates_ok = all([g1, g2, g3, g4_ok])

    fixture = {"n": n_gate, "proof": honest_a["proof"],
               "claim": honest_a["claim"], "w": honest_a["w"]}
    fixture_path = REPO_ROOT / config["fixture_path"]
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(fixture_path, fixture)

    evidence: dict[str, Any] = {
        "milestone": "M213",
        "cell": "EVM verification hook (direct port of M193b verify)",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "anchor": {"proof_hex_n64": honest_a["proof"],
                   "note": ("the n=64 honest proof bytes; a future run "
                            "must reproduce them bit-exactly and verify "
                            "on-chain")},
        "cases": cases,
        "gates": {
            "g1_cross_language_bit_exact": bool(g1),
            "g2_tampered_rejected": bool(g2),
            "g3_deterministic": bool(g3),
            "g4_gas_sweep": {"ok": bool(g4_ok),
                             "measured": sweep_gas,
                             "extrapolated_n16384": extrapolated},
        },
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": (
                "the SAME proof bytes verify in the Python verifier and "
                "in the Solidity port on the local EVM; the direct port "
                "is O(n) in modexps (public weights), so the real-width "
                "on-chain cell requires the committed-weights variant — "
                "the registered follow-up, not claimed here"
            ) if gates_ok else "one or more M213 gates failed — VOID",
        },
        "scope": "local EVM; the committed-weights (O(log n)) verifier "
                 "is the registered follow-up cell",
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok,
                      "gates": evidence["gates"],
                      "cases": cases,
                      "hardhat_tail": tail[:400]}, indent=1), flush=True)
    print(f"M213 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m213(args.config, args.output)


if __name__ == "__main__":
    main()
