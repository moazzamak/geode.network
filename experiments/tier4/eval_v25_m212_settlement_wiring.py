"""M212 — settlement wire evidence: orchestrator route records become
deterministic, chain-anchored CreditLedger attribution batches, and the
built payload posts to the deployed contract on the local EVM with no
revert (the cross-language gate).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(19 Aug 2026, before the build). CPU-only; the only external call is the
local Hardhat harness. Routing accuracies: the ms arm uses the sealed
M210b per-domain table; competitor arms are synthetic (stated). The
wire, not the accuracy, is the M212 claim.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from geode.core.arm import arm_from_sealed_head
from geode.core.orchestrator import Orchestrator
from geode.settlement.settlement import (
    address_of,
    build_credit_batches,
    deposit_split,
    recompute_batch_hash,
    verify_batch_rules,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m212_settlement_wiring.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v25" / "m212_settlement_wiring"


def _build_orchestrator(config: dict[str, Any]) -> Orchestrator:
    orch = Orchestrator()
    for name, spec in config["arms"].items():
        orch.register(arm_from_sealed_head(
            name, spec["family"], spec["width"], spec["accuracy"],
            spec["sealed_source"], per_task=spec["per_task"]))
    for query in config["queries"]:
        orch.serve(query["query_id"], [], task_id=query["task_id"])
    return orch


def _scenario_report(config: dict[str, Any]) -> dict[str, Any]:
    # registered scenario: syn_fast tops every task axis (0.3 > all
    # per-task competitors), so its payout address is the credited
    # party everywhere; one query pays from that same address (the
    # self-payment exclusion, C1).
    self_pay_query = (config.get("self_payment") or {}).get("query")
    top_payout = address_of("syn_fast")
    orch = _build_orchestrator(config)

    def payer_of(query_id: str) -> str:
        if query_id == self_pay_query:
            return top_payout
        return address_of(f"{config['payer_prefix']}{query_id}")

    report = build_credit_batches(
        orch, int(config["price_per_query"]),
        payer_of=payer_of,
        registration_fee=int(config.get("registration_fee", 0)))
    return report


def _post_json(report: dict[str, Any]) -> dict[str, Any]:
    return {"registration_fee": report["registration_fee"],
            "registrations": report["registrations"],
            "deposits": report["deposits"],
            "pool_expected": report["pool_expected"],
            "batches": report["batches"],
            "expected": report["expected"]}


def _run_hardhat(config: dict[str, Any], post_path: Path
                 ) -> tuple[bool, str]:
    import os
    hh = config["hardhat"]
    cwd = REPO_ROOT / hh["cwd"]
    env = dict(os.environ)
    env["POST_BATCH_PATH"] = str(post_path.resolve())
    # shell=True so Windows resolves the npx.cmd shim.
    proc = subprocess.run(subprocess.list2cmdline(list(hh["cmd"])),
                          cwd=cwd, capture_output=True, text=True,
                          shell=True, env=env,
                          timeout=int(hh.get("timeout_seconds", 600)))
    out = (proc.stdout or "") + (proc.stderr or "")
    for line in (proc.stdout or "").splitlines():
        if line.startswith("POST_OK") or line.startswith("POST_FAIL"):
            return line.startswith("POST_OK"), line
    return False, out[-2000:]


def run_m212(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()

    report = _scenario_report(config)
    report2 = _scenario_report(config)
    g1_deterministic = report["batch_hash"] == report2["batch_hash"]

    g2_violations = verify_batch_rules(report, pool=report["pool_expected"])
    g2_conforms = not g2_violations

    tampered = json.loads(json.dumps(report))
    if tampered["batches"] and tampered["batches"][0]["entries"]:
        tampered["batches"][0]["entries"][0]["amount"] += 1
    g3_tamper_detected = (
        tampered.get("batch_hash") != recompute_batch_hash(tampered)
        and "batch_hash does not recompute" in
        " | ".join(verify_batch_rules(tampered)))

    post_path = output_dir / "post_batch.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(post_path, _post_json(report))
    g4_ok, g4_line = _run_hardhat(config, post_path)

    credited_expected = sum(report["expected"]["credits"].values())
    pool_remaining_expected = report["pool_expected"] - credited_expected
    dev_expected = sum(amount - deposit_split(amount)[0]
                       for d in report["deposits"]
                       for amount in [int(d["amount"])]) \
        + int(report["registration_fee"]) \
        * len(report["registrations"])  # the fee is per registration
    g4_parsed: dict[str, Any] = {}
    if g4_ok:
        parts = dict(item.split("=") for item in g4_line.split()[1:])
        g4_parsed = {k: int(v) for k, v in parts.items()}
        g4_ok = (g4_parsed.get("credited") == credited_expected
                 and g4_parsed.get("pool_remaining")
                 == pool_remaining_expected
                 and g4_parsed.get("skipped")
                 == len(report["expected"]["skipped"])
                 and g4_parsed.get("dev_share") == dev_expected
                 and g4_parsed.get("anchored")
                 == len(report["batches"]))

    gates = {
        "g1_deterministic": bool(g1_deterministic),
        "g2_contract_rules": {"ok": bool(g2_conforms),
                              "violations": g2_violations},
        "g3_tamper_detected": bool(g3_tamper_detected),
        "g4_cross_language_post": {"ok": bool(g4_ok), "line": g4_line,
                                   "parsed": g4_parsed,
                                   "expected": {
                                       "credited": credited_expected,
                                       "pool_remaining":
                                           pool_remaining_expected,
                                       "skipped": len(
                                           report["expected"]["skipped"]),
                                       "dev_share": dev_expected}},
    }
    gates_ok = all([gates["g1_deterministic"],
                    gates["g2_contract_rules"]["ok"],
                    gates["g3_tamper_detected"],
                    gates["g4_cross_language_post"]["ok"]])

    evidence: dict[str, Any] = {
        "milestone": "M212",
        "cell": "settlement wire (orchestrator -> CreditLedger)",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "anchor": {"batch_hash": report["batch_hash"],
                   "note": ("the settlement report content hash for the "
                            "registered scenario; future runs must "
                            "reproduce it bit-exactly")},
        "report": report,
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": (
                "the wire is closed end-to-end: ledger route records "
                "become a deterministic, chain-anchored attribution "
                "batch; the Python-built payload posts to the deployed "
                "CreditLedger with no revert and credits exactly the "
                "Python-side expected amounts, including the "
                "self-payment exclusion (payer = payout address) "
                "mirroring the contract"
            ) if gates_ok else "one or more M212 gates failed — VOID",
        },
        "scope": ("off-chain side only; the librarian transaction itself "
                  "is the production submitter's step (M194 anchoring "
                  "and a real network remain deferred)"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok,
                      "batch_hash": report["batch_hash"],
                      "gates": gates}, indent=1), flush=True)
    print(f"M212 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m212(args.config, args.output)


if __name__ == "__main__":
    main()
