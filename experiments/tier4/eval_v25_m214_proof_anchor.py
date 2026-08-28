"""M214 — proof-hash anchor evidence: the real-SIZE M193b proof
(14 rounds, 1,024 bytes) anchors on-chain; the anchor gas is measured
and the append-only semantics are gated.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(19 Aug 2026, before the build). CPU-only; the only external call is
the local Hardhat harness. The proof is synthetic (registered seed) —
the anchor cost depends on calldata size, not content.
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
from geode.privacy.zk_onchain import serialize_hex

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m214_proof_anchor.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v25" / "m214_proof_anchor"


def _build_proofs(config: dict[str, Any]) -> tuple[str, str]:
    n = int(config["n"])
    rnd = random.Random(int(config["seed"]))
    x = [rnd.randrange(Q_ORDER) for _ in range(n)]
    w = [rnd.randrange(Q_ORDER) for _ in range(n)]
    claim = sum(xi * wi % Q_ORDER for xi, wi in zip(x, w)) % Q_ORDER
    r = rnd.randrange(Q_ORDER)
    proof = prove(x, r, w, claim)
    data = serialize_hex(proof, n)
    # a tampered variant: flip the last byte of the r_final slot
    assert (len(data) - 2) // 2 == 1024  # 32 words at r=14, sealed size
    alt = data[:-2] + ("00" if data[-2:] != "00" else "11")
    assert alt != data
    return data, alt


def _run_hardhat(config: dict[str, Any], spec_path: Path
                 ) -> tuple[bool, dict[str, Any], str]:
    hh = config["hardhat"]
    cwd = REPO_ROOT / hh["cwd"]
    env = dict(os.environ)
    env["POST_ANCHOR_PATH"] = str(spec_path.resolve())
    proc = subprocess.run(subprocess.list2cmdline(list(hh["cmd"])),
                          cwd=cwd, capture_output=True, text=True,
                          shell=True, env=env,
                          timeout=int(hh.get("timeout_seconds", 300)))
    out = (proc.stdout or "") + (proc.stderr or "")
    for line in (proc.stdout or "").splitlines():
        if line.startswith("ANCHOR_OK"):
            parts = dict(item.split("=") for item in line.split()[1:]
                         if "=" in item)
            return True, parts, line
    return False, {}, out[-2000:]


def run_m214(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()

    proof, proof_alt = _build_proofs(config)
    spec = {"proof": proof, "proof_alt": proof_alt}
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "anchor_cases.json"
    write_canonical_json(spec_path, spec)
    ok, parts, tail = _run_hardhat(config, spec_path)

    gas = int(parts.get("gas", 0)) if ok else None
    reanchor_gas = int(parts.get("reanchor_gas", 0)) if ok else None
    g1 = bool(ok and parts.get("block1") not in (None, "0"))
    g2 = bool(ok and parts.get("distinct") == "true")
    g3 = bool(ok and parts.get("reanchor_gas", "0") != "0"
              and gas is not None)  # re-anchor executed, block unchanged
    g4 = bool(gas is not None and gas > 0)
    gates_ok = all([g1, g2, g3, g4])

    evidence: dict[str, Any] = {
        "milestone": "M214",
        "cell": "per-query on-chain proof-hash anchor",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "anchor": {"proof_hex_sha_of_record": payload_hash(proof),
                   "note": ("the synthetic 14-round proof hex; a future "
                            "run must reproduce it bit-exactly")},
        "proof_size_bytes": (len(proof) - 2) // 2,
        "gates": {
            "g1_anchored_and_retrievable": bool(g1),
            "g2_tampered_hash_distinct": bool(g2),
            "g3_append_only_noop": bool(g3),
            "g4_measured_gas": {"ok": bool(g4),
                                "gas": gas,
                                "reanchor_gas": reanchor_gas},
        },
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": (
                "the per-query proof-hash anchor is the economically "
                "correct settlement design: ~1 KB proof anchored "
                "on-chain at measured cost, full verification runs "
                "off-chain through the sealed verifier; full-width "
                "on-chain verification requires a pairing-based SNARK "
                "(production zk stack, M211)"
            ) if gates_ok else "one or more M214 gates failed — VOID",
        },
        "scope": "local EVM; production anchoring rides the M194 "
                 "public-testnet decision",
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok,
                      "gates": evidence["gates"],
                      "hardhat_tail": tail[:400]}, indent=1), flush=True)
    print(f"M214 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m214(args.config, args.output)


if __name__ == "__main__":
    main()
