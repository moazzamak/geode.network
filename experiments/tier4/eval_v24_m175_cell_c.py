"""M175 cell C — cross-modality routing: contract guard demonstration.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` section 12
(18 Aug 2026). Route-level only; NO new accuracy, no data cost. The
router's contract guard (C1, geode/router.py, unit-tested 16/16) plus
this demonstration of the sealed registry:

- vision arms (sealed): spm1923 DomainNet (0.2274/0.2786), deep-patch
  SPM (0.590), dense r70 (0.3118), kind classification-vision.
- text arms (sealed): wikitext uniform-w2 (ppl 9.9152), Wikipedia
  uniform-w2 (ppl 9.5142), kind next-token-text.
- task manifests: fingerprints are registered profile vectors built
  from the sealed numbers (the claim is the GUARD, not fingerprint
  quality).

Registered checks: (1) the vision task's chain contains ONLY vision
arms; (2) the text task's chain contains ONLY text arms; (3) a
cross-contract query (vision fingerprint + text contract) contains no
wrong-kind arm. Verdict: pass iff (1) and (2) hold and (3) shows no
wrong-kind arm in any chain.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from geode.core.router import Router

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m175_cell_c.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v24" / "m175_cell_c"

VISION_KIND = "classification-vision"
TEXT_KIND = "next-token-text"


def _arm(arm_id: str, fp: list[float], kind: str, acc: float,
         general: bool = False) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "fingerprint": fp,
        "output_contract": {"kind": kind,
                            "note": "registered in the sealed evidence"},
        "held_out_accuracy": {arm_id: acc},
        "selection_accuracy": acc,
        "availability": {"contract_hash": "sealed", "payload_hash": "sealed",
                         "healthy": True},
        "price": 0.0,
        "general": general,
        "primitive": False,
    }


def run_m175_cell_c(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    arms = config["arms"]
    router = Router()
    for spec in arms:
        router.add_arm(_arm(spec["arm_id"], spec["fingerprint"],
                            spec["kind"], spec["sealed_score"],
                            spec.get("general", False)))
    registry_hash = router.content_hash()

    checks: dict[str, Any] = {}
    for task_name, task in config["tasks"].items():
        chain = router.chain(task["fingerprint"],
                             contract_kind=task["contract_kind"])
        chain_ids = [a["arm_id"] for a in chain]
        kinds = {a["output_contract"]["kind"] for a in chain}
        only_own_kind = kinds == {task["contract_kind"]}
        wrong_kind = [a["arm_id"] for a in chain
                      if a["output_contract"]["kind"]
                      != task["contract_kind"]]
        checks[task_name] = {
            "contract_kind": task["contract_kind"],
            "chain": chain_ids,
            "chain_kinds": sorted(kinds),
            "only_own_kind": only_own_kind,
            "wrong_kind_arms": wrong_kind,
        }
        print(f"  {task_name}: chain {chain_ids} "
              f"only_own_kind={only_own_kind}", flush=True)

    cross = router.chain(config["tasks"]["vision"]["fingerprint"],
                         contract_kind=TEXT_KIND)
    cross_ids = [a["arm_id"] for a in cross]
    cross_wrong = [a["arm_id"] for a in cross
                   if a["output_contract"]["kind"] != TEXT_KIND]
    checks["cross_contract_query"] = {
        "query": "vision fingerprint + text contract",
        "chain": cross_ids,
        "wrong_kind_arms": cross_wrong,
    }
    print(f"  cross-contract: chain {cross_ids} "
          f"wrong_kind={cross_wrong}", flush=True)

    pass_checks = (checks["vision"]["only_own_kind"]
                   and checks["text"]["only_own_kind"]
                   and not cross_wrong
                   and not checks["vision"]["wrong_kind_arms"]
                   and not checks["text"]["wrong_kind_arms"])

    evidence: dict[str, Any] = {
        "milestone": "M175",
        "cell": "C cross-modality routing (contract guard demonstration)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "registry_hash": registry_hash,
        "checks": checks,
        "verdict": {
            "passed": pass_checks,
            "reading": ("no modality confusion: every task's chain contains "
                        "only output-contract-matching arms, including under "
                        "a cross-contract query")
            if pass_checks else
            ("guard failed: a wrong-kind arm appeared in a chain"),
        },
        "security_note": ("the guard is a safety property: a task can never "
                          "silently receive a wrong-modality arm; wrong-kind "
                          "arms are unreachable, not just ranked last"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"passed": pass_checks, "checks": checks}, indent=1),
          flush=True)
    print(f"M175 cell C complete -> {output_dir / 'evidence.json'}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m175_cell_c(args.config, args.output)


if __name__ == "__main__":
    main()
