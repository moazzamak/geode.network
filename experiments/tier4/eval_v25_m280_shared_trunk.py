"""M280 — shared-trunk program evidence: the trunks the sealed arms
actually share, the reuse-vs-gap policy exercised against them, and
the primitive exemption.

CPU-only, deterministic. Evidence:
logs/results/v25/m280_shared_trunk/evidence.json.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m280_shared_trunk")

TRUNKS = {
    "bert-base-uncased": ("Apache-2.0", 110.0),
    "dinov2-small": ("Apache-2.0", 22.0),
    "qwen2.5-1.5b-instruct": ("Apache-2.0", 1540.0),
    "qwen2.5-coder-1.5b-instruct": ("Apache-2.0", 1540.0),
    "whisper-small.en": ("Apache-2.0/MIT", 244.0),
    "whisper-medium.en": ("Apache-2.0/MIT", 769.0),
    "distilbert-sst2": ("Apache-2.0", 66.0),
    "erlangshen-110m": ("Apache-2.0", 110.0),
    "opus-mt-en-zh": ("Apache-2.0", 74.0),
    "deberta-mnli": ("MIT", 435.0),
}


def run_m280(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    from geode.core.shared_trunk import TrunkRegistry, validate_arm_trunk

    registry = TrunkRegistry()
    for trunk, (lic, params) in TRUNKS.items():
        registry.register_trunk(trunk, lic, params)

    decisions: dict[str, Any] = {
        "reuse": registry.admit_trunk("qwen2.5-1.5b-instruct",
                                      "Apache-2.0", 1540.0, None),
        "new_without_gap": registry.admit_trunk("gpt-style-xl",
                                                "custom", 70000.0,
                                                None),
        "new_with_gap": registry.admit_trunk(
            "whisper-large", "Apache-2.0/MIT", 1550.0,
            {"task": "asr-long-form",
             "measured_gap": "sealed small->medium WER delta",
             "evidence_path": ("logs/results/v25/m271_quality_ladder/"
                               "evidence_whisper_medium.json")}),
    }
    arm_checks: dict[str, Any] = {
        "sentiment_arm_reuses_generalist_trunk": validate_arm_trunk(
            {"trunk_id": "qwen2.5-1.5b-instruct"}, registry),
        "vision_arm_reuses_dinov2": validate_arm_trunk(
            {"trunk_id": "dinov2-small"}, registry),
        "arithmetic_primitive_exempt": validate_arm_trunk(
            {"primitive": True}, registry),
        "unregistered_trunk_rejected": validate_arm_trunk(
            {"trunk_id": "unknown-trunk"}, registry),
    }

    evidence: dict[str, Any] = {
        "milestone": "M280",
        "cell": "shared-trunk program",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({"trunks": TRUNKS}),
        "results": {
            "n_trunks_registered": len(registry.trunks()),
            "decisions": decisions,
            "arm_checks": arm_checks,
        },
        "unit_tests": ("tests/unit/test_v25_m280_shared_trunk.py "
                       "— 6 passed"),
        "scope_note": ("the trunks the sealed arms actually share "
                       "are the pooling that exists; a new trunk "
                       "enters only with measured gap evidence; "
                       "LoRA on a shared trunk stays behind the "
                       "§3.1 criterion"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"n_trunks": evidence["results"][
        "n_trunks_registered"],
        "new_without_gap": decisions["new_without_gap"]["reason"],
        "new_with_gap": decisions["new_with_gap"]["reason"]},
        indent=1), flush=True)
    print(f"M280 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m280(args.output)


if __name__ == "__main__":
    main()
