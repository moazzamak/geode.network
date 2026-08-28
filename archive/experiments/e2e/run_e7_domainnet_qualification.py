"""Fail-closed DomainNet and Ray readiness qualification."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import platform

from experiments.common.data_cache import data_cache_root
from src.runtime.domainnet_manifest import DomainNetManifest
from src.runtime.ray_executor import RayExecutor, RayUnavailableError
from src.runtime.schemas import ModelSelectionContract


def run_preflight(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported E7 configuration schema")
    selection = ModelSelectionContract(
        validation_domains=tuple(config["validation_domains"]),
        final_domains=tuple(config["final_domains"]),
        selection_rule=str(config["selection_rule"]),
        primary_metric=str(config["primary_metric"]),
    )
    blockers = []
    manifest_report = None
    cache_root = data_cache_root()
    manifest_path = cache_root / config["manifest_path"]
    if not manifest_path.exists():
        blockers.append("verified_domainnet_manifest_missing")
    else:
        try:
            manifest = DomainNetManifest.load(manifest_path)
            manifest_report = manifest.verify(cache_root / config["data_root"])
        except ValueError as error:
            blockers.append(f"domainnet_verification_failed:{error}")

    ray_available = importlib.util.find_spec("ray") is not None
    resource_report = None
    worker_loss_recovery = {
        "attempted": False,
        "passed": False,
        "reason": "Ray cluster is unavailable",
    }
    if not ray_available:
        blockers.append("ray_unavailable_for_active_python")
    else:
        try:
            executor = RayExecutor(address=config["ray"]["address"])
            resource_report = executor.resource_report().to_dict()
            if resource_report["nodes"] < int(config["ray"]["minimum_nodes"]):
                blockers.append("ray_cluster_has_insufficient_nodes")
            worker_loss_recovery["reason"] = (
                "not attempted until verified data exists"
                if manifest_report is None
                else "not attempted until minimum Ray node count is satisfied"
            )
            executor.shutdown()
        except RayUnavailableError as error:
            blockers.append(f"ray_start_failed:{error}")

    return {
        "schema_version": 1,
        "milestone": "E7",
        "qualification_status": "passed" if not blockers else "blocked",
        "gate_passed": not blockers,
        "blockers": blockers,
        "proxy_results_used_as_flagship": False,
        "manifest": manifest_report,
        "model_selection": selection.to_dict(),
        "ray": {
            "available": ray_available,
            "resource_report": resource_report,
            "worker_loss_recovery": worker_loss_recovery,
        },
        "environment": {
            "data_cache_root": str(cache_root),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "full_run": {
            "attempted": False,
            "complete_histories": False,
            "reason": "preflight blockers must be resolved first",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("experiments/configs/e7_domainnet_qualification.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/results/e7_domainnet_preflight.json"),
    )
    parser.add_argument("--allow-blocked", action="store_true")
    arguments = parser.parse_args()
    result = run_preflight(arguments.config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gate_passed"] and not arguments.allow_blocked:
        raise RuntimeError(f"E7 qualification blocked: {result['blockers']}")


if __name__ == "__main__":
    main()