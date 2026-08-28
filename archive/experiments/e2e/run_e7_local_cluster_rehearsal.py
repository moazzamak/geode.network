"""Run the E7 logical multi-node rehearsal without claiming multi-host evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from experiments.common.data_cache import data_cache_root
from src.runtime.distributed_qualification import DistributedQualificationEvidence
from src.runtime.domainnet_manifest import DomainNetManifest
from src.runtime.ray_executor import RayExecutor


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deterministic_task(item: int) -> dict[str, Any]:
    import platform
    import ray

    return {
        "item": item,
        "square": item * item,
        "node_id": ray.get_runtime_context().get_node_id(),
        "python": platform.python_version(),
        "ray": ray.__version__,
    }


def _crash_worker_process_once(marker_path: str) -> dict[str, Any]:
    import ray

    marker = Path(marker_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return {
            "recovered": True,
            "node_id": ray.get_runtime_context().get_node_id(),
        }
    with os.fdopen(descriptor, "w", encoding="ascii") as stream:
        stream.write("first worker process terminated\n")
        stream.flush()
        os.fsync(stream.fileno())
    os._exit(70)


def run_rehearsal(
    config_path: Path,
    *,
    address: str,
    shared_marker_path: str,
    host_marker_path: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported E7 configuration schema")

    cache_root = data_cache_root()
    manifest = DomainNetManifest.load(cache_root / config["manifest_path"])
    manifest_report = manifest.verify(cache_root / config["data_root"])
    host_marker_path.parent.mkdir(parents=True, exist_ok=True)
    host_marker_path.unlink(missing_ok=True)

    executor = RayExecutor(address=address)
    try:
        resources = executor.resource_report().to_dict()
        items = list(range(16))
        first = executor.map_on_nodes(
            _deterministic_task, items,
            node_ids=resources["node_ids"], max_retries=1,
        )
        second = executor.map_on_nodes(
            _deterministic_task, items,
            node_ids=resources["node_ids"], max_retries=1,
        )
        expected = [item * item for item in items]
        complete_histories = (
            [record["square"] for record in first] == expected
            and [record["square"] for record in second] == expected
        )
        first_values = [
            {"item": record["item"], "square": record["square"]}
            for record in first
        ]
        second_values = [
            {"item": record["item"], "square": record["square"]}
            for record in second
        ]
        first_hash = _canonical_hash(first_values)
        second_hash = _canonical_hash(second_values)
        fault = executor.map_on_nodes(
            _crash_worker_process_once,
            [shared_marker_path],
            node_ids=resources["worker_node_ids"][:1],
            max_retries=int(config["ray"]["max_retries"]),
        )[0]
    finally:
        executor.shutdown()

    evidence = DistributedQualificationEvidence(
        scope="local_simulation",
        logical_nodes=int(resources["nodes"]),
        executing_nodes=len({record["node_id"] for record in first}),
        physical_hosts=1,
        task_retry_passed=bool(fault["recovered"]),
        worker_process_loss_recovered=bool(fault["recovered"]),
        worker_node_loss_recovered=False,
        complete_histories=complete_histories,
        artifact_identity_verified=first_hash == second_hash,
    ).evaluate()
    return {
        "schema_version": 1,
        "milestone": "E7-local-simulation",
        "qualification_status": (
            "passed_local_simulation"
            if evidence["local_simulation_gate_passed"]
            else "failed_local_simulation"
        ),
        "domainnet": manifest_report,
        "ray": {
            "address": address,
            "resources": resources,
            "task_node_ids": sorted({record["node_id"] for record in first}),
            "node_runtimes": sorted({
                (record["node_id"], record["python"], record["ray"])
                for record in first
            }),
            "fault_recovery_node_id": fault["node_id"],
        },
        "histories": {
            "items": len(items),
            "complete": complete_histories,
            "first_hash": first_hash,
            "second_hash": second_hash,
        },
        "qualification": evidence,
        "limitations": [
            "all Ray containers share one physical host",
            "worker process loss is tested; physical worker-node loss is not",
            "this artifact cannot satisfy the final E7 multi-host gate",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("experiments/configs/e7_domainnet_qualification.json"),
    )
    parser.add_argument("--address", default="ray://127.0.0.1:10001")
    parser.add_argument(
        "--shared-marker-path",
        default="/evidence/e7-worker-process-loss.marker",
    )
    parser.add_argument(
        "--host-marker-path", type=Path,
        default=Path("logs/results/e7_local_shared/e7-worker-process-loss.marker"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/results/e7_local_cluster_rehearsal.json"),
    )
    arguments = parser.parse_args()
    result = run_rehearsal(
        arguments.config,
        address=arguments.address,
        shared_marker_path=arguments.shared_marker_path,
        host_marker_path=arguments.host_marker_path,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["qualification"]["local_simulation_gate_passed"]:
        raise RuntimeError("E7 local cluster rehearsal failed")


if __name__ == "__main__":
    main()