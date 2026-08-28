"""Evidence contracts for local and physical multi-host E7 qualification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


QualificationScope = Literal["local_simulation", "multihost"]


@dataclass(frozen=True)
class DistributedQualificationEvidence:
    scope: QualificationScope
    logical_nodes: int
    executing_nodes: int
    physical_hosts: int
    task_retry_passed: bool
    worker_process_loss_recovered: bool
    worker_node_loss_recovered: bool
    complete_histories: bool
    artifact_identity_verified: bool

    def evaluate(self) -> dict[str, Any]:
        local_checks = {
            "logical_cluster_has_multiple_nodes": self.logical_nodes >= 2,
            "tasks_executed_on_multiple_nodes": self.executing_nodes >= 2,
            "task_retry_passed": self.task_retry_passed,
            "worker_process_loss_recovered": self.worker_process_loss_recovered,
            "complete_histories": self.complete_histories,
            "artifact_identity_verified": self.artifact_identity_verified,
        }
        multihost_checks = {
            **local_checks,
            "physical_hosts_are_distinct": self.physical_hosts >= 2,
            "worker_node_loss_recovered": self.worker_node_loss_recovered,
        }
        local_simulation_gate_passed = all(local_checks.values())
        multihost_gate_passed = (
            self.scope == "multihost" and all(multihost_checks.values())
        )
        return {
            "scope": self.scope,
            "local_simulation_gate_passed": local_simulation_gate_passed,
            "multihost_gate_passed": multihost_gate_passed,
            "e7_gate_passed": multihost_gate_passed,
            "local_checks": local_checks,
            "multihost_checks": multihost_checks,
            "evidence": {
                "logical_nodes": self.logical_nodes,
                "executing_nodes": self.executing_nodes,
                "physical_hosts": self.physical_hosts,
                "task_retry_passed": self.task_retry_passed,
                "worker_process_loss_recovered": self.worker_process_loss_recovered,
                "worker_node_loss_recovered": self.worker_node_loss_recovered,
                "complete_histories": self.complete_histories,
                "artifact_identity_verified": self.artifact_identity_verified,
            },
        }