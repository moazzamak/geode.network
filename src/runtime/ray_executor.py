"""Optional Ray task adapter with explicit resource and retry reporting."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


class RayUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class RayResourceReport:
    address: str
    nodes: int
    node_ids: tuple[str, ...]
    worker_node_ids: tuple[str, ...]
    cluster_resources: dict[str, float]
    available_resources: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "nodes": self.nodes,
            "node_ids": list(self.node_ids),
            "worker_node_ids": list(self.worker_node_ids),
            "cluster_resources": self.cluster_resources,
            "available_resources": self.available_resources,
        }


class RayExecutor:
    """Submit immutable map tasks to Ray without importing it at module load."""

    def __init__(
        self,
        *,
        address: str | None = None,
        num_cpus: int | None = None,
    ) -> None:
        try:
            import ray
        except ImportError as error:
            raise RayUnavailableError(
                "Ray is unavailable; use Python <3.14 and install requirements"
            ) from error
        self._ray = ray
        if not ray.is_initialized():
            options: dict[str, Any] = {"ignore_reinit_error": True}
            if address is not None:
                options["address"] = address
            elif num_cpus is not None:
                options["num_cpus"] = num_cpus
            ray.init(**options)

    def map(
        self,
        function: Callable[[Any], Any],
        items: Sequence[Any],
        *,
        max_retries: int = 1,
        num_cpus: float = 1.0,
    ) -> list[Any]:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        remote = self._ray.remote(
            max_retries=max_retries,
            num_cpus=num_cpus,
        )(function)
        references = [remote.remote(item) for item in items]
        return list(self._ray.get(references))

    def map_on_nodes(
        self,
        function: Callable[[Any], Any],
        items: Sequence[Any],
        *,
        node_ids: Sequence[str],
        max_retries: int = 1,
        num_cpus: float = 1.0,
    ) -> list[Any]:
        if not node_ids:
            raise ValueError("node_ids must not be empty")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        remote = self._ray.remote(
            max_retries=max_retries,
            num_cpus=num_cpus,
        )(function)
        references = [
            remote.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_ids[index % len(node_ids)], soft=False,
            )).remote(item)
            for index, item in enumerate(items)
        ]
        return list(self._ray.get(references))

    def resource_report(self) -> RayResourceReport:
        context = self._ray.get_runtime_context()
        live_nodes = [node for node in self._ray.nodes() if node["Alive"]]
        node_ids = tuple(sorted(str(node["NodeID"]) for node in live_nodes))
        worker_node_ids = tuple(sorted(
            str(node["NodeID"])
            for node in live_nodes
            if "node:__internal_head__" not in node.get("Resources", {})
        ))
        return RayResourceReport(
            address=str(context.gcs_address),
            nodes=len(live_nodes),
            node_ids=node_ids,
            worker_node_ids=worker_node_ids,
            cluster_resources={
                key: float(value)
                for key, value in self._ray.cluster_resources().items()
            },
            available_resources={
                key: float(value)
                for key, value in self._ray.available_resources().items()
            },
        )

    def shutdown(self) -> None:
        self._ray.shutdown()