"""Explicit checkpoint adapter for iterative SDF refinement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.runtime.checkpoint import LocalCheckpointStore
from src.runtime.schemas import CheckpointMetadata, LifecycleState
from src.sdf_optimizer import SDFOptimizer


@dataclass(frozen=True)
class RestoredRefinementState:
    metadata: CheckpointMetadata
    epoch_history: tuple[dict[str, Any], ...]
    sampler_state: dict[str, Any]


class RefinementCheckpointAdapter:
    """Save and restore model, optimizer, RNG, sampler, and epoch history."""

    SCHEMA_VERSION = 1

    def __init__(self, store: LocalCheckpointStore) -> None:
        self.store = store

    def save(
        self,
        run_id: str,
        attempt_id: str,
        stage_name: str,
        epoch: int,
        global_step: int,
        *,
        optimizer: SDFOptimizer,
        rng: np.random.Generator,
        epoch_history: Sequence[Mapping[str, Any]],
        sampler_state: Mapping[str, Any],
        input_hashes: Mapping[str, str] | None = None,
    ) -> CheckpointMetadata:
        optimizer_state, arrays = optimizer.export_state()
        state = {
            "adapter_schema_version": self.SCHEMA_VERSION,
            "optimizer": optimizer_state,
            "rng_state": rng.bit_generator.state,
            "epoch_history": [dict(record) for record in epoch_history],
            "sampler_state": dict(sampler_state),
        }
        return self.store.save(
            run_id,
            attempt_id,
            stage_name,
            epoch,
            global_step,
            state=state,
            arrays=arrays,
            input_hashes=input_hashes,
            lifecycle_state=LifecycleState.GEOMETRY_READY,
        )

    def restore_latest(
        self,
        run_id: str,
        attempt_id: str,
        stage_name: str,
        *,
        optimizer: SDFOptimizer,
        rng: np.random.Generator,
    ) -> RestoredRefinementState | None:
        loaded = self.store.latest(run_id, attempt_id, stage_name)
        if loaded is None:
            return None
        required = {
            "adapter_schema_version", "optimizer", "rng_state",
            "epoch_history", "sampler_state",
        }
        if set(loaded.state) != required or loaded.state["adapter_schema_version"] != 1:
            raise ValueError("unsupported refinement checkpoint state")
        rng_state = loaded.state["rng_state"]
        if rng_state.get("bit_generator") != rng.bit_generator.__class__.__name__:
            raise ValueError("checkpoint RNG bit generator does not match")
        optimizer.import_state(loaded.state["optimizer"], loaded.arrays)
        rng.bit_generator.state = rng_state
        return RestoredRefinementState(
            metadata=loaded.metadata,
            epoch_history=tuple(dict(record) for record in loaded.state["epoch_history"]),
            sampler_state=dict(loaded.state["sampler_state"]),
        )

    def latest_metadata(
        self,
        run_id: str,
        attempt_id: str,
        stage_name: str,
    ) -> CheckpointMetadata | None:
        loaded = self.store.latest(run_id, attempt_id, stage_name)
        return None if loaded is None else loaded.metadata