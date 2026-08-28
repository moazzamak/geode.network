"""Local orchestration for ordered, immutable GEODE stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.runtime.artifact_store import LocalArtifactStore
from src.runtime.schemas import LifecycleState, StageManifest


StageWriter = Callable[[Path], None]


class StageExecutionStatus(str, Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    COMMITTED = "committed"
    CORRUPT = "corrupt"


@dataclass(frozen=True)
class StageSpec:
    name: str
    state: LifecycleState
    writer: StageWriter
    input_hashes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StageExecution:
    manifest: StageManifest
    reused: bool


@dataclass(frozen=True)
class StageStatus:
    name: str
    status: StageExecutionStatus
    state: LifecycleState | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "state": None if self.state is None else self.state.value,
            "error": self.error,
        }


@dataclass(frozen=True)
class RunStatus:
    run_id: str
    attempt_id: str
    stages: tuple[StageStatus, ...]

    @property
    def complete(self) -> bool:
        return bool(self.stages) and all(
            stage.status is StageExecutionStatus.COMMITTED for stage in self.stages
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "complete": self.complete,
            "stages": [stage.to_dict() for stage in self.stages],
        }


class LocalExecutor:
    """Execute ordered local stages and inspect their durable state."""

    def __init__(self, store: LocalArtifactStore) -> None:
        self.store = store

    def execute_stage(
        self,
        run_id: str,
        attempt_id: str,
        spec: StageSpec,
    ) -> StageExecution:
        reused = self.store.is_committed(run_id, attempt_id, spec.name)
        manifest = self.store.commit_stage(
            run_id,
            attempt_id,
            spec.name,
            spec.writer,
            state=spec.state,
            input_hashes=spec.input_hashes,
        )
        return StageExecution(manifest=manifest, reused=reused)

    def run(
        self,
        run_id: str,
        attempt_id: str,
        stages: Sequence[StageSpec],
    ) -> tuple[StageExecution, ...]:
        names = [stage.name for stage in stages]
        if len(names) != len(set(names)):
            raise ValueError("stage names must be unique")
        return tuple(
            self.execute_stage(run_id, attempt_id, stage) for stage in stages
        )

    def resume(
        self,
        run_id: str,
        attempt_id: str,
        stages: Sequence[StageSpec],
    ) -> tuple[StageExecution, ...]:
        return self.run(run_id, attempt_id, stages)

    def status(
        self,
        run_id: str,
        attempt_id: str,
        expected_stages: Sequence[str] = (),
    ) -> RunStatus:
        attempt_path = self.store.stage_path(
            run_id, attempt_id, "stage-placeholder"
        ).parent
        discovered: set[str] = set()
        if attempt_path.is_dir():
            for child in attempt_path.iterdir():
                if child.is_dir():
                    discovered.add(
                        child.name[:-8] if child.name.endswith(".partial") else child.name
                    )
        names = list(dict.fromkeys((*expected_stages, *sorted(discovered))))
        statuses = tuple(
            self._stage_status(run_id, attempt_id, name) for name in names
        )
        return RunStatus(run_id=run_id, attempt_id=attempt_id, stages=statuses)

    def _stage_status(
        self,
        run_id: str,
        attempt_id: str,
        stage_name: str,
    ) -> StageStatus:
        stage_path = self.store.stage_path(run_id, attempt_id, stage_name)
        partial_path = stage_path.with_name(f"{stage_path.name}.partial")
        if stage_path.exists():
            if partial_path.exists():
                return StageStatus(
                    name=stage_name,
                    status=StageExecutionStatus.CORRUPT,
                    error="partial directory exists beside committed stage",
                )
            try:
                manifest = self.store.read_stage(run_id, attempt_id, stage_name)
            except (OSError, TypeError, ValueError) as error:
                return StageStatus(
                    name=stage_name,
                    status=StageExecutionStatus.CORRUPT,
                    error=str(error),
                )
            return StageStatus(
                name=stage_name,
                status=StageExecutionStatus.COMMITTED,
                state=manifest.state,
            )
        if partial_path.exists():
            return StageStatus(
                name=stage_name,
                status=StageExecutionStatus.PARTIAL,
            )
        return StageStatus(name=stage_name, status=StageExecutionStatus.PENDING)