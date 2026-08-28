"""Explicit, atomic checkpoint storage without whole-process pickling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from src.runtime.artifact_store import LocalArtifactStore
from src.runtime.schemas import CheckpointMetadata, LifecycleState


CHECKPOINT_PREFIX = "checkpoint--"
CHECKPOINT_METADATA = "checkpoint.json"
STATE_NAME = "state.json"


def _safe_array_name(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError("checkpoint array names must be safe path components")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(_canonical_json(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _checkpoint_stage_name(stage_name: str, epoch: int, global_step: int) -> str:
    if (
        not isinstance(stage_name, str)
        or "--" in stage_name
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", stage_name) is None
    ):
        raise ValueError("checkpoint stage_name must be a safe path component")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise ValueError("global_step must be a non-negative integer")
    return f"{CHECKPOINT_PREFIX}{stage_name}--{epoch:08d}--{global_step:012d}"


def _checkpoint_prefix(stage_name: str) -> str:
    validated = _checkpoint_stage_name(stage_name, 0, 0)
    return validated.rsplit("--", 2)[0] + "--"


@dataclass(frozen=True)
class LoadedCheckpoint:
    metadata: CheckpointMetadata
    state: dict[str, Any]
    arrays: dict[str, np.ndarray]


class LocalCheckpointStore:
    """Store checkpoint state as explicit JSON and non-pickled array files."""

    def __init__(self, artifact_store: LocalArtifactStore) -> None:
        self.artifact_store = artifact_store

    def save(
        self,
        run_id: str,
        attempt_id: str,
        stage_name: str,
        epoch: int,
        global_step: int,
        *,
        state: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray],
        input_hashes: Mapping[str, str] | None = None,
        lifecycle_state: LifecycleState = LifecycleState.CREATED,
    ) -> CheckpointMetadata:
        checkpoint_stage = _checkpoint_stage_name(stage_name, epoch, global_step)
        state_payload = dict(state)
        _canonical_json(state_payload)
        normalized_arrays = {
            _safe_array_name(name): np.asarray(array)
            for name, array in arrays.items()
        }
        if any(array.dtype.hasobject for array in normalized_arrays.values()):
            raise ValueError("checkpoint arrays may not use object dtype")
        content_digest = hashlib.sha256()
        content_digest.update(_canonical_json(state_payload).encode("utf-8"))
        for name, array in sorted(normalized_arrays.items()):
            content_digest.update(name.encode("utf-8"))
            content_digest.update(_array_hash(array).encode("ascii"))
        checkpoint_inputs = dict(input_hashes or {})
        checkpoint_inputs["checkpoint_payload"] = content_digest.hexdigest()

        def write_checkpoint(path: Path) -> None:
            state_path = path / STATE_NAME
            state_path.write_text(
                _canonical_json(state_payload) + "\n", encoding="utf-8", newline="\n",
            )
            arrays_path = path / "arrays"
            arrays_path.mkdir()
            for name, array in sorted(normalized_arrays.items()):
                with (arrays_path / f"{name}.npy").open("xb") as stream:
                    np.save(stream, array, allow_pickle=False)

            artifact_paths = [state_path, *sorted(arrays_path.glob("*.npy"))]
            artifact_hashes = tuple(
                (item.relative_to(path).as_posix(), _sha256_file(item))
                for item in artifact_paths
            )
            metadata = CheckpointMetadata(
                run_id=run_id,
                attempt_id=attempt_id,
                stage_name=stage_name,
                epoch=epoch,
                global_step=global_step,
                created_at=datetime.now(timezone.utc).isoformat(),
                artifact_hashes=tuple(sorted(artifact_hashes)),
            )
            (path / CHECKPOINT_METADATA).write_text(
                _canonical_json(metadata.to_dict()) + "\n",
                encoding="utf-8",
                newline="\n",
            )

        self.artifact_store.commit_stage(
            run_id,
            attempt_id,
            checkpoint_stage,
            write_checkpoint,
            state=lifecycle_state,
            input_hashes=checkpoint_inputs,
        )
        return self.load(run_id, attempt_id, checkpoint_stage).metadata

    def load(
        self,
        run_id: str,
        attempt_id: str,
        checkpoint_stage: str,
    ) -> LoadedCheckpoint:
        self.artifact_store.read_stage(run_id, attempt_id, checkpoint_stage)
        path = self.artifact_store.stage_path(run_id, attempt_id, checkpoint_stage)
        try:
            metadata = CheckpointMetadata.from_dict(json.loads(
                (path / CHECKPOINT_METADATA).read_text(encoding="utf-8"),
            ))
            state = json.loads((path / STATE_NAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid checkpoint at {path}") from error

        arrays: dict[str, np.ndarray] = {}
        for array_path in sorted((path / "arrays").glob("*.npy")):
            with array_path.open("rb") as stream:
                arrays[array_path.stem] = np.load(stream, allow_pickle=False)
        return LoadedCheckpoint(metadata=metadata, state=state, arrays=arrays)

    def list_checkpoints(
        self,
        run_id: str,
        attempt_id: str,
        stage_name: str,
    ) -> list[LoadedCheckpoint]:
        parent = self.artifact_store.stage_path(run_id, attempt_id, "placeholder").parent
        prefix = _checkpoint_prefix(stage_name)
        checkpoints = [
            self.load(run_id, attempt_id, path.name)
            for path in parent.glob(f"{prefix}*")
            if path.is_dir() and not path.name.endswith(".partial")
        ]
        return sorted(
            checkpoints,
            key=lambda item: (item.metadata.epoch, item.metadata.global_step),
        )

    def latest(
        self,
        run_id: str,
        attempt_id: str,
        stage_name: str,
    ) -> LoadedCheckpoint | None:
        checkpoints = self.list_checkpoints(run_id, attempt_id, stage_name)
        return checkpoints[-1] if checkpoints else None