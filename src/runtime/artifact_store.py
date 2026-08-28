"""Atomic local artifact publication for resumable GEODE stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from src.runtime.schemas import LifecycleState, StageManifest


MANIFEST_NAME = "stage_manifest.json"
SUCCESS_NAME = "SUCCESS"
RESERVED_NAMES = {MANIFEST_NAME, SUCCESS_NAME}


def _safe_component(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError(f"{name} must be a safe path component")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_durable_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


class LocalArtifactStore:
    """Publish immutable stage directories using a sibling partial directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def stage_path(self, run_id: str, attempt_id: str, stage_name: str) -> Path:
        run_id = _safe_component(run_id, "run_id")
        attempt_id = _safe_component(attempt_id, "attempt_id")
        stage_name = _safe_component(stage_name, "stage_name")
        return self.root / "runs" / run_id / "attempts" / attempt_id / stage_name

    def is_committed(self, run_id: str, attempt_id: str, stage_name: str) -> bool:
        path = self.stage_path(run_id, attempt_id, stage_name)
        return path.is_dir() and (path / MANIFEST_NAME).is_file() and (path / SUCCESS_NAME).is_file()

    def commit_stage(
        self,
        run_id: str,
        attempt_id: str,
        stage_name: str,
        writer: Callable[[Path], None],
        *,
        state: LifecycleState,
        input_hashes: Mapping[str, str] | None = None,
    ) -> StageManifest:
        """Run ``writer`` in a partial directory and atomically publish it."""
        final_path = self.stage_path(run_id, attempt_id, stage_name)
        expected_inputs = tuple(sorted((input_hashes or {}).items()))
        if final_path.exists():
            manifest = self.read_stage(run_id, attempt_id, stage_name)
            if manifest.input_hashes != expected_inputs:
                raise ValueError("committed stage input hashes do not match retry inputs")
            return manifest

        final_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = final_path.with_name(f"{final_path.name}.partial")
        if partial_path.exists():
            shutil.rmtree(partial_path)
        partial_path.mkdir()

        writer(partial_path)
        for reserved_name in RESERVED_NAMES:
            if (partial_path / reserved_name).exists():
                raise ValueError(f"stage writer created reserved file {reserved_name}")

        self._fsync_outputs(partial_path)
        output_hashes = self._hash_outputs(partial_path)
        manifest = StageManifest(
            run_id=run_id,
            attempt_id=attempt_id,
            stage_name=stage_name,
            state=state,
            created_at=datetime.now(timezone.utc).isoformat(),
            input_hashes=expected_inputs,
            output_hashes=tuple(sorted(output_hashes.items())),
        )
        manifest_text = _canonical_json(manifest.to_dict())
        _write_durable_text(partial_path / MANIFEST_NAME, manifest_text + "\n")
        _write_durable_text(
            partial_path / SUCCESS_NAME,
            hashlib.sha256(manifest_text.encode("utf-8")).hexdigest() + "\n",
        )
        os.replace(partial_path, final_path)
        return manifest

    def read_stage(self, run_id: str, attempt_id: str, stage_name: str) -> StageManifest:
        """Load and verify a committed stage, including every output hash."""
        path = self.stage_path(run_id, attempt_id, stage_name)
        if not self.is_committed(run_id, attempt_id, stage_name):
            raise FileNotFoundError(f"stage is not committed: {path}")
        try:
            payload = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid stage manifest: {path}") from error
        manifest = StageManifest.from_dict(payload)
        if (
            manifest.run_id != run_id
            or manifest.attempt_id != attempt_id
            or manifest.stage_name != stage_name
        ):
            raise ValueError("stage manifest identity does not match its path")

        expected_success = hashlib.sha256(
            _canonical_json(manifest.to_dict()).encode("utf-8"),
        ).hexdigest()
        actual_success = (path / SUCCESS_NAME).read_text(encoding="utf-8").strip()
        if actual_success != expected_success:
            raise ValueError("SUCCESS marker does not match stage manifest")
        actual_outputs = tuple(sorted(self._hash_outputs(path).items()))
        if actual_outputs != manifest.output_hashes:
            raise ValueError("committed stage output hashes do not match manifest")
        return manifest

    @staticmethod
    def _fsync_outputs(path: Path) -> None:
        for item in sorted(path.rglob("*")):
            if item.is_symlink():
                raise ValueError(f"stage outputs may not contain symlinks: {item}")
            if item.is_file():
                with item.open("r+b") as stream:
                    os.fsync(stream.fileno())

    @staticmethod
    def _hash_outputs(path: Path) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for item in sorted(path.rglob("*")):
            if item.is_symlink():
                raise ValueError(f"stage outputs may not contain symlinks: {item}")
            relative = item.relative_to(path).as_posix()
            if item.is_file() and relative not in RESERVED_NAMES:
                hashes[relative] = _sha256_file(item)
        return hashes