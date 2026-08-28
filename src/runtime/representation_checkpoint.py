"""Explicit checkpoints for Torch representation training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.runtime.checkpoint import LocalCheckpointStore
from src.runtime.schemas import CheckpointMetadata, LifecycleState


_TORCH_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
}


@dataclass(frozen=True)
class RestoredRepresentationState:
    metadata: CheckpointMetadata
    epoch_history: tuple[dict[str, Any], ...]
    sampler_state: dict[str, Any]


def _pack_value(
    value: Any,
    array_name: str,
    arrays: dict[str, np.ndarray],
) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().copy()
        arrays[array_name] = raw
        return {
            "kind": "torch_tensor",
            "array": array_name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _pack_value(item, f"{array_name}-{index:06d}", arrays)
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _pack_value(item, f"{array_name}-{index:06d}", arrays)
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "items": [
                [
                    _pack_value(key, f"{array_name}-key-{index:06d}", arrays),
                    _pack_value(item, f"{array_name}-value-{index:06d}", arrays),
                ]
                for index, (key, item) in enumerate(value.items())
            ],
        }
    raise TypeError(f"unsupported checkpoint value: {type(value).__name__}")


def _unpack_value(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    if not isinstance(value, dict) or "kind" not in value:
        return value
    kind = value["kind"]
    if kind == "torch_tensor":
        dtype = _TORCH_DTYPES.get(value.get("dtype"))
        array_name = value.get("array")
        if dtype is None or array_name not in arrays:
            raise ValueError("invalid Torch tensor checkpoint metadata")
        raw = np.asarray(arrays[array_name])
        if raw.dtype != np.uint8 or raw.ndim != 1:
            raise ValueError("invalid Torch tensor checkpoint array")
        try:
            return torch.from_numpy(raw.copy()).view(dtype).reshape(value["shape"])
        except (RuntimeError, TypeError) as error:
            raise ValueError("invalid Torch tensor checkpoint shape") from error
    if kind in {"list", "tuple"}:
        items = [_unpack_value(item, arrays) for item in value.get("items", [])]
        return items if kind == "list" else tuple(items)
    if kind == "mapping":
        return {
            key: _unpack_value(item, arrays)
            for key, item in value.get("items", [])
        }
    raise ValueError("unsupported checkpoint value kind")


class RepresentationCheckpointAdapter:
    """Save and restore explicit Torch model and training state."""

    SCHEMA_VERSION = 1

    def __init__(self, store: LocalCheckpointStore) -> None:
        self.store = store

    @staticmethod
    def _optimizer_topology(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> list[list[str]]:
        names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
        try:
            return [
                [names_by_id[id(parameter)] for parameter in group["params"]]
                for group in optimizer.param_groups
            ]
        except KeyError as error:
            raise ValueError("optimizer contains a parameter outside the model") from error

    def save(
        self,
        run_id: str,
        attempt_id: str,
        stage_name: str,
        epoch: int,
        global_step: int,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rng: np.random.Generator,
        epoch_history: Sequence[Mapping[str, Any]],
        sampler_state: Mapping[str, Any],
        scheduler: Any | None = None,
        scaler: Any | None = None,
        input_hashes: Mapping[str, str] | None = None,
    ) -> CheckpointMetadata:
        arrays: dict[str, np.ndarray] = {}
        model_state = _pack_value(model.state_dict(), "model", arrays)
        optimizer_state = _pack_value(optimizer.state_dict(), "optimizer", arrays)
        scheduler_state = None if scheduler is None else _pack_value(
            scheduler.state_dict(), "scheduler", arrays,
        )
        scaler_state = None if scaler is None else _pack_value(
            scaler.state_dict(), "scaler", arrays,
        )
        torch_rng_state = _pack_value(torch.get_rng_state(), "torch-rng", arrays)
        cuda_rng_state = _pack_value(
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "cuda-rng", arrays,
        )
        state = {
            "adapter_schema_version": self.SCHEMA_VERSION,
            "torch_version": torch.__version__,
            "model_topology": list(model.state_dict()),
            "optimizer_topology": self._optimizer_topology(model, optimizer),
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler_state,
            "scaler": scaler_state,
            "torch_rng_state": torch_rng_state,
            "cuda_rng_state": cuda_rng_state,
            "numpy_rng_state": rng.bit_generator.state,
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
            lifecycle_state=LifecycleState.REPRESENTATION_READY,
        )

    def restore_latest(
        self,
        run_id: str,
        attempt_id: str,
        stage_name: str,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rng: np.random.Generator,
        scheduler: Any | None = None,
        scaler: Any | None = None,
    ) -> RestoredRepresentationState | None:
        loaded = self.store.latest(run_id, attempt_id, stage_name)
        if loaded is None:
            return None
        state = loaded.state
        required = {
            "adapter_schema_version", "torch_version", "model_topology",
            "optimizer_topology", "model", "optimizer", "scheduler", "scaler",
            "torch_rng_state", "cuda_rng_state", "numpy_rng_state",
            "epoch_history", "sampler_state",
        }
        if set(state) != required or state["adapter_schema_version"] != self.SCHEMA_VERSION:
            raise ValueError("unsupported representation checkpoint state")
        if state["torch_version"] != torch.__version__:
            raise ValueError("checkpoint Torch version does not match")
        if state["model_topology"] != list(model.state_dict()):
            raise ValueError("checkpoint model topology does not match")
        if state["optimizer_topology"] != self._optimizer_topology(model, optimizer):
            raise ValueError("checkpoint optimizer topology does not match")
        if (state["scheduler"] is None) != (scheduler is None):
            raise ValueError("checkpoint scheduler presence does not match")
        if (state["scaler"] is None) != (scaler is None):
            raise ValueError("checkpoint scaler presence does not match")
        numpy_rng_state = state["numpy_rng_state"]
        if numpy_rng_state.get("bit_generator") != rng.bit_generator.__class__.__name__:
            raise ValueError("checkpoint NumPy RNG bit generator does not match")

        model.load_state_dict(_unpack_value(state["model"], loaded.arrays), strict=True)
        optimizer.load_state_dict(_unpack_value(state["optimizer"], loaded.arrays))
        if scheduler is not None:
            scheduler.load_state_dict(_unpack_value(state["scheduler"], loaded.arrays))
        if scaler is not None:
            scaler.load_state_dict(_unpack_value(state["scaler"], loaded.arrays))
        torch.set_rng_state(_unpack_value(state["torch_rng_state"], loaded.arrays))
        cuda_rng_state = _unpack_value(state["cuda_rng_state"], loaded.arrays)
        if cuda_rng_state:
            if not torch.cuda.is_available() or len(cuda_rng_state) != torch.cuda.device_count():
                raise ValueError("checkpoint CUDA RNG topology does not match")
            torch.cuda.set_rng_state_all(cuda_rng_state)
        rng.bit_generator.state = numpy_rng_state
        return RestoredRepresentationState(
            metadata=loaded.metadata,
            epoch_history=tuple(dict(record) for record in state["epoch_history"]),
            sampler_state=dict(state["sampler_state"]),
        )

    def latest_metadata(
        self,
        run_id: str,
        attempt_id: str,
        stage_name: str,
    ) -> CheckpointMetadata | None:
        loaded = self.store.latest(run_id, attempt_id, stage_name)
        return None if loaded is None else loaded.metadata