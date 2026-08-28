"""Resumable Torch representation training at deterministic epoch boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from src.runtime.representation_checkpoint import RepresentationCheckpointAdapter


@dataclass(frozen=True)
class RepresentationTrainingResult:
    epoch_history: tuple[dict[str, Any], ...]
    global_step: int
    resumed_from_epoch: int
    checkpoints_written: int


def _sampler_state(
    sampler: DistributedSampler | None,
    epoch: int,
    position: int,
) -> dict[str, Any]:
    if sampler is None:
        return {"epoch": epoch, "position": position, "type": "numpy_permutation"}
    return {
        "drop_last": bool(sampler.drop_last),
        "epoch": epoch,
        "num_replicas": int(sampler.num_replicas),
        "position": position,
        "rank": int(sampler.rank),
        "seed": int(sampler.seed),
        "shuffle": bool(sampler.shuffle),
        "type": "torch_distributed",
    }


def train_representation(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    checkpoint_adapter: RepresentationCheckpointAdapter | None = None,
    checkpoint_run_id: str = "representation-run",
    checkpoint_attempt_id: str = "attempt-1",
    checkpoint_stage_name: str = "representation",
    checkpoint_input_hashes: Mapping[str, str] | None = None,
    fail_after_epoch: int | None = None,
    sampler: DistributedSampler | None = None,
) -> RepresentationTrainingResult:
    """Train a classifier representation with exact epoch-boundary resume."""
    if features.ndim < 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features and labels must have matching sample dimensions")
    if len(features) == 0:
        raise ValueError("representation training requires samples")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if sampler is not None and len(sampler.dataset) != len(features):
        raise ValueError("distributed sampler dataset length does not match features")

    start_epoch = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    if checkpoint_adapter is not None:
        restored = checkpoint_adapter.restore_latest(
            checkpoint_run_id,
            checkpoint_attempt_id,
            checkpoint_stage_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            rng=rng,
        )
        if restored is not None:
            start_epoch = restored.metadata.epoch
            global_step = restored.metadata.global_step
            history = list(restored.epoch_history)
            expected_position = len(features) if sampler is None else len(sampler)
            if restored.sampler_state != _sampler_state(
                sampler, start_epoch, expected_position,
            ):
                raise ValueError("checkpoint sampler state does not match epoch boundary")
            if len(history) != start_epoch:
                raise ValueError("checkpoint history does not match completed epochs")
    if start_epoch > epochs:
        raise ValueError("checkpoint epoch exceeds requested training epochs")

    checkpoints_written = 0
    model.train()
    for epoch in range(start_epoch, epochs):
        if sampler is None:
            permutation = rng.permutation(len(features))
        else:
            sampler.set_epoch(epoch)
            permutation = np.asarray(list(sampler), dtype=np.int64)
        batch_losses = []
        for start in range(0, len(permutation), batch_size):
            indices = torch.as_tensor(
                permutation[start:start + batch_size],
                dtype=torch.long,
                device=features.device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(features.index_select(0, indices)),
                labels.index_select(0, indices),
            )
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch_losses.append(float(loss.detach()))
            global_step += 1
        if scheduler is not None:
            scheduler.step()
        history.append({
            "epoch": epoch + 1,
            "batch_training_loss": float(np.mean(batch_losses)),
            "global_step": global_step,
        })
        if checkpoint_adapter is not None:
            checkpoint_adapter.save(
                checkpoint_run_id,
                checkpoint_attempt_id,
                checkpoint_stage_name,
                epoch + 1,
                global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                rng=rng,
                epoch_history=history,
                sampler_state=_sampler_state(
                    sampler, epoch + 1, len(permutation),
                ),
                input_hashes=checkpoint_input_hashes,
            )
            checkpoints_written += 1
        if fail_after_epoch == epoch + 1:
            raise RuntimeError(
                f"injected failure after representation epoch {epoch + 1}"
            )
    return RepresentationTrainingResult(
        epoch_history=tuple(history),
        global_step=global_step,
        resumed_from_epoch=start_epoch,
        checkpoints_written=checkpoints_written,
    )