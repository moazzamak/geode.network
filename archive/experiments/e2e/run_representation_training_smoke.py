import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.runtime import (
    LocalArtifactStore,
    LocalCheckpointStore,
    RepresentationCheckpointAdapter,
    train_representation,
)


def _objects(seed: int):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.Tanh(),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(5, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.8,
    )
    return model, optimizer, scheduler


def _nested_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_exact(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_exact(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _tensor_hash(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_qualification() -> dict:
    features = torch.tensor([
        [-1.0, -0.5, 0.2], [-0.8, -0.2, 0.1], [-0.4, -0.6, 0.3],
        [0.4, 0.6, -0.3], [0.8, 0.2, -0.1], [1.0, 0.5, -0.2],
    ])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    full_model, full_optimizer, full_scheduler = _objects(29)
    full_result = train_representation(
        full_model,
        full_optimizer,
        features,
        labels,
        epochs=4,
        batch_size=3,
        rng=np.random.default_rng(43),
        scheduler=full_scheduler,
    )

    with tempfile.TemporaryDirectory() as directory:
        adapter = RepresentationCheckpointAdapter(
            LocalCheckpointStore(LocalArtifactStore(directory))
        )
        interrupted_model, interrupted_optimizer, interrupted_scheduler = _objects(29)
        try:
            train_representation(
                interrupted_model,
                interrupted_optimizer,
                features,
                labels,
                epochs=4,
                batch_size=3,
                rng=np.random.default_rng(43),
                scheduler=interrupted_scheduler,
                checkpoint_adapter=adapter,
                checkpoint_input_hashes={"training_data": "fixed-data-v1"},
                fail_after_epoch=2,
            )
        except RuntimeError as error:
            if str(error) != "injected failure after representation epoch 2":
                raise
        else:
            raise AssertionError("qualification failure was not injected")

        resumed_model, resumed_optimizer, resumed_scheduler = _objects(999)
        resumed_result = train_representation(
            resumed_model,
            resumed_optimizer,
            features,
            labels,
            epochs=4,
            batch_size=3,
            rng=np.random.default_rng(999),
            scheduler=resumed_scheduler,
            checkpoint_adapter=adapter,
            checkpoint_input_hashes={"training_data": "fixed-data-v1"},
        )
        checkpoints = adapter.store.list_checkpoints(
            "representation-run", "attempt-1", "representation",
        )
        final_checkpoint = checkpoints[-1]

    model_exact = _nested_exact(
        resumed_model.state_dict(), full_model.state_dict(),
    )
    optimizer_exact = _nested_exact(
        resumed_optimizer.state_dict(), full_optimizer.state_dict(),
    )
    scheduler_exact = _nested_exact(
        resumed_scheduler.state_dict(), full_scheduler.state_dict(),
    )
    history_exact = resumed_result.epoch_history == full_result.epoch_history
    if not all((model_exact, optimizer_exact, scheduler_exact, history_exact)):
        raise AssertionError("resumed production training diverged")
    return {
        "schema_version": 1,
        "run_id": "e2-production-representation-smoke",
        "production_loop": "src.runtime.train_representation",
        "interruption_epoch": 2,
        "resumed_from_epoch": resumed_result.resumed_from_epoch,
        "final_epoch": len(resumed_result.epoch_history),
        "final_global_step": resumed_result.global_step,
        "checkpoint_count": len(checkpoints),
        "final_checkpoint_array_count": len(final_checkpoint.arrays),
        "history_exact": history_exact,
        "model_exact": model_exact,
        "optimizer_exact": optimizer_exact,
        "scheduler_exact": scheduler_exact,
        "model_sha256": {
            name: _tensor_hash(tensor)
            for name, tensor in resumed_model.state_dict().items()
        },
        "performance_claim": False,
        "torch_version": torch.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify production representation checkpoint resume.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_qualification()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(
        f"Qualified production resume at epoch {result['resumed_from_epoch']} "
        f"with {result['checkpoint_count']} checkpoints."
    )


if __name__ == "__main__":
    main()