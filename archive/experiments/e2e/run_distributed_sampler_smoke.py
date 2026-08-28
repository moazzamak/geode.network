import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from experiments.e2e.run_representation_training_smoke import _nested_exact, _objects
from src.runtime import (
    LocalArtifactStore,
    LocalCheckpointStore,
    RepresentationCheckpointAdapter,
    train_representation,
)


def _sampler(features: torch.Tensor, rank: int) -> DistributedSampler:
    return DistributedSampler(
        features,
        num_replicas=2,
        rank=rank,
        shuffle=True,
        seed=47,
        drop_last=False,
    )


def _rank_shards(features: torch.Tensor, rank: int) -> list[list[int]]:
    sampler = _sampler(features, rank)
    shards = []
    for epoch in range(4):
        sampler.set_epoch(epoch)
        shards.append([int(index) for index in sampler])
    return shards


def _qualify_rank(
    features: torch.Tensor,
    labels: torch.Tensor,
    rank: int,
) -> dict:
    full_model, full_optimizer, full_scheduler = _objects(31)
    full_result = train_representation(
        full_model,
        full_optimizer,
        features,
        labels,
        epochs=4,
        batch_size=2,
        rng=np.random.default_rng(53),
        scheduler=full_scheduler,
        sampler=_sampler(features, rank),
    )
    with tempfile.TemporaryDirectory() as directory:
        adapter = RepresentationCheckpointAdapter(
            LocalCheckpointStore(LocalArtifactStore(directory))
        )
        model, optimizer, scheduler = _objects(31)
        try:
            train_representation(
                model,
                optimizer,
                features,
                labels,
                epochs=4,
                batch_size=2,
                rng=np.random.default_rng(53),
                scheduler=scheduler,
                sampler=_sampler(features, rank),
                checkpoint_adapter=adapter,
                fail_after_epoch=2,
            )
        except RuntimeError as error:
            if str(error) != "injected failure after representation epoch 2":
                raise
        else:
            raise AssertionError("qualification failure was not injected")

        mismatch_rejected = False
        mismatched_model, mismatched_optimizer, mismatched_scheduler = _objects(999)
        try:
            train_representation(
                mismatched_model,
                mismatched_optimizer,
                features,
                labels,
                epochs=4,
                batch_size=2,
                rng=np.random.default_rng(999),
                scheduler=mismatched_scheduler,
                sampler=_sampler(features, 1 - rank),
                checkpoint_adapter=adapter,
            )
        except ValueError as error:
            mismatch_rejected = str(error) == (
                "checkpoint sampler state does not match epoch boundary"
            )

        resumed_model, resumed_optimizer, resumed_scheduler = _objects(999)
        resumed_result = train_representation(
            resumed_model,
            resumed_optimizer,
            features,
            labels,
            epochs=4,
            batch_size=2,
            rng=np.random.default_rng(999),
            scheduler=resumed_scheduler,
            sampler=_sampler(features, rank),
            checkpoint_adapter=adapter,
        )
    model_exact = _nested_exact(resumed_model.state_dict(), full_model.state_dict())
    optimizer_exact = _nested_exact(
        resumed_optimizer.state_dict(), full_optimizer.state_dict(),
    )
    scheduler_exact = _nested_exact(
        resumed_scheduler.state_dict(), full_scheduler.state_dict(),
    )
    history_exact = resumed_result.epoch_history == full_result.epoch_history
    if not all((
        mismatch_rejected,
        model_exact,
        optimizer_exact,
        scheduler_exact,
        history_exact,
    )):
        raise AssertionError(f"distributed qualification failed for rank {rank}")
    return {
        "rank": rank,
        "resumed_from_epoch": resumed_result.resumed_from_epoch,
        "final_global_step": resumed_result.global_step,
        "shards": _rank_shards(features, rank),
        "rank_mismatch_rejected": mismatch_rejected,
        "history_exact": history_exact,
        "model_exact": model_exact,
        "optimizer_exact": optimizer_exact,
        "scheduler_exact": scheduler_exact,
    }


def run_qualification() -> dict:
    features = torch.tensor([
        [-1.0, -0.5, 0.2], [-0.8, -0.2, 0.1], [-0.4, -0.6, 0.3],
        [-0.2, -0.1, 0.4], [0.2, 0.1, -0.4], [0.4, 0.6, -0.3],
        [0.8, 0.2, -0.1], [1.0, 0.5, -0.2],
    ])
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    ranks = [_qualify_rank(features, labels, rank) for rank in range(2)]
    full_coverage = all(
        sorted(ranks[0]["shards"][epoch] + ranks[1]["shards"][epoch])
        == list(range(len(features)))
        and not set(ranks[0]["shards"][epoch]) & set(ranks[1]["shards"][epoch])
        for epoch in range(4)
    )
    if not full_coverage:
        raise AssertionError("distributed shards do not form an exact partition")
    return {
        "schema_version": 1,
        "run_id": "e2-distributed-sampler-smoke",
        "sampler": "torch.utils.data.DistributedSampler",
        "world_size": 2,
        "epochs": 4,
        "full_disjoint_coverage": full_coverage,
        "ranks": ranks,
        "performance_claim": False,
        "torch_version": torch.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify two-rank distributed sampler checkpoint resume.",
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
        f"Qualified {result['world_size']} sampler ranks across "
        f"{result['epochs']} epochs."
    )


if __name__ == "__main__":
    main()