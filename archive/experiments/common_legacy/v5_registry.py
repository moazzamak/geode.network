from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping

from experiments.common.experiment_manifest import canonical_json, experiment_id
from experiments.common.v5_protocol import DataStage, seeds_for_stage


@dataclass(frozen=True)
class ExperimentCell:
    milestone: str
    stage: DataStage
    dataset: str
    seed: int
    representation: str
    head: str
    readout: str
    split_hash: str
    feature_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "milestone": self.milestone,
            "stage": self.stage.value,
            "dataset": self.dataset,
            "seed": self.seed,
            "representation": self.representation,
            "head": self.head,
            "readout": self.readout,
            "split_hash": self.split_hash,
            "feature_hash": self.feature_hash,
        }

    @property
    def cell_id(self) -> str:
        return experiment_id(self.to_dict())


def expand_matrix(
    *,
    milestone: str,
    stage: DataStage,
    dataset: str,
    representations: list[str],
    heads: list[str],
    readouts: list[str],
    split_hashes: Mapping[str, str],
    feature_hashes: Mapping[str, str],
    declared_seeds: tuple[int, ...] | None = None,
) -> list[ExperimentCell]:
    if not milestone or not dataset:
        raise ValueError("milestone and dataset are required.")
    for name, values in (
        ("representations", representations),
        ("heads", heads),
        ("readouts", readouts),
    ):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must contain unique values.")
    if set(split_hashes) != set(representations):
        raise ValueError("split_hashes must cover every representation exactly.")
    if set(feature_hashes) != set(representations):
        raise ValueError("feature_hashes must cover every representation exactly.")

    seeds = seeds_for_stage(stage, declared_seeds)
    cells = [
        ExperimentCell(
            milestone=milestone,
            stage=stage,
            dataset=dataset,
            seed=seed,
            representation=representation,
            head=head,
            readout=readout,
            split_hash=split_hashes[representation],
            feature_hash=feature_hashes[representation],
        )
        for seed, representation, head, readout in product(
            seeds, representations, heads, readouts
        )
    ]
    return sorted(cells, key=lambda cell: canonical_json(cell.to_dict()))


def validate_required_controls(
    cells: list[ExperimentCell],
    required_heads: set[str],
) -> None:
    if not cells:
        raise ValueError("Experiment matrix is empty.")
    grouped: dict[tuple[DataStage, str, int, str, str], set[str]] = {}
    split_hashes: dict[tuple[DataStage, str, int, str, str], set[str]] = {}
    for cell in cells:
        key = (
            cell.stage,
            cell.dataset,
            cell.seed,
            cell.representation,
            cell.readout,
        )
        grouped.setdefault(key, set()).add(cell.head)
        split_hashes.setdefault(key, set()).add(cell.split_hash)
    for key, heads in grouped.items():
        missing = required_heads - heads
        if missing:
            raise ValueError(f"Matrix group {key} is missing controls {sorted(missing)}.")
        if len(split_hashes[key]) != 1:
            raise ValueError(f"Matrix group {key} contains unequal split hashes.")
    invariant_values = {
        (cell.milestone, cell.stage, cell.dataset) for cell in cells
    }
    if len(invariant_values) != 1:
        raise ValueError("One matrix cannot mix milestones, stages, or datasets.")
    matrix_split_hashes = {cell.split_hash for cell in cells}
    if len(matrix_split_hashes) != 1:
        raise ValueError("Compared representations must use identical split hashes.")


def validate_matched_comparison(cells: list[ExperimentCell]) -> None:
    if not cells:
        raise ValueError("Comparison requires at least one cell.")
    invariant_values = {
        (cell.milestone, cell.stage, cell.dataset, cell.seed) for cell in cells
    }
    if len(invariant_values) != 1:
        raise ValueError(
            "Compared cells must share milestone, stage, dataset, and seed."
        )
    split_hashes = {cell.split_hash for cell in cells}
    if len(split_hashes) != 1:
        raise ValueError("Compared cells must have identical split hashes.")
