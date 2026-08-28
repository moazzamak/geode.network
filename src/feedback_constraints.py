from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from src.rejection_buffer import RejectionRecord


@dataclass(frozen=True)
class PairwiseConstraint:
    left_record_id: int
    right_record_id: int
    relation: str

    def __post_init__(self) -> None:
        if self.left_record_id == self.right_record_id:
            raise ValueError("A pairwise constraint requires two records.")
        if self.relation not in {"must_link", "cannot_link"}:
            raise ValueError("relation must be must_link or cannot_link.")


@dataclass(frozen=True)
class ConstraintMetric:
    feature_weights: tuple[float, ...]
    must_link_count: int
    cannot_link_count: int
    version: str = "pairwise-diagonal-v1"


@dataclass(frozen=True)
class ConstraintConsistency:
    is_consistent: bool
    direct_conflict_count: int
    transitive_conflict_count: int


@dataclass(frozen=True)
class ConstraintConfirmation:
    accepted: tuple[PairwiseConstraint, ...]
    disagreement_count: int


def confirm_pairwise_constraints(
    first_answers: tuple[PairwiseConstraint, ...],
    second_answers: tuple[PairwiseConstraint, ...],
) -> ConstraintConfirmation:
    def by_pair(
        constraints: tuple[PairwiseConstraint, ...],
    ) -> dict[tuple[int, int], PairwiseConstraint]:
        indexed = {
            tuple(sorted((item.left_record_id, item.right_record_id))): item
            for item in constraints
        }
        if len(indexed) != len(constraints):
            raise ValueError("Each answer set must contain unique pairs.")
        return indexed

    first_by_pair = by_pair(first_answers)
    second_by_pair = by_pair(second_answers)
    if first_by_pair.keys() != second_by_pair.keys():
        raise ValueError("Answer sets must cover identical pairs.")
    accepted = tuple(
        first_by_pair[pair]
        for pair in sorted(first_by_pair)
        if first_by_pair[pair].relation == second_by_pair[pair].relation
    )
    return ConstraintConfirmation(
        accepted=accepted,
        disagreement_count=len(first_answers) - len(accepted),
    )


def validate_pairwise_constraints(
    constraints: tuple[PairwiseConstraint, ...],
) -> ConstraintConsistency:
    relations_by_pair: dict[tuple[int, int], set[str]] = {}
    record_ids = set()
    for constraint in constraints:
        pair = tuple(sorted((
            constraint.left_record_id,
            constraint.right_record_id,
        )))
        relations_by_pair.setdefault(pair, set()).add(constraint.relation)
        record_ids.update(pair)

    direct_conflicts = {
        pair for pair, relations in relations_by_pair.items()
        if len(relations) > 1
    }
    parents = {record_id: record_id for record_id in record_ids}

    def find(record_id: int) -> int:
        while parents[record_id] != record_id:
            parents[record_id] = parents[parents[record_id]]
            record_id = parents[record_id]
        return record_id

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for pair, relations in relations_by_pair.items():
        if "must_link" in relations:
            union(*pair)
    transitive_conflicts = sum(
        pair not in direct_conflicts and find(pair[0]) == find(pair[1])
        for pair, relations in relations_by_pair.items()
        if "cannot_link" in relations
    )
    return ConstraintConsistency(
        is_consistent=not direct_conflicts and transitive_conflicts == 0,
        direct_conflict_count=len(direct_conflicts),
        transitive_conflict_count=transitive_conflicts,
    )


def build_pairwise_constraints(
    record_ids: np.ndarray,
    confirmed_labels: np.ndarray,
    *,
    maximum_per_relation: int = 256,
) -> tuple[PairwiseConstraint, ...]:
    record_ids = np.asarray(record_ids, dtype=np.int64)
    confirmed_labels = np.asarray(confirmed_labels, dtype=np.int64)
    if record_ids.ndim != 1 or confirmed_labels.shape != record_ids.shape:
        raise ValueError("record_ids and confirmed_labels must be aligned vectors.")
    if maximum_per_relation <= 0:
        raise ValueError("maximum_per_relation must be positive.")

    constraints = []
    relation_counts = {"must_link": 0, "cannot_link": 0}
    for left_index, right_index in combinations(range(len(record_ids)), 2):
        relation = (
            "must_link"
            if confirmed_labels[left_index] == confirmed_labels[right_index]
            else "cannot_link"
        )
        if relation_counts[relation] >= maximum_per_relation:
            continue
        constraints.append(PairwiseConstraint(
            left_record_id=int(record_ids[left_index]),
            right_record_id=int(record_ids[right_index]),
            relation=relation,
        ))
        relation_counts[relation] += 1
    return tuple(constraints)


def fit_diagonal_constraint_metric(
    embeddings_by_record_id: dict[int, np.ndarray],
    constraints: tuple[PairwiseConstraint, ...],
) -> ConstraintMetric:
    if not embeddings_by_record_id:
        raise ValueError("At least one embedding is required.")
    embedding_dimension = len(next(iter(embeddings_by_record_id.values())))
    deltas = {"must_link": [], "cannot_link": []}
    for constraint in constraints:
        left = np.asarray(
            embeddings_by_record_id[constraint.left_record_id], dtype=np.float64,
        )
        right = np.asarray(
            embeddings_by_record_id[constraint.right_record_id], dtype=np.float64,
        )
        if left.shape != (embedding_dimension,) or right.shape != left.shape:
            raise ValueError("Constraint embeddings must share one dimension.")
        deltas[constraint.relation].append((left - right) ** 2)

    must_link_count = len(deltas["must_link"])
    cannot_link_count = len(deltas["cannot_link"])
    if must_link_count == 0 or cannot_link_count == 0:
        weights = np.ones(embedding_dimension, dtype=np.float64)
    else:
        within = np.mean(deltas["must_link"], axis=0)
        between = np.mean(deltas["cannot_link"], axis=0)
        weights = between / np.maximum(within, 1e-8)
        weights = np.clip(weights / np.mean(weights), 0.1, 10.0)

    return ConstraintMetric(
        feature_weights=tuple(float(value) for value in weights),
        must_link_count=must_link_count,
        cannot_link_count=cannot_link_count,
    )


def refine_rejection_partition(
    clusters: tuple[tuple[RejectionRecord, ...], ...],
    constraints: tuple[PairwiseConstraint, ...],
) -> tuple[tuple[RejectionRecord, ...], ...]:
    records = {
        int(record.source_sample_id): record
        for cluster in clusters
        for record in cluster
        if record.source_sample_id is not None
    }
    if not records or not constraints:
        return clusters

    parents = {record_id: record_id for record_id in records}

    def find(record_id: int) -> int:
        while parents[record_id] != record_id:
            parents[record_id] = parents[parents[record_id]]
            record_id = parents[record_id]
        return record_id

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for constraint in constraints:
        if (
            constraint.relation == "must_link"
            and constraint.left_record_id in records
            and constraint.right_record_id in records
        ):
            union(constraint.left_record_id, constraint.right_record_id)

    cannot_links = {
        frozenset((find(constraint.left_record_id), find(constraint.right_record_id)))
        for constraint in constraints
        if constraint.relation == "cannot_link"
        and constraint.left_record_id in records
        and constraint.right_record_id in records
        and find(constraint.left_record_id) != find(constraint.right_record_id)
    }
    provisional_groups: list[set[int]] = []
    for cluster in clusters:
        component_ids = sorted({
            find(int(record.source_sample_id))
            for record in cluster
            if record.source_sample_id is not None
        })
        cluster_groups: list[set[int]] = []
        for component_id in component_ids:
            compatible_group = next((
                group for group in cluster_groups
                if all(
                    frozenset((component_id, member)) not in cannot_links
                    for member in group
                )
            ), None)
            if compatible_group is None:
                cluster_groups.append({component_id})
            else:
                compatible_group.add(component_id)
        provisional_groups.extend(cluster_groups)

    merged = True
    while merged:
        merged = False
        for left_index in range(len(provisional_groups)):
            for right_index in range(left_index + 1, len(provisional_groups)):
                if provisional_groups[left_index] & provisional_groups[right_index]:
                    provisional_groups[left_index].update(
                        provisional_groups.pop(right_index)
                    )
                    merged = True
                    break
            if merged:
                break

    component_members: dict[int, list[RejectionRecord]] = {}
    for record_id, record in records.items():
        component_members.setdefault(find(record_id), []).append(record)
    return tuple(
        tuple(
            record
            for component_id in sorted(group)
            for record in component_members[component_id]
        )
        for group in provisional_groups
    )


def select_constraint_queries(
    constraints: tuple[PairwiseConstraint, ...],
    embeddings_by_record_id: dict[int, np.ndarray],
    partition_record_ids: tuple[tuple[int, ...], ...],
    *,
    budget: int,
    strategy: str,
    random_state: int = 0,
) -> tuple[PairwiseConstraint, ...]:
    if budget <= 0:
        raise ValueError("budget must be positive.")
    if strategy not in {"active", "random", "dense"}:
        raise ValueError("strategy must be active, random, or dense.")
    if strategy == "dense" or budget >= len(constraints):
        return constraints

    ordered_constraints = sorted(
        constraints,
        key=lambda item: (item.left_record_id, item.right_record_id),
    )
    if strategy == "random":
        indices = np.random.default_rng(random_state).permutation(
            len(ordered_constraints)
        )[:budget]
        return tuple(ordered_constraints[int(index)] for index in indices)

    membership = {
        record_id: group_index
        for group_index, group in enumerate(partition_record_ids)
        for record_id in group
    }

    def distance(constraint: PairwiseConstraint) -> float:
        left = np.asarray(embeddings_by_record_id[constraint.left_record_id])
        right = np.asarray(embeddings_by_record_id[constraint.right_record_id])
        return float(np.linalg.norm(left - right))

    within = sorted((
        constraint for constraint in ordered_constraints
        if membership.get(constraint.left_record_id) is not None
        and membership.get(constraint.left_record_id)
        == membership.get(constraint.right_record_id)
    ), key=lambda item: (-distance(item), item.left_record_id, item.right_record_id))
    across = sorted((
        constraint for constraint in ordered_constraints
        if constraint not in within
    ), key=lambda item: (distance(item), item.left_record_id, item.right_record_id))
    selected = []
    while len(selected) < budget and (within or across):
        if within:
            selected.append(within.pop(0))
        if len(selected) < budget and across:
            selected.append(across.pop(0))
    return tuple(selected)
