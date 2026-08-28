from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import sha256_file
from experiments.common.v5_protocol import require_sha256


def _repository_path(root: Path, relative_path: str, kind: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"{kind} path must be repository-relative.")
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{kind} path must remain inside the repository.")
    return resolved


def validate_parent_file_locks(
    locks: Sequence[Mapping[str, Any]],
    repository_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(repository_root).resolve()
    if not locks:
        raise ValueError("At least one v6.1 parent file lock is required.")
    verified = []
    identifiers = set()
    for lock in locks:
        if set(lock) != {"id", "path", "sha256"}:
            raise ValueError("Parent file locks have an unsupported schema.")
        identifier = str(lock["id"])
        if not identifier or identifier in identifiers:
            raise ValueError("Parent file lock identifiers must be unique.")
        identifiers.add(identifier)
        path = _repository_path(root, str(lock["path"]), "Parent file lock")
        expected = require_sha256(str(lock["sha256"]), "parent_file.sha256")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Parent file {identifier!r} hash mismatch: "
                f"expected {expected}, got {actual}."
            )
        verified.append(
            {
                "id": identifier,
                "path": Path(str(lock["path"])).as_posix(),
                "sha256": actual,
                "bytes": path.stat().st_size,
            }
        )
    return verified


def validate_indexed_parent_locks(
    locks: Sequence[Mapping[str, Any]],
    repository_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(repository_root).resolve()
    if not locks:
        raise ValueError("At least one indexed v6 parent lock is required.")
    verified = []
    identifiers = set()
    for lock in locks:
        if set(lock) != {
            "id",
            "root",
            "index_path",
            "index_sha256",
            "required_paths",
        }:
            raise ValueError("Indexed parent locks have an unsupported schema.")
        identifier = str(lock["id"])
        if not identifier or identifier in identifiers:
            raise ValueError("Indexed parent identifiers must be unique.")
        identifiers.add(identifier)
        artifact_root = _repository_path(root, str(lock["root"]), "Artifact root")
        index_path = _repository_path(
            root, str(lock["index_path"]), "Artifact index"
        )
        expected_index_hash = require_sha256(
            str(lock["index_sha256"]), "index_sha256"
        )
        if sha256_file(index_path) != expected_index_hash:
            raise ValueError(f"Artifact index hash mismatch for {identifier!r}.")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            set(index) != {"schema_version", "artifacts"}
            or index["schema_version"] != 1
            or not isinstance(index["artifacts"], list)
        ):
            raise ValueError(f"Unsupported artifact index for {identifier!r}.")
        indexed_paths = set()
        total_bytes = 0
        for item in index["artifacts"]:
            if set(item) != {"path", "sha256", "bytes"}:
                raise ValueError("Artifact index entries have an unsupported schema.")
            relative = str(item["path"])
            if relative in indexed_paths:
                raise ValueError("Artifact index paths must be unique.")
            indexed_paths.add(relative)
            path = _repository_path(
                artifact_root, relative, "Indexed artifact"
            )
            expected_hash = require_sha256(
                str(item["sha256"]), "artifact.sha256"
            )
            if sha256_file(path) != expected_hash:
                raise ValueError(
                    f"Indexed artifact hash mismatch for {identifier!r}/{relative}."
                )
            actual_bytes = path.stat().st_size
            if actual_bytes != int(item["bytes"]):
                raise ValueError(
                    f"Indexed artifact byte count mismatch for "
                    f"{identifier!r}/{relative}."
                )
            total_bytes += actual_bytes
        required = {str(path) for path in lock["required_paths"]}
        missing = required - indexed_paths
        if missing:
            raise ValueError(
                f"Indexed parent {identifier!r} is missing required paths "
                f"{sorted(missing)}."
            )
        verified.append(
            {
                "id": identifier,
                "root": Path(str(lock["root"])).as_posix(),
                "index_path": Path(str(lock["index_path"])).as_posix(),
                "index_sha256": expected_index_hash,
                "artifact_count": len(indexed_paths),
                "artifact_bytes": total_bytes,
                "required_paths": sorted(required),
            }
        )
    return verified


def validate_representation_lineage(
    records: Sequence[Mapping[str, Any]],
    *,
    m30_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_seed = m30_evidence.get("seed_results")
    if not isinstance(by_seed, Mapping):
        raise ValueError("M30 parent evidence does not contain seed results.")
    if set(by_seed) != {"11", "23", "37"}:
        raise ValueError("M30 parent evidence must contain S2 seeds 11, 23, and 37.")
    if len(records) != 3:
        raise ValueError("v6.1 requires exactly three S2 lineage records.")
    verified = []
    seen = set()
    for record in records:
        required = {
            "seed",
            "parent_representation_hash",
            "directional_representation_hash",
            "training_split_hash",
            "development_split_hash",
        }
        if set(record) != required:
            raise ValueError("Representation lineage has an unsupported schema.")
        seed = int(record["seed"])
        if seed in seen or seed not in {11, 23, 37}:
            raise ValueError("Representation lineage seeds must be unique S2 seeds.")
        seen.add(seed)
        parent = by_seed.get(str(seed))
        if not isinstance(parent, Mapping):
            raise ValueError(f"M30 parent evidence is missing seed {seed}.")
        expected = {
            "parent_representation_hash": parent["parent_representation_hash"],
            "directional_representation_hash": parent[
                "directional_representation_hash"
            ],
            "training_split_hash": parent["split_hashes"]["train"],
            "development_split_hash": parent["split_hashes"]["dev"],
        }
        observed = {}
        for field, expected_value in expected.items():
            value = require_sha256(str(record[field]), field)
            if value != expected_value:
                raise ValueError(
                    f"Seed {seed} {field} does not match the M30 parent evidence."
                )
            observed[field] = value
        verified.append({"seed": seed, **observed})
    return sorted(verified, key=lambda item: item["seed"])


def validate_v61_config(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "amendment",
        "stage",
        "parent_file_locks",
        "indexed_parent_locks",
        "representation_lineage",
        "amended_primitive",
        "weighted_readout",
        "budgets",
        "claims",
        "closed_branches",
        "test_labels_opened",
    }
    if set(payload) != required or payload.get("schema_version") != 1:
        raise ValueError("Unsupported v6.1 A0 configuration schema.")
    if payload["amendment"] != "v6.1" or payload["stage"] != "A0":
        raise ValueError("The amendment lock must identify v6.1/A0.")
    if payload["test_labels_opened"] is not False:
        raise PermissionError("v6.1 A0 cannot open final-test labels.")

    primitive = payload["amended_primitive"]
    if not isinstance(primitive, Mapping) or set(primitive) != {
        "family",
        "rank",
        "minimum_seed_rule",
        "minimum_support",
        "normalization",
        "direction",
        "angular_radius_units",
        "residual_scale",
        "primary_score",
    }:
        raise ValueError("The amended primitive schema is unsupported.")
    if (
        primitive["family"] != "tangent_cap"
        or int(primitive["rank"]) != 32
        or primitive["minimum_seed_rule"] != "r_plus_2"
        or int(primitive["minimum_support"]) != 34
        or primitive["normalization"] != "explicit_l2"
        or primitive["direction"] != "unit_mean"
        or primitive["angular_radius_units"] != "radians"
        or primitive["residual_scale"] != "isotropic"
        or primitive["primary_score"] != "normalized_tangent_radial"
    ):
        raise ValueError("The v6.1 tangent-cap contract does not match the plan.")

    readout = payload["weighted_readout"]
    if not isinstance(readout, Mapping) or set(readout) != {
        "family",
        "constraint",
        "parameterization",
        "optimizer",
        "regularization",
        "maximum_iterations",
        "gradient_tolerance",
        "initialization",
        "temperature_policy",
    }:
        raise ValueError("The weighted-readout schema is unsupported.")
    if (
        readout["family"] != "nonnegative_component_mixture"
        or readout["constraint"] != "per_class_simplex"
        or readout["parameterization"] != "softmax_log_weights"
        or readout["optimizer"] != "lbfgs"
        or float(readout["regularization"]) != 1e-4
        or int(readout["maximum_iterations"]) != 500
        or float(readout["gradient_tolerance"]) != 1e-8
        or readout["initialization"] != "zero_equal_weights"
        or readout["temperature_policy"] != "one_global"
    ):
        raise ValueError("The v6.1 weighted-readout contract does not match the plan.")

    budgets = payload["budgets"]
    if not isinstance(budgets, list) or len(budgets) != 5:
        raise ValueError("v6.1 requires the five registered predictive budgets.")
    budget_ids = set()
    for budget in budgets:
        if set(budget) != {
            "id",
            "components",
            "parameter_limit",
            "candidate_evaluation_limit",
        }:
            raise ValueError("A v6.1 budget has an unsupported schema.")
        identifier = str(budget["id"])
        if identifier in budget_ids:
            raise ValueError("v6.1 budget identifiers must be unique.")
        budget_ids.add(identifier)
        if (
            int(budget["components"]) != 46
            or int(budget["parameter_limit"]) < 1
            or int(budget["candidate_evaluation_limit"]) != 5000
        ):
            raise ValueError("A v6.1 predictive budget is invalid.")

    claims = payload["claims"]
    if not isinstance(claims, list) or not claims:
        raise ValueError("v6.1 A0 requires a non-empty claim snapshot.")
    claim_ids = set()
    allowed_statuses = {"supported", "negative", "blocked", "eligible"}
    for claim in claims:
        if set(claim) != {"id", "status", "statement", "evidence"}:
            raise ValueError("A v6.1 claim has an unsupported schema.")
        if claim["id"] in claim_ids or claim["status"] not in allowed_statuses:
            raise ValueError("Claim identifiers and statuses must be valid.")
        claim_ids.add(claim["id"])
        if not claim["statement"] or not claim["evidence"]:
            raise ValueError("Claims require statements and evidence.")

    required_closed = {
        "M32_topology",
        "M33_ood",
        "M35_predictive_confirmation",
        "M36_migration",
        "M37_amortization",
    }
    if set(payload["closed_branches"]) != required_closed:
        raise ValueError("v6.1 must preserve all branches closed by M31.")
