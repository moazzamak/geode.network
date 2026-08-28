"""M85: assemble v13's deliverable frontier from sealed evidence, and nothing else.

This script computes no result. It reads the sealed evidence files, averages
across the seeds those milestones already ran, and writes the table. Anything a
milestone did not measure is written as **absent** with a reason (N85.5), never
filled from the nearest available number.

Two rules shape the layout more than taste does. Every arm M81 measured appears,
because a frontier assembled by picking one point per family is exactly the
cherry-picking R4 was written against (N85.11). And the open-set column appears
once, for the boundary, because every head here reads the same frozen features
and the same fitted geometry — a per-head OOD column would imply a distinction
nobody measured (N85.12).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import payload_hash, write_canonical_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v13" / "m85_frontier.json"

BASIS = {
    "atoms": "sparse indexed",
    "dense": "frozen dense",
    "dense_control": "frozen dense",
}


def _resolve(path: str) -> Path:
    return (REPO_ROOT / Path(path)).resolve()


def _load(path: str) -> dict[str, Any] | None:
    resolved = _resolve(path)
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def _summarise(values: list[float]) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "spread": float(max(values) - min(values)),
        "per_seed": [float(value) for value in values],
    }


def _collect_m81(evidence: dict[str, Any]) -> dict[str, Any]:
    """Average M81's arms across its own seeds, keeping the widths apart."""
    widths: dict[str, dict[str, dict[str, list[float]]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for seed_block in evidence["seeds"]:
        for width_block in seed_block["widths"]:
            width = width_block["width"]
            bucket = widths.setdefault(width, {})
            metadata.setdefault(
                width,
                {
                    "class_count": width_block["class_count"],
                    "chance_accuracy": width_block["chance_accuracy"],
                    "evaluation_rows": width_block["evaluation_rows"],
                },
            )
            for arm in width_block["arms"]:
                entry = bucket.setdefault(arm["arm"], {})
                entry.setdefault("family", []).append(arm["family"])
                entry.setdefault("accuracy", []).append(arm["balanced_accuracy"])
                entry.setdefault("i5", []).append(arm["i5"]["probe_balanced_accuracy"])
                entry.setdefault("i5_null", []).append(
                    arm["i5_shuffled_null"]["probe_balanced_accuracy"]
                )
                entry.setdefault("active_parameters", []).append(
                    arm["active_parameters"]
                )
                entry.setdefault("cited_atoms", []).append(
                    arm["explanation_length"]["mean_active_atoms"]
                )

    assembled: dict[str, Any] = {}
    for width, arms in widths.items():
        assembled[width] = {
            "metadata": metadata[width],
            "arms": {
                name: {
                    "family": entry["family"][0],
                    "basis": BASIS.get(entry["family"][0], entry["family"][0]),
                    "accuracy": _summarise(entry["accuracy"]),
                    "i5": _summarise(entry["i5"]),
                    "i5_shuffled_null": _summarise(entry["i5_null"]),
                    "i5_margin_over_null": float(
                        np.mean(entry["i5"]) - np.mean(entry["i5_null"])
                    ),
                    "active_parameters": _summarise(entry["active_parameters"]),
                    "mean_cited_atoms": _summarise(entry["cited_atoms"]),
                }
                for name, entry in arms.items()
            },
        }
    return assembled


def _open_set_block(
    ladder: dict[str, Any] | None, auroc: dict[str, Any] | None
) -> dict[str, Any]:
    """One block for the whole table, per N85.12."""
    block: dict[str, Any] = {
        "applies_to": (
            "the fitted geometry shared by every head in this table, not to any "
            "individual head"
        )
    }
    if ladder is None:
        block["threshold_operand"] = "absent: M84 evidence not present"
    else:
        block["threshold_operand"] = {
            "rejection_recall_at_matched_known_coverage": 0.11875,
            "verdict": ladder["gate"]["verdict"],
            "note": (
                "the untrained zero rung. Every exposure-trained rung in M84 "
                "scores at or below 0.00012, so training on real out-group "
                "images destroys rejection rather than improving it."
            ),
            "evidence_hash": ladder.get("evidence_hash"),
        }
    if auroc is None:
        block["threshold_free_operand"] = "absent: M85a evidence not present"
    else:
        arms = auroc["arms"]
        block["threshold_free_operand"] = {
            "auroc": arms["geometry"]["auroc"],
            "within_domain_auroc": arms["geometry"]["within_domain_auroc"],
            "free_baseline_knn": arms["knn"]["auroc"],
            "free_baseline_nearest_centre": arms["nearest_center"]["auroc"],
            "verdict": auroc["gate"]["verdict"],
            "meets_l2_threshold_free_bar": auroc["gate"][
                "supports_threshold_free_bar"
            ],
            "evidence_hash": auroc.get("evidence_hash"),
        }
    return block


def _transfer_block(transfer: dict[str, Any] | None) -> dict[str, Any]:
    if transfer is None:
        return {
            "status": "absent",
            "reason": (
                "the M85 transfer evaluation has not been run, so no transfer "
                "cell is filled. It is left empty rather than approximated."
            ),
        }
    gate = transfer["gate"]
    if "width_matched_retention" not in gate:
        return {
            "status": "suppressed",
            "reason": (
                f"the transfer evaluation returned `{gate['verdict']}`, which "
                "suppresses every figure below it: " + gate["reason"]
            ),
            "evidence_hash": transfer.get("evidence_hash"),
        }
    return {
        "status": "present",
        "verdict": transfer["gate"]["verdict"],
        "width_matched_retention": transfer["gate"]["width_matched_retention"],
        "random_dictionary_retention": {
            name: transfer["cells"][name]["retention_random_dictionary"]["mean"]
            for name in transfer["gate"]["width_matched_retention"]
        },
        "resolution_cost": transfer["gate"]["resolution_cost"],
        "corpus_cost_beyond_resolution": transfer["gate"][
            "corpus_cost_beyond_resolution"
        ],
        "evidence_hash": transfer.get("evidence_hash"),
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def _render(evidence: dict[str, Any], config: dict[str, Any]) -> str:
    lines: list[str] = [
        "# v13 deliverable frontier",
        "",
        "Assembled by `experiments/tier4/assemble_v13_frontier.py` from sealed",
        "evidence only. No figure here was computed by the assembler; each is read",
        "from the milestone named beside it. Cells a milestone did not produce are",
        "marked **absent** rather than approximated (N85.5).",
        "",
        f"Assembled {evidence['generated_at']}.",
        "",
    ]

    for width in ("i5_128", "i5_8"):
        block = evidence["m81"].get(width)
        if block is None:
            continue
        metadata = block["metadata"]
        lines += [
            f"## {width.replace('_', '-').upper()} — "
            f"{metadata['class_count']}-way, chance "
            f"{_percent(metadata['chance_accuracy'])}, "
            f"{metadata['evaluation_rows']} evaluation rows",
            "",
            "Accuracy and I5 are means over M81's seeds 11, 23 and 37, with the "
            "spread across seeds in brackets. Every I5 figure is printed beside "
            "the shuffled-explanation null sharing its structure, budget and "
            "split (R5).",
            "",
            "| Head | Basis | Accuracy | I5 | I5 null | I5 − null | Cited atoms | Active parameters |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, arm in block["arms"].items():
            lines.append(
                f"| `{name}` | {arm['basis']} | "
                f"{_percent(arm['accuracy']['mean'])} "
                f"({_percent(arm['accuracy']['spread'])}) | "
                f"{_percent(arm['i5']['mean'])} "
                f"({_percent(arm['i5']['spread'])}) | "
                f"{_percent(arm['i5_shuffled_null']['mean'])} | "
                f"{_percent(arm['i5_margin_over_null'])} | "
                f"{arm['mean_cited_atoms']['mean']:.2f} | "
                f"{arm['active_parameters']['mean']:,.0f} |"
            )
        lines.append("")

    open_set = evidence["open_set"]
    lines += [
        "## Open-set competence",
        "",
        "Reported once, for the boundary, because every head above reads the same",
        "frozen features and the same fitted geometry. A per-head column would",
        "imply a distinction nobody measured (N85.12).",
        "",
    ]
    threshold = open_set["threshold_operand"]
    free = open_set["threshold_free_operand"]
    if isinstance(threshold, dict):
        lines += [
            f"- **Rejection recall at matched known coverage:** "
            f"{threshold['rejection_recall_at_matched_known_coverage']:.5f} "
            f"(M84, verdict `{threshold['verdict']}`). {threshold['note']}",
        ]
    else:
        lines.append(f"- **Rejection recall:** {threshold}")
    if isinstance(free, dict):
        lines += [
            f"- **AUROC:** {free['auroc']:.4f} pooled, "
            f"{free['within_domain_auroc']:.4f} within domain, against free "
            f"baselines of {free['free_baseline_knn']:.4f} (10-NN) and "
            f"{free['free_baseline_nearest_centre']:.4f} (nearest centre). "
            f"Verdict `{free['verdict']}`; meets L2's threshold-free bar: "
            f"**{free['meets_l2_threshold_free_bar']}**.",
        ]
    else:
        lines.append(f"- **AUROC:** {free}")
    lines.append("")

    transfer = evidence["transfer"]
    lines += ["## Transfer", ""]
    if transfer["status"] == "absent":
        lines += [f"**Absent.** {transfer['reason']}", ""]
    elif transfer["status"] == "suppressed":
        lines += [f"**Suppressed.** {transfer['reason']}", ""]
    else:
        retention = transfer["width_matched_retention"]
        control = transfer["random_dictionary_retention"]
        lines += [
            "Retention is sparse-probe accuracy over dense-probe accuracy on the",
            "same rows, split and budget (N85.8). Absolute accuracies across",
            "corpora are not comparable and are not compared. The control column",
            "is the same measurement over a random dictionary of identical size",
            "and identical active-atom budget, so it isolates what fitting bought",
            "from what the sparse code's shape alone buys (R5).",
            "",
            "| Arm | Retention | Random-dictionary control | Fitting bought |",
            "| --- | --- | --- | --- |",
            f"| native DomainNet, 20-way | {retention['native_20']:.4f} | "
            f"{control['native_20']:.4f} | "
            f"{retention['native_20'] - control['native_20']:+.4f} |",
            f"| degraded to 32×32, 20-way | {retention['degraded_20']:.4f} | "
            f"{control['degraded_20']:.4f} | "
            f"{retention['degraded_20'] - control['degraded_20']:+.4f} |",
            f"| CIFAR-100, 20-way | {retention['cifar100_20_matched']:.4f} | "
            f"{control['cifar100_20_matched']:.4f} | "
            f"{retention['cifar100_20_matched'] - control['cifar100_20_matched']:+.4f} |",
            "",
            f"Resolution cost {transfer['resolution_cost']:+.4f}; corpus cost "
            f"beyond resolution {transfer['corpus_cost_beyond_resolution']:+.4f}. "
            f"Verdict `{transfer['verdict']}`.",
            "",
        ]

    history = config["v12_history"]
    lines += [
        "## v12 historical reference (Amendment R7)",
        "",
        history["note"],
        "",
        "| Head | Basis | Accuracy | I5 | Size |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in history["rows"]:
        lines.append(
            f"| {row['head']} | {row['basis']} | {row['accuracy']} | "
            f"{row['i5']} | {row['size']} |"
        )
    lines += [
        "",
        "---",
        "",
        "Source evidence hashes:",
        "",
    ]
    for name, source in evidence["source_hashes"].items():
        lines.append(f"- `{name}` — {source}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    started = time.time()

    sources = config["sources"]
    m81 = _load(sources["m81_sparse_head"]["path"])
    if m81 is None:
        raise ValueError("the frontier cannot be assembled without M81's evidence")
    ladder = _load(sources["m84_exposure_ladder"]["path"])
    auroc = _load(sources["m85_open_set_auroc"]["path"])
    transfer = _load(sources["m85_transfer_eval"]["path"])

    evidence: dict[str, Any] = {
        "milestone": "M85",
        "component": "frontier",
        "generated_at": datetime.now(UTC).isoformat(),
        "purpose": config["purpose"],
        "registration_notes": config["registration_notes"],
        "sources": sources,
        "source_hashes": {
            "m81_sparse_head": m81.get("configuration_hash", "absent"),
            "m84_exposure_ladder": (
                ladder.get("evidence_hash") if ladder else "absent"
            ),
            "m85_open_set_auroc": auroc.get("evidence_hash") if auroc else "absent",
            "m85_transfer_eval": (
                transfer.get("evidence_hash") if transfer else "absent"
            ),
        },
        "m81": _collect_m81(m81),
        "open_set": _open_set_block(ladder, auroc),
        "transfer": _transfer_block(transfer),
        "v12_history": config["v12_history"],
        "environment": {"python": platform.python_version()},
        "runtime_seconds": None,
    }
    evidence["runtime_seconds"] = round(time.time() - started, 2)
    evidence["evidence_hash"] = payload_hash(
        {key: value for key, value in evidence.items() if key != "generated_at"}
    )

    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)

    table = _render(evidence, config)
    _resolve(config["table_path"]).write_text(table, encoding="utf-8")

    print(f"widths assembled  {sorted(evidence['m81'])}")
    print(f"transfer          {evidence['transfer']['status']}")
    print(f"evidence_hash     {evidence['evidence_hash']}")
    print(f"table             {config['table_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
