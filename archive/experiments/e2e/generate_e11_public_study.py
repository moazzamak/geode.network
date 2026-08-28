"""Reproduce E11 publication summaries from immutable artifacts only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _artifact_index(config: dict[str, Any], repository: Path) -> dict[str, Any]:
    records = []
    for specification in config["artifacts"]:
        relative = Path(specification["path"])
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing E11 artifact: {relative.as_posix()}")
        payload = _read_json(path)
        records.append({
            "kind": specification["kind"],
            "milestone": specification["milestone"],
            "path": relative.as_posix(),
            "schema_version": payload.get("schema_version"),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        })
    return {
        "schema_version": 1,
        "study": config["study"],
        "repository_evidence_cutoff": config["repository_evidence_cutoff"],
        "milestone_status": config["milestone_status"],
        "artifacts": records,
    }


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _selected_ood(record: dict[str, Any], domain: str) -> dict[str, float]:
    selected = record["selected_ood_score"]
    entry = record["final_ood"][selected][domain]
    return entry.get("detection", entry)


def _study_summary(repository: Path, config: dict[str, Any]) -> dict[str, Any]:
    results = repository / "logs" / "results"
    e4 = _read_json(results / "e4_cifar_qualification.json")
    e5 = _read_json(results / "e5_routing_qualification.json")
    e6 = _read_json(results / "e6_transfer_qualification.json")
    e7 = _read_json(results / "e7_domainnet_preflight_python312.json")
    e8 = _read_json(results / "e8_cross_modal_qualification.json")
    e9 = _read_json(results / "e9_transactional_adaptation.json")
    e10 = _read_json(results / "e10_production_rehearsal.json")
    if e4["seeds"] != config["principal_seeds"]:
        raise ValueError("E4 seeds do not match the frozen E11 principal seeds")

    methods = ("geode_multinomial", "logistic_regression", "rbf_svm")
    classification = []
    for method in methods:
        values = [
            float(record["final_classification"][method]["balanced_accuracy"])
            for record in e4["records"]
        ]
        classification.append({
            "method": method,
            "mean_balanced_accuracy": _mean(values),
            "stdev_balanced_accuracy": _stdev(values),
            "per_seed": values,
        })
    near = [_selected_ood(record, "near") for record in e4["records"]]
    far = [_selected_ood(record, "far") for record in e4["records"]]
    elapsed = [float(record["elapsed_seconds"]) for record in e4["records"]]
    experts = [int(record["model_counts"]["experts"]) for record in e4["records"]]
    primitives = [int(record["model_counts"]["primitives"]) for record in e4["records"]]
    summary = {
        "schema_version": 1,
        "study": config["study"],
        "principal_seeds": e4["seeds"],
        "classification": classification,
        "ood": {
            "selected_score": "maximum_probability",
            "near": {
                "mean_auroc": _mean([float(item["auroc"]) for item in near]),
                "mean_fpr95": _mean([float(item["fpr95"]) for item in near]),
            },
            "far": {
                "mean_auroc": _mean([float(item["auroc"]) for item in far]),
                "mean_fpr95": _mean([float(item["fpr95"]) for item in far]),
            },
        },
        "transfer": {
            "geode_balanced_accuracy": e6["variants"]["geode_head"]["target_metrics"]["balanced_accuracy"],
            "linear_probe_balanced_accuracy": e6["variants"]["linear_probe"]["target_metrics"]["balanced_accuracy"],
            "supervised_adapter_balanced_accuracy": e6["variants"]["supervised_adapter"]["target_metrics"]["balanced_accuracy"],
            "source_forgetting": e6["source_replay"]["forgetting"],
        },
        "cross_modal": {
            "text_top_1_accuracy": e8["text"]["accuracy"],
            "text_linear_top_1_accuracy": e8["text"]["linear_context_accuracy"],
            "point_geode_mean_abs_sdf": e8["pointcloud"]["geode_test_mean_abs_sdf"],
            "point_sphere_mean_abs_sdf": e8["pointcloud"]["single_sphere_test_mean_abs_sdf"],
        },
        "recovery": {
            "bad_canary_seconds": e10["bad_canary"]["recovery_seconds"],
            "coordinator_loss_seconds": e10["coordinator_loss"]["recovery_seconds"],
            "rto_seconds": e10["recovery_objectives"]["rto_seconds"],
            "rpo_requests": e10["recovery_objectives"]["rpo_requests"],
        },
        "cost": {
            "e4_total_seconds": sum(elapsed),
            "e4_mean_seconds_per_seed": _mean(elapsed),
            "e4_mean_experts": _mean(experts),
            "e4_mean_primitives": _mean(primitives),
        },
        "negative_results": [
            {
                "milestone": "E4",
                "result": "GEODE passed non-inferiority but trailed logistic and RBF controls.",
            },
            {
                "milestone": "E5",
                "result": f"No router promoted ({e5['summary']['promotion_eligible_count']} eligible of {e5['summary']['candidate_count']}).",
            },
            {
                "milestone": "E7",
                "result": f"Blocked: {', '.join(e7['blockers'])}.",
            },
            {
                "milestone": "E8",
                "result": "Text GEODE trailed matched linear and n-gram controls.",
            },
        ],
        "gates": {
            "e4": e4["core_gates_passed"],
            "e5": e5["summary"]["gate_passed"],
            "e6": e6["core_gates_passed"],
            "e7_expected_blocked": e7["qualification_status"] == "blocked",
            "e8": e8["gate_passed"],
            "e9": e9["gate_passed"],
            "e10": e10["gate_passed"],
        },
    }
    if not all(summary["gates"].values()):
        raise ValueError("one or more E11 source-artifact gates failed")
    return summary


def _principal_markdown(summary: dict[str, Any]) -> str:
    rows = [
        "# E11 Principal Results",
        "",
        "All values are reproduced from locked artifacts without training.",
        "",
        "## Five-seed CIFAR-100 classification",
        "",
        "| Method | Mean balanced accuracy | Standard deviation |",
        "| --- | ---: | ---: |",
    ]
    for record in summary["classification"]:
        rows.append(
            f"| {record['method']} | {record['mean_balanced_accuracy']:.4f} | "
            f"{record['stdev_balanced_accuracy']:.4f} |"
        )
    rows.extend([
        "",
        "## OOD and transfer",
        "",
        "| Endpoint | Value |",
        "| --- | ---: |",
        f"| Near-OOD AUROC | {summary['ood']['near']['mean_auroc']:.4f} |",
        f"| Near-OOD FPR95 | {summary['ood']['near']['mean_fpr95']:.4f} |",
        f"| Far-OOD AUROC | {summary['ood']['far']['mean_auroc']:.4f} |",
        f"| Far-OOD FPR95 | {summary['ood']['far']['mean_fpr95']:.4f} |",
        f"| Transfer GEODE balanced accuracy | {summary['transfer']['geode_balanced_accuracy']:.4f} |",
        f"| Transfer source forgetting | {summary['transfer']['source_forgetting']:.4f} |",
        "",
        "## Negative results and blocked work",
        "",
    ])
    rows.extend(
        f"- {item['milestone']}: {item['result']}"
        for item in summary["negative_results"]
    )
    return "\n".join(rows) + "\n"


def _cost_markdown(summary: dict[str, Any]) -> str:
    cost = summary["cost"]
    recovery = summary["recovery"]
    return "\n".join([
        "# E11 Cost and Recovery Report",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| E4 five-seed total wall time | {cost['e4_total_seconds']:.3f} s |",
        f"| E4 mean wall time per seed | {cost['e4_mean_seconds_per_seed']:.3f} s |",
        f"| E4 mean experts | {cost['e4_mean_experts']:.1f} |",
        f"| E4 mean primitives | {cost['e4_mean_primitives']:.1f} |",
        f"| Bad-canary recovery | {recovery['bad_canary_seconds']:.6f} s |",
        f"| Coordinator-loss recovery | {recovery['coordinator_loss_seconds']:.6f} s |",
        f"| Recovery time objective | {recovery['rto_seconds']:.3f} s |",
        f"| Recovery point objective | {recovery['rpo_requests']} requests |",
        "",
        "Wall times describe the recorded local hosts and are not normalized hardware benchmarks.",
        "",
    ])


def _bar_chart(title: str, labels: list[str], values: list[float], maximum: float) -> str:
    width, height = 720, 90 + 54 * len(values)
    plot_left, plot_width = 210, 460
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="sans-serif" font-size="20" fill="#17212b">{title}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = 62 + index * 54
        bar_width = plot_width * value / maximum
        elements.extend([
            f'<text x="24" y="{y + 22}" font-family="sans-serif" font-size="14" fill="#17212b">{label}</text>',
            f'<rect x="{plot_left}" y="{y}" width="{bar_width:.3f}" height="28" fill="#267365"/>',
            f'<text x="{plot_left + bar_width + 8:.3f}" y="{y + 20}" font-family="sans-serif" font-size="13" fill="#17212b">{value:.4f}</text>',
        ])
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def reproduce_public_study(
    config_path: Path,
    output_directory: Path,
    *,
    repository: Path = Path("."),
    lock_path: Path | None = None,
    refresh_lock: bool = False,
) -> dict[str, Any]:
    repository = repository.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported E11 configuration schema")
    index = _artifact_index(config, repository)
    if lock_path is not None:
        if refresh_lock:
            _write_json(lock_path, index)
        elif _read_json(lock_path) != index:
            raise ValueError("E11 artifact lock does not match repository artifacts")
    summary = _study_summary(repository, config)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_json(output_directory / "principal_results.json", summary)
    (output_directory / "principal_results.md").write_text(
        _principal_markdown(summary), encoding="utf-8", newline="\n",
    )
    (output_directory / "cost_report.md").write_text(
        _cost_markdown(summary), encoding="utf-8", newline="\n",
    )
    classification = summary["classification"]
    (output_directory / "classification.svg").write_text(
        _bar_chart(
            "Five-seed balanced accuracy",
            [item["method"] for item in classification],
            [item["mean_balanced_accuracy"] for item in classification],
            1.0,
        ),
        encoding="utf-8",
        newline="\n",
    )
    recovery = summary["recovery"]
    (output_directory / "recovery.svg").write_text(
        _bar_chart(
            "Recovery time (seconds)",
            ["bad canary", "coordinator loss"],
            [recovery["bad_canary_seconds"], recovery["coordinator_loss_seconds"]],
            recovery["rto_seconds"],
        ),
        encoding="utf-8",
        newline="\n",
    )
    return {path.name: _sha256(path) for path in sorted(output_directory.iterdir())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("experiments/configs/e11_public_study.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/results/e11_public_study"),
    )
    parser.add_argument(
        "--lock", type=Path,
        default=Path("logs/results/e11_artifact_index.json"),
    )
    parser.add_argument("--refresh-lock", action="store_true")
    arguments = parser.parse_args()
    hashes = reproduce_public_study(
        arguments.config,
        arguments.output,
        lock_path=arguments.lock,
        refresh_lock=arguments.refresh_lock,
    )
    print(json.dumps(hashes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()