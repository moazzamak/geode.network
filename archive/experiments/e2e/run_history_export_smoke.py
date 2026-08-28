import argparse
import json
import tempfile
from pathlib import Path

from src.runtime import MetricEvent, MetricLedger, export_metric_history


def run_qualification(output_directory: str | Path) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        ledger = MetricLedger(Path(directory) / "metrics.jsonl")
        events = (
            MetricEvent(
                event_id="train-loss-1",
                run_id="e2-history-export-smoke",
                attempt_id="attempt-1",
                stage_name="representation",
                split="train",
                metric_name="loss",
                value=0.8,
                sample_count=128,
                created_at="2026-07-26T00:00:00Z",
                epoch=1,
                global_step=10,
                namespace="exploratory",
            ),
            MetricEvent(
                event_id="train-loss-2",
                run_id="e2-history-export-smoke",
                attempt_id="attempt-2",
                stage_name="representation",
                split="train",
                metric_name="loss",
                value=0.6,
                sample_count=128,
                created_at="2026-07-26T00:01:00Z",
                epoch=2,
                global_step=20,
                namespace="exploratory",
            ),
            MetricEvent(
                event_id="validation-loss-2",
                run_id="e2-history-export-smoke",
                attempt_id="attempt-2",
                stage_name="representation",
                split="validation",
                metric_name="loss",
                value=0.65,
                sample_count=64,
                created_at="2026-07-26T00:02:00Z",
                epoch=2,
                global_step=20,
                namespace="selection",
            ),
        )
        for event in events:
            ledger.append(event)
        first_hashes = export_metric_history(ledger, output_directory)
        second_hashes = export_metric_history(ledger, output_directory)

    if first_hashes != second_hashes:
        raise AssertionError("idempotent history export changed artifact hashes")
    history = json.loads(
        (Path(output_directory) / "history.json").read_text(encoding="utf-8")
    )
    expected_files = {"dashboard.html", "events.csv", "history.json"}
    if set(first_hashes) != expected_files or history["event_count"] != len(events):
        raise AssertionError("history export is incomplete")
    return {
        "schema_version": 1,
        "run_id": "e2-history-export-smoke",
        "source_of_truth": "append-only MetricLedger JSONL",
        "event_count": history["event_count"],
        "attempt_count": len(history["attempt_ids"]),
        "series_count": len(history["series"]),
        "namespace_counts": history["namespace_counts"],
        "artifact_sha256": first_hashes,
        "idempotent_reexport": True,
        "standalone_dashboard": True,
        "external_service_required": False,
        "performance_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify deterministic local metric history exports.",
    )
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    result = run_qualification(args.output_directory)
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(
        f"Exported {result['event_count']} events into "
        f"{len(result['artifact_sha256'])} deterministic artifacts."
    )


if __name__ == "__main__":
    main()