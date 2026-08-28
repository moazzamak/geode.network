import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.tier6.eval_temporal_text_prediction import (
    run_text_prediction_experiment,
)


def run_ablation(config: dict) -> dict:
    results = {}
    excluded = {"artifact_path", "representations"}
    common = {key: value for key, value in config.items() if key not in excluded}
    for representation in config["representations"]:
        print(f"\n=== Representation: {representation} ===")
        results[representation] = run_text_prediction_experiment(
            representation=representation,
            **common,
        )
    selected = max(results, key=lambda name: results[name]["cv_acc_mean"])
    return {
        "config": config,
        "results": results,
        "selection_metric": "forward_validation_accuracy",
        "selected_representation": selected,
        "test_used_for_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare causal Tier 6 representations.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_ablation(config)
    output = Path(config["artifact_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()