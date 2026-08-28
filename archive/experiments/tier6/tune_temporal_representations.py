import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.tier6.eval_temporal_text_prediction import (
    forward_chaining_splits,
    linear_context_accuracy,
    prepare_text_corpus,
    sample_context_pairs,
    sample_ensemble_state_pairs,
    sample_hybrid_state_pairs,
    sample_temporal_state_pairs,
)


def _sample_representation(
    train_ids: np.ndarray,
    candidate: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    representation = candidate["representation"]
    common = {"lag": 1, "max_samples": None, "seed": seed}
    if representation == "window":
        return sample_context_pairs(
            train_ids, window=candidate["window"], **common,
        )
    if representation == "temporal_state":
        return sample_temporal_state_pairs(
            train_ids,
            state_dim=candidate["state_dim"],
            encoder_seed=seed,
            warmup=candidate.get("warmup", 32),
            recurrence=candidate.get("recurrence", 0.8),
            **common,
        )
    if representation in {"multi_timescale", "multi_seed"}:
        return sample_ensemble_state_pairs(
            train_ids,
            state_dim=candidate["state_dim"],
            variant=representation,
            encoder_seed=seed,
            warmup=candidate.get("warmup", 32),
            recurrence=candidate.get("recurrence", 0.8),
            recurrences=tuple(candidate.get("recurrences", (0.3, 0.7, 0.95))),
            member_count=candidate.get("member_count", 3),
            **common,
        )
    if representation == "hybrid":
        return sample_hybrid_state_pairs(
            train_ids,
            window=candidate["window"],
            state_dim=candidate["state_dim"],
            encoder_seed=seed,
            warmup=candidate.get("warmup", 32),
            recurrence=candidate.get("recurrence", 0.8),
            **common,
        )
    raise ValueError(f"Unsupported representation: {representation}")


def run_tuning(config: dict) -> dict:
    train_ids, _ = prepare_text_corpus(
        dataset=config["dataset"],
        max_chars=config["max_chars"],
        seed=config["seed"],
    )
    if config.get("max_tuning_chars") is not None:
        train_ids = train_ids[:config["max_tuning_chars"]]
    results = []
    for candidate in config["candidates"]:
        features, targets = _sample_representation(
            train_ids, candidate, config["seed"],
        )
        fold_scores = []
        for train_indices, validation_indices in forward_chaining_splits(
            len(features), config["n_folds"], gap=candidate.get("window", 1),
        ):
            fold_scores.append(linear_context_accuracy(
                features[train_indices],
                targets[train_indices],
                features[validation_indices],
                targets[validation_indices],
                seed=config["seed"],
                max_iter=config.get("linear_max_iter", 30),
            ))
        results.append({
            "name": candidate["name"],
            "candidate": candidate,
            "fold_accuracies": fold_scores,
            "mean_forward_accuracy": float(np.mean(fold_scores)),
            "std_forward_accuracy": float(np.std(fold_scores)),
            "feature_dimension": int(features.shape[1]),
        })
    selected = max(results, key=lambda result: result["mean_forward_accuracy"])
    return {
        "config": config,
        "results": results,
        "selected": selected,
        "selection_metric": "train_only_linear_forward_validation_accuracy",
        "test_sequence_used": False,
        "tuning_character_count": len(train_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune causal temporal representations.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_tuning(config)
    output = Path(config["artifact_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {result['selected']['name']}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()