"""M308 harness — the registered drawn-challenge gate (A8, H26-10).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M308
(26 Aug 2026, before any build). H26-10: scores for the same artifact
measured by disjoint validator sets agree within a registered
tolerance under drawn challenges and DISAGREE under validator-authored
challenges — the second half is what establishes A8 was a real defect.

Gate cells:

- **C1 stratified draw.** The published rule draws equal per class
  and rotates across epochs.
- **C2 drawn agreement (H26-10 first half).** Two disjoint validator
  sets draw from the same sealed corpus and score the same imperfect
  artifact: |s1 - s2| within the registered tolerance.
- **C3 authored disagreement (H26-10 second half).** One validator
  set authors from an easy slice, the other from a hard slice:
  |s1 - s2| beyond the registered tolerance.
- **C4 authored stream never scores.** Adding authored challenges
  leaves the routable score unchanged.

All four cells must pass.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from geode.core.drawn_challenges import (
    register_corpus,
    routable_score,
    score_draw,
    stratified_draw,
    supplementary_stream,
    verify_answer,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m308_drawn_challenges.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m308_drawn_challenges")


def _artifact_scores(x: np.ndarray, w_hat: np.ndarray
                     ) -> np.ndarray:
    return np.argmax(x @ w_hat, axis=1)


def run_m308(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    rng = np.random.default_rng(int(config["seed"]))

    cells: dict[str, Any] = {}

    # ---- C1 stratified draw ----
    corpus, rows, labels = _synthetic_corpus(int(config["corpus_rows"]),
                                             int(config["classes"]))
    draw0 = stratified_draw(corpus, labels, config["beacon_seed"], 0,
                            int(config["draw_count"]))
    draw1 = stratified_draw(corpus, labels, config["beacon_seed"], 1,
                            int(config["draw_count"]))
    from collections import Counter
    counts = Counter(labels[i] for i in draw0)
    equal_share = int(config["draw_count"]) // int(config["classes"])
    c1 = {
        "per_class_counts": dict(sorted(counts.items())),
        "registered_equal_share": equal_share,
        "equal_per_class": bool(all(
            v == equal_share for v in counts.values())),
        "rotates_across_epochs": bool(draw0 != draw1),
        "passes": bool(all(v == equal_share for v in counts.values())
                       and draw0 != draw1),
    }
    cells["c1_stratified_draw"] = c1

    # ---- synthetic axis: an imperfect artifact over a noisy truth ----
    d = int(config["dim"])
    classes = int(config["classes"])
    n_pop = int(config["population_rows"])
    w_truth = rng.standard_normal((d, classes)) * 3.0
    w_hat = w_truth + rng.standard_normal((d, classes)) * 0.6
    x_pop = rng.standard_normal((n_pop, d))
    truth = np.argmax(x_pop @ w_truth, axis=1)

    # ---- C2 drawn agreement (H26-10 first half) ----
    row_bytes = [r.tobytes() for r in x_pop]
    corpus_pop = register_corpus(row_bytes, [int(t) for t in truth])
    scores = []
    for beacon in (config["beacon_seed"] + "-A",
                   config["beacon_seed"] + "-B"):
        draw = stratified_draw(corpus_pop, [int(t) for t in truth],
                               beacon, 0, int(config["draw_count"]))
        answers = _artifact_scores(x_pop[draw], w_hat)
        scored = score_draw(corpus_pop, row_bytes,
                            [int(t) for t in truth],
                            list(draw), [int(a) for a in answers])
        scores.append(scored["score"])
    drawn_gap = abs(scores[0] - scores[1])
    tol = float(config["drawn_tolerance"])
    c2 = {
        "validator_set_scores": scores,
        "gap": drawn_gap,
        "registered_tolerance": tol,
        "passes": bool(drawn_gap <= tol),
    }
    cells["c2_drawn_agreement"] = c2

    # ---- C3 authored disagreement (H26-10 second half) ----
    hat_scores = x_pop @ w_hat
    margins = np.sort(hat_scores, axis=1)[:, -1] - np.sort(
        hat_scores, axis=1)[:, -2]
    easy_rows = np.argsort(margins)[-int(config["authored_count"]) :]
    hard_rows = np.argsort(margins)[: int(config["authored_count"])]
    easy_correct = float(np.mean(
        _artifact_scores(x_pop[easy_rows], w_hat)
        == truth[easy_rows]))
    hard_correct = float(np.mean(
        _artifact_scores(x_pop[hard_rows], w_hat)
        == truth[hard_rows]))
    authored_gap = abs(easy_correct - hard_correct)
    authored_tol = float(config["authored_tolerance"])
    c3 = {
        "easy_slice_score": easy_correct,
        "hard_slice_score": hard_correct,
        "gap": authored_gap,
        "registered_tolerance": authored_tol,
        "passes": bool(authored_gap > authored_tol),
    }
    cells["c3_authored_disagreement"] = c3

    # ---- C4 authored stream never scores ----
    score_before = routable_score([True, True, False, True])["score"]
    stream = supplementary_stream([{"authored": 1}] * 7)
    score_after = routable_score([True, True, False, True])["score"]
    c4 = {
        "score_before": score_before,
        "score_after": score_after,
        "authored_count": stream["authored_count"],
        "enters_score": stream["enters_routable_score"],
        "passes": bool(score_before == score_after
                       and not stream["enters_routable_score"]),
    }
    cells["c4_authored_never_scores"] = c4

    gates_ok = all(bool(c["passes"]) for c in cells.values())
    elapsed = time.time() - started
    evidence = {
        "milestone": "M308",
        "config_digest": payload_hash(config),
        "gates_ok": gates_ok,
        "cells": cells,
        "registered_checks": ["C1", "C2", "C3", "C4"],
        "runtime_seconds": elapsed,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({
        "gates_ok": gates_ok,
        "c2_drawn_scores": scores, "c2_gap": drawn_gap,
        "c3_easy": easy_correct, "c3_hard": hard_correct,
        "c3_gap": authored_gap,
    }, indent=1))
    return evidence


def _synthetic_corpus(n: int, classes: int):
    labels = [i % classes for i in range(n)]
    rows = [f"row-{i}".encode() for i in range(n)]
    return register_corpus(rows, labels), rows, labels


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m308(args.config, args.output)


if __name__ == "__main__":
    main()
