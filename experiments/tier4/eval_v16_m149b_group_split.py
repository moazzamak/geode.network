"""M149b — group splitting against the REAL specialist incumbent (train-score).

Registered after M149's sealed negative (14 Aug 2026): M149's held-out
incumbent re-fit (1400 rows) weakened the baseline. This cell uses the M143b
train-score cache: split children are fit on the group's TRAIN rows, the
fusion is fit on train rows (penalty ladder on a train-valid slice), and the
incumbent is the real cached specialist's predictions on the sealed test rows.

Per group (domain): proposal = seeded 2-means over the domain's 2415-dim
train score vectors -> children X, Y; child ridges on their cluster train
rows; fusion = stacking over [X_scores, Y_scores] (ladder selected on a
seeded 80/20 slice of the group's train rows); evaluated on the group's TEST
rows vs the incumbent specialist and a random-split control.

Gate: a domain PASSES iff fused(2-means) >= incumbent + 0.005 AND
fused(2-means) >= fused(random) on the group's test rows. M149b fires iff
ZERO domains pass. Row floor: groups with < 2 x floor train rows in a
cluster are SKIPPED, not failed.

Reproduce with::

    .\\.venv\\Scripts\\python.exe -m experiments.tier4.eval_v16_m149b_group_split
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.data_cache import data_cache_root
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v16_m143_integration import (
    CLASSES,
    DOMAINS,
    _select_penalty,
    _stacking_fit,
)
from experiments.tier4.eval_v16_m149_group_split import _fit_child, _kmeans_2
from experiments.tier4.eval_v16_m109_trunk import _load_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m149b_group_split.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m149b_group_split"


def run_m149b(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    corpus, _train_index, _test_index = _load_corpus(config)
    train_domains = corpus["train_domains"]
    test_domains = corpus["test_domains"]
    del _train_index, _test_index

    train_payload = np.load(
        data_cache_root() / config["train_cache"]["cache_relpath"]
        / "train_scores.npz", allow_pickle=False)
    specialist_train = train_payload["specialist_train"]  # (6, n_train, 345)
    global_train = train_payload["global_train"]          # (n_train, 345)
    train_labels = train_payload["train_labels"]
    n_train = len(train_labels)

    test_payload = np.load(
        data_cache_root() / config["test_cache"]["cache_relpath"]
        / "scores.npz", allow_pickle=False)
    specialist_test = test_payload["specialist_scores"]
    test_labels = test_payload["test_labels"]
    n_test = len(test_labels)

    def concat_rows(spec: np.ndarray, glob: np.ndarray, n: int) -> np.ndarray:
        return np.concatenate(
            [spec.reshape(DOMAINS, n, CLASSES).transpose(1, 0, 2)
                .reshape(n, -1), glob], axis=1)

    train_concat = concat_rows(specialist_train, global_train, n_train)
    test_concat = concat_rows(specialist_test, test_payload["global_scores"],
                              n_test)

    margin = float(config["gate"]["margin"])
    floor = int(config["gate"]["row_floor"])
    ladder = [float(x) for x in config["phase2"]["penalty_ladder"]]
    kmeans_seed = int(config["phase2"]["kmeans_seed"])
    random_seed = int(config["phase2"]["random_seed"])
    valid_frac = float(config["phase2"]["valid_frac"])
    valid_seed = int(config["phase2"]["valid_seed"])

    domain_report: dict[str, Any] = {}
    for domain in range(DOMAINS):
        tr = np.where(train_domains == domain)[0]
        te = np.where(test_domains == domain)[0]
        print(f"domain {domain}: {len(tr)} train / {len(te)} test rows",
              flush=True)
        if len(tr) < 4 * floor:
            domain_report[str(domain)] = {"skipped": True,
                                          "reason": "train row floor",
                                          "train_rows": int(len(tr))}
            continue

        cluster_labels = _kmeans_2(train_concat[tr], kmeans_seed + domain)
        if min(int((cluster_labels == 0).sum()),
               int((cluster_labels == 1).sum())) < floor:
            domain_report[str(domain)] = {"skipped": True,
                                          "reason": "degenerate cluster",
                                          "train_rows": int(len(tr))}
            continue

        def _children_features(children, rows):
            feat = np.zeros((len(rows), 2 * CLASSES), dtype=np.float64)
            feat[:, :CLASSES] = children[0](train_concat[rows])
            feat[:, CLASSES:] = children[1](train_concat[rows])
            return feat

        # 2-means children
        pred_0 = _fit_child(tr[cluster_labels == 0], train_concat, train_labels)
        pred_1 = _fit_child(tr[cluster_labels == 1], train_concat, train_labels)
        # random-split children (control)
        rng_r = np.random.default_rng(random_seed + domain)
        rand_labels = rng_r.integers(0, 2, size=len(tr))
        pred_r0 = _fit_child(tr[rand_labels == 0], train_concat, train_labels)
        pred_r1 = _fit_child(tr[rand_labels == 1], train_concat, train_labels)

        # penalty ladder on a seeded 80/20 slice of the group's train rows
        rng = np.random.default_rng(valid_seed + domain)
        order = rng.permutation(len(tr))
        cut = int(valid_frac * len(tr))
        ft = tr[order[:cut]]
        fv = tr[order[cut:]]

        def metric(penalty, children=(pred_0, pred_1)):
            predict = _stacking_fit(_children_features(children, ft),
                                    train_labels[ft], penalty)
            return float((predict(_children_features(children, fv))
                          == train_labels[fv]).mean())

        def metric_r(penalty, children=(pred_r0, pred_r1)):
            predict = _stacking_fit(_children_features(children, ft),
                                    train_labels[ft], penalty)
            return float((predict(_children_features(children, fv))
                          == train_labels[fv]).mean())

        best_penalty, ladder_scores = _select_penalty(metric, ladder)
        best_penalty_r, ladder_scores_r = _select_penalty(metric_r, ladder)

        stacking = _stacking_fit(_children_features((pred_0, pred_1), tr),
                                 train_labels[tr], best_penalty)
        stacking_r = _stacking_fit(_children_features((pred_r0, pred_r1), tr),
                                   train_labels[tr], best_penalty_r)

        # evaluate on the group's TEST rows
        test_feat = np.zeros((len(te), 2 * CLASSES), dtype=np.float64)
        test_feat[:, :CLASSES] = pred_0(test_concat[te])
        test_feat[:, CLASSES:] = pred_1(test_concat[te])
        fused_acc = float((stacking(test_feat) == test_labels[te]).mean())
        test_feat_r = np.zeros((len(te), 2 * CLASSES), dtype=np.float64)
        test_feat_r[:, :CLASSES] = pred_r0(test_concat[te])
        test_feat_r[:, CLASSES:] = pred_r1(test_concat[te])
        fused_r_acc = float((stacking_r(test_feat_r)
                             == test_labels[te]).mean())
        # incumbent = the REAL cached specialist on its own domain's test rows
        inc_acc = float((np.argmax(specialist_test[domain, te], axis=1)
                         == test_labels[te]).mean())

        passed = (fused_acc >= inc_acc + margin) and (fused_acc >= fused_r_acc)
        domain_report[str(domain)] = {
            "train_rows": int(len(tr)),
            "test_rows": int(len(te)),
            "incumbent_specialist_accuracy": inc_acc,
            "fused_2means_accuracy": fused_acc,
            "fused_random_accuracy": fused_r_acc,
            "selected_penalty": best_penalty,
            "ladder_scores": ladder_scores,
            "passed": passed,
        }
        print(f"  incumbent {inc_acc:.4f}; fused(2means) {fused_acc:.4f}; "
              f"fused(random) {fused_r_acc:.4f} -> passed={passed}", flush=True)

    passing = [d for d, r in domain_report.items() if r.get("passed")]
    fired = len(passing) == 0 and not inadmissible
    evidence: dict[str, Any] = {
        "milestone": "M149b",
        "admissible_as_evidence": not inadmissible,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": ("against the real cached specialist incumbent and with "
                     "children fit on the group's train rows, does the "
                     "transactional split recover structure without losing "
                     "accuracy?"),
        "domain_report": domain_report,
        "passing_domains": passing,
        "gate": {
            "registered": config["gate"]["registered"],
            "fired": fired,
            "consequence": (config["gate"]["consequence_fired"] if fired
                            else config["gate"]["consequence_passed"]),
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM149b complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m149b(args.config, args.output)


if __name__ == "__main__":
    main()
