"""M139a — the routing slack: a 6-way linear domain router on the sealed codes.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v21.md`` (M139 amendment,
13 Aug 2026) and ``experiments/configs/v16/m139a_routing_slack.json``.

Question. The specialist buy-back (M139b) routes by domain. How much routing
slack does the sealed code carry: can a 6-way linear domain router fit on the
frozen 6144-atom codes identify the domain of a test row, and with what
confidence? M139a measures that operand. It has NO kill switch — it is a
prerequisite diagnostic; the buy-back gate lives in M139b.

Method. ONE streaming pass over the sealed f6144 train memmap accumulates the
Gram and BOTH cross-product targets — the 345-class one-hot (for the t1 anchor)
and the 6-domain one-hot (the router). Both heads are closed-form ridges at
lambda = 1.0 with intercept, the exact RidgeAccumulator standardisation.

t1 anchor: the class head on the same codes must reproduce M117's sealed
Q(6144, 138000) = 0.2248695652173913 within 0.002 — validating the
codes-to-labels pairing without any encode.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m139a_routing_slack
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v16_m109_trunk import _load_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m139a_routing_slack.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m139a_routing_slack"

T1_TOLERANCE = 0.002
T1_REFERENCE = 0.2248695652173913
CLASSES = 345
DOMAINS = 6


class DualAccumulator:
    """One-pass normal equations for TWO target sets sharing one Gram.

    The expensive part of the ridge fit is the Gram accumulation; the class head
    (t1 anchor) and the domain router share exactly the same X'X. One pass
    produces both systems with the RidgeAccumulator's standardisation recovered
    from the raw Gram (same arithmetic, same convention).
    """

    def __init__(self, width: int, class_count: int, domain_count: int) -> None:
        self.width = width
        self.classes = class_count
        self.domains = domain_count
        self.gram = np.zeros((width, width), dtype=np.float64)
        self.column_sum = np.zeros(width, dtype=np.float64)
        self.cross_class = np.zeros((width, class_count), dtype=np.float64)
        self.cross_domain = np.zeros((width, domain_count), dtype=np.float64)
        self.class_count = np.zeros(class_count, dtype=np.float64)
        self.domain_count = np.zeros(domain_count, dtype=np.float64)
        self.rows = 0

    def add(self, features: np.ndarray, labels: np.ndarray,
            domains: np.ndarray) -> None:
        block = np.asarray(features, dtype=np.float64)
        n = len(block)
        y = np.zeros((n, self.classes), dtype=np.float64)
        y[np.arange(n), labels] = 1.0
        d = np.zeros((n, self.domains), dtype=np.float64)
        d[np.arange(n), domains] = 1.0
        self.gram += block.T @ block
        self.column_sum += block.sum(axis=0)
        self.cross_class += block.T @ y
        self.cross_domain += block.T @ d
        self.class_count += y.sum(axis=0)
        self.domain_count += d.sum(axis=0)
        self.rows += n

    def _standardiser(self) -> tuple[np.ndarray, np.ndarray]:
        centre = self.column_sum / self.rows
        variance = np.diag(self.gram) / self.rows - np.square(centre)
        scale = np.sqrt(np.maximum(variance, 0.0)) + 1e-8
        return centre.astype(np.float32), scale.astype(np.float32)

    def solve(self, penalty: float, cross: np.ndarray,
              counts: np.ndarray) -> np.ndarray:
        """Closed-form weights for one target set (classes or domains)."""
        centre64 = (self.column_sum / self.rows).astype(np.float64)
        scale64 = (np.sqrt(np.maximum(np.diag(self.gram) / self.rows
                                       - np.square(centre64), 0.0))
                   + 1e-8)
        inverse = 1.0 / scale64
        centred = (self.gram - np.outer(self.column_sum, centre64))
        centred *= inverse[:, None]
        centred *= inverse[None, :]
        cross_s = (cross - np.outer(centre64, counts)) * inverse[:, None]
        centred.flat[:: self.width + 1] += penalty
        weights = np.linalg.solve(centred, cross_s)
        intercept = counts / self.rows
        return np.vstack([weights, intercept[None, :]])


def _score(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    scores = features @ weights[:-1] + weights[-1]
    return np.argmax(scores, axis=1).astype(np.int64)


def _blocks(mem_test: np.ndarray, block: int) -> Iterator[np.ndarray]:
    for start in range(0, len(mem_test), block):
        yield np.asarray(mem_test[start:start + block])


def run_m139a(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if "_smoke_note" in config and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    configure_external_cache_environment()
    block = int(config["numerics"]["block"])
    smoke = bool(config.get("_smoke_skip_gates", False))
    started = time.time()

    print("loading corpus labels (subsample digest verified by _load_corpus)", flush=True)
    corpus, _, _ = _load_corpus(config)
    train_labels = corpus["train_labels"]
    test_labels = corpus["test_labels"]
    test_domains = corpus["test_domains"]

    print("opening sealed 6144-atom code memmaps", flush=True)
    codes_dir = data_cache_root() / config["sealed_codes"]["cache_relpath"]
    train_path = codes_dir / config["sealed_codes"]["train_file"]
    test_path = codes_dir / config["sealed_codes"]["test_file"]
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(f"sealed code memmaps missing under {codes_dir}")
    mem_train = np.load(train_path, mmap_mode="r")
    mem_test = np.load(test_path, mmap_mode="r")
    width = int(config["sealed_codes"]["width"])
    if mem_train.shape != (len(train_labels), width):
        raise SystemExit(f"train code shape {mem_train.shape} != expected")

    n = int(config["domain_head"]["n"])
    penalty = float(config["domain_head"]["penalty"])
    print(f"dual accumulate over n={n} rows (classes + domains, one Gram)", flush=True)
    acc = DualAccumulator(width, CLASSES, DOMAINS)
    for start in range(0, n, block):
        stop = min(start + block, n)
        acc.add(mem_train[start:stop], train_labels[start:stop],
                corpus["train_domains"][start:stop])

    centre, scale = acc._standardiser()
    w_class = acc.solve(penalty, acc.cross_class, acc.class_count)
    w_domain = acc.solve(penalty, acc.cross_domain, acc.domain_count)

    # ---- scoring -----------------------------------------------------------
    class_hits = 0
    domain_hits = 0
    confusion = np.zeros((DOMAINS, DOMAINS), dtype=np.int64)
    domain_counts = np.zeros(DOMAINS, dtype=np.int64)
    domain_confidence: list[float] = []
    seen = 0
    for block_rows in _blocks(mem_test, block):
        standardised = (block_rows - centre) / scale
        class_pred = _score(w_class, standardised)
        domain_pred = _score(w_domain, standardised)
        class_hits += int((class_pred == test_labels[seen:seen + len(block_rows)]).sum())
        domain_hits += int((domain_pred == test_domains[seen:seen + len(block_rows)]).sum())
        for d_true, d_pred in zip(test_domains[seen:seen + len(block_rows)],
                                  domain_pred):
            confusion[int(d_true), int(d_pred)] += 1
            domain_counts[int(d_true)] += 1
        scores = standardised @ w_domain[:-1] + w_domain[-1]
        ranked = np.sort(scores, axis=1)
        domain_confidence.extend((ranked[:, -1] - ranked[:, -2]).tolist())
        seen += len(block_rows)

    class_accuracy = class_hits / seen
    domain_accuracy = domain_hits / seen
    per_domain = (np.diag(confusion).astype(np.float64)
                  / np.maximum(domain_counts, 1)).tolist()
    evidence: dict[str, Any] = {
        "milestone": "M139a",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "codes": {"dir": str(codes_dir), "width": width},
        "domain_head": {"n": n, "penalty": penalty, "classes": DOMAINS},
        "class_head": {
            "accuracy": class_accuracy,
            "t1_reference": T1_REFERENCE,
            "t1_delta": class_accuracy - T1_REFERENCE,
            "t1_tolerance": T1_TOLERANCE,
        },
        "domain_router": {
            "accuracy": domain_accuracy,
            "per_domain_accuracy": per_domain,
            "confusion": confusion.tolist(),
            "confidence_margin": {
                "q25": float(np.quantile(domain_confidence, 0.25)),
                "q50": float(np.quantile(domain_confidence, 0.50)),
                "q75": float(np.quantile(domain_confidence, 0.75)),
                "q95": float(np.quantile(domain_confidence, 0.95)),
                "mean": float(np.mean(domain_confidence)),
            },
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    if not smoke:
        if abs(evidence["class_head"]["t1_delta"]) > T1_TOLERANCE:
            evidence["void"] = True
            evidence["void_reason"] = "t1 anchor reproduction failed"
        else:
            print(f"  t1 anchor delta {evidence['class_head']['t1_delta']:+.6f} "
                  f"(<= {T1_TOLERANCE})", flush=True)
    print(f"  domain router accuracy {domain_accuracy:.4f} "
          f"per-domain {[round(p, 4) for p in per_domain]}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"wrote {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    run_m139a(Path(args.config), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
