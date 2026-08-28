"""M175 cell A0 — the FIRST text encoder: additive next-token fit-and-report
on wikitext-103 (the M131 machinery, new corpus, anchors + held-out).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` section 12
(18 Aug 2026) and ``analysis/v24_m175_transfer_cells.md``.

Question. Does the M131 additive next-token construction (exact-order
counts, uniform Jelinek-Mercer over orders 1..w, add-alpha smoothing)
build a frozen text arm on wikitext-103 character tokens — and does the
window dial behave as measured on the DSL (perplexity falls, footprint
grows, more memory does not hurt)?

Corpus: cached ``data/tier6/wikitext103_100000.npz`` (vocab_fingerprint
5c6f0917b424e3a7, cache_version 2): 80,000 train / 20,000 test
character-level token ids. No labels; fit-and-report is self-supervised.

Model: ``ProgrammaticMemory`` exact-order counts over the train ids;
per-token probability is the uniform mixture over orders 1..w of
add-alpha smoothed exact-order counts (alpha = 1.0). Test is read-only.

Anchors (registered before running):
- g1 bigram closed form: the window-1 cell's test perplexity must equal
  an INDEPENDENT closed-form bigram computation (C(a,b)+alpha over
  N(a)+alpha*V, from train counts) to <= 1e-9.
- g2 footprint law: footprint_bytes is monotone non-decreasing in window.

Gate (N90.5 adapted): test_ppl(w=max) <= test_ppl(w=1) * 1.10.
Uniform baseline: vocab size. This encoder is the frozen text arm reused
by cells A and D. No novelty claim (n-gram/PPM territory, M129).

Reproduce with::

    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v24_m175_cell_a0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    write_canonical_json,
)
from src.programmatic_memory import ProgrammaticMemory

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m175_cell_a0.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v24" / "m175_cell_a0"

G1_TOLERANCE = 1e-9


def _load_wikitext(config: dict[str, Any], smoke_tokens: int
                   ) -> tuple[list[int], list[int], str, int]:
    """Train/test token-id lists + the registered cache fingerprint."""
    cache = np.load(REPO_ROOT / config["corpus"]["cache_file"])
    train = [int(x) for x in cache["train_ids"]]
    test = [int(x) for x in cache["test_ids"]]
    if smoke_tokens:
        train = train[:smoke_tokens]
        test = test[:max(1, smoke_tokens // 4)]
    fingerprint = str(cache["vocab_fingerprint"])
    cache_version = int(cache["cache_version"])
    expected_fp = config["corpus"]["vocab_fingerprint"]
    expected_cv = int(config["corpus"]["cache_version"])
    if fingerprint != expected_fp or cache_version != expected_cv:
        raise SystemExit(
            f"M175 A0 VOID: wikitext cache pin failed "
            f"(fp {fingerprint} != {expected_fp}, cv {cache_version} "
            f"!= {expected_cv})")
    return train, test, fingerprint, cache_version


def _vocabulary(tokens: list[int]) -> list[int]:
    return sorted(set(tokens))


def _order_log_prob(memory: ProgrammaticMemory, context: list[int],
                    token: int, order: int, alpha: float,
                    vocab: list[int]) -> tuple[float, int]:
    result = memory.exact_continuations(context, order)
    counts = dict(result.counts)
    total = result.total()
    count = counts.get(token, 0)
    prob = (count + alpha) / (total + alpha * len(vocab))
    return math.log(prob), len(result.counts)


def _interpolated_log_prob(memory: ProgrammaticMemory, context: list[int],
                           token: int, window: int, alpha: float,
                           vocab: list[int]) -> float:
    total = 0.0
    for order in range(1, window + 1):
        log_p, _ = _order_log_prob(memory, context, token, order, alpha,
                                   vocab)
        total += math.exp(log_p)
    return math.log(total / window)


def _perplexity(memory: ProgrammaticMemory, tokens: list[int], window: int,
                alpha: float, vocab: list[int], ops: list[int]) -> float:
    nll_sum = 0.0
    n = 0
    for i in range(1, len(tokens)):
        context = tokens[max(0, i - memory.window):i]
        log_p = _interpolated_log_prob(memory, context, tokens[i], window,
                                       alpha, vocab)
        nll_sum -= log_p
        entries = 0
        for order in range(1, window + 1):
            entries += len(memory.exact_continuations(context, order).counts)
        ops.append(window + entries)
        n += 1
    return math.exp(nll_sum / n)


def _bigram_closed_form(train: list[int], test: list[int], alpha: float,
                        vocab: list[int]) -> float:
    """g1: window-1 = bigram, computed independently in closed form."""
    pair_counts: dict[tuple[int, int], int] = {}
    prefix_totals: dict[int, int] = {}
    for i in range(1, len(train)):
        pair = (train[i - 1], train[i])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        prefix_totals[train[i - 1]] = prefix_totals.get(train[i - 1], 0) + 1
    v = len(vocab)
    nll_sum = 0.0
    n = 0
    for i in range(1, len(test)):
        prev = test[i - 1]
        prob = (pair_counts.get((prev, test[i]), 0) + alpha) / (
            prefix_totals.get(prev, 0) + alpha * v)
        nll_sum -= math.log(prob)
        n += 1
    return math.exp(nll_sum / n)


def _backoff_perplexity(memory: ProgrammaticMemory, tokens: list[int],
                        alpha: float, vocab: list[int],
                        ops: list[int]) -> tuple[float, dict[int, int]]:
    """Longest-match backoff read (the machinery's ``continuations``
    primitive), add-alpha at the matched order; a novel context gets the
    uniform floor 1/V. Returns (ppl, matched_length_histogram)."""
    from collections import Counter

    histogram: Counter[int] = Counter()
    v = len(vocab)
    nll_sum = 0.0
    n = 0
    for i in range(1, len(tokens)):
        context = tokens[max(0, i - memory.window):i]
        result = memory.continuations(context)
        counts = dict(result.counts)
        total = result.total()
        prob = (counts.get(tokens[i], 0) + alpha) / (total + alpha * v)
        nll_sum -= math.log(prob)
        histogram[result.matched_length] += 1
        ops.append(result.matched_length + len(result.counts))
        n += 1
    return math.exp(nll_sum / n), dict(sorted(histogram.items()))


def run_m175_cell_a0(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible
    smoke_tokens = int(config.get("_smoke_tokens", 0))

    train, test, fingerprint, cache_version = _load_wikitext(
        config, smoke_tokens)
    vocab = _vocabulary(train)
    alpha = float(config["model"]["alpha"])
    ladder = [int(w) for w in config["model"]["window_ladder"]]
    max_window = int(config["model"]["max_window"])
    print(f"wikitext: train {len(train)} / test {len(test)} tokens, "
          f"vocab {len(vocab)} (fp {fingerprint}, cv {cache_version})",
          flush=True)

    cells: dict[str, Any] = {}
    cells_raw: dict[str, float] = {}
    backoff_cells: dict[str, Any] = {}
    for window in ladder:
        memory = ProgrammaticMemory(window=window)
        memory.register(train)
        ops_test: list[int] = []
        test_ppl = _perplexity(memory, test, window, alpha, vocab, ops_test)
        cells_raw[str(window)] = test_ppl
        cells[str(window)] = {
            "window": window,
            "test_perplexity": round(test_ppl, 4),
            "footprint_bytes": memory.footprint_bytes(),
            "context_count": memory.context_count,
            "entry_count": memory.entry_count,
            "ops_per_token_test_mean": round(float(np.mean(ops_test)), 2),
        }
        ops_back: list[int] = []
        back_ppl, histogram = _backoff_perplexity(
            memory, test, alpha, vocab, ops_back)
        backoff_cells[str(window)] = {
            "window": window,
            "test_perplexity": round(back_ppl, 4),
            "matched_length_histogram": histogram,
            "ops_per_token_test_mean": round(float(np.mean(ops_back)), 2),
        }
        print(f"  window {window}: uniform {test_ppl:.4f} backoff {back_ppl:.4f}"
              f" footprint {memory.footprint_bytes():,}B "
              f"ops/tok {np.mean(ops_test):.1f}", flush=True)

    # ---- g1: window-1 == independent closed-form bigram --------------------
    closed = _bigram_closed_form(train, test, alpha, vocab)
    g1_delta = abs(cells_raw["1"] - closed)
    g1_ok = g1_delta <= G1_TOLERANCE
    print(f"g1 bigram closed form: model {cells_raw['1']:.12f} "
          f"closed {closed:.12f} delta {g1_delta:.3e} ok={g1_ok}", flush=True)

    # ---- g2: footprint monotone in window ---------------------------------
    footprints = [cells[str(w)]["footprint_bytes"] for w in ladder]
    g2_ok = all(b >= a for a, b in zip(footprints, footprints[1:]))
    print(f"g2 footprint monotone: {footprints} ok={g2_ok}", flush=True)

    if (not g1_ok or not g2_ok) and not smoke:
        evidence = {
            "milestone": "M175", "cell": "A0",
            "admissible_as_evidence": False,
            "void": True,
            "void_reason": "anchor check failed",
            "g1_delta": g1_delta, "g1_ok": g1_ok, "g2_ok": g2_ok,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        return evidence

    # ---- gate: more memory must not hurt (both variants) ------------------
    ppl_w1 = cells["1"]["test_perplexity"]
    ppl_wmax = cells[str(max_window)]["test_perplexity"]
    gate_fired = bool(ppl_wmax > ppl_w1 * 1.10)
    back_wmax = backoff_cells[str(max_window)]["test_perplexity"]
    better_variant = ("backoff" if back_wmax < ppl_wmax else "uniform")
    evidence: dict[str, Any] = {
        "milestone": "M175",
        "cell": "A0 first text encoder (additive next-token on wikitext-103)",
        "admissible_as_evidence": not smoke,
        "registered_in": config.get("registered_in"),
        "config_file": Path(config_path).name,
        "config": config,
        "corpus": {
            "kind": "wikitext103_100000_char_tokens",
            "cache_file": config["corpus"]["cache_file"],
            "vocab_fingerprint": fingerprint,
            "cache_version": cache_version,
            "train_tokens": len(train),
            "test_tokens": len(test),
            "vocabulary_size": len(vocab),
            "uniform_baseline_perplexity": len(vocab),
        },
        "model": {
            "interpolation": config["model"]["interpolation"],
            "alpha": alpha,
        },
        "per_window": cells,
        "per_window_backoff": backoff_cells,
        "anchors": {
            "g1_bigram_closed_form": {"model": cells_raw["1"],
                "closed_form": closed,
                "delta": g1_delta, "tolerance": G1_TOLERANCE, "ok": g1_ok},
            "g2_footprint_monotone": {"footprints": footprints, "ok": g2_ok},
        },
        "gate": {
            "more_memory_hurts_uniform_fired": gate_fired,
            "window1_test_ppl": ppl_w1,
            "max_window_test_ppl": ppl_wmax,
            "threshold": round(ppl_w1 * 1.10, 4),
        },
        "verdict": {
            "uniform_max_window_ppl": ppl_wmax,
            "backoff_max_window_ppl": back_wmax,
            "max_window_winner": better_variant,
            "arm_selection": ("NONE — A0 selects no arm from test numbers; "
                              "cells A/D pin their arm in A's own "
                              "registration using a validation slice"),
        },
        "role": ("this encoder is the frozen text arm reused by cells A "
                 "(text->text transfer) and D (license-clean Wikipedia "
                 "fit-and-report)"),
        "disclosure": ("small-transformer comparison NOT run (N90.6); "
                       "additive model reported vs the uniform baseline and "
                       "its own window sweep only; no novelty claim (M129)"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"g1_delta": g1_delta, "uniform_wmax": ppl_wmax,
                      "backoff_wmax": back_wmax,
                      "gate_fired": gate_fired,
                      "better_variant": better_variant}, indent=1),
          flush=True)
    print(f"M175 cell A0 complete -> {output_dir / 'evidence.json'}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m175_cell_a0(args.config, args.output)


if __name__ == "__main__":
    main()
