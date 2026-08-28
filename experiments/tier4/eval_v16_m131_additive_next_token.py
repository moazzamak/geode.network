"""M131 - registered engineering measurement: additive next-token model on a DSL.

v20 B4b (``analysis/ENGINEERING_PLAN_v20.md``). Measures the B4a
:class:`ProgrammaticMemory` as an additive interpolated next-token model:

- Corpus: a constrained DSL generated deterministically (seeded), split by
  seed (train/valid/test programs never share a seed). Zero network, byte-
  reproducible (N90.3).
- Model: p(next|context) = (1/K) * sum_{k=1..K} smoothed exact-order counts
  (Jelinek-Mercer with uniform lambda over orders) - an ADDITIVE construction
  built from ProgrammaticMemory.exact_continuations (N90.4).
- Measured: perplexity (valid/test) vs the window ladder (the "how far back"
  dial), footprint bytes, integer operations per token, matched-length
  histogram of the pure backoff model. Gate: more memory must not hurt by
  >10% (N90.5). Transformer comparison disclosed as not run (N90.6).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    write_canonical_json,
)
from src.programmatic_memory import ProgrammaticMemory

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m131_additive_next_token.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m131_additive_next_token"


# ---------------------------------------------------------------------------
# DSL generator (deterministic, seeded)
# ---------------------------------------------------------------------------


def _generate_program(rng: np.random.Generator, max_depth: int) -> list[str]:
    n_statements = int(rng.integers(4, 13))
    tokens: list[str] = []
    for _ in range(n_statements):
        tokens.extend(_generate_statement(rng, max_depth))
    return tokens


def _generate_statement(rng: np.random.Generator, depth: int) -> list[str]:
    kind = int(rng.integers(0, 4))
    if kind == 0:  # let v = expr;
        return (
            ["let", " ", f"v{int(rng.integers(0, 10))}", " ", "=", " "]
            + _generate_expr(rng, depth) + [";"]
        )
    if kind == 1:  # return expr;
        return ["return", " "] + _generate_expr(rng, depth) + [";"]
    if kind == 2:  # if (c) { ... } [else { ... }]
        body = []
        for _ in range(int(rng.integers(1, 4))):
            body.extend(_generate_statement(rng, max(0, depth - 1)))
        result = ["if", " ", "(", *_generate_cond(rng, depth), ")", " ",
                  "{", " ", *body, " ", "}"]
        if rng.random() < 0.5:
            else_body = []
            for _ in range(int(rng.integers(1, 4))):
                else_body.extend(_generate_statement(rng, max(0, depth - 1)))
            result.extend([" ", "else", " ", "{", " ", *else_body, " ", "}"])
        return result
    # kind == 3: call f(args);
    args = []
    for _ in range(int(rng.integers(1, 4))):
        args.extend(_generate_expr(rng, max(0, depth - 1)))
        args.append(",")
        args.append(" ")
    if args:
        args = args[:-2]  # drop trailing ", "
    return ["call", " ", f"f{int(rng.integers(0, 6))}", "(", *args, ")", ";"]


def _generate_cond(rng: np.random.Generator, depth: int) -> list[str]:
    left = _generate_expr(rng, depth)
    op = rng.choice(["==", "<", ">"])
    right = _generate_expr(rng, depth)
    return [*left, " ", op, " ", *right]


def _generate_expr(rng: np.random.Generator, depth: int) -> list[str]:
    if depth <= 0 or rng.random() < 0.6:
        kind = int(rng.integers(0, 3))
        if kind == 0:
            return [f"n{int(rng.integers(0, 10))}"]
        if kind == 1:
            return [f"v{int(rng.integers(0, 10))}"]
        return ["(", *[f"n{int(rng.integers(0, 10))}"], ")"]
    left = _generate_expr(rng, depth - 1)
    op = rng.choice(["+", "-", "*", "/"])
    right = _generate_expr(rng, depth - 1)
    return ["(", *left, " ", op, " ", *right, ")"]


def _generate_corpus(config: dict[str, Any], limit_programs: int | None = None
                     ) -> dict[str, list[str]]:
    corpus_cfg = config["corpus"]
    out: dict[str, list[str]] = {}
    for split in ("train", "valid", "test"):
        seed = int(corpus_cfg["seeds"][split])
        rng = np.random.default_rng(seed)
        n_programs = int(corpus_cfg["programs"][split])
        if limit_programs:
            n_programs = min(n_programs, limit_programs)
        tokens: list[str] = []
        for _ in range(n_programs):
            tokens.extend(
                _generate_program(rng, int(corpus_cfg["max_expr_depth"]))
            )
        out[split] = tokens
    return out


# ---------------------------------------------------------------------------
# Additive interpolated next-token model
# ---------------------------------------------------------------------------


def _vocabulary(tokens: list[str]) -> list[str]:
    return sorted(set(tokens))


def _order_log_probs(
    memory: ProgrammaticMemory, context: list[str], token: str,
    order: int, alpha: float, vocab: list[str],
) -> tuple[float, int]:
    """Add-alpha smoothed probability of *token* under EXACT order-*order*
    counts, plus the number of stored entries in that order's continuation."""
    result = memory.exact_continuations(context, order)
    counts = dict(result.counts)
    total = result.total()
    entries = len(result.counts)
    count = counts.get(token, 0)
    prob = (count + alpha) / (total + alpha * len(vocab))
    return math.log(prob), entries


def _interpolated_log_prob(
    memory: ProgrammaticMemory, context: list[str], token: str,
    window: int, alpha: float, vocab: list[str],
) -> tuple[float, int]:
    """Uniform mixture over orders 1..window: log p = log((1/K) sum_k p_k)."""
    total_log = 0.0
    for order in range(1, window + 1):
        log_p, _ = _order_log_probs(memory, context, token, order, alpha, vocab)
        total_log += math.exp(log_p)
    return math.log(total_log / window), 0


def _perplexity(
    memory: ProgrammaticMemory, tokens: list[str], window: int,
    alpha: float, vocab: list[str], ops: list[int],
) -> float:
    """Average negative log-likelihood under the additive model, exponentiated.

    ops accumulates per-token integer operations (K lookups + summed entries)
    for the energy/footprint story; the same ops list is shared across cells.
    """
    nll_sum = 0.0
    n = 0
    for i in range(1, len(tokens)):
        context = tokens[max(0, i - memory.window):i]
        log_p, _ = _interpolated_log_prob(
            memory, context, tokens[i], window, alpha, vocab)
        nll_sum -= log_p
        # count operations: K lookups + summed continuation entries
        entries = 0
        for order in range(1, window + 1):
            result = memory.exact_continuations(context, order)
            entries += len(result.counts)
        ops.append(window + entries)
        n += 1
    return math.exp(nll_sum / n)


def run_m131(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    smoke_rows = int(config.get("_smoke_rows", 0))
    smoke_skip = bool(config.get("_smoke_skip_gates", False))
    alpha = float(config["model"]["alpha"])
    ladder = [int(w) for w in config["model"]["window_ladder"]]
    max_window = int(config["model"]["max_window"])

    corpus = _generate_corpus(
        config, limit_programs=(smoke_rows or None))
    train, valid, test = corpus["train"], corpus["valid"], corpus["test"]
    vocab = _vocabulary(train)
    print(f"corpus: train {len(train)} / valid {len(valid)} / test {len(test)} "
          f"tokens, vocab {len(vocab)}", flush=True)

    cells: dict[str, Any] = {}
    for window in ladder:
        memory = ProgrammaticMemory(window=window)
        memory.register(train)
        ops_train: list[int] = []
        ops_test: list[int] = []
        valid_ppl = _perplexity(memory, valid, window, alpha, vocab, ops_train)
        test_ppl = _perplexity(memory, test, window, alpha, vocab, ops_test)
        cells[str(window)] = {
            "window": window,
            "valid_perplexity": round(valid_ppl, 4),
            "test_perplexity": round(test_ppl, 4),
            "footprint_bytes": memory.footprint_bytes(),
            "context_count": memory.context_count,
            "entry_count": memory.entry_count,
            "ops_per_token_train_mean": round(float(np.mean(ops_train)), 2),
            "ops_per_token_test_mean": round(float(np.mean(ops_test)), 2),
        }
        print(f"  window {window}: valid {valid_ppl:.3f} test {test_ppl:.3f} "
              f"footprint {memory.footprint_bytes():,}B "
              f"ops/tok {np.mean(ops_test):.1f}", flush=True)

    # pure backoff (longest-match) model matched-length histogram on test
    backoff_memory = ProgrammaticMemory(window=max_window)
    backoff_memory.register(train)
    histogram = backoff_memory.matched_length_histogram(test)

    test_ppl_w1 = cells["1"]["test_perplexity"]
    test_ppl_w8 = cells["8"]["test_perplexity"]
    kill_switch_fired = bool(test_ppl_w8 > test_ppl_w1 * 1.10)
    if kill_switch_fired and not smoke_skip:
        print(f"KILL SWITCH: window 8 ({test_ppl_w8}) > window 1 "
              f"({test_ppl_w1}) * 1.10", flush=True)

    evidence: dict[str, Any] = {
        "milestone": "M131",
        "admissible_as_evidence": True,
        "registered_in": config.get("registered_in"),
        "config_file": Path(config_path).name,
        "config": config,
        "hypothesis": ("additive interpolated next-token test perplexity "
                       "decreases with the window (how far back); footprint "
                       "grows with window; per-token operations grow slowly"),
        "corpus": {
            "kind": "generated_dsl",
            "train_tokens": len(train),
            "valid_tokens": len(valid),
            "test_tokens": len(test),
            "vocabulary_size": len(vocab),
            "uniform_baseline_perplexity": len(vocab),
        },
        "model": {
            "interpolation": config["model"]["interpolation"],
            "alpha": alpha,
        },
        "per_window": cells,
        "backoff_matched_length_histogram_test": histogram,
        "gate": {
            "kill_switch_more_memory_hurts": kill_switch_fired,
            "window1_test_ppl": test_ppl_w1,
            "window8_test_ppl": test_ppl_w8,
            "threshold": test_ppl_w1 * 1.10,
            "gate_skipped": smoke_skip,
        },
        "disclosure": ("small-transformer comparison NOT run (compute GPU = "
                       "display GPU, TDR risk); additive model reported vs "
                       "uniform baseline and its own window sweep only (N90.6)"),
        "note": "engineering measurement on a generated constrained DSL; no "
                "novelty claim (N90.1)",
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM131 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m131(args.config, args.output)


if __name__ == "__main__":
    main()
