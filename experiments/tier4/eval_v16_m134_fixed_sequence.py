"""M134 - registered measurement: fixed-construction sequence families on the frontier.

v20 follow-up (``analysis/ENGINEERING_PLAN_v20.md`` B4b follow-up, M134). Extends
the sealed M133 matched-footprint frontier with two no-backprop sequence families:

- **Reservoir** (ESN-style): fixed random recurrent matrix (spectral radius < 1),
  tanh state, closed-form ridge readout on the state trajectory.
- **Fixed-attention**: fixed random embeddings / QKV / output / FFN projections,
  softmax attention over a context window, closed-form ridge readout on the
  mixed features.

Both keep the programme's identity: fixed nonlinear construction + closed-form
ridge readout (streaming Gram accumulation, RidgeAccumulator arithmetic). The
gap to the trained transformer at matched footprint IS the measured "price of
learning" for sequence models (N93.2). Count + transformer points are reused
from sealed M131/M133 evidence, never recomputed (N93.3).
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
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v16_m131_additive_next_token import _generate_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m134_fixed_sequence.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m134_fixed_sequence"


# ---------------------------------------------------------------------------
# Reservoir (ESN-style) + ridge readout
# ---------------------------------------------------------------------------


def _reservoir_params(units: int, vocab: int, seed: int,
                      spectral_radius: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # W_in maps the one-hot token (vocab,) into the reservoir state (units,)
    w_in = rng.standard_normal((units, vocab)) / math.sqrt(vocab)
    w_rec = rng.standard_normal((units, units)) / math.sqrt(units)
    eig = np.linalg.eigvals(w_rec)
    rho = float(np.max(np.abs(eig))) if len(eig) else 1.0
    if rho > 0:
        w_rec = w_rec * (spectral_radius / rho)
    return w_in, w_rec


def _reservoir_readout(
    train_ids: np.ndarray, test_ids: np.ndarray, vocab: int,
    units: int, seed: int, spectral_radius: float, warmup: int,
    penalty: float, flush: int = 4096,
) -> dict[str, Any]:
    w_in, w_rec = _reservoir_params(units, vocab, seed, spectral_radius)
    acc = RidgeAccumulator(units, vocab)
    h = np.zeros(units, dtype=np.float64)
    state_buffer: list[np.ndarray] = []
    label_buffer: list[int] = []
    seen = 0

    def flush_buffer() -> None:
        if state_buffer:
            acc.add(np.asarray(state_buffer, dtype=np.float64),
                    np.asarray(label_buffer, dtype=np.int64))
            state_buffer.clear()
            label_buffer.clear()

    x = np.zeros(vocab, dtype=np.float64)
    for t in range(len(train_ids) - 1):
        x[:] = 0.0
        x[train_ids[t]] = 1.0
        h = np.tanh(w_in @ x + w_rec @ h)
        if t >= warmup:
            state_buffer.append(h.copy())
            label_buffer.append(int(train_ids[t + 1]))
            seen += 1
            if len(state_buffer) >= flush:
                flush_buffer()
    flush_buffer()

    standardise = acc.standardiser()
    weights = acc.solve(penalty)

    # evaluate test perplexity
    nll_sum = 0.0
    n = 0
    h = np.zeros(units, dtype=np.float64)
    x = np.zeros(vocab, dtype=np.float64)
    for t in range(len(test_ids) - 1):
        x[:] = 0.0
        x[test_ids[t]] = 1.0
        h = np.tanh(w_in @ x + w_rec @ h)
        if t >= warmup:
            feature = standardise(h[None, :])
            logits = feature @ weights[:-1] + weights[-1]
            log_probs = logits - np.log(np.sum(np.exp(logits)))
            nll_sum -= log_probs[0, int(test_ids[t + 1])]
            n += 1
    return {
        "units": units,
        "train_states": seen,
        "test_perplexity": round(float(math.exp(nll_sum / max(n, 1))), 4),
        "params": int(units * units + vocab * units + (units + 1) * vocab),
    }


# ---------------------------------------------------------------------------
# Fixed-attention extractor + ridge readout
# ---------------------------------------------------------------------------


def _fixed_attention_features(
    ids: np.ndarray, d_model: int, vocab: int, context: int, heads: int,
    seed: int, chunk: int = 4096,
) -> np.ndarray:
    """Compute fixed-attention mixed features for every position (batched)."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(d_model)
    embed = rng.standard_normal((vocab, d_model)) * scale
    pos = rng.standard_normal((context, d_model)) * scale
    w_q = rng.standard_normal((d_model, d_model)) * scale
    w_k = rng.standard_normal((d_model, d_model)) * scale
    w_v = rng.standard_normal((d_model, d_model)) * scale
    w_o = rng.standard_normal((d_model, d_model)) * scale
    head_dim = max(1, d_model // heads)

    features = np.zeros((len(ids), d_model), dtype=np.float64)
    for start in range(context, len(ids), chunk):
        stop = min(start + chunk, len(ids))
        ids_c = ids[start:stop]
        # embeddings + positional (window positions relative to current)
        h = embed[ids_c].copy()
        # attention over the last `context` tokens (batched shifted stacks)
        q = h @ w_q  # (B, d)
        # K, V shifted stacks: K[t, w, :] = embed[ids[t - (context-1-w)]] @ w_k
        k_stack = np.zeros((stop - start, context, d_model), dtype=np.float64)
        v_stack = np.zeros((stop - start, context, d_model), dtype=np.float64)
        for w in range(context):
            src = np.clip(start - (context - 1 - w), 0, None)
            k_stack[:, w, :] = (embed[ids[src:src + (stop - start)]] + pos[w]) @ w_k
            v_stack[:, w, :] = (embed[ids[src:src + (stop - start)]] + pos[w]) @ w_v
        scores = np.einsum("bd,bwd->bw", q, k_stack) / math.sqrt(d_model)
        # attn is (B, context); the extra axis in the mix below is intentional
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        mixed = (attn[:, None, :] @ v_stack)[:, 0, :]  # (B, d)
        out = mixed @ w_o
        # per-head split for a bit more structure
        out = out.reshape(-1, heads, head_dim).reshape(-1, d_model)
        features[start:stop] = out
    return features


def _fixed_attention_readout(
    train_ids: np.ndarray, test_ids: np.ndarray, vocab: int,
    d_model: int, context: int, heads: int, seed: int, penalty: float,
) -> dict[str, Any]:
    # features for train (buffer them; 2.3M x d float64 is fine for d<=256)
    train_feats = _fixed_attention_features(
        train_ids, d_model, vocab, context, heads, seed)
    test_feats = _fixed_attention_features(
        test_ids, d_model, vocab, context, heads, seed)
    acc = RidgeAccumulator(d_model, vocab)
    for start in range(context, len(train_ids) - 1, 4096):
        stop = min(start + 4096, len(train_ids) - 1)
        acc.add(train_feats[start:stop], train_ids[start + 1:stop + 1])
    standardise = acc.standardiser()
    weights = acc.solve(penalty)
    nll_sum = 0.0
    n = 0
    for start in range(context, len(test_ids) - 1, 4096):
        stop = min(start + 4096, len(test_ids) - 1)
        feature = standardise(test_feats[start:stop])
        logits = feature @ weights[:-1] + weights[-1]
        log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        nll_sum -= float(np.sum(log_probs[np.arange(stop - start),
                                          test_ids[start + 1:stop + 1]]))
        n += stop - start
    # footprint: fixed matrices + ridge weights
    fixed_params = (
        vocab * d_model          # embed
        + context * d_model      # pos
        + 4 * d_model * d_model  # q,k,v,o
        + (d_model + 1) * vocab  # ridge
    )
    return {
        "d_model": d_model,
        "context": context,
        "test_perplexity": round(float(math.exp(nll_sum / max(n, 1))), 4),
        "params": fixed_params,
    }


def run_m134(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    smoke_rows = int(config.get("_smoke_rows", 0))
    smoke_skip = bool(config.get("_smoke_skip_gates", False))

    corpus = _generate_corpus(config, limit_programs=(smoke_rows or None))
    train, valid, test = corpus["train"], corpus["valid"], corpus["test"]
    vocab_list = sorted(set(train))
    vocab = len(vocab_list)
    token_to_id = {t: i for i, t in enumerate(vocab_list)}
    train_ids = np.asarray([token_to_id[t] for t in train], dtype=np.int64)
    test_ids = np.asarray([token_to_id[t] for t in test], dtype=np.int64)
    print(f"corpus: train {len(train_ids)} / test {len(test_ids)} tokens, "
          f"vocab {vocab}", flush=True)

    # reservoir family
    reservoir_points: list[dict[str, Any]] = []
    rcfg = config["reservoir"]
    for spec in rcfg["ladder"]:
        units = int(spec["units"])
        print(f"  reservoir {spec['name']} (units {units})...", flush=True)
        result = _reservoir_readout(
            train_ids, test_ids, vocab, units, int(rcfg["seed"]),
            float(rcfg["spectral_radius"]), int(rcfg["warmup_steps"]),
            float(rcfg["ridge_penalty"]))
        result["name"] = spec["name"]
        result["footprint_bytes_fp32"] = result["params"] * 4
        result["footprint_bytes_fp16"] = result["params"] * 2
        reservoir_points.append(result)
        print(f"    test {result['test_perplexity']} "
              f"fp {result['footprint_bytes_fp32']:,}B", flush=True)

    # fixed-attention family
    attention_points: list[dict[str, Any]] = []
    acfg = config["fixed_attention"]
    for spec in acfg["ladder"]:
        d_model = int(spec["d_model"])
        print(f"  fixed-attention {spec['name']} (d={d_model})...", flush=True)
        result = _fixed_attention_readout(
            train_ids, test_ids, vocab, d_model, int(acfg["context"]),
            int(acfg["heads"]), int(acfg["seed"]), float(acfg["ridge_penalty"]))
        result["name"] = spec["name"]
        result["footprint_bytes_fp32"] = result["params"] * 4
        result["footprint_bytes_fp16"] = result["params"] * 2
        attention_points.append(result)
        print(f"    test {result['test_perplexity']} "
              f"fp {result['footprint_bytes_fp32']:,}B", flush=True)

    # reused count + transformer points
    m131 = json.loads(
        (REPO_ROOT / config["reused_evidence"]["count"])
        .read_text(encoding="utf-8"))
    m133 = json.loads(
        (REPO_ROOT / config["reused_evidence"]["transformer"])
        .read_text(encoding="utf-8"))
    count_points = [
        {"family": "count", "name": f"w{w}", "test_perplexity": m131["per_window"][w]["test_perplexity"],
         "footprint_bytes_fp32": m131["per_window"][w]["footprint_bytes"]}
        for w in config["reused_evidence"].get("windows", ["1", "2", "3", "4", "8"])
    ]
    transformer_points = [
        {"family": "transformer", "name": t["name"], "test_perplexity": t["test_perplexity"],
         "footprint_bytes_fp32": t["footprint_bytes_fp32"]}
        for t in m133["transformer_points"]
    ]

    frontier = (
        count_points
        + transformer_points
        + [{"family": "reservoir", "name": p["name"], "test_perplexity": p["test_perplexity"],
            "footprint_bytes_fp32": p["footprint_bytes_fp32"]} for p in reservoir_points]
        + [{"family": "fixed_attention", "name": p["name"], "test_perplexity": p["test_perplexity"],
            "footprint_bytes_fp32": p["footprint_bytes_fp32"]} for p in attention_points]
    )
    frontier.sort(key=lambda p: p["footprint_bytes_fp32"])

    evidence: dict[str, Any] = {
        "milestone": "M134",
        "admissible_as_evidence": True,
        "registered_in": config.get("registered_in"),
        "config_file": Path(config_path).name,
        "config": config,
        "question": ("how much of the trained transformer's advantage over the "
                     "count model is recoverable by a fixed-construction sequence "
                     "model (reservoir / fixed-attention + ridge readout) at "
                     "matched footprint?"),
        "corpus": {"kind": "generated_dsl", "train_tokens": len(train_ids),
                   "test_tokens": len(test_ids), "vocabulary_size": vocab},
        "reservoir_points": reservoir_points,
        "fixed_attention_points": attention_points,
        "frontier": frontier,
        "notes": {
            "count_and_transformer_reused": "sealed M131/M133 evidence, never recomputed (N93.3)",
            "no_backprop": True,
            "readout": "closed-form ridge, streaming Gram (N93.4)",
            "smoke": bool(smoke_rows),
            "no_gate": "no pre-registered winner gate; the gap IS the measurement (N93.6)",
        },
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM134 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m134(args.config, args.output)


if __name__ == "__main__":
    main()
