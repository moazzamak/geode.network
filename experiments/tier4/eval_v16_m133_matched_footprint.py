"""M133 - registered measurement: matched-footprint perplexity frontier.

v20 follow-up (``analysis/ENGINEERING_PLAN_v20.md`` B4b). Completes the
transformer arm M131 disclosed as not run (N90.6): for each stored-footprint
of the additive count model (reused from sealed M131 evidence), train a tiny
transformer of the same footprint on the same deterministic DSL train split
and measure test perplexity. The matched-footprint frontier (win/tie/lose per
footprint) is the measured object; the outcome in the sub-MB regime is unknown
before measurement (N92.2).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from experiments.common.v5_artifacts import (
    build_artifact_index,
    write_canonical_json,
)
from experiments.tier4.eval_v16_m131_additive_next_token import _generate_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m133_matched_footprint.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m133_matched_footprint"
M131_EVIDENCE = (
    REPO_ROOT / "logs" / "results" / "v16" / "m131_additive_next_token" / "evidence.json"
)


class TinyTransformer(nn.Module):
    """A minimal GPT-style decoder with a tied embedding head."""

    def __init__(self, vocab: int, d_model: int, layers: int, heads: int,
                 ff: int, context: int = 64) -> None:
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, context, d_model))
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(d_model),
                nn.MultiheadAttention(d_model, heads, batch_first=True),
                nn.LayerNorm(d_model),
                nn.Sequential(
                    nn.Linear(d_model, ff), nn.GELU(), nn.Linear(ff, d_model)),
            ])
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # tied LM head
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens) + self.pos[:, : tokens.shape[1], :]
        mask = torch.triu(
            torch.full((tokens.shape[1], tokens.shape[1]), float("-inf")),
            diagonal=1,
        ).to(tokens.device)
        for ln1, attn, ln2, mlp in self.blocks:
            x = x + attn(ln1(x), ln1(x), ln1(x), attn_mask=mask, need_weights=False)[0]
            x = x + mlp(ln2(x))
        return self.head(self.norm(x))


def _parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _make_batch(tokens: np.ndarray, rng: np.random.Generator,
                seq: int, batch: int) -> torch.Tensor:
    rows = []
    for _ in range(batch):
        start = int(rng.integers(0, max(1, len(tokens) - seq)))
        rows.append(tokens[start:start + seq])
    return torch.tensor(np.stack(rows), dtype=torch.long)


def _train(model: nn.Module, tokens: np.ndarray, vocab: int, steps: int,
           batch_tokens: int, lr: float, seed: int, torch_threads: int,
           device: str) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(torch_threads)
    seq = model.context
    batch = max(1, batch_tokens // seq)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for step in range(steps):
        optimizer.zero_grad()
        x = _make_batch(tokens, rng, seq, batch).to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return {
        "final_loss": float(losses[-1]),
        "mean_loss_last_500": float(np.mean(losses[-500:])),
    }


@torch.no_grad()
def _perplexity(model: nn.Module, tokens: np.ndarray, vocab: int, seq: int,
                device: str) -> float:
    model.eval()
    nll_sum = 0.0
    n = 0
    for start in range(0, len(tokens) - seq, seq):
        block = tokens[start:start + seq]
        x = torch.tensor(block[None, :], dtype=torch.long, device=device)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        nll_sum += float(loss.item()) * (seq - 1)
        n += seq - 1
    return float(math.exp(nll_sum / max(n, 1)))


def run_m133(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    smoke_rows = int(config.get("_smoke_rows", 0))
    smoke_steps = int(config.get("_smoke_steps", 0))
    smoke_skip = bool(config.get("_smoke_skip_gates", False))

    # corpus (deterministic, same seeds as M131)
    corpus = _generate_corpus(config, limit_programs=(smoke_rows or None))
    train, valid, test = corpus["train"], corpus["valid"], corpus["test"]
    vocab_list = sorted(set(train))
    vocab = len(vocab_list)
    token_to_id = {t: i for i, t in enumerate(vocab_list)}
    train_ids = np.asarray([token_to_id[t] for t in train], dtype=np.int64)
    valid_ids = np.asarray([token_to_id[t] for t in valid], dtype=np.int64)
    test_ids = np.asarray([token_to_id[t] for t in test], dtype=np.int64)
    print(f"corpus: train {len(train_ids)} / valid {len(valid_ids)} / "
          f"test {len(test_ids)} tokens, vocab {vocab}", flush=True)

    # count-model points from sealed M131 evidence (never recomputed)
    m131 = json.loads(M131_EVIDENCE.read_text(encoding="utf-8"))
    count_points: list[dict[str, Any]] = []
    for w in config["count_model"]["window_points"]:
        cell = m131["per_window"][w]
        count_points.append({
            "window": int(cell["window"]),
            "test_perplexity": cell["test_perplexity"],
            "footprint_bytes": cell["footprint_bytes"],
        })
    count_points.sort(key=lambda p: p["footprint_bytes"])

    # transformer points
    tcfg = config["transformer"]
    steps = smoke_steps or int(tcfg["train_steps"])
    device = "cpu"  # registered: CPU (no TDR risk; tiny models)
    torch_threads = int(tcfg["torch_threads"])
    transformer_points: list[dict[str, Any]] = []
    for spec in tcfg["ladder"]:
        model = TinyTransformer(
            vocab=vocab, d_model=int(spec["d_model"]),
            layers=int(spec["layers"]), heads=int(spec["heads"]),
            ff=int(spec["ff"]))
        params = _parameter_count(model)
        fp32_bytes = params * 4
        print(f"  training {spec['name']}: d={spec['d_model']} L={spec['layers']} "
              f"H={spec['heads']} ff={spec['ff']} -> {params:,} params "
              f"({fp32_bytes:,}B fp32)", flush=True)
        training = _train(
            model, train_ids, vocab, steps, int(tcfg["batch_tokens"]),
            float(tcfg["lr"]), int(tcfg["seed"]), torch_threads, device)
        valid_ppl = _perplexity(model, valid_ids, vocab, model.context, device)
        test_ppl = _perplexity(model, test_ids, vocab, model.context, device)
        transformer_points.append({
            "name": spec["name"],
            "params": params,
            "footprint_bytes_fp32": fp32_bytes,
            "footprint_bytes_fp16": params * 2,
            "valid_perplexity": round(valid_ppl, 4),
            "test_perplexity": round(test_ppl, 4),
            "training": training,
            "hyperparameters": {
                "d_model": spec["d_model"], "layers": spec["layers"],
                "heads": spec["heads"], "ff": spec["ff"], "steps": steps,
            },
        })
        print(f"    valid {valid_ppl:.3f} test {test_ppl:.3f}", flush=True)

    # matched-footprint comparison (nearest transformer fp32 to each count point)
    transformer_points.sort(key=lambda p: p["footprint_bytes_fp32"])
    matches: list[dict[str, Any]] = []
    for cp in count_points:
        nearest = min(
            transformer_points,
            key=lambda tp: abs(tp["footprint_bytes_fp32"] - cp["footprint_bytes"]),
        )
        margin = nearest["test_perplexity"] - cp["test_perplexity"]
        matches.append({
            "count_window": cp["window"],
            "count_footprint_bytes": cp["footprint_bytes"],
            "count_test_perplexity": cp["test_perplexity"],
            "nearest_transformer": nearest["name"],
            "transformer_footprint_bytes_fp32": nearest["footprint_bytes_fp32"],
            "transformer_test_perplexity": nearest["test_perplexity"],
            "delta_transformer_minus_count": round(margin, 4),
            "winner": ("transformer" if margin < 0
                       else "count" if margin > 0 else "tie"),
        })

    evidence: dict[str, Any] = {
        "milestone": "M133",
        "admissible_as_evidence": True,
        "registered_in": config.get("registered_in"),
        "config_file": Path(config_path).name,
        "config": config,
        "question": ("at matched footprint on the deterministic DSL, does the "
                     "additive count model beat, tie, or lose to a genuinely "
                     "trained tiny transformer?"),
        "corpus": {
            "kind": "generated_dsl",
            "train_tokens": len(train_ids),
            "valid_tokens": len(valid_ids),
            "test_tokens": len(test_ids),
            "vocabulary_size": vocab,
            "uniform_baseline_perplexity": vocab,
        },
        "count_model_points": count_points,
        "transformer_points": transformer_points,
        "matched_footprint": matches,
        "notes": {
            "count_points_reused": "sealed M131 evidence, never recomputed (N92.3)",
            "footprint": "stored fp32 bytes (4/param); fp16 reported alongside (N92.4)",
            "device": device,
            "torch_threads": torch_threads,
            "gate": "no pre-registered winner gate (N92.6); a surprise direction is "
                    "reported as a measured fact with the crossover point (N92.7)",
            "smoke": bool(smoke_rows or smoke_steps),
        },
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM133 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m133(args.config, args.output)


if __name__ == "__main__":
    main()
