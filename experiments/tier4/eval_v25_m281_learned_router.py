"""M281 — the learned-router study: the true trained-on-N bridge.

Registered 22 Aug 2026 (plan v25, the M272-M281 wave). The
whitepaper's own rule: a learned policy may replace the
deterministic router only behind a MEASURED gate. The candidate: a
closed-form ridge over frozen BERT features, trained on NEW
generated marker-free examples (the cell-4 generators, different
seeds) and tested on the SEALED cell-4 700-item mix. Admission:
strictly fewer misroutes than the incumbent embedding router, or
equal misroutes with strictly higher routed accuracy — a tie is NOT
admission. The deterministic router stays the incumbent unless the
gate passes.

CPU-only. Evidence: logs/results/v25/m281_learned_router/evidence.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

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
from experiments.tier4.eval_v25_m268_natural_routing import (
    _OP_WORDS,
    _NUMBER_WORDS,
    _gen_bool_question,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SEALED_CELL4 = (REPO_ROOT / "logs" / "results" / "v25"
                / "m268_routing_study"
                / "evidence_natural_routing.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m281_learned_router")

FAMILIES = ["sentiment", "arithmetic", "logic"]
TRAIN_SEEDS = {"arithmetic": 20260824, "logic": 20260825}


def generate_train(config: dict[str, Any]) -> list[tuple[str, str]]:
    """NEW marker-free examples (same generators, fresh seeds)."""
    out: list[tuple[str, str]] = []
    ar_cfg = config["mix"]["arithmetic"]
    rng = random.Random(TRAIN_SEEDS["arithmetic"])
    templates = ar_cfg["templates"]
    for _ in range(300):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(["+", "-", "*"])
        w1, w2 = _OP_WORDS[op]
        text = rng.choice(templates).format(
            a=_NUMBER_WORDS[a], b=_NUMBER_WORDS[b],
            op_word=w1, op_word2=w2)
        out.append(("arithmetic", text))
    rng = random.Random(TRAIN_SEEDS["logic"])
    for _ in range(300):
        question, _ = _gen_bool_question(rng)
        out.append(("logic", question))
    return out


def rebuild_test_mix(config: dict[str, Any]) -> list[dict[str, Any]]:
    """The sealed cell-4 mix, rebuilt from the registered seeds."""
    import random
    mix: list[dict[str, Any]] = []
    ar_cfg = config["mix"]["arithmetic"]
    rng = random.Random(ar_cfg["seed"])
    templates = ar_cfg["templates"]
    for _ in range(ar_cfg["n"]):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(["+", "-", "*"])
        w1, w2 = _OP_WORDS[op]
        text = rng.choice(templates).format(
            a=_NUMBER_WORDS[a], b=_NUMBER_WORDS[b],
            op_word=w1, op_word2=w2)
        mix.append({"task": "arithmetic", "reference": "x",
                    "input": text})
    bo_cfg = config["mix"]["boolean"]
    rng = random.Random(bo_cfg["seed"])
    for _ in range(bo_cfg["n"]):
        question, _ = _gen_bool_question(rng)
        mix.append({"task": "logic", "reference": "x",
                    "input": question})
    from datasets import load_dataset as _hf_load
    se_cfg = config["mix"]["sentiment"]
    rows = se_cfg["rows"]
    ds = _hf_load(se_cfg["hf_id"], split=se_cfg["split"]).select(
        range(rows[0], rows[1]))
    for row in ds:
        mix.append({"task": "sentiment", "reference": "x",
                    "input": row["text"]})
    return mix


def run_m281(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    configure_external_cache_environment()

    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased",
                                        local_files_only=True)
    model = AutoModel.from_pretrained("bert-base-uncased",
                                      local_files_only=True).eval()

    def embed(texts: list[str]) -> np.ndarray:
        enc = tok(texts, padding=True, truncation=True, max_length=128,
                  return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.numpy().astype(np.float64)

    cell4_config = json.loads(Path(
        REPO_ROOT / "experiments" / "configs" / "v25"
        / "m268_natural_routing.json").read_text(encoding="utf-8"))
    sealed = json.loads(SEALED_CELL4.read_text(encoding="utf-8"))

    # ---- train the candidate on NEW examples -------------------------
    train = generate_train(cell4_config)
    from datasets import load_dataset as _hf_load
    ds = _hf_load("stanfordnlp/imdb", split="test").select(
        range(3000, 3300))
    train += [("sentiment", r["text"]) for r in ds]
    X_train = embed([t for _, t in train])
    y_train = np.array([FAMILIES.index(f) for f, _ in train])

    def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0
                  ) -> tuple[np.ndarray, np.ndarray]:
        n_classes = len(FAMILIES)
        Y = np.zeros((len(y), n_classes))
        Y[np.arange(len(y)), y] = 1.0
        Yc = Y - Y.mean(axis=0, keepdims=True)
        Xc = X - X.mean(axis=0, keepdims=True)
        W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]),
                            Xc.T @ Yc)
        b = Y.mean(axis=0) - X.mean(axis=0) @ W
        return W, b

    W, b = ridge_fit(X_train, y_train)

    # ---- test on the SEALED cell-4 mix --------------------------------
    test_mix = rebuild_test_mix(cell4_config)
    assert len(test_mix) == 700
    X_test = embed([m["input"] for m in test_mix])
    scores = X_test @ W + b
    learned = scores.argmax(axis=1)
    true = np.array([FAMILIES.index(m["task"]) for m in test_mix])
    misroutes = int((learned != true).sum())

    # routed accuracy reconstruction: where the learned route equals
    # the sealed embed route, the arm answer is the sealed one
    sealed_items = sealed["per_item"]
    agree = sum(1 for i in range(700)
                if FAMILIES[learned[i]] == sealed_items[i]["embed_route"])
    scored_correct = sum(1 for i in range(700)
                         if FAMILIES[learned[i]]
                         == sealed_items[i]["embed_route"]
                         and sealed_items[i]["embed_correct"])
    reconstruction = {
        "n_route_agreement_with_incumbent": agree,
        "n_scored": agree,
        "n_scored_correct": scored_correct,
        "routed_accuracy_where_reconstructable":
            round(scored_correct / agree, 4) if agree else None,
    }

    incumbent = {
        "misroutes": sealed["results"]["n_embed_misroutes"],
        "routed_accuracy": sealed["results"]["embed_router"][
            "overall"]["routed_accuracy"],
    }
    # the registered admission rule: strictly better, a tie is NOT
    admitted = (
        misroutes < incumbent["misroutes"]
        or (misroutes == incumbent["misroutes"]
            and reconstruction["routed_accuracy_where_reconstructable"]
            is not None
            and reconstruction["routed_accuracy_where_reconstructable"]
            > incumbent["routed_accuracy"])
    )

    evidence: dict[str, Any] = {
        "milestone": "M281",
        "cell": "learned-router study",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "candidate": "ridge over frozen BERT features",
            "train_seeds": TRAIN_SEEDS,
            "train_n": len(train),
            "test": "the sealed cell-4 700-item mix",
        }),
        "results": {
            "candidate_misroutes": misroutes,
            "incumbent_misroutes": incumbent["misroutes"],
            "incumbent_routed_accuracy": incumbent["routed_accuracy"],
            "reconstruction": reconstruction,
            "admitted": admitted,
            "verdict": ("ADMITTED" if admitted else
                        "NOT ADMITTED — the deterministic incumbent "
                        "stands (a tie is not admission, registered)"),
        },
        "scope_note": ("the candidate is trained on NEW generated "
                       "examples only; the test is the sealed mix; "
                       "no val-selected numbers; the deterministic "
                       "router stays the incumbent unless the gate "
                       "passes"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M281 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m281(args.output)


if __name__ == "__main__":
    main()
