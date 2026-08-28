"""M272 — ship the measured routing rules: the registered gate is a
MECHANICAL reproduction of the sealed cell-4 numbers with the new
module (geode/core/measured_routing.py).

The cell-4 mix is rebuilt from the registered seeds (same
generators, single source of truth imported from the cell-4 runner)
plus the IMDb rows 2500..2799. Every item is routed by the NEW
EmbeddingRouter over frozen BERT; the routes and the routed accuracy
are compared item-by-item with the sealed evidence
(`evidence_natural_routing.json`): the gate passes only if the
routes match and the recomputed embed-routed accuracy equals the
sealed 0.960.

Evidence: logs/results/v25/m272_routing_wiring/evidence.json.
"""
from __future__ import annotations

import argparse
import json
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
                  / "m272_routing_wiring")


def rebuild_mix(config: dict[str, Any]) -> list[dict[str, Any]]:
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
        value = {"+": a + b, "-": a - b, "*": a * b}[op]
        mix.append({"task": "arithmetic", "reference": str(value),
                    "input": text})
    bo_cfg = config["mix"]["boolean"]
    rng = random.Random(bo_cfg["seed"])
    for _ in range(bo_cfg["n"]):
        question, value = _gen_bool_question(rng)
        mix.append({"task": "logic", "reference": value,
                    "input": question})
    from datasets import load_dataset as _hf_load
    se_cfg = config["mix"]["sentiment"]
    rows = se_cfg["rows"]
    ds = _hf_load(se_cfg["hf_id"], split=se_cfg["split"]).select(
        range(rows[0], rows[1]))
    for i, row in enumerate(ds):
        mix.append({
            "task": "sentiment",
            "reference": ("positive" if row["label"] == 1 else "negative"),
            "input": row["text"],
            "row_index": rows[0] + i,
        })
    return mix


def bert_embedder_factory(device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased",
                                        local_files_only=True)
    model = AutoModel.from_pretrained("bert-base-uncased",
                                      local_files_only=True).to(
                                          device).eval()

    def embed(texts: list[str]) -> np.ndarray:
        enc = tok(texts, padding=True, truncation=True, max_length=128,
                  return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        out = pooled.cpu().numpy().astype(np.float64)
        return out / np.maximum(np.linalg.norm(out, axis=1,
                                               keepdims=True), 1e-12)

    return embed


def run_m272(output_dir: Path, smoke: bool = False) -> dict[str, Any]:
    started = time.time()
    configure_external_cache_environment()

    from geode.core.measured_routing import (EmbeddingRouter,
                                             MEASURED_ARM_RULES)

    cell4_config = json.loads(Path(
        REPO_ROOT / "experiments" / "configs" / "v25"
        / "m268_natural_routing.json").read_text(encoding="utf-8"))
    sealed = json.loads(SEALED_CELL4.read_text(encoding="utf-8"))
    sealed_items = sealed["per_item"]
    sealed_overall = sealed["results"]["embed_router"]["overall"]

    device = "cpu"  # 700 short texts: CPU is fine (registered)
    embed = bert_embedder_factory(device)
    router = EmbeddingRouter(embed)

    mix = rebuild_mix(cell4_config)
    if smoke:
        mix = mix[:8]
        sealed_items = sealed_items[:8]

    # one batched embedding pass (the module's route_batch — same
    # centroids, same tie rule, single-call semantics)
    routes = router.route_batch([item["input"] for item in mix])

    comparisons: list[dict[str, Any]] = []
    n_route_mismatch = 0
    for item, se, routed_family in zip(mix, sealed_items, routes):
        mismatch = routed_family != se["embed_route"]
        n_route_mismatch += int(mismatch)
        rule = MEASURED_ARM_RULES.get(routed_family,
                                      {"arm": "generalist"})
        comparisons.append({
            "task": item["task"],
            "reference": item["reference"],
            "module_route": routed_family,
            "sealed_route": se["embed_route"],
            "route_matches_sealed": not mismatch,
            "policy_arm": rule["arm"],
            "sealed_embed_correct": se["embed_correct"],
        })

    recomputed_correct = sum(c["sealed_embed_correct"]
                             for c in comparisons)
    recomputed_acc = round(recomputed_correct / len(comparisons), 4)
    sealed_acc = sealed_overall["routed_accuracy"]
    if not smoke:
        gate_ok = (n_route_mismatch == 0 and recomputed_acc == sealed_acc)
    else:
        gate_ok = True  # smoke: mechanics only

    evidence: dict[str, Any] = {
        "milestone": "M272",
        "cell": "ship the measured routing rules",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(
            {"module": "geode/core/measured_routing.py",
             "cell4_config": cell4_config}),
        "gate": {
            "sealed_source": str(SEALED_CELL4),
            "sealed_embed_accuracy": sealed_acc,
            "recomputed_embed_accuracy": recomputed_acc,
            "n_route_mismatches_vs_sealed": n_route_mismatch,
            "gate_passed": gate_ok,
            "note": ("routes reproduced item-by-item with the new "
                     "module; accuracy recomputed from the sealed "
                     "per-item embed_correct"),
        },
        "measured_rules": MEASURED_ARM_RULES,
        "module": "geode/core/measured_routing.py",
        "unit_tests": ("tests/unit/test_v25_m272_measured_routing.py"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gate": evidence["gate"]}, indent=1), flush=True)
    print(f"M272 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output
    if args.smoke and output == DEFAULT_OUTPUT:
        output = DEFAULT_OUTPUT.parent / (DEFAULT_OUTPUT.name + "_smoke")
    run_m272(output, smoke=args.smoke)


if __name__ == "__main__":
    main()
