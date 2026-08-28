"""M284b — the verdict-spoof repair: evidence run.

g1: the v1 authored suite (8 verdict-spoof probes) reads 0/8
misroutes through the shipped policy (claim pre-pass -> verdict
rule -> embedding router).
g2 REGRESSION: the v0 authored suite may only IMPROVE (its
post-repair reading was 0.0625; the verdict rule must not create
new misroutes there).
g3: the M284 claim items are untouched — the claim pre-pass still
owns the 100 M281b confounders + the 2 grammar probes.

CPU-only. Evidence:
logs/results/v25/m284b_verdict_route/evidence.json.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.data_cache import configure_external_cache_environment
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v25_m281b_discriminating_router import (
    gen_confounders,
)
from geode.core.claim_route import claim_answer, detect_verdict
from geode.core.measured_routing import FAMILY_ORDER, FAMILY_SAMPLES
from geode.core.router_probes import RouterProbeSuite

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m284b_verdict_route")


def run_m284b(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    configure_external_cache_environment()

    import torch
    torch.backends.cudnn.enabled = False
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
        out = pooled.numpy().astype(np.float64)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    centroids: dict[str, np.ndarray] = {}
    for family in FAMILY_ORDER:
        c = embed(FAMILY_SAMPLES[family]).mean(axis=0)
        centroids[family] = c / (np.linalg.norm(c) + 1e-12)

    def shipped_policy_route(text: str) -> str:
        """The shipped policy: claim pre-pass -> verdict rule ->
        embedding router (the route_policy family decision)."""
        if claim_answer(text) is not None:
            return "logic"
        if detect_verdict(text):
            return "sentiment"
        vec = embed([text])[0]
        best, best_cos = None, -2.0
        for family in FAMILY_ORDER:
            cos = float(np.dot(vec, centroids[family]))
            if cos > best_cos:
                best, best_cos = family, cos
        return best

    suite_v0 = RouterProbeSuite()
    suite_v1 = RouterProbeSuite(REPO_ROOT / "analysis"
                                / "router_probes_v1.json")
    g1 = suite_v1.evaluate(shipped_policy_route)
    g2 = suite_v0.evaluate(shipped_policy_route)

    config = json.loads(Path(REPO_ROOT / "experiments" / "configs"
                             / "v25"
                             / "m281b_discriminating_router.json")
                        .read_text(encoding="utf-8"))
    confounders = gen_confounders(config["mix"]["confounder"],
                                  random.Random(
                                      config["mix"]["confounder"]
                                      ["seed"]))
    g3_rows = [{"text": c["input"], "reference": c["reference"]}
               for c in confounders]
    g3_rows += [{"text": p["text"], "reference": None}
                for p in suite_v0.probes.values()
                if p["category"] == "contract_spoof"
                and claim_answer(p["text"]) is not None]
    g3 = []
    for row in g3_rows:
        decision_family = shipped_policy_route(row["text"])
        answer = claim_answer(row["text"])
        ok = (decision_family == "logic" and answer is not None
              and (row["reference"] is None
                   or answer["answer"] == row["reference"]))
        g3.append({"text": row["text"][:80], "family": decision_family,
                   "ok": bool(ok)})

    g1_ok = g1["overall_misroute_rate"] == 0.0
    g2_ok = g2["overall_misroute_rate"] <= 0.0625
    g3_ok = all(r["ok"] for r in g3)
    results = {
        "g1_v1_suite": g1,
        "g1_ok": g1_ok,
        "g2_v0_suite": g2,
        "g2_ok": g2_ok,
        "g3_claim_items": len(g3),
        "g3_ok": g3_ok,
        "verdict": ("M284b PASS — the verdict-spoof class routes to "
                    "sentiment; no regression anywhere"
                    if (g1_ok and g2_ok and g3_ok) else
                    "M284b FAIL — see g1/g2/g3"),
    }
    evidence: dict[str, Any] = {
        "milestone": "M284b",
        "cell": "verdict-spoof routing repair",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "rule": "review noun AND true/false token -> sentiment, "
                    "after the claim pre-pass",
            "g1": "v1 suite 0/8",
            "g2": "v0 suite <= 0.0625 (may only improve)",
            "g3": "M284 claim items untouched",
        }),
        "results": results,
        "scope_note": ("licensed by the M283b 3-probe criterion; "
                       "deterministic; formal boolean expressions "
                       "never match (no review nouns)"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": results}, indent=1), flush=True)
    print(f"M284b complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


if __name__ == "__main__":
    run_m284b(DEFAULT_OUTPUT)
