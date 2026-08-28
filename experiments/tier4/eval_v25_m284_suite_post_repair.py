"""M284 — post-repair suite re-measurement: the M283 authored suite
evaluated through the SHIPPED policy (M284 claim pre-pass, then the
embedding router) instead of the raw embedding router. Records the
post-repair misroute rates against the same 0.10 bar.

CPU-only. Evidence:
logs/results/v25/m284_claim_route/evidence_suite_post_repair.json.
"""
from __future__ import annotations

import json
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
from geode.core.claim_route import claim_answer
from geode.core.measured_routing import FAMILY_ORDER, FAMILY_SAMPLES
from geode.core.router_probes import RouterProbeSuite

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (REPO_ROOT / "logs" / "results" / "v25"
              / "m284_claim_route")


def run_post_repair() -> dict[str, Any]:
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
        """The shipped policy: the M284 pre-pass first, then the
        embedding router (the route_policy family decision)."""
        claim = claim_answer(text)
        if claim is not None:
            return "logic"
        vec = embed([text])[0]
        best, best_cos = None, -2.0
        for family in FAMILY_ORDER:
            cos = float(np.dot(vec, centroids[family]))
            if cos > best_cos:
                best, best_cos = family, cos
        return best

    suite = RouterProbeSuite()
    evaluation = suite.evaluate(shipped_policy_route)
    evidence: dict[str, Any] = {
        "milestone": "M284",
        "cell": "post-repair suite re-measurement",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "policy": "M284 claim pre-pass -> embedding router",
            "suite": "analysis/router_probes_v0.json",
            "bar": suite.bar,
        }),
        "results": evaluation,
        "scope_note": ("the same authored suite measured through the "
                       "SHIPPED policy (pre-pass first); the pre-"
                       "repair reading was 0.1875 overall with the "
                       "contract_spoof class at 0.50"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_canonical_json(OUTPUT_DIR / "evidence_suite_post_repair.json",
                         evidence)
    build_artifact_index(OUTPUT_DIR)
    print(json.dumps({"results": evaluation}, indent=1), flush=True)
    print(f"post-repair complete -> "
          f"{OUTPUT_DIR / 'evidence_suite_post_repair.json'}", flush=True)
    return evidence


if __name__ == "__main__":
    run_post_repair()
