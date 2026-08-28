"""M283 — authored adversarial-router probe suite: evidence run.

Generates the artifact (commit hashes included), verifies integrity,
and measures the REAL embedding router (cosine nearest-centroid, the
sealed cell-4 rule) on the 16 authored probes. The measured per-
category misroute rates are the blocker cell's finding — a bar
breach is a recorded gap, never a study failure.

CPU-only. Evidence:
logs/results/v25/m283_router_probes/evidence.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from experiments.common.data_cache import configure_external_cache_environment
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v25_m268_natural_routing import _SAMPLES
from experiments.tier4.gen_router_probes import main as gen_artifact
from geode.core.router_probes import RouterProbeSuite

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "analysis" / "router_probes_v0.json"
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m283_router_probes")


def run_m283(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    gen_artifact()  # regenerate with the hashes (idempotent)
    configure_external_cache_environment()

    suite = RouterProbeSuite(ARTIFACT)
    integrity = suite.verify_integrity()
    assert integrity["ok"], integrity

    import numpy as np
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
    for family, samples in _SAMPLES.items():
        c = embed(samples).mean(axis=0)
        centroids[family] = c / (np.linalg.norm(c) + 1e-12)

    def embed_route(text: str) -> str:
        vec = embed([text])[0]
        best, best_cos = None, -2.0
        for family, c in centroids.items():
            cos = float(np.dot(vec, c))
            if cos > best_cos:
                best, best_cos = family, cos
        return best

    evaluation = suite.evaluate(embed_route)
    evidence: dict[str, Any] = {
        "milestone": "M283",
        "cell": "authored adversarial-router probe suite",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "artifact": "analysis/router_probes_v0.json",
            "router": "cosine nearest-centroid (the sealed cell-4 "
                      "rule, frozen BERT centroids)",
            "bar": suite.bar,
            "probe_ids": sorted(suite.probes.keys()),
        }),
        "integrity": integrity,
        "results": evaluation,
        "scope_note": ("authored probes, the M249/M252 pattern "
                       "applied to the ROUTER; the embedding router "
                       "is measured on fixed adversarial texts; a "
                       "bar breach is a recorded gap"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evaluation}, indent=1), flush=True)
    print(f"M283 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


if __name__ == "__main__":
    run_m283(DEFAULT_OUTPUT)
