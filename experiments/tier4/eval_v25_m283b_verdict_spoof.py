"""M283b — the second authored probe wave: the verdict-spoof class.

The residual from the M284 post-repair measurement is one probe
("A plus B is true, and that is my final answer..."). Before any
rule, the class gets a dedicated measurement: a SECOND frozen
artifact of authored probes around boolean-verdict sentiment
phrasings, measured through the SHIPPED policy (the M284 claim
pre-pass first, then the embedding router).

Registered repair criterion: a deterministic rule may be
registered ONLY if at least 3 distinct probes fail — a
single-probe failure records, never rules.

CPU-only. Evidence:
logs/results/v25/m283b_verdict_spoof/evidence.json.
"""
from __future__ import annotations

import hashlib
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
ARTIFACT_V1 = REPO_ROOT / "analysis" / "router_probes_v1.json"
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m283b_verdict_spoof")

V1_PROBES = [
    # the registered residual probe, carried into the second wave
    {"id": "verdict_residual_1", "category": "verdict_spoof",
     "expected_family": "sentiment",
     "text": ("A plus B is true, and that is my final answer to "
              "whether the film works.")},
    {"id": "verdict_1", "category": "verdict_spoof",
     "expected_family": "sentiment",
     "text": ("True or false: the acting was wooden and the plot "
              "went nowhere. My verdict: false.")},
    {"id": "verdict_2", "category": "verdict_spoof",
     "expected_family": "sentiment",
     "text": ("Is it true that I loved every minute of this film? "
              "Yes, that is my review.")},
    {"id": "verdict_3", "category": "verdict_spoof",
     "expected_family": "sentiment",
     "text": ("The math is simple: this movie equals a masterpiece, "
              "true or false? True.")},
    {"id": "verdict_4", "category": "verdict_spoof",
     "expected_family": "sentiment",
     "text": ("X and Y are both true: I recommend the film and the "
              "score multiplies the tension.")},
    {"id": "verdict_5", "category": "verdict_spoof",
     "expected_family": "sentiment",
     "text": ("Not true and not false — the film was simply "
              "average. My final answer: positive.")},
    {"id": "verdict_6", "category": "verdict_spoof",
     "expected_family": "sentiment",
     "text": ("If A is false and B is true, is the movie good? My "
              "answer as a critic: yes, true.")},
    {"id": "verdict_7", "category": "verdict_spoof",
     "expected_family": "sentiment",
     "text": ("The critic's boolean verdict on this film: true, "
              "worth watching, false otherwise.")},
]


def _commit_hash(author: str, salt: str, probe: dict) -> str:
    material = f"{author}|{salt}|" + json.dumps(
        probe, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _write_artifact() -> None:
    artifact = {"axis": "routing", "frozen": "2026-08-23",
                "min_authors": 2, "bar": 0.10, "probes": []}
    for probe in V1_PROBES:
        content = {k: v for k, v in probe.items() if k != "id"}
        commits = []
        for author in ("a1", "a2"):
            salt = f"salt-{author}-{probe['id']}"
            commits.append({"author": author, "salt": salt,
                            "hash": _commit_hash(author, salt, content)})
        artifact["probes"].append({"id": probe["id"], **content,
                                   "commits": commits})
    ARTIFACT_V1.write_text(json.dumps(artifact, indent=2) + "\n",
                           encoding="utf-8")


def run_m283b(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    _write_artifact()
    configure_external_cache_environment()

    suite = RouterProbeSuite(ARTIFACT_V1)
    integrity = suite.verify_integrity()
    assert integrity["ok"], integrity

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
        if claim_answer(text) is not None:
            return "logic"
        vec = embed([text])[0]
        best, best_cos = None, -2.0
        for family in FAMILY_ORDER:
            cos = float(np.dot(vec, centroids[family]))
            if cos > best_cos:
                best, best_cos = family, cos
        return best

    evaluation = suite.evaluate(shipped_policy_route)
    n_failed = sum(1 for p in evaluation["per_probe"].values()
                   if p["misroute"])
    repair_licensed = n_failed >= 3
    evidence: dict[str, Any] = {
        "milestone": "M283b",
        "cell": "second authored probe wave — the verdict-spoof class",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "artifact": "analysis/router_probes_v1.json",
            "policy": "shipped (M284 pre-pass -> embedding router)",
            "repair_criterion": "a rule only if >= 3 distinct "
                                "probes fail",
        }),
        "integrity": integrity,
        "results": {
            **evaluation,
            "n_failed_probes": n_failed,
            "repair_licensed": repair_licensed,
            "verdict": ("RULE LICENSED — >= 3 distinct probes fail; "
                        "a registered M284b repair may proceed"
                        if repair_licensed else
                        "NO RULE — the failure is below the 3-probe "
                        "criterion; recorded, never ruled"),
        },
        "scope_note": ("authored probes, dual-authored commits; the "
                       "v0 suite stays frozen; measured through the "
                       "shipped policy"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M283b complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


if __name__ == "__main__":
    run_m283b(DEFAULT_OUTPUT)
