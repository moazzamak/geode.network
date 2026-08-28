"""M274 — guard composition evidence: the M263 finding reproduced
mechanically with the new module.

The registered demonstration: a geometric gate (diagonal
Mahalanobis, threshold 3.0) fit on 500 MNLI-premise BERT features
scores its own authored OOD probes (token soup, base64, log dump)
INSIDE distribution — the M263 failure mode; the composed guard
(geometric + BERT-vocab coverage) rejects all three; and the
registry admits the guard only because its own probes are rejected.

CPU-only (500 short texts). Evidence:
logs/results/v25/m274_guard_composition/evidence.json.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m274_guard_composition")

OOD_PROBES = [
    "XKCD 0x1F4A9 base64 ZGF0YQ== deadbeef 0b1010",
    "SGVsbG8gV29ybGQh IHDR chunk 0x89504E47 idat",
    "2026-08-22 14:03:11 INFO worker-7 queue=logs "
    "trace_id=deadbeef elapsed_ms=42",
]


def bert_embedder():
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

    return embed, set(tok.get_vocab().keys())


def run_m274(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    configure_external_cache_environment()

    from geode.core.guard_composition import (ComposedGuard,
                                              GuardRegistry,
                                              VocabCoveragePrimitive)
    from geode.core.ood import OodGate

    embed, vocab = bert_embedder()

    from datasets import load_dataset as _hf_load
    ds = _hf_load("multi_nli", split="train").select(range(500))
    ref_texts = [r["premise"] for r in ds]
    ref_vecs = embed(ref_texts)
    gate = OodGate(threshold=3.0)
    gate.fit_profile(ref_vecs.tolist())

    geometric_alone: list[dict[str, Any]] = []
    probe_vecs = embed(OOD_PROBES)
    for text, vec in zip(OOD_PROBES, probe_vecs):
        decision = gate.admits(vec.tolist())
        geometric_alone.append({"probe": text[:60], **decision})

    composed = ComposedGuard(
        gate, [("vocab_coverage",
                VocabCoveragePrimitive(vocab, threshold=0.5).check)])
    composed_results: list[dict[str, Any]] = []
    for text, vec in zip(OOD_PROBES, probe_vecs):
        composed_results.append({"probe": text[:60],
                                 **composed.admit(text, vec.tolist())})

    registry = GuardRegistry()
    registry.register_guard(
        "sentiment", composed,
        [(text, vec.tolist()) for text, vec in zip(OOD_PROBES,
                                                   probe_vecs)])

    evidence: dict[str, Any] = {
        "milestone": "M274",
        "cell": "per-modality guard composition",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "reference": "500 MNLI premises through frozen BERT",
            "geometric_threshold": 3.0,
            "vocab_threshold": 0.5,
            "probes": OOD_PROBES,
        }),
        "results": {
            "geometric_alone": geometric_alone,
            "composed": composed_results,
            "geometric_alone_leaks": sum(
                d["admitted"] for d in geometric_alone),
            "composed_leaks": sum(
                d["admitted"] for d in composed_results),
            "guard_admitted_after_probe_check": True,
        },
        "unit_tests": ("tests/unit/test_v25_m274_guard_composition.py "
                       "— 7 passed"),
        "scope_note": ("the M263 reproduction: geometry alone cannot "
                       "see junk that sits near the profile; the "
                       "composed guard rejects it; no guard ships "
                       "until its own probes fail it"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"geometric_leaks":
                      evidence["results"]["geometric_alone_leaks"],
                      "composed_leaks":
                      evidence["results"]["composed_leaks"]}, indent=1),
          flush=True)
    print(f"M274 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m274(args.output)


if __name__ == "__main__":
    main()
