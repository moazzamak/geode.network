"""M271 cell (d) — clean-axis transfer readings.

Cell 1: the SST-2 specialist on the IMDb test split (cross-domain
sentiment transfer — the checkpoint was never trained on IMDb).
Cell 2: the MNLI specialist on SNLI test (same genre, outside the
MoritzLaurer training mix). These are the first language-side
readings the benchmark-exposure catch cannot touch: the checkpoints
have never seen this data.

Registered and dispatched 21 Aug 2026 (plan v25, M271 cell (d) +
amendment 25), local-first, F: caches. Context anchors cited, never
gated on; one held-out read per cell.
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
                  / "m271_quality_ladder")


def run_transfer(output_dir: Path, smoke: bool = False) -> dict[str, Any]:
    started = time.time()
    configure_external_cache_environment()
    cache_root = data_cache_root() / "v25" / "m271_quality_ladder"
    cache_root.mkdir(parents=True, exist_ok=True)

    import torch
    torch.backends.cudnn.enabled = False  # registered env note (M267)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from datasets import load_dataset as _hf_load
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    def run_cell(name: str, ckpt: str, hf_id: str, split: str,
                 text_fn, label_fn, n_classes: int, batch: int,
                 context_anchor: float, exposure: str,
                 license_rec: str,
                 target_order: list[str] | None = None
                 ) -> dict[str, Any]:
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt).to(device).eval()
        target_ids = None
        if target_order is not None:
            name_to_id = {v.lower(): k
                          for k, v in model.config.id2label.items()}
            target_ids = [name_to_id[n] for n in target_order]
        ds = _hf_load(hf_id, split=split)
        cap_rows = 200 if smoke else None
        if cap_rows is not None:
            ds = ds.select(range(min(cap_rows, len(ds))))
        preds, refs = [], []
        for start in range(0, len(ds), batch):
            rows = ds.select(range(start, min(start + batch, len(ds))))
            enc = tok([text_fn(r) for r in rows], padding=True,
                      truncation=True, max_length=256,
                      return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            if target_ids is not None:
                logits = logits[:, target_ids]
            preds.extend(int(p) for p in logits.argmax(dim=1).tolist())
            refs.extend(int(label_fn(r)) for r in rows)
        acc = float((np.asarray(preds) == np.asarray(refs)).mean())
        print(f"  {name}: {acc:.4f} ({len(refs)} rows)", flush=True)
        return {"checkpoint": ckpt, "license": license_rec,
                "dataset": hf_id, "split": split,
                "accuracy": acc, "n_rows": len(refs),
                "context_anchor_source_task": context_anchor,
                "exposure": exposure}

    results: dict[str, Any] = {}
    # Cell 1: SST-2 specialist -> IMDb test (clean-axis transfer)
    results["sst2_to_imdb"] = run_cell(
        "sst2_to_imdb",
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "stanfordnlp/imdb", "test",
        text_fn=lambda r: r["text"], label_fn=lambda r: int(r["label"]),
        n_classes=2, batch=64, context_anchor=0.913,
        exposure=("clean-axis: the checkpoint was fine-tuned on "
                  "SST-2, never on IMDb"),
        license_rec="checkpoint Apache-2.0; IMDb evaluation-only")
    # Cell 2: MNLI specialist -> SNLI test (clean-axis transfer)
    results["mnli_to_snli"] = run_cell(
        "mnli_to_snli",
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "snli", "test",
        text_fn=lambda r: r["premise"] + " [SEP] " + r["hypothesis"],
        label_fn=lambda r: int(r["label"]),
        n_classes=3, batch=16, context_anchor=0.918,
        exposure=("clean-axis: the checkpoint's training mix "
                  "(MNLI/FEVER/ANLI/WANLI/ling) does not include "
                  "SNLI"),
        license_rec="checkpoint MIT; SNLI CC-BY-SA-4.0 tier-1 "
                    "evaluation only",
        target_order=["entailment", "neutral", "contradiction"])

    evidence: dict[str, Any] = {
        "milestone": "M271d",
        "cell": "clean-axis transfer readings",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "results": results,
        "verdict": {
            "reading": ("cross-domain/cross-corpus transfer — the "
                        "first language-side readings the benchmark-"
                        "exposure catch cannot touch; context anchors "
                        "cited, never gated on")
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence_transfer.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps(results, indent=1), flush=True)
    print(f"M271d complete -> {output_dir / 'evidence_transfer.json'}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output
    if args.smoke and output == DEFAULT_OUTPUT:
        output = DEFAULT_OUTPUT.parent / (DEFAULT_OUTPUT.name + "_smoke")
    run_transfer(output, smoke=args.smoke)


if __name__ == "__main__":
    main()
