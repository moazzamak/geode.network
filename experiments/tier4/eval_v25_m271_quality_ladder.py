"""M271 cell (a) — arm-quality ladder: task-specialized publisher
checkpoints for the sealed language and audio arms.

Cells: NLI (MoritzLaurer DeBERTa-v3-large-mnli-class, MIT),
SST-2 (distilbert-finetuned-sst-2, Apache-2.0), SCv2
(MIT/ast-finetuned-speech-commands-v2, BSD-3-Clause). IMDb
specialist EXCLUDED by the license gate (no metadata) — the sealed
M262 reading stands. Every cell: anchor first (published number
cited), one held-out read on the sealed splits, license fields,
guards. Expected numbers are hypotheses — measured, never assumed.
"""
from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m271_quality_ladder.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m271_quality_ladder")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_m271a(config_path: Path, output_dir: Path, smoke: bool = False
              ) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    cache_root = data_cache_root() / config["feature_cache_relpath"]
    cache_root.mkdir(parents=True, exist_ok=True)

    import torch
    torch.backends.cudnn.enabled = False  # registered env note (M267)
    device = config["generation"]["device"]
    if not torch.cuda.is_available():
        device = "cpu"
    batch = int(config["generation"]["batch"])
    sm = config["smoke"] if smoke else {}
    results: dict[str, Any] = {}

    from datasets import load_dataset as _hf_load
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    def cap(ds, n):
        return ds.select(range(min(n, len(ds)))) if n is not None else ds

    # --------------------------------------------------------------- NLI
    nli_cfg = config["cells"]["nli"]
    tok = AutoTokenizer.from_pretrained(nli_cfg["checkpoint"])
    model = AutoModelForSequenceClassification.from_pretrained(
        nli_cfg["checkpoint"]).to(device).eval()
    # MNLI label scheme: 0 entailment / 1 neutral / 2 contradiction
    name_to_id = {v.lower(): k for k, v in model.config.id2label.items()}
    mnli_name = ["entailment", "neutral", "contradiction"]
    target_id = [name_to_id[n] for n in mnli_name]
    nli_splits: dict[str, dict[str, Any]] = {}
    for split in ["validation_matched", "validation_mismatched"]:
        ds = _hf_load("multi_nli", split=split)
        ds = cap(ds, sm.get("nli_rows"))
        preds, refs = [], []
        for start in range(0, len(ds), batch):
            rows = ds.select(range(start, min(start + batch, len(ds))))
            enc = tok([r["premise"] for r in rows],
                      [r["hypothesis"] for r in rows],
                      padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            preds.extend(int(p) for p in
                         logits[:, target_id].argmax(dim=1).tolist())
            refs.extend(int(r["label"]) for r in rows)
        acc = float((np.asarray(preds) == np.asarray(refs)).mean())
        nli_splits[split] = {"accuracy": acc, "n_rows": len(refs)}
        print(f"  nli {split}: {acc:.4f} ({len(refs)} rows)", flush=True)
    results["nli"] = {
        "checkpoint": nli_cfg["checkpoint"],
        "license": nli_cfg["license_recorded"],
        "published_anchor_mnli_m": nli_cfg["published_anchor_mnli_m"],
        "splits": nli_splits,
    }

    # -------------------------------------------------------------- SST-2
    sst_cfg = config["cells"]["sst2"]
    tok2 = AutoTokenizer.from_pretrained(sst_cfg["checkpoint"])
    model2 = AutoModelForSequenceClassification.from_pretrained(
        sst_cfg["checkpoint"]).to(device).eval()
    ds = _hf_load("glue", "sst2", split="validation")
    ds = cap(ds, sm.get("sst2_rows"))
    preds, refs = [], []
    for start in range(0, len(ds), batch):
        rows = ds.select(range(start, min(start + batch, len(ds))))
        enc = tok2([r["sentence"] for r in rows], padding=True,
                   truncation=True, max_length=128,
                   return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model2(**enc).logits
        preds.extend(int(p) for p in logits.argmax(dim=1).tolist())
        refs.extend(int(r["label"]) for r in rows)
    sst_acc = float((np.asarray(preds) == np.asarray(refs)).mean())
    print(f"  sst2: {sst_acc:.4f} ({len(refs)} rows)", flush=True)
    results["sst2"] = {
        "checkpoint": sst_cfg["checkpoint"],
        "license": sst_cfg["license_recorded"],
        "published_anchor": sst_cfg["published_anchor"],
        "accuracy": sst_acc,
        "n_rows": len(refs),
    }

    # --------------------------------------------------------------- SCv2
    scv2_cfg = config["cells"]["scv2"]
    import soundfile as sf
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForAudioClassification,
    )
    extractor = AutoFeatureExtractor.from_pretrained(scv2_cfg["checkpoint"])
    ast = AutoModelForAudioClassification.from_pretrained(
        scv2_cfg["checkpoint"]).to(device).eval()
    scv2_root = (data_cache_root().parents[1] / "cache" / "speech_commands"
                 / "v0.02")
    testing_entries = [line.strip() for line in
                       (scv2_root / "testing_list.txt").read_text()
                       .splitlines() if line.strip()]
    word_to_idx = {d.name: i for i, d in enumerate(
        sorted(p for p in scv2_root.iterdir()
               if p.is_dir() and p.name != "_background_noise_"))}
    paths = [(scv2_root / e, word_to_idx[e.split("/")[0]])
             for e in testing_entries]
    paths = [p for p in paths if p[0].exists()]
    if sm.get("scv2_rows") is not None:
        paths = paths[:sm["scv2_rows"]]
    preds, refs = [], []
    for start in range(0, len(paths), batch):
        chunk = paths[start:start + batch]
        arrays = []
        for path, _label in chunk:
            arr, sr = sf.read(path, dtype="float32", always_2d=False)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            arrays.append(arr)
        feats = extractor(arrays, sampling_rate=16000,
                          return_tensors="pt").to(device)
        with torch.no_grad():
            logits = ast(**feats).logits
        for row_idx, (path, true_idx) in enumerate(chunk):
            label_name = ast.config.id2label[int(
                logits[row_idx].argmax())]
            # map by NAME into the dataset's word index space (the
            # model's id2label order differs from the sorted words)
            preds.append(int(word_to_idx.get(label_name, -1)))
        refs.extend(true_idx for _path, true_idx in chunk)
    scv2_acc = float((np.asarray(preds) == np.asarray(refs)).mean())
    print(f"  scv2: {scv2_acc:.4f} ({len(refs)} rows)", flush=True)
    results["scv2"] = {
        "checkpoint": scv2_cfg["checkpoint"],
        "license": scv2_cfg["license_recorded"],
        "published_anchor": scv2_cfg["published_anchor"],
        "accuracy": scv2_acc,
        "n_rows": len(refs),
    }
    results["imdb"] = {
        "excluded_specialist": config["cells"]["imdb"]["excluded"],
        "standing_reading": 0.8282,
        "source": "sealed M262 evidence",
    }

    evidence: dict[str, Any] = {
        "milestone": "M271a",
        "cell": "arm-quality ladder cell (a): task-specialized "
                "publisher checkpoints",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "results": results,
        "verdict": {
            "reading": ("published anchors cited; our held-out "
                        "readings reported separately — hypotheses "
                        "measured, never assumed")
        },
        "scope_note": ("publisher checkpoints frozen, never trained "
                       "here; same held-out splits as the sealed "
                       "cells; permissive-only with license fields"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps(results, indent=1), flush=True)
    print(f"M271a complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output
    if args.smoke and output == DEFAULT_OUTPUT:
        output = DEFAULT_OUTPUT.parent / (DEFAULT_OUTPUT.name + "_smoke")
    run_m271a(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
