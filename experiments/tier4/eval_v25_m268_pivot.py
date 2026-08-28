"""M268 cell 2 — the pivot-first cross-representation chain:
English -> Chinese sentiment through frozen opus-mt-en-zh and the
cleared Erlangshen Chinese sentiment arm, graded against the English
gold labels.

Registered and dispatched 22 Aug 2026 (plan v25, amendment 32),
local-first, F: cache conventions. The registered pivot rule: ship
the pivot hub first; a pairwise specialist is admitted only where
the pivot fails its target. Measured here as the pivot-vs-direct
gap: distilbert-sst-2 on the original English vs Erlangshen on the
pivot-translated Chinese, same rows, same gold labels. Arm agreement
records whether sentiment survives translation.

Licensing: opus-mt-en-zh Apache-2.0; Erlangshen Apache-2.0;
distilbert-sst-2 Apache-2.0; IMDb evaluation-only (standing
research-class). Reviews truncated to the Marian 512-token limit —
truncation count recorded, not hidden.

Evidence: logs/results/v25/m268_routing_study/evidence_pivot.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
                  / "m268_pivot.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m268_routing_study")


def run_m268_pivot(config_path: Path, output_dir: Path,
                   smoke: bool = False) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    cache_root = data_cache_root() / config["feature_cache_relpath"]
    cache_root.mkdir(parents=True, exist_ok=True)

    import torch
    torch.backends.cudnn.enabled = False  # registered M267 env note
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              AutoModelForSequenceClassification)
    from datasets import load_dataset as _hf_load

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- the pivot hub: opus-mt-en-zh --------------------------------
    mt_cfg = config["arms"]["mt"]
    mt_tok = AutoTokenizer.from_pretrained(mt_cfg["checkpoint"])
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(
        mt_cfg["checkpoint"]).to(device).eval()

    def translate(text: str) -> tuple[str, bool]:
        enc = mt_tok(text, return_tensors="pt", truncation=True,
                     max_length=512).to(device)
        truncated = enc["input_ids"].shape[1] >= 511
        with torch.no_grad():
            out = mt_model.generate(**enc, num_beams=1)
        return mt_tok.decode(out[0], skip_special_tokens=True).strip(), \
            truncated

    # ---- the Chinese sentiment arm: Erlangshen -----------------------
    zh_cfg = config["arms"]["zh_sentiment"]
    zh_tok = AutoTokenizer.from_pretrained(zh_cfg["checkpoint"])
    zh_model = AutoModelForSequenceClassification.from_pretrained(
        zh_cfg["checkpoint"]).to(device).eval()
    zh_id2label = {int(k): v for k, v in zh_cfg["id2label"].items()}

    def zh_classify(text: str) -> str:
        enc = zh_tok(text, return_tensors="pt", truncation=True,
                     max_length=512).to(device)
        with torch.no_grad():
            logits = zh_model(**enc).logits[0]
        return zh_id2label[int(logits.argmax().item())]

    # ---- the English sentiment arm: distilbert-sst-2 -----------------
    en_cfg = config["arms"]["en_sentiment"]
    en_tok = AutoTokenizer.from_pretrained(en_cfg["checkpoint"])
    en_model = AutoModelForSequenceClassification.from_pretrained(
        en_cfg["checkpoint"]).to(device).eval()
    en_id2label = {int(k): v for k, v in en_cfg["id2label"].items()}

    def en_classify(text: str) -> str:
        enc = en_tok(text, return_tensors="pt", truncation=True,
                     max_length=512).to(device)
        with torch.no_grad():
            logits = en_model(**enc).logits[0]
        return en_id2label[int(logits.argmax().item())]

    # ---- the mix ------------------------------------------------------
    mix_cfg = config["mix"]
    rows = mix_cfg["rows"]
    n_rows = config["smoke"]["rows"] if smoke else rows[1] - rows[0]
    ds = _hf_load(mix_cfg["hf_id"], split=mix_cfg["split"]).select(
        range(rows[0], rows[0] + n_rows))

    per_item: list[dict[str, Any]] = []
    n_truncated = 0
    throttle = 0.01  # the registered display-GPU TDR mitigation
    for i, row in enumerate(ds):
        gold = "positive" if row["label"] == 1 else "negative"
        direct = en_classify(row["text"])
        zh_text, truncated = translate(row["text"])
        n_truncated += int(truncated)
        pivot = zh_classify(zh_text)
        per_item.append({
            "row_index": rows[0] + i,
            "gold": gold,
            "direct_label": direct, "direct_correct": direct == gold,
            "translation_truncated": truncated,
            "pivot_label": pivot, "pivot_correct": pivot == gold,
            "arm_agree": pivot == direct,
            "source": row["text"][:300],
            "translation": zh_text[:300],
        })
        if throttle:
            time.sleep(throttle)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_rows}", flush=True)

    n = len(per_item)
    direct_ok = sum(p["direct_correct"] for p in per_item)
    pivot_ok = sum(p["pivot_correct"] for p in per_item)
    agree = sum(p["arm_agree"] for p in per_item)
    evidence: dict[str, Any] = {
        "milestone": "M268",
        "cell": "cell 2 — pivot-first cross-representation chain (en->zh)",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "results": {
            "rows": [rows[0], rows[0] + n],
            "n": n,
            "direct_accuracy": round(direct_ok / n, 4) if n else None,
            "pivot_accuracy": round(pivot_ok / n, 4) if n else None,
            "arm_agreement": round(agree / n, 4) if n else None,
            "target_gap": round((direct_ok - pivot_ok) / n, 4)
            if n else None,
            "n_truncated_translations": n_truncated,
        },
        "sample_translations": [
            {"source": p["source"], "translation": p["translation"],
             "gold": p["gold"], "pivot_label": p["pivot_label"]}
            for p in per_item[:5]
        ],
        "per_item": per_item,
        "scope_note": ("the pivot is the hub: a pairwise en-zh "
                       "specialist is admitted only where this gap "
                       "fails the registered target; no SOTA claim"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / config["evidence_filename"], evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M268 cell 2 complete -> "
          f"{output_dir / config['evidence_filename']}", flush=True)
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
    run_m268_pivot(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
