"""M268 cell 1b — the IMDb sentiment arm upgrade: the Qwen generalist
full-scale read.

Registered and dispatched 22 Aug 2026 (plan v25, the low-accuracy
flag list + amendment 32), local-first, F: cache conventions.

Why this cell exists: the sealed M262 IMDb arm reads 0.8282 on the
frozen-probe rung. M268 cell 1 measured the Qwen2.5-1.5B-Instruct
generalist at 0.959 on IMDb rows 0..999 — the measured improvement.
This cell performs the arm-upgrade seal: a fresh single read of the
remaining 24,000 IMDb test rows (1000..24999) through the same
generalist with the identical prompt and grader; the sealed cell-1
1,000 rows are combined in the evidence for the full-split reading.

Declared: the reading is product-quality on a publisher checkpoint
with an undisclosed training corpus (the standing contamination
boundary); one held-out read per configuration; no SOTA claim.

Evidence: logs/results/v25/m268_routing_study/evidence_cell1b.json.
"""
from __future__ import annotations

import argparse
import json
import re
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
                  / "m268_cell1b_imdb_arm.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m268_routing_study")


def grade_sentiment(text: str) -> str | None:
    low = text.lower()
    idx_p = low.rfind("positive")
    idx_n = low.rfind("negative")
    if idx_p == -1 and idx_n == -1:
        return None
    return "positive" if idx_p > idx_n else "negative"


def run_m268_cell1b(config_path: Path, output_dir: Path,
                    smoke: bool = False) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    cache_root = data_cache_root() / config["feature_cache_relpath"]
    cache_root.mkdir(parents=True, exist_ok=True)

    import torch
    torch.backends.cudnn.enabled = False  # registered M267 env note
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_cfg = config["arm"]
    tok = AutoTokenizer.from_pretrained(gen_cfg["checkpoint_path"],
                                        local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        gen_cfg["checkpoint_path"], local_files_only=True).to(device).eval()
    max_new = int(gen_cfg["max_new_tokens"])
    seed = int(config["mix"]["seed"])

    def generate(prompt: str) -> str:
        torch.manual_seed(seed)
        messages = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(enc, max_new_tokens=max_new,
                                 do_sample=False)
        return tok.decode(out[0][enc.shape[1]:],
                          skip_special_tokens=True).strip()

    from datasets import load_dataset as _hf_load
    mix_cfg = config["mix"]
    rows = mix_cfg["rows"]
    n_rows = (config["smoke"]["rows"] if smoke
              else rows[1] - rows[0])
    ds = _hf_load(mix_cfg["hf_id"], split=mix_cfg["split"]).select(
        range(rows[0], rows[0] + n_rows))

    per_item: list[dict[str, Any]] = []
    n_correct = 0
    n_graded = 0
    throttle = 0.01  # the registered display-GPU TDR mitigation
    for i, row in enumerate(ds):
        prompt = config["prompt"].format(text=row["text"])
        ans = generate(prompt)
        pred = grade_sentiment(ans)
        ref = "positive" if row["label"] == 1 else "negative"
        ok = pred == ref
        n_correct += int(ok)
        n_graded += int(pred is not None)
        per_item.append({"row_index": rows[0] + i, "reference": ref,
                         "prediction": pred, "correct": ok,
                         "answer": ans})
        if throttle:
            time.sleep(throttle)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_rows} running accuracy "
                  f"{n_correct / (i + 1):.4f}", flush=True)

    n = len(per_item)
    accuracy = n_correct / n if n else float("nan")
    prior = config["sealed_prior"]
    combined_ok = int(round(prior["accuracy"] * prior["n"])) + n_correct
    combined_n = prior["n"] + n
    evidence: dict[str, Any] = {
        "milestone": "M268",
        "cell": "cell 1b — IMDb sentiment arm upgrade (generalist full read)",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "results": {
            "rows": [rows[0], rows[0] + n],
            "n": n,
            "accuracy": round(accuracy, 4),
            "n_ungradable_answers": n - n_graded,
            "sealed_prior_cell1": prior,
            "combined_full_split": {
                "n": combined_n,
                "accuracy": round(combined_ok / combined_n, 4),
            },
        },
        "per_item": per_item,
        "declarations": {
            "contamination": ("product-quality reading on a publisher "
                              "checkpoint with an undisclosed training "
                              "corpus; no novelty or SOTA claim"),
            "holdout": "one held-out read per configuration (this read is new rows)",
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / config["evidence_filename"], evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M268 cell 1b complete -> "
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
    run_m268_cell1b(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
