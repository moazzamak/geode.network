"""M278 — cross-arm consistency as a weak verifier.

Registered 22 Aug 2026 (plan v25, the M272-M281 wave). Protocol: the
sentiment specialist labels a review; the NLI arm checks the
hypothesis "this review is {label}" against the review as premise;
CONTRADICTION escalates to the generalist's label, entailment/neutral
keep the specialist's label. Measured honestly: the weak-verifier
gain is reported even if ~0 — the registered M263 lesson.

Evidence: logs/results/v25/m278_weak_verifier/evidence.json.
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
                  / "m278_weak_verifier.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m278_weak_verifier")


def run_m278(config_path: Path, output_dir: Path,
             smoke: bool = False) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    cache_root = data_cache_root() / config["feature_cache_relpath"]
    cache_root.mkdir(parents=True, exist_ok=True)

    import torch
    torch.backends.cudnn.enabled = False  # registered M267 env note
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              AutoModelForSequenceClassification)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_classifier(ckpt: str):
        tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt, local_files_only=True).to(device).eval()
        return tok, model

    s_cfg = config["arms"]["sentiment_specialist"]
    s_tok, s_model = load_classifier(s_cfg["checkpoint"])
    sid2label = {int(k): v for k, v in s_cfg["id2label"].items()}
    n_cfg = config["arms"]["nli"]
    n_tok, n_model = load_classifier(n_cfg["checkpoint"])
    nid2label = {int(k): v for k, v in n_cfg["id2label"].items()}

    g_cfg = config["arms"]["generalist_escalation"]
    g_tok = AutoTokenizer.from_pretrained(g_cfg["checkpoint_path"],
                                          local_files_only=True)
    g_model = AutoModelForCausalLM.from_pretrained(
        g_cfg["checkpoint_path"], local_files_only=True).to(device).eval()
    seed = 20260822

    def classify(tok, model, text: str) -> int:
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=512).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[0]
        return int(logits.argmax().item())

    def generalist_label(text: str) -> str | None:
        torch.manual_seed(seed)
        prompt = ("Classify the sentiment of this movie review as "
                  "positive or negative. Answer with one word only.\n"
                  f"Review: {text}\nAnswer:")
        messages = [{"role": "user", "content": prompt}]
        enc = g_tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt").to(device)
        with torch.no_grad():
            out = g_model.generate(enc, max_new_tokens=int(
                g_cfg["max_new_tokens"]), do_sample=False)
        ans = g_tok.decode(out[0][enc.shape[1]:],
                           skip_special_tokens=True).lower()
        i, j = ans.rfind("positive"), ans.rfind("negative")
        if i == -1 and j == -1:
            return None
        return "positive" if i > j else "negative"

    from datasets import load_dataset as _hf_load
    mix = config["mix"]
    n_rows = config["smoke"]["rows"] if smoke else mix["rows"][1] - \
        mix["rows"][0]
    ds = _hf_load(mix["hf_id"], split=mix["split"]).select(
        range(mix["rows"][0], mix["rows"][0] + n_rows))

    per_item: list[dict[str, Any]] = []
    throttle = 0.01  # the registered display-GPU TDR mitigation
    for i, row in enumerate(ds):
        gold = "positive" if row["label"] == 1 else "negative"
        spec_label = sid2label[classify(s_tok, s_model, row["text"])]
        hypothesis = f"This review is {spec_label}."
        nli_out = nid2label[classify(n_tok, n_model,
                                     row["text"] + " [SEP] " +
                                     hypothesis)]
        if nli_out == "contradiction":
            final = generalist_label(row["text"]) or spec_label
            escalated = True
        else:
            final = spec_label
            escalated = False
        per_item.append({
            "row_index": mix["rows"][0] + i,
            "gold": gold,
            "specialist": spec_label,
            "specialist_correct": spec_label == gold,
            "nli_verdict": nli_out,
            "escalated": escalated,
            "final": final,
            "final_correct": final == gold,
        })
        if throttle:
            time.sleep(throttle)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_rows}", flush=True)

    n = len(per_item)
    spec_ok = sum(p["specialist_correct"] for p in per_item)
    final_ok = sum(p["final_correct"] for p in per_item)
    n_esc = sum(p["escalated"] for p in per_item)
    evidence: dict[str, Any] = {
        "milestone": "M278",
        "cell": "cross-arm consistency as a weak verifier",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "results": {
            "n": n,
            "specialist_accuracy": round(spec_ok / n, 4) if n else None,
            "final_accuracy_after_verifier": round(final_ok / n, 4)
            if n else None,
            "delta_from_verifier": round((final_ok - spec_ok) / n, 4)
            if n else None,
            "n_escalations": n_esc,
            "escalation_rate": round(n_esc / n, 4) if n else None,
        },
        "per_item": per_item,
        "scope_note": ("the weak-verifier gain is reported even if "
                       "~0; a weak instrument is measured, never "
                       "trusted"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / config["evidence_filename"], evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M278 complete -> "
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
    run_m278(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
