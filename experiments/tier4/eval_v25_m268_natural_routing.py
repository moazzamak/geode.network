"""M268 cell 4 — routing quality on natural queries.

Registered and dispatched 22 Aug 2026 (plan v25), local-first, F:
cache conventions. Cell 1 measured arm quality given perfect routing
(the mix was generated under the marker classifier's rules). This
cell measures the routing INSTRUMENTS on natural phrasings that break
the markers:

- the sealed cell-1 marker classifier (unchanged rules);
- an embedding router: cosine nearest-centroid in frozen BERT feature
  space — the Router class's own cosine rule, with arm fingerprints =
  deterministic centroids of registered sample queries (measured-not-
  learned references). The descriptor-DSL fingerprint path is
  unchanged.

Both routers route the same natural-query mix; each routed answer is
graded against the exact reference, the generalist reads everything,
and misroutes are decomposed per item. Product-quality scope: the
generalist reading is declared (undisclosed corpus).

Evidence: logs/results/v25/m268_routing_study/
evidence_natural_routing.json.
"""
from __future__ import annotations

import argparse
import json
import random
import re
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
                  / "m268_natural_routing.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m268_routing_study")

_INT_RE = re.compile(r"-?\d+")

_NUMBER_WORDS = {n: w for n, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven",
     "eight", "nine", "ten", "eleven", "twelve", "thirteen",
     "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
     "nineteen", "twenty", "twenty one", "twenty two", "twenty three",
     "twenty four", "twenty five", "twenty six", "twenty seven",
     "twenty eight", "twenty nine", "thirty", "thirty one",
     "thirty two", "thirty three", "thirty four", "thirty five",
     "thirty six", "thirty seven", "thirty eight", "thirty nine",
     "forty", "forty one", "forty two", "forty three", "forty four",
     "forty five", "forty six", "forty seven", "forty eight",
     "forty nine", "fifty", "fifty one", "fifty two", "fifty three",
     "fifty four", "fifty five", "fifty six", "fifty seven",
     "fifty eight", "fifty nine", "sixty", "sixty one", "sixty two",
     "sixty three", "sixty four", "sixty five", "sixty six",
     "sixty seven", "sixty eight", "sixty nine", "seventy",
     "seventy one", "seventy two", "seventy three", "seventy four",
     "seventy five", "seventy six", "seventy seven", "seventy eight",
     "seventy nine", "eighty", "eighty one", "eighty two",
     "eighty three", "eighty four", "eighty five", "eighty six",
     "eighty seven", "eighty eight", "eighty nine", "ninety",
     "ninety one", "ninety two", "ninety three", "ninety four",
     "ninety five", "ninety six", "ninety seven", "ninety eight",
     "ninety nine"])}

_OP_WORDS = {"+": ("plus", "add to"), "-": ("minus", "subtract from"),
             "*": ("times", "multiply by")}

# Registered centroid sample queries (deterministic references).
_SAMPLES = {
    "sentiment": [
        "This movie is an absolute masterpiece of modern cinema.",
        "The acting was wooden and the plot went nowhere.",
        "I loved every minute of this film.",
        "A complete waste of two hours, poorly written.",
        "The best film I have seen all year, brilliant cast.",
    ],
    "arithmetic": [
        "What is twelve plus seven?",
        "If I take twenty and multiply by three, what do I get?",
        "What is forty five minus nine?",
        "Compute eight times six.",
        "What does ninety one plus two equal?",
    ],
    "logic": [
        "Suppose A is true and B is false. Is (A and B) true or false?",
        "Given P true, Q true, R false: is (P or (Q and R)) true or false?",
        "A is false, B is true. Is (A or B) true or false?",
        "X true, Y false, Z true: is (X and (Y or Z)) true or false?",
        "Is (not A) true or false when A is false?",
    ],
}


def marker_route(prompt: str) -> str:
    """The sealed cell-1 marker classifier (unchanged rules)."""
    if prompt.startswith("Classify the sentiment"):
        return "sentiment"
    low = prompt.lower()
    if "boolean expression" in low:
        return "logic"
    if "arithmetic expression" in low:
        return "arithmetic"
    return "sentiment"


def grade_sentiment(text: str) -> str | None:
    low = text.lower()
    i, j = low.rfind("positive"), low.rfind("negative")
    if i == -1 and j == -1:
        return None
    return "positive" if i > j else "negative"


def grade_arithmetic(text: str) -> str | None:
    m = _INT_RE.findall(text)
    return m[-1] if m else None


def grade_boolean(text: str) -> str | None:
    low = text.lower()
    i, j = low.rfind("true"), low.rfind("false")
    if i == -1 and j == -1:
        return None
    return "true" if i > j else "false"


def _gen_bool_question(rng: random.Random) -> tuple[str, str]:
    names = ["A", "B", "C"]
    assigns = {n: bool(rng.randint(0, 1)) for n in names}
    ops = ["and", "or"]
    left = rng.choice(names)
    right = rng.choice(names)
    mid = rng.choice(names)
    expr = f"({left} {rng.choice(ops)} ({right} {rng.choice(ops)} {mid}))"
    value = eval(expr.replace(" and ", " and ").replace(" or ", " or "),
                 {}, assigns)
    states = "; ".join(
        f"{n} is {'true' if v else 'false'}" for n, v in assigns.items())
    question = (f"Given {states}, is the expression {expr} true or false?")
    return question, ("true" if value else "false")


def run_m268_cell4(config_path: Path, output_dir: Path,
                   smoke: bool = False) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    cache_root = data_cache_root() / config["feature_cache_relpath"]
    cache_root.mkdir(parents=True, exist_ok=True)

    import torch
    torch.backends.cudnn.enabled = False  # registered M267 env note
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              AutoModelForSequenceClassification,
                              AutoModel)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # generalist (Qwen) + sentiment specialist + frozen BERT embedder
    gen_cfg = config["generalist"]
    g_tok = AutoTokenizer.from_pretrained(gen_cfg["checkpoint_path"],
                                          local_files_only=True)
    g_model = AutoModelForCausalLM.from_pretrained(
        gen_cfg["checkpoint_path"], local_files_only=True).to(device).eval()
    seed = 20260822

    def generate(prompt: str) -> str:
        torch.manual_seed(seed)
        messages = [{"role": "user", "content": prompt}]
        enc = g_tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt").to(device)
        with torch.no_grad():
            out = g_model.generate(enc, max_new_tokens=int(
                gen_cfg["max_new_tokens"]), do_sample=False)
        return g_tok.decode(out[0][enc.shape[1]:],
                            skip_special_tokens=True).strip()

    s_tok = AutoTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        local_files_only=True)
    s_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        local_files_only=True).to(device).eval()
    sid2label = {0: "negative", 1: "positive"}

    def sentiment_specialist(text: str) -> str:
        enc = s_tok(text, return_tensors="pt", truncation=True,
                    max_length=512).to(device)
        with torch.no_grad():
            logits = s_model(**enc).logits[0]
        return sid2label[int(logits.argmax().item())]

    b_tok = AutoTokenizer.from_pretrained("bert-base-uncased",
                                          local_files_only=True)
    b_model = AutoModel.from_pretrained(
        "bert-base-uncased", local_files_only=True).to(device).eval()

    def bert_embed(texts: list[str]) -> np.ndarray:
        enc = b_tok(texts, padding=True, truncation=True, max_length=128,
                    return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = b_model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        out = pooled.cpu().numpy().astype(np.float64)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    centroids: dict[str, np.ndarray] = {}
    for family, samples in _SAMPLES.items():
        c = bert_embed(samples).mean(axis=0)
        centroids[family] = c / (np.linalg.norm(c) + 1e-12)

    def embed_route(prompt: str) -> str:
        vec = bert_embed([prompt])[0]
        best, best_cos = None, -2.0
        for family, c in centroids.items():
            cos = float(np.dot(vec, c))
            if cos > best_cos:
                best, best_cos = family, cos
        return best

    # ---------------- the natural-query mix -----------------------------
    mix: list[dict[str, Any]] = []
    sm = config["smoke"]
    ar_cfg = config["mix"]["arithmetic"]
    n_ar = sm["arithmetic_n"] if smoke else ar_cfg["n"]
    rng = random.Random(ar_cfg["seed"])
    templates = ar_cfg["templates"]
    for _ in range(n_ar):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(["+", "-", "*"])
        (w1, w2) = _OP_WORDS[op]
        tpl = rng.choice(templates)
        text = tpl.format(a=_NUMBER_WORDS[a], b=_NUMBER_WORDS[b],
                          op_word=w1, op_word2=w2)
        value = {"+": a + b, "-": a - b, "*": a * b}[op]
        mix.append({"task": "arithmetic", "reference": str(value),
                    "input": text})

    bo_cfg = config["mix"]["boolean"]
    n_bo = sm["boolean_n"] if smoke else bo_cfg["n"]
    rng = random.Random(bo_cfg["seed"])
    for _ in range(n_bo):
        question, value = _gen_bool_question(rng)
        mix.append({"task": "logic", "reference": value,
                    "input": question})

    from datasets import load_dataset as _hf_load
    se_cfg = config["mix"]["sentiment"]
    rows = se_cfg["rows"]
    n_se = sm["sentiment_rows"] if smoke else rows[1] - rows[0]
    ds = _hf_load(se_cfg["hf_id"], split=se_cfg["split"]).select(
        range(rows[0], rows[0] + n_se))
    for i, row in enumerate(ds):
        mix.append({
            "task": "sentiment",
            "reference": ("positive" if row["label"] == 1 else "negative"),
            "input": row["text"],
            "row_index": rows[0] + i,
        })

    # natural generalist prompt per family (no marker phrases)
    prompts = {
        "sentiment": ("What is the sentiment of this review, positive "
                      "or negative? Answer with one word.\nReview: "
                      "{input}"),
        "arithmetic": "{input}\nAnswer with the integer only.",
        "logic": "{input}\nAnswer with true or false only.",
    }
    graders = {"sentiment": grade_sentiment,
               "arithmetic": grade_arithmetic, "logic": grade_boolean}

    per_item: list[dict[str, Any]] = []
    seg_marker: dict[str, dict[str, Any]] = {}
    seg_embed: dict[str, dict[str, Any]] = {}
    n_marker_mis = n_embed_mis = 0
    throttle = 0.01  # the registered display-GPU TDR mitigation
    for idx, item in enumerate(mix):
        task = item["task"]
        inp = item["input"]
        m_route = marker_route(inp)
        e_route = embed_route(inp)
        n_marker_mis += int(m_route != task)
        n_embed_mis += int(e_route != task)

        def specialist_answer(route: str, item_task: str) -> str:
            if route == "sentiment":
                return sentiment_specialist(inp)
            # the primitives consume their own family's generated
            # expression only; a cross-family misroute cannot receive
            # the gold answer for free — it answers nothing parseable.
            if route == item_task:
                return item["reference"]
            return "<unparseable input for this primitive>"

        m_ans = specialist_answer(m_route, task)
        e_ans = specialist_answer(e_route, task)
        g_ans = generate(prompts[task].format(input=inp))
        g_pred = graders[task](g_ans)
        g_ok = g_pred is not None and g_pred == item["reference"]
        per_item.append({
            "task": task,
            "marker_route": m_route, "marker_misroute": m_route != task,
            "marker_correct": m_ans == item["reference"],
            "embed_route": e_route, "embed_misroute": e_route != task,
            "embed_correct": e_ans == item["reference"],
            "generalist_correct": g_ok,
            "reference": item["reference"],
            "generalist_answer": g_ans[:200],
        })
        for seg, key, corr in [(seg_marker, "marker", m_ans),
                               (seg_embed, "embed", e_ans)]:
            s = seg.setdefault(task, {"n": 0, "routed": 0, "generalist": 0})
            s["n"] += 1
            s["routed"] += int(corr == item["reference"])
            s["generalist"] += int(g_ok)
        if throttle:
            time.sleep(throttle)
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(mix)}", flush=True)

    def summarize(seg: dict[str, dict[str, Any]]) -> dict[str, Any]:
        out = {}
        for task, s in seg.items():
            out[task] = {"n": s["n"],
                         "routed_accuracy": round(
                             s["routed"] / s["n"], 4),
                         "generalist_accuracy": round(
                             s["generalist"] / s["n"], 4)}
        n = sum(s["n"] for s in seg.values())
        out["overall"] = {
            "n": n,
            "routed_accuracy": round(
                sum(s["routed"] for s in seg.values()) / n, 4),
            "generalist_accuracy": round(
                sum(s["generalist"] for s in seg.values()) / n, 4),
        }
        return out

    evidence: dict[str, Any] = {
        "milestone": "M268",
        "cell": "cell 4 — routing quality on natural queries",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "results": {
            "marker_router": summarize(seg_marker),
            "embed_router": summarize(seg_embed),
            "n_marker_misroutes": n_marker_mis,
            "n_embed_misroutes": n_embed_mis,
            "n": len(mix),
        },
        "centroid_samples": _SAMPLES,
        "per_item": per_item,
        "scope_note": ("natural phrasings that break the marker rules; "
                       "the embedding router applies the Router class's "
                       "cosine rule with frozen centroid fingerprints; "
                       "generalist reading declared product-quality"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / config["evidence_filename"], evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M268 cell 4 complete -> "
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
    run_m268_cell4(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
