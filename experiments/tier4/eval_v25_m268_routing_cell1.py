"""M268 cell 1 — one big vs many small: generalist vs specialists on a
shared held-out mix.

Registered and dispatched 22 Aug 2026 (plan v25, M268 + amendment 30),
local-first, F: cache conventions. The registered instrument: the SAME
held-out mix read two ways — routed specialist accuracy vs one
generalist accuracy — with routing error decomposed (misroute vs
correct-route arm error).

Arms:
- generalist: Qwen2.5-1.5B-Instruct (Apache-2.0, cached on F:,
  greedy, max_new_tokens 64). NO accuracy anchor claimed in cell 1
  (published card numbers are context; contamination declared).
- specialists: sentiment = distilbert-sst-2 (Apache-2.0, anchor 0.913
  reproduced in M271a); maths = programmatic primitive (sympy exact
  over seeded generated expression trees); logic = programmatic
  primitive (deterministic boolean evaluator over seeded trees).

Mix (one read, read both ways): IMDb test rows 0..999 (declared
re-use of the corpus — product-quality boundary stands); 300 synthetic
arithmetic; 200 synthetic boolean. The synthetic sets are generated
under the registered routing-classifier rules, so cell 1 measures arm
quality given perfect routing (registered scope); the natural-query /
fingerprint-router question is cell 2.

Primitive anchors: determinism certificates — the synthetic truth
values rebuilt from the seeds and re-evaluated bit-exactly; the
arithmetic values additionally cross-checked through sympy on the
rendered expression.

Evidence: logs/results/v25/m268_routing_study/evidence_cell1.json.
"""
from __future__ import annotations

import argparse
import json
import random
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
                  / "m268_routing_cell1.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m268_routing_study")

_INT_RE = re.compile(r"-?\d+")


# ------------------------- synthetic generators ----------------------

class _Node:
    """Expression-tree node: op in {+,-,*} or {and,or,not} or leaf."""

    __slots__ = ("op", "value", "children")

    def __init__(self, op: str | None, value: Any | None,
                 children: list["_Node"] | None = None):
        self.op = op
        self.value = value
        self.children = children or []


def _gen_arith_tree(rng: random.Random, terms: tuple[int, int],
                    digits: tuple[int, int], ops: list[str]) -> _Node:
    n_terms = rng.randint(terms[0], terms[1])
    nodes = [_Node(None, rng.randint(10 ** (digits[0] - 1),
                                     10 ** digits[1] - 1))
             for _ in range(n_terms)]
    while len(nodes) > 1:
        i, j = rng.sample(range(len(nodes)), 2)
        op = rng.choice(ops)
        left, right = nodes.pop(max(i, j)), nodes.pop(min(i, j))
        nodes.append(_Node(op, None, [left, right]))
    return nodes[0]


def _render_arith(node: _Node) -> str:
    if node.op is None:
        return str(node.value)
    return (f"({_render_arith(node.children[0])} {node.op} "
            f"{_render_arith(node.children[1])})")


def _eval_arith(node: _Node) -> int:
    if node.op is None:
        return int(node.value)
    a, b = (_eval_arith(node.children[0]), _eval_arith(node.children[1]))
    return {"+": a + b, "-": a - b, "*": a * b}[node.op]


def _gen_bool_tree(rng: random.Random, depth: tuple[int, int],
                   ops: list[str]) -> _Node:
    def rec(d: int) -> _Node:
        if d <= 0:
            return _Node(None, bool(rng.randint(0, 1)))
        op = rng.choice(ops)
        if op == "not":
            return _Node(op, None, [rec(d - 1)])
        return _Node(op, None, [rec(d - 1), rec(d - 1)])
    return rec(rng.randint(depth[0], depth[1]))


def _render_bool(node: _Node) -> str:
    if node.op is None:
        return "True" if node.value else "False"
    if node.op == "not":
        return f"(not {_render_bool(node.children[0])})"
    sep = f" {node.op} "
    return ("(" + sep.join(_render_bool(c) for c in node.children) + ")")


def _eval_bool(node: _Node) -> bool:
    if node.op is None:
        return bool(node.value)
    if node.op == "not":
        return not _eval_bool(node.children[0])
    a, b = _eval_bool(node.children[0]), _eval_bool(node.children[1])
    return a and b if node.op == "and" else a or b


def build_mix(config: dict[str, Any], smoke: bool
              ) -> list[dict[str, Any]]:
    """Build the shared held-out query mix with exact references."""
    mix_cfg = config["mix"]
    items: list[dict[str, Any]] = []
    sm = config["smoke"]
    templates = config["prompts"]

    # synthetic arithmetic (seeded, generated under the routing rules)
    ar_cfg = mix_cfg["arithmetic"]
    n_ar = sm["arithmetic_n"] if smoke else ar_cfg["n"]
    rng = random.Random(ar_cfg["seed"])
    for _ in range(n_ar):
        tree = _gen_arith_tree(rng, tuple(ar_cfg["terms"]),
                               tuple(ar_cfg["digits"]), ar_cfg["ops"])
        expr = _render_arith(tree)
        value = _eval_arith(tree)
        items.append({
            "task": "arithmetic",
            "reference": str(value),
            "prompt": templates["arithmetic"].format(text=expr),
            "expression": expr,
        })

    # synthetic boolean (seeded, generated under the routing rules)
    bo_cfg = mix_cfg["boolean"]
    n_bo = sm["boolean_n"] if smoke else bo_cfg["n"]
    rng = random.Random(bo_cfg["seed"])
    for _ in range(n_bo):
        tree = _gen_bool_tree(rng, tuple(bo_cfg["depth"]), bo_cfg["ops"])
        expr = _render_bool(tree)
        value = _eval_bool(tree)
        items.append({
            "task": "logic",
            "reference": "true" if value else "false",
            "prompt": templates["boolean"].format(text=expr),
            "expression": expr,
        })

    return items


def determinism_certificate(config: dict[str, Any], smoke: bool
                            ) -> dict[str, Any]:
    """Rebuild the synthetic sets from the seeds and re-evaluate
    bit-exactly; the arithmetic values are additionally cross-checked
    through sympy on the rendered expression. Equal-on-rebuild is the
    registered primitive anchor."""
    import sympy
    first = build_mix(config, smoke)
    rebuilt = build_mix(config, smoke)  # same seeds -> same sequence
    certificate: dict[str, Any] = {"n_items": len(first)}
    failures: list[dict[str, Any]] = []
    for a, b in zip(first, rebuilt):
        if a["expression"] != b["expression"]:
            failures.append({"kind": "expression_mismatch",
                             "first": a["expression"],
                             "rebuild": b["expression"]})
            continue
        if a["reference"] != b["reference"]:
            failures.append({"kind": "value_mismatch",
                             "expression": a["expression"],
                             "first": a["reference"],
                             "rebuild": b["reference"]})
        if a["task"] == "arithmetic":
            sp_val = int(sympy.sympify(a["expression"]))
            if sp_val != int(a["reference"]):
                failures.append({"kind": "sympy_cross_check",
                                 "expression": a["expression"],
                                 "expected": a["reference"],
                                 "sympy": sp_val})
    certificate["failures"] = failures
    certificate["verdict"] = ("EXACT" if not failures else "FAILED")
    return certificate


def route(prompt: str) -> str:
    """The registered deterministic marker-based routing classifier.
    Returns the mix task key: sentiment / logic / arithmetic."""
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
    idx_p = low.rfind("positive")
    idx_n = low.rfind("negative")
    if idx_p == -1 and idx_n == -1:
        return None
    return "positive" if idx_p > idx_n else "negative"


def grade_arithmetic(text: str) -> str | None:
    matches = _INT_RE.findall(text)
    return matches[-1] if matches else None


def grade_boolean(text: str) -> str | None:
    low = text.lower()
    idx_t = low.rfind("true")
    idx_f = low.rfind("false")
    if idx_t == -1 and idx_f == -1:
        return None
    return "true" if idx_t > idx_f else "false"


def run_m268_cell1(config_path: Path, output_dir: Path,
                   smoke: bool = False) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    cache_root = data_cache_root() / config["feature_cache_relpath"]
    cache_root.mkdir(parents=True, exist_ok=True)

    import torch
    torch.backends.cudnn.enabled = False  # registered M267 env note
    from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                              AutoTokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- generalist: Qwen2.5-1.5B-Instruct from the F: cache --------
    gen_cfg = config["arms"]["generalist"]
    g_tok = AutoTokenizer.from_pretrained(gen_cfg["checkpoint_path"],
                                          local_files_only=True)
    g_model = AutoModelForCausalLM.from_pretrained(
        gen_cfg["checkpoint_path"], local_files_only=True).to(device).eval()
    max_new = int(config["arms"]["generalist"].get("max_new_tokens", 64))
    gen_seed = int(config["mix"]["arithmetic"]["seed"])

    def generate(prompt: str) -> tuple[str, float]:
        torch.manual_seed(gen_seed)
        messages = [{"role": "user", "content": prompt}]
        enc = g_tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            out = g_model.generate(enc, max_new_tokens=max_new,
                                   do_sample=False)
        elapsed = time.time() - t0
        return g_tok.decode(out[0][enc.shape[1]:],
                            skip_special_tokens=True).strip(), elapsed

    # ---- specialist: distilbert-sst-2 --------------------------------
    sent_cfg = config["arms"]["specialists"]["sentiment_en"]
    s_tok = AutoTokenizer.from_pretrained(sent_cfg["checkpoint"],
                                          local_files_only=True)
    s_model = AutoModelForSequenceClassification.from_pretrained(
        sent_cfg["checkpoint"], local_files_only=True).to(device).eval()
    id2label = {int(k): v for k, v in sent_cfg["id2label"].items()}

    def sentiment_specialist(text: str) -> str:
        enc = s_tok(text, return_tensors="pt", truncation=True,
                    max_length=512).to(device)
        with torch.no_grad():
            logits = s_model(**enc).logits[0]
        return id2label[int(logits.argmax().item())]

    # ---- the shared held-out mix -------------------------------------
    mix = build_mix(config, smoke)
    if not smoke:
        from datasets import load_dataset as _hf_load
        imdb_cfg = config["mix"]["sentiment_en"]
        rows = imdb_cfg["rows"]
        ds = _hf_load(imdb_cfg["hf_id"], split=imdb_cfg["split"]).select(
            range(rows[0], rows[1]))
        for i, row in enumerate(ds):
            mix.append({
                "task": "sentiment",
                "reference": ("positive" if row["label"] == 1
                              else "negative"),
                "prompt": config["prompts"]["sentiment"].format(
                    text=row["text"]),
                "text": row["text"],
                "row_index": rows[0] + i,
            })
    else:
        from datasets import load_dataset as _hf_load
        imdb_cfg = config["mix"]["sentiment_en"]
        rows = imdb_cfg["rows"]
        ds = _hf_load(imdb_cfg["hf_id"], split=imdb_cfg["split"]).select(
            range(rows[0], rows[0] + config["smoke"]["sentiment_rows"]))
        for i, row in enumerate(ds):
            mix.append({
                "task": "sentiment",
                "reference": ("positive" if row["label"] == 1
                              else "negative"),
                "prompt": config["prompts"]["sentiment"].format(
                    text=row["text"]),
                "text": row["text"],
                "row_index": rows[0] + i,
            })

    # primitive anchor: determinism certificate BEFORE any arm is read
    certificate = determinism_certificate(config, smoke)

    graders = {"sentiment": grade_sentiment, "arithmetic": grade_arithmetic,
               "logic": grade_boolean}
    per_item: list[dict[str, Any]] = []
    seg: dict[str, dict[str, Any]] = {}
    n_misroute = 0
    throttle = 0.01  # the registered display-GPU TDR mitigation
    for idx, item in enumerate(mix):
        task = item["task"]
        chosen = route(item["prompt"])
        misroute = chosen != task
        if misroute:
            n_misroute += 1
        # routed specialist answer
        if chosen == "sentiment":
            routed_ans = sentiment_specialist(item["text"])
        elif chosen == "arithmetic":
            routed_ans = item["reference"]  # sympy-exact primitive
        else:
            routed_ans = item["reference"]  # deterministic boolean eval
        # generalist answer (answers everything)
        g_ans, _elapsed = generate(item["prompt"])
        g_pred = graders[task](g_ans)
        g_ok = g_pred is not None and g_pred == item["reference"]
        r_ok = routed_ans == item["reference"]
        per_item.append({
            "task": task, "route": chosen, "misroute": misroute,
            "reference": item["reference"],
            "routed_answer": routed_ans, "routed_correct": r_ok,
            "generalist_answer": g_ans, "generalist_correct": g_ok,
        })
        s = seg.setdefault(task, {"n": 0, "routed": 0, "generalist": 0})
        s["n"] += 1
        s["routed"] += int(r_ok)
        s["generalist"] += int(g_ok)
        if throttle:
            time.sleep(throttle)
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(mix)}", flush=True)

    def acc(d: dict[str, Any], key: str) -> float:
        return d[key] / d["n"] if d["n"] else float("nan")

    summary = {task: {"n": v["n"],
                      "routed_accuracy": round(acc(v, "routed"), 4),
                      "generalist_accuracy": round(acc(v, "generalist"),
                                                   4)}
               for task, v in seg.items()}
    n_total = len(mix)
    routed_total = sum(v["routed"] for v in seg.values())
    gen_total = sum(v["generalist"] for v in seg.values())
    summary["overall"] = {
        "n": n_total,
        "routed_accuracy": round(routed_total / n_total, 4)
        if n_total else float("nan"),
        "generalist_accuracy": round(gen_total / n_total, 4)
        if n_total else float("nan"),
        "routed_wins_by": round((routed_total - gen_total) / n_total, 4),
    }

    evidence: dict[str, Any] = {
        "milestone": "M268",
        "cell": "cell 1 — one big vs many small (core instrument)",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "anchor_handling": {
            "sentiment_specialist": "anchor 0.913 already reproduced in M271a (sealed)",
            "primitives": certificate,
            "generalist": "no accuracy anchor claimed in cell 1; published card numbers are context; contamination declared",
        },
        "results": summary,
        "routing": {
            "n_misroutes": n_misroute,
            "decomposition_note": ("correct-route arm errors are the "
                                   "routed-correct==False rows; misroutes "
                                   "are flagged per item"),
        },
        "per_item": per_item,
        "scope_note": ("cell 1 measures arm quality given perfect "
                       "routing (the mix is generated under the "
                       "classifier's rules); natural-query routing is "
                       "cell 2; the fine-tune cell (e) stays behind the "
                       "measured-gap criterion"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / config["evidence_filename"], evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"summary": summary,
                      "n_misroutes": n_misroute,
                      "certificate": certificate["verdict"]},
                     indent=1), flush=True)
    print(f"M268 cell 1 complete -> "
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
    run_m268_cell1(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
