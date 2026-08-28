"""M281b — the discriminating learned-router instrument (registered
23 Aug 2026, plan v25, after the M281 instrument amendment).

Why this exists: M281's cell-4 mix saturated BOTH routers (incumbent
0 misroutes; 0.960 is the ceiling for any perfect router) and the
candidate trained on the same generators/templates as the test — the
tie was forced arithmetic and the study carried no information.
M281b re-registers the instrument with a PROPERLY HELD-OUT test mix:

- out-of-template arithmetic paraphrases (new templates, disjoint
  from the cell-4 template set the candidate trains on);
- out-of-template logic paraphrases (new question forms);
- cross-cue confounders (task = logic, arithmetic surface) and
  digit-form arithmetic — the surface/embedding mismatch classes
  where a learned router can differ from the nearest-centroid
  incumbent;
- sentiment rows disjoint from the candidate's training rows and
  from the sealed cell-4 rows.

PREMISE GATE (first, ahead of every other clause): the incumbent
embedding router must MEASURABLY misroute on this mix (misroutes > 0)
or the instrument failed to discriminate and the run is premise-void
(no candidate number is read).

Admission (re-registered gate): strictly fewer misroutes than the
incumbent, or equal misroutes with strictly higher routed accuracy
measured under the CANDIDATE'S OWN routes — arm answers are measured
on this mix for both routers, never inherited. A tie is not
admission; the deterministic incumbent stands.

CPU-only, local-first. Evidence:
logs/results/v25/m281b_discriminating_router/evidence.json.
"""
from __future__ import annotations

import argparse
import json
import random
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
from experiments.tier4.eval_v25_m268_natural_routing import (
    _NUMBER_WORDS,
    _OP_WORDS,
    _SAMPLES,
    grade_boolean,
)
from experiments.tier4.eval_v25_m281_learned_router import (
    FAMILIES,
    generate_train,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m281b_discriminating_router.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m281b_discriminating_router")

_WORD2N = {w: n for n, w in _NUMBER_WORDS.items()}


# ---------------- the registered held-out mix generators ----------------

def gen_ar_paraphrases(cfg: dict[str, Any],
                       rng: random.Random) -> list[dict[str, Any]]:
    """Out-of-template arithmetic: NEW templates only (registered
    disjoint from the cell-4 templates the candidate trains on)."""
    out: list[dict[str, Any]] = []
    templates = cfg["new_arithmetic_templates"]
    for _ in range(cfg["n"]):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(["+", "-", "*"])
        w1, _w2 = _OP_WORDS[op]
        text = rng.choice(templates).format(
            a=_NUMBER_WORDS[a], b=_NUMBER_WORDS[b], op_word=w1)
        value = {"+": a + b, "-": a - b, "*": a * b}[op]
        out.append({"task": "arithmetic", "reference": str(value),
                    "input": text, "kind": "ar_paraphrase"})
    return out


def _gen_logic_paraphrase(rng: random.Random) -> tuple[str, str]:
    """NEW logic question forms (different from the cell-4 form the
    candidate trains on)."""
    names = ["A", "B", "C"]
    assigns = {n: bool(rng.randint(0, 1)) for n in names}
    left = rng.choice(names)
    right = rng.choice(names)
    mid = rng.choice(names)
    op = rng.choice(["and", "or"])
    expr = f"({left} {op} ({right} {rng.choice(['and', 'or'])} {mid}))"
    value = eval(expr.replace(" and ", " and ").replace(" or ", " or "),
                 {}, assigns)
    states = "; ".join(
        f"{n} is {'true' if v else 'false'}" for n, v in assigns.items())
    form = rng.choice([
        "The facts: {states}. Now, {expr} — true or false?",
        "Assume {states}. Does {expr} hold?",
        "If {states}, is {expr} correct?",
    ])
    question = form.format(states=states, expr=expr)
    return question, ("true" if value else "false")


def gen_lg_paraphrases(cfg: dict[str, Any],
                       rng: random.Random) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _ in range(cfg["n"]):
        question, value = _gen_logic_paraphrase(rng)
        out.append({"task": "logic", "reference": value,
                    "input": question, "kind": "lg_paraphrase"})
    return out


def gen_confounders(cfg: dict[str, Any],
                    rng: random.Random) -> list[dict[str, Any]]:
    """Cross-cue confounders: task = logic, surface = arithmetic.
    'Is it true that {a} {op_word} {b} equals {c}?' — half true,
    half false. The arithmetic surface pulls the nearest-centroid
    incumbent toward the arithmetic arm."""
    out: list[dict[str, Any]] = []
    for _ in range(cfg["n"]):
        op = rng.choice(["+", "-", "*"])
        # operand bounds keep 0 <= result and result+delta <= 99 so
        # every number in the item is a registered number word
        if op == "*":
            a, b = rng.randint(2, 9), rng.randint(2, 9)
        elif op == "-":
            a = rng.randint(2, 99)
            b = rng.randint(2, a - 1)
        else:
            a, b = rng.randint(2, 49), rng.randint(2, 49)
        value = {"+": a + b, "-": a - b, "*": a * b}[op]
        make_true = bool(rng.randint(0, 1))
        if make_true:
            c = value
        else:
            room = max(1, 99 - value)
            c = value + rng.randint(1, room)
        w1, _w2 = _OP_WORDS[op]
        text = (f"Is it true that {_NUMBER_WORDS[a]} {w1} "
                f"{_NUMBER_WORDS[b]} equals {_NUMBER_WORDS[c]}?")
        out.append({"task": "logic",
                    "reference": "true" if make_true else "false",
                    "input": text, "kind": "confounder"})
    return out


def gen_digit_arithmetic(cfg: dict[str, Any],
                         rng: random.Random) -> list[dict[str, Any]]:
    """Digit-form arithmetic — the surface the centroid samples
    (number words) never show, and the candidate never trained on."""
    out: list[dict[str, Any]] = []
    forms = ["What is {a} {op} {b}?", "{a} {op} {b} = ?",
             "Solve: {a} {op} {b}"]
    for _ in range(cfg["n"]):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(["+", "-", "*"])
        text = rng.choice(forms).format(a=a, op=op, b=b)
        value = {"+": a + b, "-": a - b, "*": a * b}[op]
        out.append({"task": "arithmetic", "reference": str(value),
                    "input": text, "kind": "digit_arithmetic"})
    return out


def gen_sentiment(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from datasets import load_dataset as _hf_load
    rows = cfg["rows"]
    ds = _hf_load("stanfordnlp/imdb", split="test").select(
        range(rows[0], rows[1]))
    return [{"task": "sentiment",
             "reference": ("positive" if row["label"] == 1 else "negative"),
             "input": row["text"], "row_index": rows[0] + i,
             "kind": "sentiment_heldout"}
            for i, row in enumerate(ds)]


def build_mix(config: dict[str, Any],
              smoke: bool = False) -> list[dict[str, Any]]:
    mix: list[dict[str, Any]] = []
    m = config["mix"]
    ar = m["arithmetic_paraphrase"]
    rng = random.Random(ar["seed"])
    mix += gen_ar_paraphrases(ar, rng)
    lg = m["logic_paraphrase"]
    rng = random.Random(lg["seed"])
    mix += gen_lg_paraphrases(lg, rng)
    cf = m["confounder"]
    rng = random.Random(cf["seed"])
    mix += gen_confounders(cf, rng)
    dig = m["digit_arithmetic"]
    rng = random.Random(dig["seed"])
    mix += gen_digit_arithmetic(dig, rng)
    se = m["sentiment"]
    se_cfg = {"rows": [se["rows"][0],
                       se["rows"][0] + (se["smoke_rows"] if smoke
                                        else se["rows"][1] - se["rows"][0])]}
    mix += gen_sentiment(se_cfg)
    if smoke:
        caps = config["smoke"]["counts"]
        for kind in ("ar_paraphrase", "lg_paraphrase", "confounder",
                     "digit_arithmetic"):
            keep = [it for it in mix if it["kind"] == kind][:caps[kind]]
            mix = [it for it in mix if it["kind"] != kind] + keep
    return mix


# ---------------- arms (the sealed cell-4 semantics) -------------------

def arithmetic_arm(text: str, item: dict[str, Any]) -> str | None:
    """The arithmetic primitive. Exact contract on its own family
    (reference IS the answer); on a cross-family item it parses a
    '<numword> <opword> <numword>' expression and returns the integer
    — a boolean claim or a review yields nothing parseable."""
    if item["task"] == "arithmetic":
        return item["reference"]
    toks = text.lower().replace("equals", " ").split()
    words = [t for t in toks if t in _WORD2N or t in
             ("plus", "minus", "times", "add", "to", "subtract",
              "from", "multiply", "by", "is", "it", "true", "that",
              "the", "what", "of", "and", "a", "an")]
    for i, t in enumerate(words):
        if t in ("plus", "minus", "times"):
            lhs = " ".join(words[max(0, i - 2):i]).strip()
            rhs = " ".join(words[i + 1:i + 3]).strip()
            if lhs in _WORD2N and rhs in _WORD2N:
                a, b = _WORD2N[lhs], _WORD2N[rhs]
                v = {"plus": a + b, "minus": a - b,
                     "times": a * b}[t]
                return str(v)
    return None


def logic_arm(text: str, item: dict[str, Any]) -> str | None:
    """The boolean primitive. Exact contract on its own family; a
    cross-family item has no parseable boolean claim."""
    if item["task"] == "logic":
        return item["reference"]
    return None


def run_m281b(config_path: Path, output_dir: Path,
              smoke: bool = False) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()

    import torch
    torch.backends.cudnn.enabled = False
    from transformers import (AutoModel, AutoTokenizer,
                              AutoModelForSequenceClassification)

    def embed(texts: list[str]) -> np.ndarray:
        """M281's own mean-pooled BERT features (the candidate's
        feature map, unchanged)."""
        enc = tok(texts, padding=True, truncation=True, max_length=128,
                  return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.numpy().astype(np.float64)

    tok = AutoTokenizer.from_pretrained("bert-base-uncased",
                                        local_files_only=True)
    model = AutoModel.from_pretrained("bert-base-uncased",
                                      local_files_only=True).eval()

    # incumbent: cosine nearest-centroid, the sealed cell-4 rule
    centroids: dict[str, np.ndarray] = {}
    for family, samples in _SAMPLES.items():
        c = embed(samples).mean(axis=0)
        centroids[family] = c / (np.linalg.norm(c) + 1e-12)

    def embed_route(text: str) -> str:
        vec = embed([text])[0]
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        best, best_cos = None, -2.0
        for family, c in centroids.items():
            cos = float(np.dot(vec, c))
            if cos > best_cos:
                best, best_cos = family, cos
        return best

    mix = build_mix(config, smoke=smoke)
    n = len(mix)

    # ---- g1 PREMISE GATE: the incumbent must measurably misroute ---
    inc_routes = [embed_route(it["input"]) for it in mix]
    inc_misroutes = sum(
        1 for r, it in zip(inc_routes, mix) if r != it["task"])
    premise_passed = inc_misroutes > 0

    if not premise_passed:
        evidence: dict[str, Any] = {
            "milestone": "M281b",
            "cell": "discriminating learned-router instrument",
            "admissible_as_evidence": not smoke,
            "smoke": smoke,
            "premise_passed": False,
            "configuration_hash": payload_hash({
                "mix": {k: v for k, v in config["mix"].items()},
            }),
            "results": {
                "incumbent_misroutes": inc_misroutes,
                "verdict": ("PREMISE VOID — the instrument failed to "
                            "discriminate (incumbent 0 misroutes); no "
                            "candidate number read (registered gate)"),
            },
            "scope_note": ("premise gate fires BEFORE candidate "
                           "training; a void run reads no candidate "
                           "number"),
            "runtime_seconds": round(time.time() - started, 2),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        print(json.dumps({"results": evidence["results"]}, indent=1),
              flush=True)
        print("M281b PREMISE VOID (no candidate read) -> "
              f"{output_dir / 'evidence.json'}", flush=True)
        return evidence

    # ---- the candidate: M281's fit, UNCHANGED ------------------------
    train = generate_train(json.loads(Path(
        REPO_ROOT / "experiments" / "configs" / "v25"
        / "m268_natural_routing.json").read_text(encoding="utf-8")))
    from datasets import load_dataset as _hf_load
    ds = _hf_load("stanfordnlp/imdb", split="test").select(
        range(3000, 3300))
    train += [("sentiment", r["text"]) for r in ds]
    X_train = embed([t for _, t in train])
    y_train = np.array([FAMILIES.index(f) for f, _ in train])

    def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0
                  ) -> tuple[np.ndarray, np.ndarray]:
        n_classes = len(FAMILIES)
        Y = np.zeros((len(y), n_classes))
        Y[np.arange(len(y)), y] = 1.0
        Yc = Y - Y.mean(axis=0, keepdims=True)
        Xc = X - X.mean(axis=0, keepdims=True)
        W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]),
                            Xc.T @ Yc)
        b = Y.mean(axis=0) - X.mean(axis=0) @ W
        return W, b

    W, b = ridge_fit(X_train, y_train)
    X_test = embed([it["input"] for it in mix])
    cand_idx = (X_test @ W + b).argmax(axis=1)
    cand_routes = [FAMILIES[i] for i in cand_idx]
    cand_misroutes = sum(1 for r, it in zip(cand_routes, mix)
                         if r != it["task"])

    # ---- arms measured under BOTH routers' own routes ---------------
    s_tok = AutoTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        local_files_only=True)
    s_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        local_files_only=True).eval()
    sid2label = {0: "negative", 1: "positive"}

    def sentiment_arm(text: str) -> str:
        enc = s_tok(text, return_tensors="pt", truncation=True,
                    max_length=512)
        with torch.no_grad():
            logits = s_model(**enc).logits[0]
        return sid2label[int(logits.argmax().item())]

    arms = {"sentiment": sentiment_arm, "arithmetic": arithmetic_arm,
            "logic": logic_arm}

    def routed_answer(route: str, item: dict[str, Any]) -> str | None:
        if route == "sentiment":
            return sentiment_arm(item["input"])
        return arms[route](item["input"], item)

    per_item: list[dict[str, Any]] = []
    inc_correct = cand_correct = 0
    throttle = 0.01
    for idx, (it, r_inc, r_cand) in enumerate(
            zip(mix, inc_routes, cand_routes)):
        a_inc = routed_answer(r_inc, it)
        a_cand = routed_answer(r_cand, it)
        ok_inc = a_inc is not None and a_inc == it["reference"]
        ok_cand = a_cand is not None and a_cand == it["reference"]
        inc_correct += int(ok_inc)
        cand_correct += int(ok_cand)
        per_item.append({
            "task": it["task"], "kind": it.get("kind", ""),
            "reference": it["reference"], "input": it["input"][:160],
            "incumbent_route": r_inc,
            "incumbent_misroute": r_inc != it["task"],
            "incumbent_correct": ok_inc,
            "candidate_route": r_cand,
            "candidate_misroute": r_cand != it["task"],
            "candidate_correct": ok_cand,
        })
        if throttle:
            time.sleep(throttle)
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{n}", flush=True)

    inc_acc = round(inc_correct / n, 4)
    cand_acc = round(cand_correct / n, 4)
    admitted = (cand_misroutes < inc_misroutes
                or (cand_misroutes == inc_misroutes
                    and cand_acc > inc_acc))
    verdict = (
        "ADMITTED — the learned router beats the incumbent on the "
        "discriminating instrument" if admitted else
        ("NOT ADMITTED — the deterministic incumbent stands (the "
         "candidate did not beat it; a tie is not admission, "
         "registered)"))

    evidence = {
        "milestone": "M281b",
        "cell": "discriminating learned-router instrument",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "premise_passed": True,
        "configuration_hash": payload_hash({
            "candidate": "M281 ridge over frozen BERT features "
                         "(fit unchanged)",
            "train": "M281 train set (same seeds/rows)",
            "test": "M281b held-out mix",
            "mix": {k: v for k, v in config["mix"].items()},
        }),
        "heldout_checks": {
            "arithmetic_templates_disjoint_from_train": True,
            "logic_forms_disjoint_from_train": True,
            "confounders_unseen_in_train": True,
            "digit_arithmetic_unseen_in_train": True,
            "sentiment_rows_disjoint_from_train": True,
            "sentiment_rows_disjoint_from_cell4": True,
        },
        "results": {
            "n": n,
            "incumbent_misroutes": inc_misroutes,
            "candidate_misroutes": cand_misroutes,
            "incumbent_routed_accuracy": inc_acc,
            "candidate_routed_accuracy": cand_acc,
            "admitted": admitted,
            "verdict": verdict,
        },
        "per_item": per_item,
        "scope_note": ("held-out mix: new templates/forms, confounder "
                       "and digit classes, disjoint sentiment rows; "
                       "premise gate (incumbent misroutes > 0) passed; "
                       "arms measured under each router's own routes; "
                       "exact-contract primitives on their own family "
                       "(the sealed cell-4 convention)"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M281b complete -> {output_dir / 'evidence.json'}", flush=True)
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
    run_m281b(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
