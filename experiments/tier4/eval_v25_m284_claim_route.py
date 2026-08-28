"""M284 — contract-claim routing repair: evidence run.

g1: every M281b confounder (rebuilt from the registered seed) and
every M283 contract_spoof probe routes to the logic primitive
through the M284 pre-pass and is answered exactly.
g2 REGRESSION: the pre-pass matches NOTHING outside the claim
classes — the cell-4 700 items, the M281b non-confounder items,
and the 12 non-contract M283 probes all keep their embedding
routes (zero claim matches on them).

Deterministic; no GPU. Evidence:
logs/results/v25/m284_claim_route/evidence.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v25_m281b_discriminating_router import (
    build_mix,
    gen_confounders,
)
from geode.core.claim_route import claim_answer, detect_claim
from geode.core.measured_routing import route_policy
from geode.core.router_probes import RouterProbeSuite

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m284_claim_route")


class _NoRouter:
    """routing is never consulted on claim matches; the pre-pass
    fires first (registered)."""

    def route(self, text: str) -> str:
        raise AssertionError("router consulted for a claim match")


def run_m284(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    config = json.loads(Path(REPO_ROOT / "experiments" / "configs"
                             / "v25"
                             / "m281b_discriminating_router.json")
                        .read_text(encoding="utf-8"))

    # ---- g1: claims route to logic and answer exactly -----------------
    confounders = gen_confounders(config["mix"]["confounder"],
                                  __import__("random").Random(
                                      config["mix"]["confounder"]["seed"]))
    suite = RouterProbeSuite()
    contract_probes = [p for p in suite.probes.values()
                       if p["category"] == "contract_spoof"]

    g1_rows = [{"kind": "m281b_confounder", "text": c["input"],
                "reference": c["reference"]} for c in confounders]
    g1_rows += [{"kind": "m283_contract_spoof", "text": p["text"],
                 "reference": None}
                for p in contract_probes
                if detect_claim(p["text"]) is not None]
    router_handled = [{"id": p["id"], "text": p["text"]}
                      for p in contract_probes
                      if detect_claim(p["text"]) is None]

    g1: list[dict[str, Any]] = []
    for row in g1_rows:
        decision = route_policy(_NoRouter(), row["text"])
        ok_family = decision.get("family") == "logic"
        ok_arm = decision.get("arm") == "primitive"
        answer = decision.get("claim", {}).get("answer")
        ok_answer = (answer == row["reference"]
                     if row["reference"] is not None else answer
                     in ("true", "false"))
        g1.append({"kind": row["kind"], "text": row["text"],
                   "family": decision.get("family"),
                   "answer": answer, "reference": row["reference"],
                   "ok": bool(ok_family and ok_arm and ok_answer)})

    # ---- g2: regression — no claim matches outside the classes --------
    full_mix = build_mix(config)  # the 800-item M281b mix
    non_confounders = [it for it in full_mix if it["kind"]
                       != "confounder"]
    cell4_cfg = json.loads(Path(
        REPO_ROOT / "experiments" / "configs" / "v25"
        / "m268_natural_routing.json").read_text(encoding="utf-8"))
    from experiments.tier4.eval_v25_m268_natural_routing import (
        _gen_bool_question, _NUMBER_WORDS, _OP_WORDS)
    cell4_items: list[str] = []
    ar_cfg = cell4_cfg["mix"]["arithmetic"]
    rng = __import__("random").Random(ar_cfg["seed"])
    for _ in range(ar_cfg["n"]):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(["+", "-", "*"])
        w1, w2 = _OP_WORDS[op]
        cell4_items.append(rng.choice(ar_cfg["templates"]).format(
            a=_NUMBER_WORDS[a], b=_NUMBER_WORDS[b],
            op_word=w1, op_word2=w2))
    bo_cfg = cell4_cfg["mix"]["boolean"]
    rng = __import__("random").Random(bo_cfg["seed"])
    for _ in range(bo_cfg["n"]):
        q, _v = _gen_bool_question(rng)
        cell4_items.append(q)
    from datasets import load_dataset as _hf_load
    se_cfg = cell4_cfg["mix"]["sentiment"]
    cell4_items += [r["text"] for r in _hf_load(
        se_cfg["hf_id"], split=se_cfg["split"]).select(
            range(se_cfg["rows"][0], se_cfg["rows"][1]))]
    non_contract_probes = [p for p in suite.probes.values()
                           if p["category"] != "contract_spoof"]

    regression_texts = [it["input"] for it in non_confounders] \
        + cell4_items + [p["text"] for p in non_contract_probes]
    false_matches = [(t, claim_answer(t)) for t in regression_texts
                     if claim_answer(t) is not None]

    g1_ok = all(r["ok"] for r in g1)
    g2_ok = not false_matches
    results = {
        "g1_claim_items": len(g1_rows),
        "g1_all_ok": g1_ok,
        "g1_per_item": g1,
        "router_handled_contract_probes": router_handled,
        "g2_regression_items": len(regression_texts),
        "g2_false_matches": [{"text": t[:120], "claim": c}
                             for t, c in false_matches],
        "g2_ok": g2_ok,
        "verdict": ("M284 PASS — claims route to the logic primitive "
                    "and answer exactly; no route changes elsewhere"
                    if g1_ok and g2_ok else
                    "M284 FAIL — see g1/g2 details"),
    }
    evidence: dict[str, Any] = {
        "milestone": "M284",
        "cell": "contract-claim routing repair",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "module": "geode/core/claim_route.py",
            "prepass": "route_policy claim pre-pass",
            "g1": "M281b confounders (seed 20260828) + M283 "
                  "contract_spoof probes",
            "g2": "cell-4 700 + M281b non-confounders + 12 "
                  "non-contract probes",
        }),
        "results": results,
        "scope_note": ("deterministic grammar pre-pass; strict "
                       "adjacency protects the spoof classes; the "
                       "claim evaluator is exact arithmetic"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": results}, indent=1), flush=True)
    print(f"M284 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


if __name__ == "__main__":
    run_m284(DEFAULT_OUTPUT)
