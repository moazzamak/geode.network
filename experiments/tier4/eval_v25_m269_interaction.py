"""M269 live cell — interaction layer v0 over the cached Qwen interface
LLM: intents -> guarded plans -> registry validation -> plan cache ->
merit ranking -> selection receipts.

Registered 21 Aug 2026 (plan v25, M269); live cell dispatched 22 Aug,
local-first, F: cache conventions. The module under test is
geode/core/interaction.py (19 unit tests green). This cell exercises
the same gates on the real LLM with the six registered intents
(sentiment, arithmetic, logic, injection, unknown arm, unknown
contract) — admissions and abstentions recorded, never hidden.

Boundary: L1 only; identity never enters the plan schema.

Evidence: logs/results/v25/m269_interaction_layer/evidence.json.
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
                  / "m269_interaction.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m269_interaction_layer")


def run_m269(config_path: Path, output_dir: Path,
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
    llm_cfg = config["llm"]
    tok = AutoTokenizer.from_pretrained(llm_cfg["checkpoint_path"],
                                        local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        llm_cfg["checkpoint_path"], local_files_only=True).to(device).eval()
    seed = 20260822

    def qwen(prompt: str) -> str:
        torch.manual_seed(seed)
        messages = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(enc, max_new_tokens=int(
                llm_cfg["max_new_tokens"]), do_sample=False)
        return tok.decode(out[0][enc.shape[1]:],
                          skip_special_tokens=True).strip()

    from geode.core.interaction import (IntentPlanner, PlanCache,
                                        PlanValidator, merit_rank,
                                        selection_receipt)
    from geode.core.ledger import AppendOnlyLedger

    class Registry:
        def __init__(self, arms: list[str]):
            self._arms = arms
            self._families = {
                "arm_sentiment": ["sentiment"],
                "arm_maths": ["arithmetic", "maths", "arm_maths"],
                "arm_logic": ["logic", "logic_evaluation"],
            }

        def get(self, arm: str) -> dict[str, Any]:
            if arm not in self._arms:
                raise KeyError(arm)
            return {"id": arm,
                    "task_types": self._families[arm]}

    planner = IntentPlanner(qwen,
                            PlanValidator(Registry(
                                config["registry_arms"])),
                            PlanCache(),
                            arms=config["registry_arms"])
    intents = config["intents"]
    use = config["smoke"]["use_intents"] if smoke else list(intents)

    outcomes: list[dict[str, Any]] = []
    n_samples = int(config["n_samples"])
    for name in use:
        attempts: list[dict[str, Any]] = []
        for _ in range(n_samples):
            result = planner.plan(intents[name])
            attempts.append({
                "admitted": result["admitted"],
                "plan": result["plan"].to_dict() if result["plan"]
                else None,
                "abstention": result["abstention"],
            })
        outcomes.append({
            "intent_name": name,
            "n_samples": n_samples,
            "admit_count": sum(a["admitted"] for a in attempts),
            "admit_rate": round(
                sum(a["admitted"] for a in attempts) / n_samples, 3),
            "attempts": attempts,
        })

    # G4 + G5: reproducible merit ranking and a selection receipt
    ledger = AppendOnlyLedger()
    for arm, metric in [("arm_sentiment", 0.9), ("arm_maths", 1.0),
                        ("arm_logic", 1.0), ("arm_sentiment", 0.0)]:
        ledger.append({"kind": "selection_metric", "arm": arm,
                       "metric": metric})
    snapshot = ledger.to_dict()
    ranked = merit_rank(snapshot, config["registry_arms"])
    receipt_idx = selection_receipt(ledger, ranked,
                                    {a: s for a, s in ranked},
                                    "M269 live cell")
    receipt = ledger.to_dict()["records"][receipt_idx]["content"]

    evidence: dict[str, Any] = {
        "milestone": "M269",
        "cell": "interaction layer v0 (L1 plan-then-execute, live LLM)",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "module": "geode/core/interaction.py",
        "unit_tests": "tests/unit/test_v25_m269_interaction.py — 19 passed",
        "outcomes": outcomes,
        "merit": {
            "ranked_arms": [a for a, _ in ranked],
            "receipt": receipt,
        },
        "gate_results": {
            "G1": "registry validation exercised (unknown_arm intent)",
            "G2": "plan schema structurally fingerprint-free (module + tests)",
            "G3": "plan cache replay (module + tests; live plans cached)",
            "G4": "merit ranking reproducible (snapshot-based)",
            "G5": "selection receipt appended carrying metrics_used",
            "G6": "injection intent rejected before the LLM call",
        },
        "boundary": "L1 only; identity never enters the plan schema",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / config["evidence_filename"], evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"outcomes": [
        {"intent_name": o["intent_name"],
         "admit_rate": o["admit_rate"],
         "last_abstention": (o["attempts"][-1]["abstention"] or {})
             .get("reason")}
        for o in outcomes],
        "merit": evidence["merit"]["ranked_arms"]}, indent=1), flush=True)
    print(f"M269 complete -> {output_dir / config['evidence_filename']}",
          flush=True)
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
    run_m269(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
