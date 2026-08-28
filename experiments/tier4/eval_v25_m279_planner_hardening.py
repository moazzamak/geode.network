"""M279 — interface-LLM hardening: the few-shot planner behind the
same six M269 gates, with the admit-rate bar registered before the
run.

Evidence: logs/results/v25/m279_planner_hardening/evidence.json.
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
                  / "m279_planner_hardening.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m279_planner_hardening")


def run_m279(config_path: Path, output_dir: Path,
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
                                        PlanValidator)

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
                            arms=config["registry_arms"],
                            examples=config["examples"])
    intents = config["intents"]
    use = config["smoke"]["use_intents"] if smoke else list(intents)
    n_samples = (config["smoke"]["n_samples"] if smoke
                 else config["bar"]["n_samples"])

    outcomes: list[dict[str, Any]] = []
    for name in use:
        admits = 0
        attempts: list[dict[str, Any]] = []
        for _ in range(n_samples):
            result = planner.plan(intents[name])
            admitted = bool(result["admitted"])
            admits += int(admitted)
            attempts.append({
                "admitted": admitted,
                "plan": result["plan"].to_dict() if result["plan"]
                else None,
                "abstention": result["abstention"],
            })
        outcomes.append({
            "intent_name": name,
            "n_samples": n_samples,
            "admit_rate": round(admits / n_samples, 3),
            "attempts": attempts,
        })

    bar = config["bar"]
    if smoke:
        bar_pass = True
        bar_details: dict[str, Any] = {"note": "smoke"}
    else:
        admissible = ["sentiment", "arithmetic", "logic"]
        rates = {o["intent_name"]: o["admit_rate"] for o in outcomes}
        admit_ok = all(rates.get(name, 0.0) >=
                       bar["admit_rate_per_admissible_intent"]
                       for name in admissible)
        reject_ok = all(rates.get(name, 1.0) == 0.0
                        for name in bar["reject_required"])
        bar_pass = admit_ok and reject_ok
        bar_details = {
            "admit_ok": admit_ok,
            "reject_ok": reject_ok,
            "rates": rates,
            "bar": bar["admit_rate_per_admissible_intent"],
        }

    evidence: dict[str, Any] = {
        "milestone": "M279",
        "cell": "interface-LLM hardening (few-shot)",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "results": {
            "outcomes": outcomes,
            "bar_passed": bar_pass,
            "bar_details": bar_details,
            "sealed_m269_baseline": {
                "admit_rates": {"sentiment": 1.0,
                                "arithmetic": 1.0,
                                "logic": 1.0},
                "reject_rates": {"injection": 1.0,
                                 "unknown_arm": 1.0,
                                 "unknown_contract": 1.0},
                "source": ("logs/results/v25/m269_interaction_layer/"
                           "evidence.json"),
            },
        },
        "scope_note": ("the same six M269 gates unchanged; the "
                       "admit-rate is a monitored metric going "
                       "forward; a larger interface LLM is admitted "
                       "only behind these same gates if the bar "
                       "fails"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / config["evidence_filename"], evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"bar_passed": bar_pass,
                      "bar_details": bar_details}, indent=1), flush=True)
    print(f"M279 complete -> "
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
    run_m279(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
