"""M166 — ontology bootstrap batch 1 (the frozen local LLM).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` (section
3.2; the section 12 dispatch entry, 17 Aug 2026, with the registered
1.5B amendment). Loads the pinned Qwen2.5-1.5B-Instruct snapshot
(local files only, fp16, greedy decoding) and produces DRAFT proposals
— the axis set/vocabularies, the normalisation rules, and the
candidate similarity-control task pairs. The drafts are digest-tagged
artifacts; ratification is a human step (I3). No corpus data ever
enters a prompt; no API is called.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from geode.core.descriptor import AXES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m166_ontology.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v24"
                  / "m166_ontology_draft")

MEASURED_TASKS = [
    "DomainNet image classification (345 classes, 6 domains)",
    "CIFAR-10 image classification (10 classes)",
    "Mackey-Glass one-step forecasting (chaotic series)",
    "Lorenz-63 one-step forecasting (chaotic system)",
    "Dyck grammar next-token (bracket-matching language)",
    "tabular regression (numeric features -> real value)",
]


def _extract_json(text: str) -> Any:
    """Defensive JSON extraction: first { to last }."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {"raw": text}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"raw": text[start:end + 1]}


def _generate(model, tokenizer, messages, max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=1.0)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)


def run_m166(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    local_dir = (data_cache_root() / "huggingface"
                 / "Qwen2.5-1.5B-Instruct")
    print(f"loading pinned model from {local_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(local_dir),
                                              local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(local_dir), local_files_only=True, dtype=torch.float16)
    model.to(device)
    model.eval()

    max_new = int(config["generation"]["max_new_tokens"])
    system = ("You assist a research program that learns tasks with "
              "frozen encoders and closed-form fits. Answer only with "
              "strict JSON; no prose outside the JSON.")
    provisional = json.dumps({"axes": AXES}, sort_keys=True)

    print("batch 1: axes + vocabularies", flush=True)
    text1 = _generate(model, tokenizer, [
        {"role": "system", "content": system},
        {"role": "user", "content": config["batch1"]["role"] + "\n"
         + "Provisional schema: " + provisional},
    ], max_new)
    draft1 = _extract_json(text1)

    print("batch 2: rules + candidate pairs", flush=True)
    text2 = _generate(model, tokenizer, [
        {"role": "system", "content": system},
        {"role": "user", "content": config["batch2"]["role"] + "\n"
         + "Measured tasks: " + json.dumps(MEASURED_TASKS)},
    ], max_new)
    draft2 = _extract_json(text2)

    evidence: dict[str, Any] = {
        "milestone": "M166",
        "cell": "ontology bootstrap batch 1 (draft proposals)",
        "admissible_as_evidence": True,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "model": config["model"],
        "batch1": {"raw": text1, "parsed": draft1},
        "batch2": {"raw": text2, "parsed": draft2},
        "ratification_required": True,
        "note": "proposals only; the human ratifies, then the frozen "
                "task_ontology_v0.json is written (I3).",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "draft.json", evidence)
    write_canonical_json(output_dir / "ratification_checklist.json", {
        "steps": [
            "review batch1 axes/vocabularies against the provisional schema",
            "review batch2 rules and candidate pairs",
            "edit or accept; then write analysis/task_ontology_v0.json",
            "re-run the descriptor unit tests against the ratified schema",
        ],
        "status": "pending_human",
    })
    build_artifact_index(output_dir)
    print(f"M166 draft complete -> {output_dir / 'draft.json'}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m166(args.config, args.output)


if __name__ == "__main__":
    main()
