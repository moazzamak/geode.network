"""M268 cell 3 — programming tier: code specialist vs generalist,
single-sample pass@1 on HumanEval, graded by sandboxed execution.

Registered and dispatched 22 Aug 2026 (plan v25, amendment 34),
local-first, F: cache conventions. The registered honest position:
the LLM proposes, the compiler/test-runner disposes — the only
component that can be right is the verifier. Grading runs the
problem's own check function inside a guarded subprocess (python -I
-B, no shell, 5s timeout, temp working directory); the environment
fingerprint is recorded.

Licensing: Qwen2.5-Coder-1.5B-Instruct Apache-2.0; generalist
Qwen2.5-1.5B-Instruct Apache-2.0; HumanEval from the official
MIT-licensed openai/human-eval repository (the HF mirror carried no
license metadata and was excluded by G6). Declared: HumanEval sits
in the training corpora of code LLMs — product-quality reading,
published anchors cited, never SOTA.

Evidence: logs/results/v25/m268_routing_study/evidence_code.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
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
                  / "m268_code.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m268_routing_study")

PASS_MARKER = "GEODE_CHECK_PASS"


def run_check(candidate: str, test: str, entry_point: str,
              timeout: float = 5.0) -> dict[str, Any]:
    """Execute the candidate under the registered sandbox guard."""
    script = (
        f"{candidate}\n\n"
        f"{test}\n\n"
        f"if __name__ == '__main__':\n"
        f"    check({entry_point})\n"
        f"    print('{PASS_MARKER}')\n"
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", str(path)],
                capture_output=True, timeout=timeout, cwd=td,
                env={"PATH": "", "SystemRoot":
                     r"C:\Windows", "TEMP": td, "TMP": td},
            )
            out = proc.stdout.decode(errors="replace")
            err = proc.stderr.decode(errors="replace")[-400:]
            return {"exit_code": proc.returncode,
                    "passed": PASS_MARKER in out,
                    "stdout_tail": out[-200:], "stderr_tail": err}
        except subprocess.TimeoutExpired:
            return {"exit_code": None, "passed": False,
                    "stdout_tail": "", "stderr_tail": "timeout"}


def run_m268_code(config_path: Path, output_dir: Path,
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
    sequential = bool(config["arms"].get("sequential", False))

    def load(path: str, dtype_str: str | None = None):
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        kwargs: dict[str, Any] = {}
        if dtype_str:
            kwargs["torch_dtype"] = {"float16": torch.float16,
                                     "bfloat16": torch.bfloat16}[dtype_str]
        model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, **kwargs).to(device).eval()
        return tok, model

    def arm_dtype(arm: str) -> str | None:
        return config["arms"][arm].get("torch_dtype")

    coder_path = config["arms"]["coder"]["checkpoint_path"]
    gen_path = config["arms"]["generalist"]["checkpoint_path"]

    coder_tok, coder = load(coder_path, arm_dtype("coder"))
    gen_tok, gen = load(gen_path, arm_dtype("generalist"))
    if sequential:
        del gen_tok, gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    max_new = int(config["arms"]["coder"]["max_new_tokens"])
    seed = int(config["corpus"]["n_problems"])

    def generate(tok, model, prompt: str) -> str:
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

    problems = [json.loads(line) for line in open(
        config["corpus"]["path"], encoding="utf-8")]
    if smoke:
        problems = problems[:config["smoke"]["problems"]]

    def clean(text: str) -> str:
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)

    per_item: list[dict[str, Any]] = []
    throttle = 0.01  # the registered display-GPU TDR mitigation

    if sequential:
        # one arm at a time (the 7B-class trunks do not fit together)
        coder_results: dict[str, Any] = {}
        for i, p in enumerate(problems):
            prompt = config["prompt"].format(prompt=p["prompt"])
            coder_ans = clean(generate(coder_tok, coder, prompt))
            coder_results[p["task_id"]] = run_check(
                coder_ans, p["test"], p["entry_point"])
            if throttle:
                time.sleep(throttle)
            if (i + 1) % 20 == 0:
                print(f"  coder {i + 1}/{len(problems)}", flush=True)
        del coder_tok, coder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gen_tok, gen = load(gen_path, arm_dtype("generalist"))
        for i, p in enumerate(problems):
            prompt = config["prompt"].format(prompt=p["prompt"])
            gen_ans = clean(generate(gen_tok, gen, prompt))
            gen_check = run_check(gen_ans, p["test"], p["entry_point"])
            coder_check = coder_results[p["task_id"]]
            per_item.append({
                "task_id": p["task_id"],
                "coder_pass": coder_check["passed"],
                "generalist_pass": gen_check["passed"],
                "coder_exit": coder_check["exit_code"],
                "generalist_exit": gen_check["exit_code"],
                "coder_stderr_tail": coder_check["stderr_tail"],
                "generalist_stderr_tail": gen_check["stderr_tail"],
            })
            if throttle:
                time.sleep(throttle)
            if (i + 1) % 20 == 0:
                print(f"  generalist {i + 1}/{len(problems)}", flush=True)
    else:
        for i, p in enumerate(problems):
            prompt = config["prompt"].format(prompt=p["prompt"])
            coder_ans = clean(generate(coder_tok, coder, prompt))
            gen_ans = clean(generate(gen_tok, gen, prompt))
            coder_check = run_check(coder_ans, p["test"],
                                    p["entry_point"])
            gen_check = run_check(gen_ans, p["test"], p["entry_point"])
            per_item.append({
                "task_id": p["task_id"],
                "coder_pass": coder_check["passed"],
                "generalist_pass": gen_check["passed"],
                "coder_exit": coder_check["exit_code"],
                "generalist_exit": gen_check["exit_code"],
                "coder_stderr_tail": coder_check["stderr_tail"],
                "generalist_stderr_tail": gen_check["stderr_tail"],
            })
            if throttle:
                time.sleep(throttle)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(problems)} coder "
                      f"{sum(x['coder_pass'] for x in per_item)} "
                      f"generalist "
                      f"{sum(x['generalist_pass'] for x in per_item)}",
                      flush=True)

    n = len(per_item)
    coder_passes = sum(x["coder_pass"] for x in per_item)
    gen_passes = sum(x["generalist_pass"] for x in per_item)
    evidence: dict[str, Any] = {
        "milestone": "M268",
        "cell": "cell 3 — programming tier (d-code), pass@1 n=1",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "results": {
            "n_problems": n,
            "coder_pass_at_1": round(coder_passes / n, 4)
            if n else None,
            "generalist_pass_at_1": round(gen_passes / n, 4)
            if n else None,
            "coder_wins_by": round((coder_passes - gen_passes) / n, 4)
            if n else None,
        },
        "sandbox_fingerprint": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "flags": "-I -B, no shell, timeout 5s, temp cwd",
        },
        "per_item": per_item,
        "scope_note": ("single-sample pass@1 (n=1, greedy); product-"
                       "quality reading; contamination declared"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / config["evidence_filename"], evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M268 cell 3 complete -> "
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
    run_m268_code(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
