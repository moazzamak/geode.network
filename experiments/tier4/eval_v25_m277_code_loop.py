"""M277 — execution-feedback loop on the code arm.

Registered 22 Aug 2026 (plan v25, the M272-M281 wave). The verifier
is the SEALED cell-3 sandbox (the test-runner disposes, the LLM only
proposes). Protocol: attempt 1 is the plain problem prompt; on a
failed sandbox run the stderr tail is fed back with the registered
fix prompt, up to k=3 attempts; pass@k is the measured comparison
against the sealed single-shot 0.598.

Prior art cited, not exceeded: execution-feedback repair (the
AlphaCode line, arXiv:2203.07814; self-repair-class work). One
held-out read per configuration; contamination declared.

Evidence: logs/results/v25/m277_code_loop/evidence.json.
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
                  / "m277_code_loop.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m277_code_loop")

PASS_MARKER = "GEODE_CHECK_PASS"
BASE_PROMPT = ("Complete the following Python function. Return only "
               "the function code, without any explanation or "
               "markdown fences.\n{prompt}")


def run_check(candidate: str, test: str, entry_point: str,
              timeout: float = 5.0) -> dict[str, Any]:
    """The sealed cell-3 sandbox (verbatim)."""
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
                env={"PATH": "", "SystemRoot": r"C:\Windows",
                     "TEMP": td, "TMP": td},
            )
            out = proc.stdout.decode(errors="replace")
            err = proc.stderr.decode(errors="replace")[-400:]
            return {"exit_code": proc.returncode,
                    "passed": PASS_MARKER in out,
                    "stdout_tail": out[-200:], "stderr_tail": err}
        except subprocess.TimeoutExpired:
            return {"exit_code": None, "passed": False,
                    "stdout_tail": "", "stderr_tail": "timeout"}


def clean(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def run_m277(config_path: Path, output_dir: Path,
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
    arm = config["arm"]
    tok = AutoTokenizer.from_pretrained(arm["checkpoint_path"],
                                        local_files_only=True)
    kwargs: dict[str, Any] = {}
    if arm.get("torch_dtype"):
        kwargs["torch_dtype"] = {"float16": torch.float16,
                                 "bfloat16": torch.bfloat16}[
                                     arm["torch_dtype"]]
    model = AutoModelForCausalLM.from_pretrained(
        arm["checkpoint_path"], local_files_only=True,
        **kwargs).to(device).eval()
    seed = 20260822

    def generate(prompt: str) -> str:
        torch.manual_seed(seed)
        messages = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(enc, max_new_tokens=int(
                arm["max_new_tokens"]), do_sample=False)
        return tok.decode(out[0][enc.shape[1]:],
                          skip_special_tokens=True).strip()

    problems = [json.loads(line) for line in open(
        config["corpus"]["path"], encoding="utf-8")]
    if smoke:
        problems = problems[:config["smoke"]["problems"]]

    k = int(config["loop"]["k"])
    fb_tpl = config["loop"]["feedback_prompt"]
    per_item: list[dict[str, Any]] = []
    throttle = 0.01  # the registered display-GPU TDR mitigation
    for i, p in enumerate(problems):
        attempts: list[dict[str, Any]] = []
        passed = False
        candidate = ""
        for attempt in range(1, k + 1):
            if attempt == 1:
                prompt = BASE_PROMPT.format(prompt=p["prompt"])
            else:
                prompt = fb_tpl.format(attempt=candidate,
                                       trace=attempts[-1]["stderr_tail"],
                                       prompt=p["prompt"])
            candidate = clean(generate(prompt))
            result = run_check(candidate, p["test"], p["entry_point"])
            attempts.append({"attempt": attempt, **result})
            if result["passed"]:
                passed = True
                break
        per_item.append({"task_id": p["task_id"], "passed": passed,
                         "n_attempts": len(attempts),
                         "attempts": attempts})
        if throttle:
            time.sleep(throttle)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(problems)} pass "
                  f"{sum(x['passed'] for x in per_item)}", flush=True)

    n = len(per_item)
    passes = sum(x["passed"] for x in per_item)
    evidence: dict[str, Any] = {
        "milestone": "M277",
        "cell": "execution-feedback loop on the code arm",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "results": {
            "n_problems": n,
            "pass_at_k": round(passes / n, 4) if n else None,
            "sealed_single_shot_pass_at_1": 0.5976,
            "delta": round((passes / n) - 0.5976, 4) if n else None,
            "mean_attempts": round(sum(x["n_attempts"] for x in
                                       per_item) / n, 3) if n else None,
        },
        "sandbox_fingerprint": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "flags": "-I -B, no shell, timeout 5s, temp cwd",
        },
        "per_item": per_item,
        "scope_note": ("pass@k over k greedy attempts with the sealed "
                       "sandbox's failure trace as the ONLY feedback; "
                       "product-quality reading; contamination declared"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / config["evidence_filename"], evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M277 complete -> "
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
    run_m277(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
