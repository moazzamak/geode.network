"""M263 — generative LLM-style arm: frozen Qwen2.5-1.5B-Instruct
with a measured refusal probe, the OOD input guard, and
prompt/output-hash ledger records.

Registered and dispatched 21 Aug 2026 (plan v25, M263 + amendment
18), local-first, F: caches. Honest boundary (registered): GEODE
does not train LLMs — this arm is our measurement, registration,
routing, guards, and ledger over a publisher checkpoint
(Apache-2.0), never our own pretraining. The refusal phrase
heuristic is a REGISTERED instrument, recorded as such.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
                  / "m263_generative_arm.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m263_generative_arm")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_m263(config_path: Path, output_dir: Path, smoke: bool = False
             ) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    cache_root = data_cache_root() / config["feature_cache_relpath"]
    cache_root.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = config["generation"]["device"]
    if not torch.cuda.is_available():
        device = "cpu"
    torch.backends.cudnn.enabled = False  # registered env note (M267)
    checkpoint = config["checkpoint"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True).to(device).eval()
    max_new = int(config["generation"]["max_new_tokens"])
    seed = int(config["generation"]["seed"])

    def generate(prompt: str) -> tuple[str, float]:
        torch.manual_seed(seed)
        messages = [{"role": "user", "content": prompt}]
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(enc, max_new_tokens=max_new,
                                 do_sample=False)
        elapsed = time.time() - t0
        text = tokenizer.decode(out[0][enc.shape[1]:],
                                skip_special_tokens=True).strip()
        return text, elapsed

    def is_refusal(text: str) -> bool:
        low = text.lower()
        return any(phrase in low
                   for phrase in config["refusal"]["phrase_heuristic"])

    # ---- input guard: structural + OodGate on BERT features ----------
    from transformers import AutoModel as BertModel
    from transformers import AutoTokenizer as BTok
    bert = BertModel.from_pretrained(
        "bert-base-uncased", local_files_only=True).to(device).eval()
    btok = BTok.from_pretrained("bert-base-uncased",
                                local_files_only=True)
    bert_vocab = set(btok.get_vocab().keys())

    def bert_features(texts: list[str]) -> np.ndarray:
        out = []
        for text in texts:
            e = btok(text, padding=True, truncation=True, max_length=128,
                     return_tensors="pt")
            e = {k: v.to(device) for k, v in e.items()}
            with torch.no_grad():
                h = bert(**e).last_hidden_state
            mask = e["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            out.append(pooled.cpu().numpy().astype(np.float32)[0])
        return np.stack(out)

    from geode.core.ood import OodGate
    gate = OodGate(threshold=float(config["guard"]["threshold"]))
    # self-contained reference: 5,000 MNLI premises through frozen
    # BERT (the M262 protocol), cached on F: for reuse
    ref_cache = cache_root.parent / "m263_guard_reference_5000_feat.npy"
    if ref_cache.exists():
        ref_list = np.load(ref_cache, mmap_mode="r")[:5000].astype(
            np.float64).tolist()
    else:
        from datasets import load_dataset as _hf_load
        ref_ds = _hf_load("multi_nli", split="train").select(
            range(5000))
        ref_texts = [r["premise"] for r in ref_ds]
        ref_feats = bert_features(ref_texts).astype(np.float64)
        np.save(ref_cache, ref_feats.astype(np.float32))
        ref_list = ref_feats.tolist()
    gate.fit_profile(ref_list)
    del ref_list

    def guard_admits(prompt: str) -> dict[str, Any]:
        if not prompt.strip():
            return {"admitted": False, "reason": "empty"}
        if len(prompt) > 512 or len(prompt.split()) > 512:
            return {"admitted": False, "reason": "too_long"}
        if not prompt.isprintable():
            return {"admitted": False, "reason": "not_printable"}
        # registered vocab-coverage primitive (deterministic, license-
        # free): fraction of whitespace tokens present in the BERT
        # vocab. Token soup / base64 / log dumps score near zero;
        # natural English scores high.
        tokens = prompt.lower().split()
        ratio: float | None = None
        if tokens:
            ratio = sum(1 for t in tokens if t in bert_vocab) / len(tokens)
            if ratio < float(config["guard"]["vocab_ratio_threshold"]):
                return {"admitted": False, "reason": "vocab_coverage",
                        "vocab_ratio": round(ratio, 4)}
        vec = bert_features([prompt])[0].astype(np.float64)
        verdict = gate.admits(vec.tolist())
        verdict.setdefault("vocab_ratio",
                           round(ratio, 4) if tokens else None)
        return verdict

    # ---- ledger + arms (M270 streaming contract) ----------------------
    from geode.core.arm import arm_from_sealed_head
    from geode.core.orchestrator import Orchestrator
    orch = Orchestrator()

    probes = {"benign": config["refusal"]["benign_prompts"],
              "refusal_expected":
                  config["refusal"]["refusal_expected_prompts"]}
    if smoke:
        probes = {"benign": probes["benign"][:2],
                  "refusal_expected": probes["refusal_expected"][:2]}
    results: dict[str, list[dict[str, Any]]] = {}
    latencies: list[float] = []
    for kind, prompts in probes.items():
        results[kind] = []
        for idx, prompt in enumerate(prompts):
            guard = guard_admits(prompt)
            if not guard["admitted"]:
                results[kind].append({"prompt_digest":
                                      _sha256_hex(prompt.encode()),
                                      "guarded": guard,
                                      "output": "", "refusal": None,
                                      "latency_s": 0.0})
                orch.ledger.append({"kind": "stage_abstained",
                                    "key": f"m263:{kind}:{idx}:guard",
                                    "item_index": idx,
                                    "reason": guard["reason"]})
                continue
            text, elapsed = generate(prompt)
            latencies.append(elapsed)
            refusal = is_refusal(text)
            results[kind].append({"prompt_digest":
                                  _sha256_hex(prompt.encode()),
                                  "guarded": guard,
                                  "output": text,
                                  "output_digest":
                                      _sha256_hex(text.encode()),
                                  "refusal": refusal,
                                  "latency_s": round(elapsed, 4)})
            # prompt/output-hash ledger records (M270 C3 contract)
            orch.stream_begin(f"m263:{kind}:{idx}", route_record_index=0,
                              seed=str(seed))
            orch.stream_chunk(f"m263:{kind}:{idx}", 0,
                              _sha256_hex(text.encode()))
            orch.stream_end(f"m263:{kind}:{idx}", 1,
                            _sha256_hex(text.encode()), "complete")
            if (len(latencies)) % 5 == 0:
                print(f"  {kind} {len(latencies)} done", flush=True)

    def rate(kind: str) -> float:
        rows = [r for r in results[kind] if r.get("refusal") is not None]
        return float(np.mean([r["refusal"] for r in rows])) if rows \
            else float("nan")

    benign_rate = rate("benign")
    refusal_rate = rate("refusal_expected")
    lat = np.asarray(latencies)
    latency_stats = {"p50": float(np.percentile(lat, 50)),
                     "p99": float(np.percentile(lat, 99)),
                     "n": int(len(lat))} if len(lat) else {}

    ood_results = []
    for idx, probe in enumerate(config["guard"]["ood_probes"]):
        verdict = guard_admits(probe)
        ood_results.append({"probe_digest": _sha256_hex(probe.encode()),
                            "admitted": verdict["admitted"],
                            "reason": verdict["reason"],
                            "score": verdict.get("score")})

    # ---- arm registered AFTER measurement (honest numbers only) ------
    arm_acc = float(1.0 - benign_rate)  # benign-completion rate
    spec = arm_from_sealed_head(
        "m263_qwen_generative", "text_generation", 0, arm_acc,
        config["checkpoint"]["path"],
        license={"code": "Apache-2.0", "weights": "Apache-2.0",
                 "data": ""})
    orch.register(spec)
    orch.serve("m263-arm", [], task_id=None)

    evidence: dict[str, Any] = {
        "milestone": "M263",
        "cell": "generative LLM-style arm (frozen Qwen2.5-1.5B-Instruct)",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "checkpoint": {"path": config["checkpoint"]["path"],
                       "license": config["checkpoint"][
                           "license_recorded"],
                       "frozen": True},
        "refusal_probe": {
            "benign_false_refusal_rate": benign_rate,
            "refusal_expected_refusal_rate": refusal_rate,
            "instrument": "registered phrase heuristic",
            "n_benign": len(probes["benign"]),
            "n_refusal_expected": len(probes["refusal_expected"]),
            "reading": ("the heuristic records refusal SURFACE "
                        "behaviour on the registered probe set; it is "
                        "not a harm-semantics claim"),
        },
        "input_guard": {"reference": config["guard"]["reference"],
                        "threshold": config["guard"]["threshold"],
                        "ood_probes": ood_results,
                        "ood_probes_flagged": sum(
                            1 for r in ood_results if not r["admitted"])},
        "latency": latency_stats,
        "ledger": {"verified": orch.chain_verify()["ok"],
                   "record_count": orch.chain_verify()["record_count"],
                   "reading": ("every generation carries prompt/output "
                               "digests via the M270 streaming contract "
                               "+ a route record")},
        "results": results,
        "pending_not_claimed": config["pending_not_claimed"],
        "scope_note": ("publisher checkpoint frozen, never trained; "
                       "greedy seeded generation; honest instrument "
                       "notes registered"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"benign_false_refusal_rate": benign_rate,
                      "refusal_expected_refusal_rate": refusal_rate,
                      "latency": latency_stats,
                      "ood_flagged":
                          evidence["input_guard"]["ood_probes_flagged"],
                      "ledger_ok": evidence["ledger"]["verified"]},
                     indent=1), flush=True)
    print(f"M263 complete -> {output_dir / 'evidence.json'}", flush=True)
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
    run_m263(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
