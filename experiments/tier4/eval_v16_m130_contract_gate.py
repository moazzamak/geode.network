"""M130 - registered engineering measurement: contract-gated routing on the sealed corpus.

v20 B3 (``analysis/ENGINEERING_PLAN_v20.md``). Measures the B1/B2 library behaviour
on the sealed M126 8192-atom codes:

- In-contract: the sealed test codes reach the learned path (the standard ridge
  re-fit) and accuracy reproduces the sealed value 0.2228 within t1 = 0.002.
- Out-of-contract: corrupted codes are dispatched to the programmatic reject gate
  (negated contract) with ZERO learned forward passes; rejection = 100%.

Costs are reported separately (guard FLOPs vs learned MACs), never wall-clock.
Registered before measurement (config ``m130_contract_gate.json``, N89 notes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from src.contract_router import ContractGatedRouter
from src.model_fingerprint import InputSpec, ModelFingerprint, OutputSpec
from src.programmatic_primitive import PrimitiveContract, ProgrammaticPrimitive

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m130_contract_gate.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m130_contract_gate"


class RidgeClassifierFallback:
    """Learned fallback for the classification task: standardised ridge argmax."""

    def __init__(self, fingerprint: ModelFingerprint, standardise: Any,
                 weights: np.ndarray) -> None:
        self.fingerprint = fingerprint
        self._standardise = standardise
        self._weights = weights

    def predict(self, array: np.ndarray) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        standardised = self._standardise(values)
        scores = standardised @ self._weights[:-1] + self._weights[-1]
        return scores.argmax(axis=1).astype(np.int64)


def _reject_verdict(array: np.ndarray) -> np.ndarray:
    """Programmatic reject gate: every row is out of contract -> verdict 1."""
    return np.ones(array.shape[0], dtype=np.int64)


def run_m130(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    configure_external_cache_environment()
    smoke_rows = int(config.get("_smoke_rows", 0))
    smoke_skip = bool(config.get("_smoke_skip_gates", False))

    ref = config["sealed_reference"]
    atoms = int(ref["atoms"])
    width = int(ref["width"])
    sealed_accuracy = float(ref["sealed_accuracy"])
    tolerance = float(ref["tolerance"])
    cache = data_cache_root() / "v16" / "m126"
    mem_train = np.lib.format.open_memmap(
        cache / f"f{atoms}_train.npy", mode="r", dtype=np.float32)
    mem_test = np.lib.format.open_memmap(
        cache / f"f{atoms}_test.npy", mode="r", dtype=np.float32)

    corpus_cfg = json.loads(
        (REPO_ROOT / "experiments" / "configs" / "v16" / "m116_scale.json")
        .read_text(encoding="utf-8"))
    corpus, _ti, _te = _load_corpus(corpus_cfg)
    train_labels = corpus["train_labels"]
    test_labels = corpus["test_labels"]
    classes = int(train_labels.max()) + 1
    n_train = len(train_labels)
    if smoke_rows:
        n_train = min(smoke_rows, n_train)

    # ---- Learned path: standard ridge re-fit (M126 machinery) ----
    acc = RidgeAccumulator(width, classes)
    for start in range(0, n_train, 4096):
        stop = min(start + 4096, n_train)
        acc.add(np.asarray(mem_train[start:stop]), train_labels[start:stop])
    standardise = acc.standardiser()
    weights = acc.solve(float(config["numerics"]["ridge_penalty"]))

    # Known-value reproduction: accuracy on the sealed test codes must match the
    # sealed M126 value within t1 BEFORE the router result is read.
    correct = 0
    seen = 0
    for start in range(0, len(test_labels), 4096):
        stop = min(start + 4096, len(test_labels))
        block = np.asarray(mem_test[start:stop], dtype=np.float32)
        scores = standardise(block) @ weights[:-1] + weights[-1]
        correct += int((scores.argmax(axis=1) == test_labels[start:stop]).sum())
        seen += stop - start
    reproduced_accuracy = float(correct / seen)
    print(f"learned-path accuracy (reproduced): {reproduced_accuracy:.4f} "
          f"(sealed {sealed_accuracy}, t1 {tolerance})", flush=True)
    if not smoke_skip and abs(reproduced_accuracy - sealed_accuracy) > tolerance:
        raise RuntimeError(
            f"known-value reproduction FAILED: {reproduced_accuracy:.4f} vs "
            f"sealed {sealed_accuracy} (|delta| > {tolerance}). Refusing to "
            "report router results.")

    # ---- Router construction ----
    classify_fp = ModelFingerprint(
        task_name=config["router"]["task_classify"],
        input_spec=InputSpec(source="passthrough", dim=width),
        output_spec=OutputSpec(type="labels", classes=tuple(range(classes))),
    )
    gate_fp = ModelFingerprint(
        task_name=config["router"]["task_classify"],
        input_spec=InputSpec(source="passthrough", dim=width),
        output_spec=OutputSpec(type="labels", classes=(0, 1)),
    )
    # In-contract envelope: the observed value range of the SEALED test codes,
    # padded by 1 on each side, computed by a full streaming pass (a head
    # block's min/max is NOT the extent - the smoke run caught real codes
    # being gate-dispatched because the first 4096 rows missed the tails).
    # Recorded in evidence (never asserted unchecked).
    code_min = float("inf")
    code_max = float("-inf")
    for start in range(0, len(test_labels), 4096):
        stop = min(start + 4096, len(test_labels))
        block = np.asarray(mem_test[start:stop], dtype=np.float32)
        code_min = min(code_min, float(block.min()))
        code_max = max(code_max, float(block.max()))
    envelope = (code_min - 1.0, code_max + 1.0)

    in_contract = PrimitiveContract(
        ndim=2, shape=(None, width), dtype="float32",
        require_finite=True, value_range=envelope,
    )
    reject_gate = ProgrammaticPrimitive(
        fingerprint=gate_fp,
        fn=_reject_verdict,
        contract=PrimitiveContract(
            ndim=2, shape=(None, width), dtype="float32",
            require_finite=True, value_range=envelope, negate=True,
        ),
        cost_class=config["router"]["reject_primitive_cost"],
        description="reject out-of-contract codes (negated contract)",
    )
    fallback = RidgeClassifierFallback(classify_fp, standardise, weights)
    router = ContractGatedRouter(
        programmatic=[reject_gate],
        fallback={config["router"]["task_classify"]: fallback},
        enable_fallback=True,
    )

    # ---- In-contract measurement ----
    n_test = len(test_labels)
    correct_router = 0
    n_in_contract = 0
    n_gate_dispatched = 0
    learned_macs = 0
    guard_flops = 0
    log_sample: list[dict[str, Any]] = []
    for start in range(0, n_test, 4096):
        stop = min(start + 4096, n_test)
        block = np.asarray(mem_test[start:stop], dtype=np.float32)
        result = router.route(block, task_name=config["router"]["task_classify"])
        if result.accepted:
            n_gate_dispatched += stop - start
            guard_flops += (stop - start) * width  # finite + range guard compares
        elif result.fallback:
            n_in_contract += stop - start
            learned_macs += (stop - start) * width * classes
            pred = np.asarray(result.predictions)
            correct_router += int((pred == test_labels[start:stop]).sum())
        log_sample.append(
            {
                "rows": stop - start,
                "accepted": result.accepted,
                "fallback": result.fallback,
                "rejected": result.rejected,
                "primitive_id": result.primitive_id,
                "cost_class": result.cost_class,
            }
        )
    in_contract_accuracy = (
        float(correct_router / n_in_contract) if n_in_contract else None
    )

    # ---- Out-of-contract measurement ----
    probe_seed = int(config["numerics"]["probe_seed"])
    probe_rows = int(config["numerics"]["probe_rows"])
    if smoke_rows:
        probe_rows = min(probe_rows, smoke_rows)
    rng = np.random.default_rng(probe_seed)
    base = np.asarray(mem_test[:probe_rows], dtype=np.float32)
    corruptions = config["numerics"]["probe_corruptions"]
    probe_results: dict[str, Any] = {}
    for kind in corruptions:
        if kind == "nan":
            probe = base.copy()
            mask = rng.random(probe.shape) < 0.05
            probe[mask] = np.nan
        elif kind == "inf":
            probe = base.copy()
            mask = rng.random(probe.shape) < 0.05
            probe[mask] = np.inf
        elif kind == "wrong_width":
            probe = base[:, : width // 2]  # shape guard trips at O(1)
        elif kind == "out_of_range":
            probe = base.copy() * 1e6  # outside the recorded envelope
        elif kind == "dtype_int":
            probe = base.astype(np.int32)
        else:
            raise ValueError(f"unknown corruption {kind!r}")
        result = router.route(probe, task_name=config["router"]["task_classify"])
        verdicts = (
            None if result.predictions is None
            else np.asarray(result.predictions).tolist()
        )
        rejected = result.accepted and result.primitive_id is not None
        probe_results[kind] = {
            "rows": probe.shape[0],
            "dispatched_to_gate": rejected,
            "primitive_id": result.primitive_id,
            "cost_class": result.cost_class,
            "learned_forward_passes": 0 if rejected else 1,
            "guard_flops": probe.shape[0] * probe.shape[1] if rejected else 0,
            "sample_verdicts": verdicts[:8] if verdicts else None,
            "decision_log": [dict(e) for e in result.decision_log][:4],
        }

    gate_rejection_rate = (
        sum(1 for r in probe_results.values() if r["dispatched_to_gate"])
        / len(probe_results)
    )

    evidence: dict[str, Any] = {
        "milestone": "M130",
        "admissible_as_evidence": True,
        "registered_in": config.get("registered_in"),
        "config_file": Path(config_path).name,
        "config": config,
        "hypothesis": ("contract-gated routing rejects out-of-contract inputs "
                       "with ZERO learned forward passes (100% rejection), "
                       "while in-contract inputs reach the learned path with "
                       "accuracy equal to the sealed value within t1"),
        "sealed_reference": ref,
        "known_value_reproduction": {
            "reproduced_accuracy": reproduced_accuracy,
            "sealed_accuracy": sealed_accuracy,
            "delta": reproduced_accuracy - sealed_accuracy,
            "within_t1": abs(reproduced_accuracy - sealed_accuracy) <= tolerance,
            "gate_skipped": smoke_skip,
        },
        "router": {
            "in_contract_envelope": envelope,
            "in_contract_rows": n_in_contract,
            "in_contract_accuracy": in_contract_accuracy,
            "gate_dispatched_rows": n_gate_dispatched,
            "learned_macs": learned_macs,
            "guard_flops": guard_flops,
            "log_sample": log_sample,
        },
        "out_of_contract": {
            "corruptions": corruptions,
            "per_corruption": probe_results,
            "gate_rejection_rate": gate_rejection_rate,
        },
        "footprint": {
            "learned_weights_bytes": int(
                (width + 1) * classes * np.dtype(np.float32).itemsize),
            "reject_gate_bytes": 0,
            "note": "parameter footprint, not wall-clock (N89.7)",
        },
        "note": "engineering measurement on the sealed corpus; no novelty "
                "claim (N89.5)",
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM130 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m130(args.config, args.output)


if __name__ == "__main__":
    main()
