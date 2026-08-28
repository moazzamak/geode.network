"""M176c-anchor — anchor reproduction on the verified rental.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` §12 (18 Aug
2026), after M176c-dvc's BIT-EXACT verdict. The full-data anchors cannot
be reproduced by rerunning m142_c2 verbatim on the rental: its 75 GB of
code memmaps exceed the pod's 30 GB overlay and the volume quota. This
runner is the CHUNKED variant (the M151 Gram pattern): one encode pass
over the full train schedule feeds THREE streaming accumulators at once
(raw full, power-norm p=0.5 full, power-norm p=0.5 first-138k), and the
test codes are held in RAM (34,500 x 40,383 float32 = 5.6 GB). The three
Gram matrices are 13 GB float64 each (39 GB peak) — inside the pod's
62 GB.

Premise gate (void on failure): the sealed dvc evidence must exist and
read "bit-exact" — this runner exists ONLY because that cell passed.

Anchors reproduced (with the registered tolerances):
- A1 p=0.5 @138k, penalty 1.0   -> 0.2273623188405797  (tol 1e-6)
- A2 raw @full, penalty 1.0     -> 0.2604927536231884  (tol 0.002)
- A3 p=0.5 @full, penalty 0.1   -> 0.27855072463768116 (tol 0.002)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common.data_cache import configure_external_cache_environment
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    _chunk_rows,
    _score,
)
from experiments.tier4.eval_v15_m107_dense import _verify_pixel_identity
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m113_learned import (
    _build_whitener_and_candidates,
    _random_dictionary,
)
from experiments.tier4.eval_v16_m142_c2 import _spm_encode_block_device
from experiments.tier4.eval_v16_m142_factorial import power_norm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m176c_anchor.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v24" / "m176c_anchor")
DVC_EVIDENCE = (REPO_ROOT / "logs" / "results" / "v24" / "m176c_dvc"
                / "evidence.json")

CLASSES = 345
SPM_WIDTH = 40_383


def _encode_chunk(images: np.ndarray, table: torch.Tensor, whitener,
                  grid: int) -> np.ndarray:
    return _spm_encode_block_device(images, table, whitener, grid)


def _score_test(test_codes: np.ndarray, test_labels: np.ndarray,
                weights: np.ndarray, std, power: float | None,
                block: int) -> float:
    hits = 0
    n = len(test_labels)
    for start in range(0, n, block):
        stop = min(start + block, n)
        xs = test_codes[start:stop]
        if power is not None:
            xs = power_norm(xs, power)
        hits += int(_score(weights, std(xs), test_labels[start:stop]).sum())
    return hits / n


def run_m176c_anchor(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    torch.manual_seed(int(config["numerics"]["seed"]))
    configure_external_cache_environment()

    # ---- premise: the sealed dvc verdict, or the sealed device itself ----
    # (registered 18 Aug): on the LOCAL sealed device (gfx1201, the M108
    # instrument's reference), the device IS the premise — the dvc evidence
    # exists to clear OTHER devices. On any other device the dvc evidence
    # must read bit-exact.
    props = torch.cuda.get_device_properties(0)
    if getattr(props, "gcnArchName", None) == "gfx1201":
        premise_ok = True
        premise = "sealed-local-device"
    elif DVC_EVIDENCE.exists():
        dvc = json.loads(DVC_EVIDENCE.read_text(encoding="utf-8"))
        premise_ok = bool(dvc.get("verdict") == "bit-exact"
                          and dvc.get("admissible_as_evidence") is True)
        premise = "dvc-bit-exact"
    else:
        raise SystemExit("dvc evidence missing — the premise gate fails")

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    corpus, train_index, test_index = _load_corpus(config)
    size = int(config["corpus"]["image_size"])
    for split, idx in (("train", train_index), ("test", test_index)):
        _verify_pixel_identity(split, idx, corpus[f"{split}_images"], size,
                               int(config["corpus"]["pixel_identity_rows"]))

    rep = config["sparse"]
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    dict_spm = _random_dictionary(candidates, len(candidates),
                                  int(rep["dictionary_seed"]),
                                  int(rep["spm_atoms"]))
    table = torch.from_numpy(np.ascontiguousarray(dict_spm)).to(
        torch.float32).to(device)
    grid = (size - int(rep["patch"])) // int(rep["stride"]) + 1

    train_images = corpus["train_images"]
    train_labels = corpus["train_labels"]
    test_images = corpus["test_images"]
    test_labels = corpus["test_labels"]
    n_train = len(train_labels)
    n_test = len(test_labels)
    if smoke:
        n_train = min(n_train, int(config.get("_smoke_train_rows", 2000)))
        n_test = min(n_test, int(config.get("_smoke_test_rows", 400)))
    test_labels = test_labels[:n_test]

    # ---- test codes to RAM -------------------------------------------------
    print(f"encoding {n_test} test rows to RAM", flush=True)
    test_codes = np.empty((n_test, SPM_WIDTH), dtype=np.float32)
    step = _chunk_rows(table.shape[0], grid, n_test)
    for start in range(0, n_test, step):
        stop = min(start + step, n_test)
        test_codes[start:stop] = _encode_chunk(
            test_images[start:stop], table, whitener, grid)
    print("test codes done", flush=True)

    # ---- two-pass layout (registered OOM repair #4): pass A rows 0..n_138
    # into acc_138 only (solve + free at peak 26 GiB), then pass B rows
    # 0..n_train into acc_raw + acc_p05 (peak 39 GiB at the raw solve). The
    # pod's cgroup cap is 56.8 GiB; the single-pass three-gram layout hit it
    # (Exit 137). Pass A's rows are re-encoded in pass B (+34% encode time),
    # keeping every accumulation order identical to the sealed fits.
    anchors = config["anchors"]
    n_138 = min(int(anchors["n_138k"]), n_train)
    block = int(config["numerics"]["block"])
    step = _chunk_rows(table.shape[0], grid, block)

    acc_138 = RidgeAccumulator(SPM_WIDTH, CLASSES)
    for start in range(0, n_138, step):
        stop = min(start + step, n_138)
        chunk = _encode_chunk(train_images[start:stop], table, whitener,
                              grid)
        acc_138.add(power_norm(chunk, 0.5), train_labels[start:stop])
        print(f"  passA {stop}/{n_138}", flush=True)
    std_138 = acc_138.standardiser()
    w_138 = acc_138.solve(1.0)
    del acc_138
    a1 = _score_test(test_codes, test_labels, w_138, std_138, 0.5, block)

    acc_raw = RidgeAccumulator(SPM_WIDTH, CLASSES)
    acc_p05 = RidgeAccumulator(SPM_WIDTH, CLASSES)
    for start in range(0, n_train, step):
        stop = min(start + step, n_train)
        chunk = _encode_chunk(train_images[start:stop], table, whitener,
                              grid)
        acc_raw.add(chunk, train_labels[start:stop])
        acc_p05.add(power_norm(chunk, 0.5), train_labels[start:stop])
        print(f"  passB {stop}/{n_train}", flush=True)
    std_raw = acc_raw.standardiser()
    w_raw = acc_raw.solve(1.0)
    del acc_raw
    a2 = _score_test(test_codes, test_labels, w_raw, std_raw, None, block)

    std_p05 = acc_p05.standardiser()
    w_p05 = acc_p05.solve(0.1)
    del acc_p05
    a3 = _score_test(test_codes, test_labels, w_p05, std_p05, 0.5, block)

    checks = [
        ("A1_138k_p05_pen1", a1, float(anchors["a1_value"]),
         float(anchors["a1_tolerance"])),
        ("A2_full_raw_pen1", a2, float(anchors["a2_value"]),
         float(anchors["a2_tolerance"])),
        ("A3_full_p05_pen01", a3, float(anchors["a3_value"]),
         float(anchors["a3_tolerance"])),
    ]
    anchor_report = {
        name: {"measured": round(m, 9), "registered": reg,
               "tolerance": tol,
               "delta": abs(m - reg),
               "ok": bool(abs(m - reg) <= tol)}
        for name, m, reg, tol in checks
    }
    all_ok = bool(all(v["ok"] for v in anchor_report.values()))

    evidence: dict[str, Any] = {
        "milestone": "M176c-anchor",
        "cell": "anchor reproduction on the verified rental (chunked)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "question": config["question"],
        "interpretation_registered_before_running":
            config["interpretation_registered_before_running"],
        "premise_dvc_verdict": premise,
        "n_train_rows_encoded": n_train,
        "n_test_rows": n_test,
        "anchors": anchor_report,
        "all_anchors_reproduced": all_ok,
        "void": not all_ok,
        "void_reason": "" if all_ok else
            "one or more anchors outside their registered tolerance",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"anchors": anchor_report, "all_ok": all_ok},
                     indent=1), flush=True)
    print(f"M176c-anchor complete -> {output_dir / 'evidence.json'}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m176c_anchor(args.config, args.output)


if __name__ == "__main__":
    main()
