"""M176b — the fit-cost benchmark (the L4 wall) and one iterative escape.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` (section 7
Phase D M176b; section 12 dispatch entry, 17 Aug 2026).

- anchor: the SPM ridge read at the C4 138k level (0.2273623188405797,
  tol 1e-6) pins the fit pipeline.
- real widths: the ms357 codes (13,244) and the SPM pyramid (40,383,
  measured inside the anchor solve).
- synthetic widths: seeded Gaussian rows at 53,267 / 60,000 / 70,000
  columns — the Gram dominates, so n=20k is representative; the first
  width that raises MemoryError or exceeds the time budget is the
  measured wall.
- escape: scipy LSQR vs an exact solve on the same real-code Gram
  (G w = b, seeded b); registered drop-in criterion: rel <= 1e-9.

Smoke declares inadmissibility and refuses the sealed output directory.
"""
from __future__ import annotations

import argparse
import ctypes
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
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator, _score
from experiments.tier4.eval_v16_m142_c4 import _fit_power
from experiments.tier4.eval_v16_m142_factorial import power_norm
from experiments.tier4.eval_v16_m109_trunk import _load_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m176b_fit_wall.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v24"
                  / "m176b_fit_wall")

CLASSES = 345


def _peak_ram_mb() -> int:
    """Peak working-set size of THIS process (Windows)."""
    class _COUNTERS(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_COUNTERS), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    counters = _COUNTERS()
    counters.cb = ctypes.sizeof(_COUNTERS)
    handle = ctypes.c_void_p(kernel32.GetCurrentProcess())
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters),
                                      counters.cb):
        return -1
    return int(counters.PeakWorkingSetSize) // (1024 * 1024)


def _avail_ram_gb() -> float:
    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    stat = _MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPhys / (1024 ** 3)


def _gram_budget_ok(width: int) -> bool:
    """Registered pre-flight: 3x the float64 Gram must fit in free RAM
    (the M151 factor: a 23 GB Gram peaked ~70 GB with solver temps)."""
    gram_gb = width * width * 8 / (1024 ** 3)
    return 3.0 * gram_gb <= _avail_ram_gb()


def _bench_real(width, train_mem, labels, n_train, block, power):
    started = time.time()
    acc = RidgeAccumulator(width, CLASSES)
    for start in range(0, n_train, block):
        stop = min(start + block, n_train)
        acc.add(power_norm(train_mem[start:stop], power),
                labels[start:stop])
    solve_started = time.time()
    acc.solve_many([1.0])
    return {"width": width, "n_train": n_train,
            "seconds": round(time.time() - started, 2),
            "solve_seconds": round(time.time() - solve_started, 2),
            "peak_ram_mb": _peak_ram_mb(), "status": "ok"}


def _bench_synthetic(width, n_train, block, seed):
    if not _gram_budget_ok(width):
        return {"width": width, "status": "skipped_gram_budget",
                "note": "3x the float64 Gram exceeds free RAM (the M151 "
                        "factor); this width is the measured wall marker"}
    started = time.time()
    try:
        acc = RidgeAccumulator(width, CLASSES)
        rng = np.random.default_rng(seed)
        for start in range(0, n_train, block):
            stop = min(start + block, n_train)
            xs = rng.standard_normal((stop - start, width)).astype(
                np.float32)
            ys = rng.integers(0, CLASSES, stop - start)
            acc.add(xs, ys)
        solve_started = time.time()
        acc.solve_many([1.0])
        return {"width": width, "n_train": n_train,
                "seconds": round(time.time() - started, 2),
                "solve_seconds": round(time.time() - solve_started, 2),
                "peak_ram_mb": _peak_ram_mb(), "status": "ok"}
    except MemoryError:
        return {"width": width, "status": "MemoryError",
                "seconds": round(time.time() - started, 2),
                "peak_ram_mb": _peak_ram_mb()}


def _escape_check(train_mem, n_train, penalty, seed, width):
    """LSQR vs exact solve on the SAME real-code Gram, seeded b."""
    xs = train_mem[:n_train].astype(np.float64)
    gram = xs.T @ xs
    gram.flat[:: width + 1] += penalty
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(width)
    started = time.time()
    w_exact = np.linalg.solve(gram, b)
    exact_s = time.time() - started
    from scipy.sparse.linalg import lsqr
    started = time.time()
    w_iter, *_ = lsqr(gram, b, atol=1e-14, btol=1e-14, iter_lim=5000)
    iter_s = time.time() - started
    rel = float(np.linalg.norm(w_iter - w_exact)
                / max(np.linalg.norm(w_exact), 1e-30))
    return {"rel_difference": rel, "tolerance": 1e-9,
            "exact_seconds": round(exact_s, 2),
            "lsqr_seconds": round(iter_s, 2),
            "drop_in": bool(rel <= 1e-9)}


def run_m176b(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    smoke = inadmissible
    skip_anchors = bool(config.get("_smoke_skip_anchors", False))
    smoke_n = int(config.get("_smoke_n_train", 10 ** 9))
    smoke_widths = [int(w) for w in config.get("_smoke_widths", [])]

    configure_external_cache_environment()
    block = int(config["numerics"]["block"])
    power = 0.5
    corpus, _, _ = _load_corpus(config)

    evidence: dict[str, Any] = {
        "milestone": "M176b",
        "cell": "fit-cost benchmark + iterative-escape check",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
        "interpretation": config["interpretation_registered_before_running"],
    }

    # ---- anchor: SPM ridge (the pipeline pin) ------------------------------
    if not skip_anchors:
        print("anchor: SPM ridge @138k", flush=True)
        cache = data_cache_root() / config["anchor"]["spm_cache_relpath"]
        spm = np.load(cache / config["anchor"]["spm_train_file"],
                      mmap_mode="r")
        labels = np.load(cache / config["anchor"]["labels_file"])["labels"]
        spm_test = np.load(cache / config["anchor"]["spm_test_file"],
                           mmap_mode="r")
        test_labels = corpus["test_labels"]
        solved, std = _fit_power(spm, labels, power, [1.0],
                                 int(config["anchor"]["n_train"]), block,
                                 transform=True)
        hits = 0
        for start in range(0, len(test_labels), block):
            stop = min(start + block, len(test_labels))
            xs = std(power_norm(spm_test[start:stop], power))
            hits += int(_score(solved["1.0"], xs,
                                test_labels[start:stop]).sum())
        spm_acc = hits / len(test_labels)
        ref = float(config["anchor"]["spm_ridge_reference"])
        tol = float(config["anchor"]["spm_ridge_tolerance"])
        anchors = {"spm_ridge": {"measured": spm_acc, "sealed": ref,
                                 "delta": spm_acc - ref, "tolerance": tol}}
        print(f"  ridge {spm_acc:.6f} (delta {spm_acc - ref:+.3e})",
              flush=True)
        if abs(spm_acc - ref) > tol:
            evidence.update({"void": True,
                             "void_reason": "SPM ridge anchor failed",
                             "anchors": anchors})
            _write(output_dir, evidence)
            return evidence
        evidence["anchors"] = anchors
    else:
        anchors = {}

    results: dict[str, Any] = {}

    # ---- real widths --------------------------------------------------------
    cache = data_cache_root() / config["real_widths"]["ms357_cache_relpath"]
    ms = np.load(cache / config["real_widths"]["ms357_train_file"],
                 mmap_mode="r")
    labels = np.load(data_cache_root() / "v16" / "m142_c2"
                     / "m142_c2_fulltrain_labels.npz")["labels"]
    n_real = min(int(config["real_widths"]["n_train"]), smoke_n)
    print(f"real width {config['real_widths']['width']} @ n={n_real}",
          flush=True)
    results["real"] = [_bench_real(int(config["real_widths"]["width"]), ms,
                                   labels, n_real, block, power)]
    print(f"  {results['real'][0]}", flush=True)

    # ---- synthetic widths ---------------------------------------------------
    syn = config["synthetic"]
    widths = smoke_widths if smoke else [int(w) for w in syn["widths"]]
    n_syn = min(int(syn["n_train"]), smoke_n)
    synthetic = []
    for width in widths:
        print(f"synthetic width {width} @ n={n_syn}", flush=True)
        row = _bench_synthetic(width, n_syn, block, int(syn["seed"]))
        print(f"  {row}", flush=True)
        synthetic.append(row)
    results["synthetic"] = synthetic

    # ---- escape check -------------------------------------------------------
    esc = config["escape"]
    n_esc = min(int(esc["n_train"]), smoke_n)
    print(f"escape check on ms357 Gram (n={n_esc})", flush=True)
    try:
        escape = _escape_check(ms, n_esc, float(esc["penalty"]),
                               int(esc["seed"]),
                               int(config["real_widths"]["width"]))
        print(f"  {escape}", flush=True)
    except MemoryError:
        escape = {"status": "MemoryError"}
    results["escape"] = escape

    evidence.update({"results": results,
                     "runtime_seconds": round(time.time() - started, 2)})
    _write(output_dir, evidence)
    print(f"\nM176b complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def _write(output_dir: Path, evidence: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m176b(args.config, args.output)


if __name__ == "__main__":
    main()
