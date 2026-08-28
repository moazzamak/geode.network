from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.tier5.eval_editability_scaling import (
    SCALING_AXES,
    _models,
    _routing_counts,
    _scores,
)
from src.model_editor import ModelEditor


AXES = (*SCALING_AXES, "batch_size")


def _conditions(config: dict) -> list[dict]:
    baseline = {axis: int(config["baseline"][axis]) for axis in AXES}
    conditions = [{"name": "baseline", "axis": "baseline", **baseline}]
    for axis in AXES:
        for raw_value in config["sweeps"][axis]:
            value = int(raw_value)
            if value == baseline[axis]:
                continue
            condition = baseline.copy()
            condition[axis] = value
            conditions.append({
                "name": f"{axis}={value}",
                "axis": axis,
                **condition,
            })
    if any(condition[axis] <= 0 for condition in conditions for axis in AXES):
        raise ValueError("All scaling values must be positive.")
    return conditions


def _windows_peak_working_set_bytes() -> int | None:
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    )
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(
        process,
        ctypes.byref(counters),
        counters.cb,
    ):
        return None
    return int(counters.PeakWorkingSetSize)


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _run_condition(condition: dict, config: dict) -> dict:
    models = _models(condition, config["seed"])
    rng = np.random.default_rng(config["seed"])
    points = rng.normal(
        scale=3.0,
        size=(condition["batch_size"], condition["dimensions"]),
    )
    _scores(models, points)
    latencies = []
    for _ in range(config["timing_repeats"]):
        started = time.perf_counter()
        _scores(models, points)
        latencies.append(time.perf_counter() - started)

    tracemalloc.start()
    _scores(models, points)
    _, peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    latency = _percentiles(latencies)
    counters = _routing_counts(models, len(points))
    class_pairs = condition["batch_size"] * condition["class_count"]
    primitive_pairs = class_pairs * condition["primitives_per_class"]
    counter_invariants = {
        "compatible_pairs_equal_n_times_c": (
            counters.compatible_candidate_pairs == class_pairs
        ),
        "shortlisted_pairs_equal_compatible_pairs": (
            counters.shortlisted_candidate_pairs == class_pairs
        ),
        "exact_class_pairs_equal_n_times_c": (
            counters.exact_class_sdf_pairs == class_pairs
        ),
        "primitive_pairs_equal_n_times_c_times_k": (
            counters.primitive_sdf_pairs == primitive_pairs
        ),
        "score_values_equal_n_times_c": (
            counters.score_values_materialized == class_pairs
        ),
    }
    if not all(counter_invariants.values()):
        raise AssertionError(f"Exhaustive counter invariant failed: {counter_invariants}")
    return {
        "condition": condition,
        "latency_seconds": latency,
        "throughput_samples_per_second": {
            name: condition["batch_size"] / value
            for name, value in latency.items()
        },
        "peak_python_allocation_bytes": int(peak_python_bytes),
        "process_peak_working_set_bytes": _windows_peak_working_set_bytes(),
        "serialized_model_bytes": len(ModelEditor(models).snapshot()),
        "routing_counters": asdict(counters),
        "counter_invariants": counter_invariants,
    }


def _slope(records: list[dict], axis: str, output) -> float:
    values = [record for record in records if record["condition"]["axis"] in {
        "baseline", axis,
    }]
    unique = {}
    for record in values:
        unique[record["condition"][axis]] = output(record)
    x = np.asarray(sorted(unique), dtype=np.float64)
    y = np.asarray([unique[value] for value in x], dtype=np.float64)
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def run_exhaustive_routing_cost(config: dict) -> dict:
    records = [_run_condition(condition, config) for condition in _conditions(config)]
    slopes = {
        axis: {
            "latency_p50_log_log_slope": _slope(
                records, axis, lambda item: item["latency_seconds"]["p50"],
            ),
            "primitive_sdf_pairs_log_log_slope": _slope(
                records, axis, lambda item: item["routing_counters"][
                    "primitive_sdf_pairs"
                ],
            ),
            "serialized_bytes_log_log_slope": _slope(
                records, axis, lambda item: item["serialized_model_bytes"],
            ),
        }
        for axis in AXES
    }
    return {
        "milestone": "M12.0",
        "protocol": {
            "design": "one_factor_at_a_time",
            "routing": "exhaustive_exact",
            "timing_repeats": config["timing_repeats"],
            "warmup_runs": 1,
            "shortlisting_enabled": False,
            "ray_marching_included": False,
            "memory_measures": [
                "tracemalloc_peak_python_bytes",
                "process_peak_working_set_bytes",
                "canonical_serialized_model_bytes",
            ],
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "execution_provider": "numpy_cpu",
            },
        },
        "baseline": config["baseline"],
        "all_counter_invariants_passed": all(
            all(record["counter_invariants"].values()) for record in records
        ),
        "slopes": slopes,
        "conditions": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_exhaustive_routing_cost(config)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
