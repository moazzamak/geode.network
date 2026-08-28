"""Verify and benchmark the ROCm PyTorch environment against the CPU build.

Run with the ROCm interpreter::

    .\\.venv-rocm\\Scripts\\python.exe tools\\benchmark_rocm.py

This checks that the GPU is genuinely reachable (a real matmul whose result is
validated against the CPU), then measures throughput on the shapes the GEODE
v13 milestones actually use, plus the large shapes that later milestones will
introduce. It also probes the two properties that determine whether the sealed
replay milestones could ever move to GPU: float64 support and deterministic
algorithm coverage.
"""

from __future__ import annotations

import json
import os
import platform
import time

from rocm_device import (  # noqa: E402 - must run before torch is imported
    SUPPORTED_ARCHITECTURE_PREFIX,
    ensure_discrete_gpu,
)

ensure_discrete_gpu()

import torch  # noqa: E402 - deliberately imported after the device is pinned


def _select_device() -> torch.device:
    """Return the first device the installed ROCm kernels actually support."""
    for index in range(torch.cuda.device_count()):
        architecture = torch.cuda.get_device_properties(index).gcnArchName
        if architecture.startswith(SUPPORTED_ARCHITECTURE_PREFIX):
            torch.cuda.set_device(index)
            return torch.device("cuda", index)
    raise RuntimeError(
        "No GPU matching "
        f"{SUPPORTED_ARCHITECTURE_PREFIX!r} is visible; "
        "the installed ROCm wheel cannot drive the enumerated adapters."
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_matmul(
    device: torch.device, size: int, dtype: torch.dtype, iterations: int
) -> dict[str, float]:
    left = torch.randn(size, size, device=device, dtype=dtype)
    right = torch.randn(size, size, device=device, dtype=dtype)
    for _ in range(3):  # warm up kernels and autotuning
        left @ right
    _sync(device)
    start = time.perf_counter()
    for _ in range(iterations):
        left @ right
    _sync(device)
    elapsed = time.perf_counter() - start
    flops = 2.0 * size**3 * iterations
    return {
        "seconds": elapsed,
        "tflops": flops / elapsed / 1e12,
    }


def _correctness(device: torch.device) -> dict[str, object]:
    """A result the GPU cannot produce by accident."""
    generator = torch.Generator().manual_seed(20260729)
    left = torch.randn(512, 512, generator=generator, dtype=torch.float32)
    right = torch.randn(512, 512, generator=generator, dtype=torch.float32)
    expected = left @ right
    actual = (left.to(device) @ right.to(device)).cpu()
    difference = float((expected - actual).abs().max())
    return {
        "max_absolute_difference": difference,
        "matches_cpu": difference < 1e-2,
    }


def _float64_support(device: torch.device) -> dict[str, object]:
    try:
        left = torch.randn(256, 256, device=device, dtype=torch.float64)
        result = left @ left
        return {"supported": True, "finite": bool(torch.isfinite(result).all())}
    except Exception as error:  # noqa: BLE001 - reporting a capability probe
        return {"supported": False, "error": f"{type(error).__name__}: {error}"}


def _determinism_support(device: torch.device) -> dict[str, object]:
    """Does the repo's global determinism contract hold on this device?"""
    torch.use_deterministic_algorithms(True)
    try:
        source = torch.randn(4096, 128, device=device)
        index = torch.randint(0, 4096, (8192,), device=device)
        gathered = source.index_select(0, index)
        target = torch.zeros(4096, 128, device=device)
        target.index_add_(0, index, gathered)
        _sync(device)
        return {"supported": True}
    except Exception as error:  # noqa: BLE001 - reporting a capability probe
        return {"supported": False, "error": f"{type(error).__name__}: {error}"}
    finally:
        torch.use_deterministic_algorithms(False)


def main() -> None:
    report: dict[str, object] = {
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "gpu_available": torch.cuda.is_available(),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", "<unset>"),
    }

    if not torch.cuda.is_available():
        report["verdict"] = "NO GPU BACKEND REACHABLE"
        print(json.dumps(report, indent=2))
        return

    report["enumerated_devices"] = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "architecture": torch.cuda.get_device_properties(index).gcnArchName,
        }
        for index in range(torch.cuda.device_count())
    ]

    device = _select_device()
    properties = torch.cuda.get_device_properties(device.index)
    report["selected_device_index"] = device.index
    report["device_name"] = torch.cuda.get_device_name(device.index)
    report["device_capability"] = getattr(properties, "gcnArchName", "unknown")
    report["device_total_memory_gb"] = round(properties.total_memory / 1024**3, 2)
    report["correctness"] = _correctness(device)
    report["float64"] = _float64_support(device)
    report["determinism"] = _determinism_support(device)

    benchmarks: dict[str, object] = {}
    cpu = torch.device("cpu")
    for size, iterations in ((1024, 20), (4096, 10), (8192, 5)):
        key = f"matmul_fp32_{size}"
        gpu_result = _time_matmul(device, size, torch.float32, iterations)
        cpu_result = _time_matmul(cpu, size, torch.float32, max(1, iterations // 5))
        benchmarks[key] = {
            "gpu_tflops": round(gpu_result["tflops"], 2),
            "cpu_tflops": round(cpu_result["tflops"], 2),
            "speedup": round(gpu_result["tflops"] / cpu_result["tflops"], 1),
        }

    if report["float64"]["supported"]:
        gpu_fp64 = _time_matmul(device, 2048, torch.float64, 5)
        cpu_fp64 = _time_matmul(cpu, 2048, torch.float64, 2)
        benchmarks["matmul_fp64_2048"] = {
            "gpu_tflops": round(gpu_fp64["tflops"], 3),
            "cpu_tflops": round(cpu_fp64["tflops"], 3),
            "speedup": round(gpu_fp64["tflops"] / cpu_fp64["tflops"], 1),
        }

    # The shape a v13 metric-field training step actually uses.
    geode = torch.randn(1920, 384, device=device, dtype=torch.float64)
    _sync(device)
    start = time.perf_counter()
    for _ in range(100):
        geode @ geode.T
    _sync(device)
    benchmarks["geode_v13_shape_1920x384_fp64_100x"] = {
        "gpu_seconds": round(time.perf_counter() - start, 4)
    }
    geode_cpu = geode.cpu()
    start = time.perf_counter()
    for _ in range(100):
        geode_cpu @ geode_cpu.T
    benchmarks["geode_v13_shape_1920x384_fp64_100x"]["cpu_seconds"] = round(
        time.perf_counter() - start, 4
    )

    report["benchmarks"] = benchmarks
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()