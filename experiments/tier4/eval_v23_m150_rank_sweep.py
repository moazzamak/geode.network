"""M150 — rank-and-profile sweep of the cached code family at 138k.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md`` (section 4
M150; 16 Aug 2026). A measuring stick: per cached code, compute the
standardised-Gram profile (participation-ratio effective rank, spectrum
shares, condition number — the M128/M138 definition), the ridge refit at
138k (the anchor), and a trained-head read at 138k under the M109 shared
schedule, then report the statistic-vs-outcome table.

Anchors (before any new number is read): each code's ridge refit at 138k
reproduces its sealed 138k read at penalty 1.0 within 1e-9 (penalty 1.0
everywhere, including the sqrt variants — the C4 cells_138k protocol).

No gate. Cost: fits + Gram accumulations (CPU) and six head-only trained
fits (GPU). Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v23_m150_rank_sweep
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
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
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v16_m109_trunk import (
    _load_corpus,
    _train_with_schedule,
)
from experiments.tier4.eval_v16_m121_spectrum import _standardised_gram
from experiments.tier4.eval_v16_m142_c4 import _fit_power, _score_power
from experiments.tier4.eval_v16_m142_factorial import power_norm
from experiments.tier4.eval_v16_m146_arbiter import HeadOnly

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v23"
                  / "m150_rank_sweep.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v23" / "m150_rank_sweep"

CLASSES = 345
TOLERANCE = 1e-9


def _participation_ratio(vals: np.ndarray) -> float:
    """trace^2 / sum(lambda^2) over the kept positive eigenvalues."""
    keep = vals > max(float(vals.max()) * 1e-10, 1e-12)
    positive = vals[keep]
    trace = float(positive.sum())
    return float(trace ** 2 / (positive ** 2).sum()) if trace > 0 else 0.0


def _spectrum_shares(vals: np.ndarray, keep_mask: np.ndarray,
                     top: int = 10) -> dict[str, float]:
    positive = vals[keep_mask]
    trace = float(positive.sum())
    topk = positive[:top].sum() if trace > 0 else 0.0
    return {
        "top1_share": float(positive[0] / trace) if trace > 0 else 0.0,
        f"top{top}_share": float(topk / trace) if trace > 0 else 0.0,
        "condition": (float(positive[0] / positive[-1])
                      if trace > 0 and positive[-1] > 0 else float("inf")),
    }


def _codes_factory(mem: np.ndarray, labels: np.ndarray, rows: np.ndarray,
                   power: float | None, batch: int, device: torch.device
                   ) -> Callable[[], Iterator]:
    def gen() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for start in range(0, len(rows), batch):
            take = rows[start:start + batch]
            block = np.asarray(mem[take], dtype=np.float64)
            if power is not None:
                block = power_norm(block, power)
            yield (torch.from_numpy(np.ascontiguousarray(
                block.astype(np.float32))).to(device),
                torch.from_numpy(labels[take]).to(device))
    return gen


def run_m150(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    smoke = inadmissible
    skip_anchors = bool(config.get("_smoke_skip_anchors", False))
    smoke_train = int(config.get("_smoke_train_rows", 10 ** 9))
    smoke_epochs = int(config.get("_smoke_epochs", 10 ** 9))
    block = int(config["numerics"]["block"])

    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    torch.manual_seed(109)
    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    m142_cache = data_cache_root() / config["artifacts"]["m142_cache_relpath"]
    m117_cache = data_cache_root() / config["artifacts"]["m117_cache_relpath"]

    print("loading corpus (test labels) + cached labels", flush=True)
    corpus, _train_index, _test_index = _load_corpus(config)
    test_labels = corpus["test_labels"]
    labels = np.load(m142_cache / config["artifacts"]["labels_file"])["labels"]
    n_train = min(int(config["level"]["n_train"]), smoke_train, len(labels))
    n_test = len(test_labels)
    print(f"level: {n_train} train / {n_test} test rows", flush=True)

    schedule = config["profile"]["trained_schedule"]
    batch = 64
    lr, wd, patience = 3e-4, 1e-4, 2
    epochs = min(4, smoke_epochs)
    order = np.random.default_rng(11).permutation(n_train)
    val_count = int(round(n_train * 0.05))
    train_fit = order[val_count:]
    val_rows = order[:val_count]

    evidence: dict[str, Any] = {
        "milestone": "M150",
        "cell": "rank-and-profile sweep of the cached code family",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
    }
    anchors: dict[str, Any] = {}
    table: dict[str, Any] = {}

    for code in config["codes"]:
        name = code["name"]
        width = int(code["width"])
        power = code.get("power")
        # per-code cache locations (registered in plan §6 after the first
        # dispatch crashed on the ms357 relpath): ms357* train codes live
        # under v16/m142_c3, and their TEST codes are the M151 artifact
        # v16/m151/ms357_fulltest.npy (C3 never persisted MS test codes).
        train_mem = np.load(
            data_cache_root() / code["train_relpath"] / code["file"],
            mmap_mode="r")
        test_mem = np.load(
            data_cache_root() / code["test_relpath"] / code["test_file"],
            mmap_mode="r")
        print(f"\ncode {name} ({width} cols)", flush=True)
        mem = train_mem

        # ---- ridge fit at 138k (the anchor) --------------------------------
        print("  ridge fit", flush=True)
        if power is None:
            acc = RidgeAccumulator(width, CLASSES)
            for start in range(0, n_train, block):
                stop = min(start + block, n_train)
                acc.add(mem[start:stop], labels[start:stop])
            weights = acc.solve_many([1.0])[1.0]
            standardise = acc.standardiser()
            hits = 0
            for start in range(0, n_test, block):
                stop = min(start + block, n_test)
                xs = standardise(np.asarray(test_mem[start:stop],
                                            dtype=np.float64))
                scores = xs @ weights[:-1] + weights[-1]
                hits += int((np.argmax(scores, axis=1)
                             == test_labels[start:stop]).sum())
            test_acc = hits / n_test
        else:
            solved, std = _fit_power(mem, labels, power, [1.0], n_train,
                                     block, transform=True)
            test_acc = _score_power(test_mem, test_labels,
                                    corpus["test_domains"], power,
                                    solved["1.0"], std, block,
                                    transform=True)
        anchors[name] = {"measured": test_acc,
                         "sealed": float(code["anchor_138k"]),
                         "delta": test_acc - float(code["anchor_138k"]),
                         "tolerance": TOLERANCE}
        print(f"    ridge {test_acc:.6f} (delta "
              f"{anchors[name]['delta']:+.3e})", flush=True)
        if not skip_anchors and abs(anchors[name]["delta"]) > TOLERANCE:
            evidence.update({"void": True,
                             "void_reason": f"{name} anchor failed",
                             "anchors": anchors})
            _write(output_dir, evidence)
            return evidence

        # ---- standardised-Gram profile --------------------------------------
        print("  Gram + spectrum", flush=True)
        gram = np.zeros((width, width), dtype=np.float64)
        colsum = np.zeros(width, dtype=np.float64)
        sqsum = np.zeros(width, dtype=np.float64)
        for start in range(0, n_train, block):
            stop = min(start + block, n_train)
            xs = np.asarray(mem[start:stop], dtype=np.float64)
            if power is not None:
                xs = power_norm(xs, power)
            gram += xs.T @ xs
            colsum += xs.sum(axis=0)
            sqsum += np.square(xs).sum(axis=0)
        gstd, _centre, _scale = _standardised_gram(gram, colsum, sqsum,
                                                   n_train)
        del gram
        # AMENDED 16 Aug (registered in plan §6): the M128 definition
        # needs only the EIGENVALUES (participation ratio, shares,
        # condition), so eigvalsh replaces the full eigendecomposition
        # whose 40k-dim eigenvector matrices are impractical (hours,
        # ~26 GB per code). The definition is unchanged; only the
        # decomposition routine's output is trimmed.
        vals = np.linalg.eigvalsh(gstd)
        vals = np.sort(vals)[::-1]
        del gstd
        keep = vals > max(float(vals.max()) * 1e-10, 1e-12)
        row: dict[str, Any] = {
            "width": width,
            "ridge_138k": test_acc,
            "effective_rank": _participation_ratio(vals),
            "n_positive": int(keep.sum()),
        }
        row.update(_spectrum_shares(vals, keep))
        table[name] = row
        print(f"    eff-rank {row['effective_rank']:.2f}, "
              f"top1 {row['top1_share']:.3f}, cond {row['condition']:.1f}",
              flush=True)

        # ---- trained-head read ----------------------------------------------
        print("  trained head", flush=True)
        model = HeadOnly(width, CLASSES, device)
        training = _train_with_schedule(
            model,
            _codes_factory(mem, labels, train_fit, power, batch, device),
            _codes_factory(mem, labels, val_rows, power, batch, device),
            epochs, lr, wd, device, patience)
        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for inputs, lab in _codes_factory(test_mem, test_labels,
                                              np.arange(n_test), power,
                                              batch, device)():
                logits = model(inputs)
                correct += int((logits.argmax(dim=1) == lab).sum().item())
                total += len(lab)
        row["trained_head_138k"] = correct / total
        row["trained_val"] = training["best_validation_accuracy"]
        print(f"    trained {row['trained_head_138k']:.6f}", flush=True)
        del model, mem, test_mem
        torch.cuda.empty_cache()

    evidence.update({"anchors": anchors, "table": table,
                     "runtime_seconds": round(time.time() - started, 2)})
    _write(output_dir, evidence)
    print(f"\nM150 complete -> {output_dir / 'evidence.json'}", flush=True)
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
    run_m150(args.config, args.output)


if __name__ == "__main__":
    main()
