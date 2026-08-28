"""M142 cell C4 — power-normalisation on the cached SPM codes (free fit).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v22.md`` (execution
log, 14 Aug 2026, after the sealed C2 PASS). C4 is the composition cell of
C1 (power-norm) and C2 (spatial-pyramid pooling): the classic
Fisher-vector post-processing — signed power p + per-row L2 normalisation —
applied to the CACHED C2 SPM codes, never re-encoded.

Question. Does power-normalisation lift the SPM construction the way it
lifted the sealed codes (the registered C1b prior: +1.68 points at 138k)?

Construction. The sealed C2 artifacts: ``v16/m142_c2/spm1923_fulltrain.npy``
(409,832 x 40,383) and ``m142_c2_fulltrain_labels.npz`` on the cache drive.
The SPM test codes are encoded once here and PERSISTED as
``spm1923_fulltest.npy`` (the C2 run held them in RAM only), so every later
free cell reuses them.

Anchors:
- t1: the raw fit (no transform, penalty 1.0, full data) must reproduce
  the sealed C2 read Q_SPM(1923, 409832) = 0.260493 within 0.002.
- t1b (C1b rule): p=1.0 + L2 is the identity up to a positive per-row
  scale, so its fit must track the raw fit closely (delta reported; the
  adjudication is within-instrument: best cell vs the raw same-fitter
  reference).

Gate (kill switch). The best (p, penalty) cell at full data must beat the
raw same-fitter reference at penalty 1.0 by >= +0.005. The 138k-level grid
is reported alongside (the C1b prior level). The trained-head read (p=0.5)
is the co-adaptation control; the cell closes as a scoped negative only if
BOTH the gate fires AND the trained-head read fails.

SEALED RESULT (14 Aug 2026): t1 raw refit exact (delta +3.9e-16); best
cell p=0.5 lambda=0.1 -> 0.278551, gain +0.0181 over raw 0.260493; gate
cleared. p=1.0 (L2 alone) HURTS (-0.6); the square root carries the gain.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m142_c4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator, _score
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m113_learned import (
    _build_whitener_and_candidates,
    _random_dictionary,
)
from experiments.tier4.eval_v16_m142_c2 import _append_encode
from experiments.tier4.eval_v16_m142_factorial import power_norm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m142_c4.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m142_c4"

CLASSES = 345
T1_TOLERANCE = 0.002
KS_MARGIN = 0.005


def _fit_power(mem_train: np.ndarray, labels: np.ndarray, p: float,
               penalties: list[float], n_rows: int, block: int,
               transform: bool = True
               ) -> tuple[dict[str, np.ndarray], Any]:
    """One accumulator over the first ``n_rows`` rows.

    ``transform=False`` fits the RAW codes (the t1 reference); ``True``
    applies the signed-power + per-row L2 post-processing."""
    acc = RidgeAccumulator(mem_train.shape[1], CLASSES)
    for start in range(0, n_rows, block):
        stop = min(start + block, n_rows)
        block_x = mem_train[start:stop]
        acc.add(power_norm(block_x, p) if transform else block_x,
                labels[start:stop])
    solved = acc.solve_many(penalties)
    return ({str(q): w for q, w in solved.items()},
            acc.standardiser())


def _score_power(mem_test: np.ndarray, labels: np.ndarray, domains: np.ndarray,
                 p: float, weights: np.ndarray, std, block: int,
                 transform: bool = True) -> float:
    hits = 0
    n = len(labels)
    for start in range(0, n, block):
        stop = min(start + block, n)
        block_x = mem_test[start:stop]
        xs = power_norm(block_x, p) if transform else block_x
        hits += int(_score(weights, std(xs), labels[start:stop]).sum())
    return hits / n


def run_m142_c4(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    smoke = inadmissible
    smoke_skip = bool(config.get("_smoke_skip_gates", False))
    block = int(config["numerics"]["block"])
    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    configure_external_cache_environment()

    cache = data_cache_root() / "v16" / "m142_c2"
    train_mem = np.load(cache / config["artifacts"]["spm_train_file"],
                        mmap_mode="r")
    labels = np.load(cache / config["artifacts"]["labels_file"])["labels"]
    if len(labels) != len(train_mem):
        raise SystemExit(f"labels {len(labels)} vs train rows {len(train_mem)}")

    corpus, _train_index, _test_index = _load_corpus(config)
    test_labels = corpus["test_labels"]
    test_domains = corpus["test_domains"]
    spm_width = 21 * int(config["sparse"]["spm_atoms"])

    # test codes: the full run persists them once; the smoke encodes its
    # tiny slice to RAM and NEVER touches the persisted artifact
    test_path = cache / config["artifacts"]["spm_test_file"]
    if smoke:
        _verify_device(torch)
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        rep = config["sparse"]
        size = int(config["corpus"]["image_size"])
        grid = (size - int(rep["patch"])) // int(rep["stride"]) + 1
        print("building whitener + SPM dictionary for the smoke test encode",
              flush=True)
        whitener, candidates = _build_whitener_and_candidates(config, corpus)
        dict_spm = _random_dictionary(candidates, len(candidates),
                                      int(rep["dictionary_seed"]),
                                      int(rep["spm_atoms"]))
        table = torch.from_numpy(
            np.ascontiguousarray(dict_spm)).to(torch.float32).to(device)
        n_smoke_test = int(config["_smoke_test_rows"])
        test_mem = np.empty((n_smoke_test, spm_width), dtype=np.float32)
        _append_encode(corpus["test_images"], np.arange(n_smoke_test), table,
                       whitener, grid, test_mem, 0,
                       float(config["numerics"]["encode_throttle_seconds"]))
    elif test_path.exists() and np.load(
            test_path, mmap_mode="r").shape[0] == len(test_labels):
        test_mem = np.load(test_path, mmap_mode="r")
    else:
        _verify_device(torch)
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        rep = config["sparse"]
        size = int(config["corpus"]["image_size"])
        grid = (size - int(rep["patch"])) // int(rep["stride"]) + 1
        print("building whitener + SPM dictionary for the test encode",
              flush=True)
        whitener, candidates = _build_whitener_and_candidates(config, corpus)
        dict_spm = _random_dictionary(candidates, len(candidates),
                                      int(rep["dictionary_seed"]),
                                      int(rep["spm_atoms"]))
        table = torch.from_numpy(
            np.ascontiguousarray(dict_spm)).to(torch.float32).to(device)
        n_test = len(test_labels)
        test_mem = np.lib.format.open_memmap(
            test_path, mode="w+", dtype=np.float32, shape=(n_test, spm_width))
        _append_encode(corpus["test_images"], np.arange(n_test), table,
                       whitener, grid, test_mem, 0,
                       float(config["numerics"]["encode_throttle_seconds"]))
        del test_mem
        test_mem = np.load(test_path, mmap_mode="r")
        print(f"  test codes persisted -> {test_path}", flush=True)

    if smoke:
        train_mem = train_mem[:20000]
        labels = labels[:20000]
        test_mem = test_mem[:2000]
        test_labels = test_labels[:2000]
        test_domains = test_domains[:2000]
        print("SMOKE: 20000 train / 2000 test rows", flush=True)

    n_rows = len(train_mem)
    n_138 = int(config["cell_c4"]["n_138k"]) if not smoke else n_rows
    p_ladder = [float(p) for p in config["cell_c4"]["p_ladder"]]
    penalty_ladder = [float(q) for q in config["cell_c4"]["penalty_ladder"]]

    # raw reference (NO transform), same fitter
    raw_w, raw_std = _fit_power(train_mem, labels, 1.0, penalty_ladder,
                                n_rows, block, transform=False)
    raw_acc = {str(q): _score_power(test_mem, test_labels, test_domains,
                                    1.0, raw_w[str(q)], raw_std, block,
                                    transform=False)
               for q in penalty_ladder}
    raw_ref = raw_acc["1.0"]
    t1_delta = raw_ref - float(config["anchors"]["t1_reference"])
    print(f"t1 raw refit: {raw_ref:.4f} vs sealed C2 "
          f"{config['anchors']['t1_reference']} (delta {t1_delta:+.6f})",
          flush=True)
    if not smoke_skip and abs(t1_delta) > T1_TOLERANCE:
        raise SystemExit(f"t1 anchor reproduction failed (delta {t1_delta})")

    evidence: dict[str, Any] = {
        "milestone": "M142",
        "cell": "C4 power-normalisation on the cached SPM codes (free fit)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "anchors": {"raw_refit": raw_acc,
                    "t1_delta_vs_sealed_c2": t1_delta},
        "prior": config["cell_c4"]["prior_note"],
    }

    cells: dict[str, Any] = {}
    for p in p_ladder:
        w, std = _fit_power(train_mem, labels, p, penalty_ladder, n_rows,
                            block)
        for q in penalty_ladder:
            acc = _score_power(test_mem, test_labels, test_domains, p,
                               w[str(q)], std, block)
            cells[f"p{p}_lambda{q}"] = {"accuracy": acc, "p": p,
                                        "penalty": q}
            print(f"  full p={p} lambda={q}: {acc:.4f}", flush=True)
    evidence["full_data_cells"] = cells

    cells_138: dict[str, Any] = {}
    for p in p_ladder:
        w, std = _fit_power(train_mem, labels, p, [1.0], n_138, block)
        acc = _score_power(test_mem, test_labels, test_domains, p,
                           w["1.0"], std, block)
        cells_138[f"p{p}"] = acc
        print(f"  138k p={p}: {acc:.4f}", flush=True)
    evidence["cells_138k"] = cells_138

    best_key = max(cells, key=lambda k: cells[k]["accuracy"])
    best = cells[best_key]
    gain = best["accuracy"] - raw_ref
    print(f"  best cell: {best_key} {best['accuracy']:.4f} "
          f"(raw ref {raw_ref:.4f}, gain {gain:+.4f})", flush=True)

    trained_acc = None
    if not smoke:
        print("trained-head read (p=0.5)", flush=True)
        trained_acc = _trained_head_read(
            train_mem, labels, test_mem, test_labels, 0.5, block,
            int(config["cell_c4"]["trained_epochs"]),
            float(config["cell_c4"]["trained_lr"]),
            int(config["cell_c4"]["trained_seed"]),
            torch.device("cuda:0"))

    fired = (not smoke) and (gain < KS_MARGIN)
    both_fail = fired and (trained_acc is not None
                           and trained_acc < raw_ref + KS_MARGIN)
    evidence["reads"] = {"trained_head_read": trained_acc}
    evidence["gate"] = {
        "registered": config["cell_c4"]["gate_registered"],
        "best_cell": best_key,
        "best_accuracy": best["accuracy"],
        "raw_same_fitter_reference": raw_ref,
        "gain": gain,
        "required": KS_MARGIN,
        "fired": fired,
        "consequence": (config["cell_c4"]["consequence_fired"] if fired
                        else config["cell_c4"]["consequence_passed"]),
        "closure_note": ("scoped negative requires BOTH reads to fail; "
                         f"both_fail={bool(both_fail)}"),
    }
    evidence["runtime_seconds"] = round(time.time() - started, 2)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM142 C4 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def _trained_head_read(train_mem: np.ndarray, train_labels: np.ndarray,
                       test_mem: np.ndarray, test_labels: np.ndarray, p: float,
                       block: int, epochs: int, lr: float, seed: int,
                       device: torch.device) -> float:
    """SGD linear head on the power-normalised SPM codes (co-adaptation)."""
    torch.manual_seed(seed)
    width = train_mem.shape[1]
    model = torch.nn.Linear(width, CLASSES, bias=True).to(torch.float32).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    n = len(train_labels)
    order = np.random.default_rng(seed).permutation(n)
    batch = int(np.sqrt(n))
    for _ in range(epochs):
        for start in range(0, n, batch):
            take = order[start:start + batch]
            xs = torch.from_numpy(
                power_norm(train_mem[take], p).astype(np.float32)).to(device)
            ys = torch.from_numpy(train_labels[take].astype(np.int64)).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xs), ys)
            loss.backward()
            opt.step()
    model.eval()
    hits = 0
    n_test = len(test_labels)
    with torch.no_grad():
        for start in range(0, n_test, block):
            stop = min(start + block, n_test)
            xs = torch.from_numpy(
                power_norm(test_mem[start:stop], p).astype(np.float32)
            ).to(device)
            preds = torch.argmax(model(xs), dim=1).cpu().numpy()
            hits += int((preds == test_labels[start:stop]).sum())
    return hits / n_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m142_c4(args.config, args.output)


if __name__ == "__main__":
    main()
