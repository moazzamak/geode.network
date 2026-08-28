"""M161 — the hybrid readout (ridge + trained residual head).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md`` (section 4
M161; the section 6 build entry, 17 Aug 2026). On the C4 138k context
(the cached SPM codes, p=0.5, the M146 level):

- ridge: the frozen closed-form read, penalty 1.0 — the anchor
  (0.2273623188405797, tol 1e-6).
- hybrid: the SAME ridge logits (frozen, detached) plus ONE trained
  linear head (the M146 r2 HeadOnly structure — the registered small
  residual head) trained under the M109 shared schedule (4 epochs,
  AdamW cosine lr 3e-4, wd 1e-4, patience 2, batch 64, val frac 0.05,
  shuffle_seed 11) with cross-entropy on the COMBINED logits.
- controls: the pure trained head (the r2-alone protocol, same
  schedule/seeds, no ridge) alongside the sealed r2
  0.042608695652173914, and the ridge read itself.

Gate: hybrid >= ridge + 0.005 on the test rows, else archived as a
scoped negative. Smoke declares inadmissibility and refuses the sealed
output directory.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v16_m109_trunk import (
    _load_corpus,
    _train_with_schedule,
)
from experiments.tier4.eval_v16_m142_c4 import _fit_power, _score_power
from experiments.tier4.eval_v16_m142_factorial import power_norm
from experiments.tier4.eval_v16_m146_arbiter import HeadOnly

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v23"
                  / "m161_hybrid_readout.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v23"
                  / "m161_hybrid_readout")

CLASSES = 345
RIDGE_REFERENCE = 0.2273623188405797
RIDGE_TOLERANCE = 1e-6
SEALED_R2 = 0.042608695652173914
MARGIN = 0.005


def _codes_only_factory(mem: np.ndarray, labels: np.ndarray, rows: np.ndarray,
                        power: float, batch: int,
                        device: torch.device) -> Callable[[], Iterator]:
    def gen() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for start in range(0, len(rows), batch):
            take = rows[start:start + batch]
            block = power_norm(mem[take], power).astype(np.float32)
            yield (torch.from_numpy(np.ascontiguousarray(block)).to(device),
                   torch.from_numpy(labels[take]).to(device))
    return gen


def _hybrid_factory(mem: np.ndarray, labels: np.ndarray, rows: np.ndarray,
                    power: float, batch: int, device: torch.device,
                    weights: np.ndarray, std) -> Callable[[], Iterator]:
    """Stream (codes, labels, frozen ridge logits) for the hybrid."""

    def gen() -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        for start in range(0, len(rows), batch):
            take = rows[start:start + batch]
            block = power_norm(mem[take], power).astype(np.float32)
            xs = std(block)
            rlog = torch.from_numpy(np.ascontiguousarray(
                xs @ weights[:-1] + weights[-1])).to(device)
            yield (torch.from_numpy(np.ascontiguousarray(block)).to(device),
                   torch.from_numpy(labels[take]).to(device), rlog)
    return gen


def run_m161(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    smoke_test = int(config.get("_smoke_test_rows", 10 ** 9))
    smoke_epochs = int(config.get("_smoke_epochs", 10 ** 9))

    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    torch.manual_seed(int(config["numerics"]["seed"]))
    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    block = int(config["numerics"]["block"])
    evidence: dict[str, Any] = {
        "milestone": "M161",
        "cell": "hybrid readout (ridge + trained residual head)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
    }

    print("loading corpus + cached codes", flush=True)
    corpus, _train_index, _test_index = _load_corpus(config)
    cache = data_cache_root() / config["artifacts"]["cache_relpath"]
    train_mem = np.load(cache / config["artifacts"]["spm_train_file"],
                        mmap_mode="r")
    test_mem = np.load(cache / config["artifacts"]["spm_test_file"],
                       mmap_mode="r")[:smoke_test]
    labels = np.load(cache / config["artifacts"]["labels_file"])["labels"]
    test_labels = corpus["test_labels"][:smoke_test]
    test_domains = corpus["test_domains"][:smoke_test]
    n_train = min(int(config["level"]["n_train"]), smoke_train, len(labels))
    power = float(config["sparse"]["power"])
    atoms = int(config["sparse"]["spm_atoms"])
    width = 21 * atoms
    batch = int(config["schedule"]["batch"])
    lr = float(config["schedule"]["learning_rate"])
    wd = float(config["schedule"]["weight_decay"])
    patience = int(config["schedule"]["patience"])
    epochs = int(config["schedule"]["epochs"])
    if smoke:
        epochs = min(epochs, smoke_epochs)
    print(f"level: {n_train} train / {len(test_labels)} test rows",
          flush=True)

    # ---- ridge (the anchor) ------------------------------------------------
    print("ridge fit (p=0.5, penalty 1.0)", flush=True)
    solved, std = _fit_power(train_mem, labels, power, [1.0], n_train, block,
                             transform=True)
    weights = solved["1.0"]
    ridge_acc = _score_power(test_mem, test_labels, test_domains, power,
                             weights, std, block, transform=True)
    anchors: dict[str, Any] = {
        "ridge": {"measured": ridge_acc, "sealed": RIDGE_REFERENCE,
                  "delta": ridge_acc - RIDGE_REFERENCE,
                  "tolerance": RIDGE_TOLERANCE},
    }
    print(f"  ridge {ridge_acc:.6f} (delta "
          f"{ridge_acc - RIDGE_REFERENCE:+.3e})", flush=True)
    if not skip_anchors and abs(ridge_acc - RIDGE_REFERENCE) > RIDGE_TOLERANCE:
        evidence.update({"void": True,
                         "void_reason": "ridge anchor reproduction failed",
                         "anchors": anchors})
        _write(output_dir, evidence)
        return evidence

    order = np.random.default_rng(config["corpus"]["shuffle_seed"]).permutation(
        n_train)
    val_count = int(round(n_train * float(config["schedule"]
                                          ["validation_fraction"])))
    train_fit = order[val_count:]
    val_rows = order[:val_count]
    test_rows = np.arange(len(test_labels))

    # ---- control: the pure trained head (the r2-alone protocol) -----------
    print("control: pure trained head (r2-alone protocol)", flush=True)
    model_r2 = HeadOnly(width, CLASSES, device)
    training_r2 = _train_with_schedule(
        model_r2,
        _codes_only_factory(train_mem, labels, train_fit, power, batch,
                            device),
        _codes_only_factory(train_mem, labels, val_rows, power, batch,
                            device),
        epochs, lr, wd, device, patience)
    r2_acc = _head_test_accuracy(
        model_r2, _codes_only_factory(test_mem, test_labels, test_rows,
                                      power, batch, device))
    print(f"  pure trained head {r2_acc:.6f} (val "
          f"{training_r2['best_validation_accuracy']:.6f}; sealed r2 "
          f"{SEALED_R2})", flush=True)
    del model_r2
    torch.cuda.empty_cache()

    # ---- the hybrid --------------------------------------------------------
    print("hybrid: ridge logits + trained residual head", flush=True)
    model_h = HeadOnly(width, CLASSES, device)
    training_h = _train_hybrid_head(
        model_h,
        _hybrid_factory(train_mem, labels, train_fit, power, batch, device,
                        weights, std),
        _hybrid_factory(train_mem, labels, val_rows, power, batch, device,
                        weights, std),
        epochs, lr, wd, device, patience)
    hybrid_acc = _hybrid_test_accuracy(
        model_h, _hybrid_factory(test_mem, test_labels, test_rows, power,
                                 batch, device, weights, std))
    gain = hybrid_acc - ridge_acc
    passed = gain >= MARGIN
    print(f"  hybrid {hybrid_acc:.6f} (gain vs ridge {gain:+.6f}, val "
          f"{training_h['best_validation_accuracy']:.6f})", flush=True)
    del model_h
    torch.cuda.empty_cache()

    evidence.update({
        "anchors": anchors,
        "ridge_accuracy": ridge_acc,
        "pure_trained_head": {"accuracy": r2_acc,
                              "sealed_r2": SEALED_R2,
                              "training": training_r2},
        "hybrid": {"accuracy": hybrid_acc,
                   "training": training_h},
        "gate": {
            "registered": config["gate"]["registered"],
            "gain": gain,
            "required": MARGIN,
            "passed": bool(passed),
            "consequence": (config["gate"].get("consequence_passed",
                                               "passed") if passed
                            else config["gate"].get("consequence_fired",
                                                    "fired")),
        },
        "runtime_seconds": round(time.time() - started, 2),
    })
    _write(output_dir, evidence)
    print(f"\nM161 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


# ---------------------------------------------------------------------------
# trained-head helpers (the _train_with_schedule pattern, local so the
# hybrid CE target can include the frozen ridge logits)
# ---------------------------------------------------------------------------
def _head_test_accuracy(model: HeadOnly, factory) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in factory():
            logits = model(inputs)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += len(labels)
    return correct / total


def _train_hybrid_head(model: HeadOnly, train_factory, val_factory,
                       epochs: int, lr: float, wd: float,
                       device: torch.device, patience: int) -> dict[str, Any]:
    """AdamW + cosine + patience; CE on (head(codes) + frozen ridge logits)."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                           T_max=epochs)
    best_val = 0.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience_left = patience
    history = []
    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        for inputs, labels_t, rlog in train_factory():
            optimiser.zero_grad(set_to_none=True)
            logits = model(inputs) + rlog
            loss = F.cross_entropy(logits, labels_t)
            loss.backward()
            optimiser.step()
            correct += int((logits.argmax(dim=1) == labels_t).sum().item())
            total += len(labels_t)
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels_t, rlog in val_factory():
                logits = model(inputs) + rlog
                val_correct += int((logits.argmax(dim=1)
                                    == labels_t).sum().item())
                val_total += len(labels_t)
        val_acc = val_correct / val_total
        history.append({"epoch": epoch + 1,
                        "train_accuracy": correct / total,
                        "validation_accuracy": val_acc})
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    return {"epochs_run": len(history), "best_validation_accuracy": best_val,
            "history": history}


def _hybrid_test_accuracy(model: HeadOnly, test_factory) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels_t, rlog in test_factory():
            logits = model(inputs) + rlog
            correct += int((logits.argmax(dim=1) == labels_t).sum().item())
            total += len(labels_t)
    return correct / total


def _write(output_dir: Path, evidence: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m161(args.config, args.output)


if __name__ == "__main__":
    main()
