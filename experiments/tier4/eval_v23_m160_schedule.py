"""M160 — M146 schedule sensitivity (measuring stick).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md`` (section 4
M160; the section 6 build entry, 17 Aug 2026). Re-runs M146's r3 — the
differentiable SPM+sqrt encoder (trainable dictionary + head, whitener
frozen, the M146 module and batch factories unchanged) — under three
schedules with the same model, seeds, and corpus:

- S0: the sealed schedule (8 epochs, cosine lr 3e-4, wd 1e-4,
  patience 2, batch 64, val frac 0.05) — also the anchor run, which
  must reproduce the sealed r3 0.10602898550724638 within 0.005.
- S1: 16 epochs, patience 4 (the more-epochs axis).
- S2: 8 epochs, lr 3e-5 (the different-LR axis).

Anchor: the frozen r1 reproduction (0.2273623188405797, tol 1e-6).
No gate: if any schedule's r3 clears r1, the "price of freezing"
verdict is schedule-bound and M146's shipping selection reopens; if
none does, the frozen verdict hardens. Smoke declares inadmissibility
and refuses the sealed output directory.
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
from experiments.tier4.eval_v15_m107_dense import _verify_pixel_identity
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v16_m109_trunk import (
    _load_corpus,
    _train_with_schedule,
)
from experiments.tier4.eval_v16_m113_learned import (
    _build_whitener_and_candidates,
    _random_dictionary,
)
from experiments.tier4.eval_v16_m142_c4 import _fit_power, _score_power
from experiments.tier4.eval_v16_m146_arbiter import (
    SpmSqrtModel,
    _images_factory,
    _model_test_accuracy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v23"
                  / "m160_schedule_sensitivity.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v23"
                  / "m160_schedule_sensitivity")

CLASSES = 345
PATCH_DIM = 108
T1_REFERENCE = 0.2273623188405797
T1_TOLERANCE = 1e-6
R3_SEALED = 0.10602898550724638
R3_TOLERANCE = 0.005


def run_m160(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
        "milestone": "M160",
        "cell": "M146 schedule sensitivity (r3 x three schedules)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
    }

    print("loading corpus", flush=True)
    corpus, train_index, test_index = _load_corpus(config)
    size = int(config["corpus"]["image_size"])
    for split, idx in (("train", train_index), ("test", test_index)):
        _verify_pixel_identity(split, idx, corpus[f"{split}_images"], size,
                               int(config["corpus"]
                                   ["pixel_identity_rows"]))

    cache = data_cache_root() / config["artifacts"]["cache_relpath"]
    train_mem = np.load(cache / config["artifacts"]["spm_train_file"],
                        mmap_mode="r")
    test_mem = np.load(cache / config["artifacts"]["spm_test_file"],
                       mmap_mode="r")
    labels = np.load(cache / config["artifacts"]["labels_file"])["labels"]
    test_labels = corpus["test_labels"][:smoke_test]
    test_domains = corpus["test_domains"][:smoke_test]
    test_mem = test_mem[:smoke_test]
    n_train = min(int(config["level"]["n_train"]), smoke_train, len(labels))
    print(f"level: {n_train} train rows, {len(test_labels)} test rows",
          flush=True)

    power = float(config["sparse"]["power"])
    atoms = int(config["sparse"]["spm_atoms"])
    grid = int(config["sparse"]["grid"])
    batch = int(config["schedule"]["batch"])
    val_frac = float(config["schedule"]["validation_fraction"])

    order = np.random.default_rng(config["corpus"]["shuffle_seed"]).permutation(
        n_train)
    val_count = int(round(n_train * val_frac))
    train_fit = order[val_count:]
    val_rows = order[:val_count]

    anchors: dict[str, Any] = {}

    # ---- r1 anchor: frozen ridge at 138k ----------------------------------
    print("r1 anchor: frozen codes + closed-form ridge", flush=True)
    solved, std = _fit_power(train_mem, labels, power, [1.0], n_train, block,
                             transform=True)
    r1_acc = _score_power(test_mem, test_labels, test_domains, power,
                          solved["1.0"], std, block, transform=True)
    anchors["r1"] = {"measured": r1_acc, "sealed": T1_REFERENCE,
                     "delta": r1_acc - T1_REFERENCE,
                     "tolerance": T1_TOLERANCE}
    print(f"  r1 {r1_acc:.6f} (delta {r1_acc - T1_REFERENCE:+.3e})",
          flush=True)
    if not skip_anchors and abs(r1_acc - T1_REFERENCE) > T1_TOLERANCE:
        evidence.update({"void": True,
                         "void_reason": "r1 frozen reproduction failed",
                         "anchors": anchors})
        _write(output_dir, evidence)
        return evidence

    # ---- the differentiable model (rebuilt once per schedule, same init) --
    print("building the M146 differentiable SPM+sqrt model", flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    dictionary = _random_dictionary(
        candidates, len(candidates), int(config["sparse"]["dictionary_seed"]),
        atoms)

    results: dict[str, Any] = {}
    for sched in config["schedules"]:
        name = sched["name"]
        epochs = int(sched["epochs"])
        if smoke:
            epochs = min(epochs, smoke_epochs)
        lr = float(sched["learning_rate"])
        wd = float(sched["weight_decay"])
        patience = int(sched["patience"])
        print(f"\nschedule {name}: {epochs} epochs, lr {lr}, patience "
              f"{patience}", flush=True)
        model = SpmSqrtModel(
            dictionary, whitener.mean, whitener.whiten, grid, CLASSES,
            float(config["sparse"]["contrast_epsilon"]), power, True, device)
        training = _train_with_schedule(
            model,
            _images_factory(corpus["train_images"], corpus["train_labels"],
                            train_fit, batch, device),
            _images_factory(corpus["train_images"], corpus["train_labels"],
                            val_rows, batch, device),
            epochs, lr, wd, device, patience)
        r3_acc = _model_test_accuracy(
            model, _images_factory(corpus["test_images"], test_labels,
                                   np.arange(len(test_labels)), batch,
                                   device))
        results[name] = {
            "r3_accuracy": r3_acc,
            "epochs_registered": int(sched["epochs"]),
            "epochs_run": training["epochs_run"],
            "best_validation_accuracy": training[
                "best_validation_accuracy"],
            "learning_rate": lr,
            "weight_decay": wd,
            "patience": patience,
        }
        print(f"  r3 {r3_acc:.6f} (val "
              f"{training['best_validation_accuracy']:.6f}, "
              f"{training['epochs_run']} epochs)", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---- the S0 anchor check ----------------------------------------------
    s0 = results.get("S0", {}).get("r3_accuracy")
    anchors["r3_s0"] = {"measured": s0, "sealed": R3_SEALED,
                        "delta": (s0 - R3_SEALED) if s0 is not None
                        else None,
                        "tolerance": R3_TOLERANCE}
    if s0 is not None and not skip_anchors:
        print(f"  r3 S0 anchor delta {s0 - R3_SEALED:+.6f} "
              f"(tol {R3_TOLERANCE})", flush=True)
        if abs(s0 - R3_SEALED) > R3_TOLERANCE:
            evidence.update({"void": True,
                             "void_reason": "r3 sealed-schedule "
                                            "reproduction failed",
                             "anchors": anchors})
            _write(output_dir, evidence)
            return evidence

    # ---- the measuring-stick reading --------------------------------------
    clears = {k: v["r3_accuracy"] > r1_acc for k, v in results.items()}
    reading = ("schedule-bound: at least one schedule clears r1; the price "
               "of freezing is an artefact of the sealed schedule and M146's "
               "shipping selection reopens" if any(clears.values())
               else "the frozen verdict hardens: no schedule clears r1")

    evidence.update({
        "anchors": anchors,
        "r1_accuracy": r1_acc,
        "schedules": results,
        "reading": {
            "registered": config["gate"]["registered"],
            "clears_r1": clears,
            "verdict": reading,
        },
        "runtime_seconds": round(time.time() - started, 2),
    })
    _write(output_dir, evidence)
    print(f"\nM160 complete -> {output_dir / 'evidence.json'}", flush=True)
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
    run_m160(args.config, args.output)


if __name__ == "__main__":
    main()
