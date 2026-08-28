"""M206 — DNN-component probe: a small MLP head trained on cached
f6144 codes, admitted through the M205 validator, and evaluated on the
sealed test.

Registered scope (19 Aug 2026, plan §6): this probe seals the §4.13
INTEGRATION path (train → admission contract → replay hash → sealed-
test measurement). The full four-arm coalition re-run is OUT of scope
for the probe. Deterministic where the registration contract requires
it (seeded init, fixed architecture, recorded hashes); the training
itself is logged, not bit-replayed (the §4.13 contract).
"""
from __future__ import annotations

import argparse
import hashlib
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
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from geode.core.dnn_admission import DNNSubmission, AdmissionRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m206_dnn_probe.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25" / "m206_dnn_probe")

CLASSES = 345


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_m206(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    configure_external_cache_environment()
    corpus, _ti, _tei = _load_corpus(config)
    test_labels = corpus["test_labels"]
    n_test = len(test_labels)

    f6144_cache = data_cache_root() / config["artifacts"]["f6144_cache_relpath"]
    mem_train = np.load(f6144_cache / config["artifacts"]["f6144_train_file"],
                        mmap_mode="r")
    mem_test = np.load(f6144_cache / config["artifacts"]["f6144_test_file"],
                       mmap_mode="r")
    n_train = int(config["training"]["train_rows"])
    hidden = int(config["training"]["hidden"])
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["lr"])
    batch = int(config["training"]["batch"])
    seed = int(config["training"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    print(f"device {device}; n_train {n_train}; hidden {hidden}",
          flush=True)

    # ---- standardiser (fitted on the train slice only; part of the
    # artifact — the probe's first run fed raw f6144 codes and
    # collapsed to chance, so scaling is registered as a component)
    raw_train = np.asarray(mem_train[:n_train], dtype=np.float64)
    train_mean = raw_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    train_std = raw_train.std(axis=0, dtype=np.float64)
    train_std = np.maximum(train_std, 1e-6).astype(np.float32)
    del raw_train
    standardiser = {"mean": train_mean, "std": train_std}
    standardiser_hash = _sha256_hex(
        train_mean.astype(np.float32).tobytes()
        + train_std.astype(np.float32).tobytes())

    # ---- architecture (registered, deterministic) ---------------------------
    architecture = {
        "input": 24576, "hidden": hidden, "output": CLASSES,
        "activation": "relu", "layers": ["linear", "relu", "linear"],
        "bias": True,
    }
    architecture_hash = _sha256_hex(json.dumps(
        architecture, sort_keys=True).encode("utf-8"))
    seed_hash = _sha256_hex(f"seed={seed}".encode("utf-8"))
    data_digest = _sha256_hex(f"f6144_train rows={n_train}".encode("utf-8"))
    software_hash = _sha256_hex(
        f"torch {torch.__version__}".encode("utf-8"))

    # ---- train -------------------------------------------------------------
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(24576, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, CLASSES),
    ).to(device)
    train_rows = np.asarray(mem_train[:n_train], dtype=np.float32)
    labels = corpus["train_labels"][:n_train]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.CrossEntropyLoss()
    indices = np.arange(n_train)
    log: list[float] = []
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(indices)
        total_loss = 0.0
        for start in range(0, n_train, batch):
            stop = min(start + batch, n_train)
            idx = indices[start:stop]
            x = torch.from_numpy(train_rows[idx]).to(device)
            x = (x - torch.from_numpy(train_mean).to(device)) / torch.from_numpy(
                train_std).to(device)
            y = torch.from_numpy(labels[idx]).to(device)
            optimizer.zero_grad()
            loss = lossfn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * (stop - start)
        log.append(total_loss / n_train)
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  epoch {epoch + 1}/{epochs} loss "
                  f"{log[-1]:.4f}", flush=True)
    training_log_digest = _sha256_hex(json.dumps(log).encode("utf-8"))

    # ---- weights hash + replay anchor --------------------------------------
    weights_bytes = b"".join(
        p.detach().cpu().numpy().astype(np.float32).tobytes()
        for p in model.parameters())
    weights_hash = _sha256_hex(weights_bytes)

    # ---- evaluate on the sealed test ---------------------------------------
    model.eval()
    preds = np.empty(n_test, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, n_test, batch):
            stop = min(start + batch, n_test)
            x = torch.from_numpy(
                np.asarray(mem_test[start:stop], dtype=np.float32)
            ).to(device)
            x = (x - torch.from_numpy(train_mean).to(device)) / torch.from_numpy(
                train_std).to(device)
            preds[start:stop] = model(x).argmax(dim=1).cpu().numpy()
    accuracy = float((preds == test_labels).mean())

    # ---- admission through the M205 contract --------------------------------
    submission = DNNSubmission(
        architecture_hash=architecture_hash,
        seed_hash=seed_hash,
        data_digest=data_digest,
        software_hash=software_hash,
        weights_hash=weights_hash,
        training_log_digest=training_log_digest,
        eval_report={"split": "test", "n_test": int(n_test),
                     "accuracy": accuracy},
    )
    registry = AdmissionRegistry()
    admission = registry.admit(submission)
    admission_dict = {
        "admitted": admission.admitted,
        "replay_hash": admission.replay_hash,
        "reasons": admission.reasons,
        "duplicate": admission.duplicate,
    }
    print(f"admission: {admission_dict}", flush=True)

    evidence: dict[str, Any] = {
        "milestone": "M206",
        "cell": "DNN-component probe (admission + sealed-test measurement)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "architecture": architecture,
        "hashes": {"architecture": architecture_hash, "seed": seed_hash,
                   "data": data_digest, "software": software_hash,
                   "weights": weights_hash,
                   "standardiser": standardiser_hash,
                   "training_log": training_log_digest},
        "training": {"epochs": epochs, "final_loss": log[-1]},
        "evaluation": {"test_accuracy": accuracy, "n_test": int(n_test)},
        "admission": admission_dict,
        "verdict": {
            "admitted": admission.admitted,
            "reading": ("the DNN artifact passed the M205 admission "
                        "contract and its replay hash is recorded for H6"
                        if admission.admitted
                        else "admission failed: " + "; ".join(
                            admission.reasons)),
        },
        "scope_note": ("probe scope: the admission + measurement path is "
                       "sealed; the four-arm coalition re-run with the dnn "
                       "arm is registered as out of scope for the probe"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"admitted": admission.admitted,
                      "accuracy": accuracy,
                      "replay_hash": admission.replay_hash}, indent=1),
          flush=True)
    print(f"M206 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m206(args.config, args.output)


if __name__ == "__main__":
    main()
