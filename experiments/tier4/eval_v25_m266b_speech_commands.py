"""M266b — audio classification arm: frozen wav2vec2-base features +
closed-form ridge probe on Speech Commands v2 (CC-BY-4.0).

Registered and dispatched 21 Aug 2026 (plan v25, M266 + amendment
14), local-first, F: cache conventions. The closed-form ridge probe
IS the exact fit on frozen features — the registered "head
question" of this cell. Published anchors (Baevski et al. 2020,
fine-tuned ~98.1) are context only, never a gate.

Env note (registered, from M266a): torchcodec is incompatible with
the ROCm torch build; the parquet embeds audio bytes, decoded with
soundfile from BytesIO.
"""
from __future__ import annotations

import argparse
import hashlib
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m266b_speech_commands.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m266_audio_arm")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_m266b(config_path: Path, output_dir: Path, smoke: bool = False
              ) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    cache_root = data_cache_root() / config["feature_cache_relpath"]
    cache_root.mkdir(parents=True, exist_ok=True)

    import torch
    import soundfile as sf
    from transformers import Wav2Vec2Model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2Model.from_pretrained(
        config["encoder"]["checkpoint"])
    model.to(device).eval()
    batch = int(config["extraction"]["batch"])
    throttle = float(config["extraction"]["throttle_seconds"])
    sm = config["smoke"] if smoke else {}
    train_cap = sm.get("train_samples")
    eval_cap = sm.get("eval_samples")

    # ---- local Speech Commands v0.02 layout (official source) --------
    scv2_root = (data_cache_root().parents[1] / "cache" / "speech_commands"
                 / "v0.02")

    def rows_for(split: str, cap: int | None
                 ) -> list[tuple[Path, int]]:
        """(wav_path, label_index) rows. test/validation come from the
        official testing_list.txt / validation_list.txt; train is the
        remainder (the official split convention)."""
        label_dirs = [d for d in sorted(scv2_root.iterdir())
                      if d.is_dir() and d.name != "_background_noise_"]
        labels = {d.name: i for i, d in enumerate(label_dirs)}
        if split in ("test", "validation"):
            list_name = ("testing_list.txt" if split == "test"
                         else "validation_list.txt")
            entries = [line.strip() for line in
                       (scv2_root / list_name).read_text().splitlines()
                       if line.strip()]
        else:
            listed: set[str] = set()
            for list_name in ("testing_list.txt", "validation_list.txt"):
                listed.update(line.strip() for line in
                              (scv2_root / list_name)
                              .read_text().splitlines() if line.strip())
            entries = [f"{d.name}/{f.name}" for d in label_dirs
                       for f in sorted(d.glob("*.wav"))
                       if f"{d.name}/{f.name}" not in listed]
        rows = [(scv2_root / e, labels[e.split("/")[0]]) for e in entries]
        rows = [r for r in rows if r[0].exists()]
        if cap is not None:
            # class-balanced cap (a positional head slice over the
            # dir-sorted train list covers only the first few classes,
            # and the compact label indices would then mismatch the
            # test side — the registered positional-split lesson)
            budget = max(1, -(-cap // len(label_dirs)))
            seen: dict[int, int] = {}
            balanced: list[tuple[Path, int]] = []
            for row in rows:
                if seen.get(row[1], 0) >= budget:
                    continue
                seen[row[1]] = seen.get(row[1], 0) + 1
                balanced.append(row)
                if len(balanced) >= cap:
                    break
            rows = balanced
        return rows

    def extract(rows: list[tuple[Path, int]], name: str
                ) -> tuple[np.ndarray, list[int]]:
        # cache keyed by row count so a changed selection (smoke vs
        # full, or a fixed bug) can never reuse a poisoned cache
        cache_path = cache_root / f"scv2_{name}_{len(rows)}_feat.npy"
        label_path = cache_root / f"scv2_{name}_{len(rows)}_labels.npy"
        if cache_path.exists():
            feats = np.load(cache_path, mmap_mode="r").copy()
            labels = np.load(label_path).tolist()
            return feats, labels
        labels: list[int] = []
        feats: list[np.ndarray] = []
        for path, label in rows:
            array, sr = sf.read(path, dtype="float32", always_2d=False)
            if array.ndim > 1:
                array = array.mean(axis=1)
            labels.append(label)
            with torch.no_grad():
                x = torch.from_numpy(array).unsqueeze(0).to(device)
                out = model(x).last_hidden_state  # 1 x T x 768
            feats.append(out.mean(dim=1).cpu().numpy().astype(np.float32))
            if throttle:
                time.sleep(throttle)
        feats_arr = np.concatenate(feats, axis=0)
        np.save(cache_path, feats_arr)
        np.save(label_path, np.asarray(labels, dtype=np.int64))
        return feats_arr, labels

    train_rows = rows_for("train", train_cap)
    test_rows = rows_for("test", eval_cap)
    tr_feat, tr_labels = extract(train_rows, "train")
    te_feat, te_labels = extract(test_rows, "test")

    # ---- closed-form ridge probe (the exact fit) -------------------------
    classes = sorted(set(tr_labels))
    y = np.zeros((len(tr_labels), len(classes)), dtype=np.float64)
    for i, label in enumerate(tr_labels):
        y[i, classes.index(label)] = 1.0
    mean = tr_feat.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(tr_feat.std(axis=0, dtype=np.float64), 1e-6)
    tr_norm = (tr_feat - mean) / std
    alpha = float(config["probe"]["alpha"])
    gram = tr_norm.T @ tr_norm
    rhs = tr_norm.T @ y
    w = np.linalg.solve(gram + alpha * np.eye(tr_norm.shape[1]),
                        rhs).astype(np.float32)
    b = y.mean(axis=0) - (tr_norm.mean(axis=0) @ w)
    te_norm = (te_feat - mean) / std
    scores = te_norm @ w + b
    preds = np.asarray([classes[int(i)] for i in scores.argmax(axis=1)],
                       dtype=np.int64)
    test_acc = float((preds == np.asarray(te_labels)).mean())
    np.savez(cache_root / "scv2_probe.npz", weights=w, bias=b,
             mean=mean, std=std, classes=np.asarray(classes))
    weights_hash = _sha256_hex(
        w.astype(np.float32).tobytes() + b.astype(np.float32).tobytes())

    evidence: dict[str, Any] = {
        "milestone": "M266b",
        "cell": "audio classification arm (frozen wav2vec2-base + "
                "ridge probe, Speech Commands v2)",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "encoder": {"checkpoint": config["encoder"]["checkpoint"],
                    "license_recorded":
                        config["encoder"]["license_recorded"],
                    "frozen": True},
        "probe": {"ridge_alpha": alpha, "n_classes": len(classes),
                  "weights_hash": weights_hash,
                  "train_rows": len(tr_labels)},
        "evaluation": {"split": "speech_commands/v0.02/test",
                       "test_accuracy": test_acc,
                       "n_test": len(te_labels),
                       "reading": ("frozen-probe accuracy, below the "
                                   "published fine-tuned anchor (Baevski "
                                   "et al. 2020 ~98.1) — cited, never "
                                   "exceeded")},
        "anchors_reference_only": config["anchors_reference_only"],
        "dataset_source": ("official speech_commands_v0.02 archive "
                           "(download.tensorflow.org), local extraction "
                           "on F:; the HF 'speech_commands' entry is a "
                           "legacy script dataset unsupported by "
                           "datasets 5.x"),
        "license_recorded": config["dataset"]["license_recorded"],
        "scope_note": ("publisher checkpoint frozen; closed-form ridge "
                       "is the exact fit on frozen features; one "
                       "held-out read"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence_m266b.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"test_accuracy": test_acc, "n_test": len(te_labels)},
                     indent=1), flush=True)
    print(f"M266b complete -> {output_dir / 'evidence_m266b.json'}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output
    if args.smoke and output == DEFAULT_OUTPUT:
        # a smoke run is inadmissible evidence — never write it where
        # the full run's evidence will be sealed
        output = DEFAULT_OUTPUT.parent / (DEFAULT_OUTPUT.name + "_smoke")
    run_m266b(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
