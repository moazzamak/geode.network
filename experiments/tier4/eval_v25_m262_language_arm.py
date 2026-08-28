"""M262 — language-inference arm: frozen BERT-base features + trained
logistic probes for MNLI (matched/mismatched), SST-2, and IMDb.

Registered and dispatched 21 Aug 2026 (plan v25, M262 + queue-status
amendments 9/10), local-first, F: cache conventions. Built to the
M206 pattern (config -> runner -> evidence).

Honesty notes, registered before the run:
- the encoder is a PUBLISHER checkpoint (Apache-2.0) — frozen, never
  trained here; architecture/weights hashes are recorded;
- published BERT numbers (Devlin et al. 2019) are cited as reference
  anchors only — no gate, no claim to exceed;
- dataset licensing is recorded per split in the evidence license
  field; the IMDb probe is evaluation-only until the licensing audit
  clears its commercial standing (audit C6 rule);
- M247 refusal tags and the M250 behaviour-diff baseline for this arm
  are recorded as pending (structure shipped in geode; the measured
  assembly remains the registered M247/M250 pendings) — not claimed.
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
                  / "m262_language_arm.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m262_language_arm")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_split(load_fn, hf_id: str, split: str, n_rows: int | None,
                config_name: str | None = None):
    """Load a HF dataset split; cap rows when n_rows is given
    (smoke mode). The smoke cap is CLASS-BALANCED: a positional head
    slice can be single-class (e.g. IMDb's parquet is label-sorted),
    which would break probe fitting — the registered lesson. A smoke
    run is a pipeline check, never a reading."""
    if config_name is not None:
        ds = load_fn(hf_id, config_name, split=split)
    else:
        ds = load_fn(hf_id, split=split)
    if n_rows is not None:
        budget = max(2, int(n_rows) // 2)
        seen: dict[int, int] = {}
        indices: list[int] = []
        for i, row in enumerate(ds):
            label = int(row["label"])
            if seen.get(label, 0) >= budget:
                continue
            seen[label] = seen.get(label, 0) + 1
            indices.append(i)
            if len(indices) >= int(n_rows):
                break
        ds = ds.select(indices)
    return ds


def _extract_features(model, tokenizer, texts: list[str], device,
                      batch: int, throttle: float) -> np.ndarray:
    """Mean-pooled last-hidden-state features, fp32, deterministic
    per batch (no dropout in eval mode). Throttle between batches is
    the registered display-GPU TDR mitigation."""
    feats: list[np.ndarray] = []
    model.eval()
    with _no_grad():
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            enc = tokenizer(chunk, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(
                dim=1).clamp(min=1.0)
            feats.append(pooled.cpu().numpy().astype(np.float32))
            if throttle:
                time.sleep(throttle)
    return np.concatenate(feats, axis=0)


class _no_grad:
    def __enter__(self):
        import torch
        self._torch = torch
        self._prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        return self

    def __exit__(self, *exc):
        self._torch.set_grad_enabled(self._prev)


def _content_digest(texts: list[str], labels: list[int]) -> str:
    return _sha256_hex(json.dumps(
        {"texts": texts, "labels": [int(x) for x in labels]},
        sort_keys=True).encode("utf-8"))


def run_m262(config_path: Path, output_dir: Path, smoke: bool = False
             ) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    cache_root = data_cache_root() / config["feature_cache_relpath"]
    cache_root.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    from datasets import load_dataset as _hf_load

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}; transformers {transformers.__version__}",
          flush=True)

    # ---- frozen encoder (publisher checkpoint, never trained) ----
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["encoder"]["checkpoint"])
    model = AutoModel.from_pretrained(config["encoder"]["checkpoint"])
    model.to(device)
    safetensors_path = hf_hub_download(
        config["encoder"]["checkpoint"], "model.safetensors")
    encoder_hashes = {
        "checkpoint": config["encoder"]["checkpoint"],
        "license": config["encoder"]["license"],
        "safetensors_sha256": _sha256_hex(
            Path(safetensors_path).read_bytes()),
        "note": "publisher checkpoint, frozen; never trained in this cell",
    }

    sm = config["smoke"] if smoke else {}
    train_rows = sm.get("train_rows")
    eval_rows = sm.get("eval_rows")

    # ---- extraction + probes per task ----
    tasks: dict[str, Any] = {}
    batch = int(config["extraction"]["batch"])
    throttle = float(config["extraction"]["throttle_seconds"])
    ridge_alpha = float(config["probe"].get("alpha", 1.0))

    def cached_features(name: str, split: str, texts: list[str]):
        """Features are cached per (task, split, ROW COUNT) on F: so
        later cells (arm registration, M247/M250) reuse the same
        extraction — and a smoke run (capped rows) can never poison a
        full run's cache (the registered test-path lesson)."""
        safe = split.replace("/", "_")
        stem = f"{name}_{safe}_{len(texts)}"
        feat_path = cache_root / f"{stem}_feat.npy"
        text_path = cache_root / f"{stem}_texts.json"
        if feat_path.exists():
            return np.load(feat_path, mmap_mode="r").copy()
        feats = _extract_features(model, tokenizer, texts, device,
                                  batch, throttle)
        np.save(feat_path, feats)
        Path(text_path).write_text(
            json.dumps(texts), encoding="utf-8")
        return feats

    def ridge_probe(tr_feat: np.ndarray, train_labels: list[int]):
        """Closed-form ridge least-squares probe on one-hot targets —
        convergence-free, deterministic (the M262 registered probe
        after the lbfgs iteration-cap finding; see plan amendment)."""
        classes = sorted(set(train_labels))
        n_classes = len(classes)
        y = np.zeros((len(train_labels), n_classes), dtype=np.float64)
        for i, label in enumerate(train_labels):
            y[i, classes.index(label)] = 1.0
        mean = tr_feat.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(tr_feat.std(axis=0, dtype=np.float64), 1e-6)
        tr_norm = (tr_feat - mean) / std
        # d x d Gram (d = 768): closed form, no iteration limit
        gram = tr_norm.T @ tr_norm
        rhs = tr_norm.T @ y
        w = np.linalg.solve(gram + ridge_alpha * np.eye(tr_norm.shape[1]),
                            rhs).astype(np.float32)
        b = (y.mean(axis=0) - (tr_norm.mean(axis=0) @ w))
        return {"weights": w, "bias": b, "mean": mean, "std": std,
                "classes": classes}

    def predict(probe: dict[str, Any], feats: np.ndarray) -> np.ndarray:
        norm = (feats - probe["mean"]) / probe["std"]
        scores = norm @ probe["weights"] + probe["bias"]
        return np.asarray([probe["classes"][int(i)]
                           for i in scores.argmax(axis=1)], dtype=np.int64)

    def probe_task(name: str, hf_id: str, train_split: str,
                   eval_splits: list[str], label_from_row,
                   config_name: str | None = None) -> dict:
        t0 = time.time()
        train_ds = _load_split(_hf_load, hf_id, train_split, train_rows,
                               config_name=config_name)
        train_texts = [r["premise"] + " [SEP] " + r["hypothesis"]
                       if "premise" in r else (r.get("text")
                                               or r["sentence"])
                       for r in train_ds]
        train_labels = [label_from_row(r) for r in train_ds]
        tr_feat = cached_features(name, train_split, train_texts)
        probe = ridge_probe(tr_feat, train_labels)
        per_split: dict[str, Any] = {}
        for split in eval_splits:
            ev_ds = _load_split(_hf_load, hf_id, split, eval_rows,
                                config_name=config_name)
            ev_texts = [r["premise"] + " [SEP] " + r["hypothesis"]
                        if "premise" in r else (r.get("text")
                                                or r["sentence"])
                        for r in ev_ds]
            ev_labels = [label_from_row(r) for r in ev_ds]
            ev_feat = cached_features(name, split, ev_texts)
            preds = predict(probe, ev_feat)
            acc = float((preds == np.asarray(ev_labels)).mean())
            per_split[split] = {"accuracy": acc,
                                "n_rows": len(ev_labels),
                                "data_digest":
                                    _content_digest(ev_texts, ev_labels)}
            if smoke:
                # one-time diagnostic: the iterative solver on the
                # SAME features — the iteration-cap hypothesis was
                # REFUTED (solvers tie; see diag_m262_solver.py)
                from sklearn.linear_model import LogisticRegression
                mean_s = probe["mean"].astype(np.float32)
                std_s = probe["std"].astype(np.float32)
                tr_norm = ((tr_feat - mean_s) / std_s)
                clf = LogisticRegression(C=1.0, max_iter=500,
                                         solver="lbfgs",
                                         random_state=20260821)
                clf.fit(tr_norm, np.asarray(train_labels,
                                            dtype=np.int64))
                per_split[split]["smoke_logistic_accuracy"] = float(
                    (clf.predict((ev_feat - mean_s) / std_s)
                     == np.asarray(ev_labels)).mean())
        np.savez(cache_root / f"{name}_probe.npz",
                 weights=probe["weights"], bias=probe["bias"],
                 mean=probe["mean"], std=probe["std"],
                 classes=np.asarray(probe["classes"]))
        weights_hash = _sha256_hex(
            probe["weights"].astype(np.float32).tobytes()
            + np.asarray(probe["bias"], dtype=np.float32).tobytes())
        print(f"  {name}: {per_split}", flush=True)
        return {
            "train_rows": len(train_labels),
            "n_classes": len(probe["classes"]),
            "probe": "closed-form ridge least squares (one-hot targets)",
            "ridge_alpha": ridge_alpha,
            "train_digest": _content_digest(train_texts, train_labels),
            "weights_hash": weights_hash,
            "splits": per_split,
            "seconds": round(time.time() - t0, 1),
        }

    nli = config["datasets"]["nli"]
    tasks["nli"] = probe_task(
        "nli", nli["hf_id"], "train",
        ["validation_matched", "validation_mismatched"],
        lambda r: int(r["label"]))
    sst = config["datasets"]["sst2"]
    tasks["sst2"] = probe_task(
        "sst2", sst["hf_id"], "train", ["validation"],
        lambda r: int(r["label"]), config_name=sst.get("config_name"))
    imdb = config["datasets"]["imdb"]
    tasks["imdb"] = probe_task(
        "imdb", imdb["hf_id"], "train", ["test"],
        lambda r: int(r["label"]))

    evidence: dict[str, Any] = {
        "milestone": "M262",
        "cell": "language-inference arm (frozen BERT features + probes)",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "encoder": encoder_hashes,
        "tasks": tasks,
        "anchors_reference_only": config["anchors_reference_only"],
        "licenses": {
            "encoder": config["encoder"]["license"],
            "nli": nli["license_recorded"],
            "sst2": sst["license_recorded"],
            "imdb": imdb["license_recorded"],
        },
        "pending_not_claimed": {
            "refusal_tags_m247": "measured assembly pending (registered)",
            "behavior_diff_m250": "baseline pending (registered)",
        },
        "verdict": {
            "reading": ("frozen BERT features + trained probes measured "
                        "on held-out splits; published anchors cited, "
                        "never exceeded; IMDb probe evaluation-only")
        },
        "scope_note": ("publisher checkpoint frozen; probes trained on "
                       "the registered train splits; one held-out read "
                       "per task"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({k: v["splits"] for k, v in tasks.items()},
                     indent=1), flush=True)
    print(f"M262 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny row counts; pipeline check only")
    args = parser.parse_args()
    output = args.output
    if args.smoke and output == DEFAULT_OUTPUT:
        # a smoke run is inadmissible evidence — never write it where
        # the full run's evidence will be sealed
        output = DEFAULT_OUTPUT.parent / (DEFAULT_OUTPUT.name + "_smoke")
    run_m262(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
