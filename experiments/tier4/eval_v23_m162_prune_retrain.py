"""M162 — prune + retrain dense (the industry default at keep=0.5).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md`` (section 4
M162; the section 6 build entry, 17 Aug 2026). Prunes the M144 torch
DINOv2-small at keep=0.5 (the M144 ``_prune``), retrains it under the
M109 shared schedule on the M107-materialised r56 pixels with a
trainable 1,536 -> 345 head over the SAME features the ridge readout
uses (CLS + mean-patch tokens), with the prune mask re-applied after
every optimizer step. After training the head is discarded and the
M144 ridge readout (penalty 1.0) fits the retrained encoder's r56
features.

Anchors (before any retrained number): the M144 t1 parity guard and the
t2 UNPRUNED reproduction (0.245014492753623 within 0.002). No gate:
the retrained keep=0.5 read is reported against the M144 no-retrain
read (0.1076231884057971 @ 185.0M) and the additive recipe (0.278551 @
175.2M) at the disclosed effective MACs. Smoke declares inadmissibility
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
from experiments.tier4.eval_v15_m107_dense import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _index_digest,
    _materialise_original,
    _transformer_macs,
)
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v16_m109_trunk import (
    _load_corpus,
    _parity_guard,
)
from experiments.tier4.eval_v16_m144_pruned_dense import (
    _effective_macs,
    _encode_features,
    _fit_and_score,
    _load_torch_dinov2_small,
    _prune,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v23"
                  / "m162_prune_retrain.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v23"
                  / "m162_prune_retrain")

CLASSES = 345
RESOLUTION = 56
T2_TOLERANCE = 0.002
T2_REFERENCE = 0.245014492753623
M144_NO_RETRAIN = 0.1076231884057971
M144_NO_RETRAIN_MACS = 185_029_632
ADDITIVE_ACC = 0.27855072463768116
ADDITIVE_MACS = 175_200_000


def _train_masked(model, head, device, pix, labels, train_rows, val_rows,
                  epochs: int, lr: float, wd: float, patience: int,
                  batch: int) -> dict[str, Any]:
    """Train the pruned encoder + head; re-apply the prune mask per step."""
    masks = {name: (p.detach() != 0).float()
             for name, p in model.named_parameters()
             if name.startswith("encoder.") and "mask" not in name}
    optimiser = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=lr,
        weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                           T_max=epochs)
    best_val = 0.0
    best_state = {k: v.detach().clone()
                  for k, v in {**model.state_dict(),
                               **head.state_dict()}.items()}
    patience_left = patience
    history = []

    def _forward(rows):
        block = pix[rows].astype(np.float32) / 255.0
        block = (block - IMAGENET_MEAN) / IMAGENET_STD
        block = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
        tokens = model(pixel_values=torch.from_numpy(block).to(device)
                       ).last_hidden_state
        # Torch-native M107 features (CLS + mean patch tokens) so the
        # encoder stays on the autograd graph and gradients reach the
        # kept weights (the registered retrain). The numpy `feature`
        # path remains only in the no-grad encode/anchor arms.
        feats = torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)],
                          dim=1)
        return head(feats), torch.from_numpy(labels[rows]).to(device)

    def _val_acc():
        model.eval()
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for start in range(0, len(val_rows), batch):
                take = val_rows[start:start + batch]
                logits, labels_t = _forward(take)
                correct += int((logits.argmax(dim=1) == labels_t).sum().item())
                total += len(labels_t)
        return correct / total

    for epoch in range(epochs):
        model.train()
        head.train()
        total, correct = 0, 0
        for start in range(0, len(train_rows), batch):
            take = train_rows[start:start + batch]
            optimiser.zero_grad(set_to_none=True)
            logits, labels_t = _forward(take)
            loss = F.cross_entropy(logits, labels_t)
            loss.backward()
            optimiser.step()
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in masks:
                        p.mul_(masks[name])
            correct += int((logits.argmax(dim=1) == labels_t).sum().item())
            total += len(labels_t)
        scheduler.step()

        val_acc = _val_acc()
        history.append({"epoch": epoch + 1,
                        "train_accuracy": correct / total,
                        "validation_accuracy": val_acc})
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone()
                          for k, v in {**model.state_dict(),
                                       **head.state_dict()}.items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict({k: v for k, v in best_state.items()
                           if k in model.state_dict()})
    head.load_state_dict({k: v for k, v in best_state.items()
                          if k in head.state_dict()})
    return {"epochs_run": len(history), "best_validation_accuracy": best_val,
            "history": history}


def run_m162(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    batch = int(config["numerics"]["batch"])
    keep = float(config["arm"]["keep"])

    print("loading corpus (138k subsample + raw)", flush=True)
    corpus, train_index, test_index = _load_corpus(config)
    from experiments.tier4.eval_v15_m107_dense import _dinov2_geometry
    geometry = _dinov2_geometry("small")

    digest = _index_digest({"train": train_index, "test": test_index})
    expected = config["corpus"].get("expected_subsample_sha256")
    if expected and expected != digest:
        raise SystemExit(f"M162 subsample digest {digest} != registered")
    tag = digest[:16]
    import PIL
    train_pix = _materialise_original("train", train_index, [RESOLUTION], tag)
    test_pix = _materialise_original("test", test_index, [RESOLUTION], tag)
    pix_train = np.load(train_pix[RESOLUTION], mmap_mode="r")
    pix_test = np.load(test_pix[RESOLUTION], mmap_mode="r")
    print(f"  pixels: {train_pix[RESOLUTION]} / {test_pix[RESOLUTION]} "
          f"(materialised under .venv PIL 12.3.0; runner env PIL "
          f"{PIL.__version__} reads only)", flush=True)
    del corpus["train_images"], corpus["test_images"]

    evidence: dict[str, Any] = {
        "milestone": "M162",
        "cell": "prune + retrain dense (DINOv2-small keep=0.5, r56)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
        "geometry": geometry,
        "pixels": {"tag": tag, "train": str(train_pix[RESOLUTION]),
                   "test": str(test_pix[RESOLUTION])},
    }

    # ---- anchors: t1 parity + t2 unpruned reproduction ---------------------
    print("t1 parity guard", flush=True)
    parity = _parity_guard(torch, config, device)
    evidence["parity_guard"] = parity
    print(f"  parity worst relative difference "
          f"{parity['worst_relative_difference']:.3e} "
          f"(bound {parity['bound']})", flush=True)
    if parity["worst_relative_difference"] > parity["bound"] \
            and not skip_anchors:
        raise SystemExit("M162 t1 parity guard failed")

    n_train = (len(train_index) if not smoke else smoke_train)
    n_test = (len(test_index) if not smoke else smoke_test)
    train_rows = np.arange(n_train)
    test_rows = np.arange(n_test)

    anchors: dict[str, Any] = {}
    if not skip_anchors:
        print("t2: unpruned r56 reproduction", flush=True)
        model0 = _load_torch_dinov2_small(device)
        feat0_tr = _encode_features(model0, device, pix_train, train_rows,
                                    batch)
        feat0_te = _encode_features(model0, device, pix_test, test_rows,
                                    batch)
        t2 = _fit_and_score(feat0_tr, corpus["train_labels"][:n_train],
                            feat0_te, corpus["test_labels"][:n_test], block)
        t2_delta = t2["accuracy"] - T2_REFERENCE
        anchors["t2_unpruned"] = {"accuracy": t2["accuracy"],
                                  "reference": T2_REFERENCE,
                                  "delta": t2_delta,
                                  "tolerance": T2_TOLERANCE}
        print(f"  t2 unpruned {t2['accuracy']:.4f} (delta {t2_delta:+.6f})",
              flush=True)
        del model0, feat0_tr, feat0_te
        torch.cuda.empty_cache()
        if abs(t2_delta) > T2_TOLERANCE:
            evidence.update({"void": True,
                             "void_reason": "t2 unpruned reproduction "
                                            "failed",
                             "anchors": anchors})
            _write(output_dir, evidence)
            return evidence

    # ---- prune + retrain ---------------------------------------------------
    print(f"pruning at keep={keep} + retraining", flush=True)
    model = _load_torch_dinov2_small(device)
    stats = _prune(model, keep)
    print(f"  prune stats: {stats}", flush=True)
    head = torch.nn.Linear(2 * model.config.hidden_size, CLASSES,
                           bias=True).to(device)
    torch.nn.init.normal_(head.weight, std=0.01)
    torch.nn.init.zeros_(head.bias)

    order = np.random.default_rng(config["corpus"]["shuffle_seed"]).permutation(
        n_train)
    val_count = int(round(n_train * float(config["schedule"]
                                          ["validation_fraction"])))
    train_fit = order[val_count:]
    val_rows = order[:val_count]
    epochs = int(config["schedule"]["epochs"])
    if smoke:
        epochs = min(epochs, smoke_epochs)
    lr = float(config["schedule"]["learning_rate"])
    wd = float(config["schedule"]["weight_decay"])
    patience = int(config["schedule"]["patience"])

    training = _train_masked(
        model, head, device, pix_train, corpus["train_labels"], train_fit,
        val_rows, epochs, lr, wd, patience, batch)
    print(f"  training: {training['epochs_run']} epochs, best val "
          f"{training['best_validation_accuracy']:.6f}", flush=True)
    del head
    torch.cuda.empty_cache()

    print("re-encoding with the retrained pruned encoder", flush=True)
    feat_tr = _encode_features(model, device, pix_train, train_rows, batch)
    feat_te = _encode_features(model, device, pix_test, test_rows, batch)
    result = _fit_and_score(feat_tr, corpus["train_labels"][:n_train],
                            feat_te, corpus["test_labels"][:n_test], block)
    result["effective_macs"] = _effective_macs(geometry, stats)
    result["prune_stats"] = stats
    result["training"] = training
    print(f"  retrained keep={keep}: {result['accuracy']:.4f} @ "
          f"{result['effective_macs']} MACs", flush=True)

    evidence.update({
        "anchors": anchors,
        "retrained_arm": result,
        "context": {
            "m144_no_retrain": {"accuracy": M144_NO_RETRAIN,
                                "effective_macs": M144_NO_RETRAIN_MACS},
            "additive_recipe": {"accuracy": ADDITIVE_ACC,
                                "macs": ADDITIVE_MACS},
            "note": "measuring stick: the retrained keep=0.5 read is "
                    "reported against the M144 no-retrain read and the "
                    "additive recipe at the disclosed effective MACs.",
        },
        "runtime_seconds": round(time.time() - started, 2),
    })
    _write(output_dir, evidence)
    print(f"\nM162 complete -> {output_dir / 'evidence.json'}", flush=True)
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
    run_m162(args.config, args.output)


if __name__ == "__main__":
    main()
