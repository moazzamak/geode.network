"""M146 — the end-to-end arbiter: gradients through the promoted additive
code (21-bin SPM + signed sqrt + L2, the C4 winner) measure the price of
freezing.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v22.md`` (section 9
M146, Remaining-milestones recipe; 15 Aug 2026). M146 is a MEASURING
STICK — no win/loss gate; its outcome selects the shipping mode.

Rungs (all at the sealed 138k level):

- r1 frozen codes + closed-form ridge: the C4 fitter over the cached
  C2/C4 SPM codes' first 138,000 rows (the subsample rows), signed sqrt
  + L2 (p=0.5). This is the t1 anchor: the PENALTY-1.0 read must
  reproduce the sealed C4 138k read 0.2273623188405797 (tol 1e-6) —
  C4's cells_138k protocol fits penalties [1.0] ONLY, so the anchor is
  the penalty-1.0 read, and the {0.1, 10.0} rungs are reported as
  diagnostics, never selected for the anchor.
- r2 frozen transformed codes + TRAINED linear head (4 epochs): the E5
  read on the promoted codes (the M109 t2 pattern).
- r3 trainable dictionary + trained head through the differentiable
  SPM+sqrt encoder (8 epochs); the whitener stays FROZEN (the M109 t3
  structure).

The shared schedule is M109's registered one, unchanged: AdamW, batch
64, cosine LR 3e-4, weight decay 1e-4, early stopping patience 2,
validation fraction 0.05 (val_seed 66). The frozen anchor stays on the
exact numpy path; the trained rungs use float32 torch arithmetic. The
differentiable transform clamps pooled activations at 1e-12 before the
square root (the training-side numerical choice; the numpy path has no
clamp).

Shipping map (registered): r3 clearly above r1 -> gradients pay on the
winner construction -> hybrid / trained-small shipping; r3 below r1 (the
M109 sealed answer on the sealed construction) -> the frozen system
ships.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m146_arbiter
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
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
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
from experiments.tier4.eval_v16_m142_c2 import (
    _spm_inference_macs,
    _spm_pool,
)
from experiments.tier4.eval_v16_m142_c4 import _fit_power, _score_power
from experiments.tier4.eval_v16_m142_factorial import power_norm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v16"
                  / "m146_arbiter.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m146_arbiter"

CLASSES = 345
PATCH_DIM = 108
T1_TOLERANCE = 1e-6
T1_REFERENCE = 0.2273623188405797
SIGNED_SQRT_CLAMP = 1e-12


# ---------------------------------------------------------------------------
# differentiable promoted encoder
# ---------------------------------------------------------------------------
class SpmSqrtModel(torch.nn.Module):
    """The C2/C4 winner as a differentiable module.

    Unfold 6x6 stride 1 -> contrast normalisation -> whiten (FROZEN
    buffers) -> cdist vs a trainable dictionary -> triangle activation ->
    21-bin SPM pooling -> signed sqrt (clamped at 1e-12) -> per-row L2 ->
    linear head. Images are (B, 3, 32, 32) float in [0, 1].
    """

    def __init__(self, dictionary: np.ndarray, mean: np.ndarray,
                 whiten: np.ndarray, grid: int, classes: int,
                 contrast_epsilon: float, power: float,
                 train_dictionary: bool, device: torch.device):
        super().__init__()
        self.grid = grid
        self.contrast_epsilon = contrast_epsilon
        self.power = power
        atoms = len(dictionary)
        self.dictionary = torch.nn.Parameter(
            torch.from_numpy(np.asarray(dictionary, dtype=np.float32).copy()
                             ).to(device),
            requires_grad=train_dictionary)
        self.register_buffer(
            "whiten_mean",
            torch.from_numpy(np.asarray(mean, dtype=np.float32).copy()
                             ).to(device))
        self.register_buffer(
            "whiten_matrix",
            torch.from_numpy(np.asarray(whiten, dtype=np.float32).copy()
                             ).to(device))
        self.head = torch.nn.Linear(21 * atoms, classes, bias=True).to(device)
        torch.nn.init.normal_(self.head.weight, std=0.01)
        torch.nn.init.zeros_(self.head.bias)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        count = images.shape[0]
        unfolded = F.unfold(images, kernel_size=6, stride=1)
        # F.unfold is channel-planar (all of R, then G, then B); the sealed
        # numpy pipeline is channel-LAST per pixel (RGB interleaved). The
        # fixed reorder below makes the whitened patches the same object
        # the M108 whitener was fitted on (parity-tested).
        patches = unfolded.transpose(1, 2).reshape(-1, 3, 36)
        patches = patches.permute(0, 2, 1).reshape(-1, PATCH_DIM)
        centre = patches.mean(dim=1, keepdim=True)
        centred = patches - centre
        variance = centred.square().mean(dim=1, keepdim=True)
        normalised = centred / torch.sqrt(variance + self.contrast_epsilon)
        white = (normalised - self.whiten_mean) @ self.whiten_matrix
        distances = torch.cdist(white, self.dictionary)
        activation = torch.clamp(distances.mean(dim=1, keepdim=True)
                                 - distances, min=0.0)
        pooled = _spm_pool(activation, count, self.grid)
        signed = torch.sqrt(pooled.clamp(min=SIGNED_SQRT_CLAMP))
        norms = signed.norm(dim=1, keepdim=True)
        norms = norms.clamp(min=SIGNED_SQRT_CLAMP)
        return signed / norms

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))


class HeadOnly(torch.nn.Module):
    """A trained 345-way head over pre-computed (frozen) codes."""

    def __init__(self, width: int, classes: int, device: torch.device):
        super().__init__()
        self.head = torch.nn.Linear(width, classes, bias=True).to(device)
        torch.nn.init.normal_(self.head.weight, std=0.01)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.head(codes)


# ---------------------------------------------------------------------------
# batch factories
# ---------------------------------------------------------------------------
def _codes_factory(memmap: np.ndarray, labels: np.ndarray, rows: np.ndarray,
                   power: float, batch: int, device: torch.device
                   ) -> Callable[[], Iterator]:
    """Stream (power_norm(codes), labels) from the cached SPM memmap."""

    def gen() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for start in range(0, len(rows), batch):
            take = rows[start:start + batch]
            block = power_norm(memmap[take], power).astype(np.float32)
            yield (torch.from_numpy(np.ascontiguousarray(block)).to(device),
                   torch.from_numpy(labels[take]).to(device))
    return gen


def _images_factory(images: np.ndarray, labels: np.ndarray, rows: np.ndarray,
                    batch: int, device: torch.device
                    ) -> Callable[[], Iterator]:
    """Stream (images, labels); images are (B, 3, 32, 32) float in [0,1]."""

    def gen() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for start in range(0, len(rows), batch):
            take = rows[start:start + batch]
            block = np.ascontiguousarray(images[take].transpose(0, 3, 1, 2))
            yield (torch.from_numpy(block).to(device).float() / 255.0,
                   torch.from_numpy(labels[take]).to(device))
    return gen


def _model_test_accuracy(model: torch.nn.Module,
                         factory: Callable[[], Iterator]) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in factory():
            logits = model(inputs)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += len(labels)
    return correct / total


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run_m146(config_path: Path, output_dir: Path) -> dict[str, Any]:
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

    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    torch.manual_seed(int(config["numerics"]["seed"]))
    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    block = int(config["numerics"]["block"])
    evidence: dict[str, Any] = {
        "milestone": "M146",
        "cell": "end-to-end arbiter on the promoted SPM+sqrt construction",
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
                               int(config["corpus"]["pixel_identity_rows"]))

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
    width = 21 * atoms
    schedule = config["schedule"]
    batch = int(schedule["batch"])
    lr = float(schedule["learning_rate"])
    wd = float(schedule["weight_decay"])
    patience = int(schedule["early_stopping"]["patience"])
    epochs = {k: int(v) for k, v in schedule["epochs"].items()}

    # val split (the registered schedule's convention)
    order = np.random.default_rng(config["corpus"]["shuffle_seed"]).permutation(
        n_train)
    val_count = int(round(n_train * float(schedule["validation_fraction"])))
    train_fit = order[val_count:]
    val_rows = order[:val_count]

    rungs: dict[str, Any] = {}
    anchors: dict[str, Any] = {}

    # ---- r1: frozen codes + closed-form ridge (the t1 anchor) -------------
    print("r1: frozen codes + closed-form ridge", flush=True)
    penalties = [0.1, 1.0, 10.0]
    solved, std = _fit_power(train_mem, labels, power, penalties, n_train,
                             block, transform=True)
    for penalty in penalties:
        acc = _score_power(test_mem, test_labels, test_domains, power,
                           solved[str(penalty)], std, block, transform=True)
        rungs[f"r1_ridge_{penalty}"] = {"accuracy": acc}
    # the anchor is the PENALTY-1.0 read: C4's sealed cells_138k protocol
    # fits penalties [1.0] only, so no other rung may be selected for it
    r1_acc = rungs["r1_ridge_1.0"]["accuracy"]
    rungs["r1"] = {"accuracy": r1_acc, "anchor_penalty": 1.0,
                   "head": "closed_form_ridge",
                   "trainable_parameters": 0,
                   "macs_per_image": _spm_inference_macs(atoms, grid,
                                                         PATCH_DIM, CLASSES),
                   "pixels": "cached C2/C4 codes (numpy path)",
                   "ladder_note": "r1_ridge_0.1/10.0 are diagnostics only"}
    print(f"  r1: {r1_acc:.6f} (anchor penalty 1.0)", flush=True)
    if not skip_anchors:
        anchors["t1"] = {"measured": r1_acc, "sealed": T1_REFERENCE,
                         "delta": r1_acc - T1_REFERENCE,
                         "tolerance": T1_TOLERANCE}
        print(f"  t1 anchor delta {r1_acc - T1_REFERENCE:+.3e}", flush=True)
        if abs(r1_acc - T1_REFERENCE) > T1_TOLERANCE:
            evidence.update({"void": True,
                             "void_reason": "t1 frozen reproduction failed",
                             "anchors": anchors, "rungs": rungs})
            _write(output_dir, evidence)
            return evidence

    # ---- r2: trained head on the frozen transformed codes -----------------
    print("r2: trained head on frozen transformed codes", flush=True)
    model = HeadOnly(width, CLASSES, device)
    training = _train_with_schedule(
        model,
        _codes_factory(train_mem, labels, train_fit, power, batch, device),
        _codes_factory(train_mem, labels, val_rows, power, batch, device),
        epochs["r2"], lr, wd, device, patience)
    r2_acc = _model_test_accuracy(
        model, _codes_factory(test_mem, test_labels,
                              np.arange(len(test_labels)), power, batch,
                              device))
    rungs["r2"] = {"accuracy": r2_acc, "head": "trained_linear",
                   "trainable_parameters": sum(p.numel() for p in
                                               model.parameters()),
                   "training": training,
                   "codes": "cached C2/C4 codes, power_norm (numpy float64, "
                            "cast float32)"}
    print(f"  r2: {r2_acc:.6f} (val {training['best_validation_accuracy']:.6f})",
          flush=True)
    del model
    torch.cuda.empty_cache()

    # ---- r3: trainable dictionary + head through the encoder --------------
    print("r3: building the differentiable SPM+sqrt model", flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    dictionary = _random_dictionary(candidates, len(candidates),
                                    int(config["sparse"]["dictionary_seed"]),
                                    atoms)
    model = SpmSqrtModel(dictionary, whitener.mean, whitener.whiten, grid,
                         CLASSES, float(config["sparse"]["contrast_epsilon"]),
                         power, True, device)
    print(f"r3: training ({epochs['r3']} epochs)", flush=True)
    training = _train_with_schedule(
        model,
        _images_factory(corpus["train_images"], corpus["train_labels"],
                        train_fit, batch, device),
        _images_factory(corpus["train_images"], corpus["train_labels"],
                        val_rows, batch, device),
        epochs["r3"], lr, wd, device, patience)
    r3_acc = _model_test_accuracy(
        model, _images_factory(corpus["test_images"], test_labels,
                               np.arange(len(test_labels)), batch, device))
    rungs["r3"] = {"accuracy": r3_acc, "head": "trained_linear",
                   "dictionary": "trainable", "whitener": "frozen",
                   "trainable_parameters": sum(p.numel() for p in
                                               model.parameters()
                                               if p.requires_grad),
                   "training": training,
                   "codes": "differentiable torch float32 forward"}
    print(f"  r3: {r3_acc:.6f} (val {training['best_validation_accuracy']:.6f})",
          flush=True)
    del model
    torch.cuda.empty_cache()

    # ---- shipping map -----------------------------------------------------
    evidence.update({
        "anchors": anchors,
        "rungs": rungs,
        "schedule": {"batch": batch, "learning_rate": lr,
                     "weight_decay": wd, "patience": patience,
                     "epochs": epochs,
                     "val_count": int(val_count),
                     "train_fit_rows": int(len(train_fit))},
        "shipping_map": {
            "registered": config["rungs"].get(
                "shipping_map",
                "r3 clearly above r1 -> hybrid/trained shipping; r3 below r1 "
                "-> the frozen system ships."),
            "r3_minus_r1": r3_acc - r1_acc,
            "reading": ("gradients pay on the winner construction "
                        "(hybrid/trained shipping)" if r3_acc > r1_acc
                        else "freezing holds on the winner construction "
                             "(the frozen system ships)"),
        },
        "runtime_seconds": round(time.time() - started, 2),
    })
    _write(output_dir, evidence)
    print(f"\nM146 complete -> {output_dir / 'evidence.json'}", flush=True)
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
    run_m146(args.config, args.output)


if __name__ == "__main__":
    main()
