"""M109 — trunk training: does the crossing survive when both families train
their own representation?

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v16.md`` section 5.2 and
``experiments/configs/v16/m109_trunk.json``.

Both families in M107 were frozen. The dense trunk was frozen because freezing
is DINOv2's own evaluation protocol; the sparse trunk was frozen because there
was no compute to train it. The compute premise is now void (plan §4). M109
climbs the v12 ladder on both sides — frozen, projection, partial, full — under
one matched schedule, and asks what happens to the crossing.

The t1 (frozen) rung of each family is a *reproduction* of the sealed M107/M108
figures: dense at resolutions {28, 42, 224} must reproduce M107's
`d4a_small_28`, `d4b_small_42`, `d1_small_224`; sparse at 3072 atoms must
reproduce M108's `a_random_3072`. A rung outside the registered tolerance voids
the run (plan section 3.3), because every trained rung is read against its
frozen predecessor.

Device placement is registered in the config. The sparse whitening stays on
numpy CPU (M107's arithmetic, validated by the M108 pre-seal check to reproduce
M107 at 8.7e-05); everything trainable runs on the GPU (torch device "cuda" =
ROCm/HIP in this interpreter). The parity guard of plan §4.6.1 runs first.

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m109_trunk
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from experiments.common.data_cache import configure_external_cache_environment
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m103_atoms import (
    Whitener,
    _contrast_normalise,
    _extract_patches,
    _fit_zca,
    _pool,
)
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    _inference_macs,
    _load_domainnet,
    _score,
)
from experiments.tier4.eval_v15_m107_dense import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _class_subsample,
    _index_digest,
    _materialise_original,
    _transformer_macs,
    _verify_pixel_identity,
)
from experiments.tier4.eval_v16_m108_dictionary import (
    _encode_block_device,
    _verify_device,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m109_trunk.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m109_trunk"
M107_EVIDENCE = REPO_ROOT / "logs" / "results" / "v15" / "m107_dense" / "evidence.json"
M108_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m108_dictionary" / "evidence.json"

T1_REPRODUCTION_TOLERANCE = 0.002

# The t1 dense reproduction targets (M107 sealed, penalty 1.0).
DENSE_T1_TARGETS = {28: "d4a_small_28", 42: "d4b_small_42", 224: "d1_small_224"}


# --------------------------------------------------------------------------
# parity guard (plan section 4.6.1)
# --------------------------------------------------------------------------
def _parity_guard(torch, config: dict[str, Any], device: torch.device
                  ) -> dict[str, Any]:
    """Re-run the onnx/CPU vs torch/CUDA parity check on a fixed input.

    A run whose worst relative disagreement exceeds the registered bound is
    VOID, not negative.
    """
    import onnxruntime as ort

    from experiments.common.data_cache import data_cache_root
    from experiments.tier4.bench_v16_parity import feature, fixed_input

    guard = config["parity_guard"]
    resolution = int(guard["resolution"])
    batch = int(guard["batch"])
    bound = float(guard["bound_relative"])
    data = fixed_input(batch, resolution)

    worst = 0.0
    details = []
    for name in guard["models"]:
        found = sorted((data_cache_root() / "huggingface" / "hub").glob(
            f"models--onnx-community--dinov2-{name}-ONNX/snapshots/*/onnx/model.onnx"))
        if not found:
            raise SystemExit(f"parity guard: no ONNX export for dinov2-{name}")
        session = ort.InferenceSession(str(found[0]),
                                       providers=["CPUExecutionProvider"])
        onnx_tokens = session.run(None, {"pixel_values": data})[0]
        onnx_feature = feature(onnx_tokens)

        from transformers import Dinov2Model
        weights = data_cache_root() / "torch" / f"dinov2-{name}"
        model = Dinov2Model.from_pretrained(str(weights), dtype=torch.float32)
        model.eval().to(device)
        with torch.no_grad():
            tokens = model(pixel_values=torch.from_numpy(data).to(device))
        torch_feature = feature(tokens.last_hidden_state.cpu().numpy())

        relative = float(np.abs(onnx_feature - torch_feature).max()
                         / (np.abs(onnx_feature).max() + 1e-12))
        worst = max(worst, relative)
        details.append({"model": name, "worst_relative_difference": relative})
        del model
        torch.cuda.empty_cache()

    if worst > bound:
        raise SystemExit(
            f"M109 VOID: parity guard worst relative disagreement {worst:.3e} "
            f"exceeds the registered bound {bound}.")
    return {"bound": bound, "worst_relative_difference": worst, "pairs": details,
            "resolution": resolution, "batch": batch}


# --------------------------------------------------------------------------
# corpus, whitener, dictionary (reproduce M108's arm (a) construction)
# --------------------------------------------------------------------------
def _load_corpus(config: dict[str, Any]) -> tuple[dict[str, np.ndarray],
                                                  np.ndarray, np.ndarray]:
    cc = config["corpus"]
    raw = _load_domainnet(cc["image_size"])
    train_index = _class_subsample(raw["train_labels"],
                                   cc["train_rows_per_class"], cc["subsample_seed"])
    test_index = _class_subsample(raw["test_labels"],
                                  cc["test_rows_per_class"], cc["subsample_seed"])
    digest = _index_digest({"train": train_index, "test": test_index})
    expected = cc.get("expected_subsample_sha256")
    if expected and expected != digest:
        raise SystemExit(f"M109 subsample digest {digest} != registered")
    corpus = {
        "train_images": raw["train_images"][train_index],
        "train_labels": raw["train_labels"][train_index],
        "train_domains": raw["train_domains"][train_index],
        "test_images": raw["test_images"][test_index],
        "test_labels": raw["test_labels"][test_index],
        "test_domains": raw["test_domains"][test_index],
    }
    del raw
    return corpus, train_index, test_index


def _build_whitener_dictionary(config: dict[str, Any],
                               corpus: dict[str, np.ndarray]
                               ) -> tuple[Whitener, np.ndarray, int, int, int]:
    """M108's exact whitener and arm (a) dictionary at the registered atoms."""
    size = config["corpus"]["image_size"]
    sparse = config["sparse"]
    patch, stride = 6, 1
    pool_grid = int(sparse["pool_grid"])
    rep = {"zca_fit_patches": 400000, "zca_fit_seed": 11,
           "contrast_epsilon": 10.0, "zca_epsilon": 0.1}

    rng = np.random.default_rng(rep["zca_fit_seed"])
    sample = corpus["train_images"][rng.choice(
        len(corpus["train_images"]), min(len(corpus["train_images"]), 20000),
        replace=False)]
    patches = _extract_patches(sample, patch, stride)
    grid = (size - patch) // stride + 1
    take = min(rep["zca_fit_patches"], len(patches))
    pool = _contrast_normalise(
        patches[rng.choice(len(patches), take, replace=False)],
        rep["contrast_epsilon"])
    mean, whiten = _fit_zca(pool, rep["zca_epsilon"])
    whitener = Whitener(patch, stride, rep["contrast_epsilon"], mean, whiten, grid)

    seed = 11
    pool_size = 8192
    srng = np.random.default_rng(seed)
    candidates = ((pool[srng.choice(len(pool), pool_size, replace=False)]
                   - mean) @ whiten).astype(np.float32)
    order = np.random.default_rng([seed, 100]).permutation(pool_size)
    atoms = int(sparse["atoms"])
    dictionary = candidates[order[:atoms]]
    dimension = patch * patch * 3
    return whitener, dictionary, grid, dimension, pool_grid


# --------------------------------------------------------------------------
# differentiable sparse model (t3/t4)
# --------------------------------------------------------------------------
class SparseModel(torch.nn.Module):
    """The sparse side as a differentiable module.

    Whitened 6x6 patches -> triangle activation against a trainable dictionary
    -> 2x2 pooling -> linear head. The whitener is a parameter only at t4, the
    dictionary only at t3/t4. The frozen rungs (t1/t2) use the exact
    numpy-whitened M108 path instead so they reproduce the sealed figures.
    """

    def __init__(self, dictionary: np.ndarray, mean: np.ndarray, whiten: np.ndarray,
                 patch: int, stride: int, grid: int, pool_grid: int, classes: int,
                 contrast_epsilon: float, train_dictionary: bool,
                 train_whitener: bool, device: torch.device):
        super().__init__()
        self.patch = patch
        self.stride = stride
        self.grid = grid
        self.pool_grid = pool_grid
        self.contrast_epsilon = contrast_epsilon
        atoms = len(dictionary)

        self.dictionary = torch.nn.Parameter(
            torch.from_numpy(dictionary.copy()).to(device),
            requires_grad=train_dictionary)
        self.whiten_mean = torch.nn.Parameter(
            torch.from_numpy(mean.astype(np.float32).copy()).to(device),
            requires_grad=train_whitener)
        self.whiten_matrix = torch.nn.Parameter(
            torch.from_numpy(whiten.astype(np.float32).copy()).to(device),
            requires_grad=train_whitener)
        self.head = torch.nn.Linear(pool_grid * pool_grid * atoms, classes,
                                    bias=True).to(device)
        torch.nn.init.normal_(self.head.weight, std=0.01)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, 3, 32, 32) float in [0,1]
        unfolded = F.unfold(images, kernel_size=self.patch, stride=self.stride)
        patches = unfolded.transpose(1, 2).reshape(-1, self.patch * self.patch * 3)
        centre = patches.mean(dim=1, keepdim=True)
        centred = patches - centre
        variance = centred.square().mean(dim=1, keepdim=True)
        normalised = centred / torch.sqrt(variance + self.contrast_epsilon)
        white = (normalised - self.whiten_mean) @ self.whiten_matrix
        distances = torch.cdist(white, self.dictionary)
        activation = torch.clamp(distances.mean(dim=1, keepdim=True) - distances,
                                 min=0.0)
        pooled = _pool(activation, len(images), self.grid, self.pool_grid)
        return self.head(pooled)


# --------------------------------------------------------------------------
# dense model + LoRA
# --------------------------------------------------------------------------
class DenseModel(torch.nn.Module):
    """Dinov2Model behind M107's feature definition plus a linear head."""

    def __init__(self, name: str, classes: int, device: torch.device,
                 from_scratch: bool = False, seed: int | None = None):
        super().__init__()
        from transformers import Dinov2Model
        from experiments.common.data_cache import data_cache_root

        weights = data_cache_root() / "torch" / f"dinov2-{name}"
        if from_scratch:
            config_model = Dinov2Model.from_pretrained(str(weights)).config
            self.trunk = Dinov2Model(config_model).to(device)
            if seed is not None:
                torch.manual_seed(seed)
                for module in self.trunk.modules():
                    if hasattr(module, "reset_parameters"):
                        try:
                            module.reset_parameters()
                        except Exception:
                            pass
        else:
            self.trunk = Dinov2Model.from_pretrained(
                str(weights), dtype=torch.float32).to(device)
        self.width = 2 * self.trunk.config.hidden_size
        self.head = torch.nn.Linear(self.width, classes).to(device)
        torch.nn.init.normal_(self.head.weight, std=0.01)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        tokens = self.trunk(pixel_values=pixels).last_hidden_state
        feature = torch.cat([tokens[:, 0, :], tokens[:, 1:, :].mean(dim=1)], dim=1)
        return self.head(feature)

    def features(self, pixels: torch.Tensor) -> torch.Tensor:
        tokens = self.trunk(pixel_values=pixels).last_hidden_state
        return torch.cat([tokens[:, 0, :], tokens[:, 1:, :].mean(dim=1)], dim=1)


class _LoRALinear(torch.nn.Module):
    """Low-rank adapter around a frozen linear projection (Hu et al., 2021)."""

    def __init__(self, base: torch.nn.Linear, rank: int):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.weight.shape[1], base.weight.shape[0]
        device = base.weight.device
        self.lora_a = torch.nn.Parameter(torch.zeros(in_f, rank, device=device))
        self.lora_b = torch.nn.Parameter(
            torch.randn(rank, out_f, device=device) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_a) @ self.lora_b


def _apply_lora(model: DenseModel, rank: int) -> int:
    """Replace the qkv/out projections of every block with LoRA adapters."""
    count = 0
    for block in model.trunk.encoder.layer:
        attn = block.attention.attention
        out = block.attention.output
        replacements = [
            (attn, "query"), (attn, "key"), (attn, "value"), (out, "dense"),
        ]
        for parent, name in replacements:
            module = getattr(parent, name)
            adapter = _LoRALinear(module, rank)
            setattr(parent, name, adapter)
            count += adapter.lora_a.numel() + adapter.lora_b.numel()
    return count


# --------------------------------------------------------------------------
# training (one schedule, both families)
# --------------------------------------------------------------------------
def _train_with_schedule(model: torch.nn.Module,
                         train_factory: Callable[[], Iterator],
                         val_factory: Callable[[], Iterator],
                         epochs: int, lr: float, weight_decay: float,
                         device: torch.device, patience: int) -> dict[str, Any]:
    """AdamW + cosine schedule + early stopping on validation accuracy.

    Identical procedure for both families (plan section 5.2 restriction 1).
    The factories return FRESH generators each call so every epoch sees all
    rows. The optimizer sees the model's trainable parameters only.
    """
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    best_val = 0.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience_left = patience
    history = []
    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        for inputs, labels in train_factory():
            optimiser.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimiser.step()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += len(labels)
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_factory():
                logits = model(inputs)
                val_correct += int((logits.argmax(dim=1) == labels).sum().item())
                val_total += len(labels)
        val_acc = val_correct / val_total
        history.append({"epoch": epoch + 1, "train_accuracy": correct / total,
                        "validation_accuracy": val_acc})
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    return {"epochs_run": len(history), "best_validation_accuracy": best_val,
            "history": history}


# --------------------------------------------------------------------------
# data batch factories
# --------------------------------------------------------------------------
def _dense_pixels(config, train_index, test_index):
    """{split: {size: memmap}} over the original-resolution pixels."""
    resolutions = [int(r) for r in config["dense"]["resolutions"]]
    tag = "m109"
    return {
        "train": _materialise_original("train", train_index, resolutions, tag),
        "test": _materialise_original("test", test_index, resolutions, tag),
    }


def _dense_batches_factory(memmap, labels, rows, batch, device):
    """Factory of (pixels, labels) generators for one resolution/split."""

    def gen():
        for start in range(0, len(rows), batch):
            take = rows[start:start + batch]
            block = np.asarray(memmap[take], dtype=np.float32) / 255.0
            block = (block - IMAGENET_MEAN) / IMAGENET_STD
            block = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
            yield (torch.from_numpy(block).to(device),
                   torch.from_numpy(labels[take]).to(device))
    return gen


def _sparse_batches_factory(images, labels, rows, batch, device):
    """Factory of (images, labels) generators for the differentiable sparse
    model. Images are (B, 3, 32, 32) float in [0,1]."""

    def gen():
        for start in range(0, len(rows), batch):
            take = rows[start:start + batch]
            block = np.ascontiguousarray(images[take].transpose(0, 3, 1, 2))
            yield (torch.from_numpy(block).to(device).float() / 255.0,
                   torch.from_numpy(labels[take]).to(device))
    return gen


def _sparse_frozen_batches_factory(images, labels, rows, batch, table, whitener,
                                   pool_grid, device):
    """Factory of (frozen_codes, labels) generators using M108's exact encode."""

    def gen():
        for start in range(0, len(rows), batch):
            take = rows[start:start + batch]
            block = _encode_block_device(images[take], table, whitener, pool_grid)
            yield (torch.from_numpy(block).to(device),
                   torch.from_numpy(labels[take]).to(device))
    return gen


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run_m109(config_path: Path, output_dir: Path, progress: bool = True
             ) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    torch.set_num_threads(config["numerics"]["torch_threads"])
    torch.manual_seed(config["numerics"]["seed"])
    configure_external_cache_environment()
    device_report = _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("parity guard at startup", flush=True)
    parity = _parity_guard(torch, config, device)

    print("loading corpus", flush=True)
    corpus, train_index, test_index = _load_corpus(config)
    test_seq = np.arange(len(test_index))
    classes = int(corpus["train_labels"].max()) + 1
    size = config["corpus"]["image_size"]
    identity_checks = [
        _verify_pixel_identity(
            split, idx, corpus[f"{split}_images"], size,
            config["corpus"]["pixel_identity_rows"],
        )
        for split, idx in (("train", train_index), ("test", test_index))
    ]

    print("building whitener and dictionary (M108 arm (a) at 3072)", flush=True)
    whitener, dictionary, grid, dimension, pool_grid = _build_whitener_dictionary(
        config, corpus)
    atoms = int(config["sparse"]["atoms"])
    sparse_macs = _inference_macs(atoms, grid, dimension, pool_grid, classes)
    order = np.random.default_rng(config["corpus"]["shuffle_seed"]).permutation(
        len(train_index))
    val_count = int(round(len(train_index) * config["schedule"]["validation_fraction"]))
    train_fit = order[val_count:]
    val_rows = order[:val_count]

    schedule = config["schedule"]
    batch = int(schedule["batch"])
    epochs = schedule["epochs"]
    lr = float(schedule["learning_rate"])
    wd = float(schedule["weight_decay"])
    patience = int(schedule["early_stopping"]["patience"])

    # ---- dense geometry, MACs, pixels, references -------------------------
    from experiments.tier4.eval_v15_m107_dense import _dinov2_geometry
    geometry = _dinov2_geometry("small")
    resolutions = [int(r) for r in config["dense"]["resolutions"]]
    dense_macs = {r: _transformer_macs(geometry, r, classes)["total"]
                  for r in resolutions}
    dense_pixels = _dense_pixels(config, train_index, test_index)

    m107 = json.loads(M107_EVIDENCE.read_text(encoding="utf-8"))
    m108 = json.loads(M108_EVIDENCE.read_text(encoding="utf-8"))
    penalty = 1.0

    results: dict[str, Any] = {"dense": {}, "sparse": {}}
    reproduction: dict[str, Any] = {}

    # ---- t1: frozen, closed-form ridge (reproduction) ---------------------
    print("t1 frozen (reproduction)", flush=True)
    dense_model = DenseModel("small", classes, device)
    dense_t1 = {}
    for r in resolutions:
        mem = np.load(dense_pixels["train"][r], mmap_mode="r")
        mem_test = np.load(dense_pixels["test"][r], mmap_mode="r")
        accumulator = RidgeAccumulator(dense_model.width, classes)
        for start in range(0, len(train_fit), 256):
            take = train_fit[start:start + 256]
            block = np.asarray(mem[take], dtype=np.float32) / 255.0
            block = (block - IMAGENET_MEAN) / IMAGENET_STD
            block = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
            with torch.no_grad():
                feat = dense_model.features(
                    torch.from_numpy(block).to(device)).cpu().numpy()
            accumulator.add(feat, corpus["train_labels"][take])
        solutions = accumulator.solve_many([penalty])
        standardise = accumulator.standardiser()
        correct = 0
        for start in range(0, len(test_seq), 256):
            take = test_seq[start:start + 256]
            block = np.asarray(mem_test[take], dtype=np.float32) / 255.0
            block = (block - IMAGENET_MEAN) / IMAGENET_STD
            block = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
            with torch.no_grad():
                feat = dense_model.features(
                    torch.from_numpy(block).to(device)).cpu().numpy()
            correct += int(_score(solutions[penalty], standardise(feat),
                                  corpus["test_labels"][test_seq[start:start + 256]]).sum())
        acc = correct / len(test_index)
        dense_t1[r] = acc
        results["dense"][f"t1_r{r}"] = {
            "accuracy": acc, "macs": dense_macs[r], "trainable_parameters": 0,
            "head": "closed_form_ridge_1.0"}
        print(f"  t1 dense r{r}: {acc:.4f}", flush=True)
    del dense_model
    torch.cuda.empty_cache()

    # sparse t1: frozen codes + closed-form ridge (reproduces M108 a@3072)
    table = torch.from_numpy(np.ascontiguousarray(dictionary)).to(torch.float32)
    table = table.to(device)
    accumulator = RidgeAccumulator(pool_grid * pool_grid * atoms, classes)
    for start in range(0, len(train_fit), 64):
        take = train_fit[start:start + 64]
        block = _encode_block_device(corpus["train_images"][take], table,
                                     whitener, pool_grid)
        accumulator.add(block, corpus["train_labels"][take])
    solutions = accumulator.solve_many([penalty])
    standardise = accumulator.standardiser()
    correct = 0
    for start in range(0, len(test_seq), 64):
        take = test_seq[start:start + 64]
        block = _encode_block_device(corpus["test_images"][take], table,
                                     whitener, pool_grid)
        correct += int(_score(solutions[penalty], standardise(block),
                              corpus["test_labels"][take]).sum())
    sparse_t1_acc = correct / len(test_index)
    results["sparse"]["t1"] = {
        "accuracy": sparse_t1_acc, "macs": sparse_macs["total"],
        "trainable_parameters": 0, "head": "closed_form_ridge_1.0"}
    print(f"  t1 sparse: {sparse_t1_acc:.4f}", flush=True)

    # ---- t1 reproduction gate ---------------------------------------------
    for r in resolutions:
        target = DENSE_T1_TARGETS[r]
        reference = float(m107["arms"][target]["accuracy_by_penalty"]["1.0"])
        reproduction[f"dense_t1_r{r}"] = {
            "measured": dense_t1[r], "m107": reference,
            "delta": dense_t1[r] - reference}
    sparse_ref = float(m108["arms"]["a_random_3072"]["accuracy_by_penalty"]["1.0"])
    reproduction["sparse_t1"] = {
        "measured": sparse_t1_acc, "m108": sparse_ref,
        "delta": sparse_t1_acc - sparse_ref}
    max_delta = max(abs(v["delta"]) for v in reproduction.values())
    smoke_skip = bool(config.get("_smoke_skip_reproduction_gate", False))
    if smoke_skip:
        reproduction["_skipped_by_smoke"] = (
            "SMOKE ONLY: the smoke corpus differs from M107's, so t1 cannot "
            "reproduce; the gate is bypassed so the t2-t4 training paths can "
            "be exercised. The sealed config does not carry this flag.")
    if max_delta > T1_REPRODUCTION_TOLERANCE and not smoke_skip:
        reproduction["_verdict"] = (
            "t1 reproduction outside the registered tolerance; M109 is VOID "
            "and the instrument is at fault, not the arms.")
        print(f"  T1 REPRODUCTION GATE FAILED (max delta {max_delta:.5f}) — "
              "M109 VOID", flush=True)
        write_canonical_json(output_dir / "evidence.json", {
            "milestone": "M109", "admissible_as_evidence": False, "void": True,
            "void_reason": "t1 reproduction gate failed",
            "reproduction": reproduction,
            "parity_guard": parity})
        return {"admissible_as_evidence": False, "void": True,
                "reproduction": reproduction}
    if smoke_skip:
        print(f"  t1 reproduction max delta {max_delta:.5f} — SMOKE: gate "
              "bypassed so the training rungs can execute", flush=True)
    else:
        print(f"  t1 reproduction max delta {max_delta:.5f} within tolerance",
              flush=True)

    # ---- t2: projection (linear head on frozen features/codes) ------------
    print("t2 projection", flush=True)
    dense_model = DenseModel("small", classes, device)
    for p in dense_model.trunk.parameters():
        p.requires_grad_(False)

    class HeadOnly(torch.nn.Module):
        def __init__(self, trunk, head):
            super().__init__()
            self.trunk = trunk
            self.head = head

        def forward(self, pixels):
            return self.head(self.trunk.features(pixels))

    for r in resolutions:
        head = torch.nn.Linear(dense_model.width, classes).to(device)
        torch.nn.init.normal_(head.weight, std=0.01)
        torch.nn.init.zeros_(head.bias)
        combined = HeadOnly(dense_model, head)
        mem = np.load(dense_pixels["train"][r], mmap_mode="r")
        mem_test = np.load(dense_pixels["test"][r], mmap_mode="r")
        train_out = _train_with_schedule(
            combined,
            _dense_batches_factory(mem, corpus["train_labels"], train_fit,
                                   batch, device),
            _dense_batches_factory(mem, corpus["train_labels"], val_rows,
                                   batch, device),
            epochs["t2_projection"], lr, wd, device, patience)
        correct = 0
        combined.eval()
        with torch.no_grad():
            for start in range(0, len(test_seq), batch):
                take = test_seq[start:start + batch]
                block = np.asarray(mem_test[take], dtype=np.float32) / 255.0
                block = (block - IMAGENET_MEAN) / IMAGENET_STD
                block = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
                logits = combined(torch.from_numpy(block).to(device))
                correct += int((logits.argmax(dim=1) ==
                                torch.from_numpy(corpus["test_labels"][take])
                                .to(device)).sum().item())
        acc = correct / len(test_index)
        results["dense"][f"t2_r{r}"] = {
            "accuracy": acc, "macs": dense_macs[r],
            "trainable_parameters": sum(p.numel() for p in head.parameters()),
            "head": "trained_linear", "training": train_out}
        print(f"  t2 dense r{r}: {acc:.4f}", flush=True)
    del dense_model
    torch.cuda.empty_cache()

    # sparse t2: linear head on frozen codes
    head = torch.nn.Linear(pool_grid * pool_grid * atoms, classes).to(device)
    torch.nn.init.normal_(head.weight, std=0.01)
    torch.nn.init.zeros_(head.bias)
    train_out = _train_with_schedule(
        head,
        _sparse_frozen_batches_factory(corpus["train_images"],
                                       corpus["train_labels"], train_fit, batch,
                                       table, whitener, pool_grid, device),
        _sparse_frozen_batches_factory(corpus["train_images"],
                                       corpus["train_labels"], val_rows, batch,
                                       table, whitener, pool_grid, device),
        epochs["t2_projection"], lr, wd, device, patience)
    correct = 0
    head.eval()
    with torch.no_grad():
        for block, labels in _sparse_frozen_batches_factory(
                corpus["test_images"], corpus["test_labels"], test_seq, batch,
                table, whitener, pool_grid, device)():
            correct += int((head(block).argmax(dim=1) == labels).sum().item())
    acc = correct / len(test_index)
    results["sparse"]["t2"] = {
        "accuracy": acc, "macs": sparse_macs["total"],
        "trainable_parameters": sum(p.numel() for p in head.parameters()),
        "head": "trained_linear", "training": train_out}
    print(f"  t2 sparse: {acc:.4f}", flush=True)

    # ---- t3: partial (LoRA + last-2 dense / dictionary-grad sparse) -------
    print("t3 partial", flush=True)
    dense_model = DenseModel("small", classes, device)
    lora_params = _apply_lora(dense_model, 16)
    for p in dense_model.trunk.parameters():
        p.requires_grad_(False)
    for block in dense_model.trunk.encoder.layer[-2:]:
        for p in block.parameters():
            p.requires_grad_(True)
    for r in resolutions:
        mem = np.load(dense_pixels["train"][r], mmap_mode="r")
        mem_test = np.load(dense_pixels["test"][r], mmap_mode="r")
        train_out = _train_with_schedule(
            dense_model,
            _dense_batches_factory(mem, corpus["train_labels"], train_fit,
                                   batch, device),
            _dense_batches_factory(mem, corpus["train_labels"], val_rows,
                                   batch, device),
            epochs["t3_partial"], lr, wd, device, patience)
        correct = 0
        dense_model.eval()
        with torch.no_grad():
            for start in range(0, len(test_seq), batch):
                take = test_seq[start:start + batch]
                block = np.asarray(mem_test[take], dtype=np.float32) / 255.0
                block = (block - IMAGENET_MEAN) / IMAGENET_STD
                block = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
                logits = dense_model(torch.from_numpy(block).to(device))
                correct += int((logits.argmax(dim=1) ==
                                torch.from_numpy(corpus["test_labels"][take])
                                .to(device)).sum().item())
        acc = correct / len(test_index)
        results["dense"][f"t3_r{r}"] = {
            "accuracy": acc, "macs": dense_macs[r],
            "trainable_parameters": sum(p.numel() for p in dense_model.parameters()
                                        if p.requires_grad),
            "head": "trained_linear", "training": train_out, "lora_parameters": lora_params}
        print(f"  t3 dense r{r}: {acc:.4f}", flush=True)
    del dense_model
    torch.cuda.empty_cache()

    sparse_model = SparseModel(
        dictionary, whitener.mean, whitener.whiten, 6, 1, grid, pool_grid,
        classes, 10.0, train_dictionary=True, train_whitener=False, device=device)
    train_out = _train_with_schedule(
        sparse_model,
        _sparse_batches_factory(corpus["train_images"], corpus["train_labels"],
                                train_fit, batch, device),
        _sparse_batches_factory(corpus["train_images"], corpus["train_labels"],
                                val_rows, batch, device),
        epochs["t3_partial"], lr, wd, device, patience)
    correct = 0
    sparse_model.eval()
    with torch.no_grad():
        for images, labels in _sparse_batches_factory(
                corpus["test_images"], corpus["test_labels"], test_seq, batch,
                device)():
            correct += int((sparse_model(images).argmax(dim=1) == labels).sum().item())
    acc = correct / len(test_index)
    results["sparse"]["t3"] = {
        "accuracy": acc, "macs": sparse_macs["total"],
        "trainable_parameters": sum(p.numel() for p in sparse_model.parameters()
                                    if p.requires_grad),
        "head": "trained_linear", "training": train_out}
    print(f"  t3 sparse: {acc:.4f}", flush=True)
    del sparse_model
    torch.cuda.empty_cache()

    # ---- t4: full (dense fine-tune / sparse dict+whitener) ----------------
    print("t4 full", flush=True)
    dense_model = DenseModel("small", classes, device)
    for r in resolutions:
        mem = np.load(dense_pixels["train"][r], mmap_mode="r")
        mem_test = np.load(dense_pixels["test"][r], mmap_mode="r")
        train_out = _train_with_schedule(
            dense_model,
            _dense_batches_factory(mem, corpus["train_labels"], train_fit,
                                   batch, device),
            _dense_batches_factory(mem, corpus["train_labels"], val_rows,
                                   batch, device),
            epochs["t4_full"], lr, wd, device, patience)
        correct = 0
        dense_model.eval()
        with torch.no_grad():
            for start in range(0, len(test_seq), batch):
                take = test_seq[start:start + batch]
                block = np.asarray(mem_test[take], dtype=np.float32) / 255.0
                block = (block - IMAGENET_MEAN) / IMAGENET_STD
                block = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
                logits = dense_model(torch.from_numpy(block).to(device))
                correct += int((logits.argmax(dim=1) ==
                                torch.from_numpy(corpus["test_labels"][take])
                                .to(device)).sum().item())
        acc = correct / len(test_index)
        results["dense"][f"t4_r{r}"] = {
            "accuracy": acc, "macs": dense_macs[r],
            "trainable_parameters": sum(p.numel() for p in dense_model.parameters()
                                        if p.requires_grad),
            "head": "trained_linear", "training": train_out}
        print(f"  t4 dense r{r}: {acc:.4f}", flush=True)
    del dense_model
    torch.cuda.empty_cache()

    # t4 from-scratch dense (plan section 5.2.6)
    r = int(config["dense"]["from_scratch_at_t4"]["resolution"])
    dense_scratch = DenseModel("small", classes, device, from_scratch=True,
                               seed=config["numerics"]["seed"])
    mem = np.load(dense_pixels["train"][r], mmap_mode="r")
    mem_test = np.load(dense_pixels["test"][r], mmap_mode="r")
    train_out = _train_with_schedule(
        dense_scratch,
        _dense_batches_factory(mem, corpus["train_labels"], train_fit, batch,
                               device),
        _dense_batches_factory(mem, corpus["train_labels"], val_rows, batch,
                               device),
        epochs["from_scratch_t4"], lr, wd, device, patience)
    correct = 0
    dense_scratch.eval()
    with torch.no_grad():
        for start in range(0, len(test_seq), batch):
            take = test_seq[start:start + batch]
            block = np.asarray(mem_test[take], dtype=np.float32) / 255.0
            block = (block - IMAGENET_MEAN) / IMAGENET_STD
            block = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
            logits = dense_scratch(torch.from_numpy(block).to(device))
            correct += int((logits.argmax(dim=1) ==
                            torch.from_numpy(corpus["test_labels"][take])
                            .to(device)).sum().item())
    acc = correct / len(test_index)
    results["dense"][f"t4_from_scratch_{r}"] = {
        "accuracy": acc, "macs": dense_macs[r],
        "trainable_parameters": sum(p.numel() for p in dense_scratch.parameters()
                                    if p.requires_grad),
        "head": "trained_linear", "training": train_out,
        "_symmetry_note": "plan section 5.2.6: same geometry, random init, same schedule"}
    print(f"  t4 dense from-scratch {r}: {acc:.4f}", flush=True)
    del dense_scratch
    torch.cuda.empty_cache()

    sparse_model = SparseModel(
        dictionary, whitener.mean, whitener.whiten, 6, 1, grid, pool_grid,
        classes, 10.0, train_dictionary=True, train_whitener=True, device=device)
    train_out = _train_with_schedule(
        sparse_model,
        _sparse_batches_factory(corpus["train_images"], corpus["train_labels"],
                                train_fit, batch, device),
        _sparse_batches_factory(corpus["train_images"], corpus["train_labels"],
                                val_rows, batch, device),
        epochs["t4_full"], lr, wd, device, patience)
    correct = 0
    sparse_model.eval()
    with torch.no_grad():
        for images, labels in _sparse_batches_factory(
                corpus["test_images"], corpus["test_labels"], test_seq, batch,
                device)():
            correct += int((sparse_model(images).argmax(dim=1) == labels).sum().item())
    acc = correct / len(test_index)
    results["sparse"]["t4"] = {
        "accuracy": acc, "macs": sparse_macs["total"],
        "trainable_parameters": sum(p.numel() for p in sparse_model.parameters()
                                    if p.requires_grad),
        "head": "trained_linear", "training": train_out}
    print(f"  t4 sparse: {acc:.4f}", flush=True)

    # ---- gate -------------------------------------------------------------
    gate = _build_gate(results, dense_macs, sparse_macs["total"], resolutions)
    results["gate"] = gate

    evidence = {
        "milestone": "M109",
        "question": "what happens to the M107 crossing when each family trains its own representation?",
        "registered_in": "analysis/RESEARCH_IMPLEMENTATION_PLAN_v16.md section 5.2",
        "admissible_as_evidence": not inadmissible,
        "config_file": Path(config_path).name,
        "config": config,
        "device": device_report,
        "parity_guard": parity,
        "corpus": {
            "train_rows": int(len(train_index)),
            "test_rows": int(len(test_index)),
            "classes": classes,
            "subsample_sha256": _index_digest({"train": train_index, "test": test_index}),
            "pixel_identity": identity_checks,
        },
        "t1_reproduction": reproduction,
        "results": results,
        "sparse_macs": sparse_macs,
        "dense_macs_per_resolution": dense_macs,
    }
    evidence["payload_sha256"] = payload_hash(evidence)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    return evidence


def _build_gate(results: dict[str, Any], dense_macs: dict[int, int],
                sparse_macs: int, resolutions: list[int]) -> dict[str, Any]:
    """The three registered kill switches of plan section 5.2.

    At each rung the sparse side is one point at ``sparse_macs``; the dense
    side has in-range points at the registered resolutions at or below the
    sparse MACs. M107's comparison rule: the sparse point is compared against
    the best dense point at or below its MACs.
    """
    dense = results["dense"]
    sparse = results["sparse"]
    out = {}
    for key in ("t1", "t2", "t3", "t4"):
        in_range = [r for r in resolutions if dense_macs[r] <= sparse_macs]
        if not in_range:
            raise RuntimeError(
                "M109 instrument failure: no dense resolution is at or below "
                "the sparse MACs; kill switches 1 and 2 are undecidable.")
        dense_curve = [(dense_macs[r], dense[f"{key}_r{r}"]["accuracy"])
                       for r in in_range]
        sparse_acc = sparse[key]["accuracy"]
        best_dense = max(a for m, a in dense_curve)
        out[key] = {
            "sparse_accuracy": sparse_acc,
            "sparse_macs": sparse_macs,
            "dense_at_or_below": [[m, a] for m, a in dense_curve],
            "best_dense_at_or_below_accuracy": best_dense,
            "kill_switch_1_dense_dominates": bool(best_dense > sparse_acc),
            "kill_switch_2_sparse_still_above": bool(sparse_acc >= best_dense),
        }
    t1_acc = sparse["t1"]["accuracy"]
    t4_acc = sparse["t4"]["accuracy"]
    moved = abs(t4_acc - t1_acc) >= 0.005
    out["kill_switch_3_sparse_at_capacity"] = {
        "t1_accuracy": t1_acc, "t4_accuracy": t4_acc, "fired": bool(not moved),
        "consequence": (
            "if the gradient-trained dictionary matches the constructed one, "
            "the sparse dictionary is at capacity, not at its optimum")}
    out["_comparison_rule"] = (
        "a sparse point at M MACs is compared against the best dense point at "
        "or below M MACs (M107 gate, unchanged); dense 224 is above the sparse "
        "MACs and is the in-distribution reference, not a comparison point")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    evidence = run_m109(args.config, args.output, progress=not args.quiet)
    if evidence.get("void"):
        print("\nM109 VOID — no figure is entitled.", flush=True)
        return
    gate = evidence["results"]["gate"]
    print("\ngate:", flush=True)
    for key in ("t1", "t2", "t3", "t4"):
        g = gate[key]
        print(f"  {key}: sparse {g['sparse_accuracy']:.4f} vs best dense "
              f"{g['best_dense_at_or_below_accuracy']:.4f} -> "
              f"KS1={g['kill_switch_1_dense_dominates']} "
              f"KS2={g['kill_switch_2_sparse_still_above']}", flush=True)
    ks3 = gate["kill_switch_3_sparse_at_capacity"]
    print(f"  KS3 sparse moved {abs(ks3['t4_accuracy'] - ks3['t1_accuracy']):.4f} "
          f"(fired={ks3['fired']})", flush=True)


if __name__ == "__main__":
    main()
