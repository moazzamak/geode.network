"""M144 — pruned-dense baseline: channel-magnitude pruning of DINOv2-small
at r56, ridge readout.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v22.md`` (section 9
M144, Remaining-milestones recipe; 14 Aug 2026). M144 is a MEASURING STICK,
not a gated cell: it builds the missing "train-big-then-prune" side of the
additive-vs-pruning comparison at the sealed dense data level (138k train
rows, the M107 ladder level) and reports accuracy-vs-MACs.

Construction. The cached torch DINOv2-small weights (the M109 path), the
M107 feature definition (CLS + mean of the patch tokens, 768 columns), at
resolution 56 (the sealed r56 arm). The pixel inputs are M107's own:
``_materialise_original`` decodes the selected rows from the ORIGINAL
parquet stream and resizes original->56 with PIL bilinear into the
digest-tagged cache ``domainnet_m107/<tag>/`` (the sealed r56 arm's exact
inputs; the 32x32 decoded cache is NOT upsampled to 56, because a two-step
resize is a different pixel pipeline and the t2 anchor voids on it).
Structured channel pruning by magnitude, applied to the attention and MLP
matrices only:

- attention: heads are scored by the L2 norm of their concatenated q/k/v
  output rows; the top round(keep x n_heads) heads are kept; a dropped
  head's q/k/v rows, the attention-output projection's matching input
  columns, and their biases are ZEROED (zeroing both sides makes the
  removal exact: the unit contributes nothing in either direction, so the
  forward pass of the pruned model equals the kept subnetwork bitwise).
- MLP: fc1 output rows scored by L2; the top round(keep x mlp_hidden)
  kept; dropped rows of fc1, the matching columns of fc2, and the biases
  zeroed.

keep in {1.0, 0.5, 0.25}. Achieved nonzero-parameter fractions are
reported, never assumed.

Anchors:
- t1 parity: the in-run ONNX-vs-torch parity guard at r56 (bound 1e-4, the
  sealed M109 guard) pins the torch backbone against the ONNX exports M107
  measured.
- t2: the UNPRUNED torch re-encode at r56 must reproduce the sealed
  r56 read 0.245014492753623 within 0.002.

Output (no win/loss gate — a measuring stick): accuracy and effective MACs
per arm (the analytic ledger scaled by the kept channel fractions), read
against the sealed dense ladder (r42 0.1972, r56 0.2450, r70 0.3118) and
the sparse frontier (138k level: pool 0.2064, MS+sqrt 0.2239, SPM+sqrt
0.2274; full data 0.2786).

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m144_pruned_dense
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
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v15_m107_dense import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _index_digest,
    _materialise_original,
    _transformer_macs,
)
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v16_m109_trunk import _load_corpus, _parity_guard
from experiments.tier4.bench_v16_parity import feature

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v16"
                  / "m144_pruned_dense.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m144_pruned_dense"

CLASSES = 345
RESOLUTION = 56
T2_TOLERANCE = 0.002
T2_REFERENCE = 0.245014492753623


def _load_torch_dinov2_small(device: torch.device):
    from transformers import Dinov2Model

    weights = data_cache_root() / "torch" / "dinov2-small"
    if not weights.exists():
        raise SystemExit(f"no torch weights at {weights}")
    model = Dinov2Model.from_pretrained(str(weights), dtype=torch.float32)
    model.eval().to(device)
    return model


def _prune(model, keep: float) -> dict[str, Any]:
    """Zero out dropped attention heads and MLP units (both sides + bias).

    Returns the achieved per-component kept fractions and total nonzero
    parameter count for the pruned matrices."""
    n_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size) // n_heads
    width = int(model.config.hidden_size)
    n_layers = len(model.encoder.layer)
    stats: dict[str, Any] = {
        "n_heads": n_heads, "head_dim": head_dim, "width": width,
        "mlp_hidden": int(model.config.mlp_ratio * width),
        "n_layers": n_layers,
    }
    dropped_heads = 0
    dropped_mlp = 0
    for layer in model.encoder.layer:
        attn = layer.attention.attention
        head_norms = torch.stack(
            [torch.stack([
                proj.weight[h * head_dim:(h + 1) * head_dim].float().pow(2).sum()
                for proj in (attn.query, attn.key, attn.value)]).sum()
             for h in range(n_heads)]
        )
        keep_heads = max(1, int(round(keep * n_heads)))
        order = torch.argsort(head_norms, descending=True)
        drop = order[keep_heads:]
        dropped_heads += len(drop)
        with torch.no_grad():
            for h in drop:
                lo, hi = h * head_dim, (h + 1) * head_dim
                for proj in (attn.query, attn.key, attn.value):
                    proj.weight[lo:hi] = 0.0
                    proj.bias[lo:hi] = 0.0
                out = layer.attention.output.dense
                out.weight[:, lo:hi] = 0.0
        mlp = layer.mlp
        fc1 = mlp.fc1.weight  # (hidden, width)
        unit_norms = fc1.float().pow(2).sum(dim=1)
        keep_units = max(1, int(round(keep * stats["mlp_hidden"])))
        drop_units = torch.argsort(unit_norms, descending=True)[keep_units:]
        dropped_mlp += len(drop_units)
        with torch.no_grad():
            mlp.fc1.weight[drop_units] = 0.0
            mlp.fc1.bias[drop_units] = 0.0
            mlp.fc2.weight[:, drop_units] = 0.0
    stats["dropped_heads"] = int(dropped_heads)
    stats["dropped_mlp_units"] = int(dropped_mlp)
    stats["kept_head_fraction"] = 1.0 - dropped_heads / (n_heads * n_layers)
    stats["kept_mlp_fraction"] = 1.0 - dropped_mlp / (stats["mlp_hidden"] * n_layers)

    nonzero = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += int((p.detach() != 0).sum())
    stats["nonzero_params"] = int(nonzero)
    stats["total_params"] = int(total)
    stats["nonzero_fraction"] = nonzero / total
    return stats


def _encode_features(model, device, images: np.ndarray, rows: np.ndarray,
                     batch: int) -> np.ndarray:
    """CLS + mean-patch-token features at resolution 56, streamed.

    ``images`` is the M107-materialised original-resolution r56 memmap
    (``domainnet_m107/<tag>/<split>_56.npy``); rows are positional into it."""
    out = np.empty((len(rows), 2 * model.config.hidden_size),
                   dtype=np.float32)
    for start in range(0, len(rows), batch):
        take = rows[start:start + batch]
        block = images[take].astype(np.float32) / 255.0  # (n, 56, 56, 3)
        block = (block - IMAGENET_MEAN) / IMAGENET_STD
        block = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
        with torch.no_grad():
            tokens = model(
                pixel_values=torch.from_numpy(block).to(device)
            ).last_hidden_state.cpu().numpy()
        out[start:start + len(take)] = feature(tokens)
    return out


def _fit_and_score(feat_train: np.ndarray, labels: np.ndarray,
                   feat_test: np.ndarray, test_labels: np.ndarray,
                   block: int) -> dict[str, Any]:
    acc = RidgeAccumulator(feat_train.shape[1], CLASSES)
    for start in range(0, len(feat_train), block):
        stop = min(start + block, len(feat_train))
        acc.add(feat_train[start:stop], labels[start:stop])
    weights = acc.solve(1.0)
    std = acc.standardiser()
    hits = 0
    for start in range(0, len(feat_test), block):
        stop = min(start + block, len(feat_test))
        scores = std(feat_test[start:stop]) @ weights[:-1] + weights[-1]
        hits += int((np.argmax(scores, axis=1)
                     == test_labels[start:stop]).sum())
    return {"accuracy": hits / len(test_labels),
            "fit_rows": int(acc.rows),
            "features": int(feat_train.shape[1])}


def _effective_macs(geometry: dict[str, int], stats: dict[str, Any]) -> int:
    base = _transformer_macs(geometry, RESOLUTION, CLASSES)
    head_keep = stats["kept_head_fraction"]
    mlp_keep = stats["kept_mlp_fraction"]
    total = (base["patch_embedding"]
             + base["projections"] * head_keep
             + base["attention"] * head_keep * head_keep
             + base["mlp"] * mlp_keep
             + base["head"])
    return int(total)


def run_m144(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    batch = int(config["numerics"]["batch"])
    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    torch.manual_seed(int(config["numerics"]["seed"]))
    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("loading corpus (138k subsample + raw)", flush=True)
    corpus, train_index, test_index = _load_corpus(config)
    from experiments.tier4.eval_v15_m107_dense import _dinov2_geometry
    geometry = _dinov2_geometry("small")
    print(f"geometry: {geometry}", flush=True)

    # pixels: the sealed r56 arm's exact inputs (original-resolution parquet
    # decode, PIL-bilinear resize to 56, digest-tagged) — NOT the 32x32 cache
    # upsampled (a two-step resize is a different pixel pipeline).
    digest = _index_digest({"train": train_index, "test": test_index})
    expected = config["corpus"].get("expected_subsample_sha256")
    if expected and expected != digest:
        raise SystemExit(f"M144 subsample digest {digest} != registered")
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

    model = _load_torch_dinov2_small(device)
    base_macs = _transformer_macs(geometry, RESOLUTION, CLASSES)["total"]
    print(f"sealed r56 ledger total: {base_macs}", flush=True)

    # ---- t1 parity guard (ONNX vs torch at r56) ---------------------------
    print("t1 parity guard", flush=True)
    parity = _parity_guard(torch, config, device)
    print(f"  parity worst relative difference "
          f"{parity['worst_relative_difference']:.3e} "
          f"(bound {parity['bound']})", flush=True)
    if parity["worst_relative_difference"] > parity["bound"] and not smoke_skip:
        raise SystemExit("t1 parity guard failed")

    # rows: the sealed dense ladder level is the 138k subsample
    n_train = len(train_index) if not smoke else int(config["_smoke_train_rows"])
    n_test = len(test_index) if not smoke else int(config["_smoke_test_rows"])
    train_rows = np.arange(n_train)
    test_rows = np.arange(n_test)

    evidence: dict[str, Any] = {
        "milestone": "M144",
        "cell": "pruned-dense baseline (DINOv2-small, r56, ridge readout)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "geometry": geometry,
        "sealed_r56_macs": base_macs,
        "parity_guard": parity,
        "pixels": {
            "source": "M107 _materialise_original: original-resolution "
                      "parquet decode, PIL-bilinear resize to 56, "
                      "digest-tagged cache",
            "provenance": "materialised under the CPU env (.venv, the "
                          "M107 interpreter family, PIL 12.3.0) on "
                          "2026-08-15; the runner only READS the "
                          "memmaps",
            "tag": tag,
            "train": str(train_pix[RESOLUTION]),
            "test": str(test_pix[RESOLUTION]),
        },
        "frontier_context": config["frontier_context"],
    }

    arms: dict[str, Any] = {}
    for keep in config["arms"]["keep_fractions"]:
        keep = float(keep)
        print(f"arm keep={keep}: preparing", flush=True)
        if keep < 1.0:
            model = _load_torch_dinov2_small(device)  # fresh unpruned copy
            stats = _prune(model, keep)
        else:
            stats = {"nonzero_fraction": 1.0, "kept_head_fraction": 1.0,
                     "kept_mlp_fraction": 1.0}
        print(f"  prune stats: {stats}", flush=True)
        print("  encoding train", flush=True)
        feat_train = _encode_features(model, device, pix_train,
                                      train_rows, batch)
        print("  encoding test", flush=True)
        feat_test = _encode_features(model, device, pix_test,
                                     test_rows, batch)
        result = _fit_and_score(feat_train, corpus["train_labels"][:n_train],
                                feat_test, corpus["test_labels"][:n_test],
                                block)
        result["effective_macs"] = _effective_macs(geometry, stats)
        result["prune_stats"] = stats
        arms[str(keep)] = result
        print(f"  keep={keep}: {result['accuracy']:.4f} "
              f"@ {result['effective_macs']} MACs", flush=True)
        del feat_train, feat_test
        torch.cuda.empty_cache()

    t2_delta = arms["1.0"]["accuracy"] - T2_REFERENCE
    evidence["t2_unpruned"] = {"accuracy": arms["1.0"]["accuracy"],
                               "reference": T2_REFERENCE,
                               "delta": t2_delta}
    print(f"t2 unpruned {arms['1.0']['accuracy']:.4f} vs sealed r56 "
          f"{T2_REFERENCE} (delta {t2_delta:+.6f})", flush=True)
    if not smoke_skip and abs(t2_delta) > T2_TOLERANCE:
        evidence["void"] = True
        evidence["void_reason"] = "t2 unpruned r56 reproduction failed"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        return evidence

    evidence["arms"] = arms
    evidence["note"] = (
        "M144 is a measuring stick, not a gated cell: the pruned-dense curve "
        "is reported against the sealed dense ladder and the sparse frontier "
        "for the additive-vs-pruning comparison."
    )
    evidence["runtime_seconds"] = round(time.time() - started, 2)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM144 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m144(args.config, args.output)


if __name__ == "__main__":
    main()
