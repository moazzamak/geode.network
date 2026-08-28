"""A5 Phase 1 — routed per-domain patch-dictionary specialists (Arm P) vs a
global dense model (Arm D).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v17.md`` section 3 (A5,
phase 1) and ``experiments/configs/v16/a5_routed.json``.

Question. M109 killed a single global 3,072-atom dictionary over 345 classes x
6 domains (it could not specialise per style domain, and the crossing closed
once both sides trained). Does allowing the sparse side to specialise PER DOMAIN
— each input routed to one floor-capped domain dictionary with a closed-form
ridge head (per A2, the ridge head is correct for sparse codes) — lift the
frozen sparse ceiling (0.2148) toward competitive accuracy, at a total per-input
cost of router + one specialist?

Arms.
- Arm D: RE-MEASURED global DINOv2-small ridge (never quoted from M107), at
  resolutions {28, 42}; the t1 reproduction gate requires it to reproduce
  M107/M109 t1 within 0.002 or the run is VOID.
- Arm P: per-domain patch-dictionary specialists. Global M108 whitener; per
  domain d a dictionary of atoms_d = floor(D_d/44.92) random whitened patches
  restricted to that domain's train rows (floor-capped: >= 11.23 rows per fitted
  dimension); a closed-form ridge head per domain.
- Routers: oracle (true test domain labels) is the PRIMARY control and carries
  the verdicts; a registered fingerprint router (per-image mean/std, nearest
  per-domain centroid) is secondary and its routing accuracy is reported as a
  gate.

Kill switches (registered).
- KS1 (ceiling lift): if oracle-routed Arm P does not beat the global frozen
  dictionary (0.2148) overall by >= 0.01, routing does not lift the sparse
  ceiling and the program consolidates.
- KS2 (vs dense): if oracle-routed Arm P beats Arm D at-or-above its cost on any
  domain, routed sparse specialists beat a trained transformer per-domain at
  matched cost (the only measured regime where the sparse family wins on
  accuracy).

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_a5_routed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
)
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    _score,
)
from experiments.tier4.eval_v15_m107_dense import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _transformer_macs,
    _verify_pixel_identity,
)
from experiments.tier4.eval_v16_m108_dictionary import (
    _encode_block_device,
    _verify_device,
)
from experiments.tier4.eval_v16_m109_trunk import (
    DenseModel,
    _build_whitener_dictionary,
    _dense_pixels,
    _load_corpus,
    _parity_guard,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "a5_routed.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "a5_routed"
M107_EVIDENCE = REPO_ROOT / "logs" / "results" / "v15" / "m107_dense" / "evidence.json"
M109_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m109_trunk" / "evidence.json"

T1_TOLERANCE = 0.002
FLOOR = 11.23          # section 3.5 rows per fitted dimension
PATTERNS = 27 * 27     # 729 patches per 32x32 image (patch 6, stride 1)
PATCH_DIM = 108
ROUTER_MACS = 2 * 32 * 32 * 3   # per-image mean + std over pixels


def _build_whitener(config: dict[str, Any],
                    corpus: dict[str, np.ndarray]) -> Whitener:
    """M108's exact global whitener (same seed, sample and ZCA), no dictionary.

    ``_build_whitener_dictionary`` also builds a global dictionary at
    ``sparse.atoms``, which A5 does not need (A5 builds per-domain dictionaries),
    so this replicates only the whitener half of M108's construction.
    """
    size = config["corpus"]["image_size"]
    rep = {"zca_fit_patches": 400000, "zca_fit_seed": 11,
           "contrast_epsilon": 10.0, "zca_epsilon": 0.1}
    patch, stride = 6, 1
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
    return Whitener(patch, stride, rep["contrast_epsilon"], mean, whiten, grid)


def _domain_candidates(corpus: dict[str, np.ndarray], domain: int,
                       whitener, n_cap: int = 20000, seed: int = 11
                       ) -> np.ndarray:
    """Random whitened-patch candidates restricted to one domain's train rows."""
    rows = np.where(corpus["train_domains"] == domain)[0]
    rng = np.random.default_rng([seed, domain])
    pool = np.empty((n_cap, PATCH_DIM), dtype=np.float32)
    filled = 0
    for start in range(0, len(rows), 256):
        imgs = corpus["train_images"][rows[start:start + 256]]
        pn = _contrast_normalise(_extract_patches(imgs, 6, 1), 10.0)
        room = n_cap - filled
        if len(pn) <= room:
            pool[filled:filled + len(pn)] = pn
            filled += len(pn)
        else:
            pool[filled:] = pn[rng.choice(len(pn), room, replace=False)]
            filled = n_cap
            break
    pool = pool[:filled]
    return ((pool - whitener.mean) @ whitener.whiten).astype(np.float32)


def _run_arm_p_domain(corpus, domain, atoms, whitener, pool_grid, train_fit,
                      classes, device):
    rows_d = np.array([i for i in train_fit
                       if corpus["train_domains"][i] == domain])
    cand = _domain_candidates(corpus, domain, whitener)
    order = np.random.default_rng([11, 100]).permutation(len(cand))[:atoms]
    dictionary = cand[order]
    table = torch.from_numpy(np.ascontiguousarray(dictionary)).to(torch.float32)
    table = table.to(device)

    acc = RidgeAccumulator(atoms * pool_grid * pool_grid, classes)
    for start in range(0, len(rows_d), 64):
        take = rows_d[start:start + 64]
        block = _encode_block_device(corpus["train_images"][take], table,
                                     whitener, pool_grid)
        acc.add(block, corpus["train_labels"][take])
    solutions = acc.solve_many([1.0])
    standardise = acc.standardiser()

    test_rows_d = np.where(corpus["test_domains"] == domain)[0]
    correct = 0
    for start in range(0, len(test_rows_d), 64):
        take = test_rows_d[start:start + 64]
        block = _encode_block_device(corpus["test_images"][take], table,
                                     whitener, pool_grid)
        correct += int(_score(solutions[1.0], standardise(block),
                              corpus["test_labels"][take]).sum())
    macs = (PATTERNS * atoms * PATCH_DIM        # encode (cdist)
            + atoms * pool_grid * pool_grid * classes)  # head
    return {
        "domain": int(domain),
        "atoms": int(atoms),
        "test_rows": int(len(test_rows_d)),
        "correct": int(correct),
        "accuracy": correct / len(test_rows_d),
        "macs_per_image": int(macs),
        "train_rows": int(len(rows_d)),
        "rows_per_fitted_dim": len(rows_d) / (atoms * pool_grid * pool_grid),
    }


def _fingerprint_router(corpus, train_fit, test_seq) -> dict[str, Any]:
    tr = corpus["train_images"][train_fit]
    train_desc = np.stack([tr.reshape(len(tr), -1).mean(1),
                           tr.reshape(len(tr), -1).std(1)], axis=1)
    train_mask = corpus["train_domains"][train_fit]
    centres = np.zeros((6, 2))
    for d in range(6):
        idx = np.where(train_mask == d)[0]
        centres[d] = train_desc[idx].mean(0) if len(idx) else 0.0
    te = corpus["test_images"][test_seq]
    test_desc = np.stack([te.reshape(len(te), -1).mean(1),
                          te.reshape(len(te), -1).std(1)], axis=1)
    scale = test_desc.std(0) + 1e-9
    routed = ((test_desc / scale)[:, None, :] - (centres / scale)[None, :, :]) ** 2
    pred = routed.sum(2).argmin(1)
    true = corpus["test_domains"][test_seq]
    return {"routing_accuracy": float((pred == true).mean()),
            "routed_counts": [int((pred == d).sum()) for d in range(6)],
            "true_counts": [int((true == d).sum()) for d in range(6)],
            "descriptor": "per-image mean/std of raw pixels",
            "distance": "normalised L2 to per-domain centroids"}


def run_a5(config_path: Path, output_dir: Path, progress: bool = True
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
    for split, idx in (("train", train_index), ("test", test_index)):
        _verify_pixel_identity(split, idx, corpus[f"{split}_images"], size,
                               config["corpus"]["pixel_identity_rows"])

    order = np.random.default_rng(config["corpus"]["shuffle_seed"]).permutation(
        len(train_index))
    # A5 has no SGD and selects no hyperparameter: every head is a closed-form
    # ridge (per A2), so a validation split has nothing to do, and carving 5%
    # off would (a) waste data and (b) make the recorded rows-per-fitted-
    # dimension fall below the registered floor (11.23) even when the floor
    # gate passes. M107's dense arm and per-domain mixture arm both fit on ALL
    # train rows (the dense arm's docstring says so explicitly); A5 does the
    # same. Repair registered in v17, A5 section (first sealed run self-voided:
    # the floor gate was computed on D_d total rows while heads fit on 95%).
    train_fit = order

    atoms_per_domain = [int(a) for a in config["sparse"]["atoms_per_domain"]]
    smoke_skip = bool(config.get("_smoke_skip_gates", False))
    m109 = json.loads(M109_EVIDENCE.read_text(encoding="utf-8"))

    results: dict[str, Any] = {"arm_d": {}, "arm_p": {}}
    gates: dict[str, Any] = {}

    # ---- Arm D: re-measured global DINOv2 ridge, per-domain eval ----------
    print("Arm D: re-measured global dense ridge", flush=True)
    from experiments.tier4.eval_v15_m107_dense import _dinov2_geometry
    geometry = _dinov2_geometry("small")
    resolutions = [int(r) for r in config["dense"]["resolutions"]]
    dense_pixels = _dense_pixels(config, train_index, test_index)
    dense_model = DenseModel("small", classes, device)
    t1_repro = {}
    for r in resolutions:
        mem = np.load(dense_pixels["train"][r], mmap_mode="r")
        mem_test = np.load(dense_pixels["test"][r], mmap_mode="r")
        acc = RidgeAccumulator(dense_model.width, classes)
        for start in range(0, len(train_fit), 256):
            take = train_fit[start:start + 256]
            block = np.asarray(mem[take], dtype=np.float32) / 255.0
            block = (block - IMAGENET_MEAN) / IMAGENET_STD
            block = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
            with torch.no_grad():
                feat = dense_model.features(
                    torch.from_numpy(block).to(device)).cpu().numpy()
            acc.add(feat, corpus["train_labels"][take])
        solutions = acc.solve_many([1.0])
        standardise = acc.standardiser()
        per_domain = {}
        total_correct = 0
        for d in range(6):
            rows_d = np.where(corpus["test_domains"] == d)[0]
            correct = 0
            for start in range(0, len(rows_d), 256):
                take = rows_d[start:start + 256]
                block = np.asarray(mem_test[take], dtype=np.float32) / 255.0
                block = (block - IMAGENET_MEAN) / IMAGENET_STD
                block = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
                with torch.no_grad():
                    feat = dense_model.features(
                        torch.from_numpy(block).to(device)).cpu().numpy()
                correct += int(_score(solutions[1.0], standardise(feat),
                                      corpus["test_labels"][take]).sum())
            per_domain[str(d)] = {"correct": int(correct),
                                  "test_rows": int(len(rows_d)),
                                  "accuracy": correct / len(rows_d)}
            total_correct += correct
        target = float(m107_arm(r))
        measured = total_correct / len(test_index)
        t1_repro[f"dense_r{r}"] = {"m107": target, "measured": measured,
                                   "delta": measured - target}
        results["arm_d"][str(r)] = {
            "accuracy": measured, "per_domain": per_domain,
            "macs_per_image": int(_transformer_macs(geometry, r, classes)["total"]),
            "head": "closed_form_ridge_1.0"}
        print(f"  Arm D r{r}: {measured:.4f} (t1 repro delta "
              f"{measured - target:+.5f})", flush=True)
    del dense_model
    torch.cuda.empty_cache()

    if not smoke_skip:
        max_t1 = max(abs(v["delta"]) for v in t1_repro.values())
        if max_t1 > T1_TOLERANCE:
            gates["_verdict"] = ("A5 VOID: Arm D does not reproduce M107 t1 "
                                 "within 0.002; instrument at fault.")
            write_canonical_json(output_dir / "evidence.json", {
                "milestone": "A5-p1", "admissible_as_evidence": False,
                "void": True, "void_reason": "Arm D t1 reproduction failed",
                "t1_reproduction": t1_repro, "parity_guard": parity})
            return {"admissible_as_evidence": False, "void": True,
                    "t1_reproduction": t1_repro}
        gates["t1_max_delta"] = max_t1
        print(f"  t1 reproduction max delta {max_t1:.5f} (<= {T1_TOLERANCE})",
              flush=True)

    # ---- Arm P: per-domain specialists + routers --------------------------
    print("building global whitener (M108)", flush=True)
    whitener = _build_whitener(config, corpus)
    pool_grid = int(config["sparse"]["pool_grid"])
    floor_check = {}
    for d in range(6):
        D_d = int((corpus["train_domains"] == d).sum())
        floor_check[str(d)] = {
            "train_rows": D_d,
            "atoms": atoms_per_domain[d],
            "floor_cap": int(D_d / (4 * FLOOR)),
            "ok": atoms_per_domain[d] <= D_d / (4 * FLOOR),
        }

    print("Arm P: per-domain specialists (oracle + fingerprint router)",
          flush=True)
    arm_p = {}
    total_correct = 0
    for d in range(6):
        row = _run_arm_p_domain(corpus, d, atoms_per_domain[d], whitener,
                                pool_grid, train_fit, classes, device)
        arm_p[str(d)] = row
        total_correct += row["correct"]
        print(f"  Arm P d{d}: {row['accuracy']:.4f} "
              f"({row['atoms']} atoms)", flush=True)
        torch.cuda.empty_cache()
    oracle_accuracy = total_correct / len(test_index)
    router = _fingerprint_router(corpus, train_fit, test_seq)

    if not smoke_skip and not all(v["ok"] for v in floor_check.values()):
        gates["_verdict"] = "A5 VOID: per-domain floor violated."
        write_canonical_json(output_dir / "evidence.json", {
            "milestone": "A5-p1", "admissible_as_evidence": False,
            "void": True, "void_reason": "per-domain floor violated",
            "floor_check": floor_check})
        return {"admissible_as_evidence": False, "void": True,
                "floor_check": floor_check}
    gates["floor_check"] = floor_check

    # ---- kill switches -----------------------------------------------------
    global_frozen = 0.2148  # M109 t1 sparse (sealed)
    ks1 = {
        "registered_prediction": "routing lifts the frozen sparse ceiling "
                                 "modestly but does not reach dense",
        "global_frozen_sparse": global_frozen,
        "oracle_routed_arm_p": oracle_accuracy,
        "margin": 0.01,
        "fired": oracle_accuracy - global_frozen < 0.01,
        "consequence": "if fired, routing does not lift the sparse ceiling and "
                       "the program consolidates",
    }
    dense_28 = results["arm_d"]["28"]
    ks2 = {
        "fired": any(arm_p[str(d)]["accuracy"] > dense_28["per_domain"][str(d)]["accuracy"]
                     for d in range(6)),
        "per_domain_dense_r28": {d: dense_28["per_domain"][str(d)]["accuracy"]
                                 for d in range(6)},
        "per_domain_arm_p": {d: arm_p[str(d)]["accuracy"] for d in range(6)},
        "consequence": "if fired, routed sparse specialists beat a trained "
                       "transformer per-domain at matched cost",
    }

    gates["oracle_router"] = {"accuracy": oracle_accuracy,
                              "note": "oracle carries the verdicts (KS1/KS2)"}
    gates["fingerprint_router"] = router
    gates["_smoke_skip"] = smoke_skip

    evidence = {
        "milestone": "A5-p1",
        "admissible_as_evidence": not inadmissible,
        "registered_in": config["registered_in"],
        "question": ("does per-domain routing lift the sparse ceiling toward "
                     "competitive accuracy at router + one specialist cost?"),
        "config_file": Path(config_path).name,
        "config": config,
        "device": device_report,
        "parity_guard": parity,
        "t1_reproduction": t1_repro,
        "results": results,
        "arm_p": arm_p,
        "gates": gates,
        "kill_switch_1_ceiling_lift": ks1,
        "kill_switch_2_vs_dense": ks2,
        "cost": {
            "router_macs_per_image": ROUTER_MACS,
            "arm_p_macs_per_image_by_domain": {d: arm_p[str(d)]["macs_per_image"]
                                               for d in range(6)},
            "arm_d_macs_per_image": {str(r): results["arm_d"][str(r)]["macs_per_image"]
                                     for r in resolutions},
        },
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    evidence["payload_sha256"] = payload_hash(evidence)
    print(f"\nA5 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(f"  oracle-routed Arm P overall: {oracle_accuracy:.4f}", flush=True)
    print(f"  KS1 fired: {ks1['fired']}  KS2 fired: {ks2['fired']}", flush=True)
    return evidence


def m107_arm(resolution: int) -> float:
    m107 = json.loads(M107_EVIDENCE.read_text(encoding="utf-8"))
    target = {28: "d4a_small_28", 42: "d4b_small_42"}[resolution]
    return float(m107["arms"][target]["accuracy_by_penalty"]["1.0"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_a5(args.config, args.output)


if __name__ == "__main__":
    main()
