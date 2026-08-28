"""M108 — grow the dictionary instead of drawing it: does discriminative
growth transfer to DomainNet?

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v16.md`` section 5.1 and
``experiments/configs/v16/m108_dictionary.json``.

M107's sparse ladder is M103's arm (a): a random draw of ZCA-whitened patches.
M103 showed on CIFAR-10 that arm (c), additive discriminative growth, reaches
arm (a)'s 1024-atom accuracy at 512 atoms (C103.1). C103.1 explicitly does not
entitle transfer. M108 asks for it: does the 2.0x survive on DomainNet at 345
classes, inside M107's own protocol?

M107's rig, one variable changed. Same corpus, same 138,000/34,500 subsample
and digest, same patch pipeline, same ZCA whitener, same ridge head, same
penalty grid, same selection rule, same budgets {128, 256, 512, 1024, 2048,
3072}. The dictionary construction is the only difference:

* **(a) random patches** — M107's exact construction, re-measured not quoted.
  The registered null. Restriction 4 requires it to reproduce M107's recorded
  ``s_generalist`` accuracy within the registered tolerance or the run voids.
* **(c) discriminative growth** — ``select_discriminative()`` (M103 L227-280),
  the group-OMP greedy forward selection against a centred one-hot residual.
  The mechanism this program is about.
* **(e) ridge-leverage sampling** — Avron et al., ICML 2017, re-implemented
  inside this protocol (plan section 5.1 and §7 prior-art table). The published
  comparator. The config registers the exact formula, lambda and seed.

Device placement is registered in the config (``device`` block): whitening and
the ridge solve stay on CPU (numpy / float64, M107's exact arithmetic); the
cdist/pool encode and arm (c)'s greedy selection run on the GPU (torch device
"cuda" = ROCm/HIP in this interpreter, verified at startup). The GPU may not
change any arm's arithmetic beyond float32 accumulation-order noise, and the
arm (a) reproduction gate bounds exactly that.

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m108_dictionary
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Iterator

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
    _pool,
    select_discriminative,
)
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    _cache_root,
    _chunk_rows,
    _inference_macs,
    _load_domainnet,
    _score,
)
from experiments.tier4.eval_v15_m107_dense import (
    _class_subsample,
    _index_digest,
    _solve_and_score,
    _verify_pixel_identity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m108_dictionary.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m108_dictionary"
M107_EVIDENCE = REPO_ROOT / "logs" / "results" / "v15" / "m107_dense" / "evidence.json"


# --------------------------------------------------------------------------
# device precondition (plan section 4.6 restriction 2)
# --------------------------------------------------------------------------
def _verify_device(torch) -> dict[str, Any]:
    """HIP_VISIBLE_DEVICES=1 is a precondition. Abort rather than land on the
    integrated GPU. Also records torch.version.hip so the record cannot be
    mistaken for NVIDIA hardware: the device string torch calls "cuda" is the
    ROCm/HIP backend in this interpreter.

    Registered override (M176c-dvc, 18 Aug 2026): a device whose
    verification cell passed may run the sealed runners when the env var
    GEODE_VERIFIED_DEVICE_EVIDENCE points at the sealed dvc evidence file
    whose verdict is "bit-exact" and whose content is admissible. The
    override exists ONLY after that cell; nothing else bypasses the gate.
    """
    import json
    import os

    override_path = os.environ.get("GEODE_VERIFIED_DEVICE_EVIDENCE")
    if override_path:
        from pathlib import Path
        ev = json.loads(Path(override_path).read_text(encoding="utf-8"))
        if (ev.get("verdict") == "bit-exact"
                and ev.get("admissible_as_evidence") is True):
            properties = torch.cuda.get_device_properties(0)
            return {
                "torch_version": torch.__version__,
                "hip_version": getattr(torch.version, "hip", None),
                "device_name": properties.name,
                "gcnArchName": getattr(properties, "gcnArchName", None),
                "device_count": int(torch.cuda.device_count()),
                "_note": ("verified-device override (M176c-dvc bit-exact): "
                          + str(Path(override_path).name)),
            }

    if torch.cuda.device_count() != 1:
        raise SystemExit(
            "M108 instrument failure: torch.cuda.device_count() != 1. "
            "HIP_VISIBLE_DEVICES=1 is a registered precondition (plan section "
            "4.6 restriction 2); the integrated Radeon poisons HIP context "
            "initialisation for both devices."
        )
    properties = torch.cuda.get_device_properties(0)
    if getattr(properties, "gcnArchName", None) != "gfx1201":
        raise SystemExit(
            f"M108 instrument failure: visible device gcnArchName is "
            f"{getattr(properties, 'gcnArchName', None)}, not gfx1201 (the "
            "RX 9070 XT). Aborting before producing a figure."
        )
    return {
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "device_name": properties.name,
        "gcnArchName": getattr(properties, "gcnArchName", None),
        "device_count": int(torch.cuda.device_count()),
        "_note": "torch's device string 'cuda' is the ROCm/HIP backend in this "
                 "interpreter; the device is an AMD RX 9070 XT, not NVIDIA.",
    }


# --------------------------------------------------------------------------
# encode (whiten on CPU, cdist/pool on the GPU)
# --------------------------------------------------------------------------
def _encode_block_device(images: np.ndarray, table: torch.Tensor,
                         whitener: Whitener, pool_grid: int
                         ) -> np.ndarray:
    """M107's triangle encode, with the distance/pool on the GPU.

    The whitening is numpy on the CPU (M107's exact arithmetic, config
    device.whiten=cpu); only the cdist, activation and pooling run on the GPU.
    ``table`` is a float32 tensor already on the GPU, shared across calls.
    """
    white = torch.from_numpy(
        np.ascontiguousarray(whitener(images))
    ).to(torch.float32).to(table.device)
    with torch.no_grad():
        distances = torch.cdist(white, table)
        activation = torch.clamp(
            distances.mean(dim=1, keepdim=True) - distances, min=0.0
        )
        pooled = _pool(activation, len(images), whitener.grid, pool_grid)
    return pooled.to(torch.float32).cpu().numpy()


def _sparse_features_device(images: np.ndarray, table: torch.Tensor,
                            whitener: Whitener, pool_grid: int,
                            rows: np.ndarray, batch: int
                            ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    step = min(batch, _chunk_rows(table.shape[0], whitener.grid, len(rows)))
    for start in range(0, len(rows), step):
        take = rows[start:start + step]
        yield _encode_block_device(images[take], table, whitener, pool_grid), take


def _sparse_arm(corpus: dict[str, np.ndarray], dictionary: np.ndarray,
                whitener: Whitener, pool_grid: int, order: np.ndarray,
                penalties: list[float], classes: int, validation_rows: int,
                device: torch.device, batch: int = 4096) -> dict[str, Any]:
    """One sparse generalist on the GPU, mirroring M107's ``_sparse_arm``.

    ``validation_rows`` is non-zero for exactly one arm — arm (a) at the
    smallest budget, where the head constant is chosen (same rule as M107).
    """
    width = pool_grid * pool_grid * len(dictionary)
    accumulator = RidgeAccumulator(width, classes)
    table = torch.from_numpy(
        np.ascontiguousarray(dictionary)
    ).to(torch.float32).to(device)
    selection_rows = len(order) - validation_rows
    fitted = order[:selection_rows]
    for block, rows in _sparse_features_device(corpus["train_images"], table,
                                               whitener, pool_grid, fitted,
                                               batch):
        accumulator.add(block, corpus["train_labels"][rows])

    validation: dict[str, float] = {}
    if validation_rows > 0:
        standardise = accumulator.standardiser()
        selection = accumulator.solve_many(penalties)
        hits = {penalty: 0 for penalty in penalties}
        held = order[selection_rows:]
        for block, rows in _sparse_features_device(corpus["train_images"],
                                                   table, whitener, pool_grid,
                                                   held, batch):
            standardised = standardise(block)
            for penalty in penalties:
                hits[penalty] += int(
                    _score(selection[penalty], standardised,
                           corpus["train_labels"][rows]).sum()
                )
            accumulator.add(block, corpus["train_labels"][rows])
        validation = {
            str(penalty): hits[penalty] / validation_rows for penalty in penalties
        }

    test_order = np.arange(len(corpus["test_labels"]))

    def _test_blocks() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for block, rows in _sparse_features_device(corpus["test_images"],
                                                   table, whitener, pool_grid,
                                                   test_order, batch):
            yield block, corpus["test_labels"][rows], corpus["test_domains"][rows]

    result = _solve_and_score(accumulator, penalties, _test_blocks())
    result["validation_accuracy_by_penalty"] = validation
    result["selection_fit_rows"] = int(selection_rows)
    return result


# --------------------------------------------------------------------------
# dictionary construction
# --------------------------------------------------------------------------
def _random_order(candidates: np.ndarray, pool_size: int, seed: int
                  ) -> np.ndarray:
    """Arm (a): M107's exact construction (a seeded permutation of the pool)."""
    return np.random.default_rng([seed, 100]).permutation(pool_size)


def _select_discriminative_gpu(features_np: np.ndarray, labels: np.ndarray,
                               atom_count: int, budget: int, pool_grid: int,
                               device: torch.device) -> tuple[np.ndarray, int]:
    """Arm (c) on the GPU: the group-OMP selection of M103 L227-280.

    The standardisation and the centred one-hot residual are computed in numpy
    exactly as the reference does, and only the greedy matmul/lstsq loop runs
    on the GPU. ``torch.linalg.lstsq(..., driver='gelsd')`` is the same LAPACK
    family as ``np.linalg.lstsq``, which is what the order-parity check against
    the numpy reference (config selection.parity_*) tests.
    """
    rows = len(features_np)
    classes = int(labels.max()) + 1
    work_np = features_np.astype(np.float32, copy=True)
    work_np -= work_np.mean(axis=0, keepdims=True)
    work_np /= work_np.std(axis=0, keepdims=True) + 1e-8
    residual_np = np.zeros((rows, classes), dtype=np.float32)
    residual_np[np.arange(rows), labels] = 1.0
    residual_np -= residual_np.mean(axis=0, keepdims=True)

    work = torch.from_numpy(np.ascontiguousarray(work_np)).to(device)
    residual = torch.from_numpy(np.ascontiguousarray(residual_np)).to(device)
    groups = np.stack(
        [np.arange(atom_count) + q * atom_count for q in range(pool_grid ** 2)],
        axis=1,
    )
    groups_t = torch.from_numpy(np.ascontiguousarray(groups)).to(device)

    order: list[int] = []
    available = torch.ones(atom_count, dtype=torch.bool, device=device)
    macs = 0
    block_width = pool_grid ** 2
    for _ in range(budget):
        correlation = work.t() @ residual
        macs += rows * work.shape[1] * classes
        corr_sq = correlation * correlation
        score = torch.zeros(atom_count, device=device)
        for q in range(block_width):
            score += corr_sq.index_select(0, groups_t[:, q]).sum(dim=1)
        score.masked_fill_(~available, -float("inf"))
        best = int(torch.argmax(score))
        order.append(best)
        available[best] = False
        block = work[:, groups_t[best]]
        # Only the QR-based 'gels' driver is supported on CUDA/ROCm (gelsd is
        # CPU-only). The order-parity check against the numpy reference
        # (gelsd) decides whether this is faithful enough; on failure the
        # runner falls back to the numpy function for the full selection.
        coefficients = torch.linalg.lstsq(block, residual, driver="gels").solution
        residual = residual - block @ coefficients
        macs += 2 * rows * block_width * classes
    return np.asarray(order, dtype=np.int64), macs


def _ridge_leverage_order(candidates: np.ndarray, penalty: float,
                          top_budget: int, seed: int
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Arm (e): ridge-leverage sampling (Avron et al., ICML 2017), registered
    formula in the config.

    ``ell_i = z_i^T (Z^T Z + lambda I)^-1 z_i`` over the shared whitened
    candidate pool Z (8192 x 108), in float64 on the CPU, then one weighted
    sample WITHOUT replacement of ``top_budget`` atoms, seed registered. The
    ladder is the prefixes of that draw, so it nests as arms (a) and (c) do.
    """
    z = candidates.astype(np.float64)
    gram = z.T @ z + penalty * np.eye(z.shape[1], dtype=np.float64)
    inverse = np.linalg.inv(gram)
    leverage = np.sum(z * (z @ inverse), axis=1)
    if np.any(leverage <= 0.0):
        raise RuntimeError(
            "M108 instrument failure: ridge-leverage scores are not all "
            "positive; the weighted sample is undefined."
        )
    rng = np.random.default_rng(seed)
    order = rng.choice(len(z), size=top_budget, replace=False,
                       p=leverage / leverage.sum())
    return np.asarray(order, dtype=np.int64), leverage


# --------------------------------------------------------------------------
# arm (a) reproduction gate (restriction 4)
# --------------------------------------------------------------------------
def _load_m107_generalist() -> dict[str, Any]:
    evidence = json.loads(M107_EVIDENCE.read_text(encoding="utf-8"))
    penalty = str(evidence["head"]["chosen_penalty"])
    arms = {}
    for name, payload in evidence["arms"].items():
        if name.startswith("s_generalist") and not payload.get("void"):
            arms[int(payload["atoms"])] = {
                "accuracy": float(payload["accuracy_by_penalty"][penalty]),
                "chosen_penalty": penalty,
            }
    if not arms:
        raise RuntimeError("M108 instrument failure: no M107 generalist arms found")
    return {"arms": arms, "chosen_penalty": float(penalty)}


def _check_arm_a_reproduction(measured: dict[int, float], reference: dict[int, dict],
                              tolerance: float) -> dict[str, Any]:
    """Restriction 4: arm (a) must reproduce M107 within the registered
    tolerance at every budget, or the run is void (not negative)."""
    violations = {}
    for budget, accuracy in measured.items():
        if budget not in reference:
            continue
        delta = accuracy - reference[budget]["accuracy"]
        if abs(delta) > tolerance:
            violations[int(budget)] = {
                "measured": accuracy, "m107": reference[budget]["accuracy"],
                "delta": delta,
            }
    overlap = sorted(set(measured) & set(reference))
    # A reproduction check that no budget exercises is a check that passed
    # without being run. On the sealed run the budgets are {128..3072} and the
    # reference covers them all, so this guard only fires on a misconfiguration
    # or a smoke config whose budgets sit below the reference ladder.
    exercised = bool(overlap)
    return {
        "reproduced": bool(exercised and not violations),
        "exercised": exercised,
        "overlapping_budgets": overlap,
        "tolerance": tolerance,
        "violations": violations,
        "max_abs_delta": float(max(
            (abs(accuracy - reference[b]["accuracy"])
             for b, accuracy in measured.items() if b in reference),
            default=0.0,
        )),
        "_note": (
            "restriction 4, arm_a_reproduction config: if any budget exceeds "
            "the registered tolerance, M108 is VOID and the instrument is at "
            "fault, per plan section 3.3. A check with no overlapping budgets "
            "is not exercised and cannot pass."
        ),
    }


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------
def _best_dense_at_or_below(dense: list[list[float]], macs: float) -> float | None:
    below = [a for m, a in dense if m <= macs]
    return max(below) if below else None


def _crossing_budget(curve: dict[int, float], dense: list[list[float]],
                     macs: dict[int, int]) -> int | None:
    """Cheapest sparse budget whose accuracy >= the best dense arm at or below
    its MACs (M107's comparison rule). Registered definition in the config."""
    crossing = None
    for budget in sorted(curve):
        reference = _best_dense_at_or_below(dense, macs[budget])
        if reference is not None and curve[budget] >= reference:
            crossing = budget
            break
    return crossing


def _build_gate(config: dict[str, Any], arms: dict[str, Any], penalty: float,
                m107_evidence: dict[str, Any]) -> dict[str, Any]:
    budgets = [int(b) for b in config["sparse"]["budgets"]]

    def curve(prefix: str) -> dict[int, float]:
        out = {}
        for budget in budgets:
            payload = arms[f"{prefix}_{budget}"]
            if not payload.get("void"):
                out[budget] = float(payload["accuracy_by_penalty"][str(penalty)])
        return out

    a_curve = curve("a_random")
    c_curve = curve("c_discriminative")
    e_curve = curve("e_leverage")

    dense = sorted(
        [float(p["macs"]["total"]),
         float(p["accuracy_by_penalty"][str(penalty)])]
        for name, p in m107_evidence["arms"].items()
        if p["family"] == "dense" and not p.get("void")
    )
    macs = {
        budget: int(arms[f"a_random_{budget}"]["macs"]["total"])
        for budget in budgets
    }

    # kill switch 1: does arm (c) beat arm (a) at matched atoms?
    c_beats_a = [b for b in budgets if b in a_curve and b in c_curve
                 and c_curve[b] > a_curve[b]]
    c_beats_a_everywhere = all(
        c_curve[b] > a_curve[b] for b in budgets if b in a_curve and b in c_curve
    )

    # kill switch 2: does arm (e) beat arm (c) at every budget?
    e_beats_c = [b for b in budgets if b in c_curve and b in e_curve
                 and e_curve[b] > c_curve[b]]
    e_beats_c_everywhere = all(
        e_curve[b] > c_curve[b] for b in budgets if b in c_curve and b in e_curve
    )

    # kill switch 3: arm (c) beats arm (a) but crossings do not move earlier.
    a_crossing = _crossing_budget(a_curve, dense, macs)
    c_crossing = _crossing_budget(c_curve, dense, macs)
    crossing_moved_earlier = (
        a_crossing is not None and c_crossing is not None
        and macs[c_crossing] < macs[a_crossing]
    )
    ks3_fires = bool(c_beats_a and not crossing_moved_earlier)

    return {
        "penalty": float(penalty),
        "curves": {
            "a_random": [[macs[b], a_curve[b]] for b in sorted(a_curve)],
            "c_discriminative": [[macs[b], c_curve[b]] for b in sorted(c_curve)],
            "e_leverage": [[macs[b], e_curve[b]] for b in sorted(e_curve)],
            "dense_m107": dense,
        },
        "kill_switch_1_c_does_not_beat_a": {
            "fired": bool(not c_beats_a),
            "budgets_c_beats_a": c_beats_a,
            "c_beats_a_at_every_budget": c_beats_a_everywhere,
            "consequence": (
                "C103.1 does not transfer to DomainNet; the additive-growth "
                "thesis has failed its first test outside CIFAR-10. Headline "
                "under plan section 3.6."
            ),
        },
        "kill_switch_2_e_beats_c_everywhere": {
            "fired": bool(e_beats_c_everywhere),
            "budgets_e_beats_c": e_beats_c,
            "consequence": (
                "every subsequent v16 sparse arm is constructed by "
                "ridge-leverage sampling, not by select_discriminative."
            ),
        },
        "kill_switch_3_crossings_do_not_move": {
            "fired": ks3_fires,
            "a_crossing_budget": a_crossing,
            "c_crossing_budget": c_crossing,
            "crossing_moved_earlier": crossing_moved_earlier,
            "consequence": (
                "dictionary learning improves accuracy without improving the "
                "efficiency operand; the two are reported as separate findings."
            ),
        },
        "_comparison_rule": (
            "a sparse point at M MACs is compared against the best dense point "
            "at or below M MACs (M107 gate, unchanged). The dense curve is "
            "M107's sealed evidence at this penalty; M108 does not re-run dense."
        ),
    }


def _gap_closing_fraction(arms: dict[str, Any], budgets: list[int],
                          penalty: float, dense_reference: float
                          ) -> dict[str, Any]:
    """M108's second operand (plan section 5.1, registered 5 August 2026).

    Registered reading of ``(arm - a) / (dense_224 - a)`` (config
    ``gap_closing``): at each readable budget B, the fraction of the gap from
    the random null at the same budget to the in-distribution dense reference
    (M107 sealed ``d1_small_224`` at the chosen penalty) that arm (c) and arm
    (e) close. A non-positive denominator or a voided arm reports as
    undefined, never as zero.
    """

    def acc(prefix: str, budget: int) -> float | None:
        payload = arms.get(f"{prefix}_{budget}")
        if payload is None or payload.get("void"):
            return None
        return float(payload["accuracy_by_penalty"][str(penalty)])

    def fraction(prefix: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for budget in budgets:
            arm_acc = acc(prefix, budget)
            null_acc = acc("a_random", budget)
            if arm_acc is None or null_acc is None:
                out[str(budget)] = "undefined (arm not readable)"
                continue
            denominator = dense_reference - null_acc
            if denominator <= 0.0:
                out[str(budget)] = "undefined (denominator not positive)"
                continue
            out[str(budget)] = {
                "fraction": round((arm_acc - null_acc) / denominator, 4),
                "arm_accuracy": arm_acc,
                "null_accuracy": null_acc,
            }
        return out

    return {
        "dense_reference": dense_reference,
        "dense_reference_note": (
            "M107 sealed d1_small_224 at the chosen penalty; the "
            "in-distribution frozen dense pole"
        ),
        "formula": "(arm(B) - a(B)) / (dense_224 - a(B)) at each readable "
                    "budget B; registered reading in the config gap_closing",
        "c_discriminative": fraction("c_discriminative"),
        "e_leverage": fraction("e_leverage"),
        "_measurement_note": (
            "a measurement, never a novelty claim: both mechanisms are "
            "published (plan section 7.1); the fraction quantifies how much "
            "of the representation gap selection closes at zero gradient "
            "training"
        ),
    }


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run_m108(config_path: Path, output_dir: Path, progress: bool = True
             ) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    # A config that names itself inadmissible must never write where the sealed
    # run's figures are read from (M107 amendment 6's lesson, carried).
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible as evidence (_smoke_note), and would write to the "
            f"SEALED output directory {DEFAULT_OUTPUT}. Pass --output with a "
            "separate directory."
        )

    torch.set_num_threads(config["numerics"]["torch_threads"])
    configure_external_cache_environment()
    device_report = _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    corpus_config = config["corpus"]
    representation = config["representation"]
    pool_grid = representation["pool_grid"]
    patch = representation["patch"]
    stride = representation["stride"]
    size = corpus_config["image_size"]
    penalties = [float(p) for p in config["head"]["regularisation_grid"]]
    budgets = list(config["sparse"]["budgets"])
    floor = config["head"]["fit_samples_per_fitted_dimension_floor"]

    print("loading DomainNet at 32x32", flush=True)
    raw = _load_domainnet(size)
    train_index = _class_subsample(
        raw["train_labels"], corpus_config["train_rows_per_class"],
        corpus_config["subsample_seed"],
    )
    test_index = _class_subsample(
        raw["test_labels"], corpus_config["test_rows_per_class"],
        corpus_config["subsample_seed"],
    )
    corpus = {
        "train_images": raw["train_images"][train_index],
        "train_labels": raw["train_labels"][train_index],
        "train_domains": raw["train_domains"][train_index],
        "test_images": raw["test_images"][test_index],
        "test_labels": raw["test_labels"][test_index],
        "test_domains": raw["test_domains"][test_index],
    }
    del raw
    digest = _index_digest({"train": train_index, "test": test_index})
    expected = corpus_config.get("expected_subsample_sha256")
    if expected and expected != digest:
        raise RuntimeError(
            f"M108 subsample digest {digest} does not match the pinned "
            f"{expected}; the shared subsample changed."
        )
    classes = int(corpus["train_labels"].max()) + 1
    print(f"  train {len(train_index)}  test {len(test_index)}  "
          f"classes {classes}  digest {digest[:16]}", flush=True)

    identity_checks = [
        _verify_pixel_identity(
            split, index, corpus[f"{split}_images"], size,
            corpus_config["pixel_identity_rows"],
        )
        for split, index in (("train", train_index), ("test", test_index))
    ]

    order = np.random.default_rng(corpus_config["shuffle_seed"]).permutation(
        len(train_index)
    )
    validation_rows = int(round(
        len(order) * config["head"]["selection_validation_fraction"]
    ))

    # ---- shared whitener and candidate pool (M107's exact construction) ----
    print("fitting the shared whitener on TRAIN patches only", flush=True)
    rng = np.random.default_rng(representation["zca_fit_seed"])
    sample_images = corpus["train_images"][
        rng.choice(len(corpus["train_images"]),
                   min(len(corpus["train_images"]), 20_000), replace=False)
    ]
    patches = _extract_patches(sample_images, patch, stride)
    grid = (size - patch) // stride + 1
    take = min(representation["zca_fit_patches"], len(patches))
    patch_pool = _contrast_normalise(
        patches[rng.choice(len(patches), take, replace=False)],
        representation["contrast_epsilon"],
    )
    mean, whiten = _fit_zca(patch_pool, representation["zca_epsilon"])
    whitener = Whitener(patch, stride, representation["contrast_epsilon"],
                        mean, whiten, grid)
    dimension = patch * patch * 3
    del patches, sample_images

    seed = config["sparse"]["dictionary_seed"]
    pool_size = config["sparse"]["candidate_pool_size"]
    seed_rng = np.random.default_rng(seed)
    candidates = ((patch_pool[
        seed_rng.choice(len(patch_pool), pool_size, replace=False)
    ] - mean) @ whiten).astype(np.float32)

    # ---- the three atom orders -------------------------------------------
    orders: dict[str, dict[str, Any]] = {}

    a_order = _random_order(candidates, pool_size, seed)
    orders["a_random"] = {"order": a_order,
                          "note": "M107's exact construction: prefix of a "
                                  "seeded permutation of the shared pool."}

    print("arm (c): discriminative growth selection", flush=True)
    sel_rng = np.random.default_rng(config["selection"]["selection_seed"])
    selection_rows = config["selection"]["selection_rows"]
    selection_index = sel_rng.choice(
        len(corpus["train_images"]), size=min(selection_rows, len(corpus["train_images"])),
        replace=False,
    )
    selection_table = torch.from_numpy(
        np.ascontiguousarray(candidates)
    ).to(torch.float32).to(device)
    step = min(4096, _chunk_rows(pool_size, grid, len(selection_index)))
    selection_features = np.empty(
        (len(selection_index), pool_grid * pool_grid * pool_size),
        dtype=np.float32,
    )
    selection_started = time.time()
    for start in range(0, len(selection_index), step):
        take = selection_index[start:start + step]
        selection_features[start:start + step] = _encode_block_device(
            corpus["train_images"][take], selection_table, whitener, pool_grid
        )
    selection_encode_seconds = time.time() - selection_started
    del selection_table
    print(f"    encoded {len(selection_index)} rows against the {pool_size}-atom "
          f"pool in {selection_encode_seconds:.0f}s", flush=True)

    # Order-parity check: the GPU port must reproduce the numpy reference's
    # atom order on a subset before the full selection is trusted.
    parity_rows = config["selection"]["parity_subset_rows"]
    parity_budget = config["selection"]["parity_budget"]
    parity_features = selection_features[:parity_rows]
    parity_labels = corpus["train_labels"][selection_index[:parity_rows]]
    reference_order, _ = select_discriminative(
        parity_features, parity_labels, pool_size, parity_budget, pool_grid
    )
    gpu_order, _ = _select_discriminative_gpu(
        parity_features, parity_labels, pool_size, parity_budget, pool_grid, device
    )
    parity_ok = bool(np.array_equal(reference_order, gpu_order))
    if not parity_ok:
        first_mismatch = int(np.flatnonzero(reference_order != gpu_order)[0])
        print(f"    ORDER-PARITY CHECK FAILED at atom {first_mismatch}; "
              "falling back to the numpy reference for the full selection",
              flush=True)
    else:
        print(f"    order-parity check passed on {parity_rows} rows / "
              f"{parity_budget} atoms (identical order)", flush=True)

    selection_started = time.time()
    if parity_ok:
        selection_order, selection_macs = _select_discriminative_gpu(
            selection_features,
            corpus["train_labels"][selection_index],
            pool_size, max(budgets), pool_grid, device,
        )
        selection_backend = "cuda"
    else:
        selection_order, selection_macs = select_discriminative(
            selection_features,
            corpus["train_labels"][selection_index],
            pool_size, max(budgets), pool_grid,
        )
        selection_backend = "cpu"
    selection_seconds = time.time() - selection_started
    del selection_features
    selection_total_macs = (
        _selection_encode_macs(len(selection_index), pool_size, grid,
                               dimension, classes)
        + int(selection_macs)
    )
    orders["c_discriminative"] = {
        "order": selection_order,
        "backend": selection_backend,
        "selection_rows": int(len(selection_index)),
        "selection_encode_seconds": round(selection_encode_seconds, 2),
        "selection_seconds": round(selection_seconds, 2),
        "selection_macs": int(selection_total_macs),
        "parity_checked": bool(parity_ok),
        "note": "group-OMP greedy forward selection (M103 select_discriminative).",
    }

    print("arm (e): ridge-leverage sampling", flush=True)
    e_order, leverage = _ridge_leverage_order(
        candidates, config["ridge_leverage"]["penalty"],
        config["ridge_leverage"]["top_budget"],
        config["ridge_leverage"]["seed"],
    )
    orders["e_leverage"] = {
        "order": e_order,
        "penalty": config["ridge_leverage"]["penalty"],
        "seed": config["ridge_leverage"]["seed"],
        "leverage_min": float(leverage.min()),
        "leverage_max": float(leverage.max()),
        "leverage_mean": float(leverage.mean()),
        "top_atoms_by_leverage": np.argsort(-leverage)[:32].tolist(),
        "note": "ridge-leverage sampling, Avron et al. ICML 2017; formula in "
                "the config.",
    }

    # ---- arms -------------------------------------------------------------
    arms: dict[str, Any] = {}
    partial = output_dir / "partial_arms.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _record(name: str, payload: dict[str, Any]) -> None:
        ratio = payload.get("rows_per_fitted_dimension")
        if ratio is not None and ratio < floor:
            payload["void"] = True
            payload["void_reason"] = (
                f"section 5.3: {ratio:.2f} fit rows per fitted dimension is "
                f"below the floor of {floor}. The arm is void, not negative."
            )
        arms[name] = payload
        write_canonical_json(partial, {"arms": arms})

    chosen_penalty: float | None = None
    for arm_name, state in orders.items():
        prefix = arm_name
        for budget in budgets:
            label = f"{prefix}_{budget}"
            print(f"  {label}", flush=True)
            dictionary = candidates[state["order"][:budget]]
            selecting = validation_rows if (arm_name == "a_random"
                                            and budget == budgets[0]) else 0
            payload = _sparse_arm(corpus, dictionary, whitener, pool_grid,
                                  order, penalties, classes, selecting, device)
            payload["family"] = "sparse"
            payload["arm"] = prefix
            payload["atoms"] = budget
            payload["macs"] = _inference_macs(budget, grid, dimension,
                                              pool_grid, classes)
            if selecting:
                best = max(payload["validation_accuracy_by_penalty"].items(),
                           key=lambda kv: (kv[1], -float(kv[0])))
                chosen_penalty = float(best[0])
                payload["_note"] = (
                    "the head constant is chosen HERE, on arm (a) at the "
                    "smallest budget, on held-out train rows, and applied "
                    "unchanged to every other arm and budget (same rule as "
                    "M107 section 7.14 design item 5)"
                )
                print(f"    head constant chosen on arm (a) at {budget}: "
                      f"{chosen_penalty}", flush=True)
            _record(label, payload)
    if chosen_penalty is None:
        raise RuntimeError("M108 instrument failure: no head constant was chosen")
    if arms[f"a_random_{budgets[0]}"].get("void"):
        raise RuntimeError(
            "M108 instrument failure: the arm the head constant is chosen on "
            "is below the section 5.3 floor; every other arm inherits that "
            "constant and the run is inadmissible."
        )

    # ---- arm (a) reproduction gate (restriction 4) ------------------------
    m107_ref = _load_m107_generalist()
    measured_a = {
        budget: float(arms[f"a_random_{budget}"]["accuracy_by_penalty"]
                      [str(chosen_penalty)])
        for budget in budgets if not arms[f"a_random_{budget}"].get("void")
    }
    reproduction = _check_arm_a_reproduction(
        measured_a, m107_ref["arms"], config["arm_a_reproduction"]["tolerance_accuracy"]
    )
    penalty_match = bool(
        abs(chosen_penalty - m107_ref["chosen_penalty"]) < 1e-9
    )
    if not reproduction["reproduced"] or not penalty_match:
        reproduction["penalty_match"] = penalty_match
        reproduction["_verdict"] = (
            "arm (a) did not reproduce M107; restriction 4 makes M108 VOID "
            "and the instrument is at fault, not the arms. Per plan section "
            "3.3 a failing instrument produces no figure."
        )
        write_canonical_json(output_dir / "evidence.json", {
            "milestone": "M108", "admissible_as_evidence": False,
            "void": True, "void_reason": "arm (a) reproduction gate failed",
            "reproduction": reproduction, "chosen_penalty": chosen_penalty,
            "m107_chosen_penalty": m107_ref["chosen_penalty"],
        })
        print("  ARM (a) REPRODUCTION GATE FAILED — M108 is VOID", flush=True)
        return {"admissible_as_evidence": False, "void": True,
                "reproduction": reproduction}

    # ---- evidence ---------------------------------------------------------
    evidence = {
        "milestone": "M108",
        "question": (
            "does discriminative growth (arm (c)) beat a random draw (arm (a)) "
            "at matched atoms on DomainNet, and does ridge-leverage sampling "
            "(arm (e), Avron et al. 2017) beat arm (c)?"
        ),
        "registered_in": "analysis/RESEARCH_IMPLEMENTATION_PLAN_v16.md section 5.1",
        "admissible_as_evidence": not inadmissible,
        "config_file": Path(config_path).name,
        "config": config,
        "device": device_report,
        "corpus": {
            "train_rows": int(len(train_index)),
            "test_rows": int(len(test_index)),
            "classes": classes,
            "subsample_sha256": digest,
            "pixel_identity": identity_checks,
            "train_rows_per_domain": np.bincount(
                corpus["train_domains"], minlength=6).tolist(),
            "test_rows_per_domain": np.bincount(
                corpus["test_domains"], minlength=6).tolist(),
        },
        "orders": {
            name: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in state.items()}
            for name, state in orders.items()
        },
        "head": {
            "chosen_penalty": chosen_penalty,
            "chosen_on": "a_random_128",
            "fit_samples_per_fitted_dimension_floor": floor,
            "_rule": (
                "chosen once on arm (a) at the smallest budget, applied "
                "unchanged to every arm; the full grid is recorded per arm as "
                "sensitivity only"
            ),
        },
        "arm_a_reproduction": reproduction,
        "gap_closing_fraction": _gap_closing_fraction(
            arms, budgets, chosen_penalty,
            float(config.get("gap_closing", {})
                  .get("dense_reference_accuracy", 0.5375)),
        ),
        "m107_generalist_reference": m107_ref["arms"],
        "arms": arms,
        "asymmetries": {
            "training_data": (
                "DINOv2 is pre-trained on LVD-142M; the sparse dictionaries are "
                "drawn from this corpus's own train patches and nothing else. "
                "Affects M107's dense figures, not the within-M108 comparison."
            ),
            "selection": (
                "arm (c) pays a selection cost arms (a) and (e) do not; it is "
                "reported, not netted against inference (M103 compute_ledger)"
            ),
        },
    }
    evidence["gate"] = _build_gate(
        config, arms, chosen_penalty,
        json.loads(M107_EVIDENCE.read_text(encoding="utf-8")),
    )
    evidence["payload_sha256"] = payload_hash(evidence)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    return evidence


def _selection_encode_macs(rows: int, pool_size: int, grid: int,
                           dimension: int, classes: int) -> int:
    """Encode cost of the selection rows against the full pool (M103 style)."""
    return int(rows * grid * grid * (dimension * dimension
                                     + pool_size * dimension))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    evidence = run_m108(args.config, args.output, progress=not args.quiet)
    if evidence.get("void"):
        print("\nM108 VOID — no figure is entitled.", flush=True)
        return
    gate = evidence["gate"]
    print("\ngate:", flush=True)
    for key, value in gate.items():
        if isinstance(value, dict) and "fired" in value:
            verdict = "FIRED" if value["fired"] else "not fired"
            print(f"  {key}: {verdict}", flush=True)
    if gate["kill_switch_1_c_does_not_beat_a"]["fired"]:
        print("  HEADLINE: C103.1 does not transfer to DomainNet.", flush=True)
    if gate["kill_switch_2_e_beats_c_everywhere"]["fired"]:
        print("  HEADLINE: ridge-leverage dominates; all future sparse arms "
              "use it.", flush=True)
    print(f"  arm (a) reproduction: "
          f"{'PASSED' if evidence['arm_a_reproduction']['reproduced'] else 'FAILED'}",
          flush=True)
    print(f"  chosen penalty {gate['penalty']}", flush=True)


if __name__ == "__main__":
    main()
