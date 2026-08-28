"""M300 - hash-seeded random Fourier features against the quickdraw
wall.

Registered in ``analysis/SCIENCE_LAYER_PLAN_2026-08-28.md`` (M300
REGISTRATION AMENDMENT incl. the head-type correction, 28 Aug 2026,
before the build). H26-3: is the quickdraw wall (~0.63 across four
frozen backbones) a linearity ceiling rather than a feature ceiling?

The map: phi(z) = sqrt(2/D) [cos(omega_i^T z + b_i)] with omega ~
N(0, sigma^-2 I), b ~ U[0, 2pi), drawn from a generator seeded by
the registered seed (the artifact-hash stand-in). (D, sigma) are
selected by train-side 5-fold CV on the quickdraw train rows ONLY;
the sealed quickdraw test is evaluated ONCE at the selected pair.

Head: the M233/M236 TRAINED-PROBE recipe (nn.Linear + AdamW, 30
epochs, lr 1e-3, wd 1e-4, batch 1024, seed 11) - all four wall
references are trained probes, so the same head isolates the
nonlinearity contribution.

Arms:
- clip_linear: the exact M236 reproduction on RAW CLIP-L features
  (the g2 instrument-identity anchor 0.626655234828999).
- concat_linear: probe on the per-block L2-normalised CLIP+dino
  concat (the linear reference under the registered input form).
- clip_rff: probe on [clip_norm, phi(clip_norm)] - the
  nonlinearity contribution on one backbone.
- rff_concat: probe on [concat_norm, phi(concat_norm)] - the H26-3
  arm.

Gates: g1 premise; g2 the clip_linear arm reproduces 0.626655234828999
at 1e-9; g3 the RFF map is deterministic under the registered seed;
g4 the CV selection touches train rows only; g5 accuracies valid.
"""
from __future__ import annotations

import argparse
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
from experiments.tier4.eval_v15_m104_experts import (
    _load_domainnet,
)
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m140_data_extension import _extension_indices
from experiments.tier4.eval_v16_m141_data_full import _rest_extension_indices
from experiments.tier4.eval_v25_m233_trained_probes import (
    _probe_score,
    _train_probe,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m300_rff_quickdraw.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m300_rff_quickdraw")

FULL_TRAIN_ROWS = 409832
CLASSES = 345
QUICKDRAW = 3
CLIP_L_LINEAR_ANCHOR = 0.626655234828999   # sealed M236 quickdraw probe
ANCHOR_TOL = 1e-9


# ---------------------------------------------------------------------------
# the hash-seeded random feature map
# ---------------------------------------------------------------------------
# chunk target for the projection: ~64 MiB of float32 per temporary.
# The M300b lesson: an unchunked projection at D=65536 needs arg +
# cos-temp + product + astype copy ~= 4 x 23.6 GiB simultaneously and
# dies on a 63 GB host; chunked in-place it needs one ~64 MiB temp.
_RFF_CHUNK_ELEMENTS = 1 << 24


def rff_project(block: np.ndarray, omega: np.ndarray, phase: np.ndarray,
                out: np.ndarray | None = None,
                chunk_rows: int | None = None,
                ) -> np.ndarray:
    """phi(z) = sqrt(2/D) cos(z @ omega + b) for one block of rows.
    fp32 in, fp32 out (the cached feature dtype); the map is
    deterministic given (omega, phase). The scale is a float32
    scalar: a float64 scalar would promote the whole output to
    float64 (the M300b memory lesson - 47 GiB at D=65536).

    Chunked and in-place: each row chunk is projected with matmul
    into a ~64 MiB temporary, then phase-added, cos'd, and scaled
    in place, so peak memory is one chunk regardless of D. When
    ``out`` is given it must be a float32 array of shape
    (len(block), omega.shape[1]) and the projection is written
    directly into it (no full-size temporary at all). The per-row
    arithmetic is identical to the unchunked form, so results are
    bitwise equal for any chunk size."""
    z = np.asarray(block, dtype=np.float32)
    n, D = z.shape[0], omega.shape[1]
    if out is None:
        result = np.empty((n, D), dtype=np.float32)
    else:
        result = out
        if result.shape != (n, D) or result.dtype != np.float32:
            raise ValueError(
                "out must be float32 with shape "
                f"{(n, D)}, got {result.shape} {result.dtype}")
    if chunk_rows is None:
        chunk_rows = max(1, _RFF_CHUNK_ELEMENTS // max(1, D))
    scale = np.float32(np.sqrt(2.0 / D))
    for s in range(0, n, chunk_rows):
        e = min(s + chunk_rows, n)
        arg = z[s:e] @ omega          # (chunk, D) float32 temp
        arg += phase[None, :]
        np.cos(arg, out=arg)
        arg *= scale
        result[s:e] = arg
    return result


def build_design(features: np.ndarray, omega: np.ndarray,
                 phase: np.ndarray) -> np.ndarray:
    """The registered arm input form [features, phi(features)] as a
    single preallocated float32 matrix: the features block is copied
    into the left columns and the RFF projection is written directly
    into the right columns (``rff_project(..., out=...)``). This
    avoids the np.concatenate full-size copy that, at D=65536, adds
    another ~24 GiB on top of the projection. Bitwise equal to
    np.concatenate([features, rff_project(features, omega, phase)],
    axis=1)."""
    features = np.asarray(features, dtype=np.float32)
    n, d = features.shape
    D = omega.shape[1]
    design = np.empty((n, d + D), dtype=np.float32)
    design[:, :d] = features
    rff_project(features, omega, phase, out=design[:, d:])
    return design


def rff_params(dim: int, n_features: int, sigma: float,
               seed: int) -> tuple[np.ndarray, np.ndarray]:
    """omega ~ N(0, sigma^-2 I), b ~ U[0, 2pi) from the registered
    seed. Deterministic: the same (dim, D, sigma, seed) always
    produces the same (omega, phase)."""
    rng = np.random.default_rng(seed)
    omega = (rng.standard_normal((dim, n_features))
             / float(sigma)).astype(np.float32)
    phase = (rng.random(n_features) * 2.0 * np.pi).astype(np.float32)
    return omega, phase


def _l2_normalise_rows(block: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    return block / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
# train-side 5-fold CV over the registered (D, sigma) grid
# ---------------------------------------------------------------------------
def cv_select(train_quickdraw: np.ndarray, labels_q: np.ndarray,
              grid_dims: list[int], grid_sigma: list[float],
              folds: int, cv_seed: int, rff_seed: int,
              train_probe_cfg: dict[str, Any], device
              ) -> dict[str, Any]:
    """Select (D, sigma) by 5-fold CV on the quickdraw TRAIN rows
    only. The fold split is seeded; the RFF params are seeded per
    (dim, sigma) so every fold scores the same map. The probe recipe
    is the registered M233/M236 one. Returns the selected pair and
    the full grid table."""
    n = len(train_quickdraw)
    rng = np.random.default_rng(cv_seed)
    fold_of = rng.permutation(n) % folds
    dim = train_quickdraw.shape[1]
    t = train_probe_cfg
    epochs, lr, wd, batch, seed = (int(t["epochs"]), float(t["lr"]),
                                   float(t["weight_decay"]),
                                   int(t["batch"]), int(t["seed"]))
    table: dict[str, dict[str, Any]] = {}
    best: tuple[float, int, float] | None = None
    for sigma in grid_sigma:
        for n_feat in grid_dims:
            omega, phase = rff_params(dim, n_feat, sigma, rff_seed)
            accs: list[float] = []
            for fold in range(folds):
                tr_rows = np.flatnonzero(fold_of != fold)
                va_rows = np.flatnonzero(fold_of == fold)
                design_tr = build_design(train_quickdraw[tr_rows],
                                         omega, phase)
                design_va = build_design(train_quickdraw[va_rows],
                                         omega, phase)
                w, b = _train_probe(design_tr, labels_q[tr_rows],
                                    epochs, lr, wd, batch, seed, device)
                accs.append(_probe_score(w, b, design_va,
                                         labels_q[va_rows]))
                del design_tr, design_va
            mean_acc = float(np.mean(accs))
            key = f"D{n_feat}_sigma{sigma}"
            table[key] = {"mean_cv_accuracy": mean_acc,
                          "fold_accuracies": [float(a) for a in accs]}
            if best is None or mean_acc > best[0]:
                best = (mean_acc, n_feat, sigma)
            print(f"  cv {key}: {mean_acc:.4f}", flush=True)
    assert best is not None
    return {"selected_D": best[1], "selected_sigma": best[2],
            "selected_cv_accuracy": best[0], "grid": table}


def run_m300(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    root = data_cache_root()

    corpus, train_index, test_index = _load_corpus(config)
    raw = _load_domainnet(32)
    ext600_indices, _ = _extension_indices(raw["train_labels"],
                                           train_index, 600, CLASSES)
    rest_indices = _rest_extension_indices(raw["train_labels"],
                                           train_index, CLASSES,
                                           per_class_take=200)
    perm = np.concatenate([train_index, ext600_indices, rest_indices])
    labels = np.load(root / config["artifacts"]["labels_file"])["labels"]
    g1_schedule = (len(perm) == FULL_TRAIN_ROWS
                   and np.array_equal(raw["train_labels"][perm], labels))
    train_domains = raw["train_domains"][perm]
    test_labels = corpus["test_labels"]
    test_domains = corpus["test_domains"]
    del raw
    import gc
    gc.collect()

    f = config["features"]
    clip_tr = np.asarray(np.load(REPO_ROOT / f["clip_train"],
                                 mmap_mode="r")[perm])
    clip_te = np.asarray(np.load(REPO_ROOT / f["clip_test"],
                                 mmap_mode="r")[test_index])
    dino_tr = np.asarray(np.load(REPO_ROOT / f["dino_train"],
                                 mmap_mode="r")[perm])
    dino_te = np.asarray(np.load(REPO_ROOT / f["dino_test"],
                                 mmap_mode="r")[test_index])

    q_rows = np.flatnonzero(train_domains == QUICKDRAW)
    q_trows = np.flatnonzero(test_domains == QUICKDRAW)
    q_labels = labels[q_rows]
    q_test_labels = test_labels[q_trows]

    import torch
    device = torch.device("cuda")
    t = config["training"]
    epochs, lr, wd, batch, seed = (int(t["epochs"]), float(t["lr"]),
                                   float(t["weight_decay"]),
                                   int(t["batch"]), int(t["seed"]))

    # ---- arm (a): the exact M236 reproduction on RAW CLIP features --
    # (the g2 instrument-identity anchor: same recipe, same seed, raw
    # features - the M236 quickdraw probe)
    w_a, b_a = _train_probe(clip_tr[q_rows], q_labels, epochs, lr, wd,
                            batch, seed, device)
    acc_a = _probe_score(w_a, b_a, clip_te[q_trows], q_test_labels)
    g2 = bool(abs(acc_a - CLIP_L_LINEAR_ANCHOR) <= ANCHOR_TOL)
    print(f"clip_linear (raw, M236 reproduction): {acc_a:.6f} "
          f"(anchor {CLIP_L_LINEAR_ANCHOR}, g2={g2})", flush=True)

    # per-block L2 normalisation (the registered input form for the
    # remaining arms)
    clip_q = _l2_normalise_rows(clip_tr[q_rows])
    dino_q = _l2_normalise_rows(dino_tr[q_rows])
    clip_qte = _l2_normalise_rows(clip_te[q_trows])
    dino_qte = _l2_normalise_rows(dino_te[q_trows])
    concat_q = np.concatenate([clip_q, dino_q], axis=1)
    concat_qte = np.concatenate([clip_qte, dino_qte], axis=1)
    del clip_tr, clip_te, dino_tr, dino_te
    gc.collect()

    g1 = bool(g1_schedule and clip_q.shape[1] == 768
              and dino_q.shape[1] == 768
              and len(q_rows) > 0 and len(q_trows) > 0)

    # ---- arm (b): concat linear (the registered input form) ---------
    w_b, b_b = _train_probe(concat_q, q_labels, epochs, lr, wd,
                            batch, seed, device)
    acc_b = _probe_score(w_b, b_b, concat_qte, q_test_labels)
    print(f"concat_linear: {acc_b:.6f}", flush=True)

    # ---- g3: RFF determinism under the registered seed -------------
    r = config["rff"]
    rff_seed = int(r["seed"])
    om1, ph1 = rff_params(clip_q.shape[1], 512, 1.0, rff_seed)
    om2, ph2 = rff_params(clip_q.shape[1], 512, 1.0, rff_seed)
    probe1 = rff_project(clip_q[:64], om1, ph1)
    probe2 = rff_project(clip_q[:64], om2, ph2)
    g3 = bool(np.array_equal(probe1, probe2))
    print(f"g3 rff determinism: {g3}", flush=True)

    # ---- the CV selection (train rows only, g4) ---------------------
    print("cv selection over the registered grid:", flush=True)
    cv = cv_select(clip_q, q_labels, [int(d) for d in r["grid_dims"]],
                   [float(s) for s in r["grid_sigma"]],
                   int(r["cv_folds"]), int(r["cv_seed"]), rff_seed,
                   config["training"], device)
    sel_D, sel_sigma = cv["selected_D"], cv["selected_sigma"]
    print(f"selected D={sel_D} sigma={sel_sigma} "
          f"(cv {cv['selected_cv_accuracy']:.4f})", flush=True)
    g4 = True   # the selection function touches train rows only, by
    # construction; the fold split never sees test rows. Recorded.

    # ---- arm (c): RFF on CLIP-L -------------------------------------
    om_c, ph_c = rff_params(clip_q.shape[1], sel_D, sel_sigma, rff_seed)
    design_c_tr = build_design(clip_q, om_c, ph_c)
    design_c_te = build_design(clip_qte, om_c, ph_c)
    w_c, b_c = _train_probe(design_c_tr, q_labels, epochs, lr, wd,
                            batch, seed, device)
    acc_c = _probe_score(w_c, b_c, design_c_te, q_test_labels)
    print(f"clip_rff: {acc_c:.6f}", flush=True)
    del design_c_tr, design_c_te
    gc.collect()

    # ---- arm (d): RFF on the concat (the H26-3 arm) ------------------
    om_d, ph_d = rff_params(concat_q.shape[1], sel_D, sel_sigma, rff_seed)
    design_d_tr = build_design(concat_q, om_d, ph_d)
    design_d_te = build_design(concat_qte, om_d, ph_d)
    w_d, b_d = _train_probe(design_d_tr, q_labels, epochs, lr, wd,
                            batch, seed, device)
    acc_d = _probe_score(w_d, b_d, design_d_te, q_test_labels)
    print(f"rff_concat: {acc_d:.6f}", flush=True)

    g5 = all(0.0 <= a <= 1.0 for a in (acc_a, acc_b, acc_c, acc_d))

    # ---- the registered reading --------------------------------------
    wall = float(config["wall_references"]["mlp_concat"])
    margin = float(config["gate_h26_3"]["margin"])
    bar = float(config["gate_h26_3"]["bar"])
    stroke = float(config["wall_references"]["stroke_arm_m238"])
    h26_3_pass = bool(acc_d >= bar)
    beats_stroke = bool(acc_d >= stroke)
    if h26_3_pass:
        reading = ("H26-3 PASS: the RFF map clears the wall by the "
                   "registered margin - the quickdraw wall is a "
                   "linearity ceiling, not a feature ceiling")
    else:
        reading = ("H26-3 FAIL: the RFF map does not clear the "
                   "registered margin - the wall is recorded as a "
                   "feature ceiling at this scale; the kill-criterion "
                   "input of plan section 5")
    if beats_stroke:
        reading += "; the RFF arm also beats the M238 stroke arm"

    gates = {
        "g1_premise": {"ok": g1, "schedule_alignment": bool(g1_schedule),
                       "quickdraw_train_rows": int(len(q_rows)),
                       "quickdraw_test_rows": int(len(q_trows))},
        "g2_clip_linear_anchor": {"ok": g2, "measured": acc_a,
                                  "sealed": CLIP_L_LINEAR_ANCHOR,
                                  "tolerance": ANCHOR_TOL},
        "g3_rff_determinism": {"ok": g3},
        "g4_cv_train_only": {"ok": g4,
                             "note": "the fold split and the grid "
                                     "scoring touch quickdraw train "
                                     "rows only"},
        "g5_accuracies_valid": {"ok": bool(g5)},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M300",
        "cell": ("hash-seeded RFF map against the quickdraw wall "
                 "(CLIP-L + dino-b, the M233/M236 trained-probe head, "
                 "CV-selected (D, sigma))"),
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "arms": {
            "clip_linear": acc_a,
            "concat_linear": acc_b,
            "clip_rff": acc_c,
            "rff_concat": acc_d,
        },
        "cv_selection": cv,
        "wall_references": config["wall_references"],
        "h26_3": {"pass": h26_3_pass, "bar": bar,
                  "margin": margin, "wall": wall,
                  "beats_stroke_arm": beats_stroke},
        "reading": reading,
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": bool(gates_ok),
                      "arms": evidence["arms"],
                      "h26_3": evidence["h26_3"],
                      "reading": reading}, indent=1), flush=True)
    print(f"M300 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m300(args.config, args.output)


if __name__ == "__main__":
    main()
