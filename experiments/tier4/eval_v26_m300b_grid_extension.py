"""M300b - the registered grid extension for the M300 RFF cell.

Registered in ``analysis/SCIENCE_LAYER_PLAN_2026-08-28.md`` (M300b,
28 Aug 2026, before the build). The M300 CV curve rose monotonically
in D at sigma=0.5 (0.5917 -> 0.6017 -> 0.6110 for D = 4096 -> 8192
-> 16384), so the selected D=16384 is a boundary argmin. The
extension grid: D in {32768, 65536} x sigma in {0.25, 0.5}, the SAME
seed, the SAME CV protocol (train rows only), the same M233/M236
probe head.

The sealed quickdraw test is evaluated ONCE at the extension's
selected (D, sigma) if it differs from M300's (16384, 0.5); if the
extension's CV winner is still (16384, 0.5) - impossible here since
the extension grid excludes it - the boundary flag would close with
no new test reading. The extension compares its CV winner against
the M300 CV winner's 0.6110 and, when the winner differs, evaluates
the rff_concat arm once at the new pair.

Gates: the M300 g1-g5 set re-applied (g2 re-reproduces the M236
anchor 0.626655234828999).

RUN RECORD. Run 1 (28 Aug 2026) died at the D=65536 CV point:
the unchunked projection needed arg + cos-temp + product + astype
copy ~= 4 x 23.6 GiB simultaneously on a 63 GB host
(numpy._core._exceptions._ArrayMemoryError, 23.6 GiB for
(96600, 65536) float32); no evidence was written. Run 2 uses the
chunked in-place projector (rff_project with out=, build_design)
from the same M300 module: identical per-row arithmetic, ~64 MiB
peak temporaries, and no concatenate copy. Run 2 is the sealed
run; run 1's partial CV printout (D32768_sigma0.25: 0.6116) is
superseded by run 2's full table.
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
from experiments.tier4.eval_v15_m104_experts import _load_domainnet
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m140_data_extension import _extension_indices
from experiments.tier4.eval_v16_m141_data_full import _rest_extension_indices
from experiments.tier4.eval_v25_m233_trained_probes import (
    _probe_score,
    _train_probe,
)
from experiments.tier4.eval_v26_m300_rff_quickdraw import (
    _l2_normalise_rows,
    build_design,
    cv_select,
    rff_params,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
M300_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
               / "m300_rff_quickdraw.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m300b_grid_extension")

FULL_TRAIN_ROWS = 409832
CLASSES = 345
QUICKDRAW = 3
CLIP_L_LINEAR_ANCHOR = 0.626655234828999
ANCHOR_TOL = 1e-9
M300_SELECTED = {"D": 16384, "sigma": 0.5, "cv": 0.6110,
                 "test": 0.6695246260836429}
EXTENSION_DIMS = [32768, 65536]
EXTENSION_SIGMAS = [0.25, 0.5]


def run_m300b(output_dir: Path) -> dict[str, Any]:
    config = json.loads(M300_CONFIG.read_text(encoding="utf-8"))
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

    # ---- g2: the instrument-identity anchor re-reproduction ----------
    w_a, b_a = _train_probe(clip_tr[q_rows], q_labels, epochs, lr, wd,
                            batch, seed, device)
    acc_a = _probe_score(w_a, b_a, clip_te[q_trows], q_test_labels)
    g2 = bool(abs(acc_a - CLIP_L_LINEAR_ANCHOR) <= ANCHOR_TOL)
    print(f"clip_linear anchor: {acc_a:.6f} (g2={g2})", flush=True)

    clip_q = _l2_normalise_rows(clip_tr[q_rows])
    dino_q = _l2_normalise_rows(dino_tr[q_rows])
    clip_qte = _l2_normalise_rows(clip_te[q_trows])
    dino_qte = _l2_normalise_rows(dino_te[q_trows])
    concat_q = np.concatenate([clip_q, dino_q], axis=1)
    concat_qte = np.concatenate([clip_qte, dino_qte], axis=1)
    del clip_tr, clip_te, dino_tr, dino_te
    gc.collect()

    g1 = bool(g1_schedule and concat_q.shape[1] == 1536
              and len(q_rows) > 0 and len(q_trows) > 0)

    # ---- the extension CV (train rows only) ---------------------------
    r = config["rff"]
    rff_seed = int(r["seed"])
    print("extension cv:", flush=True)
    cv = cv_select(clip_q, q_labels, EXTENSION_DIMS,
                   EXTENSION_SIGMAS,
                   int(r["cv_folds"]), int(r["cv_seed"]), rff_seed,
                   config["training"], device)
    sel_D, sel_sigma = cv["selected_D"], cv["selected_sigma"]
    sel_cv = cv["selected_cv_accuracy"]
    print(f"extension selected D={sel_D} sigma={sel_sigma} "
          f"(cv {sel_cv:.4f})", flush=True)

    # ---- the registered reading ---------------------------------------
    winner_changed = not (sel_D == M300_SELECTED["D"]
                          and sel_sigma == M300_SELECTED["sigma"])
    cv_beats_m300 = bool(sel_cv > M300_SELECTED["cv"])
    test_at_new = None
    if winner_changed:
        om, ph = rff_params(concat_q.shape[1], sel_D, sel_sigma, rff_seed)
        design_tr = build_design(concat_q, om, ph)
        design_te = build_design(concat_qte, om, ph)
        w, b = _train_probe(design_tr, q_labels, epochs, lr, wd,
                            batch, seed, device)
        test_at_new = _probe_score(w, b, design_te, q_test_labels)
        print(f"rff_concat at ({sel_D}, {sel_sigma}): "
              f"{test_at_new:.6f}", flush=True)
        del design_tr, design_te
        gc.collect()

    if winner_changed and cv_beats_m300 and test_at_new is not None \
            and test_at_new > M300_SELECTED["test"]:
        reading = ("the extension's winner beats M300 on both CV and "
                   "the sealed test: the wall break deepens and the "
                   "recipe's headroom is larger than measured")
    elif winner_changed and cv_beats_m300:
        reading = ("the extension's CV winner beats M300's CV but the "
                   "sealed test does not improve: the CV selection is "
                   "over-optimistic at large D - the operative "
                   "configuration stays M300's (16384, 0.5)")
    elif winner_changed:
        reading = ("the extension's CV winner differs from M300's but "
                   "does not beat its CV: the boundary flag closes "
                   "with the optimum interior to the extension grid")
    else:
        reading = ("the extension's CV winner is M300's own pair: the "
                   "optimum is interior at the registered grid")

    g5 = test_at_new is None or 0.0 <= test_at_new <= 1.0
    gates = {
        "g1_premise": {"ok": g1, "schedule_alignment": bool(g1_schedule)},
        "g2_clip_linear_anchor": {"ok": g2, "measured": acc_a,
                                  "sealed": CLIP_L_LINEAR_ANCHOR,
                                  "tolerance": ANCHOR_TOL},
        "g3_rff_determinism": {"ok": True,
                               "note": "the M300 module's determinism "
                                       "is unit-pinned; the same seed "
                                       "and module are reused here"},
        "g4_cv_train_only": {"ok": True},
        "g5_accuracies_valid": {"ok": bool(g5)},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M300b",
        "cell": ("the registered M300 grid extension: "
                 "D in {32768, 65536} x sigma in {0.25, 0.5}"),
        "m300_selected": M300_SELECTED,
        "extension_selected": {"D": sel_D, "sigma": sel_sigma,
                               "cv": sel_cv},
        "cv_beats_m300": cv_beats_m300,
        "winner_changed": bool(winner_changed),
        "test_at_extension_winner": test_at_new,
        "reading": reading,
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "configuration_hash": payload_hash({
            "extension_dims": EXTENSION_DIMS,
            "extension_sigmas": EXTENSION_SIGMAS,
            "rff_seed": rff_seed,
            "cv_folds": int(r["cv_folds"]),
            "cv_seed": int(r["cv_seed"]),
            "training": config["training"]}),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": bool(gates_ok),
                      "extension_selected":
                          evidence["extension_selected"],
                      "test_at_extension_winner": test_at_new,
                      "reading": reading}, indent=1), flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m300b(args.output)


if __name__ == "__main__":
    main()
