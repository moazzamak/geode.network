"""M143b — integration layer fitted on TRAIN scores (the registered rescue).

Registered after M143's sealed negative (14 Aug 2026): M143's fusion and
competence router were fit on held-out test rows at 40 rows/class and lost to
the global arm (fused 0.1463 vs global 0.2251). A diagnostic on the cached
scores showed the fitter is sound (global-only stacking 0.159 vs global 0.2251
is a rows-per-class effect, not a bug). The rescue cell fits the same fusion
and router on the arms' OWN train scores (400 rows/class, the stacking
protocol the arms themselves saw), evaluated on the sealed test rows.

Phase 1 (GPU, ~2.5h): score the full TRAIN split with the six 512-atom
specialists (exact M143/M139b construction) and the global head (dual
accumulate on the sealed f6144 memmaps); cache to
data_cache_root()/v16/m143b/train_scores.npz.

Phase 2 (CPU): stacking + competence fit on the train rows (penalty ladder
selected on a seeded train-validation slice), evaluated on the sealed TEST
scores from v16/m143/scores.npz (never recomputed).

Gate (same shape as M143): fired if fused < global - 0.005 on the test rows,
OR competence < identity, OR identity < random.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m143b_train_fusion
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
from experiments.tier4.eval_v15_m107_dense import _score, _verify_pixel_identity
from experiments.tier4.eval_v16_a5_routed import _build_whitener, _domain_candidates
from experiments.tier4.eval_v16_m108_dictionary import _encode_block_device, _verify_device
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m139a_routing_slack import DualAccumulator, _score as _score_plain
from experiments.tier4.eval_v16_m143_integration import (
    ARMS,
    CLASSES,
    DOMAINS,
    _competence_fit,
    _random_router_accuracy,
    _select_penalty,
    _stacking_fit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m143b_train_fusion.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m143b_train_fusion"

T1_TOLERANCE = 0.002


def run_m143b(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    evidence: dict[str, Any] = {
        "milestone": "M143b",
        "admissible_as_evidence": not inadmissible,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": ("fitted on the arms' own TRAIN scores instead of held-out "
                     "rows, does the integration layer (late fusion + "
                     "competence routing) recover the specialist gains that "
                     "hard routing lost?"),
    }

    phase1 = bool(config.get("phase1", True))
    phase2 = bool(config.get("phase2", True))
    train_path = data_cache_root() / config["score_cache"]["cache_relpath"]
    train_file = train_path / "train_scores.npz"

    if phase1:
        torch.set_num_threads(int(config["numerics"]["torch_threads"]))
        torch.manual_seed(int(config["numerics"]["seed"]))
        configure_external_cache_environment()
        _verify_device(torch)
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)

        batch = int(config["numerics"]["batch"])
        throttle = float(config["numerics"]["encode_throttle_seconds"])
        block = int(config["numerics"]["block"])

        print("loading corpus", flush=True)
        corpus, train_index, test_index = _load_corpus(config)
        size = int(config["corpus"]["image_size"])
        for split, idx in (("train", train_index), ("test", test_index)):
            _verify_pixel_identity(split, idx, corpus[f"{split}_images"], size,
                                   int(config["corpus"]["pixel_identity_rows"]))

        atoms = int(config["sparse"]["atoms_per_domain"])
        pool_grid = int(config["sparse"]["pool_grid"])
        n_train = min(len(corpus["train_labels"]),
                      int(config["sparse"].get("_smoke_train_score_rows", 10 ** 9)))
        n_test = len(corpus["test_labels"])
        train_cap = int(config["sparse"].get("_smoke_train_cap", 10 ** 9))

        print("building global whitener (M108 exact)", flush=True)
        whitener = _build_whitener(config, corpus)

        specialist_train = np.empty((DOMAINS, n_train, CLASSES),
                                    dtype=np.float32)
        for domain in range(DOMAINS):
            print(f"domain {domain}: candidates + dictionary ({atoms} atoms)",
                  flush=True)
            candidates = _domain_candidates(corpus, domain, whitener)
            order = np.random.default_rng([11, 100]).permutation(len(candidates))
            dictionary = candidates[order[:atoms]]
            table = torch.from_numpy(
                np.ascontiguousarray(dictionary)).to(torch.float32).to(device)

            rows_d = np.where(corpus["train_domains"] == domain)[0][:train_cap]
            n_d = len(rows_d)
            acc = RidgeAccumulator(atoms * pool_grid * pool_grid, CLASSES)
            print(f"domain {domain}: fitting on {n_d} train rows", flush=True)
            for start in range(0, n_d, batch):
                take = rows_d[start:start + batch]
                block_t = _encode_block_device(corpus["train_images"][take],
                                               table, whitener, pool_grid)
                acc.add(block_t, corpus["train_labels"][take])
                if throttle > 0:
                    time.sleep(throttle)
            weights = acc.solve_many([1.0])[1.0]
            standardise = acc.standardiser()

            print(f"domain {domain}: scoring the full train split "
                  f"({n_train} rows)", flush=True)
            for start in range(0, n_train, batch):
                stop = min(start + batch, n_train)
                block_t = _encode_block_device(
                    corpus["train_images"][start:stop], table, whitener,
                    pool_grid)
                xs = standardise(block_t)
                specialist_train[domain, start:stop] = (
                    xs @ weights[:-1] + weights[-1])
                if throttle > 0:
                    time.sleep(throttle)
            print(f"domain {domain}: train scores cached", flush=True)

        print("dual accumulate on sealed f6144 codes (class + domain heads)",
              flush=True)
        codes_dir = data_cache_root() / config["sealed_codes"]["cache_relpath"]
        mem_train = np.load(codes_dir / config["sealed_codes"]["train_file"],
                            mmap_mode="r")
        width = int(config["sealed_codes"]["width"])
        dual = DualAccumulator(width, CLASSES, DOMAINS)
        n_full = min(int(config["corpus"]["train_rows_per_class"] * CLASSES),
                     n_train)
        for start in range(0, n_full, block):
            stop = min(start + block, n_full)
            dual.add(mem_train[start:stop], corpus["train_labels"][start:stop],
                     corpus["train_domains"][start:stop])
        centre, scale = dual._standardiser()
        w_class = dual.solve(1.0, dual.cross_class, dual.class_count)

        global_train = np.empty((n_train, CLASSES), dtype=np.float32)
        for start in range(0, n_train, block):
            stop = min(start + block, n_train)
            xs = (np.asarray(mem_train[start:stop]) - centre) / scale
            global_train[start:stop] = xs @ w_class[:-1] + w_class[-1]

        train_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            train_file,
            specialist_train=specialist_train,
            global_train=global_train,
            train_labels=corpus["train_labels"],
        )
        print(f"cached train scores -> {train_file}", flush=True)

    if phase2:
        payload = np.load(train_file, allow_pickle=False)
        specialist_train = payload["specialist_train"]
        global_train = payload["global_train"]
        train_labels = payload["train_labels"]
        n_train = len(train_labels)

        test_payload = np.load(
            data_cache_root() / config["test_cache"]["cache_relpath"]
            / "scores.npz", allow_pickle=False)
        specialist_test = test_payload["specialist_scores"]
        global_test = test_payload["global_scores"]
        test_labels = test_payload["test_labels"]
        router_preds = test_payload["router_preds"]
        n_test = len(test_labels)

        def concat_rows(spec: np.ndarray, glob: np.ndarray, n: int) -> np.ndarray:
            return np.concatenate(
                [spec.reshape(DOMAINS, n, CLASSES).transpose(1, 0, 2)
                    .reshape(n, -1), glob], axis=1)

        train_concat = concat_rows(specialist_train, global_train, n_train)
        test_concat = concat_rows(specialist_test, global_test, n_test)
        arm_scores = np.concatenate(
            [specialist_test, global_test[None, :, :]], axis=0)
        train_arm_scores = np.concatenate(
            [specialist_train, global_train[None, :, :]], axis=0)

        rng = np.random.default_rng(int(config["phase2"]["valid_seed"]))
        order = rng.permutation(n_train)
        cut = int(config["phase2"]["valid_frac"] * n_train)
        ft, fv = order[:cut], order[cut:]
        ladder = [float(x) for x in config["phase2"]["penalty_ladder"]]
        print(f"phase2: fit {cut} / valid {n_train - cut} train rows; "
              f"test {n_test}", flush=True)

        def _stack_metric(penalty):
            predict = _stacking_fit(train_concat[ft], train_labels[ft], penalty)
            return float((predict(train_concat[fv])
                          == train_labels[fv]).mean())

        fusion_penalty, fusion_ladder_scores = _select_penalty(
            _stack_metric, ladder)
        stacking = _stacking_fit(train_concat, train_labels, fusion_penalty)
        fused_preds = stacking(test_concat)
        fused_acc = float((fused_preds == test_labels).mean())

        def _comp_metric(penalty):
            predict = _competence_fit(train_concat[ft], train_labels[ft],
                                      penalty)
            picks = predict(train_concat[fv])
            preds = np.argmax(train_arm_scores[picks, fv], axis=1)
            return float((preds == train_labels[fv]).mean())

        router_penalty, router_ladder_scores = _select_penalty(
            _comp_metric, ladder)
        competence = _competence_fit(train_concat, train_labels,
                                     router_penalty)
        comp_preds = competence(test_concat)
        comp_acc = float((np.argmax(arm_scores[comp_preds,
                                               np.arange(n_test)], axis=1)
                          == test_labels).mean())
        identity_acc = float((np.argmax(
            arm_scores[router_preds, np.arange(n_test)], axis=1)
            == test_labels).mean())
        random_acc = _random_router_accuracy(
            arm_scores, test_labels, int(config["phase2"]["random_seed"]))

        global_acc = float((np.argmax(global_test, axis=1)
                            == test_labels).mean())
        print(f"  fused {fused_acc:.4f}; competence {comp_acc:.4f}; "
              f"identity {identity_acc:.4f}; random {random_acc:.4f}; "
              f"global {global_acc:.4f}", flush=True)

        fired = (fused_acc < global_acc - 0.005) or (
            comp_acc < identity_acc) or (identity_acc < random_acc)
        evidence["phase2"] = {
            "fit_rows": int(n_train),
            "fusion_penalty_selected": fusion_penalty,
            "fusion_ladder_scores": fusion_ladder_scores,
            "router_penalty_selected": router_penalty,
            "router_ladder_scores": router_ladder_scores,
            "fused_accuracy": fused_acc,
            "competence_accuracy": comp_acc,
            "identity_accuracy": identity_acc,
            "random_router_accuracy": random_acc,
            "global_accuracy": global_acc,
            "gate": {
                "registered": config["gate"]["registered"],
                "fused_ok": fused_acc >= global_acc - 0.005,
                "competence_ok": comp_acc >= identity_acc,
                "identity_ok": identity_acc >= random_acc,
                "fired": fired,
                "consequence": (config["gate"]["consequence_fired"] if fired
                                else config["gate"]["consequence_passed"]),
            },
        }

    evidence["train_cache"] = {"path": str(train_file)}
    evidence["runtime_seconds"] = round(time.time() - started, 2)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM143b complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m143b(args.config, args.output)


if __name__ == "__main__":
    main()
