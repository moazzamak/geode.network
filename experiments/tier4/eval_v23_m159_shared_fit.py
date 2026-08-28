"""M159 — shared-fit specialists (head-level data sharing).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md`` (section 4
M159, section 6; 16 Aug 2026). The build registration resolved the
"138k/410k" row scale: heads fit on ALL 138k train rows (the M143b
train-protocol scale, where the score-level M154 positive was measured);
the 410k variant is a cost-gated follow-up run only if this cell passes.

Phase 1 (GPU): rebuild the M108 whitener + the six domain candidate
pools (the M143 phase-1 construction: ``_build_whitener`` +
``_domain_candidates`` per domain, [11,100] prefix, 512 atoms); encode
all 138k train rows and the 34,500 test rows per specialist dictionary
(codes persisted under ``v16/m159/``). Anchors: each specialist's
own-domain reproduction from the M119 sealed curve values (tol 0.002,
the M143 a2 pattern — the head refit on that domain's rows only).

Phase 2 (CPU): each specialist head = ridge penalty 1.0 on ALL 138k
train rows; the fusion = the M143b protocol over [6 specialists,
global] on the train scores (valid seed 55, frac 0.8, ladder
{1,10,100,1000,10000}), evaluated on the test scores; the global arm =
the cached M143b global_train/global_test scores (unchanged). Anchors:
the M143b fused/global reads (0.22431884057971013 / 0.22460869565217392,
tol 1e-9). Controls: competence vs identity routing reported alongside.
Gate: fused >= global + 0.005 on the sealed test scores, else scoped
negative. Smoke declares inadmissibility and refuses the sealed output
directory.
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
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    _score,
)
from experiments.tier4.eval_v15_m107_dense import _verify_pixel_identity
from experiments.tier4.eval_v16_a5_routed import (
    _build_whitener,
    _domain_candidates,
)
from experiments.tier4.eval_v16_m108_dictionary import (
    _encode_block_device,
    _verify_device,
)
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m143_integration import (
    _competence_fit,
    _random_router_accuracy,
    _select_penalty,
    _stacking_fit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v23"
                  / "m159_shared_fit.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v23"
                  / "m159_shared_fit")

CLASSES = 345
DOMAINS = 6
TOLERANCE = 1e-9
A2_TOLERANCE = 0.002
MARGIN = 0.005
VALID_FRAC = 0.8
VALID_SEED = 55
LADDER = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def _score_matrix(codes: np.ndarray, weights: np.ndarray, std,
                  block: int) -> np.ndarray:
    """Blocked score matrix (n, CLASSES) from cached codes."""
    n = len(codes)
    out = np.empty((n, CLASSES), dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        xs = std(np.asarray(codes[start:stop]))
        out[start:stop] = xs @ weights[:-1] + weights[-1]
    return out


def _concat_rows(spec: np.ndarray, glob: np.ndarray, n: int,
                 classes: int = CLASSES) -> np.ndarray:
    """(DOMAINS, n, C) + (n, C) -> (n, (DOMAINS+1)*C)."""
    return np.concatenate(
        [spec.reshape(DOMAINS, n, classes).transpose(1, 0, 2)
            .reshape(n, -1), glob], axis=1)


def run_m159(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    atoms = int(config["sparse"].get("_smoke_atoms", 0)) or int(
        config["sparse"]["atoms_per_domain"])
    phase1 = bool(config.get("phase1", True))
    phase2 = bool(config.get("phase2", True))

    evidence: dict[str, Any] = {
        "milestone": "M159",
        "cell": "shared-fit specialists (head-level data sharing at 138k)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
    }
    anchors: dict[str, Any] = {}
    cache = data_cache_root() / "v16" / "m159"
    cache.mkdir(parents=True, exist_ok=True)

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
        pool_grid = int(config["sparse"]["pool_grid"])

        print("loading corpus", flush=True)
        corpus, train_index, test_index = _load_corpus(config)
        size = int(config["corpus"]["image_size"])
        for split, idx in (("train", train_index), ("test", test_index)):
            _verify_pixel_identity(split, idx, corpus[f"{split}_images"],
                                   size,
                                   int(config["corpus"]
                                       ["pixel_identity_rows"]))

        print("building global whitener (M108 exact)", flush=True)
        whitener = _build_whitener(config, corpus)
        m119 = json.loads(
            (REPO_ROOT / config["anchors"]["m119_evidence"]).read_text(
                encoding="utf-8"))

        n_train = min(len(corpus["train_images"]), smoke_train)
        n_test = min(len(corpus["test_images"]), smoke_test)
        print(f"rows: train {n_train} / test {n_test}", flush=True)

        for domain in range(DOMAINS):
            print(f"domain {domain}: candidates + dictionary ({atoms} "
                  "atoms)", flush=True)
            candidates = _domain_candidates(corpus, domain, whitener)
            order = np.random.default_rng([11, 100]).permutation(
                len(candidates))
            dictionary = candidates[order[:atoms]]
            table = torch.from_numpy(
                np.ascontiguousarray(dictionary)).to(torch.float32).to(device)

            train_codes_path = cache / f"spec_d{domain}_train.npy"
            test_codes_path = cache / f"spec_d{domain}_test.npy"
            train_mem = np.lib.format.open_memmap(
                train_codes_path, mode="w+", dtype=np.float32,
                shape=(n_train, atoms * pool_grid * pool_grid))
            test_mem = np.lib.format.open_memmap(
                test_codes_path, mode="w+", dtype=np.float32,
                shape=(n_test, atoms * pool_grid * pool_grid))
            print(f"domain {domain}: encode train {n_train} rows", flush=True)
            for start in range(0, n_train, batch):
                stop = min(start + batch, n_train)
                train_mem[start:stop] = _encode_block_device(
                    corpus["train_images"][start:stop], table, whitener,
                    pool_grid)
                if throttle > 0:
                    time.sleep(throttle)
            print(f"domain {domain}: encode test {n_test} rows", flush=True)
            for start in range(0, n_test, batch):
                stop = min(start + batch, n_test)
                test_mem[start:stop] = _encode_block_device(
                    corpus["test_images"][start:stop], table, whitener,
                    pool_grid)
                if throttle > 0:
                    time.sleep(throttle)
            del table
            torch.cuda.empty_cache()

            # own-domain anchor (head on the domain's rows only, the
            # M143 a2 pattern, against the M119 sealed curve)
            if not skip_anchors and not smoke:
                rows_d = np.where(corpus["train_domains"] == domain)[0]
                n_d = len(rows_d)
                acc = RidgeAccumulator(atoms * pool_grid * pool_grid,
                                       CLASSES)
                for start in range(0, n_d, batch):
                    take = rows_d[start:start + batch]
                    acc.add(train_mem[take],
                            corpus["train_labels"][take])
                w_d = acc.solve_many([1.0])[1.0]
                std_d = acc.standardiser()
                own_rows = np.where(corpus["test_domains"] == domain)[0]
                own_rows = own_rows[own_rows < n_test]
                hits = 0
                for start in range(0, len(own_rows), batch):
                    stop = min(start + batch, len(own_rows))
                    hits += int(_score(w_d, std_d(test_mem[own_rows[
                        start:stop]]),
                        corpus["test_labels"][own_rows[start:stop]]
                    ).sum())
                own_acc = hits / len(own_rows)
                sealed_n = int(m119["specialist_curves"][str(domain)]
                               ["n_domain_rows"])
                sealed_acc = float(
                    m119["specialist_curves"][str(domain)]["accuracy"]
                    [str(sealed_n)])
                anchors[f"specialist_d{domain}"] = {
                    "measured": own_acc, "sealed": sealed_acc,
                    "delta": own_acc - sealed_acc,
                    "tolerance": A2_TOLERANCE}
                print(f"domain {domain}: own-domain {own_acc:.4f} "
                      f"(sealed {sealed_acc:.4f}, delta "
                      f"{own_acc - sealed_acc:+.6f})", flush=True)
                if abs(own_acc - sealed_acc) > A2_TOLERANCE:
                    evidence.update({
                        "void": True,
                        "void_reason": "specialist own-domain anchor failed",
                        "anchors": anchors})
                    _write(output_dir, evidence)
                    return evidence

        # ---- shared heads on ALL train rows + score matrices --------------
        print("shared heads on all train rows", flush=True)
        specialist_train = np.empty((DOMAINS, n_train, CLASSES),
                                    dtype=np.float32)
        specialist_test = np.empty((DOMAINS, n_test, CLASSES),
                                   dtype=np.float32)
        for domain in range(DOMAINS):
            train_mem = np.load(cache / f"spec_d{domain}_train.npy",
                                mmap_mode="r")
            test_mem = np.load(cache / f"spec_d{domain}_test.npy",
                               mmap_mode="r")
            acc = RidgeAccumulator(atoms * pool_grid * pool_grid, CLASSES)
            for start in range(0, n_train, block):
                stop = min(start + block, n_train)
                acc.add(np.asarray(train_mem[start:stop]),
                        corpus["train_labels"][start:stop])
            weights = acc.solve_many([1.0])[1.0]
            std = acc.standardiser()
            specialist_train[domain] = _score_matrix(train_mem, weights,
                                                     std, block)
            specialist_test[domain] = _score_matrix(test_mem, weights,
                                                    std, block)
            print(f"domain {domain}: shared head fit + scores", flush=True)

        np.savez_compressed(
            cache / "scores.npz",
            specialist_train=specialist_train,
            specialist_test=specialist_test,
            train_labels=corpus["train_labels"][:n_train],
            test_labels=corpus["test_labels"][:n_test],
            test_domains=corpus["test_domains"][:n_test],
            n_train=np.asarray(n_train, dtype=np.int64),
            n_test=np.asarray(n_test, dtype=np.int64),
        )
        print(f"cached scores -> {cache / 'scores.npz'}", flush=True)

    if phase2:
        configure_external_cache_environment()
        payload = np.load(cache / "scores.npz", allow_pickle=False)
        specialist_train = payload["specialist_train"]
        specialist_test = payload["specialist_test"]
        train_labels = payload["train_labels"]
        test_labels = payload["test_labels"]
        test_domains = payload["test_domains"]
        n_train = len(train_labels)
        n_test = len(test_labels)

        root = data_cache_root()
        m143b_train = np.load(root / config["score_caches"]["m143b_train"],
                              allow_pickle=False)
        global_train = m143b_train["global_train"][:n_train]
        m143_test = np.load(root / config["score_caches"]["m143"],
                            allow_pickle=False)
        global_test = m143_test["global_scores"][:n_test]
        if n_train > len(m143b_train["global_train"]):
            raise SystemExit("M159 premise failure: train rows exceed the "
                             "M143b train cache")
        if n_test > len(m143_test["global_scores"]):
            raise SystemExit("M159 premise failure: test rows exceed the "
                             "M143 test cache")

        # ---- anchor: the M143b protocol reproduced -------------------------
        print("anchor: M143b stacking reproduction", flush=True)
        spec_train_143b = m143b_train["specialist_train"][:, :n_train, :]
        glob_train_143b = m143b_train["global_train"][:n_train]
        lbl_train_143b = m143b_train["train_labels"][:n_train]

        train_concat_143b = _concat_rows(spec_train_143b,
                                         glob_train_143b, n_train)
        test_concat_143b = _concat_rows(
            m143_test["specialist_scores"][:, :n_test, :],
            m143_test["global_scores"][:n_test], n_test)
        order = np.random.default_rng(VALID_SEED).permutation(n_train)
        cut = int(VALID_FRAC * n_train)
        ft, fv = order[:cut], order[cut:]

        def _metric(feats, penalty):
            predict = _stacking_fit(feats[ft], lbl_train_143b[ft], penalty)
            return float((predict(feats[fv])
                          == lbl_train_143b[fv]).mean())

        penalty, _ladder = _select_penalty(
            lambda p: _metric(train_concat_143b, p), LADDER)
        stacking = _stacking_fit(train_concat_143b, lbl_train_143b,
                                 penalty)
        fused_m143b = float(
            (stacking(test_concat_143b) == test_labels).mean())
        global_acc = float(
            (np.argmax(global_test, axis=1) == test_labels).mean())
        anchors["m143b_fused"] = {
            "measured": fused_m143b,
            "sealed": float(config["anchors"]["m143b_fused"]),
            "delta": fused_m143b
            - float(config["anchors"]["m143b_fused"]),
            "tolerance": TOLERANCE}
        anchors["m143b_global"] = {
            "measured": global_acc,
            "sealed": float(config["anchors"]["m143b_global"]),
            "delta": global_acc
            - float(config["anchors"]["m143b_global"]),
            "tolerance": TOLERANCE}
        print(f"  fused {fused_m143b:.6f}; global {global_acc:.6f}",
              flush=True)
        if not skip_anchors and (
                abs(anchors["m143b_fused"]["delta"]) > TOLERANCE
                or abs(anchors["m143b_global"]["delta"]) > TOLERANCE):
            evidence.update({"void": True,
                             "void_reason": "M143b anchor reproduction "
                                            "failed",
                             "anchors": anchors})
            _write(output_dir, evidence)
            return evidence

        # ---- shared-fit fusion ---------------------------------------------
        print("shared-fit fusion", flush=True)
        train_concat = _concat_rows(specialist_train, global_train, n_train)
        test_concat = _concat_rows(specialist_test, global_test, n_test)
        pen_s, ladder_s = _select_penalty(
            lambda p: _metric(train_concat, p), LADDER)
        stacking_s = _stacking_fit(train_concat, train_labels, pen_s)
        fused_s = float((stacking_s(test_concat) == test_labels).mean())
        gain = fused_s - global_acc
        passed = gain >= MARGIN
        print(f"  shared-fit fused {fused_s:.6f} (gain {gain:+.6f}, "
              f"penalty {pen_s})", flush=True)

        # ---- routing controls (reported alongside, the M143b protocol) -----
        router_preds = m143_test["router_preds"][:n_test]
        arm_scores = np.concatenate(
            [specialist_test, global_test[None, :, :]], axis=0)
        train_arm_scores = np.concatenate(
            [specialist_train, global_train[None, :, :]], axis=0)
        identity_acc = float((np.argmax(
            arm_scores[router_preds, np.arange(n_test)], axis=1)
            == test_labels).mean())
        random_acc = _random_router_accuracy(arm_scores, test_labels, 44)

        def _comp_metric(penalty):
            predict = _competence_fit(train_concat[ft], train_labels[ft],
                                      penalty)
            picks = predict(train_concat[fv])
            preds = np.argmax(train_arm_scores[picks, fv], axis=1)
            return float((preds == train_labels[fv]).mean())

        router_penalty, _router_ladder = _select_penalty(_comp_metric,
                                                         LADDER)
        competence = _competence_fit(train_concat, train_labels,
                                     router_penalty)
        comp_preds = competence(test_concat)
        competence_acc = float((np.argmax(
            arm_scores[comp_preds, np.arange(n_test)], axis=1)
            == test_labels).mean())

        evidence.update({
            "anchors": anchors,
            "shared_fit_fusion": {
                "fused": fused_s, "penalty": pen_s, "ladder": ladder_s,
                "n_features": int(train_concat.shape[1]),
            },
            "global_accuracy": global_acc,
            "routing": {
                "competence": competence_acc,
                "identity": identity_acc,
                "random": random_acc,
            },
            "gate": {
                "registered": config["gate"]["registered"],
                "gain": gain,
                "required": MARGIN,
                "passed": bool(passed),
                "consequence": (config["gate"].get(
                    "consequence_passed", "passed") if passed
                    else config["gate"].get("consequence_fired", "fired")),
            },
            "runtime_seconds": round(time.time() - started, 2),
        })
        _write(output_dir, evidence)
        print(f"\nM159 complete -> {output_dir / 'evidence.json'}",
              flush=True)
        return evidence
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
    run_m159(args.config, args.output)


if __name__ == "__main__":
    main()
