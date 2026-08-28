"""M143 — the integration layer on cached specialist scores.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v22.md`` (section 6
integration layer, section 9 M143, 14 Aug 2026).

Question (registered before measurement). E10 sealed that hard routing loses
(routed 18.8% / oracle ceiling 20.5% < global 22.5%). The integration layer
was never measured: (a) LATE FUSION — one closed-form fit over the
concatenated scores of the six specialists plus the global model; (b)
COMPETENCE ROUTING — a router that predicts WHICH arm will be right for a
row, instead of which domain the row belongs to. Do they recover the
specialist gains that hard routing lost?

Protocol (registered):
- Phase 1 (GPU): recompute the six per-domain specialist score vectors
  (512-atom A5 dictionaries, exact M139b construction) + the global head and
  the domain router from the sealed f6144 codes, and CACHE the score matrices
  to the GEODE cache (data_cache_root()/v16/m143/scores.npz). t1 anchors:
  class head 0.22487, router 0.7559, specialist own-domain vs M119 (0.002).
- Phase 2 (CPU): deterministic 50/50 split of the sealed test rows (seed
  registered). FIT half: stacking fit (ridge over 7x345 concatenated scores)
  + competence-router fit (ridge 2415 -> 7 arms, target = argmax-score arm).
  EVAL half: all arm accuracies, fused accuracy, competence-router accuracy,
  identity-router accuracy, random-router expectation (10 seeded draws).
- Gates (kill switches, registered): fired if (fused < global - 0.005 on the
  eval half) OR (competence < identity) OR (identity < random). The stacking
  guarantee (fused >= best arm) holds on the FIT half by construction; the
  gate is whether it survives to the eval half and whether competence routing
  beats identity routing.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m143_integration
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
from experiments.tier4.eval_v15_m104_experts import _inference_macs
from experiments.tier4.eval_v15_m107_dense import _score, _verify_pixel_identity
from experiments.tier4.eval_v16_a5_routed import _build_whitener, _domain_candidates
from experiments.tier4.eval_v16_m108_dictionary import _encode_block_device, _verify_device
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m139a_routing_slack import DualAccumulator, _score as _score_plain

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m143_integration_layer.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m143_integration_layer"

T1_TOLERANCE = 0.002
CLASSES = 345
DOMAINS = 6
PATCH_DIM = 108
ARMS = 7  # six specialists + global

import torch  # noqa: E402  (kept after constants to mirror the v16 layout)


# ---------------------------------------------------------------------------
# Phase 2 pure functions (CPU, unit-testable)
# ---------------------------------------------------------------------------

def _split_indices(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    mid = n_rows // 2
    return order[:mid], order[mid:]


def _stacking_fit(scores: np.ndarray, labels: np.ndarray, penalty: float = 1.0):
    """Ridge over concatenated arm scores -> classes. Returns (w, b, standardiser)."""
    features = np.asarray(scores, dtype=np.float64)
    centre = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    xs = (features - centre) / scale
    targets = np.zeros((len(labels), CLASSES), dtype=np.float64)
    targets[np.arange(len(labels)), np.asarray(labels, dtype=np.int64)] = 1.0
    d = xs.shape[1]
    w = np.linalg.solve(xs.T @ xs + penalty * np.eye(d), xs.T @ targets)
    b = targets.mean(axis=0)

    def predict(block: np.ndarray) -> np.ndarray:
        z = (np.asarray(block, dtype=np.float64) - centre) / scale
        logits = z @ w + b
        return np.argmax(logits, axis=1)

    return predict


def _competence_fit(scores: np.ndarray, labels: np.ndarray, penalty: float = 1.0):
    """Ridge 2415 -> 7 arms. Target for row i = the lowest-index arm whose
    argmax matches the row's label (ties -> lowest index), the standard
    classifier-selection protocol: labels on the FIT half only (amended from
    the score-derived target after the smoke showed raw-score maxima are not
    comparable across arms)."""
    features = np.asarray(scores, dtype=np.float64)
    centre = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    xs = (features - centre) / scale
    blocks = features.reshape(len(features), ARMS, CLASSES)
    correct = np.argmax(blocks, axis=2) == np.asarray(labels, dtype=np.int64)[:, None]
    best_arm = np.argmax(correct, axis=1)  # lowest index among correct arms
    targets = np.zeros((len(best_arm), ARMS), dtype=np.float64)
    targets[np.arange(len(best_arm)), best_arm] = 1.0
    d = xs.shape[1]
    w = np.linalg.solve(xs.T @ xs + penalty * np.eye(d), xs.T @ targets)
    b = targets.mean(axis=0)

    def predict(block: np.ndarray) -> np.ndarray:
        z = (np.asarray(block, dtype=np.float64) - centre) / scale
        logits = z @ w + b
        return np.argmax(logits, axis=1)

    return predict


def _arm_accuracy(arm_scores: np.ndarray, labels: np.ndarray, arm: int) -> float:
    preds = np.argmax(arm_scores[arm], axis=1)
    return float((preds == labels).mean())


def _random_router_accuracy(arm_scores: np.ndarray, labels: np.ndarray,
                            seed: int, draws: int = 10) -> float:
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(draws):
        picks = rng.integers(0, ARMS, size=len(labels))
        preds = np.argmax(arm_scores[picks, np.arange(len(labels))], axis=1)
        accs.append(float((preds == labels).mean()))
    return float(np.mean(accs))


def _select_penalty(metric, ladder: list[float]):
    """Pick the ladder penalty maximising *metric(penalty)* (first max wins).

    Returns (best_penalty, {str(penalty): score})."""
    best_penalty = ladder[0]
    best_score = -1.0
    scores: dict[str, float] = {}
    for penalty in ladder:
        score = float(metric(penalty))
        scores[str(penalty)] = score
        if score > best_score:
            best_score = score
            best_penalty = penalty
    return best_penalty, scores


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_m143(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    phase1 = bool(config.get("phase1", True))
    phase2 = bool(config.get("phase2", True))
    scores_path = data_cache_root() / config["score_cache"]["cache_relpath"]
    scores_file = scores_path / "scores.npz"

    evidence: dict[str, Any] = {
        "milestone": "M143",
        "admissible_as_evidence": not inadmissible,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": ("does the integration layer (late fusion + competence "
                     "routing) recover the specialist gains that hard routing "
                     "lost (E10)?"),
    }
    anchors: dict[str, Any] = {}

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
        train_cap = int(config["sparse"].get("_smoke_train_cap", 10 ** 9))
        test_rows_n = int(config["sparse"].get("_smoke_test_rows", 10 ** 9))

        print("building global whitener (M108 exact)", flush=True)
        whitener = _build_whitener(config, corpus)

        test_labels = corpus["test_labels"][:test_rows_n]
        test_domains = corpus["test_domains"][:test_rows_n]
        n_test = len(test_labels)

        specialist_scores = np.empty((DOMAINS, n_test, CLASSES),
                                     dtype=np.float32)
        own_domain_accuracy: dict[int, float] = {}
        m119 = json.loads(
            (REPO_ROOT / config["anchors"]["m119_evidence"]).read_text(
                encoding="utf-8"))

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
            from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
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

            # own-domain t1 anchor vs M119 sealed
            own_rows = np.where(corpus["test_domains"] == domain)[0]
            own_rows = own_rows[own_rows < n_test]
            if len(own_rows):
                hits = 0
                for start in range(0, len(own_rows), batch):
                    take = own_rows[start:start + batch]
                    block_t = _encode_block_device(
                        corpus["test_images"][take], table, whitener, pool_grid)
                    hits += int(_score(weights, standardise(block_t),
                                       corpus["test_labels"][take]).sum())
                    if throttle > 0:
                        time.sleep(throttle)
                own_acc = hits / len(own_rows)
                own_domain_accuracy[domain] = own_acc
                sealed_n = int(m119["specialist_curves"][str(domain)]["n_domain_rows"])
                sealed_acc = float(m119["specialist_curves"][str(domain)]
                                   ["accuracy"][str(sealed_n)])
                print(f"domain {domain}: own-domain {own_acc:.4f} "
                      f"(sealed {sealed_acc:.4f}, "
                      f"delta {own_acc - sealed_acc:+.6f})", flush=True)
                anchors[f"specialist_d{domain}"] = {
                    "measured": own_acc, "sealed": sealed_acc,
                    "delta": own_acc - sealed_acc}

            # full-test SCORE matrix for this specialist
            for start in range(0, n_test, batch):
                stop = min(start + batch, n_test)
                block_t = _encode_block_device(
                    corpus["test_images"][start:stop], table, whitener,
                    pool_grid)
                xs = standardise(block_t)
                specialist_scores[domain, start:stop] = (
                    xs @ weights[:-1] + weights[-1])
                if throttle > 0:
                    time.sleep(throttle)
            print(f"domain {domain}: scores cached ({n_test} rows)", flush=True)

        # global head + domain router from sealed f6144 codes
        print("dual accumulate on sealed f6144 codes (class + domain heads)",
              flush=True)
        codes_dir = data_cache_root() / config["sealed_codes"]["cache_relpath"]
        mem_train = np.load(codes_dir / config["sealed_codes"]["train_file"],
                            mmap_mode="r")
        mem_test = np.load(codes_dir / config["sealed_codes"]["test_file"],
                           mmap_mode="r")
        width = int(config["sealed_codes"]["width"])
        dual = DualAccumulator(width, CLASSES, DOMAINS)
        n_full = int(config["corpus"]["train_rows_per_class"] * CLASSES)
        for start in range(0, n_full, block):
            stop = min(start + block, n_full)
            dual.add(mem_train[start:stop], corpus["train_labels"][start:stop],
                     corpus["train_domains"][start:stop])
        centre, scale = dual._standardiser()
        w_class = dual.solve(1.0, dual.cross_class, dual.class_count)
        w_domain = dual.solve(1.0, dual.cross_domain, dual.domain_count)

        global_scores = np.empty((n_test, CLASSES), dtype=np.float32)
        router_preds = np.empty(n_test, dtype=np.int64)
        router_margins = np.empty(n_test, dtype=np.float64)
        for start in range(0, n_test, block):
            stop = min(start + block, n_test)
            xs = (np.asarray(mem_test[start:stop]) - centre) / scale
            global_scores[start:stop] = xs @ w_class[:-1] + w_class[-1]
            scores_d = xs @ w_domain[:-1] + w_domain[-1]
            router_preds[start:stop] = np.argmax(scores_d, axis=1)
            ranked = np.sort(scores_d, axis=1)
            router_margins[start:stop] = ranked[:, -1] - ranked[:, -2]

        class_accuracy = float(
            (np.argmax(global_scores, axis=1) == test_labels).mean())
        router_accuracy = float((router_preds == test_domains).mean())
        anchors["class_head"] = {"measured": class_accuracy,
                                 "delta": class_accuracy - 0.2248695652173913}
        anchors["router"] = {"measured": router_accuracy,
                             "delta": router_accuracy - 0.7559}
        print(f"  class head {class_accuracy:.4f}; router {router_accuracy:.4f}",
              flush=True)

        scores_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            scores_file,
            specialist_scores=specialist_scores,
            global_scores=global_scores,
            router_preds=router_preds,
            router_margins=router_margins,
            test_labels=test_labels,
            test_domains=test_domains,
            own_domain_accuracy=np.asarray(
                [own_domain_accuracy.get(d, np.nan) for d in range(DOMAINS)]),
        )
        print(f"cached scores -> {scores_file}", flush=True)

    if phase2:
        payload = np.load(scores_file, allow_pickle=False)
        specialist_scores = payload["specialist_scores"]  # (6, n, 345)
        global_scores = payload["global_scores"]          # (n, 345)
        router_preds = payload["router_preds"]
        test_labels = payload["test_labels"]
        test_domains = payload["test_domains"]
        n_test = len(test_labels)

        arm_scores = np.concatenate(
            [specialist_scores, global_scores[None, :, :]], axis=0)  # (7, n, 345)
        concat = np.concatenate(
            [specialist_scores.reshape(DOMAINS, n_test, CLASSES)
                .transpose(1, 0, 2).reshape(n_test, -1),
             global_scores], axis=1)  # (n, 2415)

        split_seed = int(config["phase2"]["split_seed"])
        fit_idx, eval_idx = _split_indices(n_test, split_seed)
        valid_split = int(config["phase2"]["valid_frac"] * len(fit_idx))
        rng = np.random.default_rng(int(config["phase2"]["valid_seed"]))
        fit_order = rng.permutation(len(fit_idx))
        fit_train_idx = fit_idx[fit_order[:valid_split]]
        fit_valid_idx = fit_idx[fit_order[valid_split:]]
        ladder = [float(x) for x in config["phase2"]["penalty_ladder"]]
        print(f"phase2: fit {len(fit_idx)} / eval {len(eval_idx)} rows; "
              f"penalty ladder {ladder}", flush=True)

        def _stack_metric(penalty):
            predict = _stacking_fit(concat[fit_train_idx],
                                    test_labels[fit_train_idx], penalty)
            return float((predict(concat[fit_valid_idx])
                          == test_labels[fit_valid_idx]).mean())

        fusion_penalty, fusion_ladder_scores = _select_penalty(
            _stack_metric, ladder)
        stacking = _stacking_fit(concat[fit_idx], test_labels[fit_idx],
                                 fusion_penalty)

        # competence router: select penalty by arm-pick accuracy on the valid slice
        def _comp_metric(penalty):
            predict = _competence_fit(concat[fit_train_idx],
                                      test_labels[fit_train_idx], penalty)
            picks = predict(concat[fit_valid_idx])
            preds = np.argmax(arm_scores[picks, fit_valid_idx], axis=1)
            return float((preds == test_labels[fit_valid_idx]).mean())

        router_penalty, router_ladder_scores = _select_penalty(
            _comp_metric, ladder)
        competence = _competence_fit(concat[fit_idx], test_labels[fit_idx],
                                     router_penalty)

        fused_preds = stacking(concat[eval_idx])
        comp_preds = competence(concat[eval_idx])
        eval_labels = test_labels[eval_idx]

        fused_acc = float((fused_preds == eval_labels).mean())
        comp_acc = float(
            (np.argmax(arm_scores[comp_preds, eval_idx], axis=1)
             == eval_labels).mean())
        identity_acc = float(
            (np.argmax(arm_scores[router_preds[eval_idx], eval_idx], axis=1)
             == eval_labels).mean())
        random_acc = _random_router_accuracy(
            arm_scores[:, eval_idx], eval_labels,
            int(config["phase2"]["random_seed"]))

        arm_accs = {"global": float((np.argmax(global_scores[eval_idx], axis=1)
                                     == eval_labels).mean()),
                    "routed": identity_acc,
                    "oracle": float((np.argmax(
                        arm_scores[test_domains[eval_idx], eval_idx], axis=1)
                        == eval_labels).mean())}
        print(f"  fused {fused_acc:.4f}; competence {comp_acc:.4f}; "
              f"identity {identity_acc:.4f}; random {random_acc:.4f}",
              flush=True)
        print(f"  arms {arm_accs}", flush=True)

        fired = (fused_acc < arm_accs["global"] - 0.005) or (
            comp_acc < identity_acc) or (identity_acc < random_acc)
        evidence["phase2"] = {
            "split_seed": split_seed,
            "fit_rows": int(len(fit_idx)),
            "eval_rows": int(len(eval_idx)),
            "fusion_penalty_selected": fusion_penalty,
            "fusion_ladder_scores": fusion_ladder_scores,
            "router_penalty_selected": router_penalty,
            "router_ladder_scores": router_ladder_scores,
            "fused_accuracy": fused_acc,
            "competence_accuracy": comp_acc,
            "identity_accuracy": identity_acc,
            "random_router_accuracy": random_acc,
            "arm_accuracies": arm_accs,
            "gate": {
                "registered": config["gate"]["registered"],
                "fused_ok": fused_acc >= arm_accs["global"] - 0.005,
                "competence_ok": comp_acc >= identity_acc,
                "identity_ok": identity_acc >= random_acc,
                "fired": fired,
                "consequence": (config["gate"]["consequence_fired"] if fired
                                else config["gate"]["consequence_passed"]),
            },
        }

    evidence["anchors"] = anchors
    evidence["score_cache"] = {"path": str(scores_file),
                               "relpath": config["score_cache"]["cache_relpath"]}
    evidence["runtime_seconds"] = round(time.time() - started, 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM143 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m143(args.config, args.output)


if __name__ == "__main__":
    main()
