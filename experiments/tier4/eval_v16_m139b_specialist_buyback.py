"""M139b — the assembled specialist buy-back.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v21.md`` (M139 amendment,
13 Aug 2026) and ``experiments/configs/v16/m139b_specialist_buyback.json``.

Question. The parts are sealed: per-domain specialists are super-additive at
~5.6x fewer MACs (M119/M124) and the code-router dispatches domains at 75.6%
(M139a). The ASSEMBLED system was never run. Does it beat the sealed dense r28
arm per domain at ~2.4x fewer per-image MACs, with NO domain labels at
inference?

Arms (registered): global (sealed f6144 head), oracle (specialist of the true
domain - the ceiling), routed (specialist by router argmax - the deployed arm),
gated(tau) for tau in {0.0, 0.2046} (router margin >= tau -> routed, else
global; 0.2046 = the sealed M139a margin q25).

t1 anchors: per-domain specialist accuracy on its OWN test rows reproduces the
M119 sealed 512-atom full-n values (0.002); the class head reproduces M117's
0.22487; the router reproduces M139a's 0.7559.

Kill switch: the ROUTED arm must beat the sealed dense r28 per-domain accuracy
on >= 4/6 domains; fewer -> the assembled buy-back fails the A5 pattern and the
negative is sealed.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m139b_specialist_buyback
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
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator, _inference_macs
from experiments.tier4.eval_v15_m107_dense import _score, _verify_pixel_identity
from experiments.tier4.eval_v16_a5_routed import _build_whitener, _domain_candidates
from experiments.tier4.eval_v16_m108_dictionary import _encode_block_device, _verify_device
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m139a_routing_slack import DualAccumulator, _score as _score_plain

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m139b_specialist_buyback.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m139b_specialist_buyback"

T1_TOLERANCE = 0.002
CLASSES = 345
DOMAINS = 6
M119_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m119_specialist_scale" / "evidence.json"
PATCH_DIM = 108


def _per_domain_accuracy(predictions: np.ndarray, labels: np.ndarray,
                         domains: np.ndarray) -> tuple[float, list[float]]:
    correct = predictions == labels
    per_domain = [float(correct[domains == d].mean()) if (domains == d).any() else 0.0
                  for d in range(DOMAINS)]
    return float(correct.mean()), per_domain


def _assemble_arms(specialist_predictions: np.ndarray, router_predictions: np.ndarray,
                   router_margins: np.ndarray, global_predictions: np.ndarray,
                   true_domains: np.ndarray, taus: list[float]
                   ) -> dict[str, np.ndarray]:
    """The four registered arms as per-row 345-class predictions.

    specialist_predictions: (domains, rows). oracle = specialist of the TRUE
    domain; routed = specialist chosen by the router; gated(tau) = routed where
    the router margin >= tau, else global.
    """
    rows = np.arange(len(true_domains))
    routed = specialist_predictions[router_predictions, rows]
    arms: dict[str, np.ndarray] = {
        "global": global_predictions,
        "oracle": specialist_predictions[true_domains, rows],
        "routed": routed,
    }
    for tau in taus:
        arms[f"gated({tau})"] = np.where(router_margins >= tau, routed,
                                         global_predictions)
    return arms


def run_m139b(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if "_smoke_note" in config and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    torch.manual_seed(int(config["numerics"]["seed"]))
    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    smoke = bool(config.get("_smoke_skip_gates", False))
    batch = int(config["numerics"]["batch"])
    throttle = float(config["numerics"]["encode_throttle_seconds"])
    started = time.time()

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

    # ---- per-domain specialists -------------------------------------------
    m119 = json.loads(M119_EVIDENCE.read_text(encoding="utf-8"))
    specialist_predictions = np.empty((DOMAINS, n_test), dtype=np.int64)
    own_domain_accuracy: dict[int, float] = {}
    t1_specialist: dict[int, dict[str, float]] = {}
    specialist_macs = _inference_macs(atoms, whitener.grid, PATCH_DIM, pool_grid,
                                      CLASSES)["total"]
    print(f"specialist per-image MACs: {specialist_macs/1e6:.1f}M", flush=True)

    for domain in range(DOMAINS):
        print(f"domain {domain}: candidates + dictionary ({atoms} atoms)", flush=True)
        candidates = _domain_candidates(corpus, domain, whitener)
        order = np.random.default_rng([11, 100]).permutation(len(candidates))[:atoms]
        dictionary = candidates[order]
        table = torch.from_numpy(
            np.ascontiguousarray(dictionary)).to(torch.float32).to(device)

        rows_d = np.where(corpus["train_domains"] == domain)[0][:train_cap]
        n_d = len(rows_d)
        acc = RidgeAccumulator(atoms * pool_grid * pool_grid, CLASSES)
        print(f"domain {domain}: fitting on {n_d} train rows", flush=True)
        for start in range(0, n_d, batch):
            take = rows_d[start:start + batch]
            block = _encode_block_device(corpus["train_images"][take], table,
                                         whitener, pool_grid)
            acc.add(block, corpus["train_labels"][take])
            if throttle > 0:
                time.sleep(throttle)
        weights = acc.solve_many([1.0])[1.0]
        standardise = acc.standardiser()

        # own-domain test accuracy (t1 anchor vs M119 sealed)
        own_rows = np.where(corpus["test_domains"] == domain)[0]
        own_rows = own_rows[own_rows < n_test]
        if len(own_rows) == 0:
            own_domain_accuracy[domain] = float("nan")
            t1_specialist[domain] = {"measured": None, "sealed": None,
                                     "delta": None, "note": "no own-domain rows in slice"}
            print(f"domain {domain}: no own-domain test rows in slice", flush=True)
        else:
            hits = 0
            for start in range(0, len(own_rows), batch):
                take = own_rows[start:start + batch]
                block = _encode_block_device(corpus["test_images"][take], table,
                                             whitener, pool_grid)
                hits += int(_score(weights, standardise(block),
                                   corpus["test_labels"][take]).sum())
                if throttle > 0:
                    time.sleep(throttle)
            own_acc = hits / len(own_rows)
            own_domain_accuracy[domain] = own_acc
            sealed_n = int(m119["specialist_curves"][str(domain)]["n_domain_rows"])
            sealed_acc = float(m119["specialist_curves"][str(domain)]["accuracy"][str(sealed_n)])
            t1_specialist[domain] = {"measured": own_acc, "sealed": sealed_acc,
                                     "delta": own_acc - sealed_acc}
            print(f"domain {domain}: own-domain acc {own_acc:.4f} "
                  f"(sealed {sealed_acc:.4f}, delta {own_acc - sealed_acc:+.6f})", flush=True)

        # full-test predictions for the arms
        preds = np.empty(n_test, dtype=np.int64)
        for start in range(0, n_test, batch):
            stop = min(start + batch, n_test)
            block = _encode_block_device(corpus["test_images"][start:stop], table,
                                         whitener, pool_grid)
            preds[start:stop] = np.argmax(standardise(block) @ weights[:-1]
                                          + weights[-1], axis=1)
            if throttle > 0:
                time.sleep(throttle)
        specialist_predictions[domain] = preds

    # ---- global head + domain router (dual accumulate, sealed f6144) -------
    print("dual accumulate on sealed f6144 codes (class + domain heads)", flush=True)
    codes_dir = data_cache_root() / config["sealed_codes"]["cache_relpath"]
    mem_train = np.load(codes_dir / config["sealed_codes"]["train_file"], mmap_mode="r")
    mem_test = np.load(codes_dir / config["sealed_codes"]["test_file"], mmap_mode="r")
    width = int(config["sealed_codes"]["width"])
    block = int(config["numerics"]["block"])
    dual = DualAccumulator(width, CLASSES, DOMAINS)
    n_full = int(config["corpus"]["train_rows_per_class"] * CLASSES)
    for start in range(0, n_full, block):
        stop = min(start + block, n_full)
        dual.add(mem_train[start:stop], corpus["train_labels"][start:stop],
                 corpus["train_domains"][start:stop])
    centre, scale = dual._standardiser()
    w_class = dual.solve(1.0, dual.cross_class, dual.class_count)
    w_domain = dual.solve(1.0, dual.cross_domain, dual.domain_count)

    global_preds = np.empty(n_test, dtype=np.int64)
    router_preds = np.empty(n_test, dtype=np.int64)
    router_margins = np.empty(n_test, dtype=np.float64)
    for start in range(0, n_test, block):
        stop = min(start + block, n_test)
        xs = (np.asarray(mem_test[start:stop]) - centre) / scale
        global_preds[start:stop] = _score_plain(w_class, xs)
        scores = xs @ w_domain[:-1] + w_domain[-1]
        router_preds[start:stop] = np.argmax(scores, axis=1)
        ranked = np.sort(scores, axis=1)
        router_margins[start:stop] = ranked[:, -1] - ranked[:, -2]

    class_accuracy = float((global_preds == test_labels).mean())
    router_accuracy = float((router_preds == test_domains).mean())
    t1_class = class_accuracy - 0.2248695652173913
    t1_router = router_accuracy - 0.7559
    print(f"  class head {class_accuracy:.4f} (t1 delta {t1_class:+.6f}); "
          f"router {router_accuracy:.4f} (t1 delta {t1_router:+.6f})", flush=True)

    # ---- arms ---------------------------------------------------------------
    arm_preds = _assemble_arms(specialist_predictions, router_preds,
                               router_margins, global_preds, test_domains,
                               [float(t) for t in config["routing"]["tau_ladder"]])
    arms: dict[str, Any] = {name: {"predictions": preds}
                            for name, preds in arm_preds.items()}

    evidence: dict[str, Any] = {
        "milestone": "M139b",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "specialist_macs": specialist_macs,
        "dense_r28": {
            "macs": int(config["gate"]["dense_r28_macs"]),
            "global_accuracy": float(config["gate"]["dense_r28_global_accuracy"]),
            "per_domain": [float(x) for x in config["gate"]["dense_r28_per_domain"]],
        },
        "anchors": {
            "t1_tolerance": T1_TOLERANCE,
            "specialist_own_domain": t1_specialist,
            "class_head": {"measured": class_accuracy, "delta": t1_class},
            "router": {"measured": router_accuracy, "delta": t1_router},
        },
        "arms": {},
    }
    for name, arm in arms.items():
        accuracy, per_domain = _per_domain_accuracy(arm["predictions"],
                                                    test_labels, test_domains)
        evidence["arms"][name] = {"accuracy": accuracy,
                                  "per_domain": per_domain}
        print(f"  {name:12s}: {accuracy:.4f} per_domain="
              f"{[round(p, 4) for p in per_domain]}", flush=True)

    # ---- kill switch ---------------------------------------------------------
    if not smoke:
        dense_pd = [float(x) for x in config["gate"]["dense_r28_per_domain"]]
        routed_pd = evidence["arms"]["routed"]["per_domain"]
        wins = [d for d in range(DOMAINS) if routed_pd[d] > dense_pd[d]]
        fired = len(wins) < int(config["gate"]["min_domains"])
        evidence["gate"] = {
            "registered": config["gate"]["kill_switch_buyback"],
            "routed_per_domain": routed_pd,
            "dense_r28_per_domain": dense_pd,
            "winning_domains": wins,
            "min_domains": int(config["gate"]["min_domains"]),
            "fired": fired,
            "consequence": (config["gate"]["consequence_fired"] if fired
                            else config["gate"]["consequence_passed"]),
            "oracle_ceiling": evidence["arms"]["oracle"],
        }
    evidence["runtime_seconds"] = round(time.time() - started, 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"wrote {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    run_m139b(Path(args.config), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
