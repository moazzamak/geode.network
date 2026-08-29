"""M375 - a measured two-stage chain: router then specialist.

Registered in ``analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md`` under
G11: assumption 3 argues *composition* (chaining, where one stage's
output feeds the next) and then supports it with *fusion* evidence
(concatenating code blocks into one head). Those are different
claims. No chain is measured anywhere in the paper.

**Deviation from the registered design, declared before running.**
G11 names "Whisper ASR -> frozen-BERT+ridge intent head". Neither
checkpoint is in the local cache and no speech-to-text corpus is
either, so that cell cannot run offline. This runs the other chain
the paper rests on, and arguably the more load-bearing one: the
router-then-specialist composition of Figure 2, on the sealed
DomainNet selection. It satisfies the gate's substance -- a real
two-stage chain, each stage alone, the identity-substituted
coalitions, and a contract check -- on data that is already sealed.

The chain:

    stage A (router)      DINOv2 features -> one of 6 domains
    stage B (specialist)  DINOv2 features + domain -> one of 345 classes

The coalitions the attribution rule needs, with the null artifact
of each stage's own contract substituted for an absent player:

    v({})    no routing, no head: the most frequent class
    v({A})   routing with a null head: the most frequent class
             *within the predicted domain*
    v({B})   a null router with the head: one monolithic 345-way
             head over all domains -- the sealed M144 anchor
    v({A,B}) the chain

An oracle-routed arm is measured alongside. It is not a coalition;
it separates two very different negatives, because "the chain lost
because routing is inaccurate" and "the chain lost because
specialising costs more training data than it buys" call for
opposite responses.
"""
from __future__ import annotations

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
from experiments.tier4.bench_v16_parity import feature
from experiments.tier4.eval_v15_m107_dense import (
    IMAGENET_MEAN,
    IMAGENET_STD,
)

CLASSES = 345
DOMAINS = 6
RESOLUTION = 56
RIDGE_ALPHA = 1.0
SEALED_TAG = "63f590097008f749"
ANCHOR = 0.245014492753623      # sealed M144 unpruned r56 read
ANCHOR_TOL = 0.002
BATCH = 256


def _images(split: str) -> np.ndarray:
    root = data_cache_root() / "domainnet_m107" / SEALED_TAG
    return np.load(root / f"{split}_{RESOLUTION}.npy", mmap_mode="r")


def _labels_and_domains() -> dict[str, np.ndarray]:
    """The sealed selection's labels, in the same positional order as
    the materialised r56 caches."""
    from experiments.tier4.eval_v16_m109_trunk import _load_corpus
    corpus, _tr, _te = _load_corpus({
        "corpus": {"image_size": 32, "train_rows_per_class": 400,
                   "test_rows_per_class": 100, "subsample_seed": 107,
                   "expected_subsample_sha256":
                       "63f590097008f749f3f1828b29d6f154de7b21a68"
                       "28a7b017ac141c0615fa09d"}})
    return {k: np.asarray(v) for k, v in corpus.items()
            if k.endswith("labels") or k.endswith("domains")}


def _extract(split: str, n_rows: int) -> np.ndarray:
    """DINOv2-small CLS + mean-patch features at r56, cached by
    (split, row count) so a smoke run cannot poison a full run."""
    cache = (data_cache_root() / "v26" / "m375_chain"
             / f"{split}_{n_rows}_dinov2s_r56.npy")
    if cache.exists():
        print(f"reusing {cache.name}", flush=True)
        return np.load(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)

    from transformers import Dinov2Model
    device = torch.device("cuda")
    weights = data_cache_root() / "torch" / "dinov2-small"
    model = Dinov2Model.from_pretrained(str(weights),
                                        dtype=torch.float32)
    model.eval().to(device)

    images = _images(split)
    out = np.empty((n_rows, 2 * model.config.hidden_size),
                   dtype=np.float32)
    started = time.time()
    for start in range(0, n_rows, BATCH):
        stop = min(start + BATCH, n_rows)
        block = images[start:stop].astype(np.float32) / 255.0
        block = (block - IMAGENET_MEAN) / IMAGENET_STD
        block = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
        with torch.no_grad():
            tokens = model(
                pixel_values=torch.from_numpy(block).to(device)
            ).last_hidden_state.cpu().numpy()
        out[start:stop] = feature(tokens)
        if start % (BATCH * 100) == 0:
            print(f"  {split} {start}/{n_rows} "
                  f"({time.time() - started:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    np.save(cache, out)
    print(f"{split}: {n_rows} rows in {time.time() - started:.0f}s",
          flush=True)
    return out


def _ridge(x: np.ndarray, y: np.ndarray, k: int,
           mean: np.ndarray, std: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray]:
    onehot = np.zeros((x.shape[0], k), dtype=np.float64)
    onehot[np.arange(x.shape[0]), y] = 1.0
    xn = (x - mean) / std
    w = np.linalg.solve(xn.T @ xn + RIDGE_ALPHA * np.eye(xn.shape[1]),
                        xn.T @ onehot)
    b = onehot.mean(axis=0) - xn.mean(axis=0) @ w
    return w, b


def _scores(x: np.ndarray, w: np.ndarray, b: np.ndarray,
            mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std) @ w + b


class ContractError(RuntimeError):
    """A chain whose stages do not compose is refused at
    registration, not at serving time."""


def check_contract(out_kind: str, out_cardinality: int,
                   in_kind: str, in_cardinality: int) -> None:
    """C_out(h) subset C_in(g): the paper's chaining condition."""
    if out_kind != in_kind or out_cardinality != in_cardinality:
        raise ContractError(
            f"stage A emits {out_kind}[{out_cardinality}]; stage B "
            f"consumes {in_kind}[{in_cardinality}]")


def main() -> int:
    started = time.time()
    configure_external_cache_environment()
    meta = _labels_and_domains()
    ytr, dtr = meta["train_labels"], meta["train_domains"]
    yte, dte = meta["test_labels"], meta["test_domains"]

    xtr = np.asarray(_extract("train", len(ytr)), dtype=np.float64)
    xte = np.asarray(_extract("test", len(yte)), dtype=np.float64)
    mean = xtr.mean(axis=0)
    std = np.maximum(xtr.std(axis=0), 1e-6)

    payload: dict[str, Any] = {
        "milestone": "M375",
        "finding": "G11 -- composition asserted from fusion evidence",
        "registered_in": "analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md",
        "deviation_from_registered_design": {
            "registered": "Whisper ASR -> frozen-BERT+ridge intent",
            "run": "DomainNet router -> per-domain specialist",
            "reason": "neither Whisper nor BERT nor any speech-to-text "
                      "corpus is in the local cache; the router/"
                      "specialist chain is the composition Figure 2 "
                      "actually rests on and its data is sealed",
            "declared_before_running": True,
        },
        "corpus": {"train_rows": int(len(ytr)),
                   "test_rows": int(len(yte)),
                   "classes": CLASSES, "domains": DOMAINS,
                   "sealed_selection_tag": SEALED_TAG},
    }

    # ---- gate clause 1: reproduce the sealed monolith -------------
    w_mono, b_mono = _ridge(xtr, ytr, CLASSES, mean, std)
    mono_pred = _scores(xte, w_mono, b_mono, mean, std).argmax(axis=1)
    v_b = float((mono_pred == yte).mean())
    reproduced = abs(v_b - ANCHOR) <= ANCHOR_TOL
    payload["anchor_reproduction"] = {
        "registered_m144_r56_read": ANCHOR,
        "measured": round(v_b, 6), "tolerance": ANCHOR_TOL,
        "reproduced": reproduced}
    print(f"\nanchor: registered {ANCHOR:.6f} measured {v_b:.6f} "
          f"-> {reproduced}")
    out_path = Path("analysis/m375_measured_chain.json")
    if not reproduced:
        payload["verdict"] = ("VOID -- the monolithic head does not "
                              "reproduce the sealed M144 read")
        out_path.write_text(json.dumps(payload, indent=2),
                            encoding="utf-8")
        return 1

    # ---- stage A: the router --------------------------------------
    w_r, b_r = _ridge(xtr, dtr, DOMAINS, mean, std)
    route_pred = _scores(xte, w_r, b_r, mean, std).argmax(axis=1)
    router_acc = float((route_pred == dte).mean())

    # ---- stage B: per-domain specialists --------------------------
    heads = {}
    for d in range(DOMAINS):
        rows = dtr == d
        heads[d] = _ridge(xtr[rows], ytr[rows], CLASSES, mean, std)

    def _specialised(assignment: np.ndarray) -> float:
        pred = np.empty(len(yte), dtype=np.int64)
        for d in range(DOMAINS):
            rows = assignment == d
            if not rows.any():
                continue
            w, b = heads[d]
            pred[rows] = _scores(xte[rows], w, b, mean,
                                 std).argmax(axis=1)
        return float((pred == yte).mean())

    v_ab = _specialised(route_pred)
    oracle = _specialised(dte)

    # ---- the null-substituted coalitions --------------------------
    counts = np.bincount(ytr, minlength=CLASSES)
    v_empty = float((yte == int(counts.argmax())).mean())
    per_domain_mode = np.array(
        [int(np.bincount(ytr[dtr == d], minlength=CLASSES).argmax())
         for d in range(DOMAINS)])
    v_a = float((per_domain_mode[route_pred] == yte).mean())

    # ---- Shapley over two players ---------------------------------
    phi_a = 0.5 * ((v_a - v_empty) + (v_ab - v_b))
    phi_b = 0.5 * ((v_b - v_empty) + (v_ab - v_a))

    payload["coalitions"] = {
        "v_empty (no router, no head)": round(v_empty, 6),
        "v_A (router + null head)": round(v_a, 6),
        "v_B (null router + monolithic head)": round(v_b, 6),
        "v_AB (the chain)": round(v_ab, 6),
    }
    payload["diagnostics"] = {
        "router_domain_accuracy": round(router_acc, 6),
        "oracle_routed_chain": round(oracle, 6),
        "routing_loss_vs_oracle": round(oracle - v_ab, 6),
        "specialisation_gain_at_oracle": round(oracle - v_b, 6),
    }
    payload["shapley"] = {
        "stage_A_router": round(phi_a, 6),
        "stage_B_head": round(phi_b, 6),
        "router_share": round(phi_a / (phi_a + phi_b), 6),
        "sums_to_v_AB_minus_v_empty":
            abs((phi_a + phi_b) - (v_ab - v_empty)) < 1e-12,
    }

    # ---- G12: the two rules the paper uses, on the same chain -----
    loo_a = v_ab - v_b
    loo_b = v_ab - v_a
    loo_share = loo_a / (loo_a + loo_b)
    payload["two_attribution_rules"] = {
        "note": "G12 says the paper divides chains by Shapley in one "
                "section and by leave-one-out in another. On this "
                "measured chain the two rules disagree about what "
                "the router is owed.",
        "leave_one_out_router": round(loo_a, 6),
        "leave_one_out_head": round(loo_b, 6),
        "loo_router_share": round(loo_share, 6),
        "shapley_router_share": round(phi_a / (phi_a + phi_b), 6),
        "disagreement_ratio":
            round(loo_share / (phi_a / (phi_a + phi_b)), 4),
        "loo_sums_to_v_AB_minus_v_empty":
            abs((loo_a + loo_b) - (v_ab - v_empty)) < 1e-12,
    }

    # ---- disclosure: the arms are not parameter-matched -----------
    payload["capacity_disclosure"] = {
        "monolith_head_params": int(xtr.shape[1] * CLASSES),
        "chain_params": int(xtr.shape[1] * DOMAINS
                            + DOMAINS * xtr.shape[1] * CLASSES),
        "note": "the chain carries six 345-way heads where the "
                "monolith carries one, so the comparison is not "
                "parameter-matched. It is artifact-matched, which "
                "is what the fee rule divides over: two stages are "
                "two priced artifacts regardless of their size. "
                "Each specialist is fitted on ~1/6 of the rows, so "
                "the chain trades parameters for training data.",
        "rows_per_specialist": [int((dtr == d).sum())
                                for d in range(DOMAINS)],
    }

    # ---- gate clause 3: the contract check ------------------------
    contract = {}
    try:
        check_contract("domain_label", DOMAINS, "domain_label", DOMAINS)
        contract["matched_pairing_admitted"] = True
    except ContractError as exc:
        contract["matched_pairing_admitted"] = False
        contract["unexpected_refusal"] = str(exc)
    try:
        check_contract("class_label", CLASSES, "domain_label", DOMAINS)
        contract["mismatched_pairing_refused"] = False
    except ContractError as exc:
        contract["mismatched_pairing_refused"] = True
        contract["refusal_reason"] = str(exc)
    payload["contract_check"] = contract

    chain_wins = v_ab > max(v_a, v_b)
    payload["verdict"] = {
        "chain_beats_strongest_single_stage": bool(chain_wins),
        "coalitions_all_computed": True,
        "contract_refuses_mismatch":
            bool(contract["mismatched_pairing_refused"]),
        "assumption_3b": "supported" if chain_wins else "WITHDRAWN",
    }
    payload["runtime_seconds"] = round(time.time() - started, 1)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nv(empty)={v_empty:.6f}  v(A)={v_a:.6f}  "
          f"v(B)={v_b:.6f}  v(AB)={v_ab:.6f}")
    print(f"router domain accuracy {router_acc:.6f}  "
          f"oracle-routed {oracle:.6f}")
    print(f"shapley: router {phi_a:.6f}  head {phi_b:.6f}")
    print(f"router share -- shapley "
          f"{phi_a / (phi_a + phi_b):.4f}  loo {loo_share:.4f}")
    print(f"chain beats strongest stage: {chain_wins}")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
