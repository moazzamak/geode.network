"""M225 - semantics axis + inverse-relation analogies (prototype).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (M225
REGISTERED, 20 Aug 2026). The user's example made mechanical: the
direction from an operation to its INVERSE must generalise across
inverse pairs, word2vec-style:

    cos(emb[integration] - emb[differentiation],
        emb[fft] - emb[ifft])  >= 0.5        (G6, held-out pair 1)
    cos(emb[integration] - emb[differentiation],
        emb[convolution] - emb[deconvolution]) >= 0.5  (G6, held-out pair 2)

Scope: an EXTENDED encoder schema (the frozen v0 axes + the authored
``task.transform`` axis from
``analysis/fingerprint_relations_v1_transform.json``) trained with the
M224 v1 signal mix plus the inverse-pair hinge. PROTOTYPE ONLY: this
milestone does NOT migrate the product ontology - that is M226
(registered) and runs only if G1-G3+G6 pass. The v0 tasks are
registered with ``task.transform=identity``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from experiments.common.data_cache import configure_external_cache_environment
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v25_m224_fingerprint_v1_train import (
    DISSIMILAR_PAIRS,
    SIMILAR_PAIRS,
    TASK_DESCRIPTORS,
    _axis_key,
    _rel_loss,
)
from geode.core.descriptor import AXES, normalise
from geode.core.fingerprint import FingerprintEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m225_transform_analogies.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m225_transform_analogies")
RELATIONS_PATH = REPO_ROOT / "analysis" / "fingerprint_relations_v0.json"
TRANSFORM_REL_PATH = (REPO_ROOT / "analysis"
                      / "fingerprint_relations_v1_transform.json")

# the extended transform task descriptors (all numeric-series ->
# regression; only task.transform differs from mackey_glass's family)
_TRANSFORM_BASE = {"input.modality": "numeric-series",
                   "input.value_kind": "continuous",
                   "input.temporal_structure": "sequential",
                   "output.kind": "regression",
                   "output.ordinality": "cardinal",
                   "latent.recurrence": "markov",
                   "latent.stationarity": "stationary",
                   "latent.noise_regime": "low",
                   "latent.label_cardinality": 1,
                   "latent.sample_regime": "medium",
                   "coupling": "single-task"}
TRANSFORM_TASKS = {
    "integrate_series": {**_TRANSFORM_BASE, "task.transform": "integration"},
    "differentiate_series": {**_TRANSFORM_BASE,
                             "task.transform": "differentiation"},
    "fourier_series": {**_TRANSFORM_BASE, "task.transform": "fft"},
    "ifft_series": {**_TRANSFORM_BASE, "task.transform": "ifft"},
    "deconvolve_task": {**_TRANSFORM_BASE, "task.transform": "deconvolution"},
}
# registered M225 pair additions: a transform task is similar to the
# mackey-glass family; dissimilar to tabular regression on iid data.
M225_SIMILAR = SIMILAR_PAIRS + [("mackey_glass", "integrate_series")]
M225_DISSIMILAR = DISSIMILAR_PAIRS + [("integrate_series", "tabular")]

# the FROZEN G3 set: exactly the six registered v0 tasks x two axes
# (the 12 quadruples of the traversability artifact). Run 1 measured
# G3 over ALL tasks including the new transform tasks - an
# instrument-scope defect, not a verdict (registered 20 Aug).
FROZEN_G3_TASKS = ["domainnet", "cifar10", "mackey_glass", "lorenz",
                   "dyck", "tabular"]


def _token_vec(enc: FingerprintEncoder, axis: str, token: str
               ) -> torch.Tensor:
    idx = enc.token_index[(axis, token)]
    return enc.token_emb.weight[idx]


def _transform_rel_loss(enc: FingerprintEncoder, trel: dict[str, Any],
                        tau: float, rho: float,
                        exclude: list[list[str]]) -> torch.Tensor:
    loss = torch.zeros(())
    ref = trel["reference_inverse_pair"]
    r_dir = _token_vec(enc, trel["axis"], ref[0]) \
        - _token_vec(enc, trel["axis"], ref[1])
    for a, b in trel["inverse_pairs"]:
        if [a, b] == ref or [a, b] in exclude:
            continue
        d = _token_vec(enc, trel["axis"], a) \
            - _token_vec(enc, trel["axis"], b)
        cos = F.cosine_similarity(d.unsqueeze(0), r_dir.unsqueeze(0))
        loss = loss + torch.clamp(tau - cos, min=0.0)
    for a, b in trel["polar_pairs"]:
        va = _token_vec(enc, trel["axis"], a)
        vb = _token_vec(enc, trel["axis"], b)
        cos = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0))
        loss = loss + torch.clamp(cos + rho, min=0.0)
    return loss


def _frozen_swap_loss(enc: FingerprintEncoder,
                      descs: dict[str, Any], tau_swap: float
                      ) -> torch.Tensor:
    """M225c: the frozen-axis swap directions as an explicit training
    term (G3's reading is then TRAINED-not-held-out)."""
    loss = torch.zeros(())
    for name in FROZEN_G3_TASKS:
        desc = descs[name]
        for axis in ["input.modality", "output.kind"]:
            vocab = AXES[axis]
            alt = vocab[0] if desc.axes[axis] != vocab[0] else vocab[-1]
            alt_desc = normalise({**{a: v for a, v in desc.axes.items()},
                                  axis: alt})
            f_orig = enc.forward(desc)
            f_swap = enc.forward(alt_desc)
            e_new = enc.token_emb.weight[enc.token_index[(axis, alt)]]
            e_old = enc.token_emb.weight[enc.token_index[(axis,
                                                          desc.axes[axis])]]
            direction = e_new - e_old
            cos = F.cosine_similarity((f_swap - f_orig).unsqueeze(0),
                                      direction.unsqueeze(0))
            loss = loss + torch.clamp(tau_swap - cos, min=0.0)
    return loss


def _train(enc, descs, similar, dissimilar, relations, trel, steps, lr,
           tau, rho, margin, inv_tau, schema_axes, exclude, tau_swap):
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    names = list(descs)
    history = []
    for step in range(steps):
        opt.zero_grad()
        loss = torch.zeros(())
        for a, b in similar:
            fa, fb = enc.forward(descs[a]), enc.forward(descs[b])
            logits = []
            for n in names:
                if n == a:
                    continue
                logits.append((fa * enc.forward(descs[n])).sum())
            logits = torch.stack(logits)
            loss = loss - fb.dot(fa) + torch.logsumexp(logits, dim=0)
        if dissimilar:
            cos_sim = [float(F.cosine_similarity(
                enc.forward(descs[a]).unsqueeze(0),
                enc.forward(descs[b]).unsqueeze(0)))
                for a, b in similar]
            floor = min(cos_sim)
            for a, b in dissimilar:
                cos = F.cosine_similarity(enc.forward(descs[a]).unsqueeze(0),
                                          enc.forward(descs[b]).unsqueeze(0))
                loss = loss + torch.clamp(cos - floor + margin, min=0.0)
        for name in names:
            desc = descs[name]
            axis = list(schema_axes)[step % len(schema_axes)]
            fi = enc.forward(desc)
            token = desc.axes.get(axis)
            vocab = schema_axes[axis]
            target = vocab.index(token) if token in vocab else 0
            logits_axis = enc.attr_heads[_axis_key(axis)](fi.unsqueeze(0))
            loss = loss + F.cross_entropy(
                logits_axis, torch.tensor([target], dtype=torch.long))
        loss = loss + _rel_loss(enc, relations, tau, rho)
        loss = loss + _transform_rel_loss(enc, trel, inv_tau, rho, exclude)
        if tau_swap is not None:
            loss = loss + _frozen_swap_loss(enc, descs, tau_swap)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return history


def _gates(enc, descs, relations, trel, tau, rho, inv_tau,
           exclude: list[list[str]] | None = None,
           tau_swap: float | None = None):
    exclude = exclude or []
    names = list(descs)
    fingerprints = {n: enc.fingerprint(descs[n]) for n in names}
    g1 = all(torch.equal(fingerprints[n], enc.fingerprint(descs[n]))
             for n in names)
    sim_cos = [float(F.cosine_similarity(fingerprints[a].unsqueeze(0),
                                         fingerprints[b].unsqueeze(0)))
               for a, b in M225_SIMILAR]
    dis_cos = [float(F.cosine_similarity(fingerprints[a].unsqueeze(0),
                                         fingerprints[b].unsqueeze(0)))
               for a, b in M225_DISSIMILAR]
    g2_margin = sum(sim_cos) / len(sim_cos) - sum(dis_cos) / len(dis_cos)
    g2 = bool(g2_margin >= 0.05)
    # G3: the FROZEN 12 quadruples over the frozen axes only, on the
    # six registered v0 tasks ONLY (the frozen artifact's scope)
    g3_scores = []
    for name in FROZEN_G3_TASKS:
        desc = descs[name]
        for axis in ["input.modality", "output.kind"]:
            vocab = AXES[axis]
            alt = vocab[0] if desc.axes[axis] != vocab[0] else vocab[-1]
            alt_desc = normalise({**{a: v for a, v in desc.axes.items()},
                                  axis: alt})
            f_orig, f_swap = enc.fingerprint(desc), enc.fingerprint(alt_desc)
            e_new = enc.token_emb.weight[enc.token_index[(axis, alt)]]
            e_old = enc.token_emb.weight[enc.token_index[(axis,
                                                          desc.axes[axis])]]
            direction = e_new - e_old
            g3_scores.append(float(F.cosine_similarity(
                (f_swap - f_orig).unsqueeze(0), direction.unsqueeze(0))))
    g3 = bool(min(g3_scores) >= 0.5)
    # G5: the v0 relational constraints still hold on the extended schema
    ordered_scores = []
    for axis, order in relations["ordered_axes"].items():
        vecs = [_token_vec(enc, axis, t) for t in order]
        for i in range(len(vecs) - 2):
            d1 = vecs[i + 1] - vecs[i]
            d2 = vecs[i + 2] - vecs[i + 1]
            ordered_scores.append(float(F.cosine_similarity(
                d1.unsqueeze(0), d2.unsqueeze(0))))
    polar_scores = []
    for pair in relations["polar_pairs"]:
        a = _token_vec(enc, pair["axis"], pair["a"])
        b = _token_vec(enc, pair["axis"], pair["b"])
        polar_scores.append(float(F.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0))))
    g5_ordered = bool(min(ordered_scores) >= tau)
    g5_polar = bool(max(polar_scores) <= -rho)
    # G6: the inverse-relation analogy (the user's example)
    ref = trel["reference_inverse_pair"]
    r_dir = (_token_vec(enc, trel["axis"], ref[0])
             - _token_vec(enc, trel["axis"], ref[1]))
    analogy_cos = {}
    held_out = [tuple(p) for p in exclude]
    for a, b in trel["inverse_pairs"]:
        if [a, b] == ref:
            continue
        d = _token_vec(enc, trel["axis"], a) - _token_vec(enc, trel["axis"], b)
        analogy_cos[f"{a}-{b}"] = float(F.cosine_similarity(
            d.unsqueeze(0), r_dir.unsqueeze(0)))
    g6 = bool(min(analogy_cos.values()) >= inv_tau)
    transform_polar = {}
    for a, b in trel["polar_pairs"]:
        va = _token_vec(enc, trel["axis"], a)
        vb = _token_vec(enc, trel["axis"], b)
        transform_polar[f"{a}/{b}"] = float(F.cosine_similarity(
            va.unsqueeze(0), vb.unsqueeze(0)))
    g6_polar = bool(max(transform_polar.values()) <= -rho)
    return {
        "g1_deterministic": g1,
        "g2_similarity_ordering": {"passed": g2, "margin": g2_margin,
                                   "similar_cos": sim_cos,
                                   "dissimilar_cos": dis_cos},
        "g3_traversality": {"passed": g3, "min_cos": min(g3_scores),
                            "scores": g3_scores,
                            "frozen_tasks": FROZEN_G3_TASKS},
        "g4_continuity": {"passed": None, "note": "DEFERRED"},
        "g5_relational_recall": {
            "ordered_triple_min": min(ordered_scores),
            "ordered_passed": g5_ordered,
            "polar_max": max(polar_scores),
            "polar_passed": g5_polar,
            "passed": bool(g5_ordered and g5_polar)},
        "g6_inverse_analogy": {
            "reference": f"{ref[0]}-{ref[1]}",
            "analogy_cos": analogy_cos,
            "passed": g6,
            "threshold": inv_tau,
            "held_out_pairs": [f"{a}-{b}" for a, b in held_out],
            "transform_polar_cos": transform_polar,
            "polar_passed": g6_polar},
        "g3_trained_note": (None if tau_swap is None else
                            "the frozen-axis swap directions joined the "
                            "training objective on this arm - G3 is "
                            "TRAINED-not-held-out (registered)"),
    }


def run_m225(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()

    trel = json.loads(TRANSFORM_REL_PATH.read_text(encoding="utf-8"))
    relations = json.loads(RELATIONS_PATH.read_text(encoding="utf-8"))
    schema = {**AXES, trel["axis"]: trel["vocabulary"]}

    f_dim = int(config["model"]["f_dim"])
    mlp_hidden = int(config["model"]["mlp_hidden"])
    seed = int(config["model"]["seed"])
    lr = float(config["training"]["lr"])
    steps = int(config["training"]["steps"])
    tau = float(config["training"]["rel_ordered_triple_tau"])
    rho = float(config["training"]["rel_polar_rho"])
    margin = float(config["training"]["margin_dissimilar"])
    inv_tau = float(config["training"]["inverse_analogy_tau"])
    tau_swap = config["training"].get("frozen_swap_tau")
    exclude = [list(p) for p in config.get("train_exclude", [])]

    # the v0 tasks get task.transform = identity (registered)
    descs = {}
    for name, d in TASK_DESCRIPTORS.items():
        nd = normalise({**d, "task.transform": "identity"})
        nd.axes[trel["axis"]] = "identity"  # normalise drops unknown axes
        descs[name] = nd
    for name, d in TRANSFORM_TASKS.items():
        nd = normalise(d)
        nd.axes[trel["axis"]] = d["task.transform"]
        descs[name] = nd

    enc = FingerprintEncoder(f_dim=f_dim, mlp_hidden=mlp_hidden,
                             seed=seed, axes=schema)
    enc.attr_heads = torch.nn.ModuleDict({
        _axis_key(axis): torch.nn.Linear(f_dim, len(vocab))
        for axis, vocab in schema.items()})

    print(f"training {steps} steps", flush=True)
    history = _train(enc, descs, M225_SIMILAR, M225_DISSIMILAR,
                     relations, trel, steps, lr, tau, rho, margin,
                     inv_tau, schema, exclude, tau_swap)
    print(f"  final loss {history[-1]:.4f}", flush=True)

    print("gates", flush=True)
    gates = _gates(enc, descs, relations, trel, tau, rho, inv_tau,
                   exclude, tau_swap)
    print(json.dumps(gates, indent=1)[:2000], flush=True)

    void = not gates["g1_deterministic"]
    evidence: dict[str, Any] = {
        "milestone": "M225",
        "cell": "semantics axis + inverse-relation analogies (prototype)",
        "admissible_as_evidence": True,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
        "interpretation_registered_before_running":
            config["interpretation_registered_before_running"],
        "schema_axes": schema,
        "transform_relations_used": trel,
        "gates": gates,
        "training": {"steps": steps, "final_loss": history[-1]},
        "not_shipped": "prototype only; the product ontology migration "
                       "is M226 (registered) and runs only if G1-G3+G6 "
                       "pass",
        "void": void,
        "void_reason": "G1 determinism failed" if void else "",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM225 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m225(args.config, args.output)


if __name__ == "__main__":
    main()
