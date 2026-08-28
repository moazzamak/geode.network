"""M224 - fingerprint v1 training and the gates G1-G5.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (M224
REGISTERED, 20 Aug 2026, before the build). v1 scope, on the FROZEN
v0 ontology (12 axes, no schema change):

- training signals: InfoNCE over ALL registered similar pairs (three,
  the v0 used two - the domainnet-sketch/real pair is added with
  registered descriptors), a margin ranking loss over the registered
  dissimilar pairs as hard negatives, the CBOW attribute-reconstruction
  auxiliary, and relational constraints over the TOKEN embeddings from
  the authored analytical set ``analysis/fingerprint_relations_v0.json``
  (ordered-axis consecutive triples + the stationary/non-stationary
  polar pair).
- gates: G1 determinism (void on failure); G2 similarity ordering on
  the registered pair set (margin >= 0.05, scoped negative on failure);
  G3 the FROZEN 12-quadruple traversability set at 0.5 (regression vs
  M169 v0); G4 continuity DEFERRED; G5 NEW relational recall over the
  token embeddings (ordered-triple cos >= 0.5, polar cos <= -0.3).
- artifact: the trained state_dict (token_emb + mlp only, no auxiliary
  heads) is written to the output dir as fingerprint_v1.pt with a
  manifest; the PRODUCT shipping into geode/core/assets happens only
  if G1-G3 pass (registered ship protocol).

Smoke declares inadmissibility and refuses the sealed output directory.
"""
from __future__ import annotations

import argparse
import hashlib
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
from geode.core.descriptor import AXES, normalise
from geode.core.fingerprint import FingerprintEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m224_fingerprint_v1_train.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m224_fingerprint_v1_train")
RELATIONS_PATH = REPO_ROOT / "analysis" / "fingerprint_relations_v0.json"

# ---- the registered v1 task set (v0 set + the sketch/real pair) ----------
TASK_DESCRIPTORS = {
    "domainnet": {"input.modality": "image",
                  "input.submodality": "camera-RGB",
                  "input.value_kind": "continuous",
                  "input.temporal_structure": "iid",
                  "output.kind": "class", "output.ordinality": "nominal",
                  "latent.recurrence": "none",
                  "latent.stationarity": "stationary",
                  "latent.noise_regime": "medium",
                  "latent.label_cardinality": 345,
                  "latent.sample_regime": "large",
                  "coupling": "single-task"},
    "cifar10": {"input.modality": "image",
                "input.submodality": "camera-RGB",
                "input.value_kind": "continuous",
                "input.temporal_structure": "iid",
                "output.kind": "class", "output.ordinality": "nominal",
                "latent.recurrence": "none",
                "latent.stationarity": "stationary",
                "latent.noise_regime": "low",
                "latent.label_cardinality": 10,
                "latent.sample_regime": "medium",
                "coupling": "single-task"},
    "mackey_glass": {"input.modality": "numeric-series",
                     "input.value_kind": "continuous",
                     "input.temporal_structure": "sequential",
                     "output.kind": "regression",
                     "output.ordinality": "cardinal",
                     "latent.recurrence": "chaotic",
                     "latent.stationarity": "non-stationary",
                     "latent.noise_regime": "low",
                     "latent.label_cardinality": 1,
                     "latent.sample_regime": "medium",
                     "coupling": "single-task"},
    "lorenz": {"input.modality": "numeric-series",
               "input.value_kind": "continuous",
               "input.temporal_structure": "sequential",
               "output.kind": "regression",
               "output.ordinality": "cardinal",
               "latent.recurrence": "chaotic",
               "latent.stationarity": "non-stationary",
               "latent.noise_regime": "low",
               "latent.label_cardinality": 1,
               "latent.sample_regime": "medium",
               "coupling": "single-task"},
    "dyck": {"input.modality": "token-text",
             "input.value_kind": "discrete",
             "input.temporal_structure": "sequential",
             "output.kind": "next-token",
             "output.ordinality": "cardinal",
             "latent.recurrence": "grammar-depth",
             "latent.stationarity": "stationary",
             "latent.noise_regime": "low",
             "latent.label_cardinality": 2,
             "latent.sample_regime": "medium",
             "coupling": "single-task"},
    "tabular": {"input.modality": "tabular",
                "input.value_kind": "mixed",
                "input.temporal_structure": "iid",
                "output.kind": "regression",
                "output.ordinality": "cardinal",
                "latent.recurrence": "none",
                "latent.stationarity": "stationary",
                "latent.noise_regime": "medium",
                "latent.label_cardinality": 1,
                "latent.sample_regime": "small",
                "coupling": "single-task"},
    "domainnet_real": {"input.modality": "image",
                       "input.submodality": "camera-RGB",
                       "input.value_kind": "continuous",
                       "input.temporal_structure": "iid",
                       "output.kind": "class", "output.ordinality": "nominal",
                       "latent.recurrence": "none",
                       "latent.stationarity": "stationary",
                       "latent.noise_regime": "medium",
                       "latent.label_cardinality": 345,
                       "latent.sample_regime": "large",
                       "coupling": "single-task"},
    "domainnet_sketch": {"input.modality": "image",
                         "input.submodality": "none",
                         "input.value_kind": "continuous",
                         "input.temporal_structure": "iid",
                         "output.kind": "class",
                         "output.ordinality": "nominal",
                         "latent.recurrence": "none",
                         "latent.stationarity": "stationary",
                         "latent.noise_regime": "medium",
                         "latent.label_cardinality": 345,
                         "latent.sample_regime": "large",
                         "coupling": "single-task"},
}

SIMILAR_PAIRS = [("domainnet", "cifar10"), ("mackey_glass", "lorenz"),
                 ("domainnet_real", "domainnet_sketch")]
DISSIMILAR_PAIRS = [("domainnet", "mackey_glass"), ("domainnet", "lorenz"),
                    ("domainnet", "dyck"), ("domainnet", "tabular"),
                    ("cifar10", "mackey_glass"), ("cifar10", "dyck")]


def _axis_key(axis: str) -> str:
    return axis.replace(".", "_")


def _token_vec(enc: FingerprintEncoder, axis: str, token: str
               ) -> torch.Tensor:
    idx = enc.token_index[(axis, token)]
    return enc.token_emb.weight[idx]


def _rel_loss(enc: FingerprintEncoder, relations: dict[str, Any],
              tau: float, rho: float) -> torch.Tensor:
    """The relational hinge losses. Returns a TENSOR - the run-1 defect
    was returning float(loss), which detaches the term from the graph so
    the constraints contributed zero gradient (registered fix)."""
    loss = torch.zeros(())
    for axis, order in relations["ordered_axes"].items():
        vecs = [_token_vec(enc, axis, t) for t in order]
        for i in range(len(vecs) - 2):
            d1 = vecs[i + 1] - vecs[i]
            d2 = vecs[i + 2] - vecs[i + 1]
            cos = F.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0))
            loss = loss + torch.clamp(tau - cos, min=0.0)
    for pair in relations["polar_pairs"]:
        a = _token_vec(enc, pair["axis"], pair["a"])
        b = _token_vec(enc, pair["axis"], pair["b"])
        cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
        loss = loss + torch.clamp(cos + rho, min=0.0)
    return loss


def _train(enc, descs, similar, dissimilar, relations, steps, lr,
           tau, rho, margin):
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    names = list(descs)
    history = []
    for step in range(steps):
        opt.zero_grad()
        loss = torch.zeros(())
        # InfoNCE over the similar pairs (negatives = the other tasks)
        for a, b in similar:
            fa, fb = enc.forward(descs[a]), enc.forward(descs[b])
            logits = []
            for n in names:
                if n == a:
                    continue
                logits.append((fa * enc.forward(descs[n])).sum())
            logits = torch.stack(logits)
            loss = loss - fb.dot(fa) + torch.logsumexp(logits, dim=0)
        # margin ranking over the dissimilar pairs as hard negatives
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
        # CBOW-style attribute reconstruction auxiliary
        for name in names:
            desc = descs[name]
            axis = list(AXES)[step % len(AXES)]
            fi = enc.forward(desc)
            token = desc.axes[axis]
            target = (AXES[axis].index(token) if token in AXES[axis]
                      else 0)  # <oov> -> index 0 (auxiliary only)
            logits_axis = enc.attr_heads[_axis_key(axis)](fi.unsqueeze(0))
            loss = loss + F.cross_entropy(
                logits_axis, torch.tensor([target], dtype=torch.long))
        # relational constraints over the TOKEN embeddings
        loss = loss + _rel_loss(enc, relations, tau, rho)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return history


def _gates(enc, descs, relations, tau, rho):
    names = list(descs)
    fingerprints = {n: enc.fingerprint(descs[n]) for n in names}
    # G1 determinism
    g1 = all(torch.equal(fingerprints[n], enc.fingerprint(descs[n]))
             for n in names)
    # G2 similarity ordering
    sim_cos = [float(F.cosine_similarity(fingerprints[a].unsqueeze(0),
                                         fingerprints[b].unsqueeze(0)))
               for a, b in SIMILAR_PAIRS]
    dis_cos = [float(F.cosine_similarity(fingerprints[a].unsqueeze(0),
                                         fingerprints[b].unsqueeze(0)))
               for a, b in DISSIMILAR_PAIRS]
    g2_margin = sum(sim_cos) / len(sim_cos) - sum(dis_cos) / len(dis_cos)
    g2 = bool(g2_margin >= 0.05)
    # G3 attribute-swap traversality (the FROZEN 12 quadruples)
    g3_scores = []
    for name in names:
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
    # G5 relational recall over the token embeddings
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
    return {
        "g1_deterministic": g1,
        "g2_similarity_ordering": {"passed": g2, "margin": g2_margin,
                                   "similar_cos": sim_cos,
                                   "dissimilar_cos": dis_cos},
        "g3_traversality": {"passed": g3, "min_cos": min(g3_scores),
                            "scores": g3_scores},
        "g4_continuity": {"passed": None, "note": "DEFERRED: no sweep "
                          "families exist; registered pending"},
        "g5_relational_recall": {
            "ordered_triple_cos": ordered_scores,
            "ordered_triple_min": min(ordered_scores),
            "ordered_passed": g5_ordered,
            "ordered_threshold": tau,
            "polar_cos": polar_scores,
            "polar_max": max(polar_scores),
            "polar_passed": g5_polar,
            "polar_threshold": rho,
            "passed": bool(g5_ordered and g5_polar)},
    }


def run_m224(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    smoke = inadmissible
    steps = int(config["training"]["steps"])
    if smoke:
        steps = min(steps, int(config.get("_smoke_steps", 10)))

    configure_external_cache_environment()
    f_dim = int(config["model"]["f_dim"])
    mlp_hidden = int(config["model"]["mlp_hidden"])
    seed = int(config["model"]["seed"])
    lr = float(config["training"]["lr"])
    tau = float(config["training"]["rel_ordered_triple_tau"])
    rho = float(config["training"]["rel_polar_rho"])
    margin = float(config["training"]["margin_dissimilar"])

    relations = json.loads(RELATIONS_PATH.read_text(encoding="utf-8"))

    descs = {n: normalise(d) for n, d in TASK_DESCRIPTORS.items()}
    enc = FingerprintEncoder(f_dim=f_dim, mlp_hidden=mlp_hidden, seed=seed)
    enc.attr_heads = torch.nn.ModuleDict({
        _axis_key(axis): torch.nn.Linear(f_dim, len(vocab))
        for axis, vocab in AXES.items()})

    print(f"training {steps} steps", flush=True)
    history = _train(enc, descs, SIMILAR_PAIRS, DISSIMILAR_PAIRS,
                     relations, steps, lr, tau, rho, margin)
    print(f"  final loss {history[-1]:.4f}", flush=True)

    print("gates", flush=True)
    gates = _gates(enc, descs, relations, tau, rho)
    print(json.dumps(gates, indent=1)[:1600], flush=True)

    void = not gates["g1_deterministic"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # the trained state_dict artifact (token_emb + mlp only - the
    # auxiliary heads are training-only) with a byte-hash manifest
    state = {"token_emb.weight": enc.token_emb.weight.detach().clone()}
    for i, layer in enumerate(enc.mlp):
        for name, p in layer.named_parameters():
            state[f"mlp.{i}.{name}"] = p.detach().clone()
    pt_path = output_dir / "fingerprint_v1.pt"
    torch.save(state, pt_path)
    pt_bytes = pt_path.read_bytes()
    manifest = {
        "artifact": "fingerprint_v1.pt",
        "sha256": hashlib.sha256(pt_bytes).hexdigest(),
        "size_bytes": len(pt_bytes),
        "model": {"f_dim": f_dim, "mlp_hidden": mlp_hidden, "seed": seed},
        "frozen_axes_schema": AXES,
        "training": {"steps": steps, "final_loss": history[-1]},
        "note": "ship into geode/core/assets ONLY if G1-G3 pass "
                "(registered ship protocol)",
    }
    write_canonical_json(output_dir / "fingerprint_v1_manifest.json",
                         manifest)

    evidence: dict[str, Any] = {
        "milestone": "M224",
        "cell": "fingerprint v1 training + gates G1-G5",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
        "interpretation_registered_before_running":
            config["interpretation_registered_before_running"],
        "relations_file": RELATIONS_PATH.name,
        "relations_used": relations,
        "gates": gates,
        "training": {"steps": steps, "final_loss": history[-1]},
        "artifact": manifest,
        "void": void,
        "void_reason": "G1 determinism failed" if void else "",
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM224 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m224(args.config, args.output)


if __name__ == "__main__":
    main()
