"""M169 — fingerprint training and the gates G1–G4.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` (section 7
Phase B M169; the section 12 dispatch entry, 17 Aug 2026). v0 scope:

- training signals: InfoNCE over the ontology-registered known-similar
  pairs plus a CBOW-style attribute-reconstruction auxiliary (predict a
  held-out axis token from the fingerprint). The ONE measured
  behavioral-transfer label (M167a's d0->d1, +0.0092) is recorded as a
  ranking constraint only — the label set is too thin to train on
  (registered, not an assumption).
- gates: G1 determinism (void on failure); G2 similarity ordering on
  the registered pair set (scoped negative on failure — the gates are
  the measurement, not the instrument); G3 attribute-swap traversality
  (movement direction aligns with the swapped embedding difference,
  cos >= 0.5); G4 continuity DEFERRED (no sweep families exist yet).

Smoke declares inadmissibility and refuses the sealed output directory.
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
from geode.core.descriptor import AXES, normalise
from geode.core.fingerprint import FingerprintEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m169_fingerprint_train.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v24"
                  / "m169_fingerprint_gates")

# ---- the registered v0 task set (descriptor mappings) ---------------------
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
}

SIMILAR_PAIRS = [("domainnet", "cifar10"), ("mackey_glass", "lorenz")]
DISSIMILAR_PAIRS = [("domainnet", "mackey_glass"), ("domainnet", "lorenz"),
                    ("domainnet", "dyck"), ("domainnet", "tabular"),
                    ("cifar10", "mackey_glass"), ("cifar10", "dyck")]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def _axis_key(axis: str) -> str:
    return axis.replace(".", "_")


def _train(enc, descs, similar, steps, lr):
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
        # attribute reconstruction auxiliary: predict one masked axis
        for name in names:
            desc = descs[name]
            axis = list(AXES)[step % len(AXES)]
            fi = enc.forward(desc)
            token = desc.axes[axis]
            target = (AXES[axis].index(token) if token in AXES[axis]
                      else 0)  # <oov> -> index 0 (auxiliary only)
            logits_axis = enc.attr_heads[_axis_key(axis)](fi.unsqueeze(0))
            loss = loss + F.cross_entropy(
                logits_axis,
                torch.tensor([target], dtype=torch.long))
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return history


def _gates(enc, descs):
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
    # G3 attribute-swap traversality
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
    return {
        "g1_deterministic": g1,
        "g2_similarity_ordering": {"passed": g2, "margin": g2_margin,
                                   "similar_cos": sim_cos,
                                   "dissimilar_cos": dis_cos},
        "g3_traversality": {"passed": g3, "min_cos": min(g3_scores),
                            "scores": g3_scores},
        "g4_continuity": {"passed": None, "note": "DEFERRED: no sweep "
                          "families exist; registered pending"},
        "measured_transfer_label": {"note": "M167a d0->d1 +0.0092 recorded "
                                    "as a ranking constraint only — the "
                                    "label set is too thin to train on"},
    }


def run_m169(config_path: Path, output_dir: Path) -> dict[str, Any]:
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

    descs = {n: normalise(d) for n, d in TASK_DESCRIPTORS.items()}
    enc = FingerprintEncoder(f_dim=f_dim, mlp_hidden=mlp_hidden, seed=seed)
    # per-axis reconstruction heads (auxiliary only)
    enc.attr_heads = torch.nn.ModuleDict({
        _axis_key(axis): torch.nn.Linear(f_dim, len(vocab))
        for axis, vocab in AXES.items()})

    print(f"training {steps} steps", flush=True)
    history = _train(enc, descs, SIMILAR_PAIRS, steps, lr)
    print(f"  final loss {history[-1]:.4f}", flush=True)

    print("gates", flush=True)
    gates = _gates(enc, descs)
    print(json.dumps(gates, indent=1)[:1200], flush=True)

    void = not gates["g1_deterministic"]
    evidence: dict[str, Any] = {
        "milestone": "M169",
        "cell": "fingerprint training + gates G1-G4 (v0)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
        "interpretation_registered_before_running":
            config["interpretation_registered_before_running"],
        "gates": gates,
        "training": {"steps": steps, "final_loss": history[-1]},
        "void": void,
        "void_reason": "G1 determinism failed" if void else "",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM169 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m169(args.config, args.output)


if __name__ == "__main__":
    main()
