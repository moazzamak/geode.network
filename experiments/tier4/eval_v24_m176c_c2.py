"""M176c candidate 2 — Fisher vectors on frozen DINOv2-small patches.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` §12 (18 Aug
2026) as a COMPARISON arm (candidate 1 already passed the per-MAC gate).
Classic Fisher vectors (Perronnin & Sánchez 2013, cited): a diagonal-cov
GMM over the deep patch tokens, per-image first/second-order statistics,
signed sqrt + per-row L2, then the sealed closed-form intercept ridge.
Same streaming 224px decode, backbone, corpus schedule, and penalties as
candidate 1. The verdict is a comparison: deep-patch SPM 0.487/0.563/0.590
and the dense ladder (r70 0.3118) are the references; Fisher earns a
deployment consequence ONLY if it wins.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common.data_cache import configure_external_cache_environment
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    _score,
)
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v24_m176c_c1 import (
    _decode_rows,
    _load_backbone,
    _patch_tokens,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m176c_c2.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v24" / "m176c_c2")

CLASSES = 345


def _fit_gmm(sample_tokens: np.ndarray, k: int,
             seed: int, max_patches: int):
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(seed)
    patches = sample_tokens.reshape(-1, sample_tokens.shape[-1])
    take = rng.choice(len(patches), min(max_patches, len(patches)),
                      replace=False)
    gmm = GaussianMixture(n_components=k, covariance_type="diag",
                          max_iter=100, n_init=1, random_state=seed,
                          verbose=1)
    gmm.fit(patches[take])
    return gmm


def _fisher_codes(tokens: torch.Tensor, means: torch.Tensor,
                  stds: torch.Tensor, logw: torch.Tensor
                  ) -> np.ndarray:
    """Per-image Fisher vectors (first + second order), signed sqrt + L2."""
    b, g, _, d = tokens.shape
    x = tokens.reshape(b, g * g, d).float()
    k = means.shape[0]
    m = means.float().to(x.device)
    s = stds.float().to(x.device).clamp(min=1e-6)
    lw = logw.float().to(x.device)
    with torch.no_grad():
        m = m.to(x.device)
        s = s.to(x.device)
        lw = lw.to(x.device)
        outs = []
        inner = 64  # VRAM guard for the (b,196,k,d) expansion
        for a in range(0, b, inner):
            xc = x[a:a + inner]
            # log p(k|x) = logw - 0.5*sum((x-m)^2/s^2) - sum log s
            diff = xc.unsqueeze(2) - m[None, None]      # (n,196,k,d)
            sq = (diff * diff) / (s[None, None] ** 2)
            logp = (lw[None, None] - 0.5 * sq.sum(-1)
                    - torch.log(s).sum())
            logp = logp - torch.logsumexp(logp, dim=-1, keepdim=True)
            gamma = torch.exp(logp)                    # (n,196,k)
            norm = diff / s[None, None]
            fo = torch.einsum("npk,npkd->nkd", gamma, norm)      # (n,k,d)
            so = torch.einsum("npk,npkd->nkd", gamma,
                              (sq - 1.0) * (2.0 ** -0.5))
            fv = torch.cat([fo, so], dim=2).reshape(
                xc.shape[0], 2 * k * d)
            fv = fv.sign() * fv.abs().sqrt()
            fv = fv / (fv.norm(dim=1, keepdim=True) + 1e-12)
            outs.append(fv.cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def run_m176c_c2(config_path: Path, output_dir: Path,
                 k_override: int | None = None) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    configure_external_cache_environment()
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    corpus, train_index, test_index = _load_corpus(config)
    print("streaming 224px decode + backbone encode", flush=True)
    processor, model = _load_backbone(config)
    model = model.to(device)

    rep = config["sparse"]
    sample_rows = int(rep["gmm_sample_images"])
    sample_images = _decode_rows("train", train_index[:sample_rows], 224)
    sample_tokens = []
    step = int(config["numerics"]["backbone_batch"])
    for start in range(0, len(sample_images), step):
        stop = min(start + step, len(sample_images))
        sample_tokens.append(_patch_tokens(model, processor, device,
                                           sample_images[start:stop])
                             .cpu().numpy())
    sample_tokens = np.concatenate(sample_tokens, axis=0)
    del sample_images

    k_ladder = ([int(k) for k in config.get("_smoke_k", [16])]
                if smoke else [int(config["cell"]["k"])])
    if k_override is not None:
        k_ladder = [int(k_override)]
    n_train_rows = int(config["_smoke_train_rows"]) if smoke else len(
        corpus["train_labels"])
    n_test_rows = int(config["_smoke_test_rows"]) if smoke else len(
        corpus["test_labels"])
    test_idx = test_index[:n_test_rows]
    train_idx = train_index[:n_train_rows]
    block = int(config["numerics"]["block"])

    results = {}
    for k in k_ladder:
        t0 = time.time()
        print(f"fitting GMM k={k}", flush=True)
        gmm = _fit_gmm(sample_tokens, k, int(rep["gmm_seed"]),
                       int(rep["gmm_fit_patches"]))
        means = torch.from_numpy(gmm.means_.astype(np.float32))
        stds = torch.from_numpy(np.sqrt(gmm.covariances_)
                                .astype(np.float32))
        logw = torch.from_numpy(np.log(gmm.weights_)
                                .astype(np.float32))
        width = 2 * k * 384
        acc = RidgeAccumulator(width, CLASSES)
        for start in range(0, n_train_rows, block):
            stop = min(start + block, n_train_rows)
            imgs = _decode_rows("train", train_idx[start:stop], 224)
            toks = _patch_tokens(model, processor, device, imgs)
            codes = _fisher_codes(toks, means, stds, logw)
            acc.add(codes, corpus["train_labels"][start:stop])
            if start % (block * 8) == 0:
                print(f"  k {k}: train {stop}/{n_train_rows}", flush=True)
        std = acc.standardiser()
        weights = {str(p): w for p, w in
                   acc.solve_many([0.1, 1.0, 10.0]).items()}
        hits = {p: 0 for p in weights}
        n_seen = 0
        for start in range(0, n_test_rows, block):
            stop = min(start + block, n_test_rows)
            imgs = _decode_rows("test", test_idx[start:stop], 224)
            toks = _patch_tokens(model, processor, device, imgs)
            codes = _fisher_codes(toks, means, stds, logw)
            for p, w in weights.items():
                hits[p] += int(_score(w, std(codes),
                                      corpus["test_labels"][start:stop]
                                      ).sum())
            n_seen += stop - start
        del acc
        torch.cuda.empty_cache()
        results[str(k)] = {
            "accuracy_by_penalty": {p: hits[p] / n_seen for p in hits},
            "width": width,
            "encode_seconds": round(time.time() - t0, 1),
        }
        # Incremental seal: this K's completed numbers survive any later crash.
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / f"progress_k{k}.json", {
            "milestone": "M176c-c2",
            "k": k,
            "accuracy_by_penalty": results[str(k)]["accuracy_by_penalty"],
            "width": width,
            "configuration_hash": payload_hash(config),
            "config_file": Path(config_path).name,
            "admissible_as_evidence": False,
            "note": ("progress record written before the run completes; "
                     "the sealed evidence is evidence.json in this directory"),
        })

    evidence: dict[str, Any] = {
        "milestone": "M176c-c2",
        "cell": "Fisher vectors on frozen DINOv2-small patches",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
        "interpretation_registered_before_running":
            config["interpretation_registered_before_running"],
        "results": results,
        "comparison_references": {
            "deep_patch_spm": {"256": 0.4871304347826087,
                               "1024": 0.5628405797101449,
                               "2048": 0.5898550724637681},
            "dense_ladder": {"r70": 0.3118},
            "sealed_spm_frontier": 0.2786},
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps(results, indent=1), flush=True)
    print(f"M176c-c2 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", dest="k_override", type=int, default=None)
    args = parser.parse_args()
    run_m176c_c2(args.config, args.output, args.k_override)


if __name__ == "__main__":
    main()
