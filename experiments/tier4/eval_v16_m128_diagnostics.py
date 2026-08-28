"""M128 — v19 ride-along diagnostics (spectral tail & eff-rank vs atoms; margins).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v19.md`` section 5.6 and
``experiments/configs/v16/m128_diagnostics.json``.

Three explanatory diagnostics on the M126 extended-atoms codes (6144/8192/
12288/16384 at full data), re-using the M121 spectral machinery where the
dimension is feasible:

1. effective rank (participation ratio trace^2/||G_std||_F^2) vs atoms;
2. spectral tail index vs atoms — FULL eigh at 6144/8192 (M121 exact); at
   12288/16384 full eigh is infeasible (width >= 49152), so the top-1
   eigenvalue/share comes from blockwise power iteration and the tail index
   is DISCLOSED as not computed there;
3. margin statistics (the argmax object: f_true - max_other) on the test set
   for the 6144/8192 cells (cheap standard ridge re-fits).

Diagnostics are explanatory, never gates. CPU only; GEODE_CACHE_DIR must point
at the healthy F: cache. Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m128_diagnostics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

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
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v15_m107_dense import _score
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m121_spectrum import (
    _gram_and_targets,
    _spectrum,
    _standardised_ft_y,
    _standardised_gram,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m128_diagnostics.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m128_diagnostics"

ATOMS = [6144, 8192, 12288, 16384]
BLOCK = 8192
WIDE_THRESHOLD = 32768   # widths above this use the memmap/truncated path


def _stats_pass(mem: np.ndarray, labels: np.ndarray, rows: int,
                classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]:
    """Streaming colsum / sqsum / ft_y / class_count (all widths)."""
    width = mem.shape[1]
    colsum = np.zeros(width, dtype=np.float64)
    sqsum = np.zeros(width, dtype=np.float64)
    ft_y = np.zeros((width, classes), dtype=np.float64)
    class_count = np.zeros(classes, dtype=np.float64)
    onehot = np.eye(classes)
    for start in range(0, rows, BLOCK):
        stop = min(start + BLOCK, rows)
        b = np.asarray(mem[start:stop], dtype=np.float64)
        lab = labels[start:stop]
        colsum += b.sum(axis=0)
        sqsum += (b ** 2).sum(axis=0)
        ft_y += b.T @ onehot[lab]
        class_count += np.bincount(lab, minlength=classes).astype(np.float64)
    return colsum, sqsum, ft_y, class_count


def _standardise_params(colsum: np.ndarray, sqsum: np.ndarray,
                        rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centre = colsum / rows
    variance = sqsum / rows - centre ** 2
    scale = np.sqrt(np.maximum(variance, 0.0)) + 1e-8
    return centre, scale, 1.0 / scale


def _gram_std_apply(gram: np.ndarray, x: np.ndarray, colsum: np.ndarray,
                    centre: np.ndarray, inv: np.ndarray, rows: int,
                    block: int = 8192) -> np.ndarray:
    """Apply the standardized Gram G_std = D^-1 (G - c1^T - 1c^T + n cc^T) D^-1."""
    width = len(x)
    xd = inv * x
    c_x = float(centre @ xd)
    out = np.empty(width, dtype=np.float64)
    for i0 in range(0, width, block):
        i1 = min(i0 + block, width)
        out[i0:i1] = inv[i0:i1] * (gram[i0:i1] @ xd
                                   - colsum[i0:i1] * c_x
                                   - centre[i0:i1] * float(colsum @ xd)
                                   + centre[i0:i1] * rows * c_x)
    return out


def _top1_power(gram: np.ndarray, colsum: np.ndarray, centre: np.ndarray,
                inv: np.ndarray, rows: int, iters: int) -> tuple[float, float]:
    """Top eigenvalue + Rayleigh share of the standardized Gram (power method)."""
    width = len(colsum)
    rng = np.random.default_rng(7)
    v = rng.standard_normal(width)
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = _gram_std_apply(gram, v, colsum, centre, inv, rows)
        lam = float(v @ w)
        nrm = np.linalg.norm(w)
        if nrm == 0.0:
            break
        v = w / nrm
    return lam, lam


def _frob2_std(gram: np.ndarray, colsum: np.ndarray, centre: np.ndarray,
               inv: np.ndarray, rows: int, block: int = 4096) -> float:
    """||G_std||_F^2 by streaming the standardized Gram (wide cells).

    G_std[i0:i1, :] = diag(inv[i0:i1]) . (G - c1^T - 1c^T + n cc^T) . diag(inv),
    with the right diag(inv) (column scaling) folded into the three rank-1
    terms and applied explicitly to the raw row-block term.
    """
    width = len(colsum)
    c_t = centre * inv
    s = 0.0
    for i0 in range(0, width, block):
        i1 = min(i0 + block, width)
        row = gram[i0:i1]
        gs = (row * inv[None, :] - np.outer(colsum[i0:i1], c_t)
              - np.outer(centre[i0:i1], colsum * inv)
              + rows * np.outer(centre[i0:i1], c_t)) * inv[i0:i1, None]
        s += float((gs ** 2).sum())
    return s


def _full_spectrum_diag(mem: np.ndarray, labels: np.ndarray, rows: int,
                        classes: int) -> dict[str, Any]:
    """M121-exact: full eigh on the standardized Gram (width <= 32768)."""
    gram, colsum, sqsum, ft_y, class_count = _gram_and_targets(
        mem, labels, rows, classes)
    gstd, centre, scale = _standardised_gram(gram, colsum, sqsum, rows)
    ft_y_std = _standardised_ft_y(ft_y, class_count, centre, scale)
    vals, vecs = _spectrum(gstd)
    keep = vals > max(float(vals.max()) * 1e-10, 1e-12)
    vals_k = vals[keep]
    vecs_k = vecs[:, keep]
    eta = vals_k / rows
    a = vecs_k.T @ ft_y_std
    w2 = (a ** 2).sum(axis=1) / vals_k
    captured = float(w2.sum()) / rows
    positive = vals_k
    trace = float(positive.sum())
    eff_rank = float(trace ** 2 / (positive ** 2).sum()) if trace > 0 else 0.0
    tail = None
    if len(positive) >= 10:
        cut = max(10, len(positive) // 2)
        x = np.log(np.arange(1, cut + 1))
        y = np.log(positive[:cut])
        tail = float(np.polyfit(x, y, 1)[0])
    return {
        "n_eigenvalues": int(len(positive)),
        "top1": float(positive[0]),
        "top1_share": float(positive[0] / trace) if trace > 0 else 0.0,
        "top10_share": float(positive[:10].sum() / trace) if trace > 0 else 0.0,
        "effective_rank": eff_rank,
        "tail_index": tail,
        "captured_target_power_fraction": captured,
        "method": "full eigh (M121 exact)",
    }


def _truncated_spectrum_diag(gram: np.ndarray, colsum: np.ndarray,
                             sqsum: np.ndarray, rows: int,
                             iters: int) -> dict[str, Any]:
    """Wide cells: eff-rank exact, top-1 by power iteration, tail NOT computed."""
    centre, scale, inv = _standardise_params(colsum, sqsum, rows)
    trace_std = float(len(colsum) * rows)   # each standardized dim has unit var
    lam, _ = _top1_power(gram, colsum, centre, inv, rows, iters)
    frob = _frob2_std(gram, colsum, centre, inv, rows)
    eff_rank = trace_std ** 2 / frob if frob > 0 else 0.0
    return {
        "trace_std": trace_std,
        "top1": float(lam),
        "top1_share": float(lam / trace_std) if trace_std > 0 else 0.0,
        "effective_rank": eff_rank,
        "tail_index": None,
        "captured_target_power_fraction": None,
        "method": ("truncated: eff-rank exact (trace^2/Frob^2), top-1 by "
                   "power iteration, tail and captured power NOT computed "
                   "(full eigh infeasible at this width)"),
    }


def _fit_and_collect_margins(mem_train: np.ndarray, mem_test: np.ndarray,
                             labels: np.ndarray, test_labels: np.ndarray,
                             test_domains: np.ndarray, classes: int,
                             n: int) -> dict[str, Any]:
    """Standard ridge re-fit; return test-set margin statistics (argmax object)."""
    width = mem_train.shape[1]
    acc = RidgeAccumulator(width, classes)
    for start in range(0, n, 4096):
        stop = min(start + 4096, n)
        acc.add(np.asarray(mem_train[start:stop]), labels[start:stop])
    standardise = acc.standardiser()
    w = acc.solve(1.0)
    margins = []
    correct = 0
    seen = 0
    for start in range(0, len(test_labels), 4096):
        stop = min(start + 4096, len(test_labels))
        block = np.asarray(mem_test[start:stop], dtype=np.float32)
        s = standardise(block)
        scores = s @ w[:-1] + w[-1]
        lab = test_labels[start:stop]
        true = scores[np.arange(len(block)), lab]
        np.put_along_axis(scores, lab[:, None], -np.inf, axis=1)
        others = scores.max(axis=1)
        m = true - others
        correct += int((m > 0).sum())
        margins.append(m)
        seen += len(block)
    m = np.concatenate(margins)
    return {
        "accuracy": float(correct / seen),
        "margin_mean": float(m.mean()),
        "margin_median": float(np.median(m)),
        "margin_q5": float(np.percentile(m, 5)),
        "margin_q25": float(np.percentile(m, 25)),
        "margin_q75": float(np.percentile(m, 75)),
        "margin_q95": float(np.percentile(m, 95)),
        "frac_positive": float((m > 0).mean()),
    }


def run_m128(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    configure_external_cache_environment()
    iters = int(config["numerics"]["topk_power_iterations"])

    corpus_cfg = json.loads(
        (REPO_ROOT / "experiments" / "configs" / "v16" / "m116_scale.json")
        .read_text(encoding="utf-8"))
    corpus, _ti, _te = _load_corpus(corpus_cfg)
    labels = corpus["train_labels"]
    test_labels = corpus["test_labels"]
    test_domains = corpus["test_domains"]
    classes = int(labels.max()) + 1
    n = len(labels)

    cache = data_cache_root() / "v16" / "m126"
    m117 = data_cache_root() / "v16" / "m117"
    out: dict[str, Any] = {}
    for atoms in ATOMS:
        width = 4 * atoms
        # the 6144 codes live in the M117 cache dir (as in the M126 runner);
        # the extended-atoms codes live in the m126 dir
        base = m117 if atoms == 6144 else cache
        mem_train = np.lib.format.open_memmap(
            base / f"f{atoms}_train.npy", mode="r", dtype=np.float32)
        mem_test = np.lib.format.open_memmap(
            base / f"f{atoms}_test.npy", mode="r", dtype=np.float32)
        print(f"atoms {atoms} (width {width}):", flush=True)
        row: dict[str, Any] = {"width": width}
        if width <= WIDE_THRESHOLD:
            # full-spectrum path (M121 exact: gram + eigh in one pass)
            diag = _full_spectrum_diag(mem_train, labels, n, classes)
            row["spectrum"] = diag
            row["margins"] = _fit_and_collect_margins(
                mem_train, mem_test, labels, test_labels, test_domains,
                classes, n)
            print(f"  eff_rank {diag['effective_rank']:.1f}, "
                  f"tail {diag['tail_index']}, top1_share "
                  f"{diag['top1_share']:.4f}", flush=True)
        else:
            # wide path: stats pass once; reuse trace-verified gram memmap
            colsum, sqsum, ft_y, class_count = _stats_pass(
                mem_train, labels, n, classes)
            gram_path = cache / f"gram{atoms}.npy"
            sum_sq_check = float(sqsum.sum())
            reuse = (gram_path.exists()
                     and gram_path.stat().st_size >= width * width * 8)
            if reuse:
                gram = np.lib.format.open_memmap(gram_path, mode="r",
                                                 dtype=np.float64)
                tr = float(np.trace(gram))
                reuse = (abs(tr - sum_sq_check) / max(sum_sq_check, 1.0)
                         < 1e-9)
            if not reuse:
                gram = np.lib.format.open_memmap(
                    gram_path, mode="w+", dtype=np.float64,
                    shape=(width, width))
                for start in range(0, n, BLOCK):
                    stop = min(start + BLOCK, n)
                    b = np.asarray(mem_train[start:stop], dtype=np.float64)
                    for i0 in range(0, width, 8192):
                        i1 = min(i0 + 8192, width)
                        gram[i0:i1] += b[:, i0:i1].T @ b
            diag = _truncated_spectrum_diag(gram, colsum, sqsum, n, iters)
            row["spectrum"] = diag
            row["margins"] = None
            print(f"  eff_rank {diag['effective_rank']:.1f}, "
                  f"top1_share {diag['top1_share']:.4f} (truncated)",
                  flush=True)
        out[str(atoms)] = row

    evidence = {
        "milestone": "M128",
        "admissible_as_evidence": True,
        "registered_in": config.get("registered_in"),
        "question": ("do the spectral tail and effective rank track atoms, and "
                     "what are the margin statistics on the extended codes?"),
        "config_file": Path(config_path).name,
        "config": config,
        "source": "M126 sealed code memmaps (F: cache)",
        "per_atoms": out,
        "note": "explanatory diagnostics only, never gates; truncated "
                "quantities marked in the spectrum.method field",
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM128 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m128(args.config, args.output)


if __name__ == "__main__":
    main()
