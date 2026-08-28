"""
GEODE Verification Pipeline
Runs each experimental tier in sequence and reports a summarized table of results.
Intermediate verbose output is suppressed; only the summary is shown.
"""
import contextlib
import io
import os
import sys
import traceback

import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# GPU acceleration
# Set USE_GPU = True to offload SDF inference to the discrete GPU via OpenCL.
# Requires PyOpenCL and an OpenCL-capable GPU (AMD APP driver on Windows).
# Affects the final test-time prediction in Tier 4; training is CPU-only.
# ---------------------------------------------------------------------------
USE_GPU = True


def _run_silently(fn):
    """Call fn(), capturing stdout. Returns (return_value, captured_log)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        value = fn()
    return value, buf.getvalue()


def _geometry_normalized_residual_score(experts, test_points, alpha=1.0):
    """Project-specific normalized residual score: 1 - SS_res / SS_tot.

    y_true = 0 everywhere (test points are on the surface).
    SS_res = sum of squared SDF predictions.
    SS_tot = sum of squared distances of test points from their centroid
             (spatial variance of the data, used as the scale baseline).
    Returns nan when the model has no experts or the data is degenerate."""
    from src.inference_engine import InferenceEngine
    if not experts:
        return float('nan')
    pts = np.asarray(test_points, dtype=np.float64)
    sdf = InferenceEngine(experts, alpha=alpha).get_fused_sdf(pts)
    ss_res = float(np.sum(sdf ** 2))
    centroid = pts.mean(axis=0)
    ss_tot = float(np.sum(np.sum((pts - centroid) ** 2, axis=1)))
    if ss_tot < 1e-10:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def _r2_score(experts, test_points, alpha=1.0):
    """Compatibility alias for historical result consumers."""
    return _geometry_normalized_residual_score(experts, test_points, alpha)


# ---------------------------------------------------------------------------
# Per-tier runners — each returns a plain dict of metrics, or raises.
# FileNotFoundError  => dataset missing (reported as SKIPPED).
# Any other exception => reported as FAILED.
# ---------------------------------------------------------------------------

def run_tier1():
    from experiments.tier1.eval_regression import ensure_train_test_files, calculate_rmse
    from experiments.common.moe_eval import run_cv_with_fixed_train_test

    CAPTURE_THRESHOLD = 0.1
    shapes = [
        ("sphere",    np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        ("ellipsoid", np.array([3.0, 0.0, 0.0]), np.array([1.5, 0.8, 1.2])),
    ]
    results = []
    for base_name, gt_center, gt_radii in shapes:
        train, test = ensure_train_test_files(base_name)
        raw, _ = _run_silently(
            lambda tr=train, te=test: run_cv_with_fixed_train_test(
                train_points=tr, test_points=te, n_splits=5, seed=42,
                consensus_threshold=0.8, capture_threshold=CAPTURE_THRESHOLD,
                alpha=1.0, max_iterations=600, nudge_iterations=50,
                nudge_learning_rate=0.01, use_gpu=USE_GPU,
            )
        )
        if not raw["final_experts"]:
            results.append((base_name.capitalize(), None))
            continue
        best_expert = raw["final_experts"][0]
        r = {
            "cv_mean_error": raw["cv_mean_error"],
            "cv_std_error": raw["cv_std_error"],
            "test_error": raw["test_error"],
            "center_rmse": calculate_rmse(gt_center, best_expert.center),
            "radii_rmse": calculate_rmse(gt_radii, best_expert.radii),
            "n_experts": len(raw["final_experts"]),
            "_final_experts": raw["final_experts"],
            "_test_points": raw["test_points"],
            "_capture_threshold": CAPTURE_THRESHOLD,
        }
        results.append((base_name.capitalize(), r))
    return results


def run_tier2():
    from experiments.tier2.eval_pointcloud_reconstruction import load_modelnet_pointclouds
    from experiments.common.moe_eval import run_cv_then_test

    points = load_modelnet_pointclouds("data/tier2/modelnet10_pointclouds.npz", max_shapes=32)
    r, _ = _run_silently(lambda: run_cv_then_test(
        points=points,
        test_fraction=0.2, n_splits=5, seed=42,
        consensus_threshold=0.15, capture_threshold=0.08, alpha=1.0,
        max_iterations=3000, nudge_iterations=30, nudge_learning_rate=0.02,
        use_gpu=USE_GPU,
    ))
    return r


def run_tier3():
    from experiments.tier3.eval_mnist_manifold import load_mnist_digit_subset
    from experiments.common.moe_eval import run_cv_then_test

    # Use a small sample limit so the verify run completes quickly.
    points = load_mnist_digit_subset(digit=0, limit=500, pca_components=3, random_seed=42)
    r, _ = _run_silently(lambda: run_cv_then_test(
        points=points,
        test_fraction=0.2, n_splits=5, seed=42,
        consensus_threshold=0.12, capture_threshold=0.08, alpha=1.0,
        max_iterations=300, nudge_iterations=25, nudge_learning_rate=0.015,
        use_gpu=USE_GPU,
    ))
    return r


def run_tier4():
    from experiments.tier4.eval_complex_classification import (
        load_cifar_npz, run_cv_and_test_classification,
    )

    X, y = load_cifar_npz(
        dataset_path="data/tier4/cifar10_features.npz",
        max_samples=7500, pca_components=128, seed=42,
        feature_extractor="cnn",
    )
    r, _ = _run_silently(
        lambda: run_cv_and_test_classification(
            X=X, y=y, seed=42, n_splits=5, pca_components=128, use_gpu=USE_GPU
        )
    )
    return r


def run_tier5():
    from experiments.tier5.eval_cifar100_superclass import (
        load_cifar100_npz, run_cv_and_test_classification as _run_t5,
    )

    # 15 000 samples → ~750/class → ~600 final-train/class >> k_size=209 (d=19)
    X, y = load_cifar100_npz(
        dataset_path="data/tier5/cifar100_superclass.npz",
        max_samples=15000, pca_components=128, seed=42,
    )
    r, _ = _run_silently(
        lambda: _run_t5(
            X=X, y=y, seed=42, n_splits=5, pca_components=128,
            consensus_threshold=0.06,   # lower → more experts per multi-modal superclass
            capture_threshold=0.08,
            alpha=2.0,
            max_iterations=2500,        # more candidates → better diversity at d=19
            use_gpu=USE_GPU,
        )
    )
    return r


def run_tier6():
    from experiments.tier6.eval_temporal_text_prediction import (
        run_text_prediction_experiment as _run_t6,
    )
    # Static quick-check. Keep EM and subtractive CSG disabled until the base
    # classifier beats the unigram baseline reliably.
    r, _ = _run_silently(
        lambda: _run_t6(
            dataset="wikitext103",
            max_chars=None,
            max_train_samples=50_000,
            max_test_samples=10_000,
            window=3,
            representation="temporal_state",
            temporal_state_dim=16,
            temporal_warmup=64,
            pca_components=8,
            n_folds=2,
            alpha=2.0,
            consensus_threshold=0.06,
            capture_threshold=0.08,
            n_refinement_iters=0,
            n_refinement_epochs=10,
            refinement_lr=0.1,
            use_subtractive=False,
            seed=42,
            use_gpu=USE_GPU,
        )
    )
    return r


def _tier6_regression_error(entry: dict) -> str | None:
    """Return a Tier 6 failure reason, or None when its metrics are coherent."""
    test_acc = entry.get("test_acc")
    unigram_acc = entry.get("unigram_acc")
    if test_acc is not None and unigram_acc is not None and test_acc < unigram_acc:
        return (
            f"Regression: test_acc={test_acc*100:.2f}% is below unigram "
            f"baseline={unigram_acc*100:.2f}%."
        )
    initial_acc = entry.get("test_acc_init")
    refined_acc = entry.get("test_acc_refined")
    if initial_acc is not None and refined_acc is not None and refined_acc < initial_acc - 0.001:
        return (
            f"Refinement regression: accuracy fell from {initial_acc*100:.2f}% "
            f"to {refined_acc*100:.2f}%."
        )
    perplexity = entry.get("ppl_init")
    if perplexity is not None and not np.isfinite(perplexity):
        return "Regression: perplexity is not finite."
    if entry.get("class_count", 0) < 2:
        return "Regression: fewer than two classes were modeled."
    return None


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def _prepare_datasets():
    """
    Download and cache any missing datasets before running the tiers.
    Failures here are non-fatal: the corresponding tier will be SKIPPED.
    """
    from experiments.common.dataset_utils import (
        MODELNET10_NPZ, CIFAR10_NPZ, CIFAR100_NPZ,
        prepare_modelnet10, prepare_cifar10, prepare_cifar100,
    )

    missing = [
        p for p in (MODELNET10_NPZ, CIFAR10_NPZ, CIFAR100_NPZ)
        if not os.path.exists(p)
    ]
    if missing:
        print("\n[Setup] Preparing missing datasets...")

    if not os.path.exists(MODELNET10_NPZ):
        print("[Setup] Tier 2 -- ModelNet10 not found, downloading...")
        try:
            prepare_modelnet10(output_path=MODELNET10_NPZ)
        except Exception as exc:
            print(f"[Setup] Tier 2 download failed: {exc}")

    if not os.path.exists(CIFAR10_NPZ):
        print("[Setup] Tier 4 -- CIFAR-10 not found, downloading...")
        try:
            prepare_cifar10(output_path=CIFAR10_NPZ)
        except Exception as exc:
            print(f"[Setup] Tier 4 download failed: {exc}")

    if not os.path.exists(CIFAR100_NPZ):
        print("[Setup] Tier 5 -- CIFAR-100 not found, downloading...")
        try:
            prepare_cifar100(output_path=CIFAR100_NPZ)
        except Exception as exc:
            print(f"[Setup] Tier 5 download failed: {exc}")


def main():
    print("=" * 60)
    print("         GEODE Verification Pipeline")
    print("=" * 60)

    _prepare_datasets()

    summary = []

    # ---- Tier 1 ----
    print("\n[Tier 1] Geometry Regression ...", end=" ", flush=True)
    _t0 = time.perf_counter()
    try:
        tier1_results = run_tier1()
        _elapsed = time.perf_counter() - _t0
        print(f"done ({_elapsed:.1f}s).")
        for name, r in tier1_results:
            if r is None:
                summary.append({
                    "tier": 1, "name": name, "status": "failed",
                    "error": "No experts found on final train split.",
                    "elapsed": _elapsed,
                })
            else:
                summary.append({
                    "tier": 1, "name": name, "status": "passed",
                    "cv_mean_error": r["cv_mean_error"],
                    "cv_std_error": r["cv_std_error"],
                    "test_error": r["test_error"],
                    "center_rmse": r.get("center_rmse"),
                    "radii_rmse": r.get("radii_rmse"),
                    "n_experts": r.get("n_experts"),
                    "geometry_normalized_residual_score": (
                        _geometry_normalized_residual_score(
                            r["_final_experts"], r["_test_points"],
                        )
                    ),
                    "r2_score": _r2_score(r["_final_experts"], r["_test_points"]),
                    "elapsed": _elapsed,
                })
    except FileNotFoundError as e:
        _elapsed = time.perf_counter() - _t0
        print("skipped.")
        summary.append({
            "tier": 1, "name": "Geometry", "status": "skipped",
            "error": str(e).splitlines()[0],
            "elapsed": _elapsed,
        })
    except Exception:
        _elapsed = time.perf_counter() - _t0
        print("failed.")
        summary.append({
            "tier": 1, "name": "Geometry", "status": "failed",
            "error": traceback.format_exc(),
            "elapsed": _elapsed,
        })

    # ---- Tier 2 ----
    print("[Tier 2] Point-Cloud Reconstruction ...", end=" ", flush=True)
    _t0 = time.perf_counter()
    try:
        r = run_tier2()
        _elapsed = time.perf_counter() - _t0
        print(f"done ({_elapsed:.1f}s).")
        from experiments.common.geometry_metrics import symmetric_chamfer_distance
        primitive_count = sum(
            len(expert.ellipsoids) for expert in r["final_experts"]
        )
        samples_per_primitive = max(8, min(64, 2048 // max(primitive_count, 1)))
        geometry_score = _geometry_normalized_residual_score(
            r["final_experts"], r["test_points"],
        )
        summary.append({
            "tier": 2, "name": "ModelNet Point-Cloud", "status": "passed",
            "cv_mean_error": r["cv_mean_error"],
            "cv_std_error": r["cv_std_error"],
            "test_error": r["test_error"],
            "n_experts": len(r["final_experts"]),
            "geometry_normalized_residual_score": geometry_score,
            "r2_score": geometry_score,
            "chamfer_distance": symmetric_chamfer_distance(
                r["final_experts"], r["test_points"],
                samples_per_ellipsoid=samples_per_primitive,
                seed=42,
                projection_steps=8,
            ),
            "elapsed": _elapsed,
        })
    except FileNotFoundError as e:
        _elapsed = time.perf_counter() - _t0
        print("skipped.")
        summary.append({
            "tier": 2, "name": "ModelNet Point-Cloud", "status": "skipped",
            "error": str(e).splitlines()[0],
            "elapsed": _elapsed,
        })
    except Exception:
        _elapsed = time.perf_counter() - _t0
        print("failed.")
        summary.append({
            "tier": 2, "name": "ModelNet Point-Cloud", "status": "failed",
            "error": traceback.format_exc(),
            "elapsed": _elapsed,
        })

    # ---- Tier 3 ----
    print("[Tier 3] MNIST digit-0 Manifold Fitting ...", end=" ", flush=True)
    _t0 = time.perf_counter()
    try:
        r = run_tier3()
        _elapsed = time.perf_counter() - _t0
        print(f"done ({_elapsed:.1f}s).")
        summary.append({
            "tier": 3, "name": "MNIST digit=0", "status": "passed",
            "cv_mean_error": r["cv_mean_error"],
            "cv_std_error": r["cv_std_error"],
            "test_error": r["test_error"],
            "n_experts": len(r["final_experts"]),
            "geometry_normalized_residual_score": (
                _geometry_normalized_residual_score(
                    r["final_experts"], r["test_points"],
                )
            ),
            "r2_score": _r2_score(r["final_experts"], r["test_points"]),
            "elapsed": _elapsed,
        })
    except Exception:
        _elapsed = time.perf_counter() - _t0
        print("failed.")
        summary.append({
            "tier": 3, "name": "MNIST digit=0", "status": "failed",
            "error": traceback.format_exc(),
            "elapsed": _elapsed,
        })

    # ---- Tier 4 ----
    print(
        "[Tier 4] CIFAR-10 Classification (pretrained MobileNetV2 features) ...",
        end=" ", flush=True,
    )
    _t0 = time.perf_counter()
    try:
        r = run_tier4()
        _elapsed = time.perf_counter() - _t0
        print(f"done ({_elapsed:.1f}s).")
        summary.append({
            "tier": 4, "name": "CIFAR-10 Classification", "status": "passed",
            "cv_mean_acc": r["cv_mean_acc"],
            "cv_std_acc": r["cv_std_acc"],
            "test_acc": r["test_acc"],
            "n_experts": r["n_experts"],
            "class_count": r["class_count"],
            "elapsed": _elapsed,
        })
    except FileNotFoundError as e:
        _elapsed = time.perf_counter() - _t0
        print("skipped.")
        summary.append({
            "tier": 4, "name": "CIFAR-10 Classification", "status": "skipped",
            "error": str(e).splitlines()[0],
            "elapsed": _elapsed,
        })
    except Exception:
        _elapsed = time.perf_counter() - _t0
        print("failed.")
        summary.append({
            "tier": 4, "name": "CIFAR-10 Classification", "status": "failed",
            "error": traceback.format_exc(),
            "elapsed": _elapsed,
        })

    # ---- Tier 5 ----
    print("[Tier 5] CIFAR-100 Superclass Classification ...", end=" ", flush=True)
    _t0 = time.perf_counter()
    try:
        r = run_tier5()
        _elapsed = time.perf_counter() - _t0
        print(f"done ({_elapsed:.1f}s).")
        summary.append({
            "tier": 5, "name": "CIFAR-100 Superclass (20-class)", "status": "passed",
            "cv_mean_acc": r["cv_mean_acc"],
            "cv_std_acc":  r["cv_std_acc"],
            "test_acc":    r["test_acc"],
            "n_experts":   r["n_experts"],
            "class_count": r["class_count"],
            "elapsed": _elapsed,
        })
    except FileNotFoundError as e:
        _elapsed = time.perf_counter() - _t0
        print("skipped.")
        summary.append({
            "tier": 5, "name": "CIFAR-100 Superclass (20-class)", "status": "skipped",
            "error": str(e).splitlines()[0],
            "elapsed": _elapsed,
        })
    except Exception:
        _elapsed = time.perf_counter() - _t0
        print("failed.")
        summary.append({
            "tier": 5, "name": "CIFAR-100 Superclass (20-class)", "status": "failed",
            "error": traceback.format_exc(),
            "elapsed": _elapsed,
        })

    # ---- Tier 6 ----
    print("[Tier 6] Temporal Text Prediction (next-char) ...", end=" ", flush=True)
    _t0 = time.perf_counter()
    try:
        r = run_tier6()
        _elapsed = time.perf_counter() - _t0
        print(f"done ({_elapsed:.1f}s).")
        summary.append({
            "tier": 6, "name": "Temporal Text Prediction (char-level)", "status": "passed",
            "cv_mean_acc":  r["cv_acc_mean"],
            "cv_std_acc":   r["cv_acc_std"],
            "test_acc":     r["test_acc_final"],
            "test_acc_init": r["test_acc_init"],
            "test_acc_raw": r.get("test_acc_raw"),
            "test_acc_refined": r.get("test_acc_refined"),
            "ppl_final":    r.get("ppl_final"),
            "class_count":  r["class_count"],
            "unigram_acc":  r.get("unigram_acc"),
            "ngram_acc":    r.get("ngram_acc"),
            "linear_acc":   r.get("linear_acc"),
            "elapsed": _elapsed,
        })
    except ImportError as e:
        _elapsed = time.perf_counter() - _t0
        print("skipped (datasets not installed).")
        summary.append({
            "tier": 6, "name": "Temporal Text Prediction (char-level)", "status": "skipped",
            "error": str(e).splitlines()[0], "elapsed": _elapsed,
        })
    except Exception:
        _elapsed = time.perf_counter() - _t0
        print("failed.")
        summary.append({
            "tier": 6, "name": "Temporal Text Prediction (char-level)", "status": "failed",
            "error": traceback.format_exc(), "elapsed": _elapsed,
        })

    # ---- Results summary ----
    for entry in summary:
        error = _tier6_regression_error(entry) if entry["tier"] == 6 else None
        if entry["status"] == "passed" and error is not None:
            entry["status"] = "failed"
            entry["error"] = error

    print("\n" + "=" * 60)
    print("                  RESULTS SUMMARY")
    print("=" * 60)

    n_passed = n_failed = n_skipped = 0
    for entry in summary:
        status = entry["status"]
        if status == "passed":
            n_passed += 1
            icon = "[PASS]"
        elif status == "failed":
            n_failed += 1
            icon = "[FAIL]"
        else:
            n_skipped += 1
            icon = "[SKIP]"

        print(f"\n{icon}  [Tier {entry['tier']}] {entry['name']}")

        if status == "passed":
            if "cv_mean_error" in entry:
                print(f"     CV MAE(|SDF|) : {entry['cv_mean_error']:.6f} +/- {entry['cv_std_error']:.6f}")
                print(f"     Test MAE(|SDF|): {entry['test_error']:.6f}")
            if entry.get("center_rmse") is not None:
                print(f"     Center RMSE    : {entry['center_rmse']:.6f}")
                print(f"     Radii RMSE     : {entry['radii_rmse']:.6f}")
            if entry.get("n_experts") is not None:
                print(f"     Experts fitted : {entry['n_experts']}")
            if "cv_mean_acc" in entry:
                print(f"     CV Accuracy    : {entry['cv_mean_acc']*100:.2f}% +/- {entry['cv_std_acc']*100:.2f}%")
                print(f"     Test Accuracy  : {entry['test_acc']*100:.2f}%")
                if entry.get("test_acc_refined") is not None:
                    delta = (
                        entry["test_acc_refined"] - entry["test_acc_init"]
                    ) * 100
                    sign = "+" if delta >= 0 else ""
                    print(
                        "     After refinement: "
                        f"{entry['test_acc_refined']*100:.2f}% "
                        f"({sign}{delta:.2f}pp)"
                    )
                if entry.get("ppl_final") is not None:
                    print(f"     Perplexity     : {entry['ppl_final']:.2f}")
                if entry.get("unigram_acc") is not None:
                    print(f"     Unigram base   : {entry['unigram_acc']*100:.2f}%")
                if entry.get("ngram_acc") is not None:
                    print(f"     N-gram base    : {entry['ngram_acc']*100:.2f}%")
                if entry.get("linear_acc") is not None:
                    print(f"     Linear control : {entry['linear_acc']*100:.2f}%")
                print(f"     Classes modeled: {entry['class_count']}")
            if "geometry_normalized_residual_score" in entry:
                score = entry["geometry_normalized_residual_score"]
                val = f"{score:.4f}" if np.isfinite(score) else "N/A"
                print(f"     Geometry score : {val}")
            if "chamfer_distance" in entry:
                print(f"     Chamfer (sq.)  : {entry['chamfer_distance']:.6f}")
        elif status == "failed":
            err = entry.get("error", "unknown error")
            short = err.strip().splitlines()[-1] if "\n" in err else err
            print(f"     Error: {short[:100]}")
        else:
            print(f"     Reason: {entry.get('error', 'dataset not found')}")
        if "elapsed" in entry:
            print(f"     Time           : {entry['elapsed']:.1f}s")

    # Deduplicate elapsed by tier (Tier 1 has two shape entries sharing one elapsed).
    _seen_tiers: set = set()
    _total_time = 0.0
    for _e in summary:
        if _e["tier"] not in _seen_tiers:
            _seen_tiers.add(_e["tier"])
            _total_time += _e.get("elapsed", 0.0)

    print()
    print("-" * 60)
    total = len(summary)
    print(f"  Total: {total}  |  Passed: {n_passed}  |  Failed: {n_failed}  |  Skipped: {n_skipped}")
    print(f"  Wall-clock total: {_total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()