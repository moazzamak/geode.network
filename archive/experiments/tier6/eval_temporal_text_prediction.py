"""Tier 6: Temporal Text Prediction with geometric classification.

Validates the hypothesis that GEODE can learn next-character prediction
through causal sampling and supervised geometric refinement:

    Initial fit:   RANSAC/greedy constructor on (context_window, next_char) pairs
    Refinement:    SDFOptimizer updates additive expert parameters on fresh
                   supervised (context, next_char) pairs

Task:      Character-level next-character prediction (language modelling)
Dataset:   Wikipedia EN via HuggingFace streaming  (~3 GB of text)
           WikiText-103 as lightweight alternative  (~100 MB, fast download)
Metric:    Top-1 accuracy, Top-5 accuracy, Perplexity, vs. n-gram baselines
Regime:    K-fold cross-validation on sequential text folds (no future leakage)
           Final evaluation on a fully held-out test portion

Preprocessing pipeline
----------------------
raw text
  → restrict to printable ASCII (96 chars incl. space) + <unk>
  → encode as integer char-ID sequence
  → sample N random context windows of `window` chars
  → one-hot expand: (N, window × vocab_size)  ← sparse, never materialised fully
  → PCA (fit on train only) → (N, pca_components)   dense
  → LDA (fit on train only) → (N, lda_components)   discriminative subspace
  → GEODE greedy constructor per character class
    → [optional] bounded supervised SDFOptimizer refinement

The LDA step reduces context dimensionality to d = min(pca_components, vocab_size-1)
which keeps k_size = d(d+3)/2 small enough that each class has many more training
samples than required for stable expert fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import warnings
from collections.abc import Mapping
from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.common.moe_eval import fit_experts
from experiments.tier4.eval_complex_classification import (
    fit_class_models,
    compute_score_scales,
    compute_raw_scores,
    add_subtractive_ellipsoids,
    invalidate_gpu_engine_cache,
)
from src.inference_engine import InferenceEngine
from src.runtime.refinement_checkpoint import RefinementCheckpointAdapter
from src.sdf_engine import EllipsoidExpert, Expert
from src.sdf_optimizer import SDFOptimizer
from src.temporal_sampler import (
    MultiSeedStateEncoder,
    MultiTimescaleStateEncoder,
    TemporalStateEncoder,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIER6_DIR   = "data/tier6"
VOCAB_CHARS = [chr(c) for c in range(32, 127)]   # printable ASCII, space to ~
UNK_CHAR    = "\x00"                              # placeholder for out-of-vocab
VOCAB       = [UNK_CHAR] + VOCAB_CHARS            # index 0 = <unk>
CHAR2ID     = {c: i for i, c in enumerate(VOCAB)}
VOCAB_SIZE  = len(VOCAB)                          # 96
CACHE_VERSION = 2
VOCAB_FINGERPRINT = hashlib.sha256("".join(VOCAB).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Text corpus utilities
# ---------------------------------------------------------------------------

def prepare_text_corpus(
    dataset: str = "wikitext103",
    max_chars: int | None = None,
    seed: int = 42,
    cache_dir: str = TIER6_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Download (once) and encode a text corpus as char-ID arrays.

    Parameters
    ----------
    dataset :
        ``"wikitext103"`` (default, ~100 MB, fast) or ``"wikipedia"``
        (~3 GB streamed, for the full multi-GB experiment).
    max_chars :
        Maximum characters to retain.  ``None`` = use the full corpus.
        Ignored for wikitext103 (already small).
    seed :
        Random seed for reproducible train/test split.
    cache_dir :
        Where to cache the encoded NPZ file.

    Returns
    -------
    train_ids : int32 array — char IDs for the training portion (80 %).
    test_ids  : int32 array — char IDs for the test portion (20 %).
    """
    from experiments.common.data_cache import configure_external_cache_environment

    configure_external_cache_environment()
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{dataset}_{max_chars or 'full'}"
    if dataset == "synthetic_variable_order":
        key = f"{key}_seed{seed}"
    cache_path = os.path.join(cache_dir, f"{key}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        if "cache_version" in data.files:
            version = int(data["cache_version"])
            fingerprint = str(data["vocab_fingerprint"])
            if version != CACHE_VERSION or fingerprint != VOCAB_FINGERPRINT:
                raise ValueError(
                    f"Incompatible Tier 6 cache: {cache_path}. Delete it and rerun "
                    "to rebuild with the current vocabulary."
                )
        else:
            print("[T6] Legacy cache without preprocessing metadata; IDs will be "
                  "validated when sampled.")
        print(f"[T6] Loaded cached corpus: {cache_path}")
        print(f"     train={len(data['train_ids']):,}  test={len(data['test_ids']):,} chars")
        return data["train_ids"], data["test_ids"]

    print(f"[T6] Downloading corpus '{dataset}'…")
    raw_text = _download_corpus(dataset, max_chars, seed=seed)
    print(f"[T6] Raw text length: {len(raw_text):,} characters")

    char_ids = _encode(raw_text)
    split = int(len(char_ids) * 0.8)
    train_ids = char_ids[:split]
    test_ids  = char_ids[split:]

    np.savez_compressed(
        cache_path,
        train_ids=train_ids,
        test_ids=test_ids,
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        vocab_fingerprint=np.array(VOCAB_FINGERPRINT),
    )
    print(f"[T6] Saved to {cache_path}")
    print(f"     train={len(train_ids):,}  test={len(test_ids):,} chars")
    return train_ids, test_ids


def _download_corpus(
    dataset: str,
    max_chars: int | None,
    seed: int = 42,
) -> str:
    """Download raw text for the requested dataset."""
    if dataset == "synthetic_periodic":
        length = max_chars or 5_000
        motif = "geometry learns through time. "
        return (motif * (length // len(motif) + 1))[:length]

    if dataset == "synthetic_variable_order":
        length = max_chars or 5_000
        symbols = "abcdefgh"
        rng = np.random.default_rng(seed)
        sequence = rng.integers(0, len(symbols), size=min(8, length)).tolist()
        while len(sequence) < length:
            next_symbol = (
                sequence[-2] + sequence[-5] + (sequence[-1] % 2)
            ) % len(symbols)
            sequence.append(next_symbol)
        return "".join(symbols[index] for index in sequence)

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "HuggingFace `datasets` is not installed.  "
            "Run: pip install datasets"
        )

    if dataset == "wikitext103":
        # "Salesforce/wikitext" is the parquet-native successor to the
        # deprecated script-based "wikitext" path.
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                          split="train+validation+test")
        text = "\n".join(ds["text"])
        return text[:max_chars] if max_chars else text

    if dataset == "wikipedia":
        # Stream so we never download more than needed.
        # Uses the Wikimedia-hosted parquet version (scripts no longer supported).
        ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                          split="train", streaming=True)
        chunks: list[str] = []
        total = 0
        for article in ds:
            body = article["text"]
            chunks.append(body)
            total += len(body)
            if max_chars and total >= max_chars:
                break
        text = "\n".join(chunks)
        return text[:max_chars] if max_chars else text

    raise ValueError(
        f"Unknown dataset: {dataset!r}. Use a synthetic language, "
        "'wikitext103', or 'wikipedia'."
    )


def _encode(text: str) -> np.ndarray:
    """Map characters to integer IDs (unknown chars → 0)."""
    ids = np.fromiter(
        (CHAR2ID.get(c, 0) for c in text),
        dtype=np.int32,
        count=len(text),
    )
    return ids


# ---------------------------------------------------------------------------
# Context-pair construction
# ---------------------------------------------------------------------------

def sample_context_pairs(
    char_ids: np.ndarray,
    window: int,
    lag: int = 1,
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (context_onehot, next_char_id) pairs from a char-ID sequence.

    Each context is the one-hot concatenation of ``window`` consecutive chars.
    The target is the char that follows the context window by ``lag`` steps.

    Parameters
    ----------
    char_ids : int32 array of shape (T,).
    window   : context window length in characters.
    lag      : how many steps ahead to predict (1 = next char).
    max_samples : cap on the number of pairs returned.
    seed     : RNG seed for the random position sample.

    Returns
    -------
    X : float32 (N, window × VOCAB_SIZE)  — one-hot context vectors.
    y : int32   (N,)                      — target char IDs.
    """
    T = len(char_ids)
    if char_ids.ndim != 1:
        raise ValueError(f"Expected a 1-D character sequence, got shape {char_ids.shape}.")
    n_starts = T - window - lag + 1
    if n_starts <= 0:
        raise ValueError(f"Sequence too short (T={T}) for window={window}, lag={lag}.")

    rng = np.random.default_rng(seed)
    N = min(max_samples, n_starts) if max_samples else n_starts
    starts = rng.choice(n_starts, size=N, replace=False)
    starts.sort()   # sequential-ish order for cache efficiency
    sampled_indices = starts[:, np.newaxis] + np.arange(window + lag)
    sampled_ids = char_ids[sampled_indices]
    if sampled_ids.min() < 0 or sampled_ids.max() >= VOCAB_SIZE:
        raise ValueError("Character sequence contains IDs outside the current vocabulary.")

    # Build context matrix in float32 to keep memory under control
    X = np.zeros((N, window * VOCAB_SIZE), dtype=np.float32)
    y = np.empty(N, dtype=np.int32)

    for i, t in enumerate(starts):
        window_ids = char_ids[t : t + window]           # (window,)
        for j, c in enumerate(window_ids):
            X[i, j * VOCAB_SIZE + c] = 1.0
        y[i] = char_ids[t + window + lag - 1]          # target char

    return X, y


def sample_temporal_state_pairs(
    char_ids: np.ndarray,
    state_dim: int,
    lag: int = 1,
    max_samples: int | None = None,
    seed: int = 42,
    encoder_seed: int = 42,
    warmup: int = 32,
    recurrence: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode one contiguous text block into causal fixed-width temporal states."""
    char_ids = np.asarray(char_ids)
    if char_ids.ndim != 1:
        raise ValueError(f"Expected a 1-D character sequence, got shape {char_ids.shape}.")
    n_available = len(char_ids) - lag
    if lag < 1 or n_available <= 0:
        raise ValueError("Sequence must be longer than the positive lag.")

    sample_count = min(max_samples, n_available) if max_samples else n_available
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, n_available - sample_count + 1))
    warm_start = max(0, start - warmup)
    stop = start + sample_count

    observed_ids = char_ids[warm_start:stop]
    target_ids = char_ids[start + lag:stop + lag]
    if (observed_ids.min() < 0 or observed_ids.max() >= VOCAB_SIZE
            or target_ids.min() < 0 or target_ids.max() >= VOCAB_SIZE):
        raise ValueError("Character sequence contains IDs outside the current vocabulary.")
    observations = np.zeros((len(observed_ids), VOCAB_SIZE), dtype=np.float32)
    observations[np.arange(len(observed_ids)), observed_ids] = 1.0
    encoder = TemporalStateEncoder(
        state_dim=state_dim, recurrence=recurrence, seed=encoder_seed,
    )
    all_states = encoder.transform(observations)
    offset = start - warm_start
    states = all_states[offset:offset + sample_count].astype(np.float32)
    targets = target_ids.astype(np.int32)
    return states, targets


def sample_ensemble_state_pairs(
    char_ids: np.ndarray,
    state_dim: int,
    variant: str,
    lag: int = 1,
    max_samples: int | None = None,
    seed: int = 42,
    encoder_seed: int = 42,
    warmup: int = 32,
    recurrence: float = 0.8,
    recurrences: tuple[float, ...] = (0.3, 0.7, 0.95),
    member_count: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a contiguous block with a fixed-width reservoir ensemble."""
    char_ids = np.asarray(char_ids)
    if char_ids.ndim != 1:
        raise ValueError(f"Expected a 1-D character sequence, got shape {char_ids.shape}.")
    n_available = len(char_ids) - lag
    if lag < 1 or n_available <= 0:
        raise ValueError("Sequence must be longer than the positive lag.")
    sample_count = min(max_samples, n_available) if max_samples else n_available
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, n_available - sample_count + 1))
    warm_start = max(0, start - warmup)
    stop = start + sample_count
    observed_ids = char_ids[warm_start:stop]
    target_ids = char_ids[start + lag:stop + lag]
    if (observed_ids.min() < 0 or observed_ids.max() >= VOCAB_SIZE
            or target_ids.min() < 0 or target_ids.max() >= VOCAB_SIZE):
        raise ValueError("Character sequence contains IDs outside the current vocabulary.")
    observations = np.zeros((len(observed_ids), VOCAB_SIZE), dtype=np.float32)
    observations[np.arange(len(observed_ids)), observed_ids] = 1.0
    if variant == "multi_timescale":
        encoder = MultiTimescaleStateEncoder(
            state_dim=state_dim,
            recurrences=tuple(recurrences),
            seed=encoder_seed,
        )
    elif variant == "multi_seed":
        encoder = MultiSeedStateEncoder(
            state_dim=state_dim,
            member_count=member_count,
            recurrence=recurrence,
            seed=encoder_seed,
        )
    else:
        raise ValueError("variant must be 'multi_timescale' or 'multi_seed'.")
    all_states = encoder.transform(observations)
    offset = start - warm_start
    return (
        all_states[offset:offset + sample_count].astype(np.float32),
        target_ids.astype(np.int32),
    )


def sample_hybrid_state_pairs(
    char_ids: np.ndarray,
    window: int,
    state_dim: int,
    lag: int = 1,
    max_samples: int | None = None,
    seed: int = 42,
    encoder_seed: int = 42,
    warmup: int = 32,
    recurrence: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine exact causal context with reservoir state at the context boundary."""
    char_ids = np.asarray(char_ids)
    if char_ids.ndim != 1:
        raise ValueError(f"Expected a 1-D character sequence, got shape {char_ids.shape}.")
    if window < 1 or lag < 1:
        raise ValueError("window and lag must be positive.")
    n_available = len(char_ids) - window - lag + 1
    if n_available <= 0:
        raise ValueError("Sequence is too short for the requested window and lag.")
    sample_count = min(max_samples, n_available) if max_samples else n_available
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, n_available - sample_count + 1))
    starts = start + np.arange(sample_count)
    warm_start = max(0, start - warmup)
    observed_stop = start + sample_count + window - 1
    observed_ids = char_ids[warm_start:observed_stop]
    target_ids = char_ids[starts + window + lag - 1]
    if (observed_ids.min() < 0 or observed_ids.max() >= VOCAB_SIZE
            or target_ids.min() < 0 or target_ids.max() >= VOCAB_SIZE):
        raise ValueError("Character sequence contains IDs outside the current vocabulary.")

    observations = np.zeros((len(observed_ids), VOCAB_SIZE), dtype=np.float32)
    observations[np.arange(len(observed_ids)), observed_ids] = 1.0
    encoder = TemporalStateEncoder(
        state_dim=state_dim, recurrence=recurrence, seed=encoder_seed,
    )
    states = encoder.transform(observations)
    boundary_offsets = starts + window - 1 - warm_start
    boundary_states = states[boundary_offsets].astype(np.float32)

    exact = np.zeros((sample_count, window * VOCAB_SIZE), dtype=np.float32)
    for position, sequence_start in enumerate(starts):
        context_ids = char_ids[sequence_start:sequence_start + window]
        columns = np.arange(window) * VOCAB_SIZE + context_ids
        exact[position, columns] = 1.0
    return np.concatenate([exact, boundary_states], axis=1), target_ids.astype(np.int32)


# ---------------------------------------------------------------------------
# Preprocessing: PCA → LDA
# ---------------------------------------------------------------------------

def fit_transform_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    pca_components: int,
    seed: int,
) -> tuple[PCA, LinearDiscriminantAnalysis, StandardScaler]:
    """Fit PCA(pca_components) → LDA → StandardScaler on training data.

    The LDA step finds the most discriminative directions in the PCA subspace
    for separating the ``vocab_size`` character classes.  Final dimensionality
    d = min(pca_components, vocab_size − 1), which keeps k_size manageable.
    """
    n_pca = min(pca_components, X_train.shape[1] - 1, X_train.shape[0] - 1)
    pca   = PCA(n_components=n_pca, whiten=True, random_state=seed)
    X_pca = pca.fit_transform(X_train)

    n_classes = len(np.unique(y_train))
    n_lda     = min(n_pca - 1, n_classes - 1)
    lda       = LinearDiscriminantAnalysis(n_components=n_lda)
    X_lda     = lda.fit_transform(X_pca, y_train)

    scaler    = StandardScaler()
    scaler.fit(X_lda)
    return pca, lda, scaler


def apply_transform_pipeline(
    X: np.ndarray,
    pca: PCA,
    lda: LinearDiscriminantAnalysis,
    scaler: StandardScaler,
) -> np.ndarray:
    """Apply a fitted PCA → LDA → StandardScaler transform."""
    return scaler.transform(lda.transform(pca.transform(X)))


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def top_k_accuracy(
    y_true: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    k: int = 5,
) -> float:
    """Fraction of samples where the true label is in the top-K predictions.

    Lower SDF = more likely, so top-K are the K columns with the lowest score.
    """
    y_true = np.asarray(y_true)
    topk_cols = np.argsort(scores, axis=1)[:, :k]   # (N, k) column indices
    topk_ids  = class_ids[topk_cols]                 # (N, k) class IDs
    correct   = np.any(topk_ids == y_true[:, np.newaxis], axis=1)
    return float(correct.mean())


def compute_perplexity(
    y_true: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    alpha: float = 2.0,
) -> float:
    """Perplexity derived from SDF scores converted to log-probabilities.

    Uses softmin: p_k = softmax(−α · SDF_k) to turn SDF scores into a valid
    probability distribution, then computes perplexity = exp(H) where
    H = −Σ (1/N) log p_{y*} is the mean cross-entropy.

    Numerical safeguards
    --------------------
    * ``inf`` scores are replaced by ``1e6`` so no class gets zero weight.
    * Per-row scores are clamped so the gap between worst and best class is
      at most ``MAX_SDF_GAP = log(MAX_PPL) / alpha``.  This bounds each
      sample's CE contribution, preventing a handful of extreme outliers from
      pushing the mean cross-entropy above float64's exp-overflow threshold
      (~709.78).  The resulting perplexity is a faithful estimate of the
      true perplexity up to ``MAX_PPL`` and returns that cap when the true
      value would exceed it.
    """
    K = scores.shape[1]
    # --- 1. Replace non-finite scores ---
    finite_scores = np.where(np.isfinite(scores), scores, 1e6)

    # --- 2. Per-row clamp so exp underflow cannot produce -inf log probs ---
    # Max representable perplexity before float64 overflows: exp(709) ≈ 8.2e307.
    # We use a slightly lower cap to leave headroom for averaging.
    MAX_PPL   = 1e15          # well below exp(709) ≈ 8.2e307, still diagnostically large
    MAX_GAP   = np.log(MAX_PPL) / alpha   # SDF gap that contributes at most log(MAX_PPL) CE
    row_min   = finite_scores.min(axis=1, keepdims=True)
    clipped   = np.minimum(finite_scores, row_min + MAX_GAP)

    # --- 3. Numerically stable log-softmax ---
    neg_alpha = -alpha * clipped
    na_max    = neg_alpha.max(axis=1, keepdims=True)
    log_z     = na_max + np.log(np.exp(neg_alpha - na_max).sum(axis=1, keepdims=True))
    log_p     = neg_alpha - log_z          # (N, K) log-probabilities; all finite

    # --- 4. Gather true-class log probs ---
    id2col = {int(c): i for i, c in enumerate(class_ids)}
    target_log_p = np.array([
        log_p[n, id2col[int(y)]] if int(y) in id2col else np.log(1e-12)
        for n, y in enumerate(y_true)
    ])
    cross_entropy = float(-target_log_p.mean())
    if cross_entropy >= np.log(MAX_PPL):
        return MAX_PPL
    return float(np.exp(cross_entropy))


def predict_char_labels(
    models: dict,
    X: np.ndarray,
    alpha: float,
    score_scales: dict | None = None,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Argmin-SDF prediction for char classes."""
    if use_gpu:
        from experiments.tier4.eval_complex_classification import compute_raw_scores as _crs
        scores = _crs(models, X, alpha, score_scales=score_scales, use_gpu=True)
        class_ids = sorted(models.keys())
        best = np.argmin(scores, axis=1)
        return np.array([class_ids[i] for i in best], dtype=np.int32), scores

    class_ids = sorted(models.keys())
    scores = np.full((len(X), len(class_ids)), np.inf, dtype=np.float64)
    for i, cid in enumerate(class_ids):
        experts = models[cid]
        if not experts:
            continue
        sdf = InferenceEngine(experts, alpha=alpha).get_fused_sdf(X)
        scale = score_scales[cid] if score_scales else 1.0
        scores[:, i] = sdf / scale
    best = np.argmin(scores, axis=1)
    return np.array([class_ids[i] for i in best], dtype=np.int32), scores


def fit_score_calibrator(scores: np.ndarray, y: np.ndarray) -> Pipeline:
    """Fit a multinomial readout that makes independently fitted SDFs comparable."""
    clean_scores = np.where(np.isfinite(scores), scores, 10.0)
    calibrator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=42,
        ),
    )
    calibrator.fit(clean_scores, y)
    return calibrator


def predict_calibrated_labels(
    calibrator: Pipeline, scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return calibrated labels and class probabilities from an SDF score matrix."""
    clean_scores = np.where(np.isfinite(scores), scores, 10.0)
    probabilities = calibrator.predict_proba(clean_scores)
    predictions = calibrator.classes_[np.argmax(probabilities, axis=1)]
    return predictions.astype(np.int32), probabilities


def probability_perplexity(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_ids: np.ndarray,
) -> float:
    """Compute perplexity from calibrated class probabilities."""
    id_to_col = {int(class_id): col for col, class_id in enumerate(class_ids)}
    target_probabilities = [
        probabilities[row, id_to_col[int(label)]]
        if int(label) in id_to_col else 1e-12
        for row, label in enumerate(y_true)
    ]
    cross_entropy = -np.log(np.clip(target_probabilities, 1e-12, 1.0)).mean()
    return float(np.exp(cross_entropy))


def top_k_probability_accuracy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_ids: np.ndarray,
    k: int = 5,
) -> float:
    """Fraction of targets contained in the K highest calibrated probabilities."""
    k = min(k, probabilities.shape[1])
    top_columns = np.argsort(probabilities, axis=1)[:, -k:]
    top_ids = np.asarray(class_ids)[top_columns]
    return float(np.any(top_ids == np.asarray(y_true)[:, np.newaxis], axis=1).mean())


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

def unigram_accuracy(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Always predict the most frequent character in training."""
    counts   = Counter(y_train.tolist())
    best_id  = max(counts, key=counts.get)
    return float((y_test == best_id).mean())


def linear_context_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    max_iter: int = 30,
) -> float:
    """Fast linear control establishing whether the context representation is useful."""
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=max_iter,
        tol=1e-3,
        average=True,
        random_state=seed,
    )
    classifier.fit(X_train, y_train)
    return float(classifier.score(X_test, y_test))


def ngram_accuracy(train_ids: np.ndarray, test_ids: np.ndarray, window: int) -> float:
    """n-gram baseline: predict the most frequent next char given last window chars."""
    # Build n-gram counts from training sequence
    ngram_counts: dict[tuple, Counter] = {}
    for t in range(len(train_ids) - window):
        ctx  = tuple(train_ids[t : t + window].tolist())
        nxt  = int(train_ids[t + window])
        if ctx not in ngram_counts:
            ngram_counts[ctx] = Counter()
        ngram_counts[ctx][nxt] += 1

    # Build default prediction from unigram
    unigram = Counter(train_ids.tolist())
    default_pred = max(unigram, key=unigram.get)

    correct = 0
    for t in range(len(test_ids) - window):
        ctx  = tuple(test_ids[t : t + window].tolist())
        true = int(test_ids[t + window])
        if ctx in ngram_counts:
            pred = max(ngram_counts[ctx], key=ngram_counts[ctx].get)
        else:
            pred = default_pred
        correct += int(pred == true)

    total = len(test_ids) - window
    return correct / max(total, 1)


def sampled_ngram_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window: int,
) -> float:
    """Evaluate n-gram counts on the exact sampled one-hot pairs used by GEODE."""
    exact_width = window * VOCAB_SIZE
    if X_train.shape[1] < exact_width or X_test.shape[1] < exact_width:
        raise ValueError("Features do not contain the requested exact context window.")
    train_contexts = X_train[:, :exact_width].reshape(
        len(X_train), window, VOCAB_SIZE,
    ).argmax(axis=2)
    test_contexts = X_test[:, :exact_width].reshape(
        len(X_test), window, VOCAB_SIZE,
    ).argmax(axis=2)
    counts: dict[tuple[int, ...], Counter] = {}
    for context, target in zip(train_contexts, y_train):
        key = tuple(int(value) for value in context)
        counts.setdefault(key, Counter())[int(target)] += 1
    fallback = Counter(int(value) for value in y_train).most_common(1)[0][0]
    predictions = []
    for context in test_contexts:
        key = tuple(int(value) for value in context)
        prediction = counts[key].most_common(1)[0][0] if key in counts else fallback
        predictions.append(prediction)
    return float(np.mean(np.asarray(predictions) == y_test))


def class_sample_adequacy(y: np.ndarray, dimension: int) -> dict[str, float | int]:
    """Summarize whether each observed class can support a full ellipsoid."""
    counts = np.unique(np.asarray(y), return_counts=True)[1]
    min_seed = dimension * (dimension + 3) // 2
    recommended = 5 * min_seed
    return {
        "class_count": int(len(counts)),
        "min_count": int(counts.min()),
        "median_count": float(np.median(counts)),
        "max_count": int(counts.max()),
        "min_seed": int(min_seed),
        "below_minimum": int(np.count_nonzero(counts < min_seed)),
        "below_recommended": int(np.count_nonzero(counts < recommended)),
    }


def fit_adaptive_class_models(
    X: np.ndarray,
    y: np.ndarray,
    class_ids: np.ndarray,
    consensus_threshold: float,
    capture_threshold: float,
    alpha: float,
    max_iterations: int | None,
    nudge_iterations: int,
    nudge_learning_rate: float,
    use_gpu: bool = False,
    seed: int = 42,
) -> tuple[dict, dict[int, str]]:
    """Fit full, diagonal, or spherical class experts according to support."""
    dimension = X.shape[1]
    min_seed = dimension * (dimension + 3) // 2
    global_scale = max(float(np.median(np.std(X, axis=0))), 1e-3)
    models = {}
    complexity = {}

    for class_position, class_id_raw in enumerate(class_ids):
        class_id = int(class_id_raw)
        class_points = X[y == class_id]
        other_points = X[y != class_id]
        experts = []
        if len(class_points) >= min_seed:
            experts = fit_experts(
                points=class_points,
                exclude_points=other_points,
                consensus_threshold=consensus_threshold,
                capture_threshold=capture_threshold,
                alpha=alpha,
                max_iterations=max_iterations,
                nudge_iterations=nudge_iterations,
                nudge_learning_rate=nudge_learning_rate,
                use_gpu=use_gpu,
                seed=seed + class_position,
            )

        if experts:
            models[class_id] = experts
            complexity[class_id] = "full"
            continue

        center = class_points.mean(axis=0)
        expert = Expert(alpha=alpha)
        if len(class_points) >= dimension + 1:
            radii = np.std(class_points, axis=0) * np.sqrt(dimension)
            radii = np.maximum(radii, global_scale * 0.1)
            mode = "diagonal"
        else:
            radius = global_scale * np.sqrt(dimension)
            radii = np.full(dimension, radius, dtype=np.float64)
            mode = "spherical"
        expert.add_ellipsoid(EllipsoidExpert(center=center, radii=radii))
        models[class_id] = [expert]
        complexity[class_id] = mode

    return models, complexity


# ---------------------------------------------------------------------------
# Sequential k-fold (no temporal leakage)
# ---------------------------------------------------------------------------

def forward_chaining_splits(n_samples: int, n_splits: int, gap: int = 0):
    """Yield expanding-window splits whose training data always precedes validation."""
    if n_splits < 1 or n_samples < n_splits + 1:
        raise ValueError("Need at least n_splits + 1 samples for forward validation.")
    if gap < 0:
        raise ValueError("gap must be non-negative.")

    boundaries = np.linspace(0, n_samples, n_splits + 2, dtype=int)
    indices = np.arange(n_samples)
    for fold in range(n_splits):
        val_start = boundaries[fold + 1]
        val_end = boundaries[fold + 2]
        train_end = val_start - gap
        if train_end <= 0:
            raise ValueError("The requested gap leaves no training samples.")
        yield indices[:train_end], indices[val_start:val_end]


def geometry_calibration_split(
    indices: np.ndarray,
    calibration_fraction: float = 0.15,
    gap: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ordered training indices into past geometry and future calibration sets."""
    ordered = np.asarray(indices)
    if ordered.ndim != 1 or len(ordered) < 3:
        raise ValueError("At least three ordered indices are required.")
    if not 0.0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be in (0, 0.5).")
    if gap < 0:
        raise ValueError("gap must be non-negative.")

    calibration_size = max(1, int(round(len(ordered) * calibration_fraction)))
    calibration_start = len(ordered) - calibration_size
    geometry_end = calibration_start - gap
    if geometry_end <= 0:
        raise ValueError("The calibration fraction and gap leave no geometry samples.")
    return ordered[:geometry_end], ordered[calibration_start:]


# ---------------------------------------------------------------------------
# EM refinement loop
# ---------------------------------------------------------------------------

def supervised_refinement(
    models: dict,
    train_ids: np.ndarray,
    pca: PCA,
    lda: LinearDiscriminantAnalysis,
    scaler: StandardScaler,
    window: int,
    alpha: float,
    score_scales: dict,
    n_iters: int,
    n_epochs: int,
    learning_rate: float,
    max_samples: int,
    seed: int,
    batch_size: int = 256,
    max_batches_per_epoch: int = 8,
    representation: str = "window",
    temporal_state_dim: int = 16,
    temporal_warmup: int = 32,
    temporal_recurrence: float = 0.8,
    temporal_recurrences: tuple[float, ...] = (0.3, 0.7, 0.95),
    temporal_ensemble_members: int = 3,
    use_gpu: bool = False,
    monitor_X: np.ndarray | None = None,
    monitor_y: np.ndarray | None = None,
    epoch_history: list[dict] | None = None,
    refinement_X: np.ndarray | None = None,
    refinement_y: np.ndarray | None = None,
    checkpoint_adapter: RefinementCheckpointAdapter | None = None,
    checkpoint_run_id: str | None = None,
    checkpoint_attempt_id: str | None = None,
    checkpoint_stage_name: str = "tier6-refinement",
    checkpoint_input_hashes: Mapping[str, str] | None = None,
    fail_after_global_epoch: int | None = None,
) -> dict:
    """Run bounded supervised gradient refinement on fixed or resampled pairs.

    Labels remain observed and the current model does not infer latent temporal
    state. The geometry is updated in-place and no new experts are created.

    Returns the refined ``models`` dict.
    """
    if (refinement_X is None) != (refinement_y is None):
        raise ValueError("refinement_X and refinement_y must be provided together.")
    if checkpoint_adapter is not None and (
        checkpoint_run_id is None or checkpoint_attempt_id is None
    ):
        raise ValueError("checkpoint run and attempt IDs are required")
    if checkpoint_adapter is not None and n_epochs <= 0:
        raise ValueError("checkpointed refinement requires positive n_epochs")
    optimizer = SDFOptimizer(
        models=models,
        alpha=alpha,
        learning_rate=learning_rate,
        momentum=0.9,
        score_scales=score_scales,
    )

    history_records = epoch_history if epoch_history is not None else []
    start_iteration = 0
    start_epoch = 0
    global_step = 0
    restored_rng = None
    if checkpoint_adapter is not None:
        latest = checkpoint_adapter.latest_metadata(
            checkpoint_run_id, checkpoint_attempt_id, checkpoint_stage_name,
        )
        if latest is not None:
            total_epochs = n_iters * n_epochs
            if latest.epoch > total_epochs:
                raise ValueError("checkpoint epoch exceeds configured refinement")
            checkpoint_iteration = max(0, (latest.epoch - 1) // n_epochs)
            restored_rng = np.random.default_rng(seed + checkpoint_iteration)
            restored = checkpoint_adapter.restore_latest(
                checkpoint_run_id,
                checkpoint_attempt_id,
                checkpoint_stage_name,
                optimizer=optimizer,
                rng=restored_rng,
            )
            expected_sampler_state = {
                "iteration": checkpoint_iteration,
                "epoch_in_iteration": (latest.epoch - 1) % n_epochs + 1,
            }
            if restored.sampler_state != expected_sampler_state:
                raise ValueError("checkpoint sampler progress does not match epoch")
            history_records[:] = restored.epoch_history
            global_step = restored.metadata.global_step
            start_iteration, start_epoch = divmod(restored.metadata.epoch, n_epochs)
            score_scales = optimizer.score_scales

    for refinement_i in range(start_iteration, n_iters):
        print(f"  Refinement iter {refinement_i + 1}/{n_iters}")

        if refinement_X is not None:
            X_em = np.asarray(refinement_X, dtype=np.float64)
            y_em = np.asarray(refinement_y)
        elif representation == "temporal_state":
            X_raw, y_em = sample_temporal_state_pairs(
                train_ids,
                state_dim=temporal_state_dim,
                lag=1,
                max_samples=max_samples,
                seed=seed + refinement_i + 1,
                encoder_seed=seed,
                warmup=temporal_warmup,
                recurrence=temporal_recurrence,
            )
        elif representation in {"multi_timescale", "multi_seed"}:
            X_raw, y_em = sample_ensemble_state_pairs(
                train_ids,
                state_dim=temporal_state_dim,
                variant=representation,
                lag=1,
                max_samples=max_samples,
                seed=seed + refinement_i + 1,
                encoder_seed=seed,
                warmup=temporal_warmup,
                recurrence=temporal_recurrence,
                recurrences=temporal_recurrences,
                member_count=temporal_ensemble_members,
            )
        elif representation == "hybrid":
            X_raw, y_em = sample_hybrid_state_pairs(
                train_ids,
                window=window,
                state_dim=temporal_state_dim,
                lag=1,
                max_samples=max_samples,
                seed=seed + refinement_i + 1,
                encoder_seed=seed,
                warmup=temporal_warmup,
                recurrence=temporal_recurrence,
            )
        else:
            X_raw, y_em = sample_context_pairs(
                train_ids, window=window, lag=1,
                max_samples=max_samples, seed=seed + refinement_i + 1,
            )
        if refinement_X is None:
            X_em = apply_transform_pipeline(
                X_raw.astype(np.float64), pca, lda, scaler,
            )
        modeled_mask = np.isin(y_em, np.fromiter(models.keys(), dtype=np.int32))
        X_em = X_em[modeled_mask]
        y_em = y_em[modeled_mask]

        # ── Bounded minibatch gradient refinement ───────────────────────────
        losses = []
        iteration_start_epoch = start_epoch if refinement_i == start_iteration else 0
        rng = (
            restored_rng
            if restored_rng is not None and iteration_start_epoch > 0
            else np.random.default_rng(seed + refinement_i)
        )
        for ep in range(iteration_start_epoch, n_epochs):
            order = rng.permutation(len(X_em))
            batch_losses = []
            for batch_number, start in enumerate(range(0, len(order), batch_size)):
                if batch_number >= max_batches_per_epoch:
                    break
                batch_indices = order[start:start + batch_size]
                batch_losses.append(optimizer.step(X_em[batch_indices], y_em[batch_indices]))
                global_step += 1
            losses.append(float(np.mean(batch_losses)))
            train_metrics = optimizer.evaluate(X_em, y_em)
            monitor_metrics = (
                optimizer.evaluate(monitor_X, monitor_y)
                if monitor_X is not None and monitor_y is not None else None
            )
            epoch_record = {
                "iteration": refinement_i + 1,
                "epoch": ep + 1,
                "batch_training_loss": losses[-1],
                "train_loss": train_metrics["loss"],
                "train_error": train_metrics["error"],
                "test_loss": (
                    monitor_metrics["loss"] if monitor_metrics is not None else None
                ),
                "test_error": (
                    monitor_metrics["error"] if monitor_metrics is not None else None
                ),
                "monitor_used_for_learning": False,
            }
            history_records.append(epoch_record)
            monitor_text = (
                f"  test_loss={monitor_metrics['loss']:.4f}"
                f"  test_error={monitor_metrics['error']:.4f}"
                if monitor_metrics is not None else ""
            )
            print(
                f"    epoch {ep + 1}/{n_epochs}"
                f"  train_loss={train_metrics['loss']:.4f}"
                f"  train_error={train_metrics['error']:.4f}{monitor_text}"
            )

            if ep + 1 == n_epochs:
                if use_gpu:
                    invalidate_gpu_engine_cache(models)
                score_scales = compute_score_scales(
                    models, X_em, alpha=alpha, use_gpu=use_gpu, class_labels=y_em,
                )
                optimizer.score_scales = score_scales
            global_epoch = refinement_i * n_epochs + ep + 1
            if checkpoint_adapter is not None:
                checkpoint_adapter.save(
                    checkpoint_run_id,
                    checkpoint_attempt_id,
                    checkpoint_stage_name,
                    global_epoch,
                    global_step,
                    optimizer=optimizer,
                    rng=rng,
                    epoch_history=history_records,
                    sampler_state={
                        "iteration": refinement_i,
                        "epoch_in_iteration": ep + 1,
                    },
                    input_hashes=checkpoint_input_hashes,
                )
            if fail_after_global_epoch == global_epoch:
                raise RuntimeError(f"injected failure after refinement epoch {global_epoch}")

        print(f"    loss  {losses[0]:.4f} → {losses[-1]:.4f}")
        start_epoch = 0
        restored_rng = None

    return models, score_scales


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_text_prediction_experiment(
    dataset: str = "wikitext103",
    max_chars: int | None = None,
    max_train_samples: int = 300_000,
    max_test_samples: int = 60_000,
    window: int = 3,
    pca_components: int = 24,
    n_folds: int = 5,
    alpha: float = 2.0,
    consensus_threshold: float = 0.06,
    capture_threshold: float = 0.08,
    max_iterations: int | None = None,
    nudge_iterations: int = 20,
    nudge_lr: float = 0.02,
    n_refinement_iters: int = 2,
    n_refinement_epochs: int = 20,
    refinement_lr: float = 0.005,
    refinement_batch_size: int = 256,
    refinement_max_batches_per_epoch: int = 8,
    calibration_fraction: float = 0.15,
    representation: str = "window",
    temporal_state_dim: int = 16,
    temporal_warmup: int = 32,
    temporal_recurrence: float = 0.8,
    temporal_recurrences: tuple[float, ...] = (0.3, 0.7, 0.95),
    temporal_ensemble_members: int = 3,
    use_subtractive: bool = True,
    seed: int = 42,
    use_gpu: bool = False,
):
    """Full Tier 6 experiment with k-fold CV, EM refinement, and baselines.

    Parameters
    ----------
    dataset :
        ``"wikitext103"`` (quick, ~100 MB) or ``"wikipedia"`` (full multi-GB).
    max_chars :
        Cap on raw text characters.  ``None`` = use full corpus.
        Recommended: ``None`` for wikitext103; ``500_000_000`` for wikipedia.
    max_train_samples, max_test_samples :
        Number of (context, next_char) pairs sampled for train/test evaluation.
    window :
        Number of previous characters used as context (= n-gram order).
    pca_components :
        PCA output dimensionality before LDA.  Final d = min(pca_components,
        vocab_size − 1).  Smaller values → faster fitting, less expressiveness.
    n_folds :
        Number of sequential k-fold cross-validation folds on the training set.
    alpha :
        Softmin concentration for GEODE inference.
    consensus_threshold :
        Fraction of unexplained pool required to lock in a new expert.
    capture_threshold :
        SDF threshold for "captured" (inside ellipsoid).
    max_iterations :
        RANSAC iterations per ellipsoid (None = auto from d).
    n_refinement_iters :
        Number of supervised refinement iterations after the initial fit.
    n_refinement_epochs :
        SDFOptimizer gradient epochs per refinement iteration.
    refinement_lr :
        Learning rate for the SDFOptimizer.
    refinement_batch_size, refinement_max_batches_per_epoch :
        Bounds on supervised refinement work per epoch.
    calibration_fraction :
        Final fraction of each ordered training region reserved for fitting
        the multinomial SDF score calibrator.
    representation :
        ``"window"`` for explicit delayed one-hot context or
        ``"temporal_state"`` for a contiguous fixed-width recurrent state,
        ``"multi_timescale"`` or ``"multi_seed"`` for fixed-width reservoir
        ensembles, or ``"hybrid"`` for exact context plus recurrent state.
    use_subtractive :
        Whether to fit subtractive CSG ellipsoids. Disable this while
        validating the base classifier and gradient refinement independently.
    """

    print("=" * 68)
    print("Tier 6: Temporal Text Prediction — GEODE")
    print("=" * 68)
    if representation not in {
        "window", "temporal_state", "multi_timescale", "multi_seed", "hybrid",
    }:
        raise ValueError(
            "Unsupported temporal representation."
        )

    # ── 1. Load / download corpus ─────────────────────────────────────────
    train_ids_full, test_ids_full = prepare_text_corpus(
        dataset=dataset, max_chars=max_chars, seed=seed,
    )

    # ── 2. Baselines (computed directly on raw char ID sequences) ─────────
    print("\n[Baselines]")
    # Limit baseline data for speed
    n_baseline = min(5_000_000, len(train_ids_full))
    unigram_acc = unigram_accuracy(train_ids_full[:n_baseline],
                                   test_ids_full[:min(500_000, len(test_ids_full))])
    ngram_acc   = ngram_accuracy(
        train_ids_full[:n_baseline],
        test_ids_full[:min(200_000, len(test_ids_full))],
        window=window,
    )
    print(f"  Unigram (most-frequent char)  : {unigram_acc * 100:.2f}%")
    print(f"  {window}-gram (most-frequent next) : {ngram_acc * 100:.2f}%")

    # ── 3. Sample (context, next_char) pairs ──────────────────────────────
    print(f"\n[Sampling pairs: window={window}, vocab={VOCAB_SIZE}]")
    if representation == "temporal_state":
        X_train_raw, y_train = sample_temporal_state_pairs(
            train_ids_full, temporal_state_dim, lag=1,
            max_samples=max_train_samples, seed=seed, encoder_seed=seed,
            warmup=temporal_warmup,
            recurrence=temporal_recurrence,
        )
        X_test_raw, y_test = sample_temporal_state_pairs(
            test_ids_full, temporal_state_dim, lag=1,
            max_samples=max_test_samples, seed=seed + 99, encoder_seed=seed,
            warmup=temporal_warmup,
            recurrence=temporal_recurrence,
        )
    elif representation in {"multi_timescale", "multi_seed"}:
        X_train_raw, y_train = sample_ensemble_state_pairs(
            train_ids_full,
            state_dim=temporal_state_dim,
            variant=representation,
            lag=1,
            max_samples=max_train_samples,
            seed=seed,
            encoder_seed=seed,
            warmup=temporal_warmup,
            recurrence=temporal_recurrence,
            recurrences=temporal_recurrences,
            member_count=temporal_ensemble_members,
        )
        X_test_raw, y_test = sample_ensemble_state_pairs(
            test_ids_full,
            state_dim=temporal_state_dim,
            variant=representation,
            lag=1,
            max_samples=max_test_samples,
            seed=seed + 99,
            encoder_seed=seed,
            warmup=temporal_warmup,
            recurrence=temporal_recurrence,
            recurrences=temporal_recurrences,
            member_count=temporal_ensemble_members,
        )
    elif representation == "hybrid":
        X_train_raw, y_train = sample_hybrid_state_pairs(
            train_ids_full,
            window=window,
            state_dim=temporal_state_dim,
            lag=1,
            max_samples=max_train_samples,
            seed=seed,
            encoder_seed=seed,
            warmup=temporal_warmup,
            recurrence=temporal_recurrence,
        )
        X_test_raw, y_test = sample_hybrid_state_pairs(
            test_ids_full,
            window=window,
            state_dim=temporal_state_dim,
            lag=1,
            max_samples=max_test_samples,
            seed=seed + 99,
            encoder_seed=seed,
            warmup=temporal_warmup,
            recurrence=temporal_recurrence,
        )
    else:
        X_train_raw, y_train = sample_context_pairs(
            train_ids_full, window=window, lag=1,
            max_samples=max_train_samples, seed=seed,
        )
        X_test_raw, y_test = sample_context_pairs(
            test_ids_full, window=window, lag=1,
            max_samples=max_test_samples, seed=seed + 99,
        )
    print(f"  Train pairs: {len(X_train_raw):,}   Test pairs: {len(X_test_raw):,}")
    matched_ngram_acc = None
    if representation in {"window", "hybrid"}:
        matched_ngram_acc = sampled_ngram_accuracy(
            X_train_raw, y_train, X_test_raw, y_test, window,
        )
        print(f"  Matched-data {window}-gram: {matched_ngram_acc * 100:.2f}%")

    class_ids = np.unique(y_train)
    n_classes = len(class_ids)
    print(f"  Active character classes in training: {n_classes}")

    # Adequacy check. Class averages hide the severe long-tail imbalance.
    d_approx  = min(pca_components - 1, n_classes - 1)
    adequacy = class_sample_adequacy(y_train, d_approx)
    print(f"  Approx d={d_approx}  k_size={adequacy['min_seed']}")
    print(f"  Per-class count: min={adequacy['min_count']}  "
          f"median={adequacy['median_count']:.0f}  max={adequacy['max_count']}")
    print(f"  Classes below minimum={adequacy['below_minimum']}/{n_classes}  "
          f"below recommended={adequacy['below_recommended']}/{n_classes}")

    # ── 4. Sequential K-fold cross-validation ─────────────────────────────
    print(f"\n[Cross-validation  ({n_folds}-fold, sequential splits)]")
    fold_accs = []
    for fold_i, (cv_train_idx, cv_val_idx) in enumerate(
        forward_chaining_splits(len(X_train_raw), n_folds, gap=window), start=1
    ):
        cv_geometry_idx, cv_calibration_idx = geometry_calibration_split(
            cv_train_idx, calibration_fraction=calibration_fraction, gap=window,
        )
        X_cv_raw_tr = X_train_raw[cv_geometry_idx].astype(np.float64)
        X_cv_raw_cal = X_train_raw[cv_calibration_idx].astype(np.float64)
        X_cv_raw_vl = X_train_raw[cv_val_idx].astype(np.float64)
        y_cv_tr     = y_train[cv_geometry_idx]
        y_cv_cal    = y_train[cv_calibration_idx]
        y_cv_vl     = y_train[cv_val_idx]

        pca_f, lda_f, scaler_f = fit_transform_pipeline(
            X_cv_raw_tr, y_cv_tr, pca_components, seed
        )
        X_cv_tr = apply_transform_pipeline(X_cv_raw_tr, pca_f, lda_f, scaler_f)
        X_cv_cal = apply_transform_pipeline(X_cv_raw_cal, pca_f, lda_f, scaler_f)
        X_cv_vl = apply_transform_pipeline(X_cv_raw_vl, pca_f, lda_f, scaler_f)

        cv_class_ids = np.unique(y_cv_tr)
        fold_models, fold_complexity = fit_adaptive_class_models(
            X=X_cv_tr, y=y_cv_tr, class_ids=cv_class_ids,
            consensus_threshold=consensus_threshold,
            capture_threshold=capture_threshold,
            alpha=alpha,
            max_iterations=max_iterations,
            nudge_iterations=nudge_iterations,
            nudge_learning_rate=nudge_lr,
            use_gpu=use_gpu,
            seed=seed + fold_i * 1_000,
        )
        mode_counts = Counter(fold_complexity.values())
        if use_subtractive:
            add_subtractive_ellipsoids(
                fold_models, X_cv_tr, y_cv_tr, cv_class_ids,
                capture_threshold=capture_threshold, alpha=alpha,
                max_iterations=max_iterations, use_gpu=use_gpu,
                seed=seed + fold_i * 1_000 + 500,
            )
        fold_scales = compute_score_scales(
            fold_models, X_cv_tr, alpha=alpha, use_gpu=use_gpu,
            class_labels=y_cv_tr,
        )
        calibration_mask = np.isin(y_cv_cal, cv_class_ids)
        _, calibration_scores = predict_char_labels(
            fold_models, X_cv_cal[calibration_mask], alpha, fold_scales,
            use_gpu=use_gpu,
        )
        fold_calibrator = fit_score_calibrator(
            calibration_scores, y_cv_cal[calibration_mask],
        )
        _, validation_scores = predict_char_labels(
            fold_models, X_cv_vl, alpha, fold_scales, use_gpu=use_gpu,
        )
        y_pred, _ = predict_calibrated_labels(fold_calibrator, validation_scores)
        fold_acc = float((y_pred == y_cv_vl).mean())
        fold_accs.append(fold_acc)
        print(
            f"  Fold {fold_i}/{n_folds}  val_acc={fold_acc * 100:.2f}%  "
            f"experts={dict(mode_counts)}"
        )

    cv_mean = np.mean(fold_accs)
    cv_std  = np.std(fold_accs)
    print(f"  CV mean={cv_mean * 100:.2f}%  std={cv_std * 100:.2f}%")

    # ── 5. Full training set fit ───────────────────────────────────────────
    print("\n[Full training fit]")
    X_train_raw_f = X_train_raw.astype(np.float64)
    X_test_raw_f  = X_test_raw.astype(np.float64)
    geometry_idx, calibration_idx = geometry_calibration_split(
        np.arange(len(X_train_raw)),
        calibration_fraction=calibration_fraction,
        gap=window,
    )
    X_geometry_raw = X_train_raw_f[geometry_idx]
    y_geometry = y_train[geometry_idx]
    y_calibration = y_train[calibration_idx]
    linear_acc = linear_context_accuracy(
        X_geometry_raw, y_geometry, X_test_raw_f, y_test, seed=seed,
    )
    print(f"  Linear context control: {linear_acc * 100:.2f}%")

    pca_full, lda_full, scaler_full = fit_transform_pipeline(
        X_geometry_raw, y_geometry, pca_components, seed
    )
    X_train = apply_transform_pipeline(X_train_raw_f, pca_full, lda_full, scaler_full)
    X_test  = apply_transform_pipeline(X_test_raw_f,  pca_full, lda_full, scaler_full)
    X_geometry = X_train[geometry_idx]
    X_calibration = X_train[calibration_idx]

    d_actual = X_train.shape[1]
    k_size_actual = d_actual * (d_actual + 3) // 2
    print(f"  d={d_actual}  k_size={k_size_actual}")

    model_class_ids = np.unique(y_geometry)
    models, model_complexity = fit_adaptive_class_models(
        X=X_geometry, y=y_geometry, class_ids=model_class_ids,
        consensus_threshold=consensus_threshold,
        capture_threshold=capture_threshold,
        alpha=alpha,
        max_iterations=max_iterations,
        nudge_iterations=nudge_iterations,
        nudge_learning_rate=nudge_lr,
        use_gpu=use_gpu,
        seed=seed + 100_000,
    )
    print(f"  Expert complexity: {dict(Counter(model_complexity.values()))}")
    if use_subtractive:
        add_subtractive_ellipsoids(
            models, X_geometry, y_geometry, model_class_ids,
            capture_threshold=capture_threshold, alpha=alpha,
            max_iterations=max_iterations, use_gpu=use_gpu,
            seed=seed + 100_500,
        )
    score_scales = compute_score_scales(
        models, X_geometry, alpha=alpha, use_gpu=use_gpu, class_labels=y_geometry,
    )

    calibration_mask = np.isin(y_calibration, model_class_ids)
    _, calibration_scores = predict_char_labels(
        models, X_calibration[calibration_mask], alpha, score_scales,
        use_gpu=use_gpu,
    )
    calibrator = fit_score_calibrator(
        calibration_scores, y_calibration[calibration_mask],
    )

    _, scores_train = predict_char_labels(models, X_train, alpha, score_scales, use_gpu=use_gpu)
    y_pred_train, _ = predict_calibrated_labels(calibrator, scores_train)
    train_acc  = float((y_pred_train == y_train).mean())
    y_pred_raw, scores_t0 = predict_char_labels(models, X_test, alpha, score_scales, use_gpu=use_gpu)
    raw_test_acc = float((y_pred_raw == y_test).mean())
    y_pred_t0, probabilities_t0 = predict_calibrated_labels(calibrator, scores_t0)
    test_acc_0 = float((y_pred_t0 == y_test).mean())
    ppl_0 = probability_perplexity(y_test, probabilities_t0, calibrator.classes_)
    top5_0 = top_k_probability_accuracy(
        y_test, probabilities_t0, calibrator.classes_, k=5,
    )

    print(f"  Initial  train={train_acc * 100:.2f}%  raw_test={raw_test_acc * 100:.2f}%  "
          f"cal_test={test_acc_0 * 100:.2f}%  top5={top5_0 * 100:.2f}%  ppl={ppl_0:.2f}")

    # ── 6. Optional supervised gradient refinement ─────────────────────────
    test_accs = [test_acc_0]
    top5s     = [top5_0]
    ppls      = [ppl_0]
    epoch_history: list[dict] = []

    if n_refinement_iters > 0:
        if use_subtractive:
            raise ValueError("Gradient refinement requires use_subtractive=False.")
        print(
            "\n[Supervised gradient refinement  "
            f"({n_refinement_iters} iters × {n_refinement_epochs} epochs)]"
        )
        models, score_scales = supervised_refinement(
            models=models,
            train_ids=train_ids_full,
            pca=pca_full,
            lda=lda_full,
            scaler=scaler_full,
            window=window,
            alpha=alpha,
            score_scales=score_scales,
            n_iters=n_refinement_iters,
            n_epochs=n_refinement_epochs,
            learning_rate=refinement_lr,
            max_samples=max_train_samples,
            seed=seed,
            batch_size=refinement_batch_size,
            max_batches_per_epoch=refinement_max_batches_per_epoch,
            representation=representation,
            temporal_state_dim=temporal_state_dim,
            temporal_warmup=temporal_warmup,
            temporal_recurrence=temporal_recurrence,
            temporal_recurrences=temporal_recurrences,
            temporal_ensemble_members=temporal_ensemble_members,
            use_gpu=use_gpu,
            monitor_X=X_test,
            monitor_y=y_test,
            epoch_history=epoch_history,
        )
        _, calibration_scores_em = predict_char_labels(
            models, X_calibration[calibration_mask], alpha, score_scales,
            use_gpu=use_gpu,
        )
        refined_calibrator = fit_score_calibrator(
            calibration_scores_em, y_calibration[calibration_mask],
        )
        _, scores_em = predict_char_labels(models, X_test, alpha, score_scales, use_gpu=use_gpu)
        refined_predictions, refined_probabilities = predict_calibrated_labels(
            refined_calibrator, scores_em,
        )
        refined_test_acc = float((refined_predictions == y_test).mean())
        refined_top5 = top_k_probability_accuracy(
            y_test, refined_probabilities, refined_calibrator.classes_, k=5,
        )
        refined_ppl = probability_perplexity(
            y_test, refined_probabilities, refined_calibrator.classes_,
        )

        test_accs.append(refined_test_acc)
        top5s.append(refined_top5)
        ppls.append(refined_ppl)
        print(f"  After refinement test={refined_test_acc * 100:.2f}%  "
              f"top5={refined_top5 * 100:.2f}%  ppl={refined_ppl:.2f}")

    # ── 7. Results summary ────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("RESULTS SUMMARY")
    print("=" * 68)
    print(f"  Dataset        : {dataset}  (chars: train={len(train_ids_full):,}  "
          f"test={len(test_ids_full):,})")
    print(f"  Context window : {window} chars  →  d={d_actual} (after PCA+LDA)")
    print(f"  Training pairs : {len(X_train):,}")
    print(f"  Test pairs     : {len(X_test):,}")
    print()
    print(f"  Unigram baseline : {unigram_acc * 100:.2f}%")
    print(f"  {window}-gram baseline  : {ngram_acc * 100:.2f}%")
    print()
    print(f"  CV accuracy      : {cv_mean * 100:.2f}% ± {cv_std * 100:.2f}%")
    print(f"  Initial  top-1   : {test_accs[0] * 100:.2f}%  "
          f"top-5: {top5s[0] * 100:.2f}%  ppl: {ppls[0]:.2f}")
    if len(test_accs) > 1:
        delta = (test_accs[-1] - test_accs[0]) * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  After refinement : {test_accs[-1] * 100:.2f}%  "
              f"top-5: {top5s[-1] * 100:.2f}%  ppl: {ppls[-1]:.2f}"
              f"  ({sign}{delta:.2f}pp vs initial)")
    print("=" * 68)

    return {
        "cv_acc_mean"   : cv_mean,
        "cv_acc_std"    : cv_std,
        "test_acc_init" : test_accs[0],
        "test_acc_raw"  : raw_test_acc,
        "test_acc_refined": test_accs[-1] if len(test_accs) > 1 else None,
        "test_acc_final": test_accs[-1],
        "top5_init"     : top5s[0],
        "top5_refined"  : top5s[-1] if len(top5s) > 1 else None,
        "ppl_init"      : ppls[0],
        "ppl_refined"   : ppls[-1] if len(ppls) > 1 else None,
        "ppl_final"     : ppls[-1],
        "unigram_acc"   : unigram_acc,
        "ngram_acc"     : ngram_acc,
        "ngram_matched_acc": matched_ngram_acc,
        "ngram_best_practical_acc": ngram_acc,
        "linear_acc"    : linear_acc,
        "class_count"   : len(model_class_ids),
        "sample_adequacy": adequacy,
        "epoch_history"  : epoch_history,
        "epoch_monitor_policy": (
            "Test metrics are observational only and are never used for gradients, "
            "sampling, early stopping, or hyperparameter selection."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Tier 6: Temporal text prediction with GEODE refinement."
    )
    p.add_argument("--dataset",    default="wikitext103",
                   choices=["wikitext103", "wikipedia"],
                   help="Text corpus (wikitext103 ≈ 100 MB; wikipedia ≈ 3 GB).")
    p.add_argument("--max_chars",  type=int, default=None,
                   help="Cap on raw characters (None = full corpus).")
    p.add_argument("--max_train",  type=int, default=300_000,
                   help="Max (context, char) training pairs.")
    p.add_argument("--max_test",   type=int, default=60_000,
                   help="Max (context, char) test pairs.")
    p.add_argument("--window",     type=int, default=3,
                   help="Context window (chars).")
    p.add_argument("--pca",        type=int, default=24,
                   help="PCA components before LDA.")
    p.add_argument("--folds",      type=int, default=5)
    p.add_argument("--alpha",      type=float, default=2.0)
    p.add_argument("--refinement-iters", "--em_iters", type=int, default=2)
    p.add_argument("--refinement-epochs", "--em_epochs", type=int, default=20)
    p.add_argument("--refinement-lr", "--em_lr", type=float, default=0.005)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--gpu",        action="store_true")
    args = p.parse_args()
    if any(argument in sys.argv for argument in ("--em_iters", "--em_epochs", "--em_lr")):
        warnings.warn(
            "The --em_* options are deprecated; use --refinement-* instead.",
            DeprecationWarning,
            stacklevel=1,
        )

    run_text_prediction_experiment(
        dataset           = args.dataset,
        max_chars         = args.max_chars,
        max_train_samples = args.max_train,
        max_test_samples  = args.max_test,
        window            = args.window,
        pca_components    = args.pca,
        n_folds           = args.folds,
        alpha             = args.alpha,
        n_refinement_iters  = args.refinement_iters,
        n_refinement_epochs = args.refinement_epochs,
        refinement_lr     = args.refinement_lr,
        seed              = args.seed,
        use_gpu           = args.gpu,
    )
