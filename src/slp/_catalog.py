"""SLP inventory -- the single source of truth for the catalog.

Every standard-library primitive has one entry here. The entry
holds the domain, the operation family, the access path, the
status, the tier, and a one-line STE summary. ``CATALOG.md`` is
generated from this list, never edited by hand.

Adding a primitive is a two-step change:
1. Build the factory (with ``PrimitiveSpec`` and a docstring).
2. Flip the entry here from PENDING to BUILT and set the access
   path.

The full backlog (beyond the launch set) lives in
``analysis/SLP_POSSIBILITY_SPACE_v1.md``.
"""

from __future__ import annotations

from typing import Any

STATUS_BUILT = "BUILT"
STATUS_PENDING = "PENDING"

_DOMAINS = (
    "memory",
    "math",
    "symbolic",
    "logic",
    "signal",
    "text",
    "image",
    "tables",
    "code",
)

# (id, domain, family, name, access, status, tier, summary)
_ENTRIES: tuple[tuple[str, str, str, str, str, str, str, str], ...] = (
    # ---- memory ----
    ("mem.count", "memory", "memory", "count-based memory",
     "src.programmatic_memory.ProgrammaticMemory", STATUS_BUILT, "A",
     "Variable-order count memory: register, backoff lookup, predict-next."),
    # ---- math ----
    ("math.scale", "math", "transform", "scale",
     "src.slp.make_scale", STATUS_BUILT, "A",
     "Multiply every feature by a scalar."),
    ("math.clip", "math", "transform", "clip",
     "src.slp.make_clip", STATUS_BUILT, "A",
     "Clip values to [low, high]."),
    ("math.affine", "math", "transform", "affine",
     "src.slp.make_affine", STATUS_BUILT, "A",
     "Affine transform Y = X @ A + b."),
    ("math.l2-normalize", "math", "transform", "l2_normalize",
     "src.slp.make_l2_normalize", STATUS_BUILT, "A",
     "Normalize each row to unit L2 length."),
    ("math.reductions", "math", "reduce", "sum, mean, var, std, min, max, argmin, argmax",
     "", STATUS_PENDING, "A",
     "Axis reductions with the axis fixed at registration."),
    ("math.softmax", "math", "transform", "softmax, log-softmax",
     "", STATUS_PENDING, "A",
     "Softmax family over a fixed axis, float64."),
    ("math.activations", "math", "transform", "sigmoid, tanh, relu, gelu",
     "", STATUS_PENDING, "A",
     "Pinned activation functions."),
    ("math.exp-log", "math", "transform", "log, exp, sqrt, pow, abs, sign",
     "", STATUS_PENDING, "A",
     "Elementwise elementary functions in float64."),
    ("math.linalg", "math", "solve", "solve, pinv, det, trace, svd, eig",
     "", STATUS_PENDING, "A",
     "Linear algebra in float64, pinned LAPACK path."),
    ("math.distances", "math", "compare", "euclidean, cosine, manhattan",
     "", STATUS_PENDING, "A",
     "Distance and similarity functions."),
    ("math.prob-tables", "math", "solve", "cdf, ppf, pdf for fixed families",
     "", STATUS_PENDING, "A",
     "Normal, student, beta, gamma probability functions."),
    # ---- symbolic ----
    ("sym.simplify", "symbolic", "transform", "parse, simplify, expand, factor, substitute",
     "", STATUS_PENDING, "B",
     "Symbolic manipulation on a pinned computer-algebra engine."),
    ("sym.calculus", "symbolic", "reduce", "differentiate, integrate",
     "", STATUS_PENDING, "B",
     "Symbolic calculus, bounded."),
    ("sym.solve", "symbolic", "solve", "linear, polynomial, small nonlinear",
     "", STATUS_PENDING, "B",
     "Symbolic solving with a resource cap."),
    ("sym.latex", "symbolic", "encode", "to-latex, to-mathml",
     "", STATUS_PENDING, "B",
     "Render an expression tree."),
    # ---- logic ----
    ("logic.threshold", "logic", "transform", "threshold",
     "src.slp.make_threshold", STATUS_BUILT, "A",
     "Binarize: 1.0 where X > value, else 0.0."),
    ("logic.and", "logic", "combine", "logical_and",
     "src.slp.make_logical_and", STATUS_BUILT, "A",
     "Column-wise AND."),
    ("logic.or", "logic", "combine", "logical_or",
     "src.slp.make_logical_or", STATUS_BUILT, "A",
     "Column-wise OR."),
    ("logic.not-xor", "logic", "combine", "not, xor, nand, nor",
     "", STATUS_PENDING, "A",
     "Remaining boolean algebra."),
    ("logic.compare", "logic", "compare", "eq, ne, lt, le, gt, ge",
     "", STATUS_PENDING, "A",
     "Elementwise comparisons."),
    ("logic.where", "logic", "select", "where, select, argwhere",
     "", STATUS_PENDING, "A",
     "Conditional selection."),
    ("logic.sets", "logic", "combine", "unique, intersect, union, difference",
     "", STATUS_PENDING, "A",
     "Set operations with deterministic ordering."),
    # ---- signal ----
    ("sig.delay", "signal", "transform", "delay",
     "src.slp.make_delay", STATUS_BUILT, "A",
     "Ring-buffer delay over a stream."),
    ("sig.mel", "signal", "convert", "mel_spectrogram",
     "geode.core.audio_primitives.mel_spectrogram", STATUS_BUILT, "A",
     "FFT/mel front-end, bit-exact (M267 stage 0)."),
    ("sig.fft", "signal", "convert", "fft, rfft, ifft",
     "", STATUS_PENDING, "A",
     "Fourier transforms on a pinned backend."),
    ("sig.stft", "signal", "convert", "stft, spectrogram",
     "", STATUS_PENDING, "B",
     "Short-time spectra with pinned windows."),
    ("sig.smoothing", "signal", "transform", "moving average, median filter",
     "", STATUS_PENDING, "A",
     "Fixed-window smoothing."),
    ("sig.resample", "signal", "convert", "resample",
     "", STATUS_PENDING, "A",
     "Fixed-ratio resampling with pinned interpolation."),
    # ---- text ----
    ("text.tokenize", "text", "transform", "tokenize",
     "", STATUS_PENDING, "B",
     "Tokenization with a pinned vocabulary and merges."),
    ("text.normalize", "text", "transform", "unicode-normalize, casefold, whitespace",
     "", STATUS_PENDING, "A",
     "Text normalization with a pinned Unicode version."),
    ("text.ngrams", "text", "transform", "n-grams, bag-of-words, tf-idf",
     "", STATUS_PENDING, "A",
     "N-gram features with stats fitted at registration."),
    ("text.edit-distance", "text", "compare", "levenshtein, damerau",
     "", STATUS_PENDING, "A",
     "Edit distances."),
    ("text.hash-encode", "text", "encode", "sha256, blake3, keccak, base64, base58, hex",
     "", STATUS_PENDING, "A",
     "Hashing and text encodings."),
    ("text.regex", "text", "select", "regex search and replace",
     "", STATUS_PENDING, "B",
     "Bounded regular expressions with a resource cap."),
    # ---- image ----
    ("img.grayscale", "image", "convert", "grayscale, invert, normalize",
     "", STATUS_PENDING, "A",
     "Pixel-space conversions with pinned coefficients."),
    ("img.colorspace", "image", "convert", "rgb-hsv, rgb-yuv, rgb-lab",
     "", STATUS_PENDING, "A",
     "Colorspace conversions with pinned matrices."),
    ("img.geometry", "image", "transform", "flip, transpose, rotate-90",
     "", STATUS_PENDING, "A",
     "Exact geometry transforms."),
    ("img.geometry-cont", "image", "transform", "rotate, resize, crop, pad",
     "", STATUS_PENDING, "B",
     "Continuous geometry with pinned interpolation."),
    ("img.filters", "image", "transform", "blur, sharpen, sobel, canny, erode, dilate",
     "", STATUS_PENDING, "A",
     "Fixed-kernel filters."),
    ("img.codec", "image", "encode", "png, jpeg, webp",
     "", STATUS_PENDING, "B",
     "Image codecs with pinned versions and normalized metadata."),
    # ---- tables ----
    ("tab.select-dims", "tables", "select", "select_dims",
     "src.slp.make_select_dims", STATUS_BUILT, "A",
     "Keep only the given columns."),
    ("tab.one-hot", "tables", "transform", "one-hot, label-encode",
     "", STATUS_PENDING, "A",
     "Encodings over a registered class list."),
    ("tab.stats", "tables", "transform", "standardize, normalize, pca",
     "", STATUS_PENDING, "A",
     "Fits at registration, applies at serve time."),
    ("tab.agg", "tables", "reduce", "group-by, pivot, crosstab",
     "", STATUS_PENDING, "A",
     "Deterministic aggregation."),
    ("tab.join", "tables", "combine", "join, concat, merge",
     "", STATUS_PENDING, "A",
     "Relational combination."),
    ("tab.io", "tables", "encode", "csv, tsv, parquet subset",
     "", STATUS_PENDING, "B",
     "Table formats with bounded parsers."),
    # ---- code ----
    ("code.programmatic", "code", "construct", "ProgrammaticPrimitive",
     "src.programmatic_primitive.ProgrammaticPrimitive", STATUS_BUILT, "B",
     "Wrap any registered pure function as a primitive."),
)


def catalog() -> list[dict[str, Any]]:
    """Return the full SLP inventory as a list of records."""
    return [
        {
            "id": id_,
            "domain": domain,
            "family": family,
            "name": name,
            "access": access,
            "status": status,
            "tier": tier,
            "summary": summary,
        }
        for (id_, domain, family, name, access, status, tier, summary) in _ENTRIES
    ]


def render_markdown() -> str:
    """Render the inventory as the checked-in CATALOG.md body."""
    lines = [
        "# GEODE Standard Library of Primitives -- catalog",
        "",
        "Generated from `src/slp/_catalog.py`. Do not edit by hand.",
        "Status: BUILT = importable today; PENDING = launch backlog.",
        "Tiers: A = launch-ready, B = needs a pinned dependency and a",
        "determinism certificate. The full possibility space:",
        "`analysis/SLP_POSSIBILITY_SPACE_v1.md`.",
        "",
        "| ID | Domain | Family | Name | Access | Status | Tier | Summary |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in catalog():
        lines.append(
            "| {id} | {domain} | {family} | {name} | {access} | {status} | "
            "{tier} | {summary} |".format(**entry)
        )
    lines += [
        "",
        "## Counts",
        "",
    ]
    built = sum(1 for e in catalog() if e["status"] == STATUS_BUILT)
    pending = sum(1 for e in catalog() if e["status"] == STATUS_PENDING)
    lines.append(f"- BUILT: {built}")
    lines.append(f"- PENDING: {pending}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_markdown())
