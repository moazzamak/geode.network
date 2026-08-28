# GEODE Standard Library of Primitives -- catalog

Generated from `src/slp/_catalog.py`. Do not edit by hand.
Status: BUILT = importable today; PENDING = launch backlog.
Tiers: A = launch-ready, B = needs a pinned dependency and a
determinism certificate. The full possibility space:
`analysis/SLP_POSSIBILITY_SPACE_v1.md`.

| ID | Domain | Family | Name | Access | Status | Tier | Summary |
|---|---|---|---|---|---|---|---|
| mem.count | memory | memory | count-based memory | src.programmatic_memory.ProgrammaticMemory | BUILT | A | Variable-order count memory: register, backoff lookup, predict-next. |
| math.scale | math | transform | scale | src.slp.make_scale | BUILT | A | Multiply every feature by a scalar. |
| math.clip | math | transform | clip | src.slp.make_clip | BUILT | A | Clip values to [low, high]. |
| math.affine | math | transform | affine | src.slp.make_affine | BUILT | A | Affine transform Y = X @ A + b. |
| math.l2-normalize | math | transform | l2_normalize | src.slp.make_l2_normalize | BUILT | A | Normalize each row to unit L2 length. |
| math.reductions | math | reduce | sum, mean, var, std, min, max, argmin, argmax |  | PENDING | A | Axis reductions with the axis fixed at registration. |
| math.softmax | math | transform | softmax, log-softmax |  | PENDING | A | Softmax family over a fixed axis, float64. |
| math.activations | math | transform | sigmoid, tanh, relu, gelu |  | PENDING | A | Pinned activation functions. |
| math.exp-log | math | transform | log, exp, sqrt, pow, abs, sign |  | PENDING | A | Elementwise elementary functions in float64. |
| math.linalg | math | solve | solve, pinv, det, trace, svd, eig |  | PENDING | A | Linear algebra in float64, pinned LAPACK path. |
| math.distances | math | compare | euclidean, cosine, manhattan |  | PENDING | A | Distance and similarity functions. |
| math.prob-tables | math | solve | cdf, ppf, pdf for fixed families |  | PENDING | A | Normal, student, beta, gamma probability functions. |
| sym.simplify | symbolic | transform | parse, simplify, expand, factor, substitute |  | PENDING | B | Symbolic manipulation on a pinned computer-algebra engine. |
| sym.calculus | symbolic | reduce | differentiate, integrate |  | PENDING | B | Symbolic calculus, bounded. |
| sym.solve | symbolic | solve | linear, polynomial, small nonlinear |  | PENDING | B | Symbolic solving with a resource cap. |
| sym.latex | symbolic | encode | to-latex, to-mathml |  | PENDING | B | Render an expression tree. |
| logic.threshold | logic | transform | threshold | src.slp.make_threshold | BUILT | A | Binarize: 1.0 where X > value, else 0.0. |
| logic.and | logic | combine | logical_and | src.slp.make_logical_and | BUILT | A | Column-wise AND. |
| logic.or | logic | combine | logical_or | src.slp.make_logical_or | BUILT | A | Column-wise OR. |
| logic.not-xor | logic | combine | not, xor, nand, nor |  | PENDING | A | Remaining boolean algebra. |
| logic.compare | logic | compare | eq, ne, lt, le, gt, ge |  | PENDING | A | Elementwise comparisons. |
| logic.where | logic | select | where, select, argwhere |  | PENDING | A | Conditional selection. |
| logic.sets | logic | combine | unique, intersect, union, difference |  | PENDING | A | Set operations with deterministic ordering. |
| sig.delay | signal | transform | delay | src.slp.make_delay | BUILT | A | Ring-buffer delay over a stream. |
| sig.mel | signal | convert | mel_spectrogram | geode.core.audio_primitives.mel_spectrogram | BUILT | A | FFT/mel front-end, bit-exact (M267 stage 0). |
| sig.fft | signal | convert | fft, rfft, ifft |  | PENDING | A | Fourier transforms on a pinned backend. |
| sig.stft | signal | convert | stft, spectrogram |  | PENDING | B | Short-time spectra with pinned windows. |
| sig.smoothing | signal | transform | moving average, median filter |  | PENDING | A | Fixed-window smoothing. |
| sig.resample | signal | convert | resample |  | PENDING | A | Fixed-ratio resampling with pinned interpolation. |
| text.tokenize | text | transform | tokenize |  | PENDING | B | Tokenization with a pinned vocabulary and merges. |
| text.normalize | text | transform | unicode-normalize, casefold, whitespace |  | PENDING | A | Text normalization with a pinned Unicode version. |
| text.ngrams | text | transform | n-grams, bag-of-words, tf-idf |  | PENDING | A | N-gram features with stats fitted at registration. |
| text.edit-distance | text | compare | levenshtein, damerau |  | PENDING | A | Edit distances. |
| text.hash-encode | text | encode | sha256, blake3, keccak, base64, base58, hex |  | PENDING | A | Hashing and text encodings. |
| text.regex | text | select | regex search and replace |  | PENDING | B | Bounded regular expressions with a resource cap. |
| img.grayscale | image | convert | grayscale, invert, normalize |  | PENDING | A | Pixel-space conversions with pinned coefficients. |
| img.colorspace | image | convert | rgb-hsv, rgb-yuv, rgb-lab |  | PENDING | A | Colorspace conversions with pinned matrices. |
| img.geometry | image | transform | flip, transpose, rotate-90 |  | PENDING | A | Exact geometry transforms. |
| img.geometry-cont | image | transform | rotate, resize, crop, pad |  | PENDING | B | Continuous geometry with pinned interpolation. |
| img.filters | image | transform | blur, sharpen, sobel, canny, erode, dilate |  | PENDING | A | Fixed-kernel filters. |
| img.codec | image | encode | png, jpeg, webp |  | PENDING | B | Image codecs with pinned versions and normalized metadata. |
| tab.select-dims | tables | select | select_dims | src.slp.make_select_dims | BUILT | A | Keep only the given columns. |
| tab.one-hot | tables | transform | one-hot, label-encode |  | PENDING | A | Encodings over a registered class list. |
| tab.stats | tables | transform | standardize, normalize, pca |  | PENDING | A | Fits at registration, applies at serve time. |
| tab.agg | tables | reduce | group-by, pivot, crosstab |  | PENDING | A | Deterministic aggregation. |
| tab.join | tables | combine | join, concat, merge |  | PENDING | A | Relational combination. |
| tab.io | tables | encode | csv, tsv, parquet subset |  | PENDING | B | Table formats with bounded parsers. |
| code.programmatic | code | construct | ProgrammaticPrimitive | src.programmatic_primitive.ProgrammaticPrimitive | BUILT | B | Wrap any registered pure function as a primitive. |

## Counts

- BUILT: 12
- PENDING: 36

