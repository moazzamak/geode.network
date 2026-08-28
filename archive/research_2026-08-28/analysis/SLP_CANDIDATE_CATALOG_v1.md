# GEODE Standard Library of Primitives — Candidate Catalog (v1, 25 Aug 2026)

The launch catalog (whitepaper, 8 categories) is the audited core.
This file is the idea register: candidates for the standard library
(SLP), ranked by admission risk. The full possibility space --- 36
domains × 18 operation families, ~720 named primitives, including
the tool-call equivalence map --- is
enumerated in `SLP_POSSIBILITY_SPACE_v1.md`. The entry rule is fixed (25 Aug):
pure, deterministic, resource-bounded, pinned to an exact dependency
version; no randomness, no network, no wall clock. Anything that
fails the rule is not a primitive. It is an off-path service or a
third-party primitive, where admission applies the same discipline.

Tiers:

- **A — launch-ready.** Pure numeric/logic/table/text code with no
  risky dependencies. Admits immediately after the standard
  determinism test.
- **B — needs a pin + certificate.** Deterministic only under a
  pinned dependency version and a resource cap (codecs, filterbanks,
  CAS, regex). Each item admits only with a reference-vs-reference
  determinism certificate measured on registered hardware.
- **C — excluded by rule.** Network access, wall clock, OS
  randomness, or unbounded I/O. Never a primitive.

---

## 1. Memory (extends the built count-based memory)

| Primitive                                          | Tier      | Note                                                 |
| -------------------------------------------------- | --------- | ---------------------------------------------------- |
| Count-based variable-order memory                  | A (built) | `src/programmatic_memory.py`; package as `Primitive` |
| Key-value store (session-scoped, capacity-bounded) | A         | No persistence; dies with the session                |
| Priority queue / top-k over a stream               | A         | Bounded capacity                                     |
| Ring buffer (already: delay)                       | A (built) |                                                      |
| Bloom filter (fixed size, pinned hash)             | A         | Probabilistic, deterministic                         |
| Trie / prefix lookup over a registered vocabulary  | A         |                                                      |
| LRU cache primitive (bounded)                      | A         |                                                      |

## 2. Numeric math (extends the built core)

| Primitive                                                     | Tier      | Note                                   |
| ------------------------------------------------------------- | --------- | -------------------------------------- |
| scale, clip, affine, l2-normalize                             | A (built) |                                        |
| reductions: sum, mean, var, std, min, max, argmin, argmax     | A         | Axis parameter fixed at registration   |
| softmax, log-softmax, sigmoid, tanh, relu, gelu               | A         | Pinned implementation                  |
| log, exp, sqrt, pow, abs, sign, floor, ceil, round            | A         | Float64                                |
| matmul, transpose, inverse, pinv, solve, det, trace, SVD, eig | A         | Float64 promotion; pinned LAPACK path  |
| exact integer arithmetic, factorial, binomial, gcd, lcm       | A         | Python ints are exact                  |
| distances: euclidean, cosine, manhattan, minkowski            | A         |                                        |
| probability tables: normal/student/beta/gamma cdf, ppf, pdf   | A         | Fixed algorithms (continued fractions) |
| discrete entropy, KL divergence                               | A         |                                        |
| seeded PRNG stream (SHAKE-based, seed is an argument)         | A         | Deterministic; seed explicit           |
| units conversion (SI, imperial, energy)                       | A         | Static tables                          |

## 3. Symbolic math (CAS)

| Primitive                                          | Tier | Note                      |
| -------------------------------------------------- | ---- | ------------------------- |
| parse, simplify, expand, factor                    | B    | Pinned CAS version        |
| substitute, evaluate (exact and float64)           | B    |                           |
| differentiate, integrate (symbolic)                | B    |                           |
| solve: linear, polynomial, small nonlinear systems | B    | Resource-bounded          |
| LaTeX render of an expression                      | B    | Pure function of the tree |
| units-of-measure arithmetic                        | B    | Static tables             |

## 4. Logic and sets

| Primitive                                            | Tier      | Note                            |
| ---------------------------------------------------- | --------- | ------------------------------- |
| threshold, logical-and, logical-or                   | A (built) |                                 |
| logical-not, xor, nand, nor                          | A         |                                 |
| comparisons: eq, ne, lt, le, gt, ge                  | A         | Elementwise                     |
| where / select / argwhere                            | A         |                                 |
| unique, intersect, union, difference (sorted output) | A         | Deterministic ordering          |
| bitwise ops on integers                              | A         |                                 |
| truth-table evaluation of small boolean expressions  | A         | Bounded expression size         |
| decision-table primitive (registered rule set)       | A         | The router pattern, generalized |

## 5. Signal processing

| Primitive                                      | Tier      | Note                                                            |
| ---------------------------------------------- | --------- | --------------------------------------------------------------- |
| delay / ring buffer                            | A (built) |                                                                 |
| FFT, rFFT, iFFT                                | A         | Pinned FFT backend                                              |
| STFT, spectrogram, mel filterbank              | B         | Filterbank coefficients pinned; mel stage exists in experiments |
| convolution, correlation                       | A         | Fixed kernels                                                   |
| moving average, median filter                  | A         |                                                                 |
| window functions: hann, hamming, blackman      | A         |                                                                 |
| resampling (fixed ratio, pinned interpolation) | A         |                                                                 |
| RMS energy, zero-crossing rate, peak detection | A         |                                                                 |
| autocorrelation                                | A         |                                                                 |

## 6. Text and documents

| Primitive                                           | Tier | Note                                |
| --------------------------------------------------- | ---- | ----------------------------------- |
| tokenizer (pinned vocab + merges)                   | B    | Vocab is registered data            |
| Unicode normalization (NFKC/NFC), casefold          | A    | Pinned Unicode version              |
| whitespace normalization, trimming                  | A    |                                     |
| character n-grams, bag-of-words, TF-IDF             | A    | Stats fitted at registration        |
| edit distance: Levenshtein, Damerau                 | A    |                                     |
| Jaccard / token-set similarity                      | A    |                                     |
| phonetic encodings: Soundex, Metaphone              | A    | Static tables                       |
| language profile detection (pinned n-gram profiles) | B    | Profile data registered             |
| regex search/replace (bounded)                      | B    | ReDoS risk → resource cap mandatory |
| hashing: sha256, blake3, keccak                     | A    |                                     |
| base64, base58, hex encode/decode                   | A    |                                     |
| JSON / CSV / MessagePack parse and emit             | B    | Depth- and size-bounded parsers     |
| template substitution                               | A    |                                     |

## 7. Image processing

| Primitive                                          | Tier | Note                                                  |
| -------------------------------------------------- | ---- | ----------------------------------------------------- |
| grayscale, invert, normalize                       | A    | Pinned coefficients                                   |
| colorspace: RGB↔HSV, RGB↔YUV, RGB↔Lab              | A    | Pinned matrices                                       |
| flip, transpose, rotate (multiples of 90°)         | A    | Exact                                                 |
| rotate (arbitrary angle), resize, crop, pad        | B    | Pinned interpolation kernel                           |
| gaussian blur, median, sobel, canny, erode, dilate | A    | Fixed kernels                                         |
| threshold (fixed and Otsu)                         | A    | Otsu is deterministic                                 |
| histogram (fixed bins), color moments              | A    |                                                       |
| LBP, HOG features                                  | A    | Fixed cell geometry                                   |
| PNG/JPEG decode and encode                         | B    | Pinned codec version + rejection of metadata variance |
| QR encode / decode                                 | B    | Pinned library                                        |

## 8. Tables and features

| Primitive                                              | Tier      | Note                   |
| ------------------------------------------------------ | --------- | ---------------------- |
| select-dims                                            | A (built) |                        |
| one-hot encode / decode, label encode                  | A         | Registered class list  |
| standardize / normalize (stats fitted at registration) | A         |                        |
| PCA / whitening (fitted at registration)               | A         | Float64                |
| imputation: mode, median, registered constant          | A         |                        |
| schema validation and type conversion                  | A         |                        |
| filter / join / group-aggregate over tabular input     | A         | Deterministic ordering |
| z-score outlier flags                                  | A         |                        |

## 9. Time series

| Primitive                                  | Tier | Note                   |
| ------------------------------------------ | ---- | ---------------------- |
| rolling mean, std, min, max                | A    | Fixed window           |
| exponential smoothing (fixed alpha)        | A    |                        |
| differencing, lag features                 | A    | delay generalizes this |
| autocorrelation, seasonality decomposition | A    | Fixed windows          |

## 10. Geometry and spatial

| Primitive                          | Tier | Note           |
| ---------------------------------- | ---- | -------------- |
| haversine distance, bounding boxes | A    |                |
| point-in-polygon (ray casting)     | A    |                |
| convex hull, area, perimeter       | A    | Float64 policy |
| WKT/WKB parse and emit (bounded)   | B    | Pinned parser  |

## 11. Cryptography-adjacent (pure, no secrets)

| Primitive                               | Tier | Note                                 |
| --------------------------------------- | ---- | ------------------------------------ |
| sha256 / blake3 / keccak hashing, HMAC  | A    |                                      |
| Merkle root of an ordered list          | A    | The ledger's own building block      |
| secp256k1 signature verification        | A    | Deterministic                        |
| key derivation from a given seed (HKDF) | A    | Seed is an argument; no key material |

## 12. Explicitly excluded (Tier C)

These are not primitives. They are off-path services or third-party
code, and the rule excludes them from the SLP: network calls (price
feeds, web fetch, DNS), wall-clock access, OS randomness, file
system access, and anything with unbounded runtime or memory. A
contributor that wants them builds a third-party primitive and
accepts the sandbox contract.

## Placement policy (25 Aug, user overrule: broad by default)

- The standard library is broad on purpose. A library with few
  options is a demo, not infrastructure. Usefulness at launch
  requires packaging as many capabilities as the entry rule
  allows.
- Every Tier-A candidate is packaged for launch. Tier-B candidates
  admit as their dependency pins and determinism certificates
  land, in parallel with launch, not instead of it.
- The development fund carries the maintenance duty for the broad
  library: security patches, pin upgrades, and re-certification
  of each pinned dependency. This is a registered cost line, not
  an afterthought.
- One boundary protects contributors: the standard library holds
  code-defined transforms only. It never holds learned models.
  A broad free library therefore cannot compete with contributor
  arms. It only feeds them.
- Third-party primitives remain for what the network does not
  maintain and for what fails the entry rule. They are paid,
  challenged at admission, sandboxed in microVMs, and
  replay-verified.
