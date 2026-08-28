# GEODE Standard Library — the Possibility Space (v1, 25 Aug 2026)

This document exhausts the primitive possibility space. It is not
the launch catalog (`SLP_CANDIDATE_CATALOG_v1.md` ranks launch
readiness). It is the map of everything that can be packaged under
the entry rule: pure, deterministic, resource-bounded, pinned; no
randomness, no network, no wall clock.

Primitives are tool calls, under discipline. An LLM tool call and
a GEODE primitive are the same idea: a named, typed function the
system can invoke. The difference is the discipline. A primitive
is registered, replayable, measured, and paid per execution. A
learned tool (an embedder, a transcriber, a generator) is an arm,
admitted by measurement, not a standard-library entry. The
tool-call equivalence map at the end of this document locates the
common LLM tool categories.

## The generator principle

The space is closed and enumerable. Every primitive is a
deterministic function over typed data. Enumerate the **domains**
(what the data is) and the **operation families** (what a function
does to it). Every primitive is a cell of the domain × family
table. Composition stitches cells into pipelines, so the reachable
space is the closure of the cells under wiring — which is exactly
what the registry and router already model.

**The 36 domains:**

1. Numbers (exact, float, complex, bignum)
2. Vectors and matrices
3. Tensors (n-dimensional arrays)
4. Sets and multisets
5. Sequences and streams
6. Time series and signals
7. Text and strings
8. Documents and corpora
9. Images (raster)
10. Audio
11. Graphs and networks
12. Trees and hierarchies
13. Tables and relational data
14. Geometry (2-D/3-D)
15. Spatial and geo
16. Probability and statistics
17. Deterministic randomness (seeded streams)
18. Symbolic mathematics
19. Formal logic and automata
20. Code and bytecode
21. Serialization and encodings
22. Compression
23. Cryptography and hashing (pure)
24. Units and measurement
25. Dates and calendars
26. Intervals and ranges
27. Finance and trading series
28. Chemistry and molecules
29. Physics and deterministic simulation
30. Knowledge representation
31. Search and indexing
32. Verification and proofs
33. Video
34. Blockchain and EVM data
35. Forms and documents of record
36. Query languages

**The 18 operation families:**

1. Construct — build a value from parameters.
2. Validate — check a contract, return typed verdicts.
3. Select — index, slice, mask, project, filter.
4. Transform — map, normalize, feature-map.
5. Reduce — aggregate, summarize, fold.
6. Combine — merge, concat, join, multiply, compose.
7. Compare — distance, similarity, alignment.
8. Order — sort, rank, top-k.
9. Encode — serialize into a standard format.
10. Decode — parse a standard format.
11. Hash — fingerprint, digest.
12. Fit-then-apply — statistics fitted at registration (standardize, PCA, vocab).
13. Search — membership, lookup, nearest neighbor over registered data.
14. Solve — linear systems, equations, optimization, root finding.
15. Simulate — one deterministic step of a model.
16. Generate — seeded, parameterized production.
17. Measure — metrics, quality scores, error rates.
18. Convert — type, unit, and coordinate conversion.

The enumeration below lists each domain's cells. ~720 concrete
primitives follow. Tiers follow the catalog convention: A =
launch-ready, B = needs a pinned dependency + determinism
certificate, C = excluded by rule (listed only to mark the
boundary).

---

## 1. Numbers

| Family    | Primitives                                                                         | Tier |
| --------- | ---------------------------------------------------------------------------------- | ---- |
| Transform | abs, sign, floor, ceil, round, sqrt, cbrt, pow, exp, log, log2, log10, trig, atan2 | A    |
| Reduce    | sum, product, min, max                                                             | A    |
| Construct | constants (pi, e), arithmetic progressions                                         | A    |
| Convert   | int↔float, degrees↔radians, exact↔float                                            | A    |
| Compare   | is-close (registered tolerance)                                                    | A    |
| Solve     | gcd, lcm, modular inverse, CRT, primality (deterministic)                          | A    |

## 2. Vectors and matrices

| Family         | Primitives                                                                           | Tier |
| -------------- | ------------------------------------------------------------------------------------ | ---- |
| Transform      | transpose, scale, shift, normalize (L1/L2/Linf), softmax, logsumexp, sign-flip, clip | A    |
| Reduce         | row/col sums, means, norms, argmin/argmax                                            | A    |
| Combine        | matmul, dot, outer, kronecker, hadamard, cat                                         | A    |
| Solve          | solve, least-squares, pinv, det, trace, rank, LU/QR/SVD/eig, ridge (the head)        | A    |
| Compare        | cosine, euclidean, mahalanobis (registered stats), correlation                       | A    |
| Fit-then-apply | standardize, whiten, PCA, centering (registered at admission)                        | A    |
| Convert        | dense↔sparse, float32↔float64                                                        | A    |

## 3. Tensors

| Family    | Primitives                                                              | Tier |
| --------- | ----------------------------------------------------------------------- | ---- |
| Transform | reshape, permute, flatten, broadcast, pad, pool (max/mean), convolution | A    |
| Select    | gather, slice, tile                                                     | A    |
| Reduce    | per-axis sums/means/mins, softmax over axis                             | A    |
| Combine   | concat, stack                                                           | A    |
| Encode    | to numpy/memmap spec                                                    | A    |

## 4. Sets and multisets

| Family    | Primitives                                                               | Tier |
| --------- | ------------------------------------------------------------------------ | ---- |
| Construct | empty, from-list                                                         | A    |
| Transform | map, filter                                                              | A    |
| Reduce    | cardinality, counts (multiset)                                           | A    |
| Combine   | union, intersection, difference, symmetric difference, cartesian product | A    |
| Compare   | Jaccard, containment, overlap coefficient                                | A    |
| Order     | sorted list form (deterministic)                                         | A    |

## 5. Sequences and streams

| Family    | Primitives                                | Tier |
| --------- | ----------------------------------------- | ---- |
| Transform | map, filter, dedupe, chunk, window        | A    |
| Reduce    | first, last, count, fold                  | A    |
| Combine   | concat, zip, interleave                   | A    |
| Select    | take, drop, slice, nth                    | A    |
| Order     | sort, stable sort, unique                 | A    |
| Search    | membership, subsequence search            | A    |
| Generate  | ranges, arithmetic/geometric progressions | A    |

## 6. Time series and signals

| Family    | Primitives                                                                                | Tier                |
| --------- | ----------------------------------------------------------------------------------------- | ------------------- |
| Transform | delay/ring buffer, differencing, rolling mean/std/min/max, exponential smoothing, detrend | A                   |
| Combine   | align, resample (fixed ratio)                                                             | A                   |
| Reduce    | mean, variance, autocorrelation, entropy                                                  | A                   |
| Compare   | DTW (registered), correlation                                                             | A                   |
| Solve     | linear prediction, AR coefficients                                                        | A                   |
| Convert   | spectrum: FFT/rFFT/iFFT, STFT, spectrogram, mel                                           | B (filterbank pins) |
| Generate  | seeded sinusoids, impulses, noise                                                         | A                   |

## 7. Text and strings

| Family    | Primitives                                                                                      | Tier |
| --------- | ----------------------------------------------------------------------------------------------- | ---- |
| Transform | casefold, trim, normalize whitespace, Unicode normalize (NFKC), strip accents, replace, reverse | A    |
| Select    | substring, split, extract by regex (bounded)                                                    | B    |
| Reduce    | length, word count, character histogram                                                         | A    |
| Combine   | concat, join, template substitution                                                             | A    |
| Compare   | Levenshtein, Damerau, Jaccard (tokens), cosine (vectors)                                        | A    |
| Order     | lexicographic sort                                                                              | A    |
| Encode    | base64, base58, hex, URL-encode                                                                 | A    |
| Decode    | the inverses                                                                                    | A    |
| Hash      | sha256, blake3, keccak                                                                          | A    |
| Search    | substring, prefix, suffix, fuzzy (bounded)                                                      | A    |
| Convert   | transliteration, phonetic codes (Soundex, Metaphone), case systems                              | A    |

## 8. Documents and corpora

| Family    | Primitives                                                                                        | Tier |
| --------- | ------------------------------------------------------------------------------------------------- | ---- |
| Transform | tokenization (pinned vocab/merges), n-grams, stopword filter (registered list), stemming (pinned) | B    |
| Reduce    | bag-of-words, TF, TF-IDF (registered stats)                                                       | A    |
| Compare   | document similarity (cosine, Jaccard, BM25 over registered index)                                 | A    |
| Select    | sentence/paragraph split, headline extraction                                                     | A    |
| Decode    | PDF/HTML/Markdown extraction (bounded parsers)                                                    | B    |
| Measure   | readability scores, perplexity vs registered model                                                | A    |

## 9. Images

| Family        | Primitives                                                                                | Tier                   |
| ------------- | ----------------------------------------------------------------------------------------- | ---------------------- |
| Transform     | grayscale, invert, threshold (fixed/Otsu), normalize, gamma                               | A                      |
| Transform     | flip, transpose, rotate (90° multiples exact; arbitrary pinned kernel), resize, crop, pad | B (interpolation pins) |
| Transform     | blur (gaussian/median), sharpen, sobel, laplacian, canny, erode, dilate                   | A                      |
| Convert       | colorspace RGB↔HSV/HSL/YUV/Lab, bit depth, channel reorder                                | A                      |
| Reduce        | histogram (fixed bins), color moments, mean/variance                                      | A                      |
| Measure       | PSNR, SSIM (pinned), LBP, HOG                                                             | A                      |
| Encode/Decode | PNG, JPEG, WebP (pinned codec versions, metadata-normalized)                              | B                      |
| Generate      | seeded gradients, checkerboards, test charts                                              | A                      |
| Search        | template match (fixed template), QR encode/decode                                         | B                      |

## 10. Audio

| Family        | Primitives                                                         | Tier |
| ------------- | ------------------------------------------------------------------ | ---- |
| Transform     | gain, clip, fade, mono↔stereo mix, resample, time-stretch (pinned) | B    |
| Reduce        | RMS, peak, zero-crossing rate, loudness (fixed standard)           | A    |
| Combine       | mix, concatenate                                                   | A    |
| Convert       | FFT, STFT, mel, MFCC                                               | B    |
| Encode/Decode | WAV, FLAC (pinned)                                                 | B    |
| Generate      | seeded tones, chirps                                               | A    |
| Measure       | SNR, spectral centroid/rolloff                                     | A    |

## 11. Graphs and networks

| Family    | Primitives                                          | Tier |
| --------- | --------------------------------------------------- | ---- |
| Construct | from edge list (ordered)                            | A    |
| Select    | neighbors, subgraph, degree filter                  | A    |
| Reduce    | degrees, components, triangle count                 | A    |
| Solve     | shortest path (Dijkstra), spanning tree, max flow   | A    |
| Compare   | isomorphism (small graphs), edit distance (bounded) | B    |
| Convert   | adjacency↔edge list, formats (GraphML, DIMACS)      | B    |

## 12. Trees and hierarchies

| Family    | Primitives                                  | Tier |
| --------- | ------------------------------------------- | ---- |
| Construct | from parent list, from nested dict          | A    |
| Transform | map over nodes, prune, relabel              | A    |
| Reduce    | depth, size, leaf count                     | A    |
| Select    | ancestors, descendants, path                | A    |
| Order     | pre/post/in-order traversal (deterministic) | A    |
| Compare   | tree edit distance (bounded)                | B    |
| Decode    | XML/JSON/EDN subset (bounded parsers)       | B    |

## 13. Tables and relational data

| Family        | Primitives                                                                                    | Tier |
| ------------- | --------------------------------------------------------------------------------------------- | ---- |
| Select        | filter rows, project columns, sample (seeded)                                                 | A    |
| Transform     | map column, one-hot, label encode, impute (registered values), standardize (registered stats) | A    |
| Reduce        | group-by aggregates, pivot, crosstab                                                          | A    |
| Combine       | join (inner/left/outer), concat, merge                                                        | A    |
| Order         | sort by keys (stable)                                                                         | A    |
| Encode/Decode | CSV, TSV, Parquet subset, Arrow subset                                                        | B    |
| Validate      | schema check, type coercion                                                                   | A    |
| Measure       | null rates, cardinalities, correlations                                                       | A    |

## 14. Geometry

| Family    | Primitives                                                                | Tier |
| --------- | ------------------------------------------------------------------------- | ---- |
| Construct | point, segment, polygon, mesh (from ordered vertices)                     | A    |
| Transform | translate, rotate, scale, affine, reflect                                 | A    |
| Reduce    | centroid, bounding box, area, perimeter, volume                           | A    |
| Compare   | distance point-to-point/segment/polygon, Hausdorff (bounded)              | A    |
| Solve     | convex hull, intersection, point-in-polygon, closest point, triangulation | A    |
| Convert   | WKT/WKB/GeoJSON (pinned parsers)                                          | B    |

## 15. Spatial and geo

| Family  | Primitives                                                                         | Tier |
| ------- | ---------------------------------------------------------------------------------- | ---- |
| Convert | coordinate projections (registered EPSG tables), tile↔coordinate (XYZ slippy maps) | A    |
| Compare | haversine, geodesic (pinned ellipsoid), bounding-box overlap                       | A    |
| Solve   | geofence containment, nearest point on route                                       | A    |
| Decode  | GeoJSON, GPX (bounded)                                                             | B    |

## 16. Probability and statistics

| Family         | Primitives                                                                                      | Tier |
| -------------- | ----------------------------------------------------------------------------------------------- | ---- |
| Transform      | softmax, log-probs, Bayes update (discrete)                                                     | A    |
| Reduce         | mean, variance, quantiles, covariance, correlation                                              | A    |
| Solve          | cdf/ppf/pdf for fixed families (normal, student, beta, gamma, binomial), MLE for fixed families | A    |
| Compare        | KL, JS divergence, entropy, mutual information (discrete)                                       | A    |
| Fit-then-apply | standardization, PCA, calibration (registered)                                                  | A    |
| Generate       | seeded draws via SHAKE-stream (explicit seed)                                                   | A    |

## 17. Deterministic randomness

| Family    | Primitives                                                                          | Tier |
| --------- | ----------------------------------------------------------------------------------- | ---- |
| Generate  | seeded PRNG stream (SHAKE), uniform/exponential/normal draws, permutations, samples | A    |
| Construct | seed = explicit argument; no OS entropy                                             | A    |

## 18. Symbolic mathematics

| Family    | Primitives                                                                        | Tier |
| --------- | --------------------------------------------------------------------------------- | ---- |
| Construct | parse expression (bounded grammar)                                                | B    |
| Transform | simplify, expand, factor, collect, substitute                                     | B    |
| Solve     | linear systems, polynomial roots, small nonlinear systems, ODE (closed form only) | B    |
| Reduce    | differentiate, integrate (symbolic), limit (bounded)                              | B    |
| Compare   | structural equality, numerical equivalence (sampled)                              | B    |
| Encode    | LaTeX, MathML rendering                                                           | B    |
| Convert   | units-of-measure arithmetic                                                       | B    |

## 19. Formal logic and automata

| Family    | Primitives                                              | Tier |
| --------- | ------------------------------------------------------- | ---- |
| Transform | normalize (CNF/DNF), negate                             | A    |
| Reduce    | truth-table evaluation, satisfiability (bounded search) | A    |
| Combine   | boolean algebra: and/or/not/xor/nand/nor                | A    |
| Construct | finite automata, regular expressions (bounded)          | B    |
| Solve     | regex matching (resource-capped), DFA/NFA simulation    | B    |
| Decode    | small declarative rule languages (decision tables)      | B    |

## 20. Code and bytecode

| Family    | Primitives                                                      | Tier |
| --------- | --------------------------------------------------------------- | ---- |
| Validate  | syntax check (pinned parser), type check (registered spec)      | B    |
| Transform | formatting (pinned formatter), AST transforms (bounded)         | B    |
| Reduce    | cyclomatic complexity, LOC, symbol table extraction             | B    |
| Decode    | parse source into AST (pinned grammar)                          | B    |
| Encode    | pretty-print, bytecode assemble (pinned)                        | B    |
| Measure   | test coverage over registered cases                             | B    |
| Execute   | the sandboxed engine itself (runs any registered pure function) | B    |

## 21. Serialization and encodings

| Family        | Primitives                                                                                           | Tier |
| ------------- | ---------------------------------------------------------------------------------------------------- | ---- |
| Encode/Decode | JSON, CSV, XML subset, YAML subset, TOML, MessagePack, CBOR, Proto (pinned schemas), Base16/32/58/64 | B    |
| Validate      | schema validation per format                                                                         | B    |
| Convert       | format-to-format (via internal table model)                                                          | B    |

## 22. Compression

| Family        | Primitives                                               | Tier |
| ------------- | -------------------------------------------------------- | ---- |
| Encode/Decode | zlib, gzip, LZ4, Zstandard, Brotli (pinned levels)       | B    |
| Measure       | compression ratio, entropy estimate                      | A    |
| Note          | lossy codecs (image/audio) live in their domain sections | —    |

## 23. Cryptography and hashing (pure, no secrets)

| Family    | Primitives                                            | Tier |
| --------- | ----------------------------------------------------- | ---- |
| Hash      | sha256, blake3, keccak, HMAC                          | A    |
| Construct | Merkle root of ordered lists                          | A    |
| Verify    | secp256k1 signature verify, Ed25519 verify            | A    |
| Convert   | public-key↔address, Base58Check                       | A    |
| Solve     | HKDF from a given seed                                | A    |
| Note      | signing lives host-side; the sandbox never holds keys | —    |

## 24. Units and measurement

| Family   | Primitives                                                                    | Tier |
| -------- | ----------------------------------------------------------------------------- | ---- |
| Convert  | SI/imperial length, mass, time, energy, power, temperature scales, data sizes | A    |
| Validate | dimensional consistency check                                                 | A    |
| Note     | currency conversion is Tier C (needs a feed)                                  | —    |

## 25. Dates and calendars

| Family    | Primitives                                           | Tier |
| --------- | ---------------------------------------------------- | ---- |
| Construct | date from components (proleptic Gregorian, UTC only) | A    |
| Transform | add days/months, weekday, week number                | A    |
| Compare   | duration between dates, ordering                     | A    |
| Convert   | ISO-8601 parse/emit, epoch seconds                   | A    |
| Note      | current time is Tier C (wall clock)                  | —    |

## 26. Intervals and ranges

| Family    | Primitives                          | Tier |
| --------- | ----------------------------------- | ---- |
| Construct | closed/open intervals, ranges       | A    |
| Combine   | union, intersection, merge overlaps | A    |
| Compare   | containment, overlap length         | A    |
| Reduce    | coverage length, gap detection      | A    |

## 27. Finance and trading series

| Family    | Primitives                                                 | Tier |
| --------- | ---------------------------------------------------------- | ---- |
| Transform | returns, log-returns, rolling volatility, drawdown         | A    |
| Reduce    | sharpe, sortino, max drawdown, CAGR                        | A    |
| Generate  | seeded price paths (GBM) for simulation                    | A    |
| Note      | live feeds are Tier C; indicators run on given series only | —    |

## 28. Chemistry and molecules

| Family    | Primitives                                         | Tier |
| --------- | -------------------------------------------------- | ---- |
| Decode    | SMILES parse (pinned grammar), SDF parse           | B    |
| Transform | canonical SMILES, substructure search              | B    |
| Reduce    | formula mass, atom counts, ECFP/MACCS fingerprints | B    |
| Compare   | Tanimoto similarity                                | B    |

## 29. Physics and deterministic simulation

| Family   | Primitives                                                                                                            | Tier |
| -------- | --------------------------------------------------------------------------------------------------------------------- | ---- |
| Simulate | fixed-step integrators (Euler, RK4), N-body (bounded), projectile/kinematics, circuit evaluation (registered netlist) | A    |
| Solve    | linear mechanics, collision time for simple bodies                                                                    | A    |
| Note     | float policy pinned; no chaotic-model reproducibility claims                                                          | —    |

## 30. Knowledge representation

| Family    | Primitives                                                         | Tier |
| --------- | ------------------------------------------------------------------ | ---- |
| Construct | triples from ordered lists                                         | A    |
| Transform | ontology subset reasoning (registered rules), entailment (bounded) | B    |
| Select    | query by pattern (registered pattern language)                     | B    |
| Decode    | RDF/Turtle subset (pinned parser)                                  | B    |

## 31. Search and indexing

| Family    | Primitives                                                  | Tier |
| --------- | ----------------------------------------------------------- | ---- |
| Construct | inverted index, k-d tree, LSH over registered data          | A    |
| Search    | exact lookup, prefix search, nearest neighbor, top-k by key | A    |
| Measure   | recall/precision vs registered gold set                     | A    |
| Note      | the index data is registered like any reference data        | —    |

## 32. Verification and proofs

| Family  | Primitives                                                                  | Tier |
| ------- | --------------------------------------------------------------------------- | ---- |
| Verify  | checksums, digest chains, Merkle proofs                                     | A    |
| Solve   | Bulletproof verification (the head's proof), replay of registered artifacts | B    |
| Measure | proof size, verification cost                                               | A    |

---

## 33. Video

| Family        | Primitives                                              | Tier |
| ------------- | ------------------------------------------------------- | ---- |
| Select        | frame extraction at fixed indices                       | B    |
| Transform     | crop, resize, rotate frames, thumbnails                 | B    |
| Reduce        | scene-cut detection, keyframe selection                 | B    |
| Encode/Decode | pinned container and codec (H.264 subset)               | B    |
| Measure       | frame hashes, duration from header                      | B    |
| Note          | learned tasks (action recognition, captioning) are arms | —    |

## 34. Blockchain and EVM data

| Family    | Primitives                                                                | Tier |
| --------- | ------------------------------------------------------------------------- | ---- |
| Encode    | ABI encode, calldata build, RLP                                           | A    |
| Decode    | ABI decode, event-log parse, calldata parse                               | A    |
| Transform | checksum addresses, transaction hash                                      | A    |
| Verify    | Merkle proofs, signature verify (secp256k1/Ed25519), JWT parse and verify | A    |
| Construct | Merkle root of ordered leaves                                             | A    |
| Note      | signing is host-side; the sandbox never holds keys                        | —    |

## 35. Forms and documents of record

| Family    | Primitives                                     | Tier |
| --------- | ---------------------------------------------- | ---- |
| Decode    | PDF form fields, form-grid extraction (pinned) | B    |
| Transform | fill registered templates, field extraction    | B    |
| Validate  | required fields, value constraints             | B    |
| Encode    | render filled form (pinned renderer)           | B    |
| Note      | handwriting and OCR are learned arms           | —    |

## 36. Query languages

| Family   | Primitives                                                                                  | Tier |
| -------- | ------------------------------------------------------------------------------------------- | ---- |
| Solve    | SQL subset over registered tables, pattern queries over documents, path queries over graphs | B    |
| Validate | query-plan bounds (time, rows)                                                              | B    |
| Note     | one registered query language per data domain                                               | —    |

---

## Tier C boundary (listed so the space is complete)

These are deterministic only with inputs the rule forbids, or are
unbounded: live price feeds, DNS/web fetch, current time, OS
entropy, filesystem access, unbounded search. They are off-path
services, never SLP primitives.

## Tool-call equivalence map

Current LLM systems expose tool categories. This table locates
each one in GEODE. "Arm" means a learned capability, admitted by
measurement in the marketplace, not a standard-library entry.
"Tier C" means the entry rule excludes it. It ships as an off-path
service.

| LLM tool category                           | GEODE home                                                 |
| ------------------------------------------- | ---------------------------------------------------------- |
| calculator / math evaluation                | domains 1–3                                                |
| unit conversion                             | domain 24                                                  |
| date arithmetic                             | domain 25                                                  |
| code interpreter                            | domain 20 + the sandbox engine                             |
| web search (live)                           | Tier C feed; offline corpus search is domain 31            |
| document / PDF reading                      | domains 8, 35                                              |
| spreadsheet and database                    | domains 13, 36                                             |
| calendar scheduling                         | domain 25 over given events; live calendars are Tier C     |
| email draft and parse                       | domain 7; delivery is Tier C                               |
| image editing                               | domain 9                                                   |
| image understanding (OCR, captioning)       | Arm                                                        |
| audio editing                               | domain 10                                                  |
| speech-to-text, text-to-speech              | Arm                                                        |
| translation                                 | Arm                                                        |
| embeddings / vector search                  | Arm (encoders) + domain 31 (index over registered vectors) |
| retrieval / RAG                             | domain 31 + Arm embedders                                  |
| maps / geocoding / routing                  | domain 15 + a registered gazetteer (B)                     |
| weather / prices / news (live)              | Tier C feeds                                               |
| crypto / transactions                       | domain 34; signing stays host-side                         |
| summarization / classification / generation | Arm                                                        |

## How to read this space

- The space is ~36 domains × 18 families, with ~720 named
  primitives above. It is exhaustive in structure, not in naming:
  any new idea falls into one of these cells or opens a new domain
  (a new input type), which is a registry decision.
- The MVP packages the Tier-A cells of the launch catalog
  (`SLP_CANDIDATE_CATALOG_v1.md`). The rest are backlog.
- A new domain is a new input type in the task descriptor. The
  unit table extends with it, under the registered rule change.
