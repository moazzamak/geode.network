# Sparse Patch-Dictionary Classification: a sealed measurement of the joint (dictionary-size × data) scaling surface

**System under study:** the GEODE sparse patch-dictionary classifier
**Data:** DomainNet (6 domains, 345 classes, 32×32 images), shared subsample
**Status:** registered-before-measurement protocol; all evidence sealed and byte-committed

---

## Abstract

We measure how a _sparse patch-dictionary classifier_ — an image encoder built from a
whitened patch dictionary plus a closed-form (non-iterative) ridge classifier head,
with **no** learned backbone — scales with its two available resources: dictionary
size (the "atoms" axis, which is the width of the head's feature space) and training
data. On a shared DomainNet subsample (138,000 train / 34,500 test rows, 345 classes,
6 domains), we find that the accuracy "ceiling" reported for a single slice of this
family was an artifact of holding one axis fixed:

1. **The joint surface is super-additive.** Scaling dictionary size and data _together_
   beats the sum of scaling either axis alone (measured excess +0.0097/+0.0101 over a
   +0.005 margin). Quality is a function of the joint (compute × data) budget, not a
   fixed ceiling.
2. **The compute is "data-elastic".** The wider the closed-form head, the more
   productively it consumes data: per-atom learning-curve steepness rises 0.094 →
   0.115 → 0.134 as the dictionary grows 1536 → 3072 → 6144 atoms. The effect is in
   the _absolute_ gain, not the learning _rate_ (the fitted rate is width-independent).
3. **The atoms axis saturates.** Extending the dictionary past ~6144 atoms at full
   data produces a flat, slowly declining ridge (0.2246 → 0.2205 from 6144 → 16384
   atoms). The remaining lever is **data**, not dictionary size.
4. **The mechanism is structural, not solver-side.** The code space is dominated by a
   fixed ~8-dimensional structure (effective rank ≈ 7.8, flat across all dictionary
   sizes), and the correct class loses the argmax for ~78% of test samples — accuracy
   lives in a thin positive-margin tail. This is why the gap to a frozen dense trunk
   (DINOv2-small + ridge, 0.22 vs 0.54 at the large end) is structurally hard to close.

All comparisons are **cost-matched** (equal per-image multiply-accumulates): the sparse
family never dominates the dense trunk in absolute accuracy; the honest claim is
cost-matched non-domination, with the sparse family winning on data-scaling steepness
at equal cost. Every number here was produced under a registered-before-measurement
protocol with kill-switch gates and exact anchor reproduction (see Methodology), so the
_measurement_, not the theory, is the contribution.

---

## 1. Background and motivation

Large frozen backbones (e.g. DINOv2 [Oquab et al. 2023]) followed by a linear probe are
the standard way to get strong features cheaply. The GEODE project studies the opposite
end of the design space: a classifier whose feature extractor is **not** a learned
neural network at all, but a fixed _patch dictionary_ — a construction, not a training
run. The question is what such a system can honestly claim: not accuracy dominance
(impossible against a big trunk), but a measured place on the cost/accuracy frontier and
a _quantitative_ understanding of how its two dials — dictionary size and data — trade
off against each other.

This report states what we learned about that trade-off, in a form that does not depend
on the project's internal milestone numbering. All quantities are reproducible from the
registered configuration (Section 4).

---

## 2. System architecture (what we actually built)

The system has three stages: **encode** (images → frozen feature codes), **whiten +
dictionary** (the fixed construction that defines the code space), and **head**
(closed-form ridge classification). A **dense reference** (frozen DINOv2-small trunk +
the same ridge head) provides the cost-matched comparison.

### 2.1 Corpus

- DomainNet [Peng et al. 2019], six domains (clipart, infograph, painting, quickdraw,
  real, sketch), 345 classes, images downscaled to 32×32.
- A shared subsample: 400 train rows and 100 test rows per class per domain, drawn with
  a fixed seed; one deterministic shuffle; the train set is 138,000 rows and the test
  set 34,500 rows, identical for every measurement. The subsample is pinned by a SHA-256
  digest so any re-derivation can be verified byte-for-byte.

### 2.2 Encoder and dictionary (the "sparse" feature extractor)

1. **Patches.** Images are cut into 6×6 patches with stride 1.
2. **Contrast normalisation** with a registered epsilon.
3. **Global whitening.** A ZCA whitener is fit once on 400,000 patches (fixed seed,
   epsilon 0.1) and applied everywhere; the whitener's mean/whiten are pinned so any
   re-derived patch stream can be checked against them.
4. **Dictionary.** The dictionary is a set of "atoms": a seeded, fixed-size pool (8192)
   drawn from the whitened patch distribution and ordered by a seeded permutation. A
   dictionary of size _A_ atoms is simply the first _A_ elements of that fixed order.
   Dictionary construction is **not** trained by gradient descent; it is a deterministic
   construction from the whitened patch stream.
5. **Codes.** Each image is mapped to a code vector of dimension **4·A** (a 2×2 spatial
   pooling folds the per-patch dictionary responses). Codes are computed once per
   dictionary size, stored as float32 memmaps, and treated as _image functions_ — they
   never change when the training-set size changes.

This is the sense in which the system is "sparse": the feature extractor is a sparse
patch dictionary plus pooling, not a dense convolutional stack.

### 2.3 Head: closed-form ridge

- Features are standardised (per-dimension mean and unit variance computed on the
  training set only, RidgeAccumulator convention).
- A ridge (L2-regularised least squares) classifier is fit with penalty λ = 1.0 and an
  intercept. Because the objective is quadratic, the solution is **closed form**:
  weights = (XᵀX + λI)⁻¹XᵀY. There is no iterative optimisation, no epochs, no learning
  rate — the fit is an exact linear solve, hence deterministic and byte-reproducible.
- Prediction is argmax over the 345 class scores (a linear readout in code space).

### 2.4 Dense reference

- A frozen DINOv2-small trunk [Oquab et al. 2023]; the feature is the CLS token
  concatenated with the mean of the patch tokens.
- The same closed-form ridge head (penalty 1.0) on top.
- Resolution is the cost dial: r42 (0.1972 at 215.6M per-image MACs) is the
  cost-matched point at-or-below the sparse family's ~254.6M per-image MACs at 6144
  atoms; the dense curve r70 (0.3118 @ 564.2M), r98 (0.4476 @ 1096M), r224 (0.5375 @
  6124M) defines the frontier.

### 2.5 Variants measured

- **Binary codes.** The same pipeline with a 108-bit Hamming code (a learned ITQ
  rotation [Gong et al. 2011] or a random projection), and a 216-bit version (two
  independent 108-bit projections). The head is a ridge over the binary codes.
- **Per-domain specialists.** One dictionary + ridge per domain (345 classes each),
  trained only on that domain's rows, at 256/512 atoms.
- **Extended dictionaries.** Dictionaries larger than the 8192-atom pool are built by
  re-drawing additional atoms from the same whitened patch distribution (new seeded
  draws, registered), always keeping the first 8192 atoms identical to the pool.

### 2.6 Cost accounting

Per-image multiply-accumulates (MACs) are the cost currency everywhere; wall-clock is
never used for comparison. The sparse family at 6144 atoms costs ~254.6M per-image
MACs; at 16384 atoms ~1.3B; per-domain specialists ~45M (512 atoms, six domains).

---

## 3. Prior art and where this measurement differs

The theoretical scaffolding is _not_ novel and is fully prior art. What is novel here
is the sealed measurement object, which differs from each prior line:

### 3.1 Random-feature / ridge learning curves

- **Bordelon–Canatar–Pehlevan (2020, arXiv:2002.02561)** and **Canatar–Bordelon–Pehlevan
  (2021, arXiv:2006.13198)** derive the ridge/wide-network test error from the feature
  Gram eigenspectrum and the label projections ("modal power"), including a closed-form
  E_g(P) learning curve. **Mei–Montanari (2019, arXiv:1908.05355)** bound random-feature
  regression error; **Xiao et al. (2022, arXiv:2205.14846)**, **Defilippis–Loureiro–
  Misiakiewicz (2024, arXiv:2405.15699)**, and **Atanasov et al. (2024, arXiv:2405.00592)**
  extend spectrum-based scaling predictions; **Bahri et al. (2022, arXiv:2102.06701)**
  explain neural scaling laws.
- **Difference measured here:** the theory predicts the _shape_ of the accuracy-vs-data
  curve from the spectrum. We measured both the shape and the spectrum of a real frozen
  dictionary system and found the MSE-proxy version of the theory **does not track the
  argmax crossing** (it predicts the sparse family ≈ chance and the dense family → 1.0;
  measurement says sparse 0.2153 vs dense 0.1972). The argmax object is not separable by
  a spectral/MSE proxy built from the Gram spectrum and label projections alone.

### 3.2 Overparameterization optimality

- **Simon–Karkada–Ghosh–Belkin (ICLR 2024, arXiv:2311.14646)** show random-feature
  models can be _optimal_ under infinite overparameterization; **Chen–Schaeffer (2021,
  arXiv:2110.11477)** analyse random-feature conditioning and the N/m complexity ratio.
- **Difference measured here:** these are one-dimensional (features OR data) statements.
  We measured the **2D surface** and found the axes interact _positively_ (super-
  additivity): more features make data more productive, and more data makes features
  more productive. That interaction, and its saturation past ~6144 atoms, is the
  measured object.

### 3.3 Dictionary learning

- **Shakeri et al. (2016, arXiv:1608.02792)**, **Schnass (2016, arXiv:1605.05284)** and
  **Schnass (2015, arXiv:1503.07027, ITKM)** bound the _sample complexity of recovering
  a generating dictionary_.
- **Difference measured here:** we do not recover anything. Our dictionary is a fixed
  construction, and we measure the _accuracy surface of the resulting classifier_ — a
  different object from dictionary-recovery guarantees.

### 3.4 Frozen trunks and linear probes

- **Shi et al. (2023, arXiv:2303.00106)** study label efficiency of linear probes on
  pretrained features; **Chowdhury et al. (2023, arXiv:2306.04073)** study patch-level
  MoE routing sample efficiency.
- **Difference measured here:** our "head-width" axis is a _property of the sparse
  encoder_ (dictionary size), and we quantify its data-elasticity (Section 5.3) — a
  mechanism that is structurally absent from a fixed frozen trunk, whose feature width
  cannot be dialled without changing the model.

### 3.5 The measurement protocol

The prior-art lines above are theoretical or empirical results; none of them is a
_registered, gated, sealed measurement_ with exact anchor reproduction. That protocol
(Section 4) is what makes the numbers trustworthy and is itself the methodological
contribution.

### 3.6 Joint (model-size × data) scaling — the scaling-laws field (established)

The central v19 conclusion — model-size scaling saturates, data scaling saturates, and
scaling BOTH jointly keeps improving — is **established prior art**, not a discovery of
this programme. A registered survey (M132, evidence at
`logs/results/v16/m132_joint_scaling_litsearch/evidence.json`; instrument live, 79
anchor hits) confirms the presence of this work directly:

- **Joint scaling and compute-optimal allocation:** Kaplan et al. (2020,
  arXiv:2001.08361) fit power laws in parameters and data; Hoffmann et al.
  (_Chinchilla_, 2022, arXiv:2203.15556) fit loss as a joint function
  $L(N,D) = E + A/N^{\alpha} + B/D^{\beta}$ and show compute-optimal training must
  scale **both** in a fixed ratio. The D5 family of M132 returned the compute-optimal
  line directly (e.g. "Compute-Optimal LLMs Provably Generalize Better With Scale").
- **Data saturation:** Muennighoff et al. (2023, arXiv:2305.16264, "Scaling
  Data-Constrained Language Models") show repeated epochs give sharply diminishing
  returns; Hestness et al. (2017, arXiv:1712.00409) found power-law-in-data with a
  saturation floor; Sharma & Kaplan (2022, arXiv:2012.00160) tie the saturation to the
  data manifold dimension. The D2 family of M132 returned this line directly.
- **Model-size saturation at fixed data:** the parameter power laws (Kaplan) and the
  double-descent analysis of Nakkiran et al. (2019, arXiv:1912.02292) are the same
  phenomenon viewed on the width axis.
- **Theoretical joint dependence:** the random-feature/ridge theory already depends on
  data _and_ the spectrum jointly (Bordelon–Canatar–Pehlevan, Section 3.1); the D4
  family of M132 returned this line (e.g. "Spectrum Dependent Learning Curves in
  Kernel Regression and Wide Neural Networks").

**What differs here is the measurement object, not the concept.** The scaling laws are
measured on _learned_ neural networks (SGD/transformers). This report measures the
same qualitative phenomena in a **non-learned** system — a patch dictionary + closed-
form ridge, no backprop — with the super-additivity and the saturation points
_quantified_ on a fixed corpus under a matched-cost protocol. The concept is prior
art; the sealed measurement of _this_ system is the contribution, and no novelty claim
is made anywhere in this report.

---

## 4. Methodology (protocol, gates, and how to replicate)

### 4.1 Registered-before-measurement protocol

Every measurement follows the same discipline, which is why the numbers can be
believed without re-running:

1. **Registration.** The hypothesis, the success gate (kill-switch), the tolerance, and
   the exact configuration are written down _before_ any accuracy is computed, in a
   committed config file. A measurement whose hypothesis was amended after seeing
   numbers is inadmissible.
2. **Anchors (t1 tolerance).** Every run must reproduce a previously sealed number
   within tolerance 0.002 (e.g. the 6144-atom/full-data accuracy 0.2249 must reproduce
   exactly from the same codes). This catches broken instruments before any new claim.
3. **Kill switches.** Each measurement has a pre-registered fail condition (e.g. "the
   atoms axis must still pay at full data: Q(16384) − Q(6144) ≥ +0.005"). A fired
   switch resolves the question in the negative and the negative is reported.
4. **Matched cost.** Sparse-vs-dense comparisons are at equal per-image MACs.
5. **Fixed test set.** One 34,500-row test set for every cell and both families.
6. **Evidence sealing.** Outputs are written as canonical JSON with a payload hash and
   an artifact index, committed to the repository.

### 4.2 What was measured

| Measurement                | Axes                                                            | Cells                 | Gate                                                            |
| -------------------------- | --------------------------------------------------------------- | --------------------- | --------------------------------------------------------------- |
| Joint surface              | atoms ∈ {1536, 3072, 6144} × n ∈ {34500, 69000, 138000}         | 9                     | super-additivity: joint gain > sum of single-axis gains + 0.005 |
| Head-width data-elasticity | steepness of Q(n) vs atoms                                      | (fits on the surface) | disclosed fit, no threshold                                     |
| Data scaling vs dense      | n-ladder {6900…138000}, atoms 3072 vs DINOv2-r42                | nested ladder         | sparse gain ≥ 0.5 × dense gain                                  |
| Binary codes               | 108/216-bit, random/ITQ, atoms {3072, 6144} × n {69000, 138000} | 8                     | joint budget narrows bit loss ≥ +0.01                           |
| Per-domain specialists     | atoms {256, 512} × n {0.4·n_d, n_d} per domain                  | 24 (4 per domain × 6) | per-domain super-additivity (4/6)                               |
| Atoms extension            | atoms {8192, 12288, 16384} × n = 138000                         | 3                     | Q(16384) − Q(6144) ≥ +0.005                                     |
| Spectral diagnostics       | Gram eigenspectra of sparse vs dense codes                      | exact + truncated     | explanatory, never gates                                        |
| Margin diagnostics         | test-set score margins (f_true − max_other)                     | 6144/8192 atoms       | explanatory, never gates                                        |

### 4.3 Reproduction (how a third party repeats this)

1. **Corpus.** DomainNet six-domain/345-class; take 400 train + 100 test rows per class
   per domain (subsample seed 107, shuffle seed 11); verify the SHA-256 digest
   `63f590097008f749f3f1828b29d6f154de7b21a6828a7b017ac141c0615fa09d`.
2. **Features.** 6×6 patches, stride 1, contrast epsilon 10.0; ZCA whitening fit on
   400,000 patches (seed 11, epsilon 0.1); dictionary = first _A_ atoms of the seeded
   permutation (seed 11) of the 8192-whitened-patch pool; codes = 2×2-pooled dictionary
   responses, width 4·A, float32, computed once per _A_. Extended dictionaries reuse the
   first 8192 atoms and append new seeded draws (seed 42) from the same whitened stream.
3. **Head.** Standardise per-dimension on train; closed-form ridge λ = 1.0 with
   intercept; argmax over 345 scores; report top-1 on the fixed 34,500-row test set,
   plus per-domain accuracy.
4. **Dense reference.** Frozen DINOv2-small; feature = CLS ⊕ mean(patch tokens);
   identical ridge head; resolution 42 for the cost-matched point, 70/98/224 for the
   curve. Per-image MACs computed for both families.
5. **Gates.** Reproduce a sealed anchor within 0.002 before trusting any new cell.

### 4.4 Environment notes (reported, not claims)

Ridge and whitening run on CPU (numpy, float64 Gram); encoding runs on GPU (ROCm/HIP).
The compute GPU also drives the display; a per-batch throttle keeps the display engine
alive (a registered hardware constraint, not a code change). Dictionary sizes above
8192 atoms require a memmap-backed Gram with chunked-output-row accumulation and an
F-order in-place solve; the arithmetic is identical to the in-RAM fit.

---

## 5. Results

### 5.1 The joint surface is super-additive (the "ceiling" was a slice artifact)

At the reference cell (1536 atoms, 34,500 rows), accuracy is 0.1028. Scaling the axes:

- atoms alone (3072 atoms, same n): 0.1004 — flat-to-negative;
- data alone (1536 atoms, 69,000 rows): 0.1457;
- **both** (3072 atoms, 69,000 rows): **0.1530** — greater than the sum of the
  single-axis gains (excess +0.0097/+0.0101 over the +0.005 margin).

The same test at the (3072 → 6144 atoms) × (69,000 → 138,000 rows) cell reproduces the
pattern. At full data, 6144 atoms reach **0.2249** vs 3072 atoms' 0.2153. The quality
axis of this family is therefore **not closed**: it is flat only on single slices, and
compute and data are _complementary_ resources.

### 5.2 Data scaling beats a cost-matched dense trunk

On a nested n-ladder (6,900 → 138,000 rows) at fixed 3072 atoms, the sparse family's
accuracy overtakes the cost-matched dense point (DINOv2-small r42) between n = 27,600
and n = 55,200, and the total data gain ratio vs dense is **1.35** (sparse gains ≥ half
the dense gain at every rung — gate passed). The sparse family is the _steeper_ learner
at matched cost, even though its absolute ceiling is lower.

### 5.3 Data-elastic compute: wider heads consume data more productively

Per-atom learning-curve steepness (absolute Q(n) gain) rises with dictionary size:

| atoms | width | absolute steepness | fitted n-exponent |
| ----- | ----- | ------------------ | ----------------- |
| 1536  | 6144  | 0.094              | 1.136             |
| 3072  | 12288 | 0.115              | 1.128             |
| 6144  | 24576 | 0.134              | 1.117             |

The fitted exponent is ~width^γ with γ ≈ **−0.012**, i.e. essentially width-independent:
the wider head buys a larger _absolute_ data gain at the same learning _rate_. Design
rule: _when data is the limiting resource, put compute into head width._ (Reported as a
trend-level rule, not a law — the exponents are not strictly monotone.)

### 5.4 Binary codes: the bit loss is intrinsic, cost-only

At 3072 atoms, 108-bit Hamming codes score 0.1842 (random projection) / 0.1820 (ITQ)
vs the float reference 0.2153. The joint-budget test (bits × data) showed:

- 216 bits (two independent 108-bit projections) buy +0.009–0.016 within the binary
  family (more at the wider head), but the gap to float **never closes** (kill-switch
  fired: the joint budget narrows the bit loss by only +0.0048 random / −0.0009 ITQ
  vs the +0.01 requirement).
- The binary ceiling is a **code-width phenomenon**, not a head-width one. The binary
  axis is a cost route only (cheap encode), permanently.

### 5.5 Per-domain specialists: the primary buy-back route

Each of the six domains gets its own 345-class dictionary+ridge trained on that
domain's rows only. On a per-domain atoms × data grid (256/512 atoms × 0.4·n_d/n_d):

- **4/6 domains are super-additive**, and all six have positive excess
  (clipart +0.0243, quickdraw +0.0091, real +0.0140, sketch +0.0120; infograph +0.0017,
  painting +0.0049). The per-domain joint surface mirrors the global one.
- At full data, 512-atom specialists match or beat the global 3072-atom arm on 4/6
  domains at **~5.6× fewer per-image MACs** (~45M vs 254.6M), with the best domain
  (quickdraw) at 0.3245.

This is the strongest cost story: cheap, per-domain, and self-scaling.

### 5.6 The atoms axis saturates at full data

Extending the dictionary past the 8192-pool cap (new seeded draws, full 138,000 rows):

| atoms | width | accuracy   | per-image MACs |
| ----- | ----- | ---------- | -------------- |
| 6144  | 24576 | **0.2246** | ~254.6M        |
| 8192  | 32768 | 0.2228     | ~339M          |
| 12288 | 49152 | 0.2235     | ~509M          |
| 16384 | 65536 | 0.2205     | ~1.3B          |

The kill-switch fired: Q(16384) − Q(6144) = **−0.0041** < +0.005. The atoms axis is a
**flat ridge** peaking at ~6144 atoms and slowly declining beyond. The remaining
joint-budget lever is **data**, not atoms (consistent with Sections 5.2–5.3).

### 5.7 Why the gap to dense is structural (diagnostics)

**Spectrum.** The sparse code space is dominated by a fixed ~8-dimensional structure:
effective rank 7.84/7.75/7.83/7.77 at 6144/8192/12288/16384 atoms (vs 7.9 at 3072), a
spectral tail index ≈ −2.2 (flat), and the top-10 modes carrying ~74% of the variance.
The one-hot label power captured by the top modes grows only slowly with atoms
(0.19 → 0.24 from 6144 → 8192 atoms, vs 0.118 at 3072) — the code space's low-rank
geometry is intrinsic to the whitened dictionary, not to the atom count.

**Margins.** On the sealed test set at 6144/8192 atoms, the score margin
(f_true − max_other) has median ≈ −0.04 and its 75th percentile is **still negative**
(−0.005/−0.006); only the 95th percentile (+0.085/+0.095) is positive. The correct
class loses the argmax for **~78% of test samples**; the model's ~22% accuracy lives
entirely in a thin positive-margin tail. Accuracy is structurally capped for this code
geometry, independent of the ridge solver.

**Certificate.** A first-principles per-class Gaussian margin model built from the Gram
spectrum + label projections (Canatar's machinery applied per class, Monte-Carlo
argmax) predicts sparse ≈ chance (0.003–0.005) and dense → 1.0; measurement says the
opposite (0.2153 vs 0.1972). The argmax crossing is not predictable from spectra +
labels alone. We report this as a _negative_ result and stopped investing in
spectral-only certificates.

---

## 6. Discussion

**What the family can honestly claim.** On the sealed corpus, the sparse
patch-dictionary family (a) scales with the joint (dictionary × data) budget
super-additively, (b) is a steeper data-learner than a cost-matched frozen DINOv2 trunk
(gain ratio 1.35), (c) offers a cheap per-domain specialist route (~5.6× fewer MACs),
and (d) has a hard, structurally-explained ceiling on the atoms axis (fixed ~8-D code
geometry + thin positive-margin tail). It never dominates the dense trunk in absolute
accuracy (0.22 vs 0.31–0.54 at matched or higher cost); the claim is cost-matched
non-domination plus mechanism.

**What is closed.** Closed-form depth (does not lift the ceiling); accuracy-preserving
binary codes (intrinsic bit loss); spectral certificates (cannot predict the argmax
crossing); dictionary growth past ~6144 atoms (flat ridge).

**What is open.** The data axis (the steep one); per-domain specialization (cheap +
super-additive); and, structurally, the thin-margin tail — any future work that wants
to close the dense gap must change the _code geometry_ (the ~8-D structure), not add
atoms or solver capacity.

---

## 7. Conclusion

The single-stage accuracy "ceiling" of the sparse patch-dictionary family was a slice
artifact. Measured on the 2D (dictionary-size × data) surface, the family is
**super-additive in its joint budget**, its compute is **data-elastic** (wider heads
consume data more productively), and its honest frontier is cost-matched
non-domination against a frozen dense trunk. The atoms axis saturates at ~6144 atoms
because the code space is a fixed low-rank structure whose argmax object wins only in a
thin margin tail. These are measurement-level results under a registered protocol; the
theory (random-feature learning curves, overparameterization optimality) is prior art,
and no novelty claim is made beyond the sealed measurements themselves.

---

## 8. References

**Random-feature / ridge learning curves and scaling laws**

1. Bordelon, B., Canatar, A., Pehlevan, C. _Spectrum Dependent Learning Curves in Kernel Regression and Wide Neural Networks._ ICML 2020. arXiv:2002.02561.
2. Canatar, A., Bordelon, B., Pehlevan, C. _Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide networks._ Nature Communications 2023. arXiv:2006.13198.
3. Mei, S., Montanari, A. _The generalization error of random features regression: Precise asymptotics and the double descent curve._ Comm. Pure Appl. Math. 2022. arXiv:1908.05355.
4. Xiao, L., et al. _Spectral density of random features and learning curves._ 2022. arXiv:2205.14846.
5. Defilippis, P., Loureiro, B., Misiakiewicz, T. _Dimension-free deterministic equivalents and scaling laws for random feature regression._ 2024. arXiv:2405.15699.
6. Atanasov, A., et al. _Scaling laws for learning with real networks._ 2024. arXiv:2405.00592.
7. Bahri, Y., et al. _Explaining neural scaling laws._ PNAS 2024. arXiv:2102.06701.

**Overparameterization and conditioning** 8. Simon, J. B., Karkada, D., Ghosh, N., Belkin, M. _More is Better in Modern Machine Learning: When Infinite Overparameterization is Optimal and Overfitting is Not._ ICLR 2024. arXiv:2311.14646. 9. Chen, Z., Schaeffer, R. _Conditioning of random feature matrices: A is better than you think._ 2021. arXiv:2110.11477.

**Dictionary learning sample complexity** 10. Shakeri, Z., et al. _Sample complexity bounds for dictionary learning._ 2016. arXiv:1608.02792. 11. Schnass, K. _On the identifiability of overcomplete dictionaries via the l1 principle._ 2016. arXiv:1605.05284. 12. Schnass, K. _A unified convergence proof for dictionary learning via iterative thresholding / ITKM._ 2015. arXiv:1503.07027.

**Linear probes, routing, and feature transfer** 13. Shi, Z., et al. _Label efficiency of linear probes on pretrained features._ 2023. arXiv:2303.00106. 14. Chowdhury, S., et al. _Patch-level routing in mixture-of-experts improves sample efficiency._ 2023. arXiv:2306.04073.

**Backbones, data, and binary codes** 15. Oquab, M., et al. _DINOv2: Learning Robust Visual Features without Supervision._ TMLR 2024. arXiv:2304.07193. 16. Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., Wang, B. _Moment Matching for Multi-Source Domain Adaptation (DomainNet)._ ICCV 2019. 17. Gong, Y., Lazebnik, S., Gordo, A., Perronnin, F. _Iterative Quantization: A Procrustean Approach to Learning Binary Codes for Large-Scale Image Retrieval._ CVPR 2011.

**Joint (model-size × data) scaling and data saturation (established, confirmed by M132)** 20. Kaplan, J., et al. _Scaling Laws for Neural Language Models._ 2020. arXiv:2001.08361. 21. Hoffmann, J., et al. _Training Compute-Optimal Large Language Models (Chinchilla)._ NeurIPS 2022. arXiv:2203.15556. 22. Muennighoff, N., et al. _Scaling Data-Constrained Language Models._ NeurIPS 2023. arXiv:2305.16264. 23. Hestness, J., et al. _Deep Learning Scaling is Predictable, Empirically._ 2017. arXiv:1712.00409. 24. Sharma, U., Kaplan, J. _Scaling Laws from the Data Manifold Dimension._ JMLR 2022. arXiv:2012.00160. 25. Nakkiran, P., et al. _Deep Double Descent: Where Bigger Models and More Data Hurt._ ICLR 2020. arXiv:1912.02292.

**Related surveys in this programme (adjacent directions, no claim)** 18. _HTN-style hierarchical routing — literature review._ Documents the LLM×HTN/routing
landscape; absence of the exact combination is unresolved, not novel. 19. _Programmatic primitives + hybrid router — literature review._ Documents
tool-augmented models, neuro-symbolic hybrids, typed tool schemas, LLM-as-
controller, reject/cascade, and energy-aware routing; technique-per-issue mapping. 26. _Joint (model-size × data) scaling — registered survey (M132)._ Confirms the
scaling-laws lines above are present; absence is not claimed anywhere.
