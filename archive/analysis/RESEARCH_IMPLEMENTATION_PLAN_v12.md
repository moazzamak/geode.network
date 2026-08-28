# GEODE Research Implementation Plan v12

## Metric-Faithful Geometric Support: Diagnostics, Eikonal Fields, and Probe-Trained Open Space

**Status:** Final Outcome E; M76 complete
**Date:** 28 July 2026  
**Claim ledger:** `analysis/CLAIM_LEDGER_v12.md`  
**Acceptance frame:** `analysis/ACCEPTANCE_CRITERIA_v12.md`  
**Immutable parents:** v6.1 D, v7 C, v8 D, v9 D, v10 D, v11 E

## 1. Program decision

### 1.1 The unified diagnosis

V9, v10, and v11 each terminated on open-space behavior, and the accumulated
evidence supports one root cause that no prior program named:

> Every failure was a failure of the score field **far from the data**, and no
> program ever trained, constrained, or even defined the field far from the
> data.

All prior geometry was fit in closed form (PCA bases, quantile extents,
conformal quantiles) from own-class data only. Probes were used to *test* the
far region, never to *shape* it. The far field was therefore an accidental
extrapolation of a near-field fit. Three measured consequences follow:

1. **No metric semantics off-data.** The residual score is normalized by a
   calibration quantile: meaningful near the patch, arbitrary far from it. V11
   M65 measured 98.8--100% acceptance of 4x tangent probes and 100%
   acceptance of cross-class bridges *at source level*. This is not component
   masking; it is a field without distance semantics.
2. **Uncontrolled score distributions.** Conformal thresholds spanned
   6.09--34.13 in a unit-scaled score. Nothing ever required a controlled
   shape.
3. **Interpenetrating supports.** Negative-guided extents were infeasible
   because other-class observations sit inside own-class tubes below the 0.90
   floor. Nothing in the pipeline ever pushed class supports apart; DINOv2's
   self-distillation objective is indifferent to class-support compactness.

This diagnosis also predicts that normalizing flows and other likelihood-based
nonlinear manifolds would fail identically, consistent with the established
finding that flows assign high likelihood to OOD inputs. That branch is
therefore deprioritized rather than untested-and-attractive.

### 1.2 Two threats to the diagnosis itself

The diagnosis rests on v11 M65 statistics that may not support it:

- **Small-sample thresholds.** M65 recorded `own_extent_count: 100` per
  class-patch. A conformal threshold at alpha = 0.08 from n = 100 is the 93rd
  order statistic of 100 samples: high variance, upward-biased by the
  finite-sample correction. The observed 6.09--34.13 spread may be
  substantially quantile-estimation noise rather than a property of deep
  features. Probe rates such as "98.8%" are 127/128 pooled outcomes, 16 per
  patch.
- **Input-regime mismatch.** `preprocess_image_dinov2` resizes CIFAR-10 32x32
  images to shortest-edge and center-crops to 224 — roughly 7x upsampling
  into a regime far outside DINOv2's LVD-142M training distribution.
  Interpolation artifacts are shared across classes, which is precisely a
  mechanism that would produce common-mode structure and interpenetrating
  supports. Supporting signal: RBF reaches 96.25% where native-resolution
  DINOv2 ViT-S/14 typically reports 97--98%.

V12 therefore **measures before it builds**. Diagnostics M70 gate the entire
architecture program.

### 1.3 Dataset representativeness

The intended environment is an open-world deployed system. The gating evidence
for v9--v11 is one narrow cell:

| Dimension | Intended | v9--v11 evidence |
|---|---|---|
| Classes | 10^2--10^3 | 8 known |
| Novelty | open, recurring | one fixed pair (CIFAR 8--9) |
| Input | native resolution | 32x32 upsampled ~7x |
| Distribution | long-tailed, drifting | balanced, clean, static |
| Calibration | large | ~100 obs/class |
| Domain shift | continuous | none |

Consequences registered explicitly:

- prior negatives must be **re-scoped**, not discarded: "fails on frozen deep
  features" becomes "fails on ~100-sample, 8-class, 7x-upsampled CIFAR-10 in
  DINOv2 space";
- any v12 positive on the same cell would be equally uninformative, so
  **L5 transfer is promoted to a gating operand** (`ACCEPTANCE_CRITERIA_v12`
  Section 2);
- some negatives may be *optimistic*: masking and interpenetration worsen
  monotonically with class count, so 8 classes understates deployment risk.

### 1.4 What v12 proposes

Conditional on diagnostics, v12 makes the far field a first-class trained
object through three mechanisms over an explicit geometric head:

1. **Eikonal constraint** `||grad f_k|| = 1`, penalized at data, probes, and
   interpolants, making each class field a true distance field. Because the
   primitives are analytic, `grad f` is closed-form: no neural field, no
   double-backward, CPU-feasible.
2. **Probes in the loss**, not only in the gate. The deterministic 8-family
   geodesic probe generator built and replayed in v10 M56 / v11 M63 moves into
   the objective as a hinge rejection term over source-component and
   system-level scores, regenerated as geometry moves.
3. **Distribution and separation shaping**: match in-class field values to a
   target distribution (collapsing the threshold ratio) and enforce a
   support-level margin between classes (attacking interpenetration).

Architecture: encoder `g_theta` -> geometric bottleneck `z` in R^m (m = 64--128)
-> per-class explicit bounded primitives -> `f_k(z)` by smooth-min over that
class's components -> predict `argmin_k f_k`, accept iff `f_{k*} <= tau_k`.

Objective: `L_cls + w1 L_eik + w2 L_probe + w3 L_dist + w4 L_sep`.

## 2. Design principles

1. **Diagnose before building.** No architecture milestone opens until M70
   establishes that the failure is real and general on this data.
2. **Learnability and inspectability are the primary axes**; editability and
   lifecycle utility are descriptive
   (`ACCEPTANCE_CRITERIA_v12.md`).
3. **Escalate representation change in stages.** Frozen head -> learned
   projection -> partial fine-tune -> full fine-tune, each with a cheap kill
   switch, respecting CPU-only determinism.
4. **Train on probes, evaluate on held-out probes and real OOD.** Passing
   probes you trained on proves nothing.
5. **Adopt standard terminology.** Inspectability operands map to existing
   named metrics wherever they exist (M69).
6. **Protocol integrity is unchanged**: preregistration, disjoint partitions,
   sealed final labels, matched controls, byte-identical replay, fail-closed
   lineage.

## 3. Frozen inputs and partitions

Reuse v9--v11 seed-specific split identities where the frozen cell is used:
`geometry_fit`, `score_calibration`, `development_eval`, `unknown_eval`,
`episode_validation`, `final_confirmation` (sealed).

New data enters only through M70/M71 manifests, which must hash-lock source,
license, preprocessing configuration (including resize/crop policy and
interpolation mode), class list, and split derivation. Any dataset introduced
for evaluation must define its own disjoint partitions under the same roles.

## 4. M69 — Prior-art audit refresh (gating prerequisite)

**Execution:** unconditional; blocks all architecture milestones

Under learnability + inspectability framing, the v7 composition shield no
longer applies. M69 must resolve displacement across four threads:

1. **Self-generated outlier synthesis** — VOS, NPOS, DREAM-OOD, CIDER,
   Outlier Exposure, G-OpenMax, OSRCI, mixup/manifold-mixup (positives vs
   negatives distinction), MixOE, ATOM, CutPaste/DRAEM/NSA/SimpleNet.
2. **Eikonal / distance-field learning applied outside 3D reconstruction** —
   DeepSDF, IGR, SIREN, SAL/SALD, DiGS; Lipschitz-certified
   distance-to-boundary work.
3. **Distance-aware uncertainty and feature collapse** — DUQ, SNGP, DDU, DUE,
   Deep SVDD/SAD; jointly trained Gaussian/GMM heads; bounded/finite-support
   class regions.
4. **Interpretable-by-design heads and faithfulness metrics** — ProtoPNet
   family, CBM, SENN, NAM/EBM; ROAR, deletion/insertion,
   comprehensiveness/sufficiency, simulatability, counterfactual distance.

### 4.1 Thread 1 result (28 July 2026)

Completed. Verdicts: virtual outlier synthesis **EXISTS** (VOS ICLR 2022 is
nearest ancestor: refits per-class Gaussians on a feature queue and samples the
low-likelihood tail, energy margin loss; NPOS ICLR 2023 is non-parametric kNN
void sampling). Outlier Exposure **EXISTS** but requires external real OOD
data. Open-space risk formalism **EXISTS** (Scheirer 2013, OpenMax, EVM).

Not displaced, and therefore the registered differentiators for the probe
mechanism are:

- deterministic displacement along **fitted principal/tangent axes at
  calibrated multiples of the fitted extent** (VOS samples the tail
  isotropically by likelihood and never eigendecomposes for sampling);
- a **multi-family probe taxonomy** from one fitted geometry (axis, corner,
  normal, mixed, same-class bridge, cross-class bridge, masking, random);
- a **source-component AND system-level** rejection requirement, where prior
  methods use a single binary OOD score;
- same-class bridge and cross-class masking interpolations used as
  **negatives**, inverting the mixup/manifold-mixup use as positives.

Required citations that do not displace: Scheirer 2013; Bendale & Boult 2016;
Rudd 2018; Hendrycks 2019; Lee 2018; Liu 2020; Ge 2017; Neal 2018; Zhang 2018;
Verma 2019; SimpleNet 2023.

Search limitations recorded: Semantic Scholar and Google Scholar were
rate-limited; 2024--2026 coverage is incomplete; MixOE/ATOM identifiers
unconfirmed.

### 4.2 Thread 2 result (28 July 2026)

Completed. Verdicts:

- **Eikonal/SDF lineage EXISTS** but is confined to 3D shape reconstruction:
  DeepSDF (Park 2019, arXiv:1901.05103), IGR (Gropp 2020, arXiv:2002.10099),
  SIREN (Sitzmann 2020), Neural-Pull (Ma 2021), SAL/SALD (Atzmon 2020/2021),
  DiGS (Ben-Shabat 2022). These must be cited as the mechanism's origin.
- **Eikonal applied to classification / OOD / open-set / anomaly detection
  APPEARS ABSENT.** Approximately 30 queries across arXiv, DBLP,
  Semantic Scholar, and OpenReview returned zero relevant results
  (`abs:eikonal AND abs:"anomaly detection"` = 0;
  `abs:eikonal AND abs:"open set"` = 0 ML-relevant;
  `abs:"signed distance" AND abs:"feature space" AND abs:"classification"`
  = 0; DBLP `eikonal classification neural` = 0). The single
  eikonal-plus-OOD hit is a quasimetric RL paper (arXiv:2512.12046), not
  classification.
- **Score-equals-distance PARTIAL.** Certified-robustness work establishes
  `score / L <= d(x, boundary)` — a **lower bound**, in input space, not
  per-class, and never enforced as equality: Hein & Andriushchenko 2017
  (arXiv:1705.08475), Tsuzuku 2018 (arXiv:1802.04034), Anil GroupSort 2019
  (arXiv:1811.05381), Leino 2021 (arXiv:2102.08452), Serrurier 2021
  (arXiv:2006.06520, closest — 1-Lipschitz binary classifier where |f(x)|
  lower-bounds boundary distance).
- **Gradient-norm penalties PARTIAL.** DUQ (van Amersfoort 2020,
  arXiv:2003.02037) is the most dangerous adjacent work but constrains
  `||J_phi(x)||_F` — the Jacobian of the **feature extractor with respect to
  the input** — not the per-class score gradient. SNGP uses spectral
  normalization, a different mechanism. DDAR (arXiv:2402.12664) relaxes DUQ's
  Lipschitz constraint.
- **Gradient-descent-fitted explicit primitives APPEARS ABSENT.** Existing
  parametric heads fit in closed form after training (Lee 2018 Mahalanobis;
  Ahuja 2019 GMM; Prototypical Networks centroids).

Registered differentiators for the eikonal mechanism:

1. `||grad_z f_k(z)|| = 1` on **per-class score fields in feature space**;
2. converting the certified-robustness **lower bound into an enforced
   equality** via the eikonal PDE;
3. **analytic** primitives whose gradients are closed-form, trained by
   gradient descent rather than closed-form estimation.

The plan must explicitly distinguish (2) from certified robustness and (1)
from DUQ's input-Jacobian penalty in any write-up. Search limitations: heavy
rate limiting on arXiv/Semantic Scholar, OpenReview inaccessible, Google
Scholar unscrapeable, 2024--2026 preprints possibly underindexed; five
follow-up queries are recorded as unexecuted.

### 4.3 Thread 3 result (28 July 2026)

Completed. Distance-aware deterministic uncertainty, feature-collapse
prevention, post-hoc Gaussian scoring, and compact one-class objectives all
**EXIST** and are standard prior art to cite:

- DUQ (van Amersfoort et al., ICML 2020, arXiv:2003.02037) jointly trains
  classwise RBF-like analytic scores and applies an input-gradient penalty.
- SNGP (Liu et al., NeurIPS 2020, arXiv:2006.10108) combines spectral
  normalization / bi-Lipschitz control with a random-feature Gaussian-process
  head.
- DUE (van Amersfoort et al., 2021, arXiv:2102.11409) explicitly identifies
  feature collapse and constrains the encoder to preserve distance.
- DDU (Mukhoti et al., 2021, arXiv:2102.11582) shows that a regularized feature
  space plus post-hoc classwise Gaussian discriminant analysis explains much
  distance-aware uncertainty performance.
- Lee et al. (2018, arXiv:1807.03888) establish post-hoc class-conditional
  Mahalanobis scoring.
- Deep SVDD (Ruff et al., ICML 2018), Deep SAD (Ruff et al., ICLR 2020,
  arXiv:1906.02694), and IGD (Chen et al., AAAI 2022,
  doi:10.1609/aaai.v36i1.19915) establish collapse-aware compact one-class
  learning and, for IGD, interpolant-based Gaussian-descriptor shaping.

These works partially overlap but do not displace the registered conjunction.
No audited work jointly trains multiclass explicit analytic primitive fields
with all of: per-class Eikonal equality in feature space, deterministic
geometry-derived probe/interpolant losses, source-component and system-level
rejection, and bounded support. In particular, Lee/DDU are post-hoc, DUQ is
prototype/RBF based with unbounded tails, SNGP/DUE are GP based, and IGD is
one-class.

V12 therefore may not claim first distance-aware uncertainty, first
collapse-aware representation, first Gaussian/Mahalanobis OOD head, first
gradient regularization, or first interpolant shaping. Collapse-prevention
ablation remains mandatory but is not itself novel. Targeted 2024--2026 checks
found adjacent preprints on conformal virtual outliers (GCOS,
arXiv:2603.08413), post-hoc box support (BBAS, arXiv:2603.22660), and
Mahalanobis variance (MahaVar, arXiv:2605.14413), none of which displaced the
conjunction.

### 4.4 Thread 4 result (28 July 2026)

Completed. Intrinsically interpretable prototype, concept, and additive models,
and standard faithfulness metrics, **EXIST**:

- ProtoPNet (Chen et al., NeurIPS 2019, arXiv:1806.10574), ProtoTree (Nauta
  et al., CVPR 2021, arXiv:2012.02046), and successors provide exact
  prototype/path-level score structure, but not analytic per-direction
  decomposition or closed-form minimum feature-space reach.
- Concept bottleneck models (Koh et al., ICML 2020, arXiv:2007.04612) provide
  intervenable semantic concepts. Frozen latent coordinates must not be called
  concepts without genuine concept supervision.
- SENN (Alvarez-Melis and Jaakkola, NeurIPS 2018, arXiv:1806.07538), NAM
  (Agarwal et al., NeurIPS 2021, arXiv:2004.13912), and EBM/GA2M (Lou et al.,
  KDD 2013) establish exact additive/decomposable scoring. Exact decomposition
  alone is not novel.
- Standard protocols are deletion/insertion (Petsiuk et al., 2018,
  arXiv:1806.07421), ROAR with retraining (Hooker et al., NeurIPS 2019,
  arXiv:1806.10758), comprehensiveness/sufficiency (DeYoung et al., ACL 2020,
  arXiv:1911.03429), human-grounded simulatability (Doshi-Velez and Kim 2017;
  Hase and Bansal, ACL 2020, arXiv:2005.01831), and minimum counterfactual
  distance/proximity with validity (Wachter et al. 2018; CARLA, Pawelczyk et
  al., NeurIPS 2021, arXiv:2108.00783).

The inspectability operands are renamed:

1. **I1 intrinsic parameter semantics**: structural audit; no canonical scalar.
2. **I2 exact score decomposition (completeness/local accuracy)**: score equals
   bias plus analytic contributions to numerical tolerance; report mean and
   maximum absolute residual.
3. **I3 deletion/comprehensiveness faithfulness**: top-k directional ablation
   versus random-k and bottom-k over multiple k values; call it ROAR only when
   the model is retrained on ablated data.
4. **I4 minimum counterfactual distance/proximity with validity**: compute the
   minimum feature-space flip, re-evaluate the displaced point, and report flip
   success and distance distribution. Plausibility is not claimed without a
   decoder or manifold constraint.
5. **I5 simulatability proxy / forward-simulation probe accuracy**: because a
   learned probe is not canonical human simulatability, compare with chance and
   no-explanation baselines and prevent example leakage.

No reviewed work displaced the combination of an explicit per-class analytic
geometric head, exact directional decomposition, and closed-form feature-space
counterfactual reach. I4 is the strongest remaining structural differentiator.
I1 and I2 alone are established prior art.

### 4.5 M69 gate disposition

No thread returned outright displacement, so no registered mechanism is
removed. M70 and M71 open. Architecture milestones M72+ remain blocked on M70.
The defended novelty is the conjunction, not any individual Gaussian,
distance-aware, additive, collapse-prevention, or probe-training mechanism.

Searches used public papers, official repositories, and public documentation.
ArXiv, Semantic Scholar, Google Scholar, and OpenReview rate or access limits
made 2024--2026 coverage targeted rather than exhaustive. MixOE/ATOM and KAR
identifiers remain unverified; no claim relies on them.

**Gate:** if any thread returns outright displacement of a mechanism, that
mechanism is removed from the architecture and the plan is amended before
execution.

## 5. M70 — Diagnostic re-examination (gating)

**Execution:** unconditional after M69 thread 1; blocks M72+

No new model. Measurement only, on existing and newly extracted features.

### 5.1 D1 — sample-size sensitivity

Using frozen v11 artifacts and features, recompute per-class conformal
threshold ratios at n = 100, 200, 400, 800 (pooling across seeds 11/23/37),
with bootstrap intervals. Report the threshold ratio distribution and its
dependence on n. Also recompute probe acceptance with per-patch counts raised
by at least 8x to reduce binomial noise.

### 5.2 D2 — resolution and domain sensitivity

Extract DINOv2 features for at least one **native-resolution** dataset
(ImageNet subset and/or DomainNet, for which `prepare_domainnet.py` and
manifests already exist) with no upsampling. On each corpus measure the same
three diagnostics under the frozen v11 rule:

1. per-class conformal threshold ratio;
2. 4x and 8x tangent-probe acceptance;
3. other-class penetration below the 0.90 own-class extent floor.

### 5.3 D3 — class-count scaling

On the native-resolution corpus, measure diagnostics 2 and 3 at 8, 32, and 128
known classes to test whether masking and interpenetration worsen with class
count as predicted.

### 5.4 Registered decision rule

| Outcome | Consequence |
|---|---|
| Pathology persists at native resolution and larger n | v9--v11 negatives are real and general; M71+ proceeds with strong justification |
| Pathology largely disappears | Negatives were dataset artifacts; **v9/v10/v11 ledgers must be amended** before any publication, and the architecture program is re-scoped or closed |
| Mixed | Only the surviving mechanism (tails or interpenetration) is targeted; the other loss term is dropped |
| Diagnostics degrade with class count | Deployment risk is understated in all prior work; L5 gating is strengthened |

An amendment obligation is registered: if D1 shows the threshold spread is
substantially estimation noise, `V11_FINAL_CLAIM_LEDGER.md` receives a recorded
caveat. Prior conclusions are not silently retained.

### 5.5 Result (28 July 2026)

M70 passed its measurement protocol and established that the open-space
pathology is neither a small-sample artifact nor a CIFAR upsampling artifact.

**D1.** Across the three frozen CIFAR seeds, median own-class threshold ratios
were 5.942, 5.595, 5.770, and 5.845 at n=100, 200, 400, and 800. The n=800
point is explicitly an empirical-bootstrap extrapolation because the disjoint
frozen pool contains 600 unique calibration scores per class. The stable
5.6--5.9 range rejects the hypothesis that the v11 spread primarily arose from
quantile-estimation noise. Probe counts were increased at least eightfold.
Source 4x acceptance was 0% on every seed, but system 4x acceptance remained
74.414--81.763%; system 8x acceptance was 66.895--75.049%. This directly
separates source-field behavior from cross-class masking.

**D2/D3.** A pinned DomainNet snapshot supplied 12,800 native images: 100 per
class for nested 8-, 32-, and 128-class evaluations, each split disjointly into
60 geometry, 20 extent, and 20 conformal observations. Every selected image had
short edge at least 256 before the frozen 256-resize/224-crop preprocessing, so
no image was upsampled. System 4x acceptance was 100% at every class count.
System 8x acceptance increased from 97.778% to 99.634% to 100%. Mean
other-class penetration below the 0.90 own-class extent floor increased
monotonically from 0.179% to 0.343% to 0.400%. Native median threshold ratios
were 972.94, 536.79, and 491.57; their magnitude is reported descriptively and
does not alter the probe-based conclusion.

The registered decision is therefore "pathology persists at native resolution
and larger n," with additional evidence that deployment risk worsens with class
count. No v9--v11 amendment is required for a dataset artifact or
small-sample-noise explanation. Their narrow CIFAR scope remains unchanged.
M72 opens with strong justification.

## 6. M71 — Gaussian head as a first-class classifier (cheap, unconditional)

**Execution:** after M69 thread 1; independent of M70 outcome

The low-rank Gaussian has only ever been evaluated as a *rejector*. The v11
control incidentally measured 95.13% known balanced accuracy versus RBF
96.25% on seed 11 — within the parity bar the project assumed unreachable.
Evaluate it as a registered classifier over seeds 11, 23, 37 against RBF,
logistic, and kNN, reporting L1--L4 and I1--I4 operands, on both the frozen
cell and (if available) the M70 native-resolution corpus.

This either establishes an inspectable near-parity head or removes the
optimism immediately. Either way it anchors the bar any trained system must
beat, and it is nearly free.

### 6.1 Result (28 July 2026)

M71 removed the near-parity optimism under the registered three-seed gate while
establishing a strong explicit baseline. Mean known balanced accuracy was
95.708% for the rank-32 Gaussian, 96.917% for RBF, 95.375% for logistic, and
95.208% for kNN. The Gaussian trailed the strongest control by 1.208 points,
outside the 1.0-point tolerance; the pooled paired 95% interval was
[-1.844, -0.564] points and the seed-level interval was [-1.567, -0.850].

Open-set competence was a formal but scientifically indeterminate near miss:
mean unknown recall was 86.833% against the registered 87.0% bar, and the
pooled exact 95% interval [83.862%, 89.435%] includes the bar. It must not be
described as evidence of practically worse open-set behavior. The Gaussian was
compact at 0.813 MB, far below the 6.02 MB kNN reference. Sample efficiency
remained weaker than RBF: 90.333% vs 94.542% at 50/class, 95.125% vs 96.208%
at 200/class, and 95.750% vs 96.917% at 1,000/class.

I1 intrinsic parameter semantics and I2 exact score decomposition passed; the
maximum decomposition residual was within numerical tolerance. Top-8
directional deletion flipped 1.56--2.34% of predictions versus 0.78--1.95% for
random-8 and 0--0.78% for bottom-8, a small descriptive faithfulness signal.
I4 failed structurally: pairwise boundaries between class-specific low-rank
Gaussian quadratic forms are general quadrics, so no registered closed-form
minimum Euclidean counterfactual displacement exists.

M71 is not retained as a fully qualified learnability-plus-inspectability
result because L1 and I4 fail. It remains the frozen baseline for M72's
registered non-regression operand.

## 7. M72 — Stage 0: probe-trained, eikonal-constrained head on frozen features

**Execution:** conditional on M69 threads 3--4 and M70

Train **only the geometric head parameters** by gradient descent on frozen
features. No representation change. This is untested territory: v9/v10/v11
used closed-form fits, and M28 distilled toward a teacher boundary, not a
field.

Implement and register:

- analytic `f_k` and closed-form `grad f_k` for the retained primitive family;
- `L_eik` sampled at data, probes, and interpolants;
- `L_probe` hinge over source-component and system scores;
- `L_dist` in-class distribution matching; `L_sep` support margin;
- deterministic optimizer state, seeds, and exact replay.

**Kill switch (registered, evaluated on calibration data and held-out probes
only):** Stage 0 advances only if
(a) the median per-class threshold ratio falls from the v11 6--34 band to at
most 2.0, and
(b) held-out-family 4x probe acceptance is below 1%, and
(c) known balanced accuracy does not regress more than 1.0 point versus the
M71 Gaussian.

If Stage 0 passes, a safe inspectable envelope exists **with no fine-tuning,
no collapse risk, and full replay determinism**. If it fails, the
representation is implicated and M73 opens.

### 7.1 Result (28 July 2026)

M72 failed two of three registered kill-switch operands on seed 11. The trained
field's median per-class threshold ratio was 1.168, below the 2.0 gate, but the
untrained initialization was already 1.168; this operand does not demonstrate a
training effect. The worst held-out 4x system acceptance was 100% because mixed
probes remained fully accepted, versus the below-1% gate. Known balanced
accuracy was 93.750%, 1.375 points below the frozen M71 Gaussian and outside
the 1.0-point non-regression tolerance.

The failure contains registered residual signal. Training improved known
accuracy from 93.250% to 93.750%, reduced trained-family axis-tangent system
acceptance from 98.926% to 56.116%, and reduced the entirely held-out corner
family from 100% to 0%. Unknown recall remained 90.5%. However, mixed, masking,
normal, and cross-class-bridge system acceptance remained 100%, and mean
absolute Eikonal error on calibration points worsened from 0.273 to 0.545. The
probe hinge itself was zero from the first epoch because its absolute margin
was below the later conformal acceptance scale; no post-hoc margin retuning is
permitted.

The analytic implementation verified exact score decomposition, closed-form
score gradients against finite differences, data/probe/interpolant Eikonal
evaluation, immutable initialization, deterministic optimizer state, and
byte-identical replay. Corner and mixed probes were excluded entirely from
training. Stage 0 does not establish a safe field, but the held-out corner and
axis reductions satisfy the registered "failure with residual signal" branch.
M73 Stage 1 opens; M74 remains blocked.

## 8. M73 — Staged representation escalation

**Execution:** conditional on M72 failure with residual signal

| Stage | Trains | Registered kill switch |
|---|---|---|
| 1 | + single learned projection (one auditable matrix) | M72 operands must improve materially |
| 2 | + last-k blocks or LoRA, reduced resolution | L1 parity within 1.0 point |
| 3 | full fine-tune | full L1--L5 and I1--I5 operand set |

Mandatory for every stage that trains the encoder:

- **collapse prevention** (spectral normalization, two-sided gradient penalty,
  or registered equivalent) **plus an ablation demonstrating it is
  load-bearing**; absence of the ablation invalidates the stage;
- feature-space diagnostics reported before and after training (threshold
  ratio, interpenetration, probe acceptance);
- inspectability claims restated in weakened form: the **decision rule** is
  exactly inspectable; the **feature semantics** are not.

**Determinism policy.** Measured environment is 8-core Ryzen 7 7800X3D, 63 GB
RAM, RX 9070 XT, torch 2.13.0+cpu with no GPU backend. Full fine-tuning is
feasible per-run on CPU but not across a multi-seed grid. Any GPU backend
introduced for gated evidence must first have a written determinism policy;
until then, gated evidence is CPU-only and byte-identical replay is retained.

### 8.1 Stage 1 registration and result (28 July 2026)

Before execution, Stage 1 froze one centered-PCA-initialized 64x384 projection,
rank-32 analytic fields, 24 joint-training epochs, a relative probe target of
twice the detached batch median own-class score, and two matched arms. The
primary arm used a weight-10 sum of row-orthonormality and initial projected
pair-distance preservation penalties; the ablation set only that weight to
zero. Material improvement required a threshold ratio no greater than 2.0 and
either at least +0.5 point known accuracy or at least a 10-point reduction in
worst held-out 4x acceptance relative to M72. Collapse prevention had to reduce
both distance drift and row-orthogonality error by at least 20% relative to the
ablation while retaining all 64 singular directions.

The constrained arm passed this registered escalation gate. Known balanced
accuracy was 95.625%, +1.875 points over M72, and the median threshold ratio was
1.282. Worst held-out 4x acceptance fell from 100% to 75%: corner acceptance
was 0%, but mixed acceptance remained 75%. The projection retained rank 64
with singular values 0.289--1.464. Mean relative pair-distance drift was 0.123
versus 0.567 without the constraint, and row-orthogonality error was 0.00695
versus 0.01295; collapse prevention was therefore load-bearing under both
registered diagnostics.

The result is not a complete open-space safety result. Masking and normal 4x
acceptance remained 100%, cross-class-bridge acceptance was 87.5%, unknown
recall was 59.5%, and calibration mean absolute Eikonal error was 0.629. The
zero-constraint ablation was operationally stronger despite its geometric
drift: 96.375% known accuracy, 77.5% unknown recall, and 0% acceptance for
mixed, masking, and normal probes. Thus the constraint preserves the registered
projection geometry but imposes a measured predictive and rejection cost; it
is not claimed to improve those outcomes. Two independent CPU runs produced
byte-identical 1,790,077-byte evidence with SHA-256
`1f35ef34378314dcd5bb1556c89c65c534a98bc49d397f0b04ef85e7746e21d6`.
M74 opens for confirmation and transfer of the constrained Stage 1 state; no
final labels were opened.

## 9. M74 — Confirmation and transfer

**Execution:** conditional on a retained M72 or M73 stage

1. three-seed confirmation on the primary corpus with paired bootstrap
   intervals;
2. **L5 transfer**: the result must reproduce on a second corpus differing in
   resolution and class count;
3. **held-out probe families**: at least two of the eight families are
   excluded from training entirely and evaluated only here;
4. **real OOD**: CIFAR-100 / SVHN / DomainNet-shift as the operative test, not
   synthetic probes;
5. controls: v7 low-rank Gaussian (87.0% unknown recall at 92% coverage) is
   the bar to beat; plus RBF, kNN, logistic, and an ablation removing each
   loss term.

If the model rejects only the families it trained on, that is Outcome E and
must be declared as such.

### 9.1 Registration and result (28 July 2026)

Before execution, M74 froze the exact M73 64D/rank-32 constrained mechanism
over seeds 11, 23, and 37, matched Gaussian/RBF/logistic/kNN controls, 2,000
paired bootstrap resamples, corner and mixed probes held out from training,
native DomainNet as cross-corpus real OOD, a 32-known/96-unknown native
DomainNet transfer cell, and seed-11 removal of each of the five Stage 1 loss
terms. Primary gates required mean accuracy within 1 point of the strongest
control, at least 87% unknown recall, median threshold ratio no greater than
2.0, less than 1% worst held-out 4x acceptance, and at least 87% real-OOD
recall. Transfer required the same accuracy, unknown-recall, and threshold-ratio
operands.

Only primary L1 parity and the threshold-ratio operands passed. Mean known
balanced accuracy was 96.083% versus 96.917% for RBF, a -0.833-point
difference just inside the one-point tolerance. The pooled paired 95% interval
was [-1.380, -0.287] points and the seed-level interval was
[-1.308, -0.359], so the field was consistently but narrowly worse. Accuracy
by seed was 95.625%, 97.125%, and 95.500%; median threshold ratios were 1.282,
1.314, and 1.297.

Open-set confirmation failed. Proxy-unknown recall was 59.5%, 69.5%, and
70.0%, averaging 66.333% versus the 87% bar. Held-out corner acceptance was 0%
for all seeds, but held-out mixed acceptance was 75%, 100%, and 75%; masking
and normal acceptance remained 100%. Native DomainNet cross-corpus OOD recall
was 76.289%, 54.570%, and 53.516%, also below the bar.

The DomainNet transfer cell failed decisively. The field reached 66.719% known
balanced accuracy, 7.344 points below logistic's 74.063%; the paired 95%
interval was [-10.796, -3.761] points. Unknown recall across 96 unseen classes
was 0.833%, AUROC was 0.622, and known coverage was 99.688%. The median
threshold ratio passed at 1.754, but several per-class ratios exceeded 2.0.
Transfer mixed, masking, normal, and cross-class-bridge acceptance were 100%;
even held-out corner acceptance was 6.25%.

The loss ablations did not identify a safety-critical term. Removing the probe
loss left seed-11 accuracy, unknown recall, threshold ratio, and every reported
probe acceptance unchanged to displayed precision despite changing the state.
Removing Eikonal or separation was similarly negligible. Distribution removal
raised accuracy to 95.875% but reduced unknown recall to 41.5%; classification
removal yielded 95.250% accuracy and 64.0% unknown recall. These results do not
support a claim that the registered probe or Eikonal objectives caused
generalized open-space rejection.

Two independent CPU runs produced byte-identical 4,637,364-byte evidence with
SHA-256
`dca6d3e300d450b0d571abdf68abbc8da8f327f68639787cfc26e926a1d70f41`.
No final labels were opened. Because rejection did not generalize to the
held-out mixed family, real OOD, or second-corpus transfer, M74 takes the
registered Outcome E branch. M75 may characterize inspectability descriptively
but cannot rescue the failed learnability/open-set claim.

## 10. M75 — Inspectability qualification

**Execution:** parallel with M74

Measure I1--I5 using the **standard metric names identified in M69 thread 4**
(ROAR / deletion-insertion / comprehensiveness-sufficiency / simulatability /
counterfactual distance), on the retained model and on RBF, kNN, and MLP
controls. I2 (exact decomposition) and I4 (closed-form counterfactual reach)
are the operands where an explicit geometric head is expected to dominate; I4
is strengthened under the eikonal constraint because the field value *is* the
displacement distance.

### 10.1 Registration and result (28 July 2026)

M75 was frozen as descriptive only against the immutable M74 Outcome E state.
It used 800 held-out seed-11 known-development examples. I3 removed the top,
random, or bottom predicted-component tangent directions at k=1,4,8 without
retraining and is therefore deletion/comprehensiveness, not ROAR. I5 trained a
logistic forward-simulation proxy on sorted explanation values with component
identity withheld, a class-stratified disjoint train/test split, and chance and
no-explanation majority controls. RBF, kNN, and one-hidden-layer MLP controls
received analogous identity-withheld proxy inputs.

I1 passed under the required weakened scope: every head parameter maps to a
named geometric quantity, but the learned coordinates themselves have no
claimed semantics. I2 passed with mean and maximum squared-score reconstruction
residuals of `1.06e-14` and `1.14e-13`.

I3 established score attribution but not useful decision faithfulness. At
k=1,4,8, top-direction deletion reduced the predicted-component score by
0.170, 0.449, and 0.660, versus random reductions of 0.026, 0.103, and 0.210
and bottom reductions of 0.00006, 0.00124, and 0.00782. Yet top-deletion
prediction-flip rates were only 0.125%, 0.125%, and 0%. Because deleting a
distance contribution makes the predicted component *closer*, these results
show completeness of the distance explanation, not comprehensiveness of
positive class evidence.

I4 failed structurally. The deployed decision is the minimum over
class-specific anisotropic quadratic scores plus conformal rejection. The
minimum Euclidean displacement to a pairwise or union boundary is not the field
value and has no registered closed form. M74's non-unit gradient norms also
invalidate the plan's expectation that score values can be read as Euclidean
counterfactual distance. No validity experiment was therefore claimed.

I5 was positive only by its weak binary operand. The field proxy reached
17.737% balanced accuracy versus 12.5% chance/no-explanation, but was below kNN
(25.246%), RBF (22.772%), and MLP (17.513% was approximately tied). This is a
small automated forward-simulation signal, not human simulatability.

The full I1--I5 conjunction is not qualified. Two independent runs produced
byte-identical 5,686-byte evidence with SHA-256
`52fbe3135b1d234681b7ce74727f2f81cbd12ec7321af41f62c472e5fdb53f03`.
Outcome E remains unchanged and M76 finalization opens.

## 11. M76 — Finalization

`analysis/V12_FINAL_CLAIM_LEDGER.md`, immutable indexes, branch dispositions,
one artifact-only verifier, two byte-identical conclusion replays, and
amendments to the v11 ledger if M70 requires them.

### 11.1 Result (28 July 2026)

M76 finalized v12 as Outcome E. The artifact-only verifier checked 7 immutable
indexes, 10 indexed artifacts, 16 conclusion operands, and 8 branch
dispositions while loading no training features and opening no final labels.
It confirmed that no v11 amendment is required: M70 strengthened the existing
narrow frozen-feature pathology rather than showing that prior results were
sampling or preprocessing artifacts.

Two independent conclusion replays produced byte-identical 3,954-byte evidence
with SHA-256
`43f9a14c8060714d85d53cc4ab75953048e5f961dea3f27a5805b2a3c7cc5a6c`.
The final ledger is `analysis/V12_FINAL_CLAIM_LEDGER.md`.

## 12. Dependency graph

```text
M69 prior-art refresh (4 threads)
   | thread 1-2 done; 3-4 open
   v
M70 diagnostics (D1 sample size, D2 resolution/domain, D3 class count)
   |                          \
   | pathology real            \ pathology is artifact
   v                            v
M71 Gaussian-as-classifier   amend v9-v11 ledgers; re-scope or close
   |
   v
M72 Stage 0: eikonal + probe training on frozen features
   | pass                      | fail with residual signal
   v                           v
M74 confirmation + transfer   M73 staged representation escalation
   |                                   |
   +-----------------+-----------------+
                     v
             M75 inspectability
                     v
             M76 finalization
```

## 13. Kill switches

- Any M69 thread returning outright displacement removes that mechanism.
- M70 showing the pathology is a dataset artifact closes the architecture
  program and triggers ledger amendments.
- M72 failing its threshold-ratio or held-out-probe operands blocks
  advancement; escalation requires demonstrated residual signal.
- Any encoder-training stage lacking a load-bearing collapse ablation is
  invalid.
- Rejection confined to trained probe families is Outcome E.
- Single-corpus results cannot satisfy L5; no learnability claim from one
  dataset.
- Partition leakage, final-label access, or replay mismatch is Outcome F.

## 14. Outcomes

- **A:** metric-faithful geometric support passes learnability,
  inspectability, held-out-probe, real-OOD, and transfer gates.
- **B:** passes on the primary corpus but fails transfer.
- **C:** Stage 0 alone suffices on frozen features (strongest practical
  result: no fine-tuning, full determinism).
- **D:** signal exists but a primary gate fails.
- **E:** the mechanism does not transfer, or rejection is confined to trained
  probe families.
- **F:** protocol integrity failure.
- **G (new):** diagnostics show prior negatives were dataset artifacts; the
  program's contribution becomes the corrected measurement and the amended
  record.

## 15. Interpretation

| Result | Interpretation |
|---|---|
| D1 collapses the threshold spread | v11's heavy-tail diagnosis was substantially small-sample noise; amend |
| D2 removes the pathology at native resolution | v9--v11 negatives were input-regime artifacts; the geometry may already work |
| D3 worsens with class count | All prior open-space results understate deployment risk |
| M71 near-parity | The "5--6 point gap" is an SDF-head property, not a property of explicit geometry |
| M72 passes | Metric-faithful fields fix open space **without touching the representation** |
| M73 needed | The representation, not the geometry, was the binding constraint |
| Held-out families fail | Probe training taught rejection of memorized directions, not open space |

The program tests whether an explicit, metric-faithful geometric support model
can learn well and be inspected exactly. It does not claim the true data
manifold, and it does not assume the prior negatives were correctly scoped.
