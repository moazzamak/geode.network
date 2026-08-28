# GEODE v10 Claim Ledger

**Program:** Safety-Calibrated Higher-Codimension Support  
**Status:** Final Outcome D; all branches dispositioned
**Registration date:** 28 July 2026  
**Immutable parent:** v9 Outcome D

## Claim boundary

V10 tests a narrower claim than the v9 surface program:

> Frozen deep-feature classes exhibit useful higher-codimension local support,
> and a dimensionless tangent-bound calibration can retain that predictive
> signal while rejecting unsupported tangent extrapolation.

V10 does not reopen the codimension-one shell branch. It also does not assume
that a fitted affine patch is the true generative manifold.

## Parent evidence

V9 established two immutable observations on seed 11:

1. rank-16 and rank-32 residual/tube scores improved known balanced accuracy by
   1.375 and 1.625 points over the frozen A2 volume head and improved unknown
   recall by more than 30 points;
2. the bounded and unbounded variants behaved identically and accepted 100% of
   synthetic probes at eight times the fitted tangent extent.

The v9 penalty was numerically too small relative to its calibrated acceptance
threshold. V10 may correct score dimensionality and calibrate safety before
development evaluation. It may not tune the penalty from v9 development
failures.

## Registered hypotheses

- **H1 — residual support:** own-class observations have smaller orthogonal
  residual than competing known and proxy-unknown observations under a stable
  low-rank tangent estimate.
- **H2 — bounded support:** dimensionless tangent overshoot rejects
  extrapolation while preserving calibrated known coverage and predictive
  performance.
- **H3 — local atlas:** multiple bounded local patches represent curved or
  multimodal support better than one global affine patch at matched budget.

H1 is diagnostic and cannot advance alone. H2 is required for an eligible
support model. H3 is conditional on H2 or on a clearly diagnosed global-patch
curvature failure.

## Frozen inputs

- DINOv2-small 384-dimensional features and split identities from v6.1-v9;
- CIFAR-10 classes 0-7 as known and classes 8-9 as proxy unknown;
- seeds 11, 23, and 37;
- v9 geometry, calibration, development, and unknown partition hashes;
- frozen A2 signed-volume, rank-32 Gaussian, kNN-support, and RBF controls;
- final labels remain sealed.

## Eligible score

For patch \(k\), tangent coordinates \(z_k\), orthogonal residual \(r_k\),
calibration residual scale \(s_{\perp,k}\), tangent extent \(a_{k,j}\), and
outer tangent scale \(b_{k,j}\):

\[
q_{\perp,k}(x)=
\frac{\|r_k(x)\|^2}
{\operatorname{Quantile}_{0.95}
 (\|r_k(X_{\mathrm{cal}})\|^2)+\epsilon},
\]

\[
q_{\parallel,k}(x)=
\frac{1}{d_k}
\sum_{j=1}^{d_k}
\left[
\max\left(
0,
\frac{|z_{k,j}(x)|-a_{k,j}}{b_{k,j}+\epsilon}
\right)
\right]^2,
\]

\[
s_k(x)=q_{\perp,k}(x)+\lambda q_{\parallel,k}(x).
\]

Every term is dimensionless. Division by rank prevents a larger rank from
receiving a larger tangent penalty solely because it has more coordinates.
Patch fusion uses minimum score. A normalized soft minimum is diagnostic only
unless separately advanced.

## Calibration policy

- Registered ranks: 8, 16, and 32.
- Registered local patch counts per class: 1, 2, and 4.
- Tangent extent quantiles: 0.90, 0.95, and 0.99.
- Outer tangent scales: calibration median overshoot and interquantile range,
  each floored by a frozen numerical epsilon.
- Penalty grid: \([1,4,16,64,256,1024]\).
- Known-coverage target: 92%.

For each geometry, select the **smallest** penalty on calibration data that:

1. gives at least 90% empirical calibration-known coverage;
2. accepts no axis-aligned tangent probes at 8x extent;
3. accepts at most 1% of probes at 4x extent;
4. does not increase rejection from 1x to 0.5x extent; and
5. satisfies the parameter and fit-work budget.

Selection uses calibration observations and label-free synthetic probes only.
If no registered penalty is feasible, that geometry is ineligible. Development
labels cannot select rank, patch count, extent, scale, or penalty.

## Primary endpoints

At the threshold calibrated to 92% known coverage:

- known-class balanced accuracy;
- remaining-unknown recall;
- accepted-known balanced accuracy;
- AUROC and FPR95;
- tangent-probe acceptance at 1x, 2x, 4x, and 8x;
- bridge, mixed tangent-normal, random-direction, and cross-patch acceptance;
- parameter count, fit work, latency, and replay identity.

## Controls

- frozen A2 signed-volume head;
- v9 unbounded orthogonal residual;
- v7 rank-32 Gaussian;
- kNN support;
- same-space RBF;
- random-orientation tangent bases;
- label-permuted classes;
- shuffled tangent extents;
- score-direction reversal.

## Advancement gates

### M56-M57 protocol and identifiability

All partition, dimensional-consistency, synthetic identifiability, negative
control, and exact-replay tests must pass. Synthetic tubes must recover their
registered rank within one grid step and reject at least 99% of 8x tangent
probes while preserving at least 90% in-support coverage.

### M58 seed-11 screen

An affine tube advances only if it:

1. improves known balanced accuracy over A2 by at least 1.0 point;
2. loses no more than 2.0 points of unknown recall;
3. loses no more than 1.0 point of accepted-known balanced accuracy;
4. accepts 0% of 8x and at most 1% of 4x tangent probes;
5. accepts at most 5% of bridge, random-direction, and mixed probes;
6. predicts at least six of eight known classes;
7. stays within 2x the A2 parameter and fit-work budgets; and
8. replays geometry, calibration, threshold, and predictions exactly.

At most one rank/extent/penalty cell advances. Ties use lower parameter count,
then lower rank, then smaller penalty.

### M59 three-seed confirmation

The retained affine tube advances only if it:

1. improves mean known balanced accuracy over A2 by at least 2.0 points;
2. has a paired 95% bootstrap lower bound above zero;
3. improves at least two of three seeds;
4. remains within 3.0 points of the Gaussian control;
5. satisfies the M58 unknown and accepted-known safety bands on every seed;
6. accepts 0% of 8x and at most 1% of 4x tangent probes on every seed;
7. passes at least 8/9 seed-by-safety-diagnostic cells; and
8. preserves exact replay and all budgets.

### M60 local-atlas gate

A local atlas advances only if, at matched total component budget, it:

1. improves mean known balanced accuracy over the retained global tube by at
   least 1.5 points;
2. does not regress unknown recall or accepted-known accuracy by more than 1.0
   point;
3. reduces bridge or cross-patch acceptance by at least 10 points;
4. improves at least two of three seeds; and
5. preserves all M59 safety, replay, and resource gates.

### M61 lifecycle gate

Lifecycle evaluation opens only after M59 or M60 passes. At the frozen v8
50-label budget, the retained support model must:

1. improve mean episode utility by at least 2.0 points over the matched Gaussian
   baseline;
2. preserve known-class balanced accuracy within 1.0 point;
3. preserve remaining-unknown recall within 2.0 points;
4. improve at least 7/9 episode cells;
5. have a paired 95% interval above zero; and
6. preserve transaction, rollback, threshold-transfer, and exhaustive-fallback
   contracts.

## Kill switches

- Stop a rank if no registered penalty is calibration-feasible.
- Stop the global affine family if M58 has no eligible cell.
- Open the atlas after global failure only when diagnostics show residual signal
  plus bridge/curvature failure; do not open it for a null residual result.
- Stop the atlas if patch assignments are unstable under bootstrap or exceed
  the component budget.
- Block lifecycle work unless a predictive/open-set model passes every safety
  gate.
- Any partition leakage, final-label access, or replay mismatch produces
  Outcome F and prohibits a positive claim.

## Outcomes

- **Outcome A:** a safe support model passes predictive, open-set, and lifecycle
  gates.
- **Outcome B:** a safe global affine tube passes M59 but not lifecycle.
- **Outcome C:** a safe local atlas passes M60 but not lifecycle.
- **Outcome D:** residual signal exists, but safety or practical predictive
  gates fail.
- **Outcome E:** no reproducible higher-codimension support signal.
- **Outcome F:** protocol integrity failure.

## Claim restrictions

V10 may not claim:

- discovery of the true data manifold;
- universal manifold support in deep representations;
- a safe tube from synthetic rejection alone;
- superiority to Gaussian, kNN, or RBF without matched endpoint evidence;
- lifecycle utility unless M61 opens and passes;
- validation of shell support;
- independent final confirmation.

## Advancement ledger

| Milestone | Decision | Status |
|---|---|---|
| M56 | Lock parents, score units, calibration, probes, and schemas | Complete; all operands passed |
| M57 | Establish synthetic identifiability and negative-control behavior | Complete; all operands passed |
| M58 | Run seed-11 global affine screen | Complete; 0/18 cells retained |
| M59 | Confirm one affine tube over three seeds | Blocked by M58 |
| M60 | Test a conditional local atlas | Closed; opening condition absent |
| M61 | Test lifecycle utility | Blocked; no retained support model |
| M62 | Final artifact replay | Complete |

M56 selected penalty 1 on its deterministic protocol fixture, preserved 92.5%
calibration-known coverage, rejected all 4x and 8x axis probes, and reproduced
evidence SHA-256
`884b2619a20b16253d57a16b6e5b4a7518846516f110150c999b1231be7a2fba`
twice. This qualifies the protocol implementation, not the real-feature claim.

M57 recovered all nine registered straight-tube ranks exactly, preserved 91.35%
pooled independent in-support coverage, rejected every 8x probe, separated
normal displacement, and passed curved, multimodal, volume, shell,
random-orientation, and random-label controls. Evidence SHA-256
`df7fc12a728c7a83863e009401abefabc7c7531472f4325dcf516ae72a8aea91`
replayed exactly. These controlled results open M58 but do not support H1 or H2
on frozen deep features.

M58 evaluated all 18 frozen cells. Sixteen were calibration-infeasible. Of the
two feasible rank-8/0.99-extent cells, one regressed known balanced accuracy by
5.75 points and the other improved it by 0.875 points, below the 1.0-point
screen gate. Both passed all registered safety operands, but predictive utility
is co-primary. Evidence SHA-256
`c444577624b04ada0490bf27b200dc6273449456ddaf3973cb9e7fcdeb57df72`
replayed exactly. No global tube was retained; H2 does not advance.
