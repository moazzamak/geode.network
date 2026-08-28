# GEODE v9 Claim Ledger

**Program:** Surface Support Versus Volumetric Containment  
**Status:** final Outcome D
**Registration date:** 27 July 2026  
**Immutable parents:** v6.1 Outcome D, v7 Outcome C, and v8 Outcome D

## Immutable parent outcomes

V9 does not revise or reopen the prior outcomes:

1. The v6.1 weighted affine head reached 91.73% balanced accuracy and remained
   5.03 points behind the same-space RBF control.
2. The v7 stage-qualified rejection, discovery, and adaptation mechanisms did
   not compose into a review-efficient lifecycle.
3. The v8 utility-selected review policy improved over core selection by 3.593
   points but missed the registered 5-point gate and violated remaining-unknown
   safety in 6/9 cells.

V9 tests a new representation-geometry hypothesis. It is not a parity rescue,
an adaptation-utility continuation, or a retrospective reinterpretation of
prior results.

## New hypothesis

The prior geometric heads primarily treated class support as a volume:
negative SDF values indicated containment and increasing negative depth
generally strengthened membership. V9 tests:

> In frozen deep-feature space, class support is better represented by
> proximity to a thin geometric surface or a bounded low-dimensional manifold
> tube than by membership in the enclosed volume.

This statement contains two distinct hypotheses:

- **H1 — shell support:** support concentrates near the zero level set of an
  existing primitive, so two-sided distance to its boundary is more informative
  than signed containment depth.
- **H2 — bounded manifold-tube support:** support concentrates near a
  lower-dimensional affine manifold with bounded tangent extent, so
  perpendicular residual plus tangent support is more informative than either
  a hypersurface shell or an unbounded subspace.

H1 and H2 are evaluated separately. Failure of a codimension-one shell does not
falsify a higher-codimension manifold tube.

## Frozen representation and data policy

- Primary representation: the immutable 384-dimensional DINOv2-small features
  used by v6.1 through v8.
- Development seeds: 11, 23, and 37.
- Primary class protocol: the frozen v6.1 CIFAR-10 class inventory and split
  identities.
- OOD protocol: the frozen v7 leave-two-class-out development episodes.
- Adaptation protocol: the frozen v8 nine development cells, only if the
  predictive and rejection gates first pass.
- Final or untouched confirmation labels remain sealed.
- No method may select rank, width, score weights, thresholds, or stopping
  epochs from evaluation labels.

## Score families

For a primitive field \(f_k(x)\), lower scores indicate stronger support.

### V0: signed-volume baseline

\[
s_k^{\mathrm{vol}}(x)=f_k(x).
\]

This is the frozen containment semantics. Deep negative values are favored.

### S1: normalized shell

\[
s_k^{\mathrm{shell}}(x)=|f_k(x)|.
\]

Interior and exterior deviations of equal normalized magnitude receive equal
penalty.

### S2: metric-corrected shell

\[
s_k^{\mathrm{metric-shell}}(x)=
\left|\frac{f_k(x)}{\|\nabla f_k(x)\|+\eta}\right|.
\]

The correction is a local first-order approximation for anisotropic
ellipsoids, not an exact Euclidean distance. The same \(\eta\) is frozen from
geometry data for all methods.

### S3: asymmetric shell

\[
s_k^{\mathrm{asym}}(x)=
\begin{cases}
|d_k(x)|/\tau_{k,\mathrm{in}}, & d_k(x)<0,\\
|d_k(x)|/\tau_{k,\mathrm{out}}, & d_k(x)\ge 0,
\end{cases}
\]

where \(d_k=f_k/\left(\|\nabla f_k\|+\eta\right)\). The two widths are estimated
from geometry/calibration partitions only. This tests whether support is a
shell without assuming equal noise on both sides.

### M1: unbounded affine residual diagnostic

\[
s_k^{\perp}(x)=
\left\|(I-U_kU_k^\top)(x-\mu_k)\right\|^2/\sigma_{\perp,k}^2.
\]

This is a diagnostic, not an eligible winner, because an infinite affine
subspace can accept unsupported points arbitrarily far along tangent
directions.

### M2: bounded manifold tube

\[
s_k^{\mathrm{tube}}(x)=
\frac{\|(I-U_kU_k^\top)(x-\mu_k)\|^2}{\sigma_{\perp,k}^2}
+
\lambda
\sum_j \rho\left(
\frac{|u_{k,j}^\top(x-\mu_k)|-a_{k,j}}{b_{k,j}}
\right).
\]

\(\rho(z)\) is zero for \(z\leq0\) and a frozen nonnegative penalty outside the
observed tangent extent. This represents a tube around a bounded
lower-dimensional patch rather than a solid ambient ellipsoid.

## Controls

- frozen v6.1 weighted affine volume head;
- rank-32 class-conditional Gaussian retained by v7;
- kNN support distance;
- RBF SVM same-space control;
- unbounded affine residual diagnostic;
- label-permuted and random-orientation negative controls;
- score-direction reversal control.

The Gaussian, kNN, and RBF controls are not required to be editable. They bound
the predictive interpretation of any geometric result.

## Primary endpoints

### Predictive endpoint

Three-seed development balanced accuracy under a frozen class-comparison and
calibration protocol.

### Open-set co-primary

At a threshold calibrated to 92% known coverage:

- remaining-unknown recall;
- accepted-known accuracy;
- AUROC;
- review precision at the frozen review budget.

A method cannot advance on balanced accuracy while failing the registered
unknown-recall or accepted-known safety bands.

### Surface diagnostic endpoints

- signed own-class depth distribution;
- metric-corrected absolute boundary distance;
- fraction of known examples in calibrated near-surface, deep-interior, and
  exterior strata;
- class-conditional error by depth stratum;
- nearest-other-class occupancy inside each primitive interior;
- score separation between own-class, competing known classes, and unknowns.

These are diagnostics and cannot establish the main claim alone.

### Conditional lifecycle endpoint

If M53 passes, the winning eligible geometric score enters the frozen v8
episode harness. Episode utility remains:

\[
U_e =
\operatorname{BA}_{K_e\cup D_e}(\text{child})
-
\operatorname{BA}_{K_e\cup D_e}(\text{parent}).
\]

The review budget remains 50 and all v8 safety and transaction invariants
remain binding.

## Registered advancement gates

### M51 diagnostic gate

The score-only study advances beyond frozen-component comparison only if:

1. at least 60% of classes on every seed show greater own-class concentration
   in a preregistered near-surface band than in an equal-mass deep-interior
   band; and
2. deep-interior points have at least 2 points worse class precision or at
   least 2 points greater competing-class occupancy than near-surface points;
   and
3. the direction is consistent on at least 7/9 seed-by-diagnostic cells.

Failure stops learned shell fitting but does not block the distinct manifold
tube diagnostic.

### M52 score-semantics gate

On identical frozen components, a shell score advances only if it:

1. improves balanced accuracy over signed volume by at least 2.0 points in
   three-seed mean;
2. has a paired 95% bootstrap lower bound above zero;
3. improves at least 2/3 seeds;
4. loses no more than 1.0 point of accepted-known accuracy;
5. loses no more than 2.0 points of unknown recall; and
6. replays selected widths, thresholds, and predictions exactly.

### M53 geometry gate

A bounded tube or fitted shell advances only if it:

1. exceeds the frozen signed-volume baseline by at least 3.0 balanced-accuracy
   points;
2. exceeds the unbounded affine residual on unknown recall by at least 5.0
   points at matched known coverage;
3. is within 3.0 points of the Gaussian control in balanced accuracy;
4. satisfies the M52 open-set safety bands;
5. improves at least 7/9 matched seed-by-protocol cells;
6. has a paired 95% bootstrap lower bound above zero against volume; and
7. satisfies parameter, fit-work, replay, and lineage budgets.

The 3-point Gaussian proximity gate is a relevance threshold, not a reopening
of v6.1 parity.

### M54 lifecycle gate

The lifecycle test opens only if M53 passes. The winning geometry must:

1. improve mean \(U_e\) by at least 2.0 points over the matched v8 Gaussian
   episode baseline at the same review set and budget;
2. preserve known-class balanced accuracy within 1.0 point;
3. preserve remaining-unknown recall within 2.0 points;
4. improve at least 7/9 episode cells;
5. pass a paired 95% interval above zero; and
6. preserve confirmation, graph, replay, rollback, and fallback invariants.

## Claim restrictions

V9 may not claim:

- that deep features universally lie on hypersurfaces;
- that a shell result proves a generative data manifold;
- exact Euclidean SDF distance for anisotropic or fused geometry;
- parity with RBF unless a future separately registered program tests it;
- lifecycle qualification from predictive or OOD results alone;
- an adaptation advantage unless M54 opens and passes;
- independent final confirmation;
- that prior volume-based outcomes were invalid.

## Outcome taxonomy

- **Outcome A:** bounded manifold-tube support passes M53 and M54.
- **Outcome B:** shell support passes M52/M53 but not lifecycle M54.
- **Outcome C:** bounded tube passes predictive/open-set gates but M54 is not
  opened or fails.
- **Outcome D:** surface/tube diagnostics show signal but practical predictive
  gates fail.
- **Outcome E:** no reproducible surface or bounded-tube signal.
- **Outcome F:** leakage, lineage, replay, or transaction failure; publication
  prohibited.

## Advancement ledger

| Milestone | Decision | Status |
|---|---|---|
| M51 | Freeze protocol and test support occupancy on existing components | Complete; failed 0/3 gate operands |
| M52 | Compare volume and shell score semantics without refitting | Closed: 0/48 diagnostics had a meaningful negative interior |
| M53 | Fit and compare bounded tube, fitted shell, density, kNN, and RBF controls | Stopped at S1: all ranks accepted 100% of 8x tangent probes |
| M54 | Test the retained geometry in frozen v8 episodes | Blocked: no M53-eligible geometry |
| M55 | Artifact-only final replay and claim ledger | Complete |

### M51 disposition

The exact A2 component zero level sets did not enclose a meaningful own-class
interior in any of 48 normalized or metric-corrected class-by-seed diagnostics.
Per-seed supporting-class fractions were 0%, the precision/occupancy practical
difference was 0 points, and 0/9 seed-by-diagnostic directions were consistent.
The independent replay produced the same evidence SHA-256
`2d80c30114eb71d23d46b0c18380a4be6bf497c8c07a2c892ed23fd0aaf4ffd1`.
H1 and M52 are closed. H2 remained open after M51 because proximity to a
bounded lower-dimensional patch does not require the prior A2 radial zero level
set to act as a support boundary; its subsequent disposition is recorded below.

### M53-S1 disposition

H2 produced predictive and OOD signal but failed the mandatory bounded-open-space
test. Rank 16 improved seed-11 known balanced accuracy by 1.375 points and rank
32 by 1.625 points; both also improved unknown recall by more than 30 points.
However, the bounded and unbounded variants were observationally identical, and
every bounded rank accepted every synthetic probe at 8x its fitted tangent
extent. No rank advances to S2. This is terminal **Outcome D** rather than
Outcome E because a reproducible lower-dimensional residual signal existed but
the practical safety gate failed.

### M55 final replay

The artifact-only verifier checked two immutable milestone indexes, seven
indexed artifacts, seven branch dispositions, and eight conclusion operands.
Two executions produced byte-identical evidence SHA-256
`119a02c47b92f0b413ef5ac8f4454592caac3d69b06a4cc5cfa4a2007d508929`.
No training features were loaded and no final labels were opened. The
authoritative final interpretation is frozen in
`analysis/V9_FINAL_CLAIM_LEDGER.md`.
