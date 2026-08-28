# GEODE v11 Claim Ledger

**Program:** Directional Conformal Support Envelopes over a Delegated Head  
**Status:** Final Outcome E; M68 artifact-only replay complete
**Registration date:** 28 July 2026  
**Immutable parents:** v9 Outcome D, v10 Outcome D (plus v6.1 D, v7 C, v8 D)

## Claim boundary

V11 changes the registered role of the geometric model:

> An explicit, editable, directionally fit, conformally calibrated bounded
> geometric envelope can decide answerability — improving open-set safety and
> lifecycle utility over the strongest frozen density support model — while a
> delegated frozen discriminative head decides class, at non-inferior
> composite accuracy.

V11 does not reopen codimension-one shells, CSG, fitter variants, per-class
temperatures, Euclidean accuracy parity, or the v6.1/v10 closed accuracy
outcomes. It does not claim the true data manifold. Nonlinear warps remain
out of scope and untested.

## Parent evidence

1. **v9 S1:** rank-16/32 bounded tubes improved seed-11 unknown recall from
   60.5% (A2) to 91.5--92.5% while all bounded ranks accepted 100% of 8x
   tangent probes — support signal real, bound ineffective.
2. **v10 M58:** with dimensionless scores, 16/18 cells stayed
   calibration-infeasible because of system-level component masking; the one
   feasible cell passed every safety operand and failed only the 1.0-point
   accuracy-superiority gate.
3. **M30.2:** cosine-native geometry beat matched Euclidean geometry by 3.97
   points over three seeds; never composed with the tube line.
4. **v8 M47:** coverage-aware review beat density-core review by 3.593
   points, establishing the lifecycle endpoint the envelope must serve.

V11 may correct the model role, the acceptance rule, and the metric. It may
not tune any parameter from v9/v10 development failures.

## Registered hypotheses

- **H1 — masking is fixable:** per-class conformal thresholds plus a
  cross-class contrast condition plus negative-guided extents reduce
  system-level masking-probe acceptance to at most 1% at 4x extent under
  controlled conditions where the v10 minimum-score rule exceeds 20%.
- **H2 — directional envelope:** a bounded directional tube/atlas under H1
  machinery improves unknown recall over the frozen rank-32 Gaussian by at
  least 2.0 points at matched 92% known coverage, at composite accuracy
  non-inferior to the frozen head, on real frozen features.
- **H3 — lifecycle value:** the retained envelope improves frozen v8 episode
  utility by at least 2.0 points over the Gaussian baseline.

H1 is a controlled-condition prerequisite; its failure ends the program at
Outcome E. H2 is the real-feature claim. H3 is conditional on H2.

## Frozen inputs

- DINOv2-small 384-dimensional features and split identities from v6.1--v10;
- CIFAR-10 classes 0--7 known, classes 8--9 proxy unknown;
- seeds 11, 23, and 37;
- v9/v10 partition hashes and evidence indexes;
- frozen A2, v7 rank-32 Gaussian, kNN-support, RBF, and logistic controls;
- delegated-head predictions hash-locked at M63 before any envelope fit;
- final labels remain sealed.

## Registered acceptance rule

Scores are the v10 dimensionless \(s_k = q_{\perp,k} + \lambda
q_{\parallel,k}\), computed in the tangent plane of each class-patch mean
direction on L2-normalized features. Per-class split conformal thresholds
\(\tau_k\) at miscoverage \(\alpha = 0.08\). Accept iff:

1. \(s_{k^*}/\tau_{k^*} \le 1\) for the normalized-minimum class \(k^*\); and
2. \(\min_{j \ne k^*} s_j/\tau_j - s_{k^*}/\tau_{k^*} \ge \delta\) or
   \(s_{k^*}/\tau_{k^*} \le 1 - \delta\).

Grids: ranks \(\{8,16,32\}\); patches per class \(\{1,2,4\}\); extent
policies quantile / negative-guided / negative-guided-IQR; contrast margins
\(\{0, 0.05, 0.1, 0.2\}\). All selection uses calibration observations and
label-free probes only.

## Primary endpoints

At matched 92% known coverage:

- remaining-unknown recall (co-primary with safety);
- composite known balanced accuracy (non-inferiority guardrail);
- accepted-known balanced accuracy;
- tangent, masking, bridge, mixed, and random-direction probe acceptance,
  source-level and system-level;
- AUROC and FPR95;
- parameter count, fit work, latency, and replay identity;
- conditional: v8 episode utility at the 50-label budget.

## Advancement gates

### M63–M64 protocol, conformal, and masking identifiability

All partition, lineage, conformal-coverage, determinism, and replay tests
pass. On the adversarial masking scene, the contrast rule cuts 4x
masking-probe acceptance to ≤1% where the replicated v10 rule exceeds 20%,
and negative-guided extents reduce masking acceptance at matched coverage.
Straight geodesic tubes recover registered ranks exactly with in-support
coverage in [90%, 94%] and 0% acceptance at 8x.

### M65 seed-11 screen

Retain at most one cell that: improves unknown recall over the Gaussian
control by ≥2.0 points; is within 1.0 point of the frozen head on composite
known balanced accuracy and of head-only on accepted-known accuracy; accepts
0% at 8x, ≤1% at 4x (tangent and masking, system level), ≤5% bridge/mixed/
random; covers ≥6/8 known classes; stays within 2x A2 budgets; passes atlas
stability where applicable; and replays exactly. Ties: fewer patches, lower
rank, policy order 2/3/1, smaller margin.

### M66 three-seed confirmation

Mean unknown-recall gain ≥2.0 points with paired 95% bootstrap lower bound
above zero, improvement on ≥2/3 seeds, and every M65 operand on every seed.

### M67 lifecycle gate

Mean episode utility gain ≥2.0 points over the Gaussian baseline, paired 95%
interval above zero, ≥7/9 cells improved, known accuracy within 1.0 point,
remaining-unknown recall within 2.0 points, exact rollback and fallback
preserved.

## Kill switches

- M64 masking-gate failure ends the program at Outcome E before any
  real-feature fitting.
- Cells with infeasible negative-guided extents, insufficient per-class
  conformal counts, or no safe contrast margin fail closed.
- Atlas instability or budget overrun closes atlas cells.
- No development-selected retry anywhere; M67 opens only after a full M66
  pass.
- Partition leakage, head retraining, final-label access, or replay mismatch
  produces Outcome F.

## Outcomes

- **Outcome A:** the envelope passes M66 and M67; end-to-end support-envelope
  result.
- **Outcome B:** the envelope passes M66 but not lifecycle.
- **Outcome C:** only an atlas cell passes, establishing curved/multimodal
  local support without lifecycle value.
- **Outcome D:** safety or non-inferiority holds but the unknown-recall gain
  does not; the envelope adds nothing over density support.
- **Outcome E:** the masking mechanism fails under controlled conditions, or
  the directional signal disappears; the envelope line closes.
- **Outcome F:** protocol integrity failure.

## Claim restrictions

V11 may not claim:

- classification superiority of geometry over any control;
- rescue of the v6.1/v10 accuracy outcomes;
- a safe envelope from synthetic evidence alone;
- lifecycle utility unless M67 opens and passes;
- the true data manifold, or anything about nonlinear representations;
- independent final confirmation.

## Advancement ledger

| Milestone | Decision | Status |
|---|---|---|
| M63 | Lock parents, roles, conformal machinery, probes, schemas | Passed after delegated-head lineage repair: known-only fit/calibration, 96.25% head accuracy, exact replay |
| M64 | Establish masking fix and directional identifiability | Passed: 9/9 exact ranks, 92.389% pooled coverage, masking 100% to 0%, exact replay |
| M65 | Run seed-11 directional envelope screen | Terminal: 0/27 retained; 18 extent-infeasible, 9 contrast-infeasible |
| M66 | Confirm one envelope over three seeds | Blocked: no M65 cell |
| M67 | Test lifecycle utility | Blocked: M66 cannot pass |
| M68 | Final artifact replay | Passed: 3 indexes, 11 artifacts, 12 conclusion operands, exact two-run replay |

## M63 observations

- Directional geometry, conformal calibration, contrast acceptance, geodesic
  probes, endpoint decomposition, and resource accounting are implemented and
  schema-validated.
- The deterministic fixture selected margin 0 and rejected every registered 4x
  and 8x tangent and 4x masking probe while covering 46/49 observations in each
  class.
- The near-tie unit construction requires margin 0.1 to demonstrate the
  registered contrast effect; the fixture's margin-0 result is not evidence for
  H1.
- Negative-guided fixture extents stayed at their policy-1 upper bounds because
  all 4x calibration negatives were already excluded. Focused tests establish
  the upper-bound and 0.90-floor invariants, but M64 must test whether the policy
  contracts extents and reduces masking at matched coverage.
- M63 opens M64 only. No real-feature, open-set, accuracy, or lifecycle claim
  follows from this protocol qualification.
- A pre-M65 audit rejected the original all-class M27 head lock because it had
  seen proxy-unknown classes 8--9. M63 was repaired before any real envelope
  fit: the replacement RBF uses only classes 0--7, fits on `geometry_fit`,
  calibrates without retraining on `score_calibration`, and hash-locks exact
  predictions. This prevented an Outcome-F execution rather than waiving the
  lineage restriction.

## M64 observations

- H1 passed under the registered controlled construction: contrast reduced 4x
  masking acceptance from 100% under the v10 rule to 0%, and negative-guided
  extents independently reduced it to 0% at matched 92.462% coverage.
- All nine straight directional tubes recovered rank exactly, pooled coverage
  was 92.389%, and all 4x/8x tangent probes were rejected.
- Individual straight-cell coverage ranged from 86.750% to 97.500%. The
  registered gate is pooled, but M65 must report every class threshold and
  coverage rather than hiding heterogeneity in the pooled result.
- Reusing observations for both extent estimation and conformal thresholding
  produced 89.514% independent coverage in the first implementation attempt.
  The passing protocol uses independent deterministic subsets for those two
  operations. This separation is mandatory in M65.
- M64 opens M65. It establishes controlled identifiability and masking
  mechanism behavior only; H2, real-feature safety, and predictive
  non-inferiority remain untested.

## M65 observations

- No cell reached evaluation eligibility. Eighteen negative-guided cells were
  infeasible above the registered 0.90 floor; nine quantile cells exhausted all
  four contrast margins without passing calibration safety.
- Quantile-cell system acceptance was 98.828--100% at 4x tangent extent,
  92.578--98.828% at 8x, and 96.875--100% on 4x masking probes. Every
  cross-class bridge was accepted.
- The contrast result is stronger than a threshold miss: margins 0--0.2 barely
  changed acceptance because probes were scored inside competing calibrated
  supports rather than as shallow near-ties.
- Extent and conformal calibration used disjoint 800-observation subsets, so
  the failure is not the same-sample undercoverage defect found and repaired in
  M64.
- H1 remains supported only in the controlled adversarial scene. H2 is not
  established; M66 and M67 are blocked, and M68 must freeze the terminal
  outcome.

## M68 finalization

- Final Outcome E closes the directional envelope line on the frozen features.
  It does not retroactively classify M64 as a failure: controlled H1 passed.
- The registered mechanism failed to transfer to M65 real features. Contrast
  could not reject probes deeply accepted by competing supports, and
  negative-guided extents could not preserve the 0.90 own-class floor while
  excluding other-class calibration observations.
- The artifact-only verifier checked three immutable indexes, 11 indexed
  artifacts, 12 conclusion operands, and six branch dispositions without
  loading training features or opening final labels.
- Two final executions produced byte-identical evidence and index files. M66
  and M67 remain blocked; no H2, non-inferiority, confirmation, or lifecycle
  claim is available.
