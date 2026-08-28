# GEODE v12 Final Claim Ledger

**Program:** Metric-Faithful Geometric Support
**Final outcome:** E
**Finalized:** 28 July 2026

## Registered question

V12 tested whether explicit multiclass analytic score fields could learn
competitively while providing bounded open-space behavior and exact,
standardized inspectability. It reordered the program around learnability and
inspectability, permitted staged representation learning, and retained strict
partition, lineage, sealed-label, and byte-identical replay contracts.

The defended contribution was the conjunction of analytic per-class fields,
Eikonal and geometry-derived probe training, source- and system-level rejection,
exact directional decomposition, and valid counterfactual reach. V12 did not
claim novelty for Gaussian scoring, distance-aware uncertainty, collapse
prevention, gradient regularization, probe training, prototype explanation, or
additive decomposition individually.

## Executed evidence

### M69: prior-art refresh

Four threads found extensive adjacent work but no outright displacement of the
full registered conjunction. Claims were narrowed accordingly, and I1--I5 were
renamed to standard intrinsic-semantics, completeness/local-accuracy,
deletion/comprehensiveness, counterfactual-proximity, and forward-simulation
protocols.

### M70: diagnostic re-examination

Threshold ratios remained approximately 5.6--5.9 through the registered sample
curve; n=800 was explicitly a bootstrap extrapolation beyond 600 unique pooled
scores per class. On 12,800 native DomainNet images, system 4x acceptance was
100% at 8, 32, and 128 classes, while other-class penetration increased with
class count. The frozen-coordinate pathology was therefore not explained by
small calibration samples or CIFAR upsampling.

M70 did not invalidate the already narrow v9--v11 claims, so no retrospective
v11 ledger amendment is required.

### M71: Gaussian classifier baseline

The rank-32 Gaussian reached 95.708% known balanced accuracy versus 96.917% for
RBF, missing the one-point L1 tolerance by 0.208 point beyond the boundary.
Unknown recall was 86.833% versus the 87% bar, with a 95% exact interval of
[83.862%, 89.435%]; this was a formal near miss, not substantive evidence of
inferior open-set competence. I1 and I2 passed, while closed-form minimum
Euclidean counterfactual reach was unavailable for general pairwise Gaussian
quadrics.

### M72: frozen-feature Stage 0

The trained analytic field's threshold ratio passed at 1.168 but was unchanged
from initialization. Known accuracy was 93.750%, 1.375 points below the M71
seed-11 Gaussian, and worst held-out 4x acceptance was 100%. Axis and held-out
corner acceptance improved, providing the registered residual signal for M73,
but the field was not safe.

### M73: learned-projection Stage 1

A jointly trained 64D projection raised seed-11 accuracy to 95.625%, kept the
threshold ratio at 1.282, and reduced worst held-out 4x acceptance from 100% to
75%. Collapse prevention was load-bearing for its registered geometric
diagnostics: pair-distance drift fell from 0.567 to 0.123 and row-orthogonality
error from 0.01295 to 0.00695 while preserving rank 64.

The unconstrained ablation was nevertheless operationally stronger, with
96.375% accuracy, 77.5% unknown recall, and 0% mixed, masking, and normal
acceptance. The constrained arm retained 100% masking and normal acceptance.
M73 passed only its escalation gate.

### M74: confirmation and transfer

Across seeds 11, 23, and 37, the constrained field averaged 96.083% accuracy
versus 96.917% for RBF. The -0.833-point mean passed the one-point L1 tolerance,
although both paired 95% intervals were wholly below zero.

All decisive open-set and transfer operands failed:

- mean proxy-unknown recall was 66.333% versus the 87% bar;
- held-out mixed acceptance was 75--100%;
- native DomainNet real-OOD recall was 53.516--76.289%;
- 32-known/96-unknown DomainNet transfer accuracy was 66.719%, 7.344 points
  below logistic; and
- DomainNet transfer unknown recall was 0.833%.

Removing the probe loss left all displayed seed-11 operational outcomes
unchanged. Eikonal and separation removal were also negligible. The registered
probe/Eikonal objectives did not establish generalized open-space rejection.

### M75: descriptive inspectability

I1 passed for the decision rule over learned coordinates, not for coordinate
semantics. I2 exact squared-score decomposition passed with maximum residual
`1.14e-13`.

I3 top-direction deletion reduced predicted-component distance scores much more
than random or bottom deletion, but flipped at most 0.125% of decisions. This is
score attribution, not meaningful decision comprehensiveness, and is not ROAR.

I4 failed structurally. The actual multiclass anisotropic
quadratic-plus-rejection decision has no registered closed-form minimum
Euclidean counterfactual displacement, and measured non-unit gradients prevent
interpreting field values as that distance.

I5 reached 17.737% balanced accuracy versus 12.5% chance/no-explanation, below
kNN at 25.246% and RBF at 22.772%. This is weak automated forward-simulation
evidence, not human simulatability. The full inspectability conjunction did not
qualify.

## Final branch dispositions

| Branch                                    | Disposition                                                         |
| ----------------------------------------- | ------------------------------------------------------------------- |
| M69 prior-art refresh                     | Complete; conjunction retained with narrowed claims                 |
| M70 sample/domain/class-count diagnostics | Complete; pathology persisted and worsened                          |
| M71 Gaussian classifier baseline          | Complete; not qualified                                             |
| M72 frozen-feature Stage 0                | Failed with residual signal                                         |
| M73 learned-projection Stage 1            | Escalation gate passed; constrained state retained for confirmation |
| M74 three-seed confirmation and transfer  | Failed; Outcome E declared                                          |
| M75 inspectability qualification          | Partial structural evidence only; full conjunction failed           |
| M76 artifact-only finalization            | Complete                                                            |

## Final interpretation

V12 has **Outcome E**. A learned linear projection made the analytic field
competitive on primary-corpus known-class accuracy, but the mechanism did not
generalize as open-space support. Rejection failed on a held-out mixed family,
real cross-corpus OOD, and native DomainNet transfer. The probe objective was
not operationally load-bearing, and the assumed closed-form counterfactual
advantage was absent for the deployed anisotropic multiclass decision.

The durable positive result is narrow: the retained head has named geometric
parameters and an exact directional decomposition, and a 64D projection can
recover near-RBF classification accuracy. These properties do not establish
open-set competence, transfer, meaningful decision faithfulness, or complete
inspectability.

## Claim restrictions

V12 does not establish:

- generalized open-space rejection;
- L2 open-set competence;
- L5 second-corpus transfer;
- held-out-family probe generalization;
- benefit from the registered probe or Eikonal loss;
- closed-form minimum Euclidean counterfactual reach for the deployed decision;
- human simulatability or strong automated forward simulation;
- semantic meaning of learned feature coordinates; or
- independent final-label confirmation.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m experiments.tier4.verify_v12_final
```

The verifier reads only immutable artifact indexes, evidence JSON, and this
ledger. It loads no training features and opens no final labels.

---

## Amendment A1 — probe objective was ill-posed (29 July 2026)

**Source:** v13 M77 probe-degeneracy forensics.
**Evidence:** `logs/results/v13/m77_probe_degeneracy/evidence.json`.
**Status of Outcome E:** unchanged.

M77 replayed the sealed M73 seed-11 configuration under instrumentation that
reproduced the v12 optimizer history with a maximum absolute delta of `0.0` and
a matching trained-state hash. It established:

- own-class probe scores for `axis_tangent`, `masking`, and `normal` are
  exactly `4.0` at every epoch, and are invariant under rescaling of every
  fitted extent across three orders of magnitude (all four trained families
  invariant below `1e-9`);
- the own class is the minimising class for 98.694% of probes, unchanged
  between the first and last epoch, so the hinge has effectively no cross-class
  component;
- the probe term's gradient norm is at most `6.78e-17` with respect to
  `log_tangent`, `3.45e-18` with respect to `log_residual`, and `1.39e-18` with
  respect to `centers`, against a total-objective gradient norm of `0.287` with
  respect to `log_tangent` at epoch 1;
- the recorded probe-loss decrease from `9.649` to `2.318` is 101.5% explained
  by the detached adaptive target `2 * median(own_scores)` falling from `13.795`
  to `6.353`; the mean minimising probe score rose by `0.078`.

The registered probe objective was therefore a constant with respect to the
geometry and could not move the open-space boundary.

**Amended statements.**

| Original ledger statement                                                                           | Amended status                                                                                                                                      |
| --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| "The registered probe/Eikonal objectives did not establish generalized open-space rejection." (M74) | Downgraded from finding to **untested condition**. The probe objective was ill-posed; no conclusion about probe training is supported.              |
| "Removing the probe loss left all displayed seed-11 operational outcomes unchanged." (M74)          | Retained as an observation, but it is **forced by construction** and carries no evidential weight.                                                  |
| "The trained analytic field's threshold ratio ... was unchanged from initialization." (M72)         | Retained as an observation; the mechanism is now identified as the degenerate probe term, not a property of analytic fields.                        |
| Claim restriction: "benefit from the registered probe or Eikonal loss"                              | Restated: the registered probe loss was **never validly exercised**. The Eikonal component is unaffected by this amendment and its negative stands. |

**Not amended.** Outcome E stands. The open-set failures on held-out probe
families, real cross-corpus OOD, and DomainNet transfer rest on evaluation
evidence that does not depend on the probe objective. All other claim
restrictions are unchanged.

---

## Amendment A2 — the transfer deficit was a rank artifact (29 July 2026)

**Source:** v13 M78 sample-adequacy forensics (amendments R1, R2).
**Evidence:** `logs/results/v13/m78_sample_adequacy/evidence.json`.
**Status of Outcome E:** unchanged.

M78 re-ran the M74 DomainNet transfer arm across rank ∈ {2, 4, 8, 16, 32} at
the M74 sample count, and across `geometry_per_class` ∈ {20, 40, 60}, over
seeds 11/23/37, on the identical M70 native feature array and partition
contract. It also measured **basis identifiability** as the mean principal
angle between per-class subspaces fitted on disjoint halves of the geometry
split through a single shared projection, referenced against a Monte-Carlo
random-subspace baseline.

It established:

- **The registered M74 configuration could not have been a fair test.** At
  rank 32 with 60 samples per class the samples-per-fitted-dimension ratio is
  **1.88**, against the v13 floor of 10, and identifiability is **0.193** — the
  fitted bases are barely distinguishable from random subspaces of the same
  shape.
- **`geometry_per_class: 60` was a ceiling, not a choice.** The M70 array holds
  exactly 100 observations per class, so the M74 partition contract left no
  headroom whatsoever.
- **Reducing rank recovers the accuracy.** At rank 4, transfer known balanced
  accuracy is **74.219%** against a logistic control of **74.063%** on the same
  partitions — a gap of **+0.156 points**. At rank 2 the gap is −0.260 points.
  The M74 figure of −7.344 points was obtained at rank 32, whose accuracy is
  65.938%.
- **Identifiability and accuracy fall together with rank**
  (0.393, 0.363, 0.346, 0.276, 0.193 at ranks 2, 4, 8, 16, 32), a monotone
  dose–response consistent with basis estimation, not with transfer.

**Amended statements.**

| Original ledger statement                                                             | Amended status                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Claim restriction: "L5 second-corpus transfer"                                        | Restated: **confounded with basis identifiability at the registered rank.** The v12 evidence does not show a transfer deficit; it shows a rank-32 estimation failure. At an adequately sampled rank the head reaches parity with logistic regression on DomainNet transfer. |
| "The geometric head loses 7.344 points against logistic under corpus transfer." (M74) | **Withdrawn.** The measurement was taken in a cell that is void under sample adequacy.                                                                                                                                                                                      |

**Not amended, and strengthened.** The **open-set** transfer negative stands.
Unknown recall at 60 samples per class is 0.50%–0.80% across every rank tested,
against 20.42% for the logistic control, and the best low-rank change is
**−0.07 points**. Sample adequacy explains the accuracy failure and explains
none of the open-set failure. Removing the accuracy confound isolates the
open-set negative rather than weakening it. Outcome E stands.
