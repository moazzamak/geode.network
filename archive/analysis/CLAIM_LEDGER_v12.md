# GEODE v12 Claim Ledger

**Program:** Metric-Faithful Geometric Support
**Status:** Final Outcome E; M76 complete
**Registration date:** 28 July 2026
**Immutable parents:** v6.1 D, v7 C, v8 D, v9 D, v10 D, v11 E

## Claim boundary

V12 tests whether explicit multiclass analytic score fields can learn
competitively and remain exactly inspectable when their far field is trained
with Eikonal, deterministic probe, distribution, and separation constraints.
The defended contribution is the conjunction. V12 does not claim novelty for
Gaussian or Mahalanobis scoring, distance-aware uncertainty, feature-collapse
prevention, gradient regularization, virtual outliers, interpolation shaping,
prototype explanations, additive decomposition, or standard faithfulness
metrics individually.

## Gating hypotheses

- **H1 diagnostic generality:** the v11 threshold, probe-acceptance, or
  interpenetration pathology persists with larger calibration samples and on a
  native-resolution corpus.
- **H2 Gaussian-head learnability:** the frozen low-rank Gaussian reaches the
  registered L1/L2 bar over seeds 11, 23, and 37 while providing measurable
  I1--I4 inspectability.
- **H3 metric-faithful field:** conditional on M70, probe-trained Eikonal fields
  reduce the median threshold ratio to at most 2.0 and held-out-family 4x probe
  acceptance below 1% without losing more than 1.0 point versus the M71
  Gaussian.
- **H4 transfer and inspectability:** a retained stage passes held-out probes,
  real OOD, second-corpus transfer, and the registered standard inspectability
  protocols.

## Prior-art boundary after M69

All four M69 threads completed without outright displacement. Required adjacent
work includes VOS/NPOS and Outlier Exposure; DeepSDF/IGR/SIREN/SAL/DiGS and
certified distance bounds; DUQ/SNGP/DUE/DDU, Mahalanobis, Deep SVDD/SAD, and
IGD; ProtoPNet/ProtoTree, CBM, SENN, NAM/EBM; and deletion/insertion, ROAR,
comprehensiveness/sufficiency, simulatability, and counterfactual proximity.

V12 may defend only:

1. per-class analytic feature-space fields with Eikonal equality rather than a
   Lipschitz lower bound or encoder-Jacobian penalty;
2. deterministic fitted-geometry probe families trained and evaluated at both
   source-component and full-system levels; and
3. the conjunction of explicit class geometry, exact directional score
   decomposition, and closed-form minimum feature-space counterfactual reach.

## Inspectability operands

- **I1 intrinsic parameter semantics**: structural audit.
- **I2 exact score decomposition (completeness/local accuracy)**: mean and
  maximum numerical residual.
- **I3 deletion/comprehensiveness faithfulness**: top-k versus random-k and
  bottom-k over multiple k; ROAR only with retraining.
- **I4 minimum counterfactual distance/proximity with validity**: flip success
  and distance distribution in feature space.
- **I5 simulatability proxy / forward-simulation probe accuracy**: chance and
  no-explanation controls with leakage prevention.

The claim concerns the head over coordinates. If the representation is learned,
the feature coordinates themselves are not claimed to be semantically
interpretable.

## Advancement ledger

| Milestone | Decision | Status |
|---|---|---|
| M69 | Refresh four prior-art threads | Passed: partial overlap, no outright displacement |
| M70 | Re-examine sample size, resolution/domain, and class count | Passed diagnostic gate: pathology persists and worsens with class count |
| M71 | Evaluate Gaussian as a classifier with L1--L4 and I1--I4 | Complete, not qualified: L1 and I4 fail; L2 is a statistically indeterminate near miss |
| M72 | Train a probe/Eikonal head on frozen features | Failed: threshold ratio passed without training effect; held-out safety and accuracy failed |
| M73 | Escalate representation change | Stage 1 passed: material M72 improvement and load-bearing collapse prevention |
| M74 | Confirm and transfer | Failed: only L1 parity and threshold ratio passed; Outcome E |
| M75 | Qualify inspectability | Partial structural result; full I1--I5 conjunction failed |
| M76 | Final artifact replay | Complete: 7 indexes, 10 artifacts, 16 operands, 8 branches |

## Protocol restrictions

Disjoint partitions, sealed final labels, matched controls, lineage locks,
byte-identical replay, and fail-closed integrity remain mandatory. M70 may
require explicit amendments to prior ledgers. No single-corpus result satisfies
transfer, no trained-probe-only rejection satisfies open-space competence, and
no automated probe is described as human simulatability.

## M70 observations

- D1 threshold ratios remained 5.6--5.9 through n=800; n=800 is explicitly a
  bootstrap extrapolation beyond 600 unique pooled scores per class.
- On frozen CIFAR features, 4x source acceptance was 0%, yet system acceptance
  remained 74.414--81.763%. Component masking, not source extrapolation alone,
  remains load-bearing.
- On 12,800 native DomainNet images, system 4x acceptance was 100% at 8, 32,
  and 128 classes; 8x acceptance rose from 97.778% to 100%.
- Other-class penetration below the q90 floor rose monotonically with class
  count. The prior narrow cell understated deployment risk.
- The artifact/noise branches are rejected. No prior-ledger amendment is
  required beyond preserving their already narrow scope.

## M71 observations

- Rank-32 Gaussian accuracy was 95.708% versus 96.917% RBF. The -1.208-point
  difference and both paired intervals miss the registered parity tolerance.
- Unknown recall was 86.833% versus the 87.0% bar. The exact 95% interval
  [83.862%, 89.435%] includes the bar, so this is a formal near miss, not
  substantive evidence of inferior open-set competence.
- I1 and I2 pass. I3 shows a small descriptive top-direction deletion effect.
  I4 is unavailable for general pairwise low-rank Gaussian quadrics.
- The Gaussian remains M72's compact 0.813 MB baseline, not a retained final
  v12 model.

## M72 observations

- Median threshold ratio was 1.168 before and after training. The formal
  threshold operand passed, but training did not cause the pass.
- Known accuracy improved 93.250% to 93.750% but remained 1.375 points below
  the M71 Gaussian, failing non-regression.
- Axis-tangent system acceptance fell 98.926% to 56.116%. Held-out corner
  acceptance fell 100% to 0%, demonstrating limited cross-family transfer.
- Held-out mixed, masking, normal, and cross-class-bridge acceptance remained
  100%; the worst held-out 4x rate therefore failed at 100%.
- Calibration Eikonal error worsened despite its registered loss. The absolute
  probe hinge was inactive from epoch 1 because it did not share the eventual
  conformal score scale.
- Exact replay and analytic score/gradient tests passed. The residual axis and
  held-out-corner improvements open only M73 Stage 1, not M74.

## M73 observations

- A single jointly trained 64x384 projection raised known balanced accuracy
  from M72's 93.750% to 95.625%; median threshold ratio remained qualified at
  1.282.
- Worst held-out 4x acceptance fell 100% to 75%. Corner probes were rejected
  completely, but mixed probes remained 75% accepted; masking and normal probes
  remained 100% accepted. Stage 1 therefore passes escalation, not open-space
  safety.
- The constrained projection remained rank 64. Its mean relative pair-distance
  drift was 0.123 versus 0.567 for the zero-constraint ablation, while row
  orthogonality error was 0.00695 versus 0.01295. Both registered 20% ablation
  contrasts passed, establishing that collapse prevention changed the intended
  geometric operands.
- The ablation was nevertheless better operationally: 96.375% versus 95.625%
  accuracy, 77.5% versus 59.5% unknown recall, and 0% rather than 75--100%
  mixed/masking/normal acceptance. Collapse prevention is load-bearing for
  geometric preservation, not beneficial for prediction or rejection in this
  cell.
- Mean absolute Eikonal error was 0.629 and cross-class-bridge acceptance was
  87.5%. These remain explicit limitations for M74.
- Two CPU runs produced byte-identical evidence; no final labels were opened.

## M74 observations

- Three-seed field accuracy averaged 96.083% versus 96.917% RBF. The
  -0.833-point mean passed the one-point L1 tolerance, although both paired 95%
  intervals were wholly below zero.
- Mean proxy-unknown recall was 66.333%, far below the 87% L2 bar. Native
  DomainNet real-OOD recall ranged from 53.516% to 76.289%.
- The held-out corner family transferred at 0% acceptance, but held-out mixed
  acceptance was 75--100%; masking and normal acceptance remained 100%.
- On native DomainNet with 32 known and 96 unseen classes, the field reached
  66.719% accuracy, 7.344 points below logistic, and only 0.833% unknown recall.
  The median ratio passed at 1.754 but did not imply open-set competence.
- Removing the probe loss left all displayed seed-11 operational metrics
  unchanged. Eikonal and separation removal were also negligible. Distribution
  removal reduced unknown recall to 41.5%, while classification removal reduced
  accuracy to 95.250%.
- M74 therefore supports only narrow primary-corpus accuracy parity. It rejects
  the claimed generalized probe-trained open-space mechanism and takes Outcome
  E. Exact replay passed and no final labels were opened.

## M75 observations

- I1 passes only for the decision rule over learned coordinates. Coordinate
  semantics remain explicitly unclaimed.
- I2 exact squared-score decomposition passes with maximum residual
  `1.14e-13`.
- I3 top-direction deletion reduces predicted-component distance scores much
  more than random or bottom deletion at k=1,4,8, but changes predictions in at
  most 0.125% of examples. This is score attribution, not meaningful decision
  comprehensiveness, and is not ROAR.
- I4 fails: the multiclass anisotropic quadratic-plus-rejection boundary has no
  registered closed-form minimum Euclidean counterfactual displacement. The
  field value is not that distance, especially with non-unit measured gradient
  norms.
- I5 reaches 17.737% balanced accuracy versus a 12.5% no-explanation baseline,
  below kNN's 25.246% and RBF's 22.772%. This is weak automated
  forward-simulation evidence, not human simulatability.
- The full inspectability claim is not qualified. Exact replay passed, no final
  labels were opened, and Outcome E is unchanged.

## M76 observations

- The artifact-only verifier reproduced all 16 conclusion operands from 7
  immutable indexes covering 10 artifacts.
- Eight branch dispositions resolve M69--M76 without loading training features
  or opening final labels.
- M70 requires no amendment to the v11 ledger.
- Two final conclusion replays were byte-identical. V12 closes as Outcome E.
