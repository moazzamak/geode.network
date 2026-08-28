# GEODE Research Implementation Plan v6.1

**Status:** preregistration draft after the completed v6 M31 gate  
**Parent protocol:** `analysis/RESEARCH_IMPLEMENTATION_PLAN_v6.md`  
**Parent evidence:** `logs/results/v6/m31_factorial_s2/evidence.json`  
**Date:** 26 July 2026

## 1. Purpose and amendment boundary

M31 completed the v6 predictive sequence. Direct-label rank-32 radial subspaces
reached 90.77% mean S2 development balanced accuracy, but remained 6.00 points
behind the 96.77% strongest same-space control. The frozen two-point parity
operand failed. Therefore:

- v6 Outcome D remains the correct result for the tested v6 family;
- M32 topology search, M33 OOD, M35 broad confirmation, M36 migration, and M37
  amortization remain closed as rescue paths;
- no v6 final-test result may be reopened for selection;
- this amendment cannot retroactively convert M31 into a passing result.

v6.1 registers two previously untested capacity hypotheses as a new exploratory
family:

1. a rank-32 tangent-space spherical cap combining directional geometry with
   intrinsic-rank local shape;
2. a nonnegative per-component weighted readout testing whether equal component
   weights caused a material part of the remaining approximation error.

It also registers a one-seed component-scaling diagnostic, an independent
Outcome-C lifecycle frontier, and a support-honest Flowers feasibility branch.

### 1.1 Prior-art attribution

The citation keys below refer to the authoritative bibliography and attribution
map in `analysis/RESEARCH_REPORT_v5.md`.

- The M29/M31 center-plus-low-rank-basis components belong to the established
  PPCA/factor-analyzer and local-subspace lineage [L68-L72]. The GEODE component
  is not identical to probabilistic PPCA: it uses axis-scaled in-subspace
  coordinates, an isotropic off-subspace residual, and a discriminative radial
  field rather than the PPCA likelihood and EM objective.
- M28's temperature-controlled teacher KL and margin matching are knowledge
  distillation/model compression [L73,L74]. TREPAN and soft decision trees are
  direct precedents for distillation into interpretable students [L75,L76].
- M30's unit directions and angular caps draw on directional statistics,
  von Mises-Fisher mixtures, cosine classifiers, imprinted weights, and
  hyperspherical prototypes [L77-L81]. Its explicit angular support boundary is
  a repository-specific use, not a claim to invent directional classification.
- A1-T's sphere logarithm map followed by tangent-plane PCA is an instance of
  the principal-geodesic/Riemannian-statistics construction [L82,L83].
- Direct-label forward component addition follows the error-reduction lineage
  of orthogonal least-squares RBF selection, kernel matching pursuit, and
  matching pursuit [L84-L86].
- A1-W follows the classic two-stage RBF pattern of constructing basis units and
  then fitting supervised output weights [L87]. Here the frozen components were
  selected with labels and the readout is restricted to nonnegative per-class
  simplex weights plus one global temperature, so it is not identical to the
  Moody-Darken training procedure.
- The principal RBF SVM control and M28 teacher are standard kernel
  support-vector machines [L88,L89]. Conformal sets [L90,L91], reservoir
  representations [L92,L93], shrinkage and robust covariance fitters [L94-L96],
  a Levina-Bickel-style local intrinsic-dimension statistic [L97],
  L-BFGS/L-BFGS-B [L98,L100], and distance-weighted kNN [L99] are likewise
  established methods used as mechanisms, diagnostics, optimizers, or controls
  rather than GEODE inventions. The M19 code uses a `k` rather than `k-1`
  numerator convention, so its diagnostic must not be described as the exact
  published Levina-Bickel estimator.

## 2. Frozen evidence and claims

Before any v6.1 model run:

1. add the M31 negative-parity result to the claim ledger and README;
2. mark v6 M32, M33, M35 predictive confirmation, M36, and M37 as closed;
3. hash-lock the v6 protocol, M29-M31 configurations, selected students,
   predictions, evidence, and artifact indexes;
4. record the three S2 parent representation hashes and their normalized child
   hashes;
5. preserve all M27-M31 negative and blocked cells.

The initial claim status is:

| Claim | Status after M31 |
| --- | --- |
| Same-space predictive parity | negative |
| Directional geometry improves matched Euclidean geometry | supported on S2 |
| Direct rank-32 subspaces improve retained explicit controls | supported, but not parity |
| Proper likelihood improves this family | negative |
| Flowers few-shot support for rank 32 | blocked |
| Audited edit/rollback advantage | eligible for independent Outcome-C evaluation |

Required outputs:

- `analysis/CLAIM_LEDGER_v6_1.md`;
- updated README Current Status table;
- an immutable v6.1 parent lock under `logs/results/v6_1/a0_parent_lock/`.

## 3. Protocol rules

### 3.1 Data and seed stages

| Stage | Purpose | Seeds/data |
| --- | --- | --- |
| A0 | API, geometry, gradients, schemas | deterministic synthetic fixtures |
| A1 | cheap falsification | frozen seed-11 DINOv2 train/development cache |
| A2 | amended retention gate | frozen seeds `11, 23, 37` |
| A3 | lifecycle frontier | frozen edit suite on the same three retained seeds |
| A4 | Flowers support feasibility | official five-shot training partition |

Seed `42` may be used only for disposable implementation debugging. It cannot
select a reported method or enter an interval.

### 3.2 Label and test policy

- Training labels may fit components and readout weights.
- Development labels may apply registered gates and select among preregistered
  cells.
- Final-test labels remain sealed throughout v6.1.
- A2 is a development retention study, not independent confirmation.
- No result in A3 or A4 may rescue a failed A2 predictive gate.

### 3.3 Fixed controls

Every applicable DINOv2 comparison includes:

- RBF SVM;
- weighted kNN;
- current spherical GEODE;
- v6 M31 direct rank-32 radial student;
- v6 M30 directional cap;
- the corresponding equal-weight readout for every weighted-readout cell.

All controls and proposed cells consume the same seed-specific cache and split.

### 3.4 Complexity and lifecycle rules

Report component count, scalar parameters, array bytes, serialized bytes,
candidate evaluations, fit wall time, and inference wall time. A weighted
readout counts every stored scalar weight. A tangent-cap basis counts all
ambient-by-rank entries.

Every retained student must pass:

- canonical serialization and fail-closed representation binding;
- byte-identical fit and prediction replay;
- deterministic component and weight ordering;
- exact rollback to the parent JSON and predictions;
- local-edit changed-region measurement;
- at least 99.9% prediction preservation outside that measured region.

Registered predictive budgets:

| Family | Components | Maximum stored scalar parameters | Maximum candidate evaluations |
| --- | ---: | ---: | ---: |
| M30 directional control | 46 | 17,710 | 5,000 |
| M31 rank-32 affine control | 46 | 584,430 | 5,000 |
| Rank-32 tangent cap | 46 | 584,476 | 5,000 |
| Weighted rank-32 affine | 46 | 584,476 | 5,000 |
| Weighted rank-32 tangent cap | 46 | 584,522 | 5,000 |

The parameter counts include every component weight but not derived array
metadata. A1 may report a cell as infeasible; it may not silently reduce rank,
omit weights, or exceed these limits. Weighted inference may use at most 1.2
times its equal-weight parent's median wall time on the same seed and hardware.

## 4. A0: amendment and parent lock

Create a v6.1 schema that records:

- amendment ID and parent v6 hashes;
- primitive family, rank, score units, normalization policy, and support rule;
- readout family, weight constraints, regularization, and global temperature;
- component, parameter, and fit-work budgets;
- data stage, seed, split hashes, and representation lineage;
- gate operands, stopped status, and test-label access.

The A0 gate passes only if deliberate mismatches in parent representation,
normalization, split, primitive rank, or readout schema fail closed and the
parent lock replays without loading training data.

## 5. A1-T: rank-32 tangent-space spherical caps

### 5.1 Primitive contract

For a normalized input direction \(x\) and unit component direction \(\mu\):

1. compute angular distance \(\theta=\arccos(\operatorname{clip}(x^\top\mu))\);
2. map \(x\) into the tangent plane at \(\mu\) using a numerically stable sphere
   logarithm map;
3. represent local tangent variation with a deterministic orthonormal rank-32
   basis and positive tangent variances;
4. represent unmodeled tangent residual with one positive isotropic variance;
5. retain an explicit angular support radius in radians.

The registered rank is 32, inherited from the retained M29 rank. The minimum
support is `r+2=34`. PCA basis signs use the existing deterministic convention.
No alternate rank is selected in A1-T.

Primary score:

- normalized tangent radial field with an angular support boundary.

Descriptive secondary output:

- factorized tangent Gaussian log likelihood, reported but not eligible for
  promotion because likelihood cells failed throughout M29/M31.

### 5.2 Matched A1 comparison

On seed 11, compare:

1. M30 angular cap;
2. M31 direct rank-32 affine subspace;
3. direct rank-32 tangent cap.

Hold teacher arrays, training examples, boundary cohort, anchor policy,
candidate count, direct-label objective, exact 46-component budget, global
temperature policy, and development split fixed. Report parameter-matched
controls where feasible; never describe a comparison as parameter-matched when
rank-32 support makes it infeasible.

### 5.3 Tests

- log-map fixtures at zero, small angles, and near the antipode;
- tangent basis orthogonality to the mean direction;
- scale invariance before explicit normalization;
- unit mean direction after fitting and edits;
- finite-difference score gradients;
- deterministic PCA signs;
- `r+2` support rejection;
- serialization, lineage mismatch rejection, replay, and rollback.

### 5.4 A1-T gate

Advance tangent caps to A2 only if they:

1. improve seed-11 development balanced accuracy by at least 0.5 points over
   both matched M30 caps and the M31 rank-32 control, **or** reduce NLL by at
   least 5% with balanced-accuracy loss no greater than 0.25 points;
2. use exactly 46 components and remain within the registered parameter and
   fit-work limits;
3. pass replay, rollback, and 99.9% outside-region preservation.

Otherwise stop A1-T and retain the affine rank-32 primitive.

## 6. A1-W: nonnegative weighted component readout

### 6.1 Readout contract

For class \(k\), component fields \(f_{km}(x)\), and component weights
\(w_{km}\):

\[
s_k(x)=\log\sum_m w_{km}\exp[-f_{km}(x)].
\]

Constraints:

- \(w_{km}\ge 0\);
- weights sum to one within each class;
- one global temperature shared by all classes;
- no per-class temperatures;
- no negative weights;
- no hidden feature network or unconstrained mixture;
- weights, optimizer state summary, and ordering are explicit and serialized.

Parameterize each class's weights by a softmax over explicit log weights. Fit
them by deterministic L-BFGS cross-entropy optimization on the frozen training
boundary cohort with centered-log-weight L2 penalty `1e-4`, at most 500
iterations, gradient tolerance `1e-8`, and zero initialization (exactly equal
initial weights). Development labels apply the gate only. These values are
hash-locked in A0 and are not swept on seed 11. The equal-weight model is the
exact nested control.

### 6.2 A1 cells

Run weighted and equal-weight readouts on:

1. the M31 direct rank-32 affine-subspace candidates;
2. tangent-cap candidates only if A1-T passed.

The component set is frozen before fitting weights. Weighted fitting cannot add,
delete, split, merge, or move a component.

### 6.3 Tests

- nonnegativity and per-class normalization;
- permutation-equivariant component ordering;
- equal weights reproduce the existing softmin;
- deterministic optimizer convergence;
- zero-weight and single-component behavior;
- global-temperature-only enforcement;
- serialization and parameter accounting;
- component-weight local edit, changed region, and exact rollback.

### 6.4 A1-W gate

Advance weighted readout to A2 only if, against its exact equal-weight parent, it:

1. improves development balanced accuracy by at least 0.5 points, **or** reduces
   NLL by at least 5% with accuracy loss no greater than 0.25 points;
2. does not increase inference wall time by more than 20%;
3. passes replay, rollback, and 99.9% outside-region preservation.

If one component receives more than 90% of its class weight in a majority of
classes, report readout collapse and do not claim a distributed geometric
mixture.

## 7. A1-B: component-budget scaling diagnostic

After A1-T and A1-W, run seed 11 for the frozen M31 direct rank-32 radial family
at exact component counts:

`[10, 20, 30, 46, 60, 80, 100, 120]`.

Use the same candidate bank and direct-label objective for every point. Report:

- development balanced accuracy and NLL;
- teacher agreement and margin error;
- components per class;
- parameter/array/serialized bytes;
- candidate evaluations, fit time, and inference time;
- marginal accuracy gain per additional ten components.

This curve is diagnostic. It cannot select a larger A2 budget or reopen a cell
that failed A1. Classify the result as:

- **budget-limited:** the 80-120 component slope remains at least 0.1 points per
  ten components;
- **saturated:** the best gain above 80 components is below 0.1 points;
- **unstable:** accuracy or NLL materially reverses as components increase.

## 8. A2: amended three-seed retention

Open A2 only for mechanisms that passed their A1 gate. Freeze the complete
primitive, readout, component count, optimizer, temperature, and edit policy
before running seeds `11, 23, 37`.

Mandatory A2 cells:

- strongest v6 same-space RBF/kNN control;
- M31 direct rank-32 radial control;
- every A1-passing tangent/weighted mechanism;
- the combination of tangent caps and weighted readout only if both individual
  mechanisms passed A1.

Primary selection is mean development balanced accuracy; tiebreakers are NLL,
serialized bytes, then fit time.

### A2 parity gate

The v6.1 predictive amendment passes only if one fully frozen cell:

1. lies within 2.0 points of the strongest same-space RBF/kNN control;
2. is not worse than M31 by more than 0.25 points;
3. passes byte-identical replay, exact rollback, and 99.9% outside-region
   preservation on every seed;
4. stays inside its registered component, parameter, inference, and fit-work
   limits.

Report seed-level paired 95% intervals and pooled per-example bootstrap
intervals. A2 does not open final-test confirmation.

### Final predictive kill switch

If every amended cell remains more than 2.0 points behind on A2:

- mark v6.1 predictive status as final Outcome D;
- stop tangent/readout predictive development;
- do not open topology, OOD, migration, amortization, or confirmation as rescue;
- publish the capacity curve and mechanism failures with the negative result.

## 9. A3: Outcome-C lifecycle frontier

A3 is independent of A2 predictive success and cannot alter Outcome D. Compare:

- M31 direct rank-32 radial GEODE;
- an A2-retained v6.1 model, if one exists;
- current spherical GEODE;
- RBF SVM;
- weighted kNN;
- one compact explicit control where edit semantics are well defined.

Use the already frozen edit tasks:

- local false-positive correction;
- known-class mode addition;
- corrupted-cluster suppression;
- bounded-shift recalibration;
- exact rollback.

Report:

- balanced accuracy and explicit accuracy cost to RBF/kNN;
- unaffected-prediction preservation;
- changed-region size;
- rollback success and rollback latency;
- accepted-edit evidence count;
- edit and inference latency;
- model, audit, and review artifact bytes;
- deterministic fit-work and operator count.

Black-box controls must receive a declared analogous update or be marked
unsupported for that operation; unsupported operations are not scored as zero
cost.

### Outcome-C gate

Claim a specialized lifecycle tradeoff only if the retained explicit head:

1. is non-dominated on the full accuracy/lifecycle frontier;
2. has a paired advantage over every accuracy-superior control in at least one
   of unaffected preservation, rollback reliability, accepted-edit evidence, or
   edit latency;
3. passes exact rollback on every seed and task;
4. reports the full predictive deficit prominently.

Otherwise report lifecycle safety qualification without a comparative
advantage claim.

## 10. A4: Flowers support tiers

Register two distinct outcomes; never mix them.

### A4-F5: official five-shot feasibility

- maximum subspace/tangent rank: 3;
- minimum support: `r+2=5`;
- no rank-32 comparison;
- objective: determine whether the primitive can fit and replay, not establish
  competitiveness.

Report linear, prototype, kNN, rank-3 affine subspace, and rank-3 tangent-cap
controls on the official development protocol.

### A4-F34: expanded-support conditional tier

Run rank 32 only if at least 34 labeled training examples per class are available
without using development or test labels for fitting. Hash and name the new
support protocol separately. If such support is unavailable, record A4-F34 as
blocked; do not combine official train and development partitions silently.

No Flowers result may rescue the DINOv2 A2 gate.

## 11. Physical E7 qualification

Physical multi-host E7 remains independent. It may proceed in parallel because
it tests distributed recovery, not predictive quality. Its result cannot promote
or select a v6.1 model.

## 12. Execution order

1. **A0:** update claim ledger/README and lock v6 parent evidence.
2. **A1-T:** implement and falsify rank-32 tangent-space caps.
3. **A1-W:** test nonnegative weighted readout on frozen candidate sets.
4. **A1-B:** run the one-seed component-budget diagnostic.
5. Freeze all A1-passing mechanisms.
6. **A2:** run three-seed amended retention and apply the final predictive kill
   switch.
7. **A3:** run the Outcome-C lifecycle frontier independently.
8. **A4:** record five-shot rank-3 feasibility and expanded-support status.
9. Finalize the claim ledger, stopped-branch table, artifact-only reproduction,
   and publication narrative.
10. Run physical E7 independently when infrastructure is available.

## 13. Final outputs

1. v6.1 parent and artifact indexes with complete lineage;
2. updated README and claim ledger;
3. tangent-cap geometry and replay report;
4. equal-versus-weighted readout table;
5. component-budget scaling curve;
6. A2 three-seed parity table and intervals, if opened;
7. accuracy-editability-lifecycle Pareto table and plots;
8. Flowers support-feasibility table;
9. complete negative, blocked, infeasible, and stopped-branch ledger;
10. artifact-only reproduction command that loads no training data.

## 14. Interpretation boundaries

- A1 is exploratory falsification, not confirmation.
- A2 is amended development retention, not independent test confirmation.
- Outcome D remains attached to v6 regardless of v6.1.
- A lifecycle advantage does not imply predictive competitiveness.
- Weighted components remain explicit, but learned weights reduce the strength
  of any “pure geometry” claim and must be described as an explicit geometric
  basis with a constrained learned readout.
- No additive-gain assumption between M29 and M30 is evidence until the tangent
  primitive is directly measured.
- If A2 fails, the durable contribution is the audited lifecycle framework plus
  a mechanism-resolved structural negative result.
