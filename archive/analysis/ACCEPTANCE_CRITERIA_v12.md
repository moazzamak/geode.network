# GEODE Revised Acceptance Criteria (v12 frame)

**Date:** 28 July 2026  
**Supersedes:** the implicit acceptance frame used in v5--v11  
**Status:** registered framing document; not a claim ledger

## 1. Decision

The program's acceptance frame is re-ordered:

| Priority | Axis | Role |
|---|---|---|
| Primary | **Learnability** | Does the system actually learn well? |
| Co-primary | **Inspectability** | Can its decisions be explained exactly and faithfully? |
| Deferred | Editability, rollback, lifecycle utility | Measured descriptively; no longer gating |

Rationale: v5--v11 optimized an artifact for editability and lifecycle
auditability, then repeatedly discovered that the artifact did not learn well
enough for those properties to matter. Three consecutive programs (v9, v10,
v11) were terminated by safety or predictive gates before any lifecycle claim
could be opened. An unlearnable model's editability is not a research result.

The frame therefore inverts: establish that the system learns competitively
and can be inspected exactly; only then re-derive the lifecycle contracts.

## 2. Learnability — definition and operands

**Definition.** The system reaches statistical parity with the strongest
matched-compute, matched-data alternative head, and does so with comparable
sample efficiency, without memorizing the training set.

Registered operands:

- **L1 accuracy parity:** known balanced accuracy within 1.0 point of the
  strongest matched control (RBF, logistic, fine-tuned softmax as applicable),
  with a paired 95% bootstrap interval, over seeds 11, 23, and 37.
- **L2 open-set competence:** unknown recall at matched known coverage at
  least equal to the strongest support control (currently the v7 low-rank
  Gaussian at 87.0%).
- **L3 sample efficiency:** performance measured at 50, 200, and 1,000
  examples per class; a component family whose minimum viable size scales with
  ambient dimension is disqualified from the few-shot regime (the W2 defect).
- **L4 not-memorization:** parameter count and serialized size materially
  below the kNN control (6.02 MB) at equal or better accuracy.
- **L5 transfer:** the result reproduces on at least one dataset beyond
  CIFAR-10 known classes.

L1 and L2 are gating. L3--L5 are reported and gate only where a specific
claim depends on them.

## 3. Inspectability — definition and operands

Inspectability has been asserted throughout the project and **never measured**.
As a co-primary axis it now requires falsifiable operands:

- **I1 intrinsic parameter semantics:** every model parameter maps to a named geometric
  or statistical quantity. Verified structurally; binary.
- **I2 exact score decomposition (completeness/local accuracy):** for any
  prediction, the responsible component and
  the per-direction contributions decompose the score **exactly** (residual
  zero to numerical tolerance), not approximately.
- **I3 deletion/comprehensiveness faithfulness:** ablating the top-k attributed
  directions changes the prediction significantly more than ablating k random
  or bottom-ranked directions over a registered k sweep. Use the name ROAR only
  if the model is retrained on ablated train and test data.
- **I4 minimum counterfactual distance/proximity with validity:** the minimum
  feature-space displacement that flips a decision is computable in closed
  form, and the displaced point empirically flips the decision. No input-space
  plausibility claim follows without a decoder or manifold constraint.
- **I5 simulatability proxy / forward-simulation probe accuracy:** a simple
  probe given only the explanation predicts the model's output on held-out
  cases better than chance and no-explanation baselines, without example
  leakage. This automated measure is not canonical human simulatability.

I1, I2, and I4 are where an explicit per-class geometric head can structurally
dominate RBF, kNN, and MLP heads. I1 and I2 are not novel alone: SENN, NAM,
EBM/GA2M, and prototype/path models establish intrinsic and exact decomposable
scoring at other abstraction levels. The defensible claim is the conjunction
of explicit class geometry, exact directional decomposition, and closed-form
feature-space counterfactual reach.

**Scope warning.** If the representation is learned, inspectability covers the
**head over learned coordinates**, not the coordinates themselves. The claim
weakens accordingly and must be stated as such: the decision rule is exactly
inspectable; the feature semantics are not.

## 4. Editability — demoted

Editability, edit locality, transactional rollback, and lifecycle utility are
retained as **descriptive measurements and engineering machinery**, not gates.
Consequences:

- the 99.9% edit-locality contract is retired as a gate (A3 measured 84.2%;
  this is no longer a program failure);
- the frozen A2 46-component budget is retired as a constraint — notably
  reopening the M31/A1-B budget curve, which peaked at 100 components;
- exhaustive-routing and transactional-adapter requirements no longer
  constrain model or inference design;
- the v8 50-label utility gate no longer blocks advancement; three programs
  (v8 M47, v10 M61, v11 M67) were held behind it.

These may be reinstated as gates once learnability and inspectability are
established.

## 5. What does not change

Protocol integrity is unaffected and remains mandatory:

- preregistration of hypotheses, grids, gates, and kill switches before
  execution;
- disjoint partitions, sealed final-confirmation labels, lineage locks;
- controls matched on data, compute, and representation;
- byte-identical replay and artifact-only verification;
- fail-closed handling of leakage, retraining, or lineage violations.

These are what make the v9--v11 negatives credible rather than anecdotal, and
they cost almost nothing. Demoting editability changes **what is optimized**,
not **how it is verified**.

## 6. Status of prior negative results under the new frame

| Prior result | Status |
|---|---|
| v9/v10/v11 hard-boundary open-space failures | **Stand**, scoped to frozen DINOv2 coordinates; they were safety failures, not editability failures |
| M28 boundary distillation into spheres (-6.30 pp) | Stands, scoped to frozen features |
| M18 metric/support policy negative | Stands |
| M4/M13 CSG subtraction null | Stands |
| A1-B budget instability at 120 components | **Reopened** — was constrained by the editability-driven budget |
| A3 lifecycle frontier locality failure | **No longer a failure**; demoted to descriptive |
| v8 M47 utility shortfall | **Deferred**, not a blocker |
| Low-rank Gaussian as a classifier | **Never tested**; the v11 control incidentally measured 95.13% vs RBF 96.25% on seed 11 |

The last row is the most consequential: the "GEODE trails by 5--6 points"
narrative is a property of the SDF heads, not of explicit geometry. The
Gaussian head has never been evaluated under a registered classification
protocol.

## 7. Representation policy

**Decision: full fine-tuning of the trunk jointly with the geometric head is
permitted.** The frozen-trunk contract is retired as an inviolable constraint.

This directly targets the v11 diagnosis — class supports are heavy-tailed and
mutually interpenetrating **in frozen coordinates** — by shaping the
representation to the geometry instead of fitting geometry to fixed features.

Three constraints follow and must be registered before any run:

### 7.1 Feature collapse is the primary technical risk

Training a distance- or density-based head end-to-end permits the trunk to
destroy distance semantics while minimizing the discriminative loss. This is
the known failure mode of the adjacent literature (DUQ, SNGP, DDU). A
collapse-prevention mechanism (spectral normalization, two-sided gradient
penalty, or an equivalent registered alternative) is **required**, and an
ablation demonstrating that it is load-bearing is a required operand.

### 7.2 Prior art moves closer, not further

Under the frozen-lifecycle frame, the v7 audit's composition finding shielded
the novelty claim. Under learnability + inspectability with a learned trunk,
the program competes directly with DUQ, SNGP, DDU, Mahalanobis/GMM heads,
prototype-based interpretable classifiers, and concept-bottleneck models —
several already listed in `PRIOR_ART_AUDIT_v7.md`. A refreshed prior-art audit
is a prerequisite, and the differentiator must be explicit: bounded support
plus registered inspectability operands plus the audited protocol, not "a
Gaussian head."

### 7.3 Compute and determinism conflict

Measured environment: 8-core Ryzen 7 7800X3D, 63 GB RAM, RX 9070 XT, and
**torch 2.13.0+cpu with no GPU backend**. Full ViT-S/14 fine-tuning is
feasible per-run on CPU but not across a multi-seed grid. Adding a GPU backend
(for example torch-directml) risks nondeterminism that would break the
byte-identical replay requirement in Section 5.

Registered resolution: **stage the representation intervention** —
(a) frozen head baseline, (b) learned projection, (c) partial fine-tuning
(last-k blocks or LoRA, reduced resolution), (d) full fine-tuning — with each
stage carrying a cheap kill switch, and full fine-tuning entered only when a
cheaper stage demonstrates the mechanism. Determinism policy for any GPU
backend must be resolved before that backend is used for gated evidence.

## 8. Immediate consequence

The cheapest high-value experiment implied by this frame is the one nobody
ran: **evaluate the existing low-rank Gaussian as a first-class classifier
over seeds 11, 23, and 37 against RBF, logistic, and kNN, with the L1--L4 and
I1--I4 operands.** Every input is already frozen; it either establishes
near-parity with an inspectable head or removes the optimism immediately, and
it anchors the bar that any fine-tuned system must beat.
