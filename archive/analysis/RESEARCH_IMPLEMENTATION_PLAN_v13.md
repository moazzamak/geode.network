# GEODE Research Implementation Plan v13

## Nameable Bases and Absolute Boundaries: Sparse Concept Geometry with Measured Out-Group Exposure

**Status:** registered; no milestone executed
**Date:** 29 July 2026
**Claim ledger:** `analysis/CLAIM_LEDGER_v13.md` (created by M79)
**Acceptance frame:** `analysis/ACCEPTANCE_CRITERIA_v13.md` (created by M79)
**Immutable parents:** v6.1 D, v7 C, v8 D, v9 D, v10 D, v11 E, v12 E

---

## 1. Program decision

### 1.1 The v12 post-mortem revealed two implementation defects, not two findings

V12 declared Outcome E on the strength of two negatives. Source inspection
after finalization shows both are mechanical artifacts of the implementation
rather than properties of geometric learning.

**Defect 1 — the probe objective is scale-relative and therefore vacuous.**

`experiments/common/v12_metric_fields.py` places every axis probe at

```python
center + sign * 4.0 * tangent_scales[class_index, axis] * basis[:, axis]
```

and the probe hinge is `relu(probe_margin - min(probe_scores))` with
`probe_margin_multiplier: 2.0`. The score of that probe under its own field is
identically `sqrt((4 sigma)^2 / sigma^2) = 4.0`, independent of `sigma`. The
own-class term of the hinge therefore evaluates `relu(2.0 - 4.0) = 0` at
initialization and at every subsequent step. It contributes exactly zero
gradient to the parameter it was registered to constrain. The term is non-zero
only when a _different_ class claims the probe point, so the loss penalizes
cross-class interference and never open-space extent.

Two v12 observations that were reported as findings follow directly:

- M72: "threshold ratio passed at 1.168 but was **unchanged from
  initialization**";
- M74: "Removing the probe loss left all displayed seed-11 operational
  outcomes **unchanged**".

The registered v12 novelty — the multi-family probe taxonomy at calibrated
multiples of fitted extent — could not have worked as specified. **A boundary
cannot be learned in units of itself.**

**Defect 2 — the transfer arm was under-powered at the basis-estimation step.**

`experiments/configs/v12/m74_confirmation_transfer.json` sets
`geometry_per_class: 60` with `rank: 32` in `projection_dimension: 64`. The
guard in `initialize_metric_fields` resolves the fitted rank to
`min(32, 63, 58) = 32`, so a rank-32 subspace is estimated from 60 points in
64 dimensions. Directions beyond roughly the first eight are noise. The bases
are then **frozen** — the optimizer receives only `[centers, log_tangent,
log_residual]` — so training never repairs the estimate.

The DomainNet transfer numbers (66.719% accuracy, 0.833% unknown recall) are
therefore confounded with basis-estimation noise. This is the W2 defect
registered at `ACCEPTANCE_CRITERIA_v12.md` Section 2 (L3), and it was not
applied when interpreting the L5 result.

### 1.2 The synthetic-negative programme was the wrong instrument

Correcting Defect 1 does not rescue probe training. Three of the eight
families are displacements along fitted axes, one is a normal displacement,
one is a random direction at 8x the tangent norm, and the remainder are
interpolants. Every one of them is a deterministic function of the fitted
in-group geometry. The training set contains **zero real out-group
observations** at every milestone of v9 through v12.

This is the concern raised against the program and it is correct. Fang et al.
(NeurIPS 2022), _Is Out-of-Distribution Detection Learnable?_, proves OOD
detection is not PAC-learnable in the general separate-space setting;
learnability requires either strong ID/OOD overlap conditions or access to
outlier data. Eight programs attempted the unlearnable case and recorded eight
negatives. The negatives are consistent with theory and carry no information
about the head family.

A second sampling argument applies even where the objective is well-formed.
Open-set rejection is a statement about a **boundary** in `d` dimensions, and
sampling coverage of a boundary scales exponentially in `d`. Eight hundred
in-group points in 64 dimensions place essentially no empirical constraint on
boundary location, so the accept region's shape is determined by the
parametric assumption rather than by data. This independently predicts M72's
unchanged-from-initialization result.

### 1.3 Exactness of explanation was optimized; usefulness was never moved

The co-primary inspectability axis produced one decisive measurement that no
ledger interpreted:

| Model                   | I5 forward-simulation balanced accuracy |
| ----------------------- | --------------------------------------- |
| chance / no explanation | 12.500%                                 |
| **GEODE v12**           | **17.737%**                             |
| RBF                     | 22.772%                                 |
| kNN                     | 25.246%                                 |

I2 exact decomposition passed at residual `1.14e-13`. I5 stayed 5.2 points
above chance and **lost to nearest-neighbour**. Across the tested heads, I2
and I5 are effectively uncorrelated.

The reason is structural. An explanation reading "score 3.7, of which
coordinate 43 contributed 0.21" is exact, complete, faithful, and
uninterpretable, because the coordinates have no names. Every program from v5
to v12 built an inspectable _head_ over an uninspectable _basis_, and the
ceiling on explanation usefulness was fixed by DINOv2's basis before any head
was fitted.

### 1.4 The accuracy trap

V12 reached L1 parity exactly once, and only by introducing a learned 64D
projection — which immediately triggered the scope warning at
`ACCEPTANCE_CRITERIA_v12.md` Section 3: inspectability then covers "the head
over learned coordinates, not the coordinates themselves."

| Program    | Head                                | Accuracy vs. control     | Basis                        |
| ---------- | ----------------------------------- | ------------------------ | ---------------------------- |
| v6         | SDF / radial subspace               | -6.00 pt                 | frozen, interpretable        |
| v6.1       | weighted readout, tangent caps      | -5.03 pt                 | frozen, interpretable        |
| v9--v11    | hard-boundary supports              | open-space failures      | frozen, interpretable        |
| v12 M72    | analytic field, frozen              | -1.375 pt                | frozen, interpretable        |
| v12 M73/74 | analytic field + learned projection | **-0.833 pt, passes L1** | **learned, uninterpretable** |

This is a treadmill: explicit geometry over frozen features is interpretable
and short on accuracy; buying the accuracy relocates the unexplainability into
the trunk. The program has not been failing to find the right head. It has
been discovering, eight times, that the head is not where the problem lives.

### 1.5 The gate structure cannot express the likely truth

L1 and L2 gate; I1--I5 must hold as a conjunction. That is roughly seven
simultaneous binary conditions. The design has no vocabulary for "0.8 points
worse, three times more simulatable, bounded support" — which is the most
probable state of the world and the most interesting one. V12 passed L1, I1,
and I2, held partial I3, and still terminated at E.

### 1.6 What v13 proposes

Four changes, each targeting one diagnosis above:

1. **Correct the record first.** Forensic milestones M77--M78 establish whether
   the two v12 negatives survive correction, and amend the v12 ledger before
   any new architecture is built.
2. **Fix the basis, not the head.** Replace the learned dense projection with a
   sparse, overcomplete, **nameable** concept dictionary over frozen features,
   and put a sparse head on top. I1 and I2 become trivially true and, for the
   first time, meaningfully true. I5 becomes the primary operand.
3. **Give the boundary an absolute length scale and real out-group data.**
   Two-phase curriculum: stabilize in-group geometry, freeze the scales, then
   supervise the boundary using displacements in **absolute** units and a
   measured ladder of real outlier exposure.
4. **Report a frontier, not a verdict.** Accuracy becomes a reported axis with
   a deployment-anchored tolerance; the deliverable is the
   accuracy--simulatability Pareto frontier with matched controls, including
   the post-hoc-explained neural network the program has never run.

**The central v13 thesis:** _explanation exactness and explanation usefulness
are separate quantities; the first is a property of the head and the second is
a property of the basis. Open-set competence is a third quantity and is a
property of supervision, not of geometry._ V12 optimized the first and
measured nothing about the other two.

---

## 2. Design principles

1. **Correct before constructing.** No architecture milestone opens until
   M77--M78 have disposed of the two v12 defects and amended the record.
2. **The basis is the object of study.** Head complexity is minimized on
   purpose; a sparse linear head is the default and anything more must earn
   its place against I5.
3. **Usefulness outranks exactness.** I5 is promoted to primary. I2 is
   retained but demoted to a structural check, because v12 showed it can be
   satisfied at `1e-13` with no usefulness whatsoever.
4. **No synthetic negative may be defined in units of the parameter it
   constrains.** All boundary supervision uses absolute units, verified by a
   registered invariance test.
5. **Out-group exposure is a measured variable, not a binary.** The
   contribution is the exposure ladder, not the endpoint.
6. **Sample adequacy is checked before any negative is recorded.** Every
   milestone that estimates a subspace registers its samples-per-fitted-
   dimension ratio, and a ratio below 10 invalidates a negative result.
7. **Report frontiers.** Gates remain only where deployability demands them
   (open-set) or where integrity demands them (protocol).
8. **Protocol integrity is unchanged**: preregistration, disjoint partitions,
   sealed final labels, matched controls, byte-identical replay, fail-closed
   lineage, CPU-only determinism.
9. **Every measurement operand ships with a positive control** that fails if
   the operand is not measuring what it names. Added 29 July 2026 after the
   first M78 execution shipped a stability operand that silently measured
   projection variance instead of basis identifiability; the run is void and
   retained at `logs/results/v13/m78_sample_adequacy_void_r1/`. An operand
   without a passing positive control cannot gate a decision.

---

## 3. Corpora and partitions

The v12 primary cell (8 CIFAR-10 knowns, 7x upsampled) is retired as primary
and retained only as a legacy comparison row. Registered corpora:

| Role               | Corpus                               | Notes                                                       |
| ------------------ | ------------------------------------ | ----------------------------------------------------------- |
| Primary            | DomainNet, native resolution         | 345 classes, 6 domains; `src/runtime/domainnet_manifest.py` |
| Transfer           | CIFAR-100 superclass                 | `data/tier5/cifar100_superclass.npz`                        |
| Legacy cell        | CIFAR-10 8-known / 2-proxy           | continuity with v9--v12 only                                |
| Outlier exposure   | held-out DomainNet domains + Flowers | **train-time only**; never evaluated                        |
| Far-OOD evaluation | SVHN                                 | `data/raw/test_32x32.mat`; evaluation only                  |

**Disjointness contract.** The outlier-exposure pool and every evaluation set
are disjoint by construction at the corpus level, not merely at the split
level. A single class or domain may not appear in both. Violation is Outcome F.

**Minimum sample contract.** Any milestone fitting a rank-`r` structure
registers `n_per_class / r >= 10`. At `rank: 32` this requires 320 per class.
Milestones that cannot meet it must reduce rank, not reduce samples, and a
negative recorded below the ratio is void.

Partition identities are inherited where the legacy cell is used:
`geometry_fit`, `score_calibration`, `development_eval`, `unknown_eval`,
`episode_validation`, `final_confirmation` (sealed). New corpora enter through
hash-locked manifests recording source, license, preprocessing including
resize/crop policy, and split assignment.

---

## 4. M77 — Probe degeneracy forensics (unconditional, gating the record)

**Hypothesis H77.** The v12 probe loss own-class term is identically zero, and
the M72 "unchanged from initialization" result and the M74 probe-ablation null
are consequences of that degeneracy rather than evidence about probe training.

**Operands.**

- O77.1 — instrument `train_metric_fields` to log, per epoch, the probe hinge
  decomposed into own-class and cross-class contributions, replayed on the
  sealed M73 seed-11 configuration.
- O77.2 — an analytic invariance test: for a family of `tangent_scales`
  spanning three orders of magnitude, the own-class probe score is `4.0` to
  within `1e-9`.
- O77.3 — gradient norm of the probe term with respect to `log_tangent`,
  reported per epoch.

**Registered decision rule.** If the own-class contribution is below `1e-12`
for all epochs and O77.2 holds, the degeneracy is confirmed and
`analysis/V12_FINAL_CLAIM_LEDGER.md` must be amended: the claim restriction
"benefit from the registered probe or Eikonal loss" is rewritten to state that
the probe objective was ill-posed and was never tested.

**Cost.** Instrumentation plus one replay. No new data, no sealed labels.

### 4.1 Result (29 July 2026)

**H77 confirmed. The registered decision rule fired.**

Artifacts: `logs/results/v13/m77_probe_degeneracy/`. Reproduce with
`.\.venv\Scripts\python.exe -m experiments.tier4.eval_v13_m77_probe_degeneracy`.

**Instrumentation faithfulness (prerequisite).** The instrumented loop
reproduced the v12 optimizer history with a maximum absolute delta of exactly
`0.0` across all 24 epochs and all nine recorded loss terms, and the trained
state hash matched. The diagnostics below therefore describe the v12
computation itself, not a re-implementation of it.

**O77.2 — scale invariance.** Rescaling every fitted extent across three
orders of magnitude (factors 0.1 to 100) left the mean own-class probe score
unchanged to below `1e-9` for **all four** trained families, not merely the
three predicted by algebraic inspection. `random_direction` is placed at
`8 * ||tangent_scales||` and is therefore invariant to a _global_ rescale even
though it is not aligned with a single fitted axis. The field cannot escape
its own probes by growing or shrinking.

**O77.1 — hinge decomposition.** Own-class probe scores, first and last epoch:

| Family             | Epoch 1      | Epoch 24     | Own-class hinge, epoch 1 | Epoch 24 |
| ------------------ | ------------ | ------------ | ------------------------ | -------- |
| `axis_tangent`     | **4.000000** | **4.000000** | 9.795                    | 2.353    |
| `masking`          | **4.000000** | **4.000000** | 9.795                    | 2.353    |
| `normal`           | **4.000000** | **4.000000** | 9.795                    | 2.353    |
| `random_direction` | 97.554       | 107.872      | 0.000                    | 0.000    |

Three families sit at exactly `4.0` for the entire run. The fourth sits two
orders of magnitude above the target and never contributes. The own class is
the minimising class for **98.694%** of probes, identically at epoch 1 and
epoch 24, so `mean_hinge` equals `mean_own_class_hinge` to displayed precision
throughout. There is effectively no cross-class component.

**O77.3 — gradients.** The probe term's gradient norm with respect to the
field parameters, averaged per epoch, never exceeded:

| Parameter      | Maximum probe-term gradient norm |
| -------------- | -------------------------------- |
| `log_tangent`  | `6.78e-17`                       |
| `log_residual` | `3.45e-18`                       |
| `centers`      | `1.39e-18`                       |

For contrast, the _total_ objective's gradient norm with respect to
`log_tangent` was `0.287` at epoch 1 and `0.059` at epoch 24. The other loss
terms train the extents; the probe term contributes machine zero.

**O77.4 — attribution of the apparent progress.** The recorded probe loss
falls from `9.649` to `2.318`, a drop of `7.331`, which reads as successful
probe training in the v12 history. It is not. The adaptive target
`2 * median(own_scores).detach()` falls from `13.795` to `6.353`, a drop of
`7.442`. The mean minimising probe score rises by only `0.078`.
**101.5% of the probe-loss decrease is explained by the target moving down
toward the probes, and none of it by probes being pushed out.**

**Mechanism.** The v12 probe objective is
`relu(2 * median(own_scores).detach() - min_k f_k(probe))`. The target is
detached, so it carries no gradient; the own-class probe score is algebraically
pinned at `4.0` independent of every field parameter; and the own class is the
minimiser for 98.7% of probes. The term is therefore a **constant with respect
to the geometry**, and its only visible dynamics are those of the distribution
loss reflected through the detached target.

**Consequences.**

1. Two v12 observations are explained and cease to be findings: M72's threshold
   ratio "unchanged from initialization" and M74's "removing the probe loss
   left all displayed outcomes unchanged". Both are forced by construction.
2. The registered v12 novelty — the multi-family probe taxonomy at calibrated
   multiples of fitted extent — was never tested. It could not have been.
3. Design principle 4 of this plan is vindicated and is now mandatory for all
   later milestones: **no synthetic negative may be defined in units of the
   parameter it constrains.**
4. `analysis/V12_FINAL_CLAIM_LEDGER.md` is amended (Amendment A1, 29 July
   2026). The v12 Outcome E is _not_ overturned — the open-set failures on
   held-out families, real OOD, and transfer stand on their own evidence — but
   the specific inference "the registered probe/Eikonal objectives did not
   establish generalized open-space rejection" is downgraded from a finding to
   an untested condition.

---

## 5. M78 — Sample-adequacy forensics (unconditional, gating the record)

**Hypothesis H78.** The M74 DomainNet transfer failure is driven by rank-32
basis estimation from 60 samples per class, not by a transfer property of the
head.

**Grid.** `geometry_per_class` in {60, 200, 600} crossed with `rank` in
{8, 32}, at the M74 DomainNet configuration, seeds 11/23/37. All other
settings byte-identical to `m74_confirmation_transfer.json`.

### 5.1 Registration amendment R1 (29 July 2026) — the registered grid is infeasible

The only DomainNet artifact in the repository is
`logs/results/v12/m70_native_domainnet/arrays/features.npy`: 12,800 rows of
384-dimensional features, **exactly 100 per class** across 128 classes. No raw
DomainNet imagery is present, so the array cannot be extended without
re-acquiring the corpus.

With `calibration_per_class: 20` and `evaluation_per_class: 20` fixed by the
M74 partition contract, the maximum available `geometry_per_class` is **60**.
This is itself a finding worth recording: **v12's `geometry_per_class: 60` was
not a design choice, it was the ceiling imposed by the extracted array**, and
the M74 transfer arm therefore had no headroom at all.

The grid is re-registered before execution as follows.

- **Axis A (rank sensitivity at fixed n)** — `rank` in {2, 4, 8, 16, 32} at
  `geometry_per_class: 60`. This is the decisive axis for H78: if low rank
  materially outperforms rank-32 at identical sample count, the binding
  constraint is basis identifiability, not transfer.
- **Axis B (sample sensitivity at fixed rank)** — `geometry_per_class` in
  {20, 40, 60} at `rank` in {8, 32}. A 3x span rather than the registered 10x.
- **Seeds** 11, 23, 37 for training; the partition is deterministic.
- **Operands** — transfer known balanced accuracy; transfer unknown recall;
  **subspace stability** as the mean principal angle between per-class bases
  fitted on disjoint halves of the geometry split; and the registered
  samples-per-fitted-dimension ratio from Section 3.

**Amended decision rule.** Because Axis B can no longer reach 600, the
5-point sample-slope test is replaced by:

- if rank-2/4/8 at `n = 60` improves transfer accuracy or unknown recall by
  more than 5 points over rank-32 at `n = 60`, the **W2 defect is confirmed as
  the mechanism**, the v12 L5 restriction is amended to "confounded with basis
  identifiability at the registered rank", and rank selection becomes
  sample-dependent by contract for all later milestones;
- if subspace stability at rank 32, `n = 60` is materially worse than at low
  rank, the basis is unidentified regardless of the accuracy outcome, and any
  negative recorded in that cell is void under Section 3;
- if every cell is flat on both axes, the v12 transfer negative stands and is
  strengthened;
- the registered 10x sample sweep is deferred to M85, which must re-extract
  DomainNet features at 640+ per class before any transfer claim is made.

> **Superseded by R1 (retained for the record).** The originally registered
> operands and decision rule were:
>
> _Operands._ Transfer known accuracy; transfer unknown recall; subspace
> stability measured as mean principal angle between bases fitted on disjoint
> halves of the geometry split.
>
> _Registered decision rule._ (a) If transfer accuracy or unknown recall
> improves by more than 5 points between 60 and 600 at either rank, the v12 L5
> claim restriction is amended to "untestable at the registered sample size"
> and the transfer question reopens. (b) If rank-8 at 60 materially
> outperforms rank-32 at 60, the W2 defect is confirmed as the mechanism and
> rank selection becomes sample-dependent by contract for all later
> milestones. (c) If all cells are flat, the v12 transfer negative stands and
> is strengthened.
>
> Clause (a) is unexecutable because the corpus ceiling is 60. Clauses (b) and
> (c) are carried into R1 unchanged in substance.

### 5.2 Registration amendment R2 (29 July 2026) — the stability operand was invalid

The first execution measured subspace stability by calling
`initialize_projected_metric_fields` independently on each disjoint half. That
function fits a global PCA projection before fitting per-class subspaces, so
each half received **its own projection frame** and the principal angles
between the two results measured projection variance, not basis
identifiability.

A registered positive control caught it: on a synthetic corpus whose per-class
subspaces are exactly rank 2 and identified by 400 samples per class, the
operand should return a near-zero angle and returned **40.45 degrees**.

That execution is void and retained at
`logs/results/v13/m78_sample_adequacy_void_r1/` with a `VOID.md` describing the
defect and its scope. Stability is re-registered as follows, and design
principle 9 is added to Section 2 as a consequence.

- The projection is fitted **once** on the full geometry split; both halves are
  projected through that single shared frame before per-class subspaces are
  fitted.
- The measured angle is reported against a **Monte-Carlo random-subspace
  reference** of the same shape, and converted to an `identifiability` score in
  `[0, 1]` where 1.0 means the halves recover the same subspace and 0.0 means
  the fitted subspace is indistinguishable from a random one.
- The registered identifiability floor is **0.5**. A cell below it has an
  unidentified basis and cannot carry a negative result.

### 5.3 Result (29 July 2026) — H78 confirmed, the M74 transfer negative is withdrawn

Executed at `logs/results/v13/m78_sample_adequacy/` over seeds 11/23/37.
Full table and discussion in `analysis/MILESTONE_LEDGER_v13.md`.

|      n |  rank |     n/dim |     acc % |   logit % |       gap | unknown % | ident |
| -----: | ----: | --------: | --------: | --------: | --------: | --------: | ----: |
|     60 |     2 |     30.00 |     73.80 |     74.06 |     −0.26 |      0.50 | 0.393 |
| **60** | **4** | **15.00** | **74.22** | **74.06** | **+0.16** |      0.52 | 0.363 |
|     60 |     8 |      7.50 |     70.52 |     74.06 |     −3.54 |      0.61 | 0.346 |
|     60 |    16 |      3.75 |     71.04 |     74.06 |     −3.02 |      0.73 | 0.276 |
|     60 |    32 |      1.88 |     65.94 |     74.06 |     −8.13 |      0.80 | 0.193 |

1. **W2 confirmed.** Rank 4 beats rank 32 by **+8.28 points** at identical
   sample count, above the registered 5-point threshold.
2. **The transfer deficit disappears.** M74 reported −7.34 points against
   logistic; at rank 4 the gap is **+0.16**. The v12 transfer negative is an
   artifact of rank selection.
3. **The M74 cell is void** at a ratio of 1.88 against a floor of 10. Seven of
   nine cells are void; only rank 2 and rank 4 at n=60 clear the floor.
4. **No cell is identified.** All nine fall below the 0.5 floor; the M74 cell
   scores 0.193. Identifiability falls monotonically with rank and accuracy
   tracks it.
5. **The open-set negative is untouched and now unconfounded.** Unknown recall
   is 0.50–0.80% across every rank at n=60 against 20.42% for logistic; the
   best low-rank gain is **−0.07 points**. Sample adequacy explains the
   accuracy failure and **none** of the open-set failure, which is thereby
   strengthened and passes to M83/M84 clean.

**Consequences.** `V12_FINAL_CLAIM_LEDGER.md` Amendment A2 withdraws the L5
transfer deficit. Rank selection becomes sample-dependent by contract for all
later milestones. M85 must re-extract DomainNet at 640+ per class before any
transfer claim.

---

## 6. M79 — Acceptance reframe (unconditional, blocks all architecture work)

Deliverables: `analysis/ACCEPTANCE_CRITERIA_v13.md` and
`analysis/CLAIM_LEDGER_v13.md`.

**Registered changes to the acceptance frame.**

1. **I5 becomes primary.** Forward-simulation is the operand the program
   exists to move. Registered floor for any positive claim: I5 at least 2x
   chance and strictly above the kNN control.
2. **I2 demoted to structural check.** Retained, binary, non-gating.
   Justification: v12 satisfied it at `1.14e-13` with I5 near chance.
3. **L1 becomes a reported axis with a 3.0-point tolerance**, replacing the
   1.0-point gate. Justification: the interpretable-by-design literature
   (Label-free CBM, LaBo, SpLiCE) accepts 1--5 points, and the 1.0 figure was
   never derived. The tolerance must be restated against a named deployment
   context, which the program has never specified and M79 must specify.
4. **L2 open-set remains gating.** It is the operand that determines
   deployability.
5. **New control: MLP + post-hoc explanation.** The program's stated goal is
   to beat a neural network on explainability and it has never measured one.
   Register an MLP head with SHAP and Integrated Gradients, scored on the
   identical I5 protocol. Without this control no comparative explainability
   claim may be made.
6. **Primary deliverable becomes the frontier.** Report `(accuracy, I5)` pairs
   for every head with confidence intervals. A frontier is a result wherever
   the points fall.
7. **Human-team utility registered as the substitution argument.** Where an
   accuracy/interpretability trade is claimed, it must be quantified as
   `U = acc + p(caught | explanation) * err - review_cost`, not asserted.

**Prior-art obligations.** M79 must audit and cite, without claiming
displacement: sparse autoencoders for feature decomposition (Cunningham et
al.; Bricken et al.), Label-free CBM (Oikarinen et al., ICLR 2023), LaBo,
SpLiCE (Bhalla et al., NeurIPS 2024), Rudin (Nature MI 2019) on the
tradeoff-as-myth argument and its restriction to structured features,
Rashomon-set results (Semenova & Rudin), Outlier Exposure (Hendrycks et al.,
ICLR 2019), VOS/NPOS, and Fang et al. (NeurIPS 2022) on OOD learnability.

**Registered novelty position for v13.** Not the sparse dictionary, not the
sparse head, not outlier exposure. The defended conjunction is: _a nameable
sparse basis evaluated under a registered simulatability protocol against
geometric, kernel, and post-hoc-neural controls; the measured decorrelation of
exactness from usefulness; and a quantified out-group exposure ladder._ If M79
finds this displaced, the program narrows or closes at M79.

---

## 7. M80 — Sparse concept dictionary (breakthrough arm, stage 1)

**Hypothesis H80.** Frozen DINOv2 embeddings admit an overcomplete sparse
decomposition whose atoms are substantially monosemantic, at a reconstruction
fidelity sufficient to preserve downstream accuracy.

**Construction.** Single-layer sparse autoencoder over precomputed frozen
features. Dictionary size `m` in {2048, 4096, 8192}; top-k activation with `k`
in {16, 32, 64}; non-negativity on codes. No trunk training, so the CPU
determinism constraint at `ACCEPTANCE_CRITERIA_v12.md` Section 7.3 is
satisfied and the torch 2.13.0+cpu environment is adequate.

**Operands.** Reconstruction R^2 on held-out features; mean active atoms per
image; atom dead-fraction; linear-probe accuracy on codes versus on raw
features (the fidelity cost of sparsification).

**Gate.** Advance if linear-probe accuracy on codes is within 3.0 points of
the raw-feature probe at mean active atoms `<= 64`. Otherwise sweep `m` and
`k` once, then close the arm.

> **Amendment R5 — standing null-control contract (29 July 2026, issued by
> M80, registered after M80 and before M81).**
>
> M80's gate passed on the cell m=8192/k=64, where an untrained random
> unit-norm dictionary of identical shape scored **0.02 points higher** on the
> same probe. The gate selected on accuracy under a sparsity ceiling and never
> required the selected cell to beat its own null, so as written it would have
> certified a random projection. The gate is **not** amended and its pass
> stands; restriction N80.2 records that only cells beating the null are
> admissible, and M81 carries m=8192/k=32.
>
> Generalized for the remainder of v13, in the same way M83's degeneracy test
> generalizes M77: **every comparative operand is reported alongside a null
> that shares its structure, its budget, and its split, and any gate that
> selects a winner does so only among candidates that beat that null.** An
> operand without a null is not evidence, whatever value it takes.

---

## 7a. M80 outcome and its consequences for the remaining milestones

Recorded because two registered figures moved and one operand proved
unreliable. Full result in `analysis/MILESTONE_LEDGER_v13.md`.

1. **The M81 basis is fixed at m=8192, k=32**, per N80.2 — the only cell that
   is both inside the fidelity tolerance with room to spare (0.51 points) and
   clear of its null (+3.77 points).
2. **Probe accuracy on top-k codes is not a fidelity measure at large `k`.**
   The random control reconstructs worse than the mean (held-out R² −0.66) yet
   ties the trained dictionary on the probe. Any later milestone quoting
   code-probe accuracy as evidence of decomposition quality must cite the
   null margin with it.
3. **Atom label entropy is comparative only.** With roughly 16 activations per
   atom against 128 classes, observable entropy is capped near 4 bits against
   a 7-bit uniform bound. M82's atom-purity operand inherits this ceiling and
   must be estimated at matched counts against its own shuffled control, never
   quoted as an absolute purity figure.
4. **Single-threaded torch is a hard requirement for gated evidence.**
   Measured, not assumed: two identical fits disagree at 8 threads and agree
   at 1.

---

## 8. M81 — Sparse head and the decisive I5 measurement (breakthrough arm, stage 2)

**Hypothesis H81.** Explanation usefulness is limited by basis nameability,
not by head structure. A sparse linear head over M80 atoms will exceed the
v12 geometric head on I5 by a wide margin despite being structurally simpler.

**Arms.** Sparse linear head (L1-regularized) over atoms; short decision list
over atoms; for reference, the v12 metric field refitted over atoms.

**Operands.** L1 accuracy; I5 forward simulation against chance, kNN, RBF, and
the new MLP+SHAP control; I2 exactness (trivial for the linear arm, retained
as a check); explanation length in active atoms per decision; L4 serialized
size against the 6.02 MB kNN bar.

**Registered decision rule.** This is the decisive milestone of the program.

- I5 at or above 40% (chance 12.5%): the basis hypothesis is confirmed and
  v13 proceeds to naming and rejection.
- I5 between 25% and 40%: partial confirmation; proceed but the frontier
  claim, not the dominance claim, becomes the deliverable.
- I5 at or below 25% (kNN control level): the basis hypothesis is refuted, the
  problem is deeper than the basis, and v13 closes at Outcome H with a
  well-supported negative that saves a ninth program.

> **Amendment R4 — the decision rule is restated at two registered task
> widths (29 July 2026, issued before M81 is executed).**
>
> **The defect.** The rule above is written in numbers that only exist on
> v12's task. Chance of 12.5% is 1/8: the v12 I5 protocol was 8-way
> forward simulation on CIFAR-10. The v13 corpus carries **128 classes**,
> where chance is **0.781%**. "At or above 40%" is therefore not the same
> test at all, and "at or below 25% (kNN control level)" is simply false as
> written, because 25.246% was the v12 kNN figure at 8-way and the v13 kNN
> control has never been measured.
>
> This is the **same defect as L2's 87.0% unknown-recall bar** — a threshold
> established on 8-class CIFAR-10 and carried across corpora as if it were a
> property of the operand. N1 exposed that one and M79 withdrew it. It
> survived here because M79 rewrote `ACCEPTANCE_CRITERIA_v13.md` and
> `CLAIM_LEDGER_v13.md` and **did not rewrite this plan**.
>
> **Why this is not a rescue.** M81 has not been run and no I5 number exists.
> The correction is also not a relaxation: restating a 40% bar corpus-relative
> at 128-way would _lower_ it dramatically, so the original absolute rule is
> **retained in full at the width where it was defined**, and a second,
> independent width is added that cannot be traded against it.
>
> **Registered widths. Both are measured; neither may be reported alone.**
>
> | Operand    | Protocol                                                                     | Chance | Rule                                                                                                                                     |
> | ---------- | ---------------------------------------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
> | **I5-8**   | 8-way forward simulation; the 8 classes drawn once by seeded rule and frozen | 12.50% | **The original rule, unchanged.** ≥40% confirms; 25–40% partial; ≤25% refutes. Preserves comparability with the v12 record.              |
> | **I5-128** | Full 128-way forward simulation over the M80 evaluation split                | 0.781% | Corpus-relative per Section 6: **strictly above the kNN control on the identical protocol**, by more than the seed spread over 11/23/37. |
>
> The v12 figures (GEODE 17.737%, RBF 22.772%, kNN 25.246%) are the I5-8
> reference points. **Every control is re-measured on the v13 corpus at both
> widths**; no v12 number is carried across as a bar.
>
> **Conjunction rule, registered before measurement.**
>
> | I5-8   | I5-128          | Verdict                                                                                                                                                                        |
> | ------ | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
> | ≥40%   | above kNN       | Basis hypothesis **confirmed**. Proceed to M82.                                                                                                                                |
> | ≥40%   | at or below kNN | **Task-width artifact.** Frontier claim only; the dominance claim is blocked. Registered explicitly because N1 showed this program's best numbers were narrow-width artifacts. |
> | 25–40% | above kNN       | **Partial.** Proceed; frontier is the deliverable.                                                                                                                             |
> | 25–40% | at or below kNN | **Partial and width-limited.** Proceed only to M82 naming; no comparative explainability claim.                                                                                |
> | ≤25%   | any             | **Refuted.** Outcome H. v13 closes with a well-supported negative.                                                                                                             |
>
> **Explanation length binds here.** M79 registered a deployment context of
> ≤10 active atoms per explanation. M80's admissible cells carry k=16 and
> k=32, which is encoding width, not explanation width. I5 explanations must
> cite **at most 10 atoms per decision**, chosen by contribution magnitude. If
> a head needs more to reach its I5 figure, that is reported as a
> deployment-context failure even where the I5 gate passes.

**Cost.** No new data, no trunk training, no sealed labels. This milestone can
be executed against existing frozen features.

---

## 9. M82 — Atom naming and human-facing explanation

**Hypothesis H82.** M80 atoms admit stable natural-language names, and naming
raises I5 further by converting coordinate indices into concepts.

**Method.** Two independent naming channels: CLIP text alignment over a
registered vocabulary, and top-activating exemplar sets. Names are assigned
before any I5 re-measurement and are frozen thereafter.

**Operands.** Name stability across seeds (agreement rate); inter-channel
agreement between CLIP and exemplar naming; I5 measured with named atoms
versus indexed atoms; atom purity via held-out label entropy per atom.

**Gate.** Naming is claimed only if inter-channel agreement exceeds a
registered floor and name assignment is seed-stable. Unstable naming is
reported as a negative and I5 reverts to the indexed measurement.

> **Amendment R6 — M82's floor is unregistered and its agreement operand has
> a confound (29 July 2026, issued by M80 before M82 is executed).**
>
> The gate above defers to "a registered floor" that **no document registers**.
> A numeric floor fixed after seeing the agreement rate is not a gate. It is
> registered now, before M82 runs.
>
> **The confound.** CLIP text alignment and exemplar naming are described as
> "two independent naming channels," but both are driven by the _same_
> top-activating images for an atom. They can agree at a high rate while both
> are wrong, and that agreement would measure shared input, not correctness.
> Independence is asserted in the plan and has never been tested.
>
> **Registered, per the R5 null-control contract.**
>
> 1. **Null.** Inter-channel agreement is re-measured under a shuffled
>    atom-to-exemplar assignment at matched counts. This is the floor: real
>    agreement must exceed shuffled agreement by a margin larger than the seed
>    spread over seeds 11/23/37. An absolute agreement rate is reported, never
>    gating.
> 2. **Independence check, reported.** Agreement between the two channels when
>    each is given a _disjoint_ half of an atom's top-activating exemplars. If
>    agreement collapses, the channels are one channel and the plan's claim of
>    two independent channels is withdrawn.
> 3. **Atom purity inherits the M80 ceiling.** Held-out label entropy per atom
>    is estimated at roughly 16 activations per atom against 128 classes, which
>    caps observable entropy near 4 bits against a 7-bit uniform bound. It is
>    admissible only against its own shuffled-label control at matched counts.
> 4. **Naming is frozen before any I5 re-measurement**, as already registered.
>    Names may not be revised after seeing an I5 number.

> **Amendment R8 — M82's I5 comparison is confounded with identity revelation,
> and its baseline no longer exists at 128-way (29 July 2026, issued by M81
> after execution).**
>
> H82 claims naming "raises I5 further by converting coordinate indices into
> concepts." Three things registered in M81 make that comparison unmeasurable
> as currently written.
>
> **1. The confound: naming is entangled with identity revelation.** M81's I5
> protocol **withholds component identity** (N81.2), as v12's did. M82's names
> necessarily reveal it. Comparing a named-atom I5 against M81's 40.22% would
> therefore measure _identity revelation_ — the probe gaining a stable
> per-atom column at all — and attribute it to naming.
>
> Registered: M82 measures **three** arms on the identical split, atom set,
> budget and explanation width.
>
> - **(a) identity withheld** — the M81 form, sorted magnitudes only.
> - **(b) identity revealed, unnamed** — atoms carried as stable arbitrary
>   indices, so the probe can learn per-atom columns but no name informs it.
> - **(c) identity revealed, named** — the M82 naming channels applied.
>
> The naming claim rests on **(c) − (b)**, never on (c) − (a). (b) − (a) is
> reported separately as the price of identity revelation and is itself a
> result. If (c) − (b) is within the seed spread, naming does not raise I5 and
> that is the finding.
>
> **2. The 128-way baseline does not exist.** M81 found **no accuracy-comparable
> atom arm inside the 10-atom budget at 128 classes, in any seed, at any
> tolerance from 2.5 to 15 points.** There is nothing at that width for naming
> to raise. Registered: M82 runs at both widths; at 128-way the naming delta is
> **reported and explicitly non-gating**, because its indexed baseline is
> inadmissible under N81.7. Any M82 claim is an 8-way claim and is stated as
> one.
>
> **3. The arm is fixed now, not after seeing names.** M82 carries
> **`sparse_linear_budget_256`** — M81's best 8-way arm (I5 40.22%, 5.7 cited
> atoms, 84.83% accuracy) — with `sparse_linear_budget_512` reported alongside.
> Selecting the arm after names are assigned would let the naming channel pick
> its own evaluand.
>
> **4. N81.7 and N81.8 are standing contracts, not M81 clauses.** Every I5
> number from here on carries its arm's accuracy against the best dense control
> on the same split, and its majority-prediction baseline and distinct-class
> count. An arm below the comparability floor or emitting fewer than half the
> classes is inadmissible wherever it appears.
>
> **5. Margins are read against seed spread.** M81's 8-way pass cleared its bar
> by 0.22 points against a 3.93-point spread across seeds 11/23/37 — which is
> why the milestone reports it as marginal rather than as a win. Registered: no
> M82 delta is claimed unless it exceeds the spread of its own three seeds.

> **Amendment R9 — the two naming channels cannot be independent, because the
> atoms are not in CLIP space (29 July 2026, issued by M82 before execution,
> from an artifact audit).**
>
> **The fact.** SpLiCE, which the prior-art audit correctly names as the method
> M82 borrows, decomposes **CLIP image embeddings**. Text alignment works there
> because a CLIP text embedding and a CLIP image embedding occupy one joint
> space, so an atom — a direction in that space — can be compared to a phrase
> **directly**.
>
> The v13 dictionary is fit over **384-dimensional DINOv2** features
> (`logs/results/v13/m80_sparse_dictionary/evidence.json`, `corpus.dimension`
> 384; backbone `data/v5/backbones/dinov2-small`). A v13 atom is a direction in
> DINOv2 space. **There is no alignment between DINOv2 space and any text
> space.** The dot product of a v13 atom with a CLIP text embedding is not
> merely weak — it is undefined, the two vectors having neither a shared basis
> nor a shared dimensionality.
>
> **The consequence.** The only route from a v13 atom to a phrase is
> `atom -> its top-activating images -> an image encoder -> text`. The exemplar
> channel is `atom -> its top-activating images -> inspection`. **Both channels
> pass through the same bottleneck, and no third route exists.** R6 recorded
> this as a confound that "can" make the channels agree while both are wrong.
> The audit upgrades it: the channels are not merely correlated by shared
> input, they are **structurally the same channel differing only in what reads
> the exemplars**. An inter-channel agreement rate cannot be evidence of
> correctness, and the plan's phrase "two independent naming channels" is
> **withdrawn**.
>
> **Registered in its place.**
>
> 1. **The operand is renamed to what it measures.** "Inter-channel agreement"
>    becomes **exemplar-resampling stability**: does a name survive being
>    computed from a disjoint half of the atom's exemplars, and from a different
>    seed's dictionary? R6's disjoint-exemplar check is promoted from a reported
>    side-check to the **primary** naming operand, since it is the only thing
>    the setup can actually test. R6's shuffled-exemplar null stands.
> 2. **A correctness instrument is registered, since one is available.**
>    Agreement alone cannot show a name is right, but DomainNet ships 128 ground
>    -truth class names, and M80 already measures per-atom label entropy. For
>    the subset of atoms that are strongly class-pure, the correct name is
>    approximately the class name. Naming accuracy on that subset is a
>    **positive control on the naming instrument**, in the same role as N1's
>    far-field and held-out controls. A channel that cannot name a
>    single-class atom is not naming anything, and its output on mixed atoms is
>    uninterpretable. This control is **gating**: it runs and passes before any
>    naming number is reported. Its complement — a shuffled atom-to-name
>    assignment — is the negative end.
> 3. **The image encoder is named as a substitution, not as CLIP alignment.**
>    Where text is used, the pipeline is CLIP-image-encode the exemplars, then
>    compare to CLIP text embeddings of the registered vocabulary. This is a
>    **caption-retrieval channel over exemplars**, and is described that way
>    everywhere. Nothing in v13 may be described as "CLIP-space sparse
>    decomposition"; that is SpLiCE, and v13 does not do it.
> 4. **CLIP enters as a sealed input artifact, never into the frozen venv.**
>    `transformers`, `open_clip` and `tokenizers` are absent from `.venv` and
>    installing them is forbidden. `openai/clip-vit-large-patch14` is cached
>    locally and complete. CLIP is therefore run **once**, outside `.venv`, to
>    emit text and exemplar embeddings as a hashed artifact — exactly the
>    treatment already given the DINOv2 backbone, which enters as a frozen ONNX
>    graph rather than a live model. M82's measurement then runs inside `.venv`
>    over that artifact and the replay guarantee is preserved.
> 5. **If exemplar-resampling stability fails, that is the M82 result.** H82's
>    "atoms admit stable natural-language names" is then answered negative, I5
>    reverts to the indexed measurement as the existing gate already directs,
>    and M81's N81.2 nameability burden is discharged as **unmet** rather than
>    left open.
>
> **This amendment restricts M82; it does not rescue it.** It removes the only
> operand that could have produced an impressive number from shared input.

---

## 10. M83 — Absolute-scale boundary supervision

**Hypothesis H83.** Boundary supervision expressed in absolute units, applied
after in-group geometry has stabilized, produces a boundary that moves during
training — which the v12 objective provably could not.

**Corrections registered against v12.**

1. Synthetic displacements are placed at fixed distances in a global
   feature-space unit (median inter-class centroid distance), never at
   multiples of a class's own fitted extent.
2. **Two-phase curriculum.** Phase A fits in-group geometry to convergence
   over the full class inventory with no boundary term. Scales are then
   frozen. Phase B trains the boundary only. Rationale: in v12 the negatives
   were regenerated every batch from live, unconverged parameters, so early
   supervision was synthesized from a geometry that did not yet exist.
3. **Degeneracy test is a required operand.** Before Phase B is interpreted,
   verify that the boundary loss has non-zero gradient with respect to the
   scale parameters, and that scaling the fitted extent by a constant changes
   the loss. Any objective failing this test is void — this is the
   generalization of the M77 finding into a standing contract.

**Operands.** Boundary-parameter displacement from initialization; held-out
synthetic family acceptance; real-OOD recall; gradient-norm trace from the
degeneracy test.

---

## 11. M84 — The out-group exposure ladder (primary empirical contribution)

**Hypothesis H84.** Open-set competence is governed principally by out-group
exposure, and the transition from zero exposure is a discontinuity rather than
a slope. Neither in-group sample count nor in-group class count substitutes
for it.

This milestone tests all three "more data" hypotheses that v9--v12 conflated:

| Variable                       | v12 status                          | v13 treatment                                                                |
| ------------------------------ | ----------------------------------- | ---------------------------------------------------------------------------- |
| (a) in-group samples per class | tested on CIFAR only; flat          | swept on DomainNet, where M78 suggests it binds                              |
| (b) known-class count          | tested; **worsened** 8 -> 32 -> 128 | retested with the M83 objective, since (b) was measured under a vacuous loss |
| (c) out-group exposure         | **never tested; N = 0**             | the ladder below                                                             |

**Ladder.** Number of real outlier training samples `N_out` in
{0, 10, 100, 1000, 10000}, drawn from the exposure pool only, crossed with
outlier diversity (number of distinct exposure classes) in {1, 10, 100}.

**Operands.** Real-OOD recall at matched known coverage on SVHN and on
held-out DomainNet classes; known accuracy degradation as a function of
`N_out`; recall as a function of exposure diversity at fixed `N_out`.

**Why this is the contribution.** The literature establishes that outlier
exposure works. It does not report **how much is needed**, how it trades
against diversity, or where the knee is. A clean exposure ladder on a
nameable-basis model, with the zero-exposure control that eight prior programs
inadvertently ran, is publishable independent of whether GEODE wins any
comparison.

---

## 12. M85 — Confirmation, transfer, and frontier assembly

Seeds 11, 23, 37. DomainNet primary at sample counts satisfying the Section 3
minimum-sample contract. CIFAR-100 transfer. Legacy CIFAR-10 cell reported for
continuity.

Assemble the deliverable frontier:

> **Amendment R7 (29 July 2026, issued with R4).** The filled cells below are
> **v12 CIFAR-10 figures at 8-way I5** and are retained only as historical
> reference. They are **not** v13 bars and may not be compared against any v13
> number. Per R4 every control is re-measured on the v13 corpus at both I5-8
> and I5-128; this table is rebuilt from those measurements at M85, and the
> accuracy column is likewise a CIFAR-10 figure against a 128-class v13 raw
> probe bar of 61.304%.

| Head            | Basis          | Accuracy | I5      | Real-OOD recall | Size    |
| --------------- | -------------- | -------- | ------- | --------------- | ------- |
| RBF             | frozen dense   | 96.917%  | 22.772% | —               | —       |
| kNN             | frozen dense   | ~96.7%   | 25.246% | —               | 6.02 MB |
| MLP + SHAP      | frozen dense   | _M79_    | _M79_   | _M84_           | _M79_   |
| MLP + IG        | frozen dense   | _M79_    | _M79_   | _M84_           | _M79_   |
| GEODE v12 field | learned dense  | 96.083%  | 17.737% | 53.5--76.3%     | _M85_   |
| sparse linear   | sparse indexed | _M81_    | _M81_   | _M84_           | _M81_   |
| decision list   | sparse indexed | _M81_    | _M81_   | _M84_           | _M81_   |
| metric field    | sparse indexed | _M81_    | _M81_   | _M84_           | _M81_   |

Four rows are already populated from existing artifacts. The empty cells are
the program.

---

## 13. M86 — Finalization and record correction

Artifact-only verifier `experiments.tier4.verify_v13_final`, byte-identical
replay, ledger finalization, and the M77/M78 amendments to
`analysis/V12_FINAL_CLAIM_LEDGER.md`. The v12 ledger is amended, never
rewritten; amendments are appended with their own dates and evidence hashes.

---

## 14. Dependency graph

```text
M77 probe degeneracy forensics ----+
M78 sample-adequacy forensics -----+--> v12 ledger amendments
                                   |
                                   v
                          M79 acceptance reframe
                          (blocks all architecture)
                                   |
                +------------------+------------------+
                v                                     v
      M80 sparse dictionary                  M83 absolute-scale
                |                             boundary supervision
                v                                     |
      M81 sparse head + I5                            |
        DECISIVE MILESTONE                            |
       /        |         \                           |
  I5<=25%   25-40%      >=40%                         |
  Outcome H  frontier   dominance                     |
                \         /                           |
                 v       v                            v
              M82 atom naming  <----------------  M84 exposure ladder
                        \                             /
                         +------------+--------------+
                                      v
                        M85 confirmation + frontier
                                      v
                              M86 finalization
```

---

## 15. Kill switches

- M77 confirming degeneracy **requires** the v12 ledger amendment before any
  architecture milestone opens; skipping the amendment is Outcome F.
- Any objective failing the M83 degeneracy test is void and its results may not
  be recorded as evidence. This applies retroactively to all v13 milestones.
- Any negative recorded at samples-per-fitted-dimension below 10 is void
  (Section 3 minimum-sample contract).
- M80 failing its 3.0-point fidelity gate after one sweep closes the
  breakthrough arm.
- M81 I5 at or below the kNN control closes the program at Outcome H.
- Any appearance of an exposure-pool class or domain in an evaluation set is
  Outcome F.
- M84 showing real-OOD recall flat across the full exposure ladder falsifies
  H84 and closes the open-set claim permanently, for this and any successor
  program.
- Naming instability at M82 forbids any "human-interpretable" phrasing in the
  ledger.
- Partition leakage, final-label access, or replay mismatch is Outcome F.

---

## 16. Outcomes

> **Amendment R10 (29 July 2026, issued by M82 on its own kill switch).** M82
> returned `names_unstable`, firing the Section 15 switch that forbids
> "human-interpretable" phrasing. "Sparse **named**/**nameable** basis" is
> withdrawn program-wide in favour of "sparse **indexed** basis", and
> **Outcome A is thereby unreachable** for v13, being defined over a named
> basis. Outcomes C and D are unaffected. The restriction is on phrasing, not
> on any measurement: explanations still cite 5.7 atoms inside M79's 10-atom
> budget, and component identity is still worth +51.9 I5 points.

- **A:** ~~sparse named basis~~ **unreachable under R10** — attains L1 within
  tolerance, dominates all controls on I5, and passes L2 with bounded exposure.
- **B:** dominates on I5 and passes L2, but fails L1 tolerance — a genuine
  frontier point, and the most likely positive.
- **C:** dominates on I5 but open-set competence requires large exposure; the
  contribution is the frontier plus the exposure ladder.
- **D:** partial I5 gain (25--40%); frontier reported, dominance not claimed.
- **E:** mechanism fails to transfer at adequate sample size.
- **F:** protocol integrity failure.
- **G:** M77/M78 show the v12 negatives were artifacts, and the program's
  contribution is the corrected record plus a reopened v12 question.
- **H (new):** the basis hypothesis is refuted at M81. Explanation usefulness
  is not basis-limited, the accuracy--interpretability frontier for vision
  foundation features is flat, and the program closes with a strong negative
  that supersedes eight weakly-scoped ones.

Outcomes G and H are both _publishable results_, not failures. This is
deliberate: the v12 gate structure could only produce E, and did.

---

## 17. Interpretation

| Result                                             | Interpretation                                                                                                                                                                    |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M77 confirms zero own-class probe gradient         | v12's registered novelty was ill-posed; its probe negative carries no information                                                                                                 |
| M78 shows 60 -> 600 improvement                    | The L5 transfer negative was an under-powered measurement, not a finding                                                                                                          |
| M78 flat across the grid                           | The transfer negative stands and is now properly supported                                                                                                                        |
| M80 fidelity gate passes                           | Frozen foundation features admit a sparse indexed basis at low accuracy cost (R10: "nameable" withdrawn)                                                                          |
| M81 I5 >= 40%                                      | **Explanation usefulness is basis-limited, not head-limited.** Eight programs optimized the wrong object                                                                          |
| M81 I5 <= 25%                                      | Usefulness is not basis-limited either; the frontier is flat and the program should end                                                                                           |
| M82 naming stable                                  | ~~The explanation is human-facing, not merely sparse~~ **struck by R10 — M82 returned `names_unstable`; the complement holds and the explanation is sparse but not human-facing** |
| M83 boundary moves from initialization             | Absolute units were the missing ingredient; v12's boundary was never trained                                                                                                      |
| M84 knee at small `N_out`                          | Open-set competence is cheap once exposure is non-zero; the zero-exposure regime was the entire problem                                                                           |
| M84 flat across the ladder                         | Rejection is not exposure-limited; close the open-set claim for good                                                                                                              |
| Sparse head beats MLP+SHAP on I5 at lower accuracy | The program's founding goal is achieved as a frontier result, not a dominance result                                                                                              |

The program tests whether explanation usefulness is a property of the basis
rather than the head, and whether open-set competence is a property of
supervision rather than geometry. It does not assume the v12 negatives were
correctly scoped, and it is structured so that refuting its own central
hypothesis is a reportable outcome rather than a ninth termination.
