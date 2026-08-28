# Claim ledger — v13

**Status:** opened by M79, 29 July 2026. Companion to
`analysis/ACCEPTANCE_CRITERIA_v13.md`.

**Immutable parents:** v6.1, v7, v8, v9, v10, v11, v12. Their ledgers are
amended, never rewritten.

**Rule of this ledger.** A finding that is not registered here before it is
measured may not be reported as a result. A finding that is registered and then
fails is a deliverable, not an embarrassment.

---

## 1. Registered hypotheses

| ID      | Hypothesis                                                                                                                                                                 | Milestone | Status                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **H77** | The v12 probe objective was gradient-dead and never trained the geometry                                                                                                   | M77       | **Confirmed**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **H78** | The v12 DomainNet transfer failure was a basis-identifiability artifact                                                                                                    | M78       | **Confirmed**, with finding 5 later withdrawn by N1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **H80** | Frozen DINOv2 embeddings admit an overcomplete sparse decomposition whose atoms are substantially monosemantic, at reconstruction fidelity sufficient to preserve accuracy | M80       | **Gate passed; clause split.** Fidelity clause supported at m=8192/k=32 (0.51 pt deficit, +3.77 pt over the random-dictionary null). Monosemanticity clause **unestablished** — measured only by a biased comparative operand. Restricted by N80.2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **H81** | Explanation usefulness is limited by basis nameability, not by head structure                                                                                              | M81       | **Gate not passed — `task_width_artifact`.** Supported at 8-way but **below the bar once read per arm**: the 40.22% headline was a per-seed best-arm maximum (N82.7); the arms' own means are 38.66% (budget_256) and 38.21% (budget_512), both under the 40.00% bar, against 35.14% best dense. **No admissible arm at 128-way** at any tolerance. Dominance blocked; frontier claim only. Restricted by N81.7, N81.8, N82.7                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **H82** | M80 atoms admit stable natural-language names, and naming raises I5                                                                                                        | M82       | **Refuted — `names_unstable`, on R9's own primary operand.** The gating instrument control **passes** (0.8954 naming accuracy on class-pure atoms vs a 0.0019 shuffled-name null), so this is not an instrument failure. Names are **less** self-consistent than their matched-size null: 0.8481 against 0.9976, margin **−0.1495** on all three seeds. Naming raises I5 by **−0.71 pt**; the 51.9-pt gain is **identity revelation**, which R8 was written to separate. N81.2's nameability burden is discharged **UNMET**                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **H83** | Boundary supervision in absolute units, applied after in-group geometry stabilises, produces a boundary that moves during training                                         | M83       | **Refuted at M83.2 (`not_separable_from_null`); M83.1 void, not negative (N83.8).** M83.1's probes lay inside the known cloud, so its figures are not operands and are not claimed. M83.2 re-runs with the ladder corrected — farthest probe 90.49 against a known 10th percentile of 24.37, held-out probe rejection 0.39–0.45 rather than identically zero — and the outcome holds: the absolute arm rejects **0.0000** of the out-of-set against **0.1226** untrained and **0.1205** null. The registered operand is met and worthless. Decisively, the trained boundary is **worse at rejecting probes than not training at all** (0.3926 vs 0.4326), so the auxiliary term is harmful rather than inert. Restricted by N83.1–N83.8                                                                                                                                                                                                                                |
| **H84** | Open-set competence is governed principally by out-group exposure, and the transition from zero exposure is a discontinuity rather than a slope                            | M84       | **Refuted (`ladder_flat`, sealed).** All premises pass first: exposure activity 0.90 against N84.6's 0.5 floor, the zero rung reproduces N84.4's pre-registered **0.11875 exactly on all three seeds**, known-class control 0.9107 against 0.85. Against that baseline the exposure arm scores **0.00000–0.00012** across all eight feasible rungs from `N_out`=10 to 10,000, and **N84.3's moment-matched null beats it at every rung** (0.0030–0.0777). No trend in count or diversity. **N84.5 confirmed as mechanism, refuted as benefit**: tangent anisotropy scales with exposure (0.4344 → 1.5201) while rejection stays at zero. Mechanism: `owner_agreement` ≤ 0.0003 across all 48 arms — every negative is ejected from its owner and re-absorbed by another of the 128 ellipsoids — and `known_false_rejection` reaches exactly 0.0 while per-class coverage holds, so the union covers the space. Restricted by N84.1–N84.7; N84.2 amended before the run |

---

## 2. Findings carried into v13

| ID        | Finding                                                                                                                                  | Source  | Bearing on v13                                                                                                   |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------- |
| **F-M77** | Probe gradient norms ≤ `6.78e-17`; probe-loss decrease 101.5% explained by a falling detached target                                     | M77     | Any objective must pass a degeneracy test before its results are interpreted. Standing contract, binding on M83. |
| **F-M78** | Sample adequacy is the hidden variable behind the v12 accuracy deficit; identifiability 0.193 at rank 32                                 | M78     | Floor of 10 samples per fitted dimension; rank capped at 53 on the current corpus.                               |
| **F-N1a** | Open-set detection is near chance for every method including free-composability controls: NCM 0.5388, kNN 0.5824, geometric 0.5898 AUROC | N1      | **Forces the L2 restatement.** See acceptance criteria Section 5.                                                |
| **F-N1b** | The v10–v12 AUROCs of 0.902–0.972 are corpus artifacts; at their own 8-known/2-unknown width this corpus yields 0.6374                   | N1      | Every open-set number carries its corpus and class count, or is inadmissible.                                    |
| **F-N1c** | At adequate sampling the geometric head reaches 23.86% recall at matched 10% FA against a 20.42% logistic bar and 19.86% kNN control     | N1      | The open-set deficit closes with sampling. M78 finding 5 withdrawn as amendment R3.                              |
| **F-I4**  | The v13 corpus is 61% quickdraw; stratified on class, not domain                                                                         | I4 / N1 | Binding on any milestone depending on semantic richness.                                                         |

---

## 3. Registered novelty position

Not the sparse dictionary, not the sparse head, not outlier exposure. Each of
those exists in the literature and this program does not claim them.

**The defended conjunction is:** a nameable sparse basis evaluated under a
registered simulatability protocol against geometric, kernel, **and post-hoc
neural** controls; the measured decorrelation of exactness from usefulness; and
a quantified out-group exposure ladder.

**N1 strengthens the second element and narrows the third.** The decorrelation
of exactness from usefulness now has a second instance: v12 achieved I2 exactness
at `1.14e-13` with I5 at 17.737%, and N1 shows the open-set operand was
near-chance for every method while being reported as 0.90–0.97. Both are cases of
a validated-looking number that did not measure the thing it was gating.

**Registered closure condition.** If M79's prior-art audit finds the conjunction
displaced, the program narrows or closes here.

---

## 4. Prior-art audit

Registered obligation from plan Section 6. Audited for **displacement**, not for
citation courtesy. No claim of displacement is made against any of these.

| Work                                                                              | Relation                                          | Does it displace the conjunction?                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --------------------------------------------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sparse autoencoders for feature decomposition (Cunningham et al.; Bricken et al.) | Direct method overlap with M80                    | **No.** They establish that sparse decomposition yields interpretable features. Neither evaluates under a registered forward-simulation protocol against kernel and post-hoc-neural controls. M80 claims no novelty in the method itself.                                                                                                                                                                                                                                                |
| Label-free CBM (Oikarinen et al., ICLR 2023)                                      | Concept bottleneck without concept labels         | **No**, and it is the closest prior art to M80–M82 jointly. It reports accuracy cost, not simulatability against controls.                                                                                                                                                                                                                                                                                                                                                               |
| LaBo                                                                              | Language-model-generated bottlenecks              | **No.** Same gap: no simulatability protocol.                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| SpLiCE (Bhalla et al., NeurIPS 2024)                                              | Sparse decomposition of CLIP into named concepts  | **No**, and it is the closest prior art to M82. Overlaps the naming channel directly; M82 must cite it as method, not as contribution. **Amended by R9:** SpLiCE decomposes CLIP, where text and image share a space; v13 decomposes **DINOv2**, where they do not. v13 therefore cannot run SpLiCE's method at all, and must not be described as CLIP-space decomposition. The overlap is narrower than this row assumed, and the difference is a limitation of v13, not a contribution |
| Rudin (Nature MI 2019)                                                            | The accuracy/interpretability tradeoff is a myth  | **Partially constrains.** The argument is explicitly restricted to structured features; this program operates on learned dense embeddings, where it does not straightforwardly apply. Section 8 of the acceptance criteria exists to keep that distinction honest.                                                                                                                                                                                                                       |
| Rashomon-set results (Semenova & Rudin)                                           | Simple models often exist that match complex ones | **Constrains the framing.** If a Rashomon set is large, a sparse head matching a dense one is expected rather than surprising. M81 must not claim surprise at parity.                                                                                                                                                                                                                                                                                                                    |
| Outlier Exposure (Hendrycks et al., ICLR 2019)                                    | Out-group exposure improves OOD detection         | **No.** It establishes that exposure works; M84 measures **how much** is needed and how it trades against diversity, which it does not report.                                                                                                                                                                                                                                                                                                                                           |
| VOS / NPOS                                                                        | Synthesised outliers                              | **No.** M83 uses absolute-scale displacements; the correction registered there is about the v12 objective, not a claim over these.                                                                                                                                                                                                                                                                                                                                                       |
| Fang et al. (NeurIPS 2022)                                                        | OOD detection is not learnable in general         | **Bounds the program, and N1 is consistent with it.** It supplies theoretical context for why N1 found near-chance detection for every method. M84 must state its claim within these bounds.                                                                                                                                                                                                                                                                                             |
| Bendale & Boult (CVPR 2015)                                                       | Open World Recognition                            | No overlap with the retained line; recorded from the abandoned branch.                                                                                                                                                                                                                                                                                                                                                                                                                   |
| NINCO (Bitterwolf et al., ICML 2023)                                              | Contamination-controlled OOD benchmark            | Not prior art but the **required instrument** if open-set is ever revisited. See acceptance criteria Section 9.5.                                                                                                                                                                                                                                                                                                                                                                        |

**Audit outcome: the conjunction is not displaced.** It is materially narrowed:
M80 and M82 claim no methodological novelty, and M84's contribution is the
quantified ladder rather than the existence of the effect.

---

## 5. Claim restrictions carried forward

Not established, and not to be asserted anywhere in v13 without new evidence:

- Generalised open-space rejection.
- Human simulatability. I5 is an automated proxy and is **not** canonical human
  simulatability; no v13 result may be described as human-validated.
- Semantic meaning of learned feature coordinates, until M82 passes its gate.
- Closed-form minimum Euclidean counterfactual reach (I4, structurally
  unavailable for multiclass anisotropic quadratics).
- Independent final-label confirmation, until M86.
- `p(caught | explanation)`, never measured; blocks every accuracy-for-
  interpretability trade claim.
- **New:** any open-set claim stated without its corpus and class count.
- **New:** atom monosemanticity or purity. The M80 entropy operand is estimated
  from roughly 16 activations per atom against a 128-way label space, which caps
  observable entropy near 4 bits against a 7-bit uniform bound. It is admissible
  as a within-grid comparison against its own shuffled-label control at matched
  counts, and **never** as an absolute purity figure.
- **New:** any accuracy-preservation claim for sparse codes at a `k` where the
  random-dictionary control is not beaten. Per N80.2, k=64 codes carry the
  feature vector as a sparse random projection and their probe accuracy is not
  attributable to the learned dictionary.
- **New:** any claim that sparse-atom explanations are more forward-simulable
  than dense ones, stated without its task width. Per M81 the claim holds at
  8-way and fails at 128-way on the same corpus, same basis and same protocol.
  The 8-way number (40.22%) may not be quoted alone, and may not be compared to
  the v12 record (GEODE 17.737, RBF 22.772, kNN 25.246) as though the widths
  were commensurable.
- **New:** any I5 comparison against a model that is not accuracy-comparable to
  its control, or that emits less than half the label space. Per N81.7 and
  N81.8 both conditions were violated by the arm that first carried the M81
  gate, and both are clauses of H81 rather than added constraints.
- **New:** any claim that the sparse head meets the M79 deployment budget at
  full corpus width. At 128 classes every arm within the 10-atom budget is 22 to
  44 balanced-accuracy points below the kNN control.

---

## 6. Registration notes

Gaps between what a hypothesis asserts and what its registered operands measure,
recorded when found rather than resolved by amending the gate afterwards.

### N80.1 — H80's monosemanticity clause was unmeasured (29 July 2026)

**Issued by:** M80, before execution. H80 asserts atoms are "substantially
monosemantic," but none of the four registered operands — reconstruction R²,
mean active atoms, dead fraction, probe accuracy — measure semantic purity. Mean
atom label entropy was added as a **reported, non-gating** operand against a
shuffled-label control. The gate was left exactly as registered.

### N80.2 — the M80 gate lacks a control clause (29 July 2026)

**Issued by:** M80, after execution. **Evidence:**
`logs/results/v13/m80_sparse_dictionary/evidence.json`.

The registered gate selects the best-accuracy cell under a sparsity ceiling and
does not require that cell to beat its own null. Its selected cell, m=8192/k=64,
scores 61.206% against a random unit-norm dictionary at **61.230%** — the
untrained control is 0.02 points better. The gate as written would have
certified a random projection.

**The gate is not amended and its pass stands as registered.** The restriction
recorded instead: a cell is admissible as evidence for H80 only if its probe
accuracy exceeds the random-dictionary control on the identical split. M81
carries **m=8192, k=32**.

### N81.1 — the SHAP arm is expected gradients, not KernelSHAP (29 July 2026)

**Issued by:** M81, before execution. `shap` is absent from the frozen replay
`.venv` and installing it would break the sealed M73/M77 hashes; KernelSHAP is
intractable at 128 classes regardless. The arm is a GradientExplainer-style
expected-gradients estimator and is named as such everywhere. It lands within
0.4 points of integrated gradients at both widths, so the substitution is not
load-bearing.

### N81.2 — the protocol cannot test nameability (29 July 2026)

**Issued by:** M81, before execution. Component identity is withheld from the
probe, as in v12. M81 therefore tests whether **sparsity** aids simulatability,
not whether atoms are **nameable**. H81's nameability clause is untouched by
this milestone and passes to M82 in full. Identity-included I5 is
ceiling-artifact-prone for linear heads and was not run.

### N81.3 — unstandardised explanations measured optimiser failure (29 July 2026)

**Issued by:** M81, during construction. Explanation columns mix per-atom
contributions with sums, whose scales differ by orders of magnitude. Unscaled,
lbfgs did not converge at `max_iter=2000`, so I5 partly measured optimiser
failure rather than explanation content. Explanations are now standardised on
the probe-training rows only, identically for every arm and every control.

### N81.4 — L1 in the loss never produced a sparse head (29 July 2026)

**Issued by:** M81, during construction. Adding `l1 * |w|` to the objective and
letting Adam differentiate it shrinks every coefficient uniformly without
reaching zero. Measured: accuracy fell to 14.89% while atoms cited per decision
stayed at 30.2 — the penalty destroyed the model without shortening the
explanation. Replaced by a proximal soft-threshold applied after each step.

### N81.5 — magnitude-ranked atom budgets select atoms that never fire (29 July 2026)

**Issued by:** M81, during construction. Ranking a per-class budget by `|w|`
selects rare atoms, which carry large weights precisely because they seldom
activate. Measured: 0.6 atoms cited per decision at 13.7% accuracy. The support
is now ranked by expected contribution mass, `|w| * mean|activation|`.

### N81.6 — explanation width is held constant across arms (29 July 2026)

**Issued by:** M81, before execution. Every arm — atom heads and dense controls
alike — has its explanation reduced to the identical withheld form of eight
sorted contribution magnitudes plus sum, max and mean. I5 differences therefore
cannot arise from one arm being handed a wider explanation vector than another.

### N81.7 — the gate omitted H81's own comparability clause (29 July 2026)

**Issued by:** M81, after execution. **Evidence:**
`logs/results/v13/m81_sparse_head/evidence.json`.

H81 claims an atom head is more simulable than a dense head **of comparable
accuracy**. The selection rule did not enforce that clause, and the first gate
output returned `confirmed` on the strength of `decision_list`, which has
**15.74%** balanced accuracy against kNN's **66.13%** at 128 classes.

Unlike N80.2 this is a defect in the **implementation of the registered
hypothesis**, not an absence in the registered rule, so enforcing comparability
is a correction rather than an amendment. The floor is 5 points below the best
dense control. Because the tolerance was fixed with the data in view, the
verdict is recomputed at 2.5, 5, 10, 15 and unconstrained tolerances; the
128-way failure holds at all of them, and the 8-way result holds from 5 points
upward but not at 2.5. Corrected verdict: **`task_width_artifact`**.

### N81.8 — I5 is inflated by prediction collapse (29 July 2026)

**Issued by:** M81, after execution. `degenerate_single_prediction` caught only
the limiting case of a head emitting exactly one class. The arm that first
carried the M81 gate emitted **36 of 128** classes, giving it a majority
baseline of **2.77%** where every other arm's was 0.78%: the probe was reading a
collapsed prediction marginal rather than an explanation. Its margin over that
baseline (+3.60) was below arms it appeared to beat. An arm must now emit at
least half the label space to be admissible, and the majority baseline is
reported per arm.

### N82.1 — the naming instrument is weakest where the corpus is densest (29 July 2026)

**Issued by:** M82, **before any atom is named**, from a zero-shot instrument
check on a 512-image stratified sample (4 per class, seed 20260729).

The check exists to validate the hand-written CLIP preprocessing, and it does:
zero-shot accuracy over the 128 in-corpus terms is **49.41%** against 0.78%
chance. But it is not uniform across rendering domains, and the non-uniformity
runs the wrong way:

| domain    | share of corpus | zero-shot accuracy |
| --------- | --------------- | ------------------ |
| clipart   | 9,939           | **91.23%**         |
| real      | 3,607           | 84.38%             |
| painting  | 7,195           | 76.36%             |
| infograph | 8,029           | 71.67%             |
| quickdraw | **44,800**      | **28.99%**         |
| sketch    | 158             | (1 sample)         |

**Quickdraw is 61% of the corpus and is the one domain CLIP cannot read.** The
naming channel is therefore weakest on the majority of the images that define
the atoms it must name. This is a compounding of the corpus defect N1 recorded,
not a new one, but it is load-bearing for M82 in a way it was not for M80 or
M81: those milestones read the DINOv2 features, which are equally happy on
doodles, whereas naming reads CLIP.

**Registered consequence.** Per-domain naming quality is reported alongside
every aggregate naming number, and no aggregate may be quoted alone. If naming
succeeds on clipart and fails on quickdraw, the finding is that atoms are
nameable _where the naming instrument works_, which is a much weaker claim than
H82 makes and must be stated as the weaker one.

### N82.2 — the naming channel has a high false-naming floor (29 July 2026)

**Issued by:** M82, before any atom is named, from the same check.

The vocabulary's 217 absent DomainNet names are the far-field control. Offered
all 345 terms, the channel picks a name **absent from the corpus** for **34.57%**
of images whose true class is present in it. The instrument therefore has a
large false-naming floor before any atom enters the picture.

**Registered consequence.** `false_naming_rate` is reported for every naming
result, and an atom's name is admissible evidence only against this floor, not
against zero. A naming agreement rate that does not exceed what this leakage
alone would produce is not evidence of naming.

### N82.3 — the registered style vocabulary is weak, and is not amended (29 July 2026)

**Issued by:** M82, before any atom is named, from the same check.

The six style terms were registered to make style-carrying atoms nameable, the
corpus being 61% quickdraw. Measured, they recover the rendering domain for
**26.17%** of images against 16.67% chance — better than chance but poor — and
for quickdraw images specifically they recover it **0.00%** of the time. The
phrase registered for quickdraw loses to `a pencil sketch` on quickdraw's own
images.

**The vocabulary is not amended.** It was sealed and committed before the
naming channel existed, at hash `e587143…83ff9`, and rewriting a registered
vocabulary after measuring that it underperforms is the exact move the
registration exists to prevent — it differs from the N81.7 correction, which
enforced a clause H81 already stated, and resembles N80.2, where a gate that
would have certified a random projection was left as written. The weakness is
recorded here instead, and any atom named by a style term is reported with this
caveat attached.

### N82.4 — R8's arm (b) is sharpened to a matched-size random grouping (29 July 2026)

**Issued by:** M82, before execution.

R8 registered the naming claim as **named minus revealed-unnamed**, with
revealed-unnamed described as "atoms carried as stable arbitrary indices."
Implemented literally that is a weak control, for a reason R8 did not
anticipate: naming does not merely reveal an atom's identity, it **groups** many
atoms under one word. A per-atom arm therefore differs from a named arm in
**width and group structure** as well as in meaning, and any gap between them
would be partly the compression rather than the semantics.

Registered: arm (b) is additionally run as a **matched-size random grouping** —
the same number of groups naming produces, with the same group-size
distribution, assigned arbitrarily. The naming claim rests on **named minus
matched-random**, where width and group structure are identical and only the
semantics differ. This is R5's null-control contract applied to the grouping
itself, and it makes the comparison stricter rather than easier: the literal
per-atom arm is retained and reported so that R8's registered wording is still
answered, but it is not what the claim rests on.

### N82.5 — the false-naming rate conflated style names with absent objects (29 July 2026)

**Issued by:** M82, on reading its own first output.

The first M82 run reported a false-naming rate of **85.86 %**, against N82.2's
single-image floor of 34.57 %. The figure is arithmetically correct and
substantively misleading. Style terms occupy vocabulary indices 345–350, above
the 128 in-corpus object terms, so `false_naming_rate` counted every
style-named atom as an atom misnamed after something the corpus does not
contain. **6,737 of 8,192 atoms were named by a style term**, which accounts
for essentially the whole figure.

An atom named "a rough black and white doodle" has not been misnamed after an
absent object; it has been named after a rendering style the corpus is 61 %
composed of. N82.3 anticipated exactly this outcome and required it be
reported, so a rate that folds it into the far-field control destroys the
control's meaning in both directions.

Registered: `far_field_rate` now returns the object-only rate alongside the raw
rate, and both are reported. The raw rate is retained rather than replaced so
the first run's figure stays reconstructible. This does not touch the gate,
which was already decided against H82 by the stability operand.

### N82.6 — `dead_atom_fraction` is wrong in M80, M81 and M82's first run (29 July 2026)

**Issued by:** M82, on reading its own evidence.

`SparseCodes.active_atom_count()` counts **atoms per row**, not rows per atom.
The M80 and M81 runners both compute

```
dead_atom_fraction = 1.0 - active_atom_count() / dictionary_size
```

which divides a per-row vector by 8,192 and yields an array of ~0.996 values.
It is not a dead-atom fraction, it is not a scalar, and it was carried into
both milestones' sealed evidence unnoticed because nothing consumed it.

The correct quantity uses `atom_usage()`, which counts rows per atom. M82
computes it that way and reports **0.0000** — every one of the 8,192 atoms
fires on at least one fit row, which the naming channel independently
corroborates by finding exemplars for all 8,192.

M80's and M81's sealed evidence is **not** retroactively edited. The field was
reported but never gated on, in either milestone, so no conclusion drawn from
them depends on it; correcting a sealed artifact after the fact would be a
worse defect than the one being corrected. The error is recorded here instead,
and any reader of those two evidence files should disregard that field.

### N82.7 — M81's headline 40.22 % was misattributed to a single arm (29 July 2026)

**Issued by:** M82, which reproduced M81's carried arm exactly and found the
numbers did not match the write-up.

M82 refits M81's `sparse_linear_budget_256` head and reproduces its sealed
evidence to four decimals on every seed — balanced accuracy 0.8477 / 0.8516 /
0.8457, I5 0.3843 / 0.3972 / 0.3783, shuffled null 0.1616 / 0.0748 / 0.1219,
5.6 / 5.84 / 5.72 cited atoms. That is a clean replication, and it makes the
discrepancy unambiguous: **the arm's I5 mean is 38.66 %, not 40.22 %.**

M81's evidence records where 40.22 % comes from. It is
`best_atom_arm_per_seed`, which selected `sparse_linear_budget_256` for seed
11 and `sparse_linear_budget_512` for seeds 23 and 37, giving 0.3843 / 0.4236 /
0.3987. The gate's own field for the citation count that goes with it is
`unconstrained_cited_atoms_mean` = **7.51**, not 5.7.

**The gate is not wrong; the sentence describing it is.** Asking whether _any_
atom arm beats the dense controls, and selecting the best atom arm per seed, is
defensible so long as the dense side is selected the same way — and it is
(`best_dense_control_per_seed` varies across `mlp_expected_gradients` and
`mlp_integrated_gradients`). What is not defensible is attributing the
resulting maximum to one named arm and quoting that arm's citation count beside
it, which is what `analysis/CLAIM_LEDGER_v13.md` and
`analysis/MILESTONE_RESULTS.md` both did.

This sharpens a weakness already recorded rather than creating a new one. M81
was noted as clearing the 40.00 % bar by **0.22 points against a 3.93-point
seed spread**; the honest single-arm means are **38.66 %** (budget_256) and
**38.21 %** (budget_512), and **both sit below the bar**. The 0.22-point margin
existed only by per-seed arm selection. M81's overall verdict was
`task_width_artifact` and nothing was claimed on the 8-way pass, so no
conclusion moves — but the prose is corrected in both documents, and the
per-arm means are stated alongside the selected-maximum wherever it appears.

---

### N83.1 — "a boundary that moves" is satisfiable by noise (29 July 2026)

**Issued by:** M83, **before any M83 code exists and while no boundary number
has been computed.**

H83 asks whether absolute-unit supervision "produces a boundary that moves
during training." Movement is the registered operand — boundary-parameter
displacement from initialization. **Any free parameter under any non-constant
loss will move.** As registered, H83 is confirmable by a learning rate.

This is the same defect class as M77's: v12's probe term was vacuous because it
was algebraically independent of the parameters it claimed to supervise, and
nobody checked. The mirror-image risk here is a term that is trivially
_dependent_ and confirms itself. Registered in addition, before execution:

1. **A shuffled-negative null**, sharing the geometry, the probe count, the
   distances, the optimizer, the step budget and the split, but assigning each
   synthetic negative to a **random class** rather than its own. Displacement
   must exceed this null. This is R5 applied to M83.
2. **Movement must be movement somewhere.** Displacement alone is reported but
   is **not** sufficient for H83. The boundary must also improve real-OOD
   recall at matched known coverage over the untrained boundary at its
   initialization.
3. **The v12-form arm is retained as the negative end of the instrument**:
   identical curriculum with probes at `4.0 × fitted extent`, which M77 proved
   degenerate. If it fails the degeneracy test and M83's arm passes, the
   instrument discriminates; if both pass, the test is not measuring what M77
   measured and M83 is void.

### N83.2 — real-OOD recall confounds novel class with novel domain (29 July 2026)

**Issued by:** M83, before execution.

The v13 corpus is **61 % quickdraw** by construction. The 217 DomainNet classes
absent from it are not distributed that way, so an out-of-set sample drawn by
availability would differ from the corpus in **domain** as well as in class. A
boundary that rejected it could be detecting rendering style, not novelty —
and M82 has already demonstrated that a channel over this corpus will happily
read style when style is the easier signal (82 % of atoms took a style name).

The out-of-set artifact is therefore **stratified to match the corpus domain
distribution**, and per-domain recall is reported beside every aggregate, on
the N82.1 rule that no aggregate is quoted alone.

### N83.3 — rejection recall is trivially satisfiable by rejecting everything (29 July 2026)

**Issued by:** M83, before execution.

Recall on out-of-set data is maximized by a boundary that accepts nothing.
Two constraints are registered against this:

1. **Matched known coverage.** Thresholds are set per class so that acceptance
   on held-in evaluation rows is a fixed 90 %, and recall is read only at that
   operating point. No recall figure is quoted at an unmatched threshold.
2. **A known-class novel-image control that must be ACCEPTED.** The sealed
   artifact carries held-out _rows_ of the 128 **known** classes — images the
   model has never seen, of classes it has. A detector reading novelty must
   accept these and reject the 217 unseen classes. One reading generalization,
   one reading familiarity; a boundary that treats them alike has learned
   "unfamiliar image," not "unknown class."

### N83.4 — Phase A "to convergence" is strengthened, not merely satisfied (29 July 2026)

**Issued by:** M83, before execution.

The registered curriculum says Phase A "fits in-group geometry to convergence
… scales are then frozen," which reads as an iterative fit halted by a
tolerance. M83 instead fits each class in **closed form** by SVD
(`fit_subspace_primitive`), which is exactly converged rather than converged to
a tolerance, and removes the optimizer, its schedule and its stopping rule from
the confound set entirely. This is stronger than what was registered and is
recorded here so the deviation is visible rather than silent. The stated
rationale for the curriculum — that v12 synthesized negatives every batch from
live, unconverged parameters, supervising a geometry that did not yet exist —
is fully served.

### N83.5 — the degeneracy contract must be read without the hinge (29 July 2026)

**Issued by:** M83, before execution, after the instrument returned a false
reading on the development corpus.

The degeneracy test asks whether the probe term can influence the boundary at
all. Reading that off the training objective is wrong, because the objective is
hinged: a probe already outside its boundary contributes exactly zero gradient.
That is correct for training and fatal for an instrument, since a **saturated
healthy** objective and a **genuinely scale-blind** one both read zero. The
absolute arm did read `0.0` under the hinge. `degeneracy_report` therefore
differentiates the **mean probe score**, which is hinge-free; the hinged
gradient and the active fraction are recorded as context and are never the
verdict.

Two further corrections were forced by the same exercise, both recorded because
each produced a confidently wrong number first:

1. **Probes must be rebuilt inside the autograd graph.** Precomputing them as
   constants freezes the extent that appears in the placement while the extent
   in the score stays live, so v12's cancellation never happens and the broken
   objective reads **healthy** (gradient 0.102 instead of 0). v12 regenerated
   probes from live parameters every batch. A negative control has to be
   reproduced mechanically, not approximately.
2. **Clamp, never add, under a square root.** An additive `1e-12` perturbed the
   exact cancellation and left a spurious `2.7e-13` gradient where the contract
   demands zero. With `torch.clamp(..., min=1e-300)` the v12 arm reads
   `5.4e-18` and a rescale spread of **exactly `0.000`**.

### N83.6 — the placement unit is one global radius, not the centroid distance (29 July 2026)

**Issued by:** M83, before execution.

The obvious absolute unit is the median inter-centroid distance, and it is the
cleaner reference length, but it is a poor ruler here: fitted boundaries sit at
roughly **4 %** of it, so a probe ladder measured in that unit lands entirely
outside the band where supervision does any work, and **both** arms read zero
effect. Probes are therefore placed at multiples of `global_scale_unit` — the
median fitted tangent scale, a **single scalar shared by every class**. Being
global, it cannot carry any individual class's extent, which is the property
that blocks the v12 cancellation; being of the right order, it puts probes where
the hinge is active. The centroid distance is retained and reported as
`absolute_unit`.

### N83.7 — the inherited evaluation partition confounded domain with novelty (29 July 2026)

**Issued by:** M83, from a smoke run, **before any evidence was sealed.**

M80 and M81 partition each class positionally: rows 0--511 fit, rows 512--575
held out. The v13 corpus is class-major and every class stores its **350
quickdraw rows first**, so that held-out set contains **zero quickdraw** while
the out-of-set artifact is **61 % quickdraw**. Under that split the rejection
comparison measures which domains the two evaluation sets happen to contain.
N83.2 was written to prevent exactly this and it held on the unseen side; it was
void on the known side, which nothing checked because the achieved mixture was
never written down.

The evidence that this was composition and not signal is direct: aggregate
distance to the global feature mean differs sharply (known 50.91, unseen 33.64),
but **within each domain the two sets are indistinguishable** — 44.15 vs 43.81,
46.09 vs 46.10, 50.42 vs 50.42, 53.76 vs 54.00. The entire gap was the mixture.

M83 therefore selects evaluation rows by **domain quota** — the same quota the
out-of-set builder used — rather than by position. Domains every class can
supply are pinned; the scarce remainder is water-filled one row at a time and
then repaired by single exchanges between classes. The result matches the two
profiles at a maximum deviation of **exactly 0.0** with no domain unmet, and
makes the fit set representative as a side effect (quickdraw **68.4 % → 60.74 %**
against a corpus share of 60.76 %). Per-class mixtures drift where a class holds
no paintings at all; the aggregate, which is what N83.2 is about, is exact.

Two conditions on this change, stated so it cannot be mistaken for a rescue:
it was found from a **smoke run and applied before any evidence was sealed**,
and it corrects **which question the measurement answers**, not an answer that
was disliked. The known-coverage control moved from 0.8250 (failing) to 0.9173
(passing) as a consequence, and H83 was refuted anyway.

### N83.7a — M80 and M81 inherit the same partition defect (29 July 2026)

**Issued by:** M83, against sealed evidence, which is **not** retouched.

The positional split is M80's and M81's. Their held-out accuracy figures were
therefore read on evaluation rows containing **no quickdraw at all**, while
their dictionaries and heads were fitted on rows that were **68 % quickdraw**.
Both milestones' evidence stands exactly as measured and is not regenerated.
What those numbers measure is narrower than their prose implies — they are
accuracy on the non-quickdraw 39 % of the corpus under a fit dominated by
quickdraw — and any later claim resting on them must say so. This does not
change M80's or M81's verdicts: M80's gate turned on a random-dictionary control
measured on the identical rows, and M81's failure was at 128-way width against
kNN measured on the identical rows, so both comparisons remain internally fair.

### N83.8 — M83.1's probes lay inside the data, and its evidence is void (29 July 2026)

**Issued by:** M83, **after** sealing M83.1 and after its result had been
written up as a refutation. The correction is recorded in full because the
milestone was already committed when the defect was found.

N83.6 chose the placement unit to be the median fitted tangent scale. Its
reasoning — that the inter-centroid unit puts probes so far out that every hinge
saturates — is correct. Its conclusion does not follow. The tangent median is
the per-direction spread inside a **rank-51 subspace of 384 dimensions**, while
the distance from a row to its own centroid is dominated by the **333 residual
dimensions that subspace does not describe**. The two quantities differ by an
order of magnitude, and nothing in the milestone compared them:

| quantity                                              | value           |
| ----------------------------------------------------- | --------------- |
| placement unit (median tangent scale)                 | 2.87            |
| probe distance from centroid, multipliers 1–3         | 2.87 – **8.61** |
| known row distance from own centroid, 10th percentile | **24.36**       |
| known row distance from own centroid, median          | 30.19           |

The **farthest** probe sits at a third of the distance of the **nearest decile**
of real data. No boundary can accept rows at 24–30 and reject points at 5–8
along the same rays. The supervision was not given a hard problem, it was given
an incoherent one, and a negative result on an incoherent problem is not a
negative result.

The signature was present in M83.1's sealed evidence and was misread as a
finding: `held_out_family_rejection` is **0.0 for all four arms**, including the
**untrained** one, which does no training at all. A figure that is identically
zero for an arm that cannot have learned anything is a property of the
instrument, not of the arms. The audit that established this also confirms the
numbers are otherwise sound — no NaN, no infinity, no non-finite radius, and the
unseen minimum-score distribution genuinely tops out at 0.99014 — so M83.1
failed for a reason of design, not of arithmetic.

**Corrections, all outcome-independent:**

1. The placement unit becomes `data_scale_unit`, the median distance from a fit
   row to its own centroid. The only requirement the unit must meet is that it
   not depend on the learnable radii, so v12's cancellation cannot reappear; a
   statistic of frozen fit data satisfies that as completely as the tangent
   median did, while landing the ladder where a negative is negative. A test
   asserts the unit is unchanged when every radius is inflated a hundredfold.
2. `probe_validity` is added and is **gating, ahead of the degeneracy
   contract**, because it is prior to it: a scale-sensitive objective supervised
   by interior points is still measuring nothing. A run whose probes do not
   reach past the known cloud returns `probe_ladder_interior` and reports no
   recall figure at all, exactly as N83.1 voids everything below a failed
   degeneracy report.
3. The placement unit is registered in the configuration rather than hardcoded,
   so the choice that voided M83.1 is a recorded decision.

The diagnostic that condemns M83.1 — farthest probe distance against the tenth
percentile of known distance — involves no radius, no arm and no verdict. That
is what distinguishes this from a rescue, and it is the same standard N83.7 was
held to. The difference, stated plainly because it matters: N83.7 was caught
from a smoke run **before** any evidence was sealed; N83.8 was caught **after** a
sealed run had been written up as a refutation. M83.1's evidence is not
retouched and its directory is not reused; M83.2 writes elsewhere.

---

### N84.1 — the exposure pool and the evaluation out-of-set are class-disjoint (29 July 2026)

Outlier exposure that trains on the classes it is scored against is not open-set
detection, it is supervised classification with the test set. The 217 unseen
DomainNet labels are therefore split **by class and before extraction**:
**128–247 (120 classes) are the exposure pool, 248–344 (97 classes) are the
evaluation out-of-set.** The split is positional and fixed by this note, so no
data-dependent choice enters it. DomainNet labels are alphabetical, which is
arbitrary with respect to semantics; it is not claimed to be a _semantically_
balanced split, and any class-similarity structure that survives the alphabet is
a confound this milestone does not control.

Consequence for comparability: M83's out-of-set was all 206 fully-stratified
classes, and M84's is a 97-class subset of them. **M83.2's figures may not be
quoted as M84's baseline.** The baseline is re-measured on M84's own evaluation
classes; see N84.4.

### N84.2 — the registered ladder is not rectangular, and the empty cells are declared now (29 July 2026)

The plan's ladder crosses `N_out` in {0, 10, 100, 1000, 10000} with exposure
diversity in {1, 10, 100}. A cell needs `N_out / diversity` rows from **each**
exposure class, and DomainNet cannot supply that at the low-diversity end. Raw
availability over the 217 unseen classes, measured from the shard label and
domain columns before any extraction:

| domain    | min     | 10th pct | median | max |
| --------- | ------- | -------- | ------ | --- |
| clipart   | 12      | 42       | 135    | 467 |
| infograph | 14      | 39       | 117    | 770 |
| painting  | 2       | 29       | 178    | 838 |
| quickdraw | **500** | 500      | 500    | 500 |
| real      | 61      | 259      | 524    | 802 |
| sketch    | 22      | 79       | 181    | 714 |

Quickdraw is exactly 500 per class for every class and takes 39 of every 64
rows under N83.2's domain quota, so it caps the pool at roughly 768 rows per
class before the 256-pixel filter and before the other domains bind sooner. The
exposure pool is built at **128 rows per class** (the quota doubled to
[18, 14, 12, 78, 6, 0]), which 204 of the 217 classes can fill on raw counts.

The feasible cells, declared before any run:

| diversity | 10  | 100 | 1000   | 10000  |
| --------- | --- | --- | ------ | ------ |
| 1         | yes | yes | **no** | **no** |
| 10        | yes | yes | yes    | **no** |
| 100       | yes | yes | yes    | yes    |

Nine cells plus the shared zero rung. The three empty cells are **infeasible in
DomainNet, not omitted**: (1000, 1) and (10000, 1) would need 1000 and 10000
images of a single class clearing the resolution filter under the corpus domain
profile, and (10000, 10) would need 1000 each from ten classes. No result may be
reported as though the ladder were complete, and the diversity comparison is
available in full only at `N_out` of 10 and 100.

#### Amendment (29 July 2026, before any M84 figure existed)

The table above is wrong in one cell and the surrounding text in one number.
Both are corrected here rather than edited in place, because a registration note
that is silently repaired is not a registration.

**(10, 100) is not a cell.** The diversity axis is defined as an equal share per
class, so `N_out = 10` spread over 100 classes asks for a tenth of a row from
each and does not exist for any dataset. This is a fourth infeasible cell, and
it differs in kind from the other three: those are infeasible _in DomainNet_ and
a larger corpus would fill them, whereas this one is infeasible _in arithmetic_.
The corrected feasible set is **eight cells plus the zero rung**, and the
diversity comparison is complete only at `N_out` of 100 and above.

| diversity | 10     | 100 | 1000   | 10000  |
| --------- | ------ | --- | ------ | ------ |
| 1         | yes    | yes | **no** | **no** |
| 10        | yes    | yes | yes    | **no** |
| 100       | **no** | yes | yes    | yes    |

**The quota is [17, 14, 13, 78, 6, 0], not [18, 14, 12, 78, 6, 0].** The note
above reached it by doubling the out-of-set quota; the builder recomputes the
largest-remainder allocation at 128 directly, which lands on 17/14/13. The
artifact was built under the builder's value, so the sealed pool is correct and
only this note was wrong.

How the cell error was caught matters for the record: not by review, but by the
runner failing on `count 10 is not divisible by diversity 100` after the sealed
run had already started. The smoke run had exercised `zero` and `n100_d10` end
to end and said nothing about the other eight. The runner now draws **every**
registered cell from the real pool with the real sampler before dispatching any
work, so an unfillable cell costs seconds instead of a run. A pre-flight that
checks one instance of a thing is not a pre-flight.

### N84.3 — the null is moment-matched, because a shuffled-owner null tests nothing here (29 July 2026)

M83's null permuted which class each probe belonged to, which works when the
negatives are constructed relative to a class. Real out-group rows have no owner
to permute. R5 still requires a null sharing structure, budget and split, so the
null for every exposure cell is **the same number of synthetic negatives drawn
from a Gaussian matched to the exposure rows' own mean and covariance**, trained
with the identical optimiser, step budget, batch size and split.

This makes the comparison say something specific. The null has the exposure
sample's first and second moments and none of its content. If exposure beats it,
what the boundary used was the structure of real out-group images; if it does
not, `N_out` is buying nothing that a covariance estimate would not.

### N84.4 — the zero rung is a known-value control, and it is re-measured (29 July 2026)

`N_out = 0` is the configuration eight prior programs ran without labelling it,
and it is this milestone's known-value control. It is **not** required to
reproduce M83.2's 0.1226, which was read on 206 classes rather than 97 (N84.1).
The baseline is measured directly: the untrained boundary at 90 % matched
coverage rejects **0.11875** of M84's own evaluation set, which after the
out-of-set artifact's own underfilled classes are dropped is **5760 rows in 90
classes**. A zero rung that disagrees with that voids the ladder, on the same
footing as N83.1's degeneracy clause and N83.8's probe-validity clause.

The zero rung is split into two arms, because "no negatives" and "no training"
are different configurations and only one of them is the control:

- **`untrained`** — the fitted ellipsoid at matched coverage, no optimisation at
  all. This is the known-value control and the 0.11875 figure above.
- **`known_only`** — the full step budget against the known term alone. This is
  literally `N_out = 0` and it isolates what the negatives contribute from what
  training contributes. Without it, any ladder effect could be the optimiser
  rather than the exposure.

### N84.5 — what M83.2 permits M84 to claim, and the mechanism it leaves open (29 July 2026)

M83.2 established that real out-of-set rows on this corpus are not radially
distinguishable from known ones: within each domain, known and unseen rows sit
at near-identical distance from the global mean. A radial boundary has no
operand there, which is why synthetic negatives placed along the fitted axes
could not work and why the trained arm ended up worse at rejecting probes than
its own initialisation.

That does **not** close M84 in advance, and the reason is worth stating before
the run rather than after. The boundary is an ellipsoid, not a sphere — 51
tangent radii and one residual radius per class. M83's probes were placed along
the in-group's _own_ fitted directions, which is where the in-group already
lives; real out-group rows need not sit there. Exposure can therefore shape the
ellipsoid anisotropically in a way a radial ladder could not, and the registered
prediction is that any improvement M84 finds will show up as **tangent-radius
anisotropy rather than a smaller mean radius**. That is reported as a
descriptive operand beside recall, whichever way the recall goes.

What M84 may **not** claim on any outcome: that exposure is unnecessary
(M83 already answered that), or that a flat ladder generalises beyond a radially
parameterised boundary on DINOv2-small features over DomainNet.

### N84.6 — training starts at matched coverage, because at the raw fit the exposure term has no gradient (29 July 2026)

Measured before M84 was written, on the sealed exposure pool against the
Phase A geometry:

| initialisation        | median min-over-class score | rejected | **hinge active** |
| --------------------- | --------------------------- | -------- | ---------------- |
| raw fitted radii      | 19.46                       | 1.0000   | **0.0000**       |
| 90 % matched coverage | 0.5448                      | 0.1313   | **0.9833**       |

At the fitted radii every exposure row already sits nineteen times outside every
boundary, so a hinge at `1 + margin` is satisfied by all of them and contributes
**exactly zero gradient from the first step**. A ladder trained there would be
flat for a purely instrumental reason and would look exactly like a real
negative result. That is the same failure mode as N83.8 — negatives the
objective cannot use — caught this time before the run rather than after it.

M84 therefore initialises `log_beta` at the 90 % matched-coverage point, where
98.33 % of exposure rows are inside some boundary and the term is live.
Displacement is reported from that point rather than from the raw fit.

This is registered rather than adopted quietly because M83 declined the same
change. There the outcome was already known, so re-initialising to improve it
would have been post-hoc; here it is a precondition measured before any M84
number exists, and the diagnostic that motivates it involves no arm, no rung and
no verdict. The gate enforces it: an **`exposure_validity`** clause reports the
active fraction at the initialisation actually used and returns
`exposure_term_inert` — suppressing every figure below it — if the negatives
cannot move the objective. Its unit tests are the direct analogue of N83.8's.

### N84.7 — each exposure row is assigned one owner, fixed at initialisation, and the drift is reported (29 July 2026)

A real out-group row has no class, so the exposure term must push it outside
whichever boundary would otherwise accept it — the argmin over 128 classes.
Recomputing that argmin at every optimiser step costs a full `(rows, classes,
dimension)` projection and is not affordable across sixty fits, so each row's
owner is computed **once, exactly, at initialisation** and held fixed, exactly as
M83's probes carried fixed owners.

The cheap approximation was rejected on measurement rather than taste. Assigning
the nearest centroid instead of the exact argmin would be wrong most of the
time: the boundary argmin is the nearest centroid for only **0.3065** of rows,
is within the nearest four for 0.6160 and within the nearest eight for 0.7530,
with a 99th-percentile rank of 56. The ellipsoids are anisotropic enough that
Euclidean proximity does not predict which one accepts a point, and a candidate
set small enough to be fast would have been a silent approximation of the wrong
quantity.

Fixed owners are still an approximation, because training moves the radii and
therefore can move the argmin. The cost is bounded and reported rather than
assumed away: every fit records the fraction of exposure rows whose exact argmin
is unchanged at the end of training. **Nothing in the verdict depends on it** —
rejection recall is always read with the exact min over all classes — so owner
drift degrades the supervision's efficiency, not the measurement's validity.

### N85.1 — the 640-per-class re-extraction is discharged as unreachable, not waived (30 July 2026)

M78 registered that "M85 must re-extract DomainNet at 640+ per class before any
transfer claim is made". Infrastructure I4 subsequently **measured** the yield
that requirement assumed: under the corpus's own ≥256 px native filter, class 65
has only **579** usable training images, class 53 has 603 and class 96 has 627.
The uniform ceiling over the 128 classes is 579, the build failed closed at
81,809 of 81,920 images, and the registration was amended to **576 per class**.

The requirement is therefore **impossible in the data**, and the distinction
matters: it is not being waived because it is inconvenient. What it existed to
guarantee — that no transfer claim rests on the sampling inadequacy M78 found in
v12 — is satisfied on the corpus that exists. Section 3's floor is 10 samples per
fitted dimension; the corpus supplies **536 fit rows per class**, which is
**10.11** at the rank cap of 53 and **16.75** at rank 32. Both clear the floor.
v12's void M74 cell, for comparison, sat at **1.88**.

M85 therefore does **not** re-extract DomainNet, and no M85 result may be
described as having met a 640-per-class bar. The bar does not exist in
DomainNet at this resolution filter.

### N85.2 — CIFAR-100 is 32×32 and the corpus is ≥256 px, so resolution is a confound the size of the effect (30 July 2026)

Measured before any M85 code was written, on 384 real corpus rows sampled at
seed 85085, holding class, backbone, graph and geometry fixed and changing
**only** the input resolution — each image degraded to 32×32 by bilinear
resampling and re-extracted through the same frozen INT8 graph:

| quantity                                       | value      |
| ---------------------------------------------- | ---------- |
| nearest-class-mean accuracy, native            | **0.5234** |
| nearest-class-mean accuracy, degraded to 32×32 | **0.3073** |
| median cosine(native, degraded)                | 0.6633     |
| median ‖shift‖                                 | 38.613     |
| median ‖row − own class mean‖                  | 31.163     |
| **median ‖shift‖ / within-class distance**     | **1.166**  |

Resolution alone destroys 41 % of the classification signal, and moves a row
**further than the distance to its own class mean**. The degradation is not a
small perturbation of the geometry; it is larger than the structure the geometry
describes.

The registered transfer corpus is CIFAR-100, whose images are 32×32 natively.
A CIFAR-100 transfer number read against the v13 geometry would therefore be a
corpus effect and a resolution effect superimposed, with no way to attribute
either. This is the same defect class as M83.1's probes: an operand that cannot
mean what it appears to mean.

**M85 therefore carries a resolution control as a third arm**, and no transfer
claim may be read without it:

| arm                    | corpus  | resolution | isolates                  |
| ---------------------- | ------- | ---------- | ------------------------- |
| native DomainNet       | same    | same       | the reference             |
| **degraded DomainNet** | same    | **32×32**  | **the resolution effect** |
| CIFAR-100              | changed | 32×32      | corpus **and** resolution |

The middle arm is the whole point: it is the identical evaluation rows, degraded
to CIFAR-100's resolution and re-extracted, so the difference between it and the
third arm is corpus and nothing else. A transfer failure that also appears in the
middle arm is a statement about 32×32 images, not about transfer.

### N85.3 — the re-extraction path is verified against the sealed corpus before it is trusted (30 July 2026)

The measurement above is only admissible if M85's extraction path is the one
that built the corpus. Re-extracting the 384 sampled rows through M85's code
reproduces the sealed corpus features with **max |diff| exactly `0.000e+00`**.

This check is **gating in every M85 builder**, not a one-off. Any M85 extraction
that fails to reproduce sealed rows bit-for-bit is a code defect, and the
milestone stops rather than emitting an artifact whose features are a different
function of the images.

### N85.4 — the open-set leg reports no AUROC, and L2 makes AUROC gating (30 July 2026)

A reporting defect found while auditing M85's inputs, recorded here in the open
as N82.5–N82.7 were.

`ACCEPTANCE_CRITERIA_v13.md` §5 registers a **threshold-free bar as gating** —
"AUROC at or above the same controls" — and §5's reporting rule states that every
open-set number is reported as `(recall, AUROC, corpus, known-class count,
samples per fitted dimension)`, a bare percentage being "not interpretable and
not admissible". The criterion says in as many words that this program "reported
recall-at-threshold for years without recording the AUROC it was already
computing".

**No v13 evidence file contains an AUROC.** M83 and M84 both report rejection
recall at matched coverage and nothing threshold-free. Their verdicts are not
withdrawn — both are refutations, and a refutation on recall is not rescued by a
statistic nobody computed — but they are **incompletely reported** against the
program's own criterion.

M85 completes them from the sealed artifacts. The boundary's minimum-over-class
score is a continuous score, so AUROC over held-out knowns against unseen rows
is computable from the sealed geometry with no re-training and no new fit. Two
things follow, and both are registered now rather than after the number is seen:

- **A threshold-free result cannot overturn a threshold result.** If AUROC comes
  out meaningfully above 0.5 while recall at matched coverage is zero, the
  finding is that the score ranks better than it thresholds — which is a
  statement about the coverage-matching rule, not a rescue of H83 or H84.
- **The direction is not predicted here**, deliberately. M84's mechanism (the
  union of 128 elongated ellipsoids covering the space) predicts a poor AUROC,
  and M84's domain decomposition predicts a good one on photographic domains.
  Both are reported, per domain and in aggregate.

Two further clauses were written into
`experiments/configs/v13/m85_open_set_auroc.json` before the run and are
transcribed here so that the ledger, not only the configuration, carries them.

- **N85.4c — R5 applies.** The geometry's AUROC is reported beside free
  baselines computed on the identical rows, the identical split and the
  identical frozen fit set: distance to the nearest class centre, which uses the
  geometry's own centres but none of its learned radii or subspaces, and
  distance to the _k_-th nearest fit row, the k-NN bar this program has used as
  its free-composability control throughout. An operand without its null is not
  evidence.
- **N85.4d — the instrument is validated at both ends before any figure is
  read**, as `ACCEPTANCE_CRITERIA_v13.md` §5 requires and as M83.1 failed to do.
  The positive control places synthetic points at five times the data's median
  radius from its own mean, where AUROC must be at or above 0.99. The negative
  control splits the held-out knowns in half and scores one against the other,
  where AUROC must sit within 0.02 of 0.5. A run that fails either control
  reports `instrument_invalid` and suppresses every figure below it.

### N85.5 — the frontier is assembled from sealed evidence, and absent cells stay absent (30 July 2026)

Every cell in M85's frontier table cites the milestone and the evidence hash it
was read from. Where a milestone did not produce a cell, the cell is recorded as
**absent** and the reason given; it is never filled from the nearest available
number, from a different task width, or from a different corpus.

Three restrictions already in force are repeated here because the frontier table
is exactly where they would be violated:

- **R7.** The four filled cells are v12 CIFAR-10 figures at 8-way I5. They are
  historical reference, are not v13 bars, and may not be compared against any
  v13 number. Every row carries its corpus and its width.
- **R4.** Both I5 widths or neither. The 8-way number may not be quoted alone,
  and M81's 8-way result is a per-seed maximum whose per-arm mean is 38.66 %
  (N82.7).
- **R5.** Every comparative operand carries a null sharing its structure, its
  budget and its split. A cell without its null is not evidence, whatever value
  it takes, and is reported with the null beside it or not at all.

**N85.6 was never issued.** The gap in the numbering is recorded rather than
closed by renumbering, so that a reader who finds a citation to N85.6 anywhere
knows it is a mistake and not a deleted note.

### N85.7 — the ≥256 px filter is waived for the two transfer artifacts, explicitly and for a stated reason (30 July 2026)

Registered in `experiments/configs/v13/m85_transfer.json` before the artifacts
were built. Every CIFAR-100 image is 32×32 and would fail the corpus's native
short-edge filter; so, by construction, would every degraded row. The filter
exists to hold the corpus's resolution uniform, and these two artifacts exist
precisely to vary it, so applying it would reject the measurement rather than
protect it. The waiver is named here so it cannot later be mistaken for an
oversight, and it applies to **nothing else in v13** — no geometry, no
dictionary, no head, and no future milestone inherits it.

### N85.8 — the transfer operand is retention, not accuracy (30 July 2026)

A 128-way DomainNet accuracy and a 20-way CIFAR-100 accuracy are not comparable,
and R7 forbids exactly this class of cross-corpus reading. What transfers or
fails to transfer is **the dictionary's value relative to the raw features it
codes**, so every cell reports sparse-probe accuracy divided by dense-probe
accuracy on the identical rows, the identical split and the identical training
budget. Absolute accuracies are recorded in the evidence file for audit and are
never compared across arms.

### N85.9 — width is matched rather than assumed away (30 July 2026)

DomainNet is read at its full 128 classes and again at labels 0–19, fixed
positionally with no selection, so that the CIFAR-100 cell has a same-width
partner. CIFAR-100 is read at the matched 32/32-per-class budget **and** at its
full 250/250, because reporting only the budget that flatters would be the R4
defect appearing in a new place.

### N85.10 — R5 nulls for every transfer cell (30 July 2026)

Every cell carries a label-shuffled null sharing its structure, budget and
split, and the dictionary carries M80's registered random-dictionary null at
identical size and identical active-atom budget. A cell is reported with its
nulls or not at all. The free-composability bar is k-NN on the raw features of
the same split.

### N85.11 — every arm M81 measured is listed, and no arm is selected (30 July 2026)

A frontier assembled by picking one point per family is the cherry-picking that
R4 was written against. The reader is given the whole measured set — fifteen arms
at each of two widths — and the selection rules are left in M81's gate where they
remain auditable. The assembler computes nothing; it reads sealed evidence and
formats it.

### N85.12 — open-set competence is a property of the boundary, not of any head (30 July 2026)

Every head in the frontier table reads the same frozen features and the same
fitted geometry, so a per-head OOD column would imply a distinction nobody
measured. It is reported once, for the boundary, carrying both its threshold
operand (rejection recall at matched coverage) and its threshold-free one
(AUROC).

### N85.13 — the frontier cells the plan attributed to M79 were never measured by M79 (30 July 2026)

Found while assembling the table. `RESEARCH_IMPLEMENTATION_PLAN_v13.md` marks
four frontier cells — the MLP + integrated-gradients and MLP + SHAP rows — as
filled by M79. **M79 measured nothing.** It was an acceptance reframe that
produced `ACCEPTANCE_CRITERIA_v13.md`; there is no `logs/results/v13/m79*`
directory and no M79 evidence file exists. Those cells are filled from M81's
`mlp_integrated_gradients` and `mlp_expected_gradients` arms, which did measure
them, and are cited to M81. The plan's attribution is left uncorrected as the
historical record; this note is the correction. Nothing was filled from a
milestone that did not produce it, which is what N85.5 exists to prevent.

### N86.1 — the verifier loads no features, trains nothing, and opens no final labels (30 July 2026)

M86 reads sealed evidence, hashes artifacts, recomputes each stored evidence
hash from its own payload, and reproduces the conclusion operands. It does not
load the feature array, does not fit a geometry, and does not read a label.
**Any v13 conclusion that cannot be reproduced from sealed evidence alone is a
finalization failure**, not a rounding issue, and fails the run closed.

### N86.2 — byte-identical replay is performed, not asserted (30 July 2026)

v12's M76 verified the `exact_replay` flags that milestones wrote **about
themselves**, which is a milestone's own word for its own reproducibility. M86
re-executes a milestone instead. **M85a is the designated replay target**, fixed
here before the replay runs: it is the cheapest milestone that exercises the
full path — closed-form Phase A subspace fit, Phase B Adam optimisation,
coverage matching, and all three scorers — at 0.4 minutes, and it already
carries an internal reproduction gate against M84's registered zero rung.

The replay writes to a scratch directory and **never over the sealed one**.
Registered before it runs: the comparison is over the evidence payload with
`generated_at`, `runtime_seconds` and `evidence_hash` removed, because the first
two are wall-clock quantities and the third is self-referential. **A difference
in any other field is a replay failure and Outcome F.** Excluding a field after
seeing a mismatch would make the check meaningless, so the exclusion list is
fixed here and nowhere else.

### N86.3 — the stored evidence hashes are recomputed from their own payloads (30 July 2026)

Verifying a file's SHA-256 against a value recorded in the same repository
proves only that the file has not changed since that value was written. It does
not prove the file is internally consistent — an evidence file edited by hand
before its hash was recorded would pass. M86 therefore **recomputes each
milestone's `evidence_hash` from the payload it sits in** and requires a match.

The v13 milestones do not all hash the same way: some hash the whole payload,
some exclude `generated_at`, and the runner that wrote M85a hashed **after**
`runtime_seconds` was set, so that field is inside the sealed hash. M86 tries a
fixed, ordered list of exclusion rules and records **which rule reproduced each
hash**, so the inconsistency is documented rather than smoothed over. A
milestone whose hash no rule reproduces fails the run closed. A milestone that
stores no `evidence_hash` is recorded as such and verified by file digest only —
it is not silently counted as passing.

### N86.4 — the v12 ledger amendments are verified, not assumed (30 July 2026)

`RESEARCH_IMPLEMENTATION_PLAN_v13.md` §15 makes skipping the M77/M78 amendment
to the v12 ledger **Outcome F**, and makes it a precondition of every
architecture milestone that has since run. M86 does not take that on trust: it
hash-locks `analysis/V12_FINAL_CLAIM_LEDGER.md` and checks that both amendment
headings are present and that both cite an evidence file that exists on disk.

### N86.5 — the final labels are not opened, and the reason is recorded (30 July 2026)

`ACCEPTANCE_CRITERIA_v13.md` §11.1 holds the final labels sealed "until M86",
and that phrasing must not be read as a confirmation run that took place.

**v13 built no final-label holdout.** The corpus partitions into fit rows and
evaluation rows and nothing else; there is no third split whose labels were
withheld. `final_labels_opened: false` is a protocol assertion that no milestone
read a label reserved for confirmation, and **it stays false**.

The disposition is honest rather than procedural. A final confirmation is
meaningful only behind a passing gating conjunction — it exists to spend a
one-shot holdout on a claim that has already earned it. v13's conjunction did
not pass: M81 blocked dominance, M82 closed nameability negative, and M83 and
M84 closed the open-set leg. There is nothing to confirm, so the seal is
recorded as **never opened** rather than opened and unused.

### N86.6 — M86 concludes; it does not measure (30 July 2026)

No new operand, no new arm, no re-fit, no new corpus. If a v13 conclusion turns
out to need a number M86 would have to compute, that is evidence the conclusion
was not supported by the milestones that were supposed to establish it, and the
correct response is to record the gap — not to close it in the finalization
step, where nothing was pre-registered and every degree of freedom points
towards a tidier story.

---

## 7. Amendments

Appended with a reason, a date, and the milestone that issued them.

### Amendment R10 — the M82 kill switch fires; "named" phrasing withdrawn program-wide (29 July 2026)

**Issued by:** M82, on its own registered kill switch. **Evidence:**
`logs/results/v13/m82_atom_naming/evidence.json`,
`evidence_hash 5d484f76…82bc3fc`.

`analysis/RESEARCH_IMPLEMENTATION_PLAN_v13.md` Section 15 registered: _"Naming
instability at M82 forbids any 'human-interpretable' phrasing in the ledger."_
M82 returned `names_unstable`. The switch fires, and is applied here rather
than being noticed at write-up time when the phrasing would be load-bearing.

Withdrawn wherever it appears, in this ledger and in the plan:

| withdrawn                         | replacement                                                            |
| --------------------------------- | ---------------------------------------------------------------------- |
| "sparse **named** basis"          | "sparse indexed basis"                                                 |
| "sparse **nameable** basis"       | "sparse indexed basis"                                                 |
| "human-interpretable"             | "sparse and low-citation"                                              |
| "the explanation is human-facing" | _(claim deleted — it was M82's conditional, and the condition failed)_ |

**This is a restriction on phrasing, not a retraction of a measurement.** What
M81 and M82 jointly support is unchanged and is stated in the surviving
vocabulary: explanations over the M80 basis cite **5.7 atoms** per decision,
inside M79's 10-atom deployment budget, and revealing _which_ atoms fired is
worth **+51.9 I5 points** over summary statistics. Both facts are about
component _identity_ and explanation _length_. Neither requires an atom to
correspond to a word, and M82 showed that none reliably does.

Consequently **Outcome A is unreachable for v13** as written, since it is
defined over a "sparse named basis". Outcomes C and D remain reachable and are
unaffected — neither mentions naming. The Section 17 interpretation row
"M82 naming stable → the explanation is human-facing" is struck; its complement
now holds.

M85's frontier table basis column is relabelled `sparse indexed` in the same
edit, so the deliverable table cannot inherit the withdrawn word.

### Amendment R4 — M81's decision rule restated at two task widths (29 July 2026)

**Issued by:** M80, **before M81 is executed and while no I5 number exists.**
**Evidence:** none required; this is a defect in a registered threshold, found
by reading it against the corpus it will be applied to.

The plan's decisive rule reads "I5 at or above 40% (chance 12.5%)" and "at or
below 25% (kNN control level)". Chance of 12.5% is 1/8 — those figures come
from v12's **8-way** CIFAR-10 forward-simulation task. The v13 corpus carries
**128 classes**, where chance is **0.781%**, and the v13 kNN control has never
been measured, so "25% (kNN control level)" is false as written.

This is the **same defect as L2's 87.0% unknown-recall bar**, which N1 exposed
and M79 withdrew. It survived because M79 rewrote this ledger and the
acceptance criteria but not the implementation plan.

The original absolute rule is **retained unchanged at I5-8**, the width where
it was defined, and a corpus-relative **I5-128** operand is added against a
re-measured kNN control. Both are measured, neither may be reported alone, and
the verdict is their conjunction — including the registered case where I5-8
passes and I5-128 does not, which is declared a **task-width artifact** in
advance because N1 established that this program's strongest numbers were
narrow-width artifacts. Full table at plan Section 8.

Restating a 40% bar corpus-relative at 128-way would have _lowered_ it; the
amendment is therefore a tightening, not a rescue.

### Amendment R5 — standing null-control contract (29 July 2026)

**Issued by:** M80. **Evidence:**
`logs/results/v13/m80_sparse_dictionary/evidence.json`.

Generalizing N80.2 as M83's degeneracy test generalizes M77: **every
comparative operand is reported alongside a null sharing its structure, budget,
and split, and any gate selecting a winner selects only among candidates that
beat that null.** An operand without a null is not evidence, whatever value it
takes.

### Amendment R6 — M82's naming floor registered and its confound named (29 July 2026)

**Issued by:** M80, before M82 is executed.

M82's gate deferred to "a registered floor" that no document registers; a floor
fixed after seeing the agreement rate is not a gate. Further, the plan's "two
independent naming channels" are both driven by the same top-activating images
for an atom, so they can agree while both are wrong. Independence was asserted
and never tested.

Registered: a shuffled atom-to-exemplar null as the floor; a disjoint-exemplar
independence check, reported; and atom purity admissible only against its own
shuffled-label control at matched counts, per the M80 entropy ceiling. Full
text at plan Section 9.

### Amendment R7 — the M85 frontier table holds CIFAR-10 figures (29 July 2026)

**Issued by:** M80, with R4.

The plan's deliverable frontier table arrives with four cells already filled
from **v12 CIFAR-10 at 8-way I5**. They are historical reference only, are not
v13 bars, and may not be compared against any v13 number; its accuracy column
is likewise CIFAR-10 against a 128-class v13 raw-probe bar of 61.304%. The
table is rebuilt at M85 from controls re-measured on the v13 corpus at both
widths.

### Amendment R8 — M82's naming delta is confounded with identity revelation (29 July 2026)

**Issued by:** M81, after execution. **Evidence:**
`logs/results/v13/m81_sparse_head/evidence.json`.

H82 claims naming raises I5 by converting indices into concepts. M81 measured
I5 with component **identity withheld** (N81.2); M82's names necessarily reveal
it. The difference against M81's number would therefore measure identity
revelation and credit it to naming. Registered: M82 measures identity-withheld,
identity-revealed-unnamed, and identity-revealed-named arms on the identical
split, atom set, budget and explanation width, and the naming claim rests on
**named minus revealed-unnamed** only.

Two further restrictions. M81 left **no accuracy-comparable atom arm inside the
10-atom budget at 128 classes in any seed at any tolerance**, so there is no
baseline at that width for naming to raise; the 128-way delta is reported and
**non-gating**, and any M82 claim is an 8-way claim. And the evaluand is fixed
in advance — M82 carries `sparse_linear_budget_256` — so the naming channel
cannot select the arm it is scored on. Deltas are read against the spread of
their own three seeds, since M81's own 8-way pass cleared its bar by 0.22
points against a 3.93-point spread. Full text at plan Section 9.

### Amendment R9 — "two independent naming channels" withdrawn (29 July 2026)

**Issued by:** M82, before execution, from an artifact audit. **Evidence:**
`logs/results/v13/m80_sparse_dictionary/evidence.json` (`corpus.dimension` 384);
`data/v5/backbones/dinov2-small`.

SpLiCE, named in Section 3 above as the closest prior art to M82, decomposes
**CLIP image embeddings**, where text alignment is possible because text and
image share one joint space. The v13 dictionary is fit over **384-dimensional
DINOv2** features. A v13 atom is a direction in DINOv2 space, which has no
alignment with any text space, so it cannot be compared to a phrase directly.

The only route from a v13 atom to a phrase runs through that atom's
top-activating images, and so does the exemplar channel. The two channels are
not merely confounded by shared input, as R6 recorded — they are the **same
channel** differing only in what reads the exemplars, and no third route
exists. Inter-channel agreement therefore cannot evidence correctness, and the
plan's claim of two independent channels is **withdrawn**.

Registered in its place: the agreement operand is renamed **exemplar-resampling
stability** and R6's disjoint-exemplar check is promoted to the primary naming
operand; a **gating positive control** is added, since one is available —
naming accuracy on strongly class-pure atoms against DomainNet's 128 ground
-truth class names, with a shuffled atom-to-name assignment as the negative
end; the text pipeline is described everywhere as **caption retrieval over
exemplars**, never as CLIP-space decomposition; and CLIP enters as a hashed
input artifact produced outside the frozen `.venv`, exactly as the DINOv2
backbone does. If stability fails, H82 is answered negative and M81's N81.2
nameability burden is discharged as **unmet**. Full text at plan Section 9.

### Amendment R3 — M78 finding 5 withdrawn (29 July 2026)

**Issued by:** N1. **Evidence:** `logs/results/v13/c3_probe.json`,
`logs/results/v13/c3_scale.json`.

M78 concluded that sample adequacy explains **none** of the open-set failure and
that the open-set negative was therefore isolated and strengthened. That
conclusion is withdrawn. Every M78 cell was sample-starved — the richest held
1.88 samples per fitted dimension at rank 32 against a floor of 10 — so the grid
could not have observed an adequately sampled open-set cell. At 536 samples per
class the geometric head reaches 23.86% recall at matched 10% false-alarm rate
against a 20.42% logistic bar and a 19.86% kNN control. The open-set deficit
closes with sampling, as the closed-set deficit does. M78 findings 1–4 stand and
its headline result is strengthened.

---

### Amendment R11 — the open-set leg's threshold-free bar is reported, and is not met (30 July 2026)

Issued by M85 under N85.4. `ACCEPTANCE_CRITERIA_v13.md` §5 makes AUROC a
**gating** bar for L2 and states that a bare recall percentage is "not
interpretable and not admissible". M83 and M84 reported rejection recall and
nothing threshold-free, so the leg's headline figures were incompletely reported
against this program's own criterion for as long as they stood. That is now
corrected rather than quietly left.

`experiments/tier4/eval_v13_m85_open_set_auroc.py`, evidence hash
`bf72f81de6f6bd7ed14f0f02101cfd13bd82a75bf1bf0791eada910f2910decb`. No training
is re-run: Phase A is closed-form on the same fit rows, at rank 51, under
N83.7's domain-quota partition and the same 90 % coverage match, and the run
**gates on reproducing M84's zero rung to the ninth decimal** — measured
`0.11875` against registered `0.11875`. The ranked object is the thresholded
object.

Instrument validated at both ends before any figure was read, for all three
scorers: far-field control `1.0000` against a floor of 0.99, held-out-known
split `0.4828`–`0.4845` against a tolerance of ±0.02 about chance.

| scorer                       | pooled AUROC | within-domain AUROC | cost        |
| ---------------------------- | ------------ | ------------------- | ----------- |
| fitted geometry              | **0.5851**   | 0.6580              | Phase A + B |
| distance to nearest centre   | 0.5617       | 0.6166              | free        |
| distance to 10th nearest row | **0.5749**   | 0.6469              | free        |

**The geometry does not clear its own free baseline.** It exceeds 10-NN distance
by 0.0102, inside the 0.02 margin registered before the run, so the verdict is
`geometry_ties_free_baselines` and `supports_threshold_free_bar` is **false**.
An operand at 0.585 on a threshold-free scale is not open-set competence, and it
is not bought by the fit — the same separation is available from a distance
computation that costs nothing.

Two further findings, both registered as reportable before the numbers existed:

**The pooled AUROC sits below every one of its parts.** Per domain the geometry
reads 0.7092 clipart, 0.5841 infograph, 0.6759 painting, 0.6404 quickdraw and
0.8691 real, weighting to 0.6580 within domain against 0.5851 pooled — a Simpson
gap of **+0.0729**, and the same gap appears in both free baselines. Pooling
destroys signal because the score separates _rendering style_ more strongly than
it separates novelty, which is precisely the failure mode N83.2 required the
per-domain split to expose. Both figures are reported; neither replaces the
other.

**Ranking and thresholding come apart completely.** Quickdraw ranks at AUROC
0.6404 while its rejection recall at matched coverage is `0.0003`; real
photographs rank at 0.8691 with recall `0.7333`. The score carries usable
ordering in a domain where the coverage-matched threshold rejects nothing at
all. Per N85.4a this is registered as a statement about the per-class
coverage-matching rule and **not** a rescue of H83 or H84: both remain refuted
on the operand they were registered against. A single global operating point
cannot serve domains whose score distributions differ this much, but acting on
that is new work and needs new registration — it is recorded here as a
diagnostic, not an operand.

---

## 8. Outcome

**Outcome C, with the interpretability leg closed negative.** v13 closes at one
of the outcomes registered in `analysis/ACCEPTANCE_CRITERIA_v13.md` Section 10.
Outcome **C** — a characterised `(accuracy, I5)` frontier without dominance — is
registered in advance as a **success**, so that no failure to dominate creates
pressure to keep altering the setup until a win appears.

M81 blocked the dominance claim and left a frontier claim standing, which is
Outcome C's shape. The registration matters here: had C not been fixed as a
success before M81 ran, the 8-way number would have been under real pressure to
be reported alone. It is not reportable alone, and it is not.

M82 then closed the nameability question negative, and did so with its
instrument validated at both ends. This is the strongest form a negative can
take: the naming channel demonstrably works — 89.54 % accuracy on class-pure
atoms against a 0.19 % shuffled-name null — and the atoms still fail to hold
stable names, at **0.8481 agreement against a 0.9976 matched-size null**. The
null exceeds the signal because 82 % of atoms are named by a rendering-style
term and only 187 distinct names cover 8,192 atoms, so a randomly assembled
exemplar set lands on the corpus's modal name every time. The channel is largely
reading the corpus's marginal image distribution rather than the atoms.

**The result that most justifies the pre-registration is R8's.** I5 runs 38.66 %
with component identity withheld, 90.54 % with identity revealed under an
**arbitrary** matched-size grouping, and 89.83 % with identity revealed and
**named**. Reported against the withheld baseline alone — which is how M81
measured, and how the literature this sits beside typically reports — this would
have been a **51.9-point naming win**. It is a 51.9-point _identity-revelation_
win with a naming contribution of **−0.71 points**, inside its own 0.72-point
seed spread. R8 was written before any of this was measured, for exactly this
reason.

Where the program now stands:

- **N81.2's nameability burden is discharged UNMET**, as R9 registered in
  advance that it would be if stability failed. H80's monosemanticity clause
  remains unestablished (N80.1, biased estimator). No surviving measurement
  supports the claim that the sparse basis is nameable.
- **What survives is narrower and better supported:** revealing _which_
  components fired is worth ~52 I5 points over summary statistics, and finer
  identity is worth more than coarser (per-atom 93.30 % beats the 187-name
  grouping's 90.54 %). That is a claim about component identity, not about
  language, and it does not require the atoms to mean anything.
- **The 128-way failure is the finding that needs explaining**, not the 8-way
  success. Either the sparse basis genuinely cannot carry a readable
  accuracy-competitive head at that width, or `k=32` is too tight for 128
  classes and the (m, k) frontier needs re-opening — the latter would be an M80
  question re-entered, which the plan permits only once.
- **Three reporting defects were found by M82 and corrected in the open**:
  N82.5 (far-field rate conflated style names with absent objects), N82.6
  (`dead_atom_fraction` is wrong in M80's and M81's sealed evidence), and N82.7
  (M81's 40.22 % was a per-seed best-arm maximum misattributed to one arm, whose
  own mean is 38.66 % and sits below the bar).

**M83 closes the open-set leg negative, at the second attempt. The first
attempt was void (N83.8) and the distinction is kept.** M83.1 placed its
synthetic negatives between the class centroid and the data — farthest probe
8.61, nearest decile of real rows 24.36 — so it measured how well four
boundaries rejected points they were simultaneously required to accept. Its
recall figures are not operands and are not claimed. The tell was in its own
evidence: every arm rejected exactly zero held-out probes, the **untrained** arm
included, and an operand that is identically zero for an arm which performs no
training is not measuring the arms.

M83.2 re-runs with the ladder corrected and both premises now checked before any
operand is read. The ladder reaches past the data (farthest probe 90.49 against
a known tenth percentile of 24.37, 85.2 % of probes beyond the known median) and
held-out probe rejection is 0.39–0.45 rather than identically zero. The
degeneracy contract separates the two placement rules by more than thirty orders
of magnitude in spread (v12-form exactly `0.000`, absolute arm 29.12).

**The outcome is unchanged, and only now does it mean anything.** At matched
90 % coverage, with the known-class control accepted at 0.9178, the absolute arm
rejects **0.0000** of the out-of-set against **0.1226** for its own untrained
initialisation and **0.1205** for the shuffled-owner null. H83's registered
operand — that the boundary moves — is met and worthless, which is what N83.1
was written before execution to catch.

**The fact M83.1 could not have produced, and the reason the re-run was worth
the compute:** the trained boundary _can_ reject synthetic probes, and is still
**worse at it than not training at all** — 0.3926 against 0.4326 untrained and
0.4456 for the null. Probe-rejection skill and novelty-rejection skill do not
merely fail to correlate here, they move in opposite directions. The auxiliary
term is actively harmful, not inert.

**Why it cannot work is the finding, and it generalises past this
parameterisation.** Real out-of-set images are not radially distinguishable from
known ones on this corpus: within each domain, known and unseen rows sit at
near-identical distance from the global mean (44.15 / 43.81, 46.09 / 46.10,
50.42 / 50.42, 53.76 / 53.997). A radius-parameterised boundary has no operand
to work with, and synthetic negatives placed radially outward supervise a
direction along which novelty does not live. That is consistent with N1's
finding that unseen-class detection on this corpus is near chance for every
method tried, and it is a stronger statement than "the supervision did not
help": no amount of probe engineering along radii would have.

Two consequences for what may be claimed:

- **Synthetic negatives are not a route to open-set competence here**, and the
  reason is geometric rather than a matter of tuning. The remaining live
  hypothesis is that the geometry must be _shown_ real out-group data.
- **M84's exposure ladder is the only remaining test of the open-set leg.** M83
  sharpens its interpretation: if rejection improves with real out-group
  exposure where it did not with synthesised probes, the operative variable is
  having seen the negatives, and the direction novelty lives along is one the
  data has to supply rather than one that can be constructed.

**M84 answers that question negatively, and the open-set leg is closed.** The
ladder is `ladder_flat` and sealed. Every premise passes before an operand is
read — exposure activity 0.90 against N84.6's 0.5 floor, the known-class control
at 0.9107 against 0.85, and the zero rung reproducing N84.4's pre-registered
**0.11875 exactly on all three seeds**, which is the strongest available
evidence that the partition, the geometry and the evaluation set are the ones
that baseline was measured on.

Against that baseline, real out-group exposure scores **0.00000 to 0.00012**
across all eight feasible rungs, from ten negatives to ten thousand. The best
cell in the entire ladder is 0.000116, three orders of magnitude below doing
nothing at all. **N84.3's moment-matched null beats real exposure at every
rung** (0.0030–0.0777), so the structure of real out-group images is worth
strictly less here than a Gaussian carrying only their first two moments. There
is no trend in either count or diversity; the discontinuity H84 predicted at the
zero rung is a cliff in the wrong direction.

**N84.5 is confirmed as a mechanism and refuted as a benefit**, which is the
most informative outcome that note could have had. The predicted response was
tangent anisotropy rather than a smaller mean radius, and that is exactly what
happens — anisotropy scales monotonically with exposure, 0.4344 untrained
through to **1.5201** at ten thousand negatives — while rejection sits at zero
throughout. The boundary moves a great deal, in the predicted direction, and
every unit of that movement costs detection.

**Two diagnostics explain why, and they generalise past this objective.**
First, `owner_agreement` is **≤ 0.0003 across all 48 trained arms**: almost every
negative escapes the class it was assigned to and is immediately re-absorbed by
a different one. Rejection requires the minimum score over all 128 classes to
exceed 1, but the objective only ever pushes a point out of one of them. Ejection
is whack-a-mole against 127 remaining ellipsoids, and the union never shrinks.
Second, `known_false_rejection` falls to **exactly 0.0** on every exposure arm
while `known_novel_acceptance` holds at ~0.913: each class boundary still holds
precisely its coverage-matched 90 %, and yet the union of 128 elongated
ellipsoids now covers everything. A boundary can be per-class coverage-correct
and useless as a detector at the same time, and anisotropy is the route between
the two.

**The decomposition that reframes the whole leg, and is the most useful thing
either milestone produced.** The 0.11875 baseline is not weak everywhere. It is
an average over domains whose detection power differs by three orders of
magnitude. Untrained unseen-versus-known rejection reads **0.733 against 0.162
on real photographs**, 0.433 against 0.195 on painting, 0.181 against 0.074 on
infograph, 0.169 against 0.031 on clipart — and **0.0003 against 0.0000 on
quickdraw**, which supplies 3,510 of the 5,760 evaluation rows because N83.2
matches the out-of-set to a corpus that is 61 % quickdraw. The geometry detects
novelty respectably wherever the images are photographic, and has no power
whatsoever on the domain that dominates the corpus by its own matching rule.

This does not rescue H83 or H84; both were registered on the aggregate operand
and both are refuted on it. But it relocates the failure. "This geometry cannot
do open-set detection" is not supported by the evidence; "this geometry cannot
do open-set detection on quickdraw, and quickdraw is most of what it was shown"
is. Whether the aggregate is a fact about the method or a fact about the corpus
composition is now a live and answerable question, and it is a **new
registration** rather than a re-reading of these runs — the domain split was
recorded as a diagnostic, not as a registered operand, and no claim may rest on
it until it is tested on its own terms.

**What may and may not be said about the open-set leg:**

- **Neither route works on the registered operand.** Synthetic negatives (M83)
  and real out-group exposure at four orders of magnitude of scale (M84) both
  fail, and both fail _worse than doing nothing_. The limitation is not the
  supply of negatives, which is what M84 was built to rule out and did.
- **The objective is refuted as registered, not in every conceivable variant.**
  N84.7 fixed owners at initialisation, and M84's own diagnostic shows every
  negative changes owner during training. A scheme that re-assigns owners each
  epoch, or that penalises the minimum over all classes rather than one, is
  **untested here** and may not be described as refuted. It is also not obviously
  promising: the same diagnostic says the obstacle is that 128 overlapping
  ellipsoids must _all_ reject a point, which greedy sequential ejection does not
  address.
- **The frontier claim is unaffected.** Open-set competence was never part of
  Outcome C, and nothing in M83 or M84 touches the `(accuracy, I5)` result.

**M85 completes the open-set leg's reporting, tests transfer, and assembles the
frontier.** Its most useful contribution is the one nobody asked for: the
threshold-free operand `ACCEPTANCE_CRITERIA_v13.md` §5 makes **gating** for L2,
which no v13 evidence file contained. M83 and M84 reported recall at a threshold
and stopped, and the criterion says in as many words that this program did that
for years while already computing the AUROC. Reported at last, the fitted
geometry reads **0.5851** against **0.5749** for distance to the tenth nearest
fit row and **0.5617** for distance to the nearest class centre — a margin of
0.0102 inside the 0.02 registered before the run. **The geometry does not clear
its own free baseline on the operand the criteria call gating**, and L2's
threshold-free bar is recorded as met: **False**.

Two things about that number matter more than the number.

- **Pooling destroyed signal.** The pooled AUROC sits below _every one of its
  parts_ — 0.7092 clipart, 0.5841 infograph, 0.6759 painting, 0.6404 quickdraw,
  0.8691 real, weighting to 0.6580 within domain. The score separates rendering
  style more strongly than it separates novelty, so an aggregate over a corpus
  that N83.2 matched to 61 % quickdraw reads worse than any domain in it.
- **Ranking and thresholding came apart entirely.** Quickdraw ranks at 0.6404
  with a rejection recall of **0.0003**; real photographs rank at 0.8691 with
  recall 0.7333. Per N85.4a, registered before the number was seen, this does
  not rescue H83 or H84 — a threshold-free result cannot overturn a threshold
  result — but it does say that one global operating point cannot serve domains
  this different, and that is a new registration rather than a re-reading.

**The transfer result is the program's clearest positive, and it exists only
because the confound was measured first.** N85.2 established, before any M85
code was written, that degrading corpus images to CIFAR-100's 32×32 costs 41 %
of nearest-class-mean accuracy and displaces a row **1.166 times** the distance
to its own class mean. A CIFAR-100 figure read against a 256 px geometry would
have been corpus and resolution superimposed with no way to attribute either.

With the third arm in place — the corpus's own held-out images, degraded to
32×32 through the identical frozen graph — retention (N85.8) reads **0.9936**
native, **0.9060** degraded and **0.9157** on CIFAR-100. Resolution costs
**+0.0876**; the corpus costs **−0.0097** beyond it, inside the 0.05 margin and
_negative_. **The sparse dictionary transfers to a corpus it never saw, and the
entire measured loss is resolution.** Read without the middle arm, the CIFAR-100
number would have been reported as a transfer loss of exactly the size that
resolution alone explains.

The dictionary also earns its keep against its own null: **+0.0784 retention
over a random dictionary of identical size and identical active-atom budget on
CIFAR-100**, its largest margin of the three arms and larger than on its own
corpus (+0.0696). What transfers is fitted structure, not the shape of a sparse
code.

---

## 9. Finalization (M86)

**The v13 conclusion is reproducible from its sealed artifacts, and this was
checked rather than asserted.** `experiments/tier4/verify_v13_final.py` loads no
features, trains nothing, and computes no operand of its own (N86.1, N86.6). It
verifies **19 conclusion operands** across ten milestones, and every one passes.

**Byte-identical replay was performed, not asserted.** v12's M76 verified the
`exact_replay` flags that milestones wrote about themselves. M86 re-executed
M85a into a scratch directory — never over the sealed one — and compared field
by field: **zero differing fields**, with the stable payload hashing identically
on both sides at `c22e67b4…`.

That replay produced a finding about the program's own bookkeeping. **The sealed
`evidence_hash` did not match the replayed one**, `bf72f81d…` against
`f535eead…`, because `runtime_seconds` sits inside the hash. The stored hash
therefore verifies **integrity** — that nothing was edited after writing — and
**not reproducibility**. Two further inconsistencies were found the same way and
are recorded rather than tidied:

- **Four of ten milestones store no `evidence_hash` at all** (M77, M78, M80,
  M81), carrying only a configuration hash. They are verified by file digest and
  recorded as `absent`, not silently counted as verified.
- **`final_labels_opened` is absent from the whole M85 family and from M82.**
  Those milestones read sealed arrays that do carry the flag, so the assertion is
  inherited rather than restated — but it is inherited, and that is weaker than
  the protocol intends.

**The v12 ledger amendments the Section 15 kill switch requires are present and
verified** (N86.4): A1 and A2 are hash-locked, and both cite evidence files that
exist.

**Both void runs are retained, verified, and recorded as void** — M83's first
attempt (N83.8) and M78's R1 stability measurement. A void run that is deleted
is indistinguishable from a run that never happened, and the difference is
exactly what a reader needs in order to judge whether the surviving result was
selected.

**The final labels were not opened, and the reason is recorded rather than
performed** (N86.5). v13 built no final-label holdout: the corpus partitions
into fit rows and evaluation rows and nothing else. A final confirmation exists
to spend a one-shot holdout on a claim that has earned it, and v13's gating
conjunction did not pass. `final_labels_opened` stays **false**, and "sealed
until M86" must not be read as a confirmation run that took place.

**v13 closes at Outcome C**, registered as a success in
`ACCEPTANCE_CRITERIA_v13.md` §10 **before M81 ran**, which is the reason the
8-way number was never reported alone. What v13 delivers is a characterised
frontier and four decisive negatives, each with an identified mechanism:

| Leg                | Disposition                      | Mechanism identified                                                    |
| ------------------ | -------------------------------- | ----------------------------------------------------------------------- |
| Accuracy / I5      | **Frontier, no dominance** (M81) | 128-way failure is a task-width effect, not a basis effect              |
| Nameability        | **Negative** (M82)               | names read the corpus's modal rendering style, not the atoms            |
| Open set, probes   | **Negative** (M83)               | novelty is not radial on this corpus; probes supervise a dead direction |
| Open set, exposure | **Negative** (M84)               | 128 overlapping ellipsoids must all reject; ejection is whack-a-mole    |
| Transfer           | **Positive, narrowly** (M85)     | retention survives a corpus change; the loss is resolution              |

**What may be claimed from v13, in full:** revealing _which_ sparse components
fired is worth ~52 I5 points over summary statistics, and finer identity beats
coarser; the dictionary's retention transfers to an unseen corpus and beats a
size-matched random dictionary by more there than at home; and open-set
competence is refuted along both registered routes with the geometry named.
**Nothing here supports a claim that the basis is nameable, that the boundary
detects novelty, or that any head dominates its controls.**
