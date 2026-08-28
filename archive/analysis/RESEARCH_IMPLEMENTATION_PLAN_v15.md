# GEODE Research Implementation Plan v15

## Additive Construction Against a Discriminative Residual: Growing a Sparse Model Instead of Fitting One

**Status:** registration draft, 2 August 2026. **One milestone has since been
executed: M102 Tier A (§7.7), whose result is recorded in
`analysis/CLAIM_LEDGER_v15.md` and summarised in §2.8.5.** No other v15
milestone has run. Sections 1–11 were registered before any v15 figure was
produced; text added after execution is marked `[recorded after execution]` and
never replaces a registration, per §5.10.

**Amendment, recorded after M102.** M102 refuted H110. Three unsealed scoping
probes were then run and are registered in **§2.9**; two of them contradict
conclusions this plan previously drew, and all three are inadmissible as
operands under §2.4. The amendment they motivate is: Lever 1 is **closed**
(§2.9.1), M99's replacement gate is **blocked** by confounded backbone data
(§2.9.2), and a new unconditional milestone **M103** (§7.9) is registered to
test the plan's central thesis on the representation rather than the head. Four
bookkeeping defects found in the same audit are corrected in place: a broken
cross-reference, a stale escape condition in prohibition 21, six unqualified
uses of the inadmissible 44.4% figure, and an undisclosed milestone-ordering
deviation (§7). The arm (d) figure quoted in C102.2 was **void, not negative**,
under the plan's own sample floor; the ledger records the correction.

**Second amendment, recorded after the §2.9 audit.** Two further scoping probes
are registered in **§2.9.4** and **§2.9.5**, and both correct the first
amendment rather than extending it. §2.9.4 finds that §2.9.3's
random-beats-k-means ordering had a **second reading** — k-means optimises
reconstruction, not discrimination — and that selecting atoms against a
discriminative residual beats random selection in **all six** seed-budget cells
tried, at 3.0–3.7× the null's own seed spread. This is the first affirmative
scoping evidence anywhere on the M103 → M99 chain, and it **reverses the
prediction of failure** registered in §7.9 and §10.2, which are contradicted in
place rather than edited. §2.9.5 records that the **0.869 Thiry bar registered
in §8.5 did not re-verify** against the authors' own repository and is marked
unconfirmed. Two prior-art defects in the first amendment are corrected: §2.9.3
reproduces a result **Thiry et al. (2021) already publish**, which the first
amendment did not disclose, and prohibition 23 is extended to §2.9.4 and §2.9.5.
No gate, kill switch, sample floor or acceptance criterion is changed by this
amendment.

**Third amendment, recorded after the M103 instrumentation run.** §2.9.6
registers a **single-seed, full-scale, unsealed instrumentation run** built to
size the sealed M103 sweep, and it returns three findings that bear on the
registration. First, §2.9.4's discriminative gain **survives to 1024 atoms**
(+0.0133 over random), which answers §2.9.4's own registered limitation (ii).
Second, §2.9.3's random-beats-k-means ordering **reverses at full scale**,
weakening the reading that gave rise to the first amendment. Third, §7.9 design
item 4's instrument check **cannot be satisfied by any run** — it required arm
(b) at 1024–2048 atoms to reach a Coates figure measured at **4000 features**,
while §7.9 restriction 4, registered in the same commit, fixes M103's top
*readable* rung at 1024 atoms — and it made an external figure a pass/fail
operand, which R7 forbids. That item is contradicted in place and replaced by
three internal conditions plus a non-gating anchor, which **tightens** R7
conformance. Prohibition 23 is extended to §2.9.6. No operand, kill switch,
sample floor or acceptance criterion is changed by this amendment, and it is
recorded before M103's sealed run.

**Fourth amendment, recorded after M103 executed.** M103 (§7.9) has **run and
confirmed**, and its result is recorded in `analysis/CLAIM_LEDGER_v15.md`
entries C103.1–C103.8. Arm (c) reaches arm (a)'s 1024-atom accuracy at **512
atoms** at all three seeds individually, beating the null in **15 of 15**
seed-budget cells; neither kill switch fired. This is the program's **first
affirmative sealed result on its own central thesis**, and it is a **Q2** result
that carries no outcome letter — Q1 remains unanswered. Two findings run against
the plan's own prior text and are contradicted in place: §2.9.6 finding 2's
k-means reversal was a **single-seed artifact** that M103 overturns at three
seeds (C103.5), and prohibition 23's escape condition for §2.9.3, §2.9.4 and
§2.9.6 is now **discharged** for the figures M103 reproduced. C103.3 records the
limitation that binds the headline: the margin narrows from 3.45× the null's
seed spread at 64 atoms to **1.03×** at 1024, and any statement of the
efficiency result that omits it is a misstatement of the evidence.

**Fifth amendment, recorded after a prior-art audit of M103.** M103's result was
audited against the literature after it was sealed, and **the audit went against
it**. C103.1's phenomenon — a discriminatively chosen dictionary reaching a
random dictionary's accuracy at a fraction of the size — is **already published,
in a stronger form and by a label-free mechanism**, in the random-features
literature (§8.10). C103.3's narrowing margin is not merely an against-interest
caveat this program volunteered; it is **predicted** by that theory. This
disclosure is mandatory under §11.2 item 22's form and is recorded in
`analysis/CLAIM_LEDGER_v15.md` as an amendment to C103.1 rather than by editing
it. **No M103 operand changes and no M103 figure is withdrawn** — what changes
is the claim's standing as a contribution, which §8.4's consequence has always
governed.

Four further scoping probes are registered in **§2.9.7**. They test whether the
theory the audit surfaced hands the program a *usable instrument*, and they
return one finding the plan has never had: **effective rank spreads 6.3× across
data types on DomainNet**, while every published mixture-of-experts sizes its
experts **uniformly**. Three new unconditional milestones are registered on that
finding — **M104** (§7.10), **M105** (§7.11), **M106** (§7.12) — each with a
structure-matched null and a kill switch registered before it runs. §6.1 P2's
NP-completeness citation is **corrected in place**: it concerned *training*, not
routing or detection, and the result actually bearing on the question is a
PAC-learnability theorem whose own statement gives sufficient conditions for
learnability (§6.1 P2). Prohibitions 23 and 25 bind §2.9.7. No M102 or M103
operand, kill switch, sample floor or acceptance criterion is changed by this
amendment.

**Q1 remains unanswered and v15 therefore has no outcome letter.** M102 and M103
are both Q2 milestones (§3.4.1).

**Claim ledger:** `analysis/CLAIM_LEDGER_v15.md` — **created by M102**, earlier
than the plan anticipated, because M102 ran first under the §3.2.1 redirect.

**Registered questions:** three, on separate axes. **Q1** (§3.1) — can a grown
sparse model reach the frontier point on the `(accuracy, explanation length)`
plane? **Q2** (§3.2) — does sparsity buy efficiency at matched accuracy? **Q3**
(§3.3) — does additive construction transfer to sequential prediction?
**v15's verdict is Q1's alone** (§3.4.1); Q2 and Q3 are measured and reported
but do not gate Q1 and do not carry outcome letters.

**Claim ledger:** see the status block above.
**Acceptance frame:** inherits `analysis/ACCEPTANCE_CRITERIA_v13.md` unchanged,
with the two additions registered in §5.7. No criterion is relaxed, and §3.4.2
forbids adding an efficiency threshold to the Q1 criteria.
**Immutable parents:** v6.1 D, v7 C, v8 D, v9 D, v10 D, v11 E, v12 E, v13 C,
v14 (H90 refuted, H91 refuted, H92 untestable, H94 refuted, H95 untestable)

**Settled since first draft:** §6.2 A5 — dense neural networks are admissible as
system components, subject to the efficiency and inspectability obligations,
which are discharged by measurement (§5.11) rather than by assertion. This
unblocks M98 and is the reason Q2 and Q3 exist.

**Redirected since first draft:** §3.2.1 — Q2's centre of gravity moves off the
head. §2.6.2's arithmetic makes the head-side efficiency answer knowable before
any experiment (0.0008%–0.05% of inference compute), so Q2 is reassigned to the
three levers that can move compute: declining to run the expensive model
(**M102**, §7.7), replacing fitting with retrieval (**H111**, §4.3), and making
the representation cheaper (**M99**, whose gate is loosened in §10.2 because it
was the only compute-relevant milestone in the plan and it was conditioned on
outcomes that cannot bear on compute). §2.8 records the measurement that
motivates the first of these; prohibition 21 marks it inadmissible until
reproduced. Q1's milestones, criteria and verdict are unchanged (§3.4.7).
**[recorded after execution: the first lever is now closed — H110 refuted by
M102, and the label-free route refuted independently in §2.9.1. Q2's affirmative
case rests on Lever 3 alone, through M103 → M99. §3.2.1 registers that
concentration risk against interest.]**

---

## 1. What v15 inherits, and one correction

### 1.1 Inherited verdicts, not reopened

| source  | verdict                                                          | status in v15    |
| ------- | ---------------------------------------------------------------- | ---------------- |
| v13     | Outcome C — frontier characterised, no dominance                 | sealed, closed   |
| v13 M80 | H80 gate passed but voided by the random-dictionary control      | sealed, closed   |
| v13 M81 | `task_width_artifact` — I5-8 confirms, I5-128 unmeasurable       | sealed, closed   |
| v14 M90 | H91 refuted — L2 normalisation inert                             | sealed, closed   |
| v14 M90.1 | H90 refuted — domain-aware mixtures do not recover rejection   | sealed, closed   |
| v14 M90.2 | H94 refuted — domain survives complete linear erasure          | sealed, closed   |
| v14 M91 | H92 **untestable** — larger backbones confounded (N91.11, N91.12) | sealed, closed  |
| v14 M92, M93 | conditional, never opened                                   | remain unopened  |

V15 does not open M92 or M93, does not revisit open-set rejection, and does not
revisit domain invariance. Those are v14's questions and v14 answered them or
established that they could not be answered on this corpus.

### 1.2 The correction v15 is registered on

Every program from v5 to v14 has had the same shape: a **frozen dense trunk**
followed by a **fitted head**, with the research effort spent on the head. The
heads have been a radial SDF, a weighted readout, hard-boundary supports, an
analytic metric field, a learned projection plus metric field, a sparse linear
head over a learned dictionary, and a decision list.

§2 shows, using **only figures already sealed inside M81's own evidence file**,
that the head was never the binding constraint, and that the program's own
free control — k-nearest-neighbours — already sits on the joint
`(accuracy, explanation length)` frontier that fourteen versions were built to
reach.

This is a correction to the program's **reading of its own evidence**, not to
any sealed number. No sealed figure changes. What changes is that a table which
existed since M81 executed has not, until now, been read as a single object.

---

## 2. The motivating measurement

### 2.1 The M81 frontier, read as one table

All rows below are read directly from
`logs/results/v13/m81_sparse_head/evidence.json`, seed 11, width `i5_128`,
configuration hash
`61181c1c878cab273e66d11d5d45ad758b365766a7e83141e159c5af2cf1a6e3`. The
explanation budget of 10 atoms is the one registered at
`ACCEPTANCE_CRITERIA_v13.md` §2. Nothing here is refitted and nothing here is
new evidence; M94 exists to make it replayable.

| arm                        | family   | balanced accuracy | mean active atoms | fraction of decisions within the 10-atom budget |
| -------------------------- | -------- | ----------------: | ----------------: | ----------------------------------------------: |
| `metric_field_shrinkage_1.0` | geometry | **0.663452**      | 32.00             | 0.0000                                          |
| `knn`                      | control  | **0.661255**      | **6.72**          | **1.0000**                                      |
| `metric_field_shrinkage_0.5` | geometry | 0.660889          | 32.00             | 0.0000                                          |
| `mlp_integrated_gradients` | control  | 0.660522          | 384.00            | 0.0000                                          |
| `mlp_expected_gradients`   | control  | 0.660522          | 384.00            | 0.0000                                          |
| `metric_field_shrinkage_0.1` | geometry | 0.636841          | 32.00             | 0.0000                                          |
| _raw 384-d linear probe (M80 bar)_ | control | _0.613037_ | _384.00_          | _0.0000_                                        |
| `sparse_linear_l1_0.0`     | atoms    | 0.607178          | 32.00             | 0.0000                                          |
| `sparse_linear_l1_0.3`     | atoms    | 0.573975          | 15.60             | 0.0607                                          |
| `rbf_nystroem`             | control  | 0.569092          | 2047.96           | 0.0000                                          |
| `sparse_linear_l1_0.03`    | atoms    | 0.520020          | 22.65             | 0.0004                                          |
| `sparse_linear_l1_0.1`     | atoms    | 0.507813          | 21.43             | 0.0004                                          |
| `sparse_linear_budget_1024`| atoms    | 0.441284          | **5.16**          | **0.9655**                                      |
| `sparse_linear_budget_512` | atoms    | 0.351563          | 3.46              | 0.9954                                          |
| `sparse_linear_budget_256` | atoms    | 0.225098          | 2.28              | 0.9995                                          |
| `decision_list`            | atoms    | 0.146362          | 0.22              | 1.0000                                          |

### 2.2 Three readings that were available and were not made

**Reading 1 — the accuracy ceiling belongs to the representation.**
Three model families with unrelated inductive biases land within **0.293
points** of one another: a class-conditional anisotropic quadratic
(`metric_field_shrinkage_1.0`, 0.663452), a nonparametric neighbour rule
(`knn`, 0.661255), and a trained multilayer perceptron (`mlp_*`, 0.660522).
Meanwhile every **linear** head lands between 0.5078 and 0.6130, with the raw
384-dimensional probe at 0.613037 and the best linear head over learned atoms
at 0.607178.

Two clusters, separated by roughly **5.0 points**, with the members of each
cluster agreeing far more closely with each other than the clusters agree
across. The natural reading is that ≈0.66 is a property of the **dinov2-small
INT8 representation on this corpus**, and that the choice of nonlinear head
within that ceiling is worth less than a third of a point. Fourteen versions of
head engineering produced fourteen near-ties because the head was not the
binding constraint.

**Reading 2 — the free control already dominates the frontier.**
`knn` achieves 0.661255 balanced accuracy while citing a mean of **6.72**
neighbours, with **100%** of its decisions inside the registered 10-atom
explanation budget. It is within 0.22 points of the most accurate arm measured,
and it is the only arm in the table that is simultaneously at the top of the
accuracy cluster and fully inside the explanation budget.

Nothing this program has built beats it on both axes. This is a stronger and
more uncomfortable statement of v13's Outcome C than v13 made: the frontier was
not merely un-dominated by GEODE, it was **occupied by the cheapest control in
the acceptance frame**.

**Reading 3 — the actual gap is at the budget, not at the accuracy.**
M81's registered accuracy-comparability floor for I5-128 was
**0.6112548828125**, which is the best control (`knn`, 0.661255) minus the
registered 5.0-point tolerance. Against that floor:

- the best atom arm **at any explanation length** reached 0.607178, short by
  **0.4077 points**, which is why `seeds_with_no_admissible_atom_arm = 3` and
  I5-128 was never measured for a single seed;
- the best atom arm **within the explanation budget** (`sparse_linear_budget_1024`,
  96.55% of decisions inside 10 atoms) reached 0.441284, short by
  **16.997 points**.

The second number is the honest size of the problem. Reported as
"sparse heads nearly reached comparability" it is four tenths of a point.
Reported at the explanation budget the acceptance frame actually registered, it
is **seventeen points**.

### 2.3 What M80 already showed about sparsity and dictionaries

From `logs/results/v13/m80_sparse_dictionary/evidence.json`, configuration hash
`35da440938af3f49ba977d3c9576e1d43b57491bb90924efff17a03c2cd9c8c9`, raw-feature
probe bar 0.613037109375:

| dictionary size | active atoms k | probe accuracy | random-dictionary control | margin (points) | mean atom label entropy (bits) |
| --------------: | -------------: | -------------: | ------------------------: | --------------: | -----------------------------: |
| 2048            | 16             | 0.526367       | 0.400513                  | **+12.585**     | 3.872                          |
| 2048            | 32             | 0.535156       | 0.478271                  | +5.688          | 4.819                          |
| 2048            | 64             | 0.552734       | 0.535645                  | +1.709          | 5.521                          |
| 4096            | 16             | 0.549927       | 0.457153                  | +9.277          | 3.069                          |
| 4096            | 32             | 0.561523       | 0.538086                  | +2.344          | 4.159                          |
| 4096            | 64             | 0.574951       | 0.584351                  | −0.940          | 4.998                          |
| 8192            | 16             | 0.583130       | 0.496338                  | **+8.679**      | **2.285**                      |
| 8192            | 32             | 0.607910       | 0.570190                  | +3.772          | 3.381                          |
| 8192            | 64             | 0.612061       | 0.612305                  | **−0.024**      | 4.338                          |

Shuffled-label entropy null: **5.1312 bits**.

M80 read this grid for its gate and correctly voided the gate, because at the
cell that best matched the raw probe (m=8192, k=64) the **random** dictionary
was as good as the trained one (0.612305 vs 0.612061). The reading that was
available and was not made is the **direction of the margin across the grid**:

> The trained dictionary's advantage over a random one is largest exactly where
> the code is sparsest, and vanishes as the code becomes dense. At k=16 the
> trained dictionary is worth **+8.679 points**; at k=64 it is worth
> **−0.024 points**. Over the same sweep, mean atom label entropy falls from
> 4.338 bits to **2.285 bits** against a 5.131-bit shuffled-label null.

Sparse coding is informationally free at high k, which is what voided the gate.
At low k it is not free at all, and the atoms it produces are also the most
class-selective ones measured. **The regime in which learning the basis is
load-bearing is the same regime in which the basis is nameable.** M80 was run to
support a head; it also measured something about bases, and that measurement has
never been used.

### 2.4 Scoping observations — recorded, and inadmissible until M94 reproduces them

The following were produced during v15 planning on the sealed corpus arrays,
seed 11, using a **class-major prefix partition** (first 512 rows per class fit,
next 64 evaluation) and **ANOVA-F** feature ranking. They do not use M81's
contribution-mass ranking, do not use N83.7's domain-quota split, are
single-seed, and carry only matched random-grouping nulls. **They are scoping
only. No v15 argument rests on them, and none may be cited as evidence until
M94 reproduces them under the shared contract.** They are recorded here so that
if M94 contradicts them, the contradiction is visible.

| scoping observation                                  | value              |
| ---------------------------------------------------- | -----------------: |
| 10-NN, same rows                                     | 0.659790           |
| random forest, 400 trees                             | 0.646851           |
| dense logistic probe, standardised, 384-d            | 0.583618           |
| histogram gradient boosting, lr 0.05, 100 iterations | 0.577393           |
| logistic probe, train / evaluation                   | 0.846313 / 0.568115 |
| top-32 **ambient coordinates**, per-class one-vs-rest | 0.522583          |
| top-16 ambient coordinates                           | 0.451050           |
| top-8 ambient coordinates                            | 0.335205           |
| best hierarchical routing arm, any branching factor  | 0.112549           |

Two of these matter enough to register now.

**Scoping observation A — hierarchical routing is not the answer, and the
failure is not error compounding.** Every hard-routing tree measured
(branching 4, 8, 16 × k ∈ {2,4,8}) landed between 0.0239 and 0.1125, against a
flat top-8 arm at 0.3352 on the identical rows. Structure-matched random-grouping
nulls sat only 1–4 points below the semantically grouped trees. Replacing hard
routing with a soft hierarchical-softmax variant did not repair it. The
mechanism is already sealed in v14 M89: a class's own domain cells sit further
apart (32.30) than its nearest foreign-class cell (19.15), and M90.2 showed this
survives complete linear domain erasure. **v15 registers hierarchical routing as
a do-not-pursue direction** (§11.2 item 9), and this is consistent with the published
record — see §8.4.

**Scoping observation B — a learning-rate artifact that nearly became a
finding.** Gradient boosting on these features first read **0.0568** at
learning rate 0.2 with 150 iterations, and **0.0977** at learning rate 0.2 with
50 iterations. Both readings are consistent with the conclusion "axis-aligned
additive models fail on distributed embeddings", and that conclusion is false:
at learning rate 0.05 the same estimator reads **0.5774**, and a random forest
on the identical rows reads **0.6469**. A **52-point** swing between two
defensible hyperparameter settings at 128 classes. This is registered as a
standing measurement requirement in §5.6.

### 2.5 An external anchor, and an inversion the program has not explained

The DINOv2 paper (Oquab et al. 2023, **arXiv:2304.07193**, Table 4, verified by
fetch) reports for the **same backbone architecture this program uses**:

| backbone         | ImageNet-1k linear probe | ImageNet-1k k-NN | ordering        |
| ---------------- | -----------------------: | ---------------: | --------------- |
| DINOv2 ViT-S/14  | **0.811**                | **0.790**        | linear **beats** k-NN by 2.1 pt |
| DINOv2 ViT-g/14  | 0.865                    | 0.835            | linear beats k-NN by 3.0 pt |

On this program's corpus, the same two heads over dinov2-small features invert:

| corpus                              | linear probe | k-NN       | ordering        |
| ----------------------------------- | -----------: | ---------: | --------------- |
| ImageNet-1k, published, fp32        | 0.811        | 0.790      | linear beats k-NN by 2.1 pt |
| v13 DomainNet, sealed, INT8         | **0.613037** | **0.661255** | k-NN **beats** linear by 4.8 pt |

**The ordering of the two cheapest heads reverses between the published setting
and this program's setting — a swing of roughly 7 points.** A linear probe that
outperforms k-NN on ImageNet is outperformed by k-NN here.

This is registered as an **observation, not a finding**, and it is confounded at
least three ways: the corpora differ, the class counts differ (1000 against
128), and this program's extraction is **INT8 quantised** while the published
figures are not. R7 forbids comparing an external number to a v15 number as
evidence, and this comparison is therefore **not an operand** anywhere in the
plan.

It is recorded because it is the strongest available reason to run M95 (§7.2)
before spending anything on head or growth design. If the ceiling this program
has been measuring for fourteen versions is depressed by its own extraction
pipeline, that is worth one cheap milestone to find out.

### 2.6 The efficiency arithmetic, which the program has never run **[new in v15]**

A5 (§6.2) makes efficiency a binding constraint rather than an aspiration. That
obliges v15 to look at what the sealed evidence already says about it. It says
something unwelcome, and it says it in three independent places.

**2.6.1 The sparse arms are larger than the dense head they lose to.** M81's
`evidence.json` carries an `active_parameters` field on every arm. It has never
been read in any v13 or v14 write-up. Read against balanced accuracy, seed 11,
`i5_128`:

| arm | balanced accuracy | active parameters | mean cited atoms | fraction in budget |
| --- | ---: | ---: | ---: | ---: |
| `metric_field_shrinkage_1.0` | 0.663452 | 2,097,152 | 32.00 | 0.0000 |
| `knn` | 0.661255 | 25,165,824 | 6.72 | 1.0000 |
| `mlp_integrated_gradients` | 0.660522 | 262,784 | 384.00 | 0.0000 |
| `sparse_linear_l1_0.0` | 0.607178 | 1,048,704 | 32.00 | 0.0000 |
| `sparse_linear_budget_1024` | 0.441284 | 131,200 | 5.16 | 0.9655 |
| `sparse_linear_budget_256` | 0.225098 | 32,896 | 2.28 | 0.9995 |
| `decision_list` | 0.146362 | 67 | 0.22 | 1.0000 |

A dense linear probe over the 384 raw features costs `384 × 128 + 128` =
**49,280** parameters and scores **0.613037** (§2.1). Therefore:

- `sparse_linear_l1_0.0` uses **1,048,704** parameters — **21.3×** the dense
  probe — to score **0.607178**, which is **lower**. It is bigger and worse.
- The most parameter-efficient route to the ≈0.66 ceiling is the **dense MLP**
  at 262,784 parameters. `metric_field_shrinkage_1.0` spends **8×** that to buy
  0.0029 accuracy. `knn` spends **95×** that, because it stores the corpus.
- The only arms genuinely smaller than the dense probe are
  `sparse_linear_budget_256` (32,896 parameters, 0.225098) and `decision_list`
  (67 parameters, 0.146362). At or below matched size, sparsity costs 39 to 47
  points.

**On the parameter axis, this program's sparse arms have never once beaten a
dense baseline at matched accuracy. The ordering is monotone against them.**

**2.6.2 The head is not where the cost is.** Every arm above sits on a frozen
DINOv2 ViT-S/14 trunk. Using the verified architecture (patch 14, 12 layers,
width 384, MLP ratio 4, 257 tokens at 224×224 — §8.7 C1) the trunk costs
**6,065,759,232 MACs per image**, ≈6.07 GMAC. Derived here and cross-checked
against timm's published 46.8 GMAC at 518×518 for the same checkpoint. Against
that:

| component | MACs per decision | share of trunk |
| --- | ---: | ---: |
| frozen ViT-S/14 trunk | 6,065,759,232 | 100% |
| dense linear head (384→128) | 49,152 | 0.00081% |
| sparse head (384→8192 coding, then 32→128) | 3,149,824 | 0.0519% |

The sparse head costs **64×** the dense head's compute — and both are noise. A
change that moves 0.05% of inference compute is not an efficiency result in any
currency a deployment cares about. **Head sparsity on a frozen dense trunk
cannot produce an efficiency win, by arithmetic, before any experiment is run.**
This is registered as prior P6 (§6.1) and it is the single most consequential
fact in this section: it means the efficiency question cannot be answered
affirmatively anywhere in M94–M98. Two milestones are addressed to it —
**M99**, which grows the representation, and **M102**, which declines to run it.
§2.8 measures the headroom the second one targets, and §3.2.1 registers the
resulting redirect.

**2.6.3 The program already tried conditional compute, and it failed.** This is
pre-v13 evidence and is inadmissible as an operand under §5.1, but it is
directly on point and §5.10 requires disclosing it. `RESEARCH_REPORT_v5.md` §6
records: *"Candidate routing does not reduce observed wall-clock latency —
synthetic M12 and real E5 candidates fail latency/quality gates and remain
shadow-only; exhaustive evaluation stays authoritative"*, interpreted there as
*"Geometric bounds alone do not guarantee systems speedup"*. §7.9 of that same
external document — not of this plan: *"Candidate routing has not produced a real
latency advantage."* Its prohibition list already forbids claiming *"that current
routing scales better in wall-clock time."*

The same document also records that the accuracy-superior primitive family used
**more primitives and more fit time**, concluding that *"accuracy superiority
does not imply efficiency."*

So the program has already discovered, and already written down, that its
sparse-routing efficiency story did not survive measurement. v15 inherits that
finding rather than rediscovering it.

**2.6.4 What this section does and does not establish.** It establishes that on
the sealed corpus, in stored parameters and in multiply-accumulates, the sparse
arms are worse than dense baselines, and that the accounting boundary decides
the answer. It does **not** establish that sparse construction is inefficient in
general. Three currencies remain unmeasured here: **wall-clock latency**,
**training cost**, and **sample efficiency**. §5.11 registers all of them before
M100 measures any of them, precisely because §2.6.1 shows that an author free to
choose the currency after the fact could report either sign.

### 2.7 The task-generality record, and why it is void **[new in v15]**

The program has attempted sequence prediction exactly once, in Tier 6, and the
result is quoted in `RESEARCH_REPORT_v5.md` §6 as *"30.36% versus 34.64%
linear, 44.50% matched 5-gram"*. The sealed artifact
`logs/results/tier6_locked_window5_confirmation.json` gives the full picture:

| arm | next-token accuracy |
| --- | ---: |
| n-gram, best practical | 0.47606901725431355 |
| n-gram, matched to the geometric model | 0.445 |
| linear head, same features | 0.3464 |
| **GEODE geometric head** | **0.3036** |
| unigram floor | 0.19215 |

WikiText-103, window 5, 24 PCA components, 81 classes, perplexity 14.2463.

Two observations, in order of importance.

**2.7.1 The result is void under this program's own rules, not negative.** The
artifact's own `sample_adequacy` block reports `min_seed: 299`,
`below_minimum: 56` of `class_count: 82`. **Fifty-six of eighty-two classes fall
below the registered sample floor.** §5.3 inherits the floor of ten fit samples
per fitted dimension and states it is never waived. Under M83.1/N83.8 an arm
that fails its own adequacy check is **void, not negative**. Further, the config
shows `n_em_iters: 0` and `n_em_epochs: 0`, and `test_acc_em: null` — **the
temporal refinement loop never ran**. The reported 0.3036 is `test_acc_init`,
the static initialisation. The corpus was `max_chars: 100000`.

So the program's one sequence experiment ran the initialiser, on a hundred
thousand characters, with two-thirds of its classes below the sample floor, and
never exercised the temporal machinery it was built to test. **The task-
generality question is open. It has not been answered negatively; it has not
been answered at all.**

**2.7.2 The ordering that survives is the one that favours sparsity.** Whatever
the adequacy defect does to the absolute numbers, it applies to every arm on
identical rows. The ordering is `n-gram > linear > geometric > unigram`, and the
**n-gram won by 14.2 points over the geometric model and 13.0 over the linear
one**. An n-gram is the sparse, additive, count-based, exactly-inspectable model
in this comparison. On the program's own single attempt at a sequence task, the
most interpretable model available was also the most accurate.

That is a reason to run the milestone, not a result. §8.8 records that this
ordering is corroborated in the forecasting literature and **reversed** in the
language-modelling literature, and §7.6 registers M101 on the side where the
evidence says a sparse model can actually win.

Finally, `RESEARCH_REPORT_v5.md` §10 already proposed *"Experiment E: task-native
temporal representations"* to remove the representation confound, and it was
never run. M101 is that experiment, scoped down to what one milestone can carry.

### 2.8 Where the compute actually is, and what a cascade could reach **[new in v15]**

§2.6.2 establishes that the head controls under 0.05% of inference compute. That
kills head sparsity as an efficiency lever but says nothing about what *would*
be one. This subsection measures the next candidate: **spending compute in
proportion to how hard each input is**, rather than spending the same 6.07 GMAC
on every input.

**Status: scoping observation, not an operand.** The figures below come from a
v15 planning probe on the sealed features under a **different split** (seed-11
permutation, 448 train / 128 test per class) and a **different kNN
configuration** (k = 10, inverse-distance weighted) from M81's. Its `knn`
balanced accuracy reads **0.6357** where M81's sealed arm reads 0.661255. The
ordering is consistent; the numbers are not interchangeable, and under §2.4's
convention nothing here is admissible until a milestone reproduces it.

**2.8.1 The measurement.** A cheap stage-1 model — nearest class mean in the
top-*k* principal subspace — classifies every test input and emits a confidence
margin. The least-confident fraction *p* is deferred to the full weighted-kNN
stage-2. Three deferral rules are compared at each *p*: the **confidence**
margin, a **random** deferral of the same size (the R5 structure-matched null),
and an **oracle** that defers exactly the inputs stage-1 gets wrong.

| stage-1 | defer | oracle | confidence | random null | gate recovers |
| --- | ---: | ---: | ---: | ---: | ---: |
| NCM(16) | 75% | 0.6535 | 0.5722 | 0.5340 | 32.0% |
| NCM(32) | 50% | 0.5990 | 0.5414 | 0.4934 | 45.4% |
| NCM(32) | 75% | 0.6638 | 0.6063 | 0.5677 | 40.1% |
| NCM(64) | 40% | 0.5980 | 0.5485 | 0.5020 | 48.5% |
| **NCM(64)** | **50%** | **0.6432** | **0.5762** | **0.5228** | **44.4%** |
| NCM(64) | 75% | 0.6649 | 0.6220 | 0.5792 | 49.9% |

"Gate recovers" is `(confidence − null) / (oracle − null)`: the fraction of the
achievable gain that the confidence signal actually captures.

**2.8.2 Two readings, and the second is the important one.**

**Reading 1 — the difficulty skew is real.** At 50% deferral with a 64-dimensional
nearest-class-mean stage-1, the **oracle cascade reaches 0.6432 against the full
model's 0.6357**. Roughly **half the inputs on this corpus do not need the
expensive model at all**, and the accuracy is not merely preserved but slightly
improved, because stage-1 is right on some inputs where kNN is wrong. Compute
proportional to difficulty is not a hopeful story here; the headroom is measured.

**Reading 2 — the gate, not the model, is the binding constraint, and a better
stage-1 does not fix it.** The confidence rule captures **32% to 50%** of the
available headroom at the deferral rates that matter, and **44.4%** at the
crossover point. Over half the saving is left on the table by the abstention
signal. The natural response — make stage-1 more capable — was measured and
**does not work**:

| deferral | NCM(8) | NCM(16) | NCM(32) | NCM(64) | fraction | absolute gap |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 25% | 96.4% | 73.1% | 63.4% | 52.8% | falls | widens |
| 40% | 74.9% | 62.0% | 53.3% | 48.5% | falls | widens |
| 50% | 49.1% | 49.6% | 45.4% | 44.4% | non-monotonic | widens |
| 75% | 31.7% | 32.0% | 40.1% | 49.9% | rises | non-monotonic |

Each cell is the recovered fraction of oracle gain; the last two columns state
the direction as stage-1 capacity grows from 8 to 64 dimensions.

**Going from 8 to 64 dimensions makes stage-1 substantially more accurate — 0.1172
to 0.4150 balanced accuracy, a factor of 3.5 — and makes its deferral decisions
relatively *worse* at 25%, 40% and 50% deferral, while widening the absolute gap
to oracle at every one of those rates.** The absolute gap at 50% deferral grows
monotonically from 0.0214 to 0.0670 as stage-1 improves.

This is the finding, and it is not the one the section was expected to produce.
**Accuracy and self-knowledge are separate capabilities, and on this evidence
they are not merely distinct but mildly opposed under a fixed confidence rule.**
Scaling the cheap model buys accuracy it cannot convert into deferral quality.
That is why H110 (§4.3) targets the *objective* rather than the capacity: the one
intervention the probe rules out is the one that would otherwise be tried first.

**2.8.3 This program has already documented the same defect, from the other
side.** `RESEARCH_REPORT_v5.md` §6 records *"Raw geometry is poorly
calibrated"* and *"Raw SDF is weak for OOD — maximum calibrated probability
beats raw/corrected field distances; FPR95 remains high"*, concluding that
*"bounded geometry in the current feature space is not reliable open-space
evidence"*. Those are statements about the same quantity §2.8.1 measures: the
model's estimate of its own reliability. The program observed the defect, filed
it under calibration and OOD, and never connected it to compute.

**2.8.4 What follows for v15.** Three things, and they are the reason §3.2.1 and
M102 exist.

1. Accuracy on this representation is capped at ≈0.66 by §2.2 Reading 1, and
   fourteen versions of head engineering have confirmed it. **Accuracy is the
   axis this program cannot move.**
2. Abstention quality is **not** capped — it is measured here at 44% of oracle,
   with 56% unclaimed — and it is the axis that converts directly into compute,
   because every correctly-withheld expensive call is 6.07 GMAC saved.
3. **No milestone in this program's history has ever optimised abstention as a
   primary objective.** It has only ever been measured as a by-product of
   accuracy work, under the names "calibration" and "OOD".

**Registered limitation, stated plainly.** Every arm in §2.8.1 consumes the
384-dimensional DINOv2 feature, so every arm pays the full trunk cost and this
probe demonstrates **no systems saving whatsoever**. It measures the statistical
precondition — is difficulty skewed, and is it cheaply detectable — and that is
all. Converting the precondition into compute requires a stage-1 that does not
invoke the trunk, which requires raw images and is registered as M102.

**2.8.5 Superseded by M102, and the defect that superseded it. [recorded after execution]**

M102 (§7.7) has now run, and it **contradicts the arithmetic of this section
while confirming its direction**. §5.10 requires that a superseded registration
be contradicted in place rather than deleted, so §2.8.1–§2.8.4 are left standing
above and corrected here.

**The defect.** This probe's oracle deferred stage-1's *errors* first, chosen at
random among them. That is an upper bound only when the error count fits inside
the deferral budget. Every stage-1 arm in §2.8.1 has an error rate **above** its
deferral rate — NCM(64) is 0.585 wrong at a 0.50 budget — so the rule was not an
upper bound, and a real gate could beat it by preferring the errors stage-2 can
actually fix. M102 caught this when arm (c) at 8 dimensions read **141.4%** of a
quantity that was supposed to cap it. M102's corrected oracle ranks rows by the
benefit of deferring them (+1 stage-1 wrong and stage-2 right, 0 no change, −1
stage-1 right and stage-2 wrong) and spends the budget on the highest benefit
first, which is exactly optimal on an exactly-balanced evaluation split and is
therefore a true bound. The runner now **raises** rather than reports if any arm
beats the oracle at any point.

**What changes.** Every recovered fraction in §2.8.1 is **too high**, because
its denominator was too small. The **44.4%** that anchors H110's bar is not
reproducible: the corrected reading at the same operating point is **30.6%** for
the best gate. The gap between what a cascade could reach and what a gate
delivers is therefore **wider** than this section claimed, not narrower.

**What survives, and is strengthened.** Reading 1 holds and is larger than
stated. §2.8.2 claimed the oracle beats the full model at 50% deferral; M102
measures the corrected oracle at 32 dimensions reaching **0.6976** against the
full model's **0.6322** at **every** rate tested including **25%**. On the
corrected instrument, **at least three quarters** of inputs provably need no
expensive model, not one half.

**What is void.** §2.8.1's headline row is NCM(64). At M102's 448 fit rows per
class a 64-dimensional class mean has **7 fit samples per fitted dimension**,
below the floor of 10 that §5.3 never waives. Under M83.1/N83.8 that arm is
**void, not negative**, and the probe took its headline from it. M102's primary
operating point is 32 dimensions (14 fit samples per dimension), which passes.

**Reading 2's direction survives its arithmetic.** The claim that gate quality
does not improve with stage-1 capacity was measured on the defective oracle. On
the corrected instrument the picture is different in detail and unchanged in
consequence: recovered fraction now generally *rises* with capacity for the
margin gates, but no gate at any dimension or rate reaches even **41%**, and the
best reading anywhere in M102 is **35.3%**. The binding constraint is still the
gate.

---

## 2.9 What was measured after M102, and where it leaves the compute question **[new — recorded after execution]**

M102 refuted H110 and closed Lever 1's *gate* route. Three further probes were
run to establish whether Lever 1 had any other route, and whether Lever 3 is
worth a milestone. **All three are scoping observations under §2.4 and
prohibition 21's convention: they are unsealed, they are not milestones, and no
figure below is admissible as an operand.** They are registered here because
§5.10 requires that the evidence a redirect rests on be visible, and because two
of them contradict conclusions this plan and its ledger previously drew.

Runners are preserved outside the repository as session artifacts:
`_probe_complementarity.py`, `_probe_curve.py`, `_probe_m99_sweep.py`,
`_macs_patch.py`.

### 2.9.1 Lever 1 is closed a second way: the oracle gap is not reachable without labels

M102's corrected oracle beats the full model at every deferral rate (§2.8.5),
which means the cheap and the expensive model are **complementary** — the cheap
model is right on rows the expensive one gets wrong. A gate is only one way to
exploit that. The other is to **combine** the two models rather than choose
between them, which needs no correctness labels at all.

Measured on the sealed v13 corpus under M102's exact protocol (448 fit / 64
evaluation rows per class, inverse-distance weighted kNN at k=10, balanced
accuracy, seeds 11/23/37), combining L2-normalised vote vectors from a PCA
stage-1 and the full 384-d model:

| stage-1 dims | cheap alone | full model | **sum of normalised votes** | oracle |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.5700 | 0.6322 | **0.6257** *(−0.0065)* | 0.6727 |
| 64 | 0.6135 | 0.6322 | **0.6338** *(+0.0016)* | 0.6691 |

**The label-free combination captures almost none of the available gap** — 0.16
points of roughly 3.7 at 64 dimensions, and it is *negative* at 32.

**Registered reading.** Two independent mechanisms — a trained gate (M102, five
arms) and a label-free ensemble (here) — both fail to convert the oracle
headroom. The most economical explanation is that the oracle gap is dominated by
**chance disagreement between two models of similar accuracy**, which an
omniscient selector can harvest and no realisable rule can. §3.2.1 and §4.3
called this headroom "the largest measured unclaimed efficiency headroom
anywhere in the program's record"; that description is **withdrawn**, and the
sections are contradicted in place per §5.10 rather than edited.

**What this does not establish.** That no cascade works on any corpus, or that
published cascade results (§8.9 D3) are wrong. Those systems cascade between
models of very *different* accuracy, where competence is closer to nested than
complementary. This program's stage-1 and stage-2 differ by 6–13 points, and the
finding is about that regime only.

### 2.9.2 The accuracy-versus-trunk-compute curve, and why half of it is void

Q2's remaining lever is the trunk (§3.2.1 Lever 3), and the program had never
measured what trunk compute buys. The v14 M91 backbones supply three points on
the same corpus with row identity already verified upstream (N91.2).

Trunk MACs are computed as `L · (12·N·d² + 2·N²·d)` with `N = 257`, which
reproduces the registered v13 figure of **6,065,759,232** exactly for ViT-S/14
and is therefore applied unchanged to the other two.

| backbone | dims | GMACs | relative | balanced accuracy | INT8 divergence vs fp32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dinov2-small | 384 | 6.07 | 1.00× | 0.6322 | 0.315 |
| dinov2-base | 768 | 23.05 | 3.80× | **0.5177** | **1.194 — VOID** |
| dinov2-large | 1024 | 80.86 | 13.33× | 0.6652 | 0.481 |

**The dinov2-base row is void, not negative.** All three feature sets are INT8
ONNX exports, and M91's own N91.6 divergence control — recorded in the sealed
v14 manifests before this probe existed — measures base's mean relative
divergence from its fp32 original at **1.194**, i.e. the quantisation error
exceeds the signal. M91 reached the same anomaly independently and its recorded
reading already named quantisation as the place to look. An arm whose instrument
is damaged more than the effect it measures carries no verdict.

**What survives.** One directional reading: **13.33× trunk compute buys +3.3
points** of balanced accuracy. Because dinov2-large is *more* INT8-damaged than
dinov2-small (0.481 against 0.315), this understates the true gap, so it is a
**lower bound** on returns and an **upper bound** on how badly they diminish.

**Registered consequence for M100.** M100 was to produce the cost ledger that
§10.2's new gate depends on. These features **cannot** supply it: quantisation
damage is confounded with model size and is not monotone in it. A usable curve
needs either fp32 extraction or a cheap point below dinov2-small, and the v13
source images are **not on disk** — the selection manifest resolves to a
HuggingFace parquet corpus that is no longer cached. §7.9 and §10.2 are written
around that constraint rather than assuming it away.

### 2.9.3 A backprop-free representation reaches 0.63 on CIFAR-10, and learning the dictionary *hurts*

§10.2's M99 is the only milestone that touches the 99.95% of compute the trunk
holds, and it had never been probed. A Coates-style patch pipeline was run on
CIFAR-10 (cached locally): 6×6 patches at stride 2, per-patch contrast
normalisation, ZCA whitening, triangle encoding, 2×2 sum pooling, multinomial
logistic head, 20,000 train / 10,000 test rows, seed 11.

Per §10.2 restriction 3 the informative null is a **random** patch dictionary of
identical size, because Thiry et al.'s dictionary is itself drawn from data
without learning. Both arms were run at every budget:

| atoms | random patches (the null) | k-means (Coates et al.) | learning gain | MMACs/image |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.5240 | 0.5169 | **−0.0071** | 3.6 |
| 128 | 0.5614 | 0.5525 | **−0.0089** | 5.0 |
| 256 | 0.5819 | 0.5800 | **−0.0019** | 7.7 |
| 512 | 0.6129 | 0.6059 | **−0.0070** | 13.1 |
| 1024 | 0.6339 | 0.6223 | **−0.0116** | 24.0 |

**Random atoms beat learned atoms at every budget, and the gap widens with
size.** Accuracy rises monotonically with atom count and has not saturated at
1024.

**Prior art — this reproduces a published result and does not discover one.
[recorded after execution, correcting this section as first committed]** Thiry
et al. (2021), already registered as this milestone family's bar in §8.5, state
in their introduction that initialising with whitened patches "leads to a
significant improvement of performances, compared to a random initialization, a
wavelet initialization **or even a learning procedure**." The ordering measured
above is therefore **theirs**, and the probe reproduces it on a smaller
pipeline. This section as first committed presented the ordering as the probe's
striking observation and did not name the source. That was a prior-art
disclosure gap under §8.6 and §11.2 item 7, and it is corrected here rather than
edited away, per §5.10. Nothing in this plan may describe random-beats-learned
as a finding of this program.

**Why this matters to the plan's central thesis.** This is the first structure
the program has measured that behaves the way §3.1 assumes a grown model should:
each atom is free, capacity is added one atom at a time, accuracy rises
monotonically with the count, and the whole representation is built without
backpropagation at **24 MMACs per image** — three orders of magnitude below the
v13 trunk's 6.07 GMACs, on a different corpus and therefore never comparable to
it (§10.2 restriction 1, R7).

**Registered consequence — it reframes M99's question and strengthens M99's
null.** §10.2 asks whether additive residual-driven construction reaches the bar
with **fewer** patches than a fixed dictionary. This probe says the comparator to
beat is not k-means, which is *worse* than random here, but **random selection
itself**. That is a harder null and a more interesting question: if atom
*choice* barely matters and only atom *count* does, that is a substantive
negative result about sparse dictionaries, and it is reportable under §11.1.
**[superseded by §2.9.4, which measures that atom choice does matter when the
criterion is discriminative. Retained per §5.10.]**

**Five limitations, registered.** (i) Single seed — the random-beats-learned
ordering is monotone across five budgets but has not been replicated across
seeds, and §7.9 requires that before it carries anything. (ii) Subsampled: 20k
of 50k training rows, stride 2 not stride 1, so the accuracies are **below** what
this family reaches and are not comparable to the 0.869 bar. (iii) The logistic
head did not converge at 400 iterations at the larger budgets. (iv) MAC figures
exclude patch extraction, normalisation and pooling. (v) CIFAR-10 accuracy is
**not comparable to any v13/v14/v15 DomainNet figure in either direction** —
10 classes against 128 — and the numerical proximity of 0.6339 here to 0.6322 in
§2.9.2 is a coincidence with no meaning.

### 2.9.4 Atom choice does matter — when the criterion is discriminative **[recorded after execution]**

§2.9.3 admits two readings that point in opposite directions, and the plan as
first amended registered M103 against the wrong one.

* **(A) Atom choice does not matter, only atom count does.** The plan's central
  thesis is then dead for dictionaries, because additive residual-driven
  selection is a form of choosing and choosing does not help.
* **(B) k-means chooses badly because it optimises reconstruction, not
  discrimination.** Under this reading k-means losing to random is *predicted
  by* the thesis, and a criterion aimed at the labels should beat both.

A fourth probe discriminates between them, on the §2.9.3 pipeline unchanged.
The selection arm is group orthogonal matching pursuit over a random candidate
pool of 1,536 atoms, on a 4,000-row subsample of the **training split only**:
the residual starts as the centred one-hot label matrix, each candidate atom
owns a block of four pooled columns, the atom whose block best explains the
current residual is selected, and that block is regressed out so the next atom
is chosen against what the selected ones leave unexplained. Atoms are added one
at a time against a discriminative residual, which is §3.1's construction
principle with the model linear and the component an atom.

| atoms | random | k-means | **discriminative** | gain over random | random seed spread |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.5521 | 0.5543 | **0.5813** | **+0.0292** | 0.0078 |
| 256 | 0.5881 | 0.5798 | **0.6041** | **+0.0160** | 0.0053 |

Three seeds (11, 23, 37) at each budget. **All six seed-budget cells favour the
discriminative arm, by 3.0–3.7× the random arm's own seed spread.** This is
reading (B).

The gain **shrinks as the budget grows**, +0.0292 to +0.0160, which is the
expected shape: selection matters most when atoms are scarce, and scarce atoms
is the low-compute regime the plan cares about.

**The same result read as compute.** The discriminative arm at **128** atoms
reaches 0.5813; the random arm reaches 0.5521 at 128 and 0.5881 at 256, so
0.5813 sits about 81% of the way between them — roughly **232 random atoms**,
i.e. **≈1.8× fewer atoms at matched accuracy**. Encoding cost is linear in atom
count, so that is a direct saving on the dominant inference term. **Stated
against the claim:** selection required encoding 1,536 candidates over 4,000
rows, a *training* cost the random arm does not pay at all. §5.11 puts these in
different columns and they are not netted here or anywhere.

**The mechanism, measured rather than asserted.** Norms of whitened atoms in
256-atom dictionaries:

| dictionary | mean ‖atom‖ | median | 5th pct | 95th pct |
| --- | ---: | ---: | ---: | ---: |
| random patches | 1.296 | 1.265 | 0.369 | 2.292 |
| k-means centroids | 1.209 | 1.179 | **0.481** | **2.064** |
| the data itself | 1.248 | 1.193 | 0.372 | 2.308 |

Random draws inherit the data's distribution almost exactly. **k-means centroids
are compressed at both tails** — the 5th percentile rises from 0.372 to 0.481 and
the 95th falls from 2.308 to 2.064. Centroids are averages, and averaging pulls
toward the mean and destroys the tail. This is a mechanism for why
reconstruction-driven selection is no better than chance here and why a
label-driven criterion is different in kind rather than merely better tuned. It
is a description of two dictionaries, not a test of that mechanism, and §7.9
does not treat it as one.

**What this changes.** §7.9 and §10.2 registered M103 with the program's own
scoping evidence predicting it would fail. That prediction is contradicted in
place in both sections. What does **not** change: M103's design, its two kill
switches, its sample floor, and its acceptance criterion, all of which were
registered before this probe ran and none of which is loosened by it.

**Limitations, registered.** (i) Unsealed, single pipeline, inadmissible under
§2.4 and §11.2 item 23. (ii) Two budgets only, and the gain is already shrinking
across them — it may vanish by 1024, which is exactly what M103 measures.
(iii) The candidate pool is 1,536, so the arm selects from a restricted set and
its advantage may be an artefact of pool size rather than of the criterion;
M103's arm (a) null must be drawn from the same pool for this reason.
(iv) Subsampled exactly as §2.9.3 was, so the absolute figures are below what
the family reaches and are not comparable to any external bar. (v) CIFAR-10
figures are never comparable to any DomainNet figure (§11.2 item 24).

### 2.9.5 The registered Thiry bar did not re-verify **[recorded after execution]**

§8.5 registers **0.869** as the Thiry et al. linear-head CIFAR-10 bar, and
§10.2 restriction 2 requires bars to be re-verified before the milestone runs,
with any discrepancy recorded as an amendment rather than absorbed silently.
Re-verification against the authors' official repository gives:

| setting | figure |
| --- | ---: |
| 2K patches, linear head | **0.8232** |
| 16K patches, linear head | **0.8562** |
| 2K patches, one hidden layer | **0.8853** |
| ImageNet-64, 2K patches, linear head | 0.3321 top-1 |

The one-hidden-layer figure reproduces §8.5's registered **0.885**. **The
linear-head figure does not reproduce 0.869 at either dictionary size.**

**Registered consequence.** The 0.869 bar is marked **unconfirmed** pending a
reading of the paper's own tables, which this search could not access. No M99 or
M103 arm may be reported as having reached or missed "the Thiry bar" until that
reading happens and the figure is restated here.

**Why this does not block M103.** M103's primary operand is internal — the atom
count at which each arm reaches **arm (a)'s** accuracy at a fixed budget — and
external bars enter only through §7.9 design item 4, the instrument-correctness
check. A milestone whose verdict depended on an external figure would now be
blocked; M103's does not, and that was a property of its registration rather
than a repair made afterwards.

### 2.9.6 The M103 instrumentation run, and a defect it found in M103's own instrument check **[recorded after execution]**

Before M103 was launched, its runner was executed at **one cell** — 1024 atoms,
seed 11, the full 50,000/10,000 split at stride 1 — to size the sealed run's
wall-clock and to exercise the code end to end. It is an instrumentation run,
not a milestone: single seed, single budget, unsealed, inadmissible under §2.4
and §11.2 item 23. It is recorded here because it changed four things and one of
them is a defect in a registration.

| arm | accuracy at 1024 atoms |
| --- | ---: |
| (a) random patches | 0.6818 |
| (b) k-means | 0.6856 |
| (c) discriminative selection | **0.6951** |
| (d) random projections | 0.6760 |

**First — §2.9.4's open question resolves in the direction §2.9.4 predicted but
could not establish.** §2.9.4's gain was shrinking with budget (+0.0292 at 128,
+0.0160 at 256) and its registered limitation (ii) was that it might vanish by
1024. At 1024 atoms and full scale it is **+0.0133** over random and still
positive. One seed. M103 is what settles it, and this changes none of M103's
bars.

**Second — §2.9.3's ordering reverses at full scale.** k-means (0.6856) beats
random (0.6818) here, the opposite of §2.9.3's reading at every budget.
§2.9.3 used 20,000 rows, stride 2 and a head that did not converge; relaxing all
three flips the sign. §2.9.3's ordering is therefore weaker still as a basis for
anything — it was already someone else's published result rather than this
program's (§2.9.3 prior art), and it now appears to be conditional on a relaxed
instrument as well.

**[contradicted after execution — this finding was single-seed and did not
survive M103. Retained per §5.10.]** M103 ran the same cell under seal at three
seeds and **reproduced seed 11 exactly** — 0.6818 / 0.6856 / 0.6951 / 0.6760 to
four decimals — so the disagreement is not an implementation difference. It is
that this paragraph read **one seed**. Across three seeds arm (a) beats arm (b)
at **four of the five readable rungs**, 1024 included (0.6879 vs 0.6839).
§2.9.3's random-beats-k-means ordering therefore **survives under seal**, and
the reversal claimed above does not. Recorded in `CLAIM_LEDGER_v15.md` C103.5.
The lesson is the one §7.9 design item 3 already registered before any of this:
a single seed does not carry an ordering, and this section published one inside
the plan's own text.

**Third — the registered instrument check in §7.9 design item 4 cannot be
satisfied by any run.** §8.5's Coates anchor is **0.796 at 4000 features**;
M103's top rung is 2048 atoms and its top *readable* rung is 1024, which the
sample floor fixes and which §7.9 restriction 4 registered before any of this.
The accuracy-versus-atom-count curve does not reach 0.796 there and cannot. As
written, item 4 would declare the instrument broken on every possible run,
which makes it a check that carries no information. Worse, it made an external
figure a **pass/fail gate**, and R7 states that external figures are anchors and
never operands. The defect is corrected in §7.9 below, before M103 runs, and the
correction **tightens** conformance to R7 rather than relaxing a bar.

**Fourth — the atom-norm mechanism of §2.9.4 reproduces at full scale and
extends.** Whitened atom norms in the 1024-atom dictionaries:

| dictionary | mean | median | 5th pct | 95th pct |
| --- | ---: | ---: | ---: | ---: |
| (a) random patches | 1.248 | 1.212 | 0.342 | 2.285 |
| (b) k-means | 1.240 | 1.205 | **0.500** | **2.119** |
| (c) discriminative | **1.734** | 1.683 | **1.032** | 2.683 |
| (d) random projections | 1.240 | 1.209 | 0.357 | 2.209 |

k-means is compressed at both tails exactly as §2.9.4 measured at 256 atoms.
Arm (d) tracks arm (a), which is the intended effect of resampling its norms
from the patch pool so that it differs from arm (a) in direction only. And the
discriminative dictionary is **shifted upward at every quantile** — its 5th
percentile, 1.032, sits above arm (a)'s median. A criterion aimed at the labels
selects the high-energy tail, which is precisely the tail averaging destroys.
This is a description of four dictionaries and not a test of that mechanism.

**Disclosed against the numbers above: the head's regularisation grid was
truncated.** The grid was `{0.003, 0.01, 0.03}` and **0.03, its top value, won
for all four arms**, so every accuracy in this section is below what this family
reaches. The grid was extended upward before the sealed run. That change was
made **after seeing data** and is recorded as such rather than presented as the
original design.

### 2.9.7 Effective rank as a sizing instrument, and a 6.3× spread the plan did not know about **[recorded after execution]**

Four scoping probes were run after M103's prior-art audit (§8.10). Like every
other probe in §2.9 they are **unsealed, single-run, inadmissible as operands
under §2.4**, and bound by prohibition 23. They are recorded because they
motivate three new milestones and because one of them **refutes a hypothesis
this plan's own author formed one turn earlier**.

The instrument is **RankMe** (Garrido, Balestriero, Najman & LeCun, ICML 2023,
§8.10), used unmodified: the exponentiated Shannon entropy of the normalised
singular-value spectrum. It is label-free and has no hyper-parameters. It was
*not* built for this program — see §8.10 and prohibition 25.

**Probe 1 — effective rank grows as `atoms^0.455`, and does not saturate.**
M103 arm (a) on CIFAR-10, 10,000 rows, seed 11:

| atoms | ambient | RankMe | useful fraction of ambient |
| ---: | ---: | ---: | ---: |
| 64 | 256 | 37.194 | 0.14529 |
| 128 | 512 | 49.055 | 0.09581 |
| 256 | 1024 | 72.344 | 0.07065 |
| 512 | 2048 | 97.907 | 0.04781 |
| 1024 | 4096 | 129.829 | 0.03170 |
| 2048 | 8192 | 177.822 | 0.02171 |

Rank rises by ×1.319–1.475 per doubling with **no saturation** — the
least-squares exponent on `log rank` versus `log atoms` is **0.4553** — while
the useful fraction of ambient dimension falls as `atoms^−0.545` and collapses
**6.69×** across the sweep. **This refutes the hypothesis that rank saturation
explains C103.3's narrowing margin**; the hypothesis was formed before the probe
and the probe killed it. Recorded as a refutation because §5.10 requires it.

**Probe 2 — specialising by class buys almost nothing.** Per-class CIFAR-10
representations at 512 atoms reach a mean rank of **63.856** against a
row-matched control mean of **70.118** — a ratio of **0.9107**. The control
design matters: a specialist sees fewer rows, and fewer rows lower rank by
themselves. The three controls span 0.63 RankMe while the specialist gap is
6.26, so the effect is real and **small**. A network of class-specialists is not
worth building.

**Probe 3 — specialising by data type spreads rank 6.3×, and the mean is the
wrong statistic.** Per-domain DomainNet representations at 512 atoms, 1,000 rows
per domain, against three row-matched controls averaging **47.247**:

| domain | RankMe | ratio to control |
| --- | ---: | ---: |
| infograph | 55.286 | 1.170 |
| clipart | 55.164 | 1.168 |
| painting | 52.585 | 1.113 |
| real | 52.558 | 1.112 |
| sketch | 22.460 | 0.475 |
| **quickdraw** | **8.752** | **0.185** |

Mean ratio **0.8706** — **which conceals the finding entirely**. The spread from
infograph to quickdraw is **6.32×**, quickdraw alone sits **5.40× below** the
mixed control, and **four of six domains sit above it**. Checked against the
obvious artifact: near-white pixel fraction is 0.798 for quickdraw against 0.025
for real, per-image pixel std 0.101 against 0.203, and the pixel-complexity
ordering tracks the rank ordering exactly. **Disclosed limitation:** this is
measured at 32×32, and a downsample plausibly penalises line art more than
photographs. The *direction* is not in doubt; the *magnitude* is unmeasured at
native resolution, and no milestone may quote 6.32× as a
resolution-independent figure.

**Probe 4 — four training-free scalars route, and they are stable under
growth.** Four quantities were read from each image's own patch spectrum, with
no labels and no fitting: RankMe, the power-law decay exponent `alpha`
(alpha-ReQ, §8.10), the root mean eigenvalue, and the top-10 spectral mass.

| domain | RankMe | alpha | scale | top-10 |
| --- | ---: | ---: | ---: | ---: |
| clipart | 43.52 | −2.838 | 1.665 | 0.820 |
| infograph | 47.74 | −2.705 | 1.493 | 0.819 |
| painting | 45.15 | −2.782 | 1.296 | 0.841 |
| quickdraw | 30.76 | **−1.247** | 0.972 | **0.693** |
| real | 41.93 | −3.002 | 1.475 | 0.846 |
| sketch | 31.40 | −2.824 | 1.038 | 0.798 |

Six-domain identification, chance 0.1667, held out: **4 intrinsic scalars →
0.49278**; 2048 pooled features → **0.69056**. Four numbers recover **71%** of a
2048-dimensional learned representation at **512×** fewer dimensions.

Then the question that decides the architecture. A router was fitted on domains
{0,1,2,3}, refitted on all six, and its assignments compared **for 1,199
held-out images of the original four** — inputs that did not change:

| router input | agreement | assignments that moved |
| --- | ---: | ---: |
| pooled features, 2048-d | 0.90659 | **9.341%** |
| intrinsic fingerprints, 4-d | 0.96664 | **3.336%** |

The intrinsic router is **2.80× more stable**, and structurally so: its features
are a function of the input alone, so growth cannot move them at all. Only the
4-dimensional decision rule is refitted.

**A distinction this probe forced, which the plan did not previously draw.**
Quickdraw is the outlier on `alpha` in the *flat* direction (−1.247 against
−2.7 to −3.0) but the outlier on population rank in the *low* direction (8.75
against ~47). These are different quantities and they disagree in sign:
**within-image** diversity is high for quickdraw because a mostly-blank image
has no dominant direction, while **across-image** diversity is low because
quickdraw images resemble each other. **Expert sizing is set by the
across-image quantity; routing is served by the within-image quantity.** M104
and M105 use different measurements for these two jobs, and conflating them
would size every expert off the wrong statistic.

**What these probes do not establish.** None of them measures accuracy. A rank
spread is not an accuracy win, a probe accuracy is not a system accuracy, and
an assignment-stability figure is not a demonstration that anything composes.
M104, M105 and M106 are registered precisely because these four probes cannot
answer their own questions.

**Provenance.** The four runners are in `experiments/tier4/rank_probes/` and
their outputs in `logs/results/v15/rank_probes/`, under an artifact index that
records each file's SHA-256. Probes 1 and 2 reuse the M103 runner's own
pipeline functions unchanged, so they measure M103's representation and not a
reimplementation of it. Every figure quoted above is recomputed from those
artifacts by `experiments/tier4/verify_v15_plan.py`, and each recomputation is
paired with a check that the plan's prose still quotes it.

---

## 3. Registered question

v15 registers **three** questions. Q1 is the plan's spine and was the whole of
v15 before A5 was settled. Q2 and Q3 are additions, and they are asked because
A5 (§6.2) converted "no dense networks" from a structural prohibition into two
measurable obligations. An obligation that is never measured is decoration.

### 3.1 Q1 — the frontier point (primary)

Given that

1. the accuracy ceiling on this representation is ≈0.66 and is reached by three
   unrelated nonlinear families (§2.2 Reading 1),
2. the free `knn` control already reaches 0.661255 inside the 10-atom
   explanation budget (§2.2 Reading 2), and
3. no fitted sparse head has come within 17 points of the accuracy floor while
   staying inside that budget (§2.2 Reading 3),

**Q1 asks whether a model that is _grown_ one component at a time against a
discriminative residual can occupy the point on the `(accuracy, explanation
length)` plane that no _fitted_ head has occupied: at or above 0.6112548828125
balanced accuracy at 128 classes, with at least 95% of decisions citing at most
10 atoms.**

And, as a prerequisite, whether the ≈0.66 ceiling is a property of the task, of
the backbone, or of the INT8 quantisation the program has been measuring through
since v13.

The question is deliberately narrow. It is not "can sparse additive models
compete in general". It is a single point on a plane, on a sealed corpus, with a
sealed floor and a sealed budget, both inherited rather than chosen.

### 3.2 Q2 — does sparsity buy efficiency at matched accuracy? **[new in v15]**

**Q2 asks whether a sparse additive model reaches a given accuracy at lower
measured whole-system cost than the best dense model that reaches the same
accuracy.**

Three things fix the meaning of that sentence, and all three are registered
before anything is measured.

**The comparator is matched on accuracy, not on size.** A sparse model that is
cheaper and less accurate has not answered Q2. Neither has a sparse model that
is more accurate and more expensive. The claim is a strict dominance claim at a
matched operating point, and §5.11.3 registers the matching rule.

**The accounting boundary includes the trunk.** §2.6.2 shows why this is the
whole question: a head on a frozen ViT-S/14 controls 0.0008% to 0.05% of
inference compute. Excluding the trunk would let v15 report a 64× "regression"
or a 21× "improvement" in head cost while total system cost moves by less than
a tenth of a percent. §5.11.1 therefore makes the whole-system figure primary
and the head-only figure a disclosed secondary.

**The honest prior is negative for everything M94–M98 will build.** §2.6 already
shows the sparse arms losing on parameters and on MACs, and §2.6.3 records that
the program's earlier conditional-compute attempt failed on wall-clock. P6 (§6.1)
registers this in advance. Q2 is asked anyway, because a registered negative
answer with the arithmetic attached is a result — and because §5.11.5 opens the
one door through which a positive answer could still arrive: **training and
sample cost, where the sparse arms have never been measured at all.**

#### 3.2.1 Where Q2 must be answered, and the redirect it forces **[new in v15]**

A registered negative is a result, but a plan that only produces one has stopped
being a research plan. §2.6 and §2.8 together say exactly where Q2 *can* be
answered affirmatively, and it is not where v15's milestones were pointed.

Three levers touch compute. Sparsifying the head is not one of them.

**Lever 1 — do not run the expensive model when it is not needed.** §2.8's oracle
cascade reaches **0.6432 against the full model's 0.6357 while skipping 50% of
the expensive calls**. The headroom is measured, not assumed. What blocks it is
the abstention signal, which recovers **44.4%** of that headroom *(this figure
is not reproducible — see §2.8.5; the corrected reading is **30.6%**, which
widens the gap rather than narrowing it)*. This is the largest measured,
unclaimed efficiency headroom anywhere in the program's record, and it is
registered as **H110** and measured by **M102**. **[recorded after execution:
H110 is refuted, this lever is closed, and the description "the largest measured
unclaimed efficiency headroom anywhere in the program's record" is
**withdrawn** — §2.9.1 measures that a label-free ensemble captures ~none of it,
so the headroom is best explained as chance disagreement an omniscient selector
harvests and no realisable rule can. The sentence is contradicted here rather
than edited, per §5.10.]**

**Lever 2 — do not train at all.** §2.1's sealed figures show `knn`, which fits
nothing, at **0.661255** against the fully-fitted `mlp_*` arms at **0.660522**.
On this corpus, at this representation, **fitting the head buys −0.000733
accuracy for its entire training cost.** If retrieval matches fitting wherever a
frozen representation is good, training compute for downstream tasks is largely
avoidable. Registered as **H111**.

**Lever 3 — make the representation cheaper.** The trunk is 99.95% of the cost,
and §8.7 A10 records that it is **31–55× more expensive** than mobile CNNs that
score *higher* on ImageNet. This is M99, and §10.2 previously gated it behind two
head-milestone outcomes. §10.2 changes that gate. **[recorded after execution:
with Lever 1 closed, this is the only remaining affirmative route to Q2, and
§2.9.2 shows that M99's replacement gate is itself blocked. §10.2's second
regate and the new unconditional milestone M103 (§7.9) exist to unblock it —
§2.9.3 records the observation they follow from.]**

**The redirect, registered.** v15's original milestone set spends M94, M96, M97
and M98 on the head. Those milestones answer Q1, which is a question about
**interpretability**, and they remain fully justified by it. But P6 makes it
knowable *in advance* that none of them can answer Q2. **A plan that left Q2
attached to milestones incapable of answering it would be asking a question it
had already arranged not to answer.** Q2 is therefore reassigned: M100 measures
and characterises the cost, M102 attacks Lever 1, H111 tests Lever 2, and M99's
gate is loosened for Lever 3.

**Registered concentration risk, stated against interest. [recorded after
execution]** With H110 refuted and Lever 1 closed by two independent mechanisms
(§2.9.1), Q2's affirmative case now rests on Lever 3 alone, reached through a
single chain: M103 → M99. Lever 2 (H111) is a *negative* efficiency result about
training if it confirms — it says fitting is unnecessary, not that inference is
cheaper — and M100 characterises cost without reducing it. **The program does
not add milestones to restore redundancy**, because §3.4.3 registers
multiplicity as the larger risk and §3.2 already registers that a well-supported
negative on Q2 is an acceptable outcome. This is recorded so that a Q2 negative
is read as the concentrated bet it was, not as a surprise. **[§2.9.4 supplies
the first affirmative scoping evidence anywhere on this chain. It does not
reduce the concentration — the chain is still a single chain — and no milestone
is added on the strength of an unsealed probe.]**

**What did not change.** Q1 remains primary (§3.4.1), its milestones are
unchanged, and its acceptance criteria are untouched. The redirect moves *the
efficiency question* to where the efficiency is. It does not move the plan's
verdict, which is still H102's alone.

### 3.3 Q3 — does additive construction extend beyond classification? **[new in v15]**

**Q3 asks whether the additive-against-a-residual construction that Q1 tests on
static classification also applies to sequential prediction, and how far behind
the appropriate published bar it lands.**

Q3 is a **generality** question, not a competitiveness question, and the
difference is registered so it cannot be blurred later. v15 does not ask whether
a grown sparse model can rival an LLM; §8.8 records verified evidence that it
cannot, and §11.2 item 13 forbids the claim. It asks the prior question of
whether the machinery **transfers at all** — whether "grow a component against
what the current model gets wrong" is a construction principle for sequence
tasks or an artifact of static classification.

The task family is chosen where the literature says a sparse or simple model can
actually win, so that a negative result is informative about the method rather
than about the choice of arena: **long-horizon multivariate forecasting**, where
Zeng et al. 2023 and Elsayed et al. 2021 both report simple models beating deep
sequence models (§8.8). Choosing autoregressive language modelling instead would
guarantee a loss that says nothing, because §8.8 C2 shows a 3.1× perplexity gap
between the best sparse count model and a Transformer on the same benchmark.

§2.7 records that the program's single prior sequence experiment is **void**
under its own sample-adequacy rule and never ran its temporal loop, so Q3 starts
from no admissible evidence in either direction.

### 3.4 How the three questions interact — and what may not leak between them **[new in v15]**

Adding questions to a pre-registered plan is dangerous in a specific way: it
multiplies the opportunities to report something positive, which is the failure
mode `ACCEPTANCE_CRITERIA_v13.md` §10 and R5 exist to prevent. Q2 and Q3 are
therefore registered as **separate axes of measurement, not as additional gates
on Q1**, and the four places where they could interfere are closed here rather
than negotiated later.

**3.4.1 Q1 remains the plan's sole success criterion.** v15 succeeds or fails on
**H102**. Q2 and Q3 produce findings; they do not produce a v15 verdict. No
summary, abstract or ledger entry may lead with a Q2 or Q3 result, and no Q2 or
Q3 result may be offered in place of Q1's answer. If H102 is refuted and H109 is
confirmed, v15's outcome is still Outcome C or D on the §11.1 taxonomy —
"additive construction transferred to forecasting" is a subordinate clause, not
a headline. §11.3 registers the reporting order.

**3.4.2 Efficiency is disclosed, not gated, for Q1.** A5's efficiency obligation
is discharged by **measuring and reporting** the §5.11.2 currencies for every
arm, not by any arm passing an efficiency threshold. There is no efficiency
floor an arm must clear to count toward Q1, and none may be introduced later.

This is registered emphatically because the opposite reading is available and
would be destructive: §2.6.2 shows the head controlling under 0.05% of
whole-system cost, so an efficiency threshold applied to Q1 arms would fail or
pass them for reasons that have nothing to do with whether additive construction
reaches the frontier point. **A gate that every arm passes or fails identically
is not a gate; it is noise with an authority.** The acceptance criteria for Q1
remain exactly L1′, L6 and the inherited L-series — unchanged from before A5 was
settled.

**3.4.3 One genuine new gate on Q1, and it is inherited rather than new.**
§5.11.4's mechanism ablation **does** gate Q1: an arm whose accuracy survives
ablation of its sparse component is reported as a dense model and does not count
toward Q1. This is not a new requirement. It is the measurable form of D7 and
§11.2 item 8, both of which already forbade delivering an explanation that is
not the computation. Before A5, that rule was enforced structurally — no dense
components existed, so nothing could hide behind the sparse part. A5 removes the
structural guarantee, so the rule now needs an operand. The gate applies **only
to arms that contain a dense component**; an arm with no dense component has
nothing to ablate and is unaffected.

**3.4.4 Ordering, so Q2 and Q3 cannot starve Q1.** M100 runs early because it is
cheap — it fits no new model, and it changes what the rest of the plan may claim.
M101 runs **after M96 and M97 have reported**, because it needs a new corpus and
a new harness and is the only expensive addition here. M101 is unconditional in
the sense that its value does not depend on Q1's answer, but it is **last in
order**. If the fit budget is exhausted before M101, M101 is reported as
`not_run` and no Q3 conclusion is drawn in either direction.

M102 (§7.7) is ordered **after M100 and before M101**. After M100, because M100's
cost ledger is what makes a cascade's saving measurable rather than asserted.
Before M101, because M102's Tier A reuses the sealed features and the existing
harness and is therefore cheap, whereas M101 needs a new corpus. If the budget is
exhausted before M102's Tier B, Tier B is reported as `not_run` on the same terms
as M101 — the gate-quality result stands, the compute result is simply absent,
and prohibition 20 already forbids describing the former as the latter.

**3.4.5 No cross-question rescue.** A hypothesis refuted on one axis may not be
rescued by a result on another. Specifically, and registered because each is a
tempting move: a Q2 efficiency finding may not be offered as mitigation for an
H102 refutation; an H109 confirmation on forecasting may not be offered as
evidence that the DomainNet shortfall is corpus-specific; and an H108 sample-
efficiency win may not be described as compensating for an accuracy shortfall
unless the accuracy shortfall is stated in the same sentence with its magnitude.

**3.4.6 Multiplicity is disclosed.** v15 now registers **twelve** hypotheses
across three questions — H100–H105 for Q1, H106–H108 and **H110–H111** for Q2,
H109 for Q3. That is more opportunities for a spurious positive than v14
carried, and the plan says so rather than leaving a reader to notice. Three
mitigations are registered: each question names one primary hypothesis (H102,
H107, H109) and the rest are explicitly secondary and non-promotable; every
hypothesis carries a structure-matched null under R5, so a spurious positive
must beat its own null and not merely a baseline; and §5.11.6 requires a
confirmation replay for the single result most likely to be an instrument fault,
which is an unexpected efficiency win.

**3.4.7 The v15 redirect does not widen the success surface. [new in v15]**
§3.2.1 moves Q2's centre of gravity to M102 and H110/H111, and §10.2 loosens
M99's gate. Both changes add reachable milestones, which is exactly the
multiplicity risk §3.4.6 names, so the containment is registered here explicitly
rather than left implied.

1. **No new hypothesis is promotable.** H107 remains Q2's primary hypothesis.
   H110 and H111 are secondary and non-promotable under §3.4.6, and prohibition
   17 already forbids promotion. A confirmed H110 is a `gate_improvable` record,
   not a Q2 primary result.
2. **No new hypothesis touches Q1.** H110 and H111 produce no operand admissible
   in any H100–H105 comparison. M102 fits no arm that competes in the M96/M97
   frontier, and §5.11.4 remains the only Q1-facing efficiency gate.
3. **The new hypotheses are harder to confirm than the ones they join, not
   easier.** H110's bar is set **above** the measured baseline (60% against
   44.4%, a figure §2.8.5 later found not reproducible — the correction moved
   the baseline **down** to 30.6%, making the registered bar harder rather than
   easier) and is fixed before M102 runs. H111 is refuted by a **single** fitted
   arm exceeding retrieval at **any** rung — a conjunction across the whole
   ladder, which is a stricter form than any other v15 hypothesis uses.
4. **M99's looser gate adds no operand.** M99 reports on a separate corpus
   against an external bar and §10.2 restriction 1 forbids comparing any M99
   number to any v13/v14/v15 number in either direction. Reaching M99 more often
   therefore cannot produce an additional way for v15 to declare success; it can
   only produce an additional external-bar report.
5. **The redirect is registered before any of it is measured.** §2.8's probe is
   marked inadmissible (prohibition 21) precisely so that the redirect rests on
   §2.6.2's arithmetic — which was registered before the probe existed — rather
   than on a number that flattered it.

---

## 4. Hypotheses and kill switches

Each hypothesis names the operand that refutes it. No hypothesis may be rescued
by substituting a corpus, relaxing the explanation budget, moving the accuracy
floor, or selecting an arm after seeing the operands.

**H100 — representational ceiling.** The ≈0.66 accuracy plateau is a property of
the representation, not of the head family. _Refuted if_ any head family
measured in M94 exceeds 0.673452 — the sealed best arm (0.663452) plus the
0.01 decisive margin — on the identical rows. A refutation means head choice is
still live and §3's premise fails.

**H101 — additive selection beats one-shot selection.** At a matched atom
budget, selecting atoms greedily against the **discriminative** residual, with
refit after each addition, beats selecting them in one shot by penalised
regression. _Refuted if_ M96's greedy arm does not exceed the matched
`sparse_linear_budget_*` arm by more than the seed spread, at every budget in
the registered sweep.

**H102 — growth reaches the frontier point.** A grown sparse additive model
reaches ≥ 0.6112548828125 balanced accuracy at 128 classes with ≥ 95% of
decisions citing ≤ 10 atoms. _Refuted if_ neither M96 nor M97 produces an arm
meeting both conditions simultaneously. This is the plan's primary hypothesis
and its refutation is a complete and reportable answer.

**H103 — quantisation depresses the ceiling.** The ≈0.66 ceiling is partly an
artifact of the INT8 backbone. _Refuted if_ an fp32 extraction of the identical
images at batch size 1 moves neither the linear probe bar (0.613037) nor the
`knn` bar (0.661255) by more than the registered decisive margin of 0.01.

**H104 — the oracle is the binding constraint.** Where greedy growth
underperforms, the cause is the atom-search step and not the additive form.
_Refuted if_ a dense teacher used **solely** as a direction-proposing oracle
(M98) fails to move the grown model's accuracy at matched budget by more than
the seed spread. **Opens only if H102 is refuted by M96 and M97.**

**H105 — the representation can be grown too.** A patch dictionary grown
additively against a discriminative residual reaches the verified fixed-patch
linear-head bar of **0.869** on CIFAR-10 (Thiry et al. 2021, §8.5) using
**fewer patches** than the fixed construction. _Refuted if_ the grown dictionary
needs at least as many patches as the fixed one to reach that accuracy, or fails
to beat a random-patch null of identical size.
**Opens only if H102 is refuted and H104 is confirmed**, and it is measured on a
separate instrument under §10.2's isolation rule.

### 4.1 Efficiency hypotheses **[new in v15]**

These answer Q2 (§3.2). All are measured under the §5.11 contract, and all
report the whole-system figure as primary.

**H106 — head sparsity is efficiency-irrelevant on a frozen dense trunk.**
Varying head sparsity across the full M81 arm set changes total inference cost
by less than **1%** in MACs and by less than the measurement noise in
single-threaded wall-clock, because the frozen trunk dominates. _Refuted if_ any
registered arm changes whole-system inference MACs by more than 1%, or moves
median batch-1 latency by more than **2×** the interquartile range of the
repeated-timing null (§5.11.2).

This hypothesis is registered **as expected to be confirmed**. §2.6.2 derives
0.0008%–0.05%. It is registered rather than assumed because confirming it is
what licenses the plan to stop asking Q2 of M94–M98 and ask it instead of M99,
M102 and H111 (§3.2.1), and because a plan that quietly dropped an inconvenient
question would be doing the thing §11.2 forbids. H106's confirmation is also the
regated opening condition for M99 (§10.2).

**H107 — sparsity does not dominate density at matched accuracy.** No sparse arm
reaches a matched accuracy at strictly lower whole-system cost than the best
dense arm at that accuracy, in any of the four registered inference currencies.
_Refuted if_ a single sparse arm is strictly cheaper in all four currencies at
an accuracy within the decisive margin of a dense comparator. Refutation would
be the first efficiency win in the program's history and would require the
§5.11.6 confirmation replay before it is written anywhere.

**H108 — sparse construction is more sample-efficient than dense fitting.** At
small labelled-sample budgets, the greedy additive arm from M96 reaches a higher
balanced accuracy than the dense MLP and the dense linear probe fitted on the
same rows. _Refuted if_ the additive arm fails to exceed both dense arms by more
than the seed spread at every budget in the registered ladder (§5.11.5).

H108 is the plan's **only structurally open efficiency question**, and it is
registered explicitly as such. Sample efficiency is the one currency in which
the trunk-dominance argument of §2.6.2 does not apply, because the trunk is
frozen and costs the same for every arm, so any difference is attributable to
the head's fitting procedure. It is also the currency in which the greedy
literature's theory (§8.1) makes an actual prediction. Note the direction of the
program's own weak prior: `knn` — which fits nothing — already reaches 0.661255,
which is what a low-sample regime rewards.

### 4.2 Generality hypothesis **[new in v15]**

**H109 — additive construction transfers to sequential prediction.** A model
grown one component at a time against a forecasting residual beats (a) its own
structure-matched null and (b) the persistence baseline, on the registered
forecasting corpus at the registered horizon. _Refuted if_ the grown arm fails
either comparison. Refutation means the additive construction principle does not
transfer out of static classification, which is a first-order finding about the
method and is reportable on its own.

**H109 is a transfer hypothesis and carries no competitiveness claim.** Beating
the verified `Linear` bar of MSE 0.140 / MAE 0.237 on Electricity at horizon 96
(§8.8 C1) is registered as a **secondary, non-gating** observation. It is not a
kill switch in either direction, because a single milestone on one corpus
against numbers produced by another codebase cannot settle competitiveness, and
R7 forbids treating the external number as an operand. §11.2 item 13 forbids
upgrading H109 into a claim about language modelling under any outcome.

### 4.3 Abstention and retrieval hypotheses **[new in v15]**

These answer Q2 through §3.2.1's Levers 1 and 2. They exist because §2.6 makes
the head-side efficiency answer knowable in advance and §2.8 locates the one
place where it is not.

**H110 — abstention quality, not accuracy, is what a sparse geometric model
contributes, and it is improvable.** A sparse model whose fitting objective is
**deferral quality** rather than accuracy recovers substantially more of the
oracle cascade headroom than an accuracy-fitted model's confidence margin.
_Refuted if_ the abstention-optimised gate fails to exceed the confidence gate's
recovered fraction by more than the seed spread, at every deferral rate in the
registered sweep — **or if the temperature-scaled arm (b′) reaches the bar
without it** (§7.7, §8.9 D6).

The bar is set from §2.8's measurement rather than chosen: the confidence gate
recovers **44.4%** of oracle-available gain at the 50% deferral crossover.
H110's registered target is **> 60%** recovery at the same operating point, and
the choice of 60% is registered now, before M102 runs, so that it cannot be
lowered to meet a result. Note that this is a bar on **gate quality**, not on
accuracy: an arm that improves accuracy and not deferral has not confirmed H110.

**[recorded after execution]** The 44.4% anchor did not survive. §2.8.5 records
that the probe's oracle was not an upper bound; the corrected baseline is
**30.6%** and no arm reached 41% at any width or rate. H110 is **refuted**
(§2.9.1, `analysis/CLAIM_LEDGER_v15.md` C102.1). The registration above is left
standing per §5.10 so that the bar can be seen to have been fixed in advance.

**H110 is not a novel idea and is not registered as one.** §8.9 D2 records that
SelectiveNet (2019), Madras et al. (2018) and Mozannar & Sontag (2020) already
train the gate jointly with the task, the last with consistency guarantees. What
is untested is whether this works for the **sparse geometric family** this
program builds, measured against an **oracle denominator** that §8.9 D5 could not
find reported anywhere. §11.2 item 22 forbids any stronger statement.

**Why this is the plan's most promising open question.** It is the only place in
the program's record where a large headroom has been *measured* and left
unclaimed. Accuracy headroom on this representation is ≈0 (§2.2 Reading 1,
fourteen versions of confirmation). Parameter and MAC headroom in the head is
≈0 (§2.6). Abstention headroom is **55.6%** of a lever worth 6.07 GMAC per
withheld call *(derived from the non-reproducible 44.4%; the corrected figure is
**69.4%** — §2.8.5)*. It is also the capability the program has repeatedly
observed to be broken — under the names calibration and OOD (§2.8.3) — without
ever making it an objective. §8.9 D3's verified external cascade results
(FrugalGPT's 80% cost reduction *with* a 1.5-point accuracy gain; CALM's ×3.53
wall-clock) are anchors showing the lever is real in other stacks, and are never
operands here.

**[recorded after execution — this paragraph's judgement did not survive.]** The
headroom is real and was confirmed larger than stated (§2.8.5), but §2.9.1
records that it was not reachable by any gate M102 tried **and** not reachable
by a label-free combination of the two stages. The description **"the largest
measured unclaimed efficiency headroom anywhere in the program's record" is
withdrawn**: a headroom only an omniscient selector can reach is not headroom in
the sense that phrase implies. "Most promising" was a judgement about an
unmeasured quantity; it is left standing per §5.10 and contradicted here.

**Why the objective, and not the capacity.** §2.8.2 Reading 2 measures the
obvious alternative and rules it out: scaling stage-1 from 8 to 64 dimensions
raises its accuracy 3.5× and *lowers* its recovered fraction at 25%, 40% and 50%
deferral while widening the absolute gap to oracle. H110 is therefore addressed
to the only remaining lever. This ordering — measure the cheap intervention,
find it fails, then register the expensive one — is stated so that H110 cannot
later be presented as the first thing tried.

**H111 — on a good frozen representation, retrieval matches fitting.** A
zero-training retrieval head is within the decisive margin of the best fitted
head at every rung of the §5.11.5 sample ladder. _Refuted if_ any fitted arm
exceeds the retrieval arm by more than 0.01 at any rung.

The sealed evidence for the full-data rung is already in: `knn` **0.661255**
against `mlp_*` **0.660522**, a difference of **−0.000733** in favour of not
training.

**The literature prior is negative, and H111 is registered as expected-refuted.**
§8.9 D4 verifies four published rows and **linear beats kNN in every one**, by
1.8 to 2.4 points — DINOv2 ViT-S/14 **0.811 vs 0.790**, DINO ViT-B/8 **0.801 vs
0.783**, iBOT ViT-B/16 **0.795 vs 0.771**. This program's sealed measurement runs
the other way, which is the §2.5 ordering inversion. Registering H111 as
expected-refuted while the program's own sealed number points the other way is
deliberate: it means a confirmation must overcome a stated prior rather than
merely agree with a house result.

**What the refutation would actually be about.** Because the published rows are
full-shot ImageNet-1k and §8.9's search found **no located result in either
direction** for low-shot or shifted regimes, the informative quantity is the
**rung at which the ordering flips**, not whether it flips. H111 is therefore
reported as a curve over the §5.11.5 ladder, and a refutation at the top rung
with a confirmation at the bottom is a **result**, not a mixed verdict — that
pattern would locate a regime boundary between retrieval and fitting, which is
the question §2.5 raised and could not answer.

**What confirmation would and would not license.** Confirmation licenses the
statement that on this corpus and representation, downstream training compute is
avoidable at no accuracy cost. It does **not** license any statement about
training the representation itself, which is where the overwhelming majority of
the field's training compute is spent and which this program does not touch.
§11.2 item 19 makes that explicit.

---

## 5. Shared contract

Carried forward from v14 §5 unchanged unless stated. Additions are marked
**[new in v15]**.

### 5.1 Corpus

`logs/results/v13/domainnet_large` at the sealed index hash
`a6485f9000654eebb6a9d06edd711f757ecc9f9c5764bc5b34bf91ec841bbf85`
(73,728 rows × 384 dimensions, 128 classes, 576 rows per class, class-major).
R7 stands: **no v12 or CIFAR-10 number is compared to a v15 number.** Every
gated v15 arm is measured on this corpus so that it is directly comparable to
the sealed v13 and v14 figures and to the other v15 arms.

### 5.2 Partition

N83.7's domain-quota split: 512 fit rows per class, evaluation quota
`[9, 7, 6, 39, 3, 0]`, 64 evaluation rows per class, calibration and report
halves domain-stratified. Unchanged. **[new in v15]** Where a v15 arm requires a
selection set distinct from the fit set — greedy selection with refit does —
the selection set is carved from the **fit** rows and never from the evaluation
rows, and the carve is registered per milestone before execution.

### 5.3 Sample adequacy

The standing floor of **10 fit samples per fitted tangent dimension** applies to
every component of every arm and is never waived. **[new in v15]** For arms that
fit no tangent subspace, the equivalent floor is **10 fit samples per free
parameter of the fitted component**, applied per class. An arm that cannot meet
its floor is not run at that setting.

### 5.4 R5 standing null contract

Every comparative operand carries a null sharing structure, budget and split.
An operand without its null is not evidence.

| arm                                       | null                                                                                             |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------ |
| greedy discriminative atom selection      | identical budget, atoms drawn uniformly at random from the identical dictionary, identical refit  |
| greedy discriminative atom selection      | identical budget, atoms selected against the **reconstruction** residual — isolates "discriminative" from "greedy" |
| grown atom (new direction per step)       | random direction at matched norm, identical budget, identical refit                              |
| grown atom                                | the M96 selected-atom arm at matched budget — isolates "grown" from "selected"                    |
| fp32 backbone                             | the identical arm on INT8 features, identical rows                                                |
| dense-teacher oracle                      | the identical growth procedure with a random-direction proposer, identical step count             |
| any tree or boosting arm                  | the identical estimator with permuted labels, identical hyperparameters                           |

**[new in v15] The two nulls on the greedy arm are both mandatory.** A greedy
discriminative arm that beats a random-atom null but does not beat a
reconstruction-residual null has demonstrated that greed helps, not that
discrimination helps, and must be reported that way.

### 5.5 Instrument validation

Every arm validates before any figure below it is read. For accuracy arms the
validation is: a **shuffled-label null** must land within the registered
tolerance of chance (0.0078125 at 128 classes), and a **known-split positive
control** — the sealed `knn` arm — must reproduce 0.6612548828125 exactly on
the identical rows. An arm failing either reports `instrument_invalid` and
suppresses every figure below it. An arm that fails its instrument is **void,
not negative** (M83.1 / N83.8 precedent).

### 5.6 Hyperparameter sweeps **[new in v15]**

Registered in direct response to §2.4 scoping observation B.

1. Any arm with a learning rate sweeps it over a registered grid of at least
   four values and **reports the entire sweep**. A single-setting reading of a
   learning-rate-sensitive estimator is inadmissible.
2. Any arm whose reported accuracy falls as its capacity rises is treated as
   **suspected defective** and is not reported until the monotonicity violation
   is explained or reproduced under a second setting.
3. Train and evaluation accuracy are reported **together** for every fitted arm.
   The v15 planning probes found a logistic head at 0.846313 train against
   0.568115 evaluation; a program that reports only one of those two numbers
   cannot distinguish a capacity problem from an overfitting problem.

### 5.7 Acceptance additions **[new in v15]**

`ACCEPTANCE_CRITERIA_v13.md` is inherited unchanged. Two additions:

- **L1′ — budgeted accuracy.** Accuracy is reported jointly with the fraction of
  decisions inside the 10-atom explanation budget. A bare accuracy figure for a
  sparse arm is not admissible in v15. This makes §2.2 Reading 3 impossible to
  repeat.
- **L6 — dominance over the free control.** Any claim that a v15 arm improves on
  the program's position must state its result against `knn` at
  `(0.661255, 6.72 atoms, 1.0000 within budget)` on identical rows. An arm that
  does not beat `knn` on **both** axes has not improved the program's position,
  whatever else it does. This is gating for any claim of progress and
  non-gating for reporting a frontier.

### 5.8 Determinism

Gated evidence is CPU-only, single-threaded torch, byte-identical replay, in the
frozen `.venv` (Python 3.14.6, numpy 2.5.1, scikit-learn 1.9.0, torch
2.13.0+cpu). Multi-threaded reduction ordering breaks byte-identical replay
(M80): where parallelism is used it is across processes, with one torch thread
inside each. The RX 9070 XT is permitted **only** for upstream feature
extraction under `.venv-rocm`, never for a gated computation, and any extraction
it produces is sealed as an artifact and verified by hash before use. DirectML
disagrees with the CPU provider by 11.7% at batch 1 and is not used at all.

### 5.9 Dictionary provenance **[new in v15]**

`logs/results/v13/m80_sparse_dictionary/` contains `evidence.json` and
`artifact_index.json` **only**. The 8192 atoms themselves were not persisted.
Any v15 milestone that selects from M80's dictionary must therefore
**regenerate** it from the sealed configuration
(`experiments/configs/v13/m80_sparse_dictionary.json`, seed 11, 40 epochs, batch
1024, learning rate 0.001, one torch thread per worker) and verify that the
regenerated dictionary reproduces M80's sealed cell figures — probe accuracy
**0.607910** at m=8192/k=32 and **0.583130** at m=8192/k=16 — before anything is
built on it. A regeneration that does not reproduce both reports
`not_m80_dictionary` and the milestone stops. The regenerated atoms are sealed
as a v15 artifact so that this is done once.

### 5.10 Disclosure

The I2 defect is carried forward and restated because v15 depends on it: the
frozen dinov2-small INT8 graph contains 49 `DynamicQuantizeLinear` operators, so
extracted features are a function of the **batch set**. All v13 extraction ran
at batch size 1. Reordering a batch changes nothing (0.000); changing batch
membership changes a great deal (1.210 at batch 32). Every v15 extraction runs
at batch size 1, and M95 exists partly because this defect means the program has
never measured its own ceiling on an unquantised trunk.

### 5.11 Efficiency accounting **[new in v15]**

Registered in full before any efficiency number is produced. §2.6.1 shows this
program's arms ordering differently in different currencies, so whoever chooses
the currency after seeing the numbers chooses the answer. This subsection
removes that freedom.

**5.11.1 Accounting boundary.** Every efficiency figure is reported twice:

- **Whole-system (primary).** The complete path from input image to emitted
  decision, including the frozen trunk, the feature transform, any sparse coding
  or dictionary projection, the head, and any calibration. This is the figure
  that discharges A5's efficiency obligation, because A5 says *overall*
  efficiency.
- **Head-only (secondary, disclosed).** The same path with the trunk excluded.

A head-only figure may never be reported without the whole-system figure beside
it, and no claim may rest on the head-only figure alone. Rationale: §2.6.2.

**5.11.2 The four inference currencies.** All four are reported for every arm
that carries an efficiency claim. No subset may be reported selectively.

1. **Stored parameters** — every value that must be persisted to reproduce the
   decision. For `knn` this includes the retained corpus; M81 already counts it
   this way at 25,165,824, and that convention is inherited unchanged.
2. **Multiply-accumulates per decision** — analytic, from the architecture, at
   the registered input resolution. The trunk figure is 6,065,759,232 MACs
   (§2.6.2) and is fixed for every arm that uses the sealed features.
3. **Peak resident bytes** — measured, single process.
4. **Wall-clock latency at batch size 1, single-threaded** — measured. Batch 1
   because §5.10's I2 defect forces it; single-threaded because §5.8 records
   that multi-threaded CPU reduction breaks byte-identical replay. Reported as
   the median of 100 timed decisions after 20 discarded warm-up decisions, with
   the interquartile range. A **repeated-timing null** — the identical arm timed
   twice in separate processes — is run for every arm, and any latency
   difference smaller than 2× that null's IQR is reported as **no difference**,
   not as a win.

**No FLOP-only efficiency claim is admissible.** Currency 2 without currency 4
may not be described as an efficiency result. §8.7 A1–A3 record the verified
literature establishing that theoretical sparsity routinely fails to convert
into measured latency; §2.6.3 records that this program has already reproduced
that failure internally. A plan that repeated it a third time would be
negligent.

**5.11.3 Accuracy matching.** An efficiency comparison is only admissible
between arms whose balanced accuracy differs by less than the decisive margin of
0.01 on identical rows. Arms are matched **before** costs are compared. If no
dense arm sits within the margin of a given sparse arm, that sparse arm has no
admissible efficiency comparison and reports `not_comparable` — it does not get
compared to a more accurate dense arm and described as cheaper.

**5.11.4 The mechanism ablation.** A5.3 permits dense components but forbids a
dense component doing the discriminative work behind a sparse explanation. This
is the **only** part of the efficiency contract that gates Q1, it applies **only
to arms containing a dense component**, and it is inherited rather than new: D7
and §11.2 item 8 already required that the explanation be the computation. Until
A5, no arm contained a dense component and the rule was enforced structurally.
A5 removes that structural guarantee, so the rule now needs an operand.

Every arm containing a dense component runs two ablations on identical rows:

- **dense-ablated** — the dense component replaced by its structure-matched
  null; and
- **sparse-ablated** — the sparse component replaced by its structure-matched
  null.

An arm whose accuracy survives sparse-ablation within the decisive margin is
**reported as a dense model**, whatever its explanation length, and may not be
counted toward Q1. An arm with no dense component has nothing to ablate, runs
neither ablation, and is unaffected by this clause.

**Nothing else in §5.11 gates anything.** The four currencies are **disclosed**,
not thresholded. There is no efficiency floor an arm must clear to count toward
Q1, Q2 or Q3, and §3.4.2 forbids introducing one later. A5's efficiency
obligation is discharged by measuring and reporting, which is the only form in
which it can be discharged honestly given §2.6.2.

**5.11.5 Training and sample cost.** Reported separately from inference cost and
never summed with it, per A5.3.

- **Training wall-clock**, single-threaded, per seed, trunk extraction excluded
  and reported separately once.
- **Sample ladder.** Every arm carrying an H108 claim is fitted at
  **{4, 8, 16, 32, 64, 128, 256, 448}** labelled samples per class, drawn
  nested — the *n*-sample set is a subset of the 2*n*-sample set — so that
  differences are not draw artifacts. 448 is the full sealed training allocation
  per class. §5.3's floor of ten fit samples per fitted dimension applies at
  **every rung**: a rung at which an arm's fitted dimension exceeds one tenth of
  its sample count is reported **void, not negative**, exactly as §2.7.1 finds
  the Tier 6 artifact void. This is expected to void the low rungs for the
  high-dimensional arms, and that voiding is itself a reportable finding about
  what sparse arms can be fitted on little data.

**5.11.6 Confirmation replay for a positive result.** Because P6 and H106/H107
register the expected answer as negative, any arm that appears to refute H107 is
**not written up on first observation**. It is re-run from the sealed seeds in a
fresh process, on a second machine state, with the timing null repeated, and it
must reproduce within the decisive margin. This asymmetry is deliberate and is
registered here rather than applied later: an unexpected positive in a program
with this prior is more likely to be an instrument fault than a discovery, and
§2.2's own history — a 52-point swing between two reasonable learning rates —
is the reason.

**5.11.7 What is out of scope.** v15 measures CPU only. No GPU, no structured
N:M sparsity, no custom sparse kernels, no quantisation beyond the inherited
INT8 trunk. §8.7 A4 records that N:M structured sparsity is the one setting
where hardware sparsity gains are verified, and §11.2 item 14 forbids v15 from
borrowing that result to describe its own unstructured sparsity as fast.

### 5.12 Sequence-task contract **[new in v15]**

Applies to M101 only. The forecasting corpus is **not** the DomainNet corpus and
the two may not be pooled, compared, or reported in a shared table; §10.2's
isolation rule applies verbatim.

- **Corpus** — a public long-horizon multivariate forecasting dataset, fixed and
  hashed before any arm is fitted, with the split protocol taken from the
  published source rather than chosen here.
- **Protocol** — chronological splits only. No shuffled split, no random k-fold,
  no target scaling fitted on anything but the training segment. §2.7's void
  artifact is a standing reminder that leakage discipline and adequacy checks
  are the first thing to fail on sequence data.
- **Nulls** — persistence (predict the last observed value) and a
  structure-matched shuffled-target null, both under R5.
- **Adequacy** — §5.3's floor applies to the number of training windows per
  fitted dimension. A configuration that fails it is void.
- **Published bars** — recorded as external anchors only, never as operands
  (R7). §8.8 C1 carries the verified figures.

---

## 6. Priors and assumptions

Registered explicitly because the plan's milestone ordering depends on them and
because a refuted prior should be visible as a refuted prior rather than
discovered as a surprise.

### 6.1 Priors carried from the literature

**P1 — greedy additive approximation attains a dimension-free rate.**
Jones (1992) and Barron (1993) establish that a greedy sum of `T` ridge
functions attains squared `L2` error `O(v_f² / T)`, equivalently `L2` error
`O(1 / √T)`, for targets of bounded variation/Barron norm, **with no explicit
dependence on the ambient input dimension**. Dimension enters only through
whether the Barron constant stays finite. See §8.1.

_What this licenses:_ the belief that "build any task additively from sparse
parts" is not blocked by an approximation-theoretic impossibility.
_What it does not license:_ any claim about a specific corpus, a specific
budget, or a reachable accuracy. The constant `v_f` is unknown here and could be
large.
_How it could be wrong here:_ if the 128-way DomainNet discriminant has a large
Barron constant on dinov2-small features, the rate is vacuous at `T ≤ 10`.
**M96's budget sweep is the empirical test of the constant**, and a flat sweep is
evidence that the constant is large.

**P2 — the binding constraint is the oracle, not the form.**
Every source in §8.1 that states a rate also states that the rate is
**existential over an oracle**: it assumes the best next atom can be chosen. The
program's working assumption is that dense backpropagation won not because dense
models are more expressive at matched size but because stochastic gradient
descent on a wide layer is a cheap approximate oracle for many atoms at once.

_Status:_ **this is an assumption, not a finding.** §8.1 records that we did not
locate a verified canonical hardness theorem for the projection-pursuit oracle
step specifically; what is established is the broader result that globally
training even a three-node network is NP-complete (Blum & Rivest 1989,
unverified — see §8.6). H104 is the milestone that tests P2 directly, and it is
registered **after** H102 precisely because P2 must not be assumed while a
cheaper explanation is untested.

**[corrected in place after the M103 prior-art audit. Retained per §5.10.]** The
Blum & Rivest citation above is about **training** a fixed three-node network,
and it has been used elsewhere in this program's reasoning as though it bore on
**routing** or on assigning inputs to distributions. It does not, and the
extension was never registered. The result that does bear on that question is
**Fang et al., NeurIPS 2022** (§8.10.5), which is a **PAC-learnability** result
whose own abstract states that its impossibility conditions *"may not hold in
some practical scenarios"* and gives *"necessary and sufficient conditions"* for
when the problem **is** learnable. The correction matters because M104–M106
(§7.10–§7.12) route among **known** experts — a closed-set problem, distinct from
the open-set problem Fang et al. analyse, and one on which this program's own
sealed v14 M90.2 already measures **0.8946**. The prior above is left standing
for the projection-pursuit oracle, where it was originally written; what is
withdrawn is any use of it to argue that routing is intractable.

**P3 — a learned overcomplete basis beats ambient coordinates at matched
sparsity.** M80's grid (§2.3) shows the trained dictionary beating a random one
by 8.679 points at k=16, and §2.4's scoping ladder puts 32 ambient coordinates
at 0.522583 against M80's 32 learned atoms at 0.607910.

_Caveat registered now:_ **those two numbers are not comparable.** They use
different partitions, different selection rules and different head forms. The
comparison is a scoping intuition and M94 exists to make it a measurement or
withdraw it.

**P4 — constructive training has already matched end-to-end training at
ImageNet scale, once, under a specific relaxation.** Belilovsky, Eickenberg &
Oyallon (2019), _Greedy Layerwise Learning Can Scale to ImageNet_
(**arXiv:1812.11446, verified by fetch**), report a VGG-11 trained **layer by
layer against shallow auxiliary problems** reaching **0.676 top-1 / 0.880 top-5**
on ImageNet against the **same architecture trained end-to-end at 0.679 / 0.880**.
Their custom SimCNN at k=3 reaches 0.697 / 0.887, and 0.716 / 0.898 as an
ensemble.

_What this licenses:_ this is the strongest published evidence available to the
programme that **greedy, stagewise construction is not intrinsically a
small-scale technique**. A three-tenths-of-a-point top-1 difference and an exact
top-5 tie at ImageNet scale is a materially stronger result for constructive
learning than anything in §8.3's additive-model lineage.

_What it does not license, and this must not be elided:_ **that method is not
backpropagation-free.** It removes *end-to-end* gradient flow; each layer is
still trained by ordinary gradient descent on an auxiliary objective. It is
evidence that **the depth-wise credit assignment path can be cut without cost**,
and it is **not** evidence that gradient-based fitting of each added component
can be removed. M96 and M97 both fit their coefficients by closed-form or
convex means, which is a stronger constraint than Belilovsky et al. adopt, and
their result therefore **bounds v15's ambition from above rather than
predicting v15's outcome**.

**P5 — removing gradients entirely has a large, verified cost at scale.**
Against P4, the record for methods that abandon gradient-based credit
assignment is poor and the poorness is measured:

- Bartunov et al. (2018), _Assessing the Scalability of Biologically-Motivated
  Deep Learning Algorithms and Architectures_ (**arXiv:1807.04587, verified**),
  report ImageNet top-1 **error** of 93.08% for feedback alignment — that is
  **6.92% accuracy** — and 98.34%, 99.36% and 99.28% for target-propagation
  variants, against **63.93% error (36.07% accuracy)** for their backpropagation
  convolutional baseline.
- Hinton (2022), _The Forward-Forward Algorithm_ (**arXiv:2212.13345,
  verified**), reports **0.59** on CIFAR-10 against a **0.63** backpropagation
  baseline in the same table, and states in the paper that the method works on
  "relatively small neural networks containing a few million connections" and is
  "unlikely to replace backpropagation" where power is not a constraint.

_What this licenses:_ nothing optimistic. P4 and P5 together locate the
programme's realistic target precisely: **cutting the end-to-end path is
affordable; cutting gradients altogether currently is not.** V15's milestones
are designed to sit on the affordable side of that line — closed-form and convex
fitting of individual components, added greedily — and §6.2 A5's scaffold
question is exactly the question of which side of the line M98 falls on.

**P6 — unstructured sparsity does not convert into measured speed, and on a
frozen trunk the head cannot convert into anything at all. [new in v15]** This
is the plan's strongest prior in either direction, and it is the reason Q2's
expected answer is registered as negative before measurement.

- Mishra et al. (2021), _Accelerating Sparse Deep Neural Networks_
  (**arXiv:2104.08378, verified**): fine-grained sparsity "maintains accuracy but
  poorly utilizes memory accesses and fails to take advantage of modern vector
  and matrix math pipelines, thus it does not outperform traditional dense models
  on processor architectures such as GPUs"; the benefit of unstructured pruning
  is "negligible and at times negative, even when pruning rate is high (e.g.
  95%)"; parameters can be pruned "nearly 13× with no loss in accuracy" in a
  pattern "not conducive to hardware acceleration".
- Gale et al. (2020), _Sparse GPU Kernels for Deep Learning_
  (**arXiv:2006.10901, verified**): with the moderate sparsity found in neural
  networks, vendor sparse kernels are "not able to outperform their dense
  counterparts"; their purpose-built kernels beat dense only **above 71%
  sparsity**, reaching 1.2–2.1× end-to-end.
- Hooker (2020), _The Hardware Lottery_ (**arXiv:2009.06489, verified**), is the
  framing: methods win partly because the stack favours them.
- **The internal arithmetic is stronger than any of these for v15's purposes.**
  §2.6.2: the frozen ViT-S/14 trunk is 6,065,759,232 MACs and the head is
  0.0008%–0.05% of it. Even a head of literally zero cost changes whole-system
  inference compute by less than one part in a thousand.

_What this licenses:_ registering H106 as expected-confirmed, registering H107
as expected-unrefuted, and — most importantly — **redirecting the efficiency
question away from the head entirely.** If v15 is to say anything affirmative
about efficiency it must be through M99 (growing the representation), through
M102 (declining to run the representation, §7.7), through retrieval instead of
fitting (H111), or through sample cost (H108). §7.5 M100 exists to establish
that redirection with measurements rather than assert it, and §3.2.1 registers
the redirect itself.

**P7 — sparse and simple models are competitive on forecasting and are not
competitive on language. [new in v15]** Both halves are verified, and the
asymmetry is why §3.3 chose forecasting.

- Zeng et al. (2023), _Are Transformers Effective for Time Series Forecasting?_
  (**arXiv:2205.13504, verified**): LTSF-Linear outperforms Transformer-based
  models on all nine datasets; Electricity horizon 96, `Linear` **MSE 0.140 /
  MAE 0.237** against FEDformer **0.193 / 0.308**.
- Elsayed et al. (2021), _Do We Really Need Deep Learning Models for Time Series
  Forecasting?_ (**arXiv:2101.02118, verified**): a gradient-boosted regression
  tree setup outperforms all eight state-of-the-art deep models evaluated,
  across nine datasets.
- Against that: Chelba et al. (2013) (**arXiv:1312.3005, verified**) put an
  unpruned Kneser–Ney 5-gram at **67.6** perplexity on One Billion Word, while
  Dai et al. (2019) (**arXiv:1901.02860, verified**) report **21.8** on the same
  benchmark — a **3.10×** gap that no sparse count-based model has closed.

_What this licenses:_ scoping M101 to forecasting, and forbidding any
extrapolation from a forecasting result to language modelling (§11.2 item 13).

### 6.2 Assumptions about the object being measured

**A1 — ≈0.66 is representation-bound.** Assumed from §2.2 Reading 1. H100 is its
kill switch. If a head family exceeds 0.673452 in M94, this assumption fails and
milestones M96–M98 are re-scoped, because their premise is that head choice is
cheap.

**A2 — INT8 damage is one-sided.** Assumed: quantisation can only lower the
ceiling, never raise it. Registered as an assumption because it is not proven.
N91.11 measured INT8-vs-fp32 divergence at 0.3150 (small), 1.1944 (base) and
0.4812 (large), against ≈1.41 for unrelated vectors — the base graph is broken,
and a broken graph is not evidence that damage is monotone. M95 measures the
direction rather than assuming it.

**A3 — the 10-atom budget is the right currency.** Inherited from
`ACCEPTANCE_CRITERIA_v13.md` §2's deployment context (assisted triage, ≤30 s
read). v15 does not renegotiate it. The budget is on the **fired set per
decision**, not on model size: a model with 8192 atoms that cites 6 per decision
is inside the budget, and a model with 32 atoms that cites all 32 is outside it.
This is the currency in which `knn` beats every engineered arm, and adopting any
other currency at this point would be selecting a metric after seeing the
result.

**A4 — the corpus supports the question.** The DomainNet corpus is 61%
quickdraw and stratified on class rather than domain. §2.4 observation A and
v14's M89/M90.2 both indicate that domain, not class, dominates its local
geometry. **A negative result from v15 is therefore strong evidence about this
corpus and weak evidence about sparse additive learning in general**, and must
be reported with that restriction attached.

**A5 — dense components are admissible, and the constraint is now two
measurements rather than a structural ban. [settled]**

v15 originally raised A5 as a narrow question: may M98 use a dense multilayer
perceptron **solely** as a discarded direction-proposing oracle? That question
has been settled, and settled more broadly than it was asked. The registered
settlement is:

> Dense neural networks are permitted as components of the delivered system,
> provided the system continues to satisfy the original requirements on
> **overall efficiency** and on **inspectability/interpretability**.

Three consequences follow, and all three are binding on the rest of this plan.

**A5.1 — the prohibition changes type.** "No dense networks" was a structural
rule: satisfiable by inspecting the artifact's parts, and never measured. It is
replaced by two obligations that can only be discharged by measurement. This is
a stricter regime, not a looser one. Under the old rule a sparse head over a
frozen 21 M-parameter dense backbone trivially complied, because the backbone
was scoped out as "the representation". Under the new rule that same system must
show its efficiency, backbone included — and §2.6 records that on the sealed
evidence it does not.

**A5.2 — efficiency is now measured, and it is disclosed rather than gated.** A
constraint that is asserted rather than measured is decoration. §3 therefore
adds efficiency as a registered axis, §5.11 registers the currencies and the
accounting boundary in advance of any measurement, and H106/H107/H108 supply the
kill switches. Registering the currency first is not bureaucratic: §2.6 shows
that this program's "sparse" arms look efficient in one currency (citations per
decision) and are catastrophically inefficient in two others (stored parameters,
multiply-accumulates per decision). Whoever picks the currency after seeing the
numbers picks the answer.

But the obligation is discharged by **reporting** the currencies, not by any arm
clearing an efficiency threshold. §3.4.2 registers this and forbids introducing
a threshold later, because §2.6.2's arithmetic means an efficiency gate applied
to head-side arms would pass or fail them for reasons unrelated to what they are
being tested for. The acceptance criteria for Q1 are unchanged by A5.

**A5.3 — the scaffold/artifact distinction survives but is no longer needed for
M98.** M98's oracle MLP was to be discarded before delivery, so it complied even
with the strict reading. It is now admissible under the broad reading as well,
and **M98 is unblocked** (§10.1). The distinction is retained in the vocabulary
because it still matters for reporting: a dense component that is discarded
before delivery incurs training cost only, whereas a dense component that is
retained incurs inference cost on every decision forever, and §5.11 requires the
two to be reported separately rather than summed.

What the settlement does **not** license: it does not permit a dense network to
absorb the discriminative work while a sparse shell supplies the explanation. §9
D7 and §11.2 item 8 already forbid delivering an explanation that is not the
computation. A5 changes which components may appear; it does not change the
requirement that the cited atoms be the mechanism that produced the decision.
An arm in which the dense component can be ablated without materially changing
accuracy is a sparse model; an arm in which the sparse part can be ablated
without materially changing accuracy is a dense model wearing an explanation,
and §5.11.4 registers the ablation that tells them apart.

### 6.3 What the plan assumes about its own instruments

**A6 — sealed figures are replayable.** Every milestone's correctness
requirement is stated as reproduction of a sealed number (§5.5, §5.9). If a
regenerated dictionary or a recomputed control does not reproduce, the milestone
reports a `not_*` verdict and stops rather than proceeding on an approximation.

**A7 — the search instrument is weak.** M88 measured this program's prior-art
search recall at **4 of 7**. §8 is therefore written as a lineage list with
disclosed verification status, not as a coverage claim, and §11.2 item 7 forbids any
novelty claim built on it.

---

## 7. Milestones

| id      | question                                                              | execution                            |
| ------- | --------------------------------------------------------------------- | ------------------------------------ |
| **M94** | Reproduce §2 as sealed, replayable evidence, and close the head question | unconditional                       |
| **M95** | Is the ceiling the quantisation?                                       | unconditional, independent of M94    |
| **M96** | Does additive selection from a fixed dictionary beat one-shot selection? | after M94 and M95                  |
| **M97** | Does growing new atoms beat selecting existing ones?                   | after M96                            |
| **M98** | Is the oracle the constraint?                                          | only if H102 refuted (A5 settled — §6.2) |
| **M99** | Can the representation itself be grown?                                | only if M100 confirms H106 (§10.2 — regated in v15) |
| **M100** | What does the system actually cost, and does sparsity ever reduce it? | unconditional, after M94             |
| **M101** | Does additive construction transfer to sequential prediction?         | unconditional, **last in order** — after M96 and M97 report |
| **M102** | Can abstention be optimised directly, and does it convert into compute? | unconditional, after M100 **[new in v15]** — **executed out of order, see below** |
| **M103** | Does additive atom selection beat a random dictionary at matched size? | unconditional **[new — §7.9]** |

Q1 is answered by M94–M99, Q2 by M100 and M102, Q3 by M101. §3.4 registers how
the three interact: Q2 and Q3 are separate axes, they do not gate Q1, and v15's
verdict is H102's alone. §3.2.1 registers why Q2's centre of gravity is M102 and
not the head milestones.

**Registered deviation from this table. [recorded after execution]** M102 was
executed **before M100**, contrary to the ordering above. The reason is that
§7.7 Tier A consumes no M100 output — it reads the sealed v13 features and its
own fitted arms, and its compute ledger is the excluded-trunk statement required
by prohibition 20 rather than anything M100 produces. The deviation is recorded
rather than waived because the program's convention is that a stated ordering is
a commitment. **What it cost:** nothing measurable for Tier A, but M100 would
have supplied the wall-clock ledger that §2.9.2 now shows is missing, and
running it first would have surfaced the INT8 defect in the v14 backbone
features before §2.9.2 relied on them. **What it may not license:** the
deviation does not establish that milestone ordering in this plan is advisory,
and no future milestone may cite it as precedent without its own recorded
reason.

### 7.1 M94 — the frontier, reproduced and completed

**Non-gating by construction**, in the M89 mould. M94 registers no pass/fail
except its own instrument check. It exists so that §2 is replayable rather than
recalled, so that M96 and M97 have a stated baseline in the same currency, and
so that H100 gets a kill switch.

**Part A — reproduction, no refitting.** Recompute the §2.1 table from the sealed
M81 evidence and the §2.3 table from the sealed M80 evidence, and emit them as a
single sealed artifact with the accuracy and budget axes joined. Correctness
requirement: reproduce `knn` at exactly **0.6612548828125**,
`sparse_linear_l1_0.0` at exactly **0.607177734375**, and the M80 raw probe bar
at exactly **0.613037109375**. A run that does not reproduce all three is
describing a different object and reports `not_v13_geometry`.

**Part B — the missing head families, to give H100 a kill switch.** M81 measured
seven head families. Two obvious nonlinear families were never measured on these
rows and are added here: a **random forest** and a **gradient-boosted tree
ensemble**, both under §5.6's mandatory sweep, both with permuted-label nulls,
both reporting train and evaluation accuracy. Registered prediction, recorded so
it is not presented later as a finding: **both are expected to land inside the
0.64–0.67 cluster and neither is expected to exceed 0.673452.** If either
exceeds it, H100 is refuted and §3's premise fails.

**Part C — the sparsity ladder, made comparable.** The §2.4 ambient-coordinate
ladder is re-measured under the shared contract, at k ∈ {2, 4, 8, 16, 32}, using
M81's contribution-mass ranking rather than ANOVA-F, on the N83.7 partition,
against the regenerated M80 atoms at matched k. This turns §6.1 P3 into a
measurement or withdraws it.

**Part D — the hierarchy record.** The §2.4 observation A arms are re-run once,
under contract, with their structure-matched nulls, and sealed. They are not
expected to succeed. They are recorded so that the do-not-pursue registration in
§11.2 item 9 rests on contract evidence rather than on a scoping probe.

**Registered in advance:** if Part B lands inside the cluster and Part C shows
learned atoms beating ambient coordinates at matched sparsity, that is the
result, and it closes the head question rather than opening a new one.

### 7.2 M95 — is the ceiling the quantisation?

Cheap, unconditional, and independent of everything else. It produces one
comparison and a limit statement.

**Procedure.** Extract fp32 dinov2-small features for the identical images at
**batch size 1** under `.venv-rocm` if and only if the extraction is sealed by
hash and every downstream figure is computed on CPU (§5.8 permits exactly this
and no more). Run the **identical** linear probe and the **identical** `knn`
arm on the identical partition.

**Operands.** Linear probe against **0.613037109375**; `knn` against
**0.6612548828125**. Decisive margin **0.01**, as M85 registered.

**Null (R5).** The identical arms on INT8 features, identical rows — that is,
the sealed values themselves, recomputed rather than quoted.

**Kill switch.** H103 is refuted if neither operand moves by more than 0.01.

**Registered in advance, three ways:**

1. If fp32 moves the bars **upward** by more than 0.01, then every v13 and v14
   accuracy figure is a statement about a **quantised** backbone. That is a
   limit on the sealed record, recorded as such. It does **not** reopen any v13
   or v14 verdict (§11.1) and it is not a v15 result.
2. If fp32 moves the bars **downward**, A2 is refuted and the direction of INT8
   damage is not what the program assumed.
3. If nothing moves, H103 is refuted, the ceiling is not the quantisation, and
   §3's premise is strengthened.

**Secondary, non-gating diagnostic.** Report the **sign of the linear-versus-kNN
ordering** on fp32 features alongside the two operands. §2.5 records that the
published DINOv2 ViT-S/14 ordering on ImageNet-1k is linear 0.811 above k-NN
0.790, while this program's sealed ordering is k-NN 0.661255 above linear
0.613037. If the fp32 extraction restores the published ordering, the inversion
is an extraction artifact; if it does not, the inversion belongs to the corpus.
**Neither branch is an operand and neither may be compared to the external
number as evidence** (R7). This is a diagnostic that tells M96 and M97 which
object they are working on, and nothing more.

**Confound disclosed now:** because of the I2 defect (§5.10) the INT8 features
are a function of batch membership even at batch size 1 in the sense that the
graph's quantisation parameters are recomputed per call. A null result here is
weaker than it looks and must be reported with that attached.

### 7.3 M96 — Residual Atom Pursuit, rung 1: additive **selection**

The first genuinely constructive milestone. It changes **how atoms are chosen**
and changes nothing else.

**Prerequisite.** §5.9's dictionary regeneration, verified at both cells.

**Procedure.** For each class, starting from an empty support and a residual
initialised to the one-hot target:

1. score every atom in the regenerated dictionary by its correlation with the
   **current multiclass discriminative residual**;
2. add the single best-scoring atom to the support;
3. **refit all coefficients on the enlarged support** (the orthogonalised step
   that distinguishes orthogonal matching pursuit from plain matching pursuit,
   §8.2);
4. recompute the residual and repeat until the atom budget is reached.

**Budget sweep.** T ∈ {2, 4, 6, 8, 10, 16, 32} atoms cited per decision. The
sweep is registered in full and reported in full; no budget is selected after
the fact.

**Nulls (both mandatory, §5.4).** (a) identical budget, atoms drawn uniformly at
random from the identical dictionary, identical refit; (b) identical budget,
atoms selected against the **reconstruction** residual, identical refit. Null
(b) is the one that isolates the plan's actual claim.

**Operands.** Balanced accuracy at each budget, jointly with the fraction of
decisions inside the 10-atom budget (L1′). Compared against the matched sealed
arms: `sparse_linear_budget_1024` at (0.441284, 0.9655) and
`sparse_linear_l1_0.0` at (0.607178, 0.0000). And, under L6, against `knn` at
(0.661255, 1.0000).

**Kill switches.** H101 is refuted if the greedy arm fails to beat the matched
one-shot arm by more than the seed spread at **every** budget. H102 is advanced,
but not confirmed, if any budget reaches ≥ 0.6112548828125 with ≥ 95% of
decisions inside 10 atoms.

**Sample adequacy.** At T = 32 the head fits 32 coefficients per class against
512 fit rows per class — 16 rows per free parameter, clearing the §5.3 floor. At
larger T the floor binds and the sweep stops there. This is why the sweep ends
at 32 and not at 64.

**Registered prediction, recorded now:** the greedy arm is expected to beat null
(a) comfortably and to beat null (b) narrowly or not at all. If it does not beat
null (b), the honest report is that **greed helped and discrimination did not**,
and §8.2's discriminative-dictionary lineage is the place that result belongs.

### 7.4 M97 — Residual Atom Pursuit, rung 2: additive **growth**

M96 selects from 8192 fixed atoms. M97 removes the dictionary and constructs each
atom, so that the model is grown rather than subset-selected.

**Procedure.** At each step, compute the next direction in **closed form** as
the leading singular direction of the residual-weighted class-difference matrix
on the fit rows, normalise it, append it to the support, refit all coefficients,
recompute the residual, repeat. The step is deterministic and therefore
byte-identically replayable under §5.8, which a stochastic oracle would not be.

**Budget sweep.** Identical to M96, T ∈ {2, 4, 6, 8, 10, 16, 32}.

**Nulls.** (a) random direction at matched norm, identical budget, identical
refit; (b) **the M96 arm at matched budget** — this is what isolates "grown"
from "selected", and without it a win here says only that atoms help.

**Operands and kill switch.** As M96. H102 is **confirmed** if any arm reaches
≥ 0.6112548828125 with ≥ 95% of decisions inside 10 atoms and beats both nulls
by more than the seed spread; H102 is **refuted** if neither M96 nor M97 does.

**Registered in advance:** if M97's grown atoms beat M96's selected atoms, the
correct reading is that **8192 atoms was not an adequate dictionary**, not that
growth is superior in principle. Distinguishing those requires a dictionary-size
sweep that this corpus's fit budget does not support at the §5.3 floor, and that
limitation is registered here rather than discovered later.

### 7.5 M100 — the cost of the system **[new in v15]** {#m100}

Answers Q2. Unconditional, cheap, and it runs early because its result changes
what the rest of the plan is allowed to claim. M100 fits no new model: it
measures arms that already exist or that M94 reproduces.

**Why this is first among the efficiency work.** §2.6.2 derives, from
architecture alone, that the head controls under 0.05% of inference compute. If
that derivation survives measurement, then **no milestone operating on the head
can produce an efficiency result**, and the plan should say so once, with
numbers, rather than leaving an unfalsifiable efficiency aspiration attached to
every milestone. If the derivation fails to survive measurement, the instrument
is wrong and everything downstream of it is suspect.

**Arms.** Every M81 arm reproduced by M94, plus the dense linear probe over the
384 raw features, plus the dense MLP, plus a zero-cost head null — a head that
emits a constant — which establishes the floor of the whole-system cost that no
head design can go below.

**Operands.** The four inference currencies of §5.11.2, whole-system and
head-only (§5.11.1), for every arm; the repeated-timing null for every arm; the
accuracy-matched pairings of §5.11.3; and the sample ladder of §5.11.5 for the
arms carrying H108.

**Kill switches.** H106 is refuted if any arm moves whole-system MACs by more
than 1% or median batch-1 latency by more than 2× the timing null's IQR. H107 is
refuted if a sparse arm is strictly cheaper in all four currencies at matched
accuracy — and §5.11.6's confirmation replay runs before that is written
anywhere. H108 is refuted if the additive arm fails to beat both dense arms at
every rung of the ladder.

**Instrument check.** The zero-cost head null must reproduce the trunk-only cost
within the timing null's IQR. If it does not, the harness is measuring something
other than what it claims and M100 reports `not_instrumented` and stops.

**Registered in advance.** The expected result is that H106 is confirmed and
H107 is not refuted, on the arithmetic of §2.6.2 and the verified literature of
§8.7. **That expected result is a reportable finding, not a null result**: it
states, with measurements, that the program's fourteen versions of head
engineering were never capable of producing an efficiency win, and that the
efficiency question belongs to the representation. Outcome C (§11.1) covers it.

The one genuinely open question in M100 is H108. Sample cost is the only
currency in which the frozen trunk cancels rather than dominates.

### 7.6 M101 — additive construction on a sequence task **[new in v15]** {#m101}

Answers Q3. Unconditional, and **last in execution order** — it runs after M96
and M97 have reported, so that it cannot consume the fit budget belonging to the
primary question (§3.4.4). If the budget is exhausted first, M101 reports
`not_run` and no Q3 conclusion is drawn in either direction. It runs on an
**isolated instrument** under §10.2's rule: a separate corpus, a separate
harness, separate seals, and no shared table with any DomainNet result at any
point.

**Why forecasting and not language.** P7 records the asymmetry: verified evidence
that simple and sparse models beat deep sequence models on forecasting, and
verified evidence of a 3.10× perplexity gap on language. Running M101 on
language would produce a guaranteed loss that discriminates nothing, because the
loss would be predicted by the arena rather than by the method. Running it on
forecasting puts the method in the one place the literature says it can win, so
that a loss is informative about the method.

**Corpus and protocol.** §5.12. Chronological splits, published split protocol,
hashed before fitting.

**Arms.**

1. **Grown additive** — components added one at a time against the forecasting
   residual, each fitted closed-form or convex, using the M96/M97 machinery
   ported to a regression target. This is the arm under test.
2. **One-shot sparse** — the same component family selected in one shot by
   penalised regression at matched component count. Isolates growth from
   sparsity, as M96 does for classification.
3. **Dense sequence head** — a small dense network on the same inputs at matched
   parameter count, admissible under A5.
4. **Persistence null** — predict the last observed value.
5. **Structure-matched shuffled-target null** — R5.

**Operands.** MSE and MAE at the registered horizon; components cited per
forecast; the four §5.11.2 currencies whole-system; and the §5.11.5 sample
ladder, adapted to training-window count.

**Kill switch.** H109 is refuted if the grown additive arm fails to beat either
the persistence null or its own structure-matched null. **Refutation is a
first-order finding** — it says the additive construction principle does not
transfer out of static classification — and it is reportable on its own without
any further milestone.

**Registered as non-gating.** Comparison to Zeng et al.'s verified `Linear` bar
(MSE 0.140 / MAE 0.237, Electricity horizon 96) is an external anchor and never
an operand (R7). M101 may not be used to argue that sparse models are or are not
competitive with LLMs; §11.2 item 13.

**Registered limitation.** One corpus, one horizon, one component family. A
positive M101 licenses the statement that the construction transfers to this
forecasting task, and nothing wider. §2.7's void artifact is the standing
reminder of what happens when a sequence experiment is run below its adequacy
floor and reported as though it were evidence.

### 7.7 M102 — abstention as the objective **[new in v15]** {#m102}

**Question.** Can a sparse model be fitted so that its *deferral decisions* are
good, rather than fitted for accuracy and read for confidence as an afterthought
— and does the improvement convert into measured compute?

**Why this milestone exists.** §2.8 measures a real, large, unclaimed headroom:
the oracle cascade beats the full model while skipping half the expensive calls,
and the confidence gate captures 44.4% of that *(not reproducible — §2.8.5;
corrected to 30.6%)*. §3.2.1 registers it as Lever 1. Nothing in M94–M101 tests
it. This is the only v15 milestone whose success would reduce inference compute
by a factor rather than a rounding error.

**Design.**

1. **Stage-1 candidates, four arms.** (a) The §2.8 nearest-class-mean-on-PCA
   baseline, reproduced under M94's seal so its numbers become admissible. (b) A
   sparse geometric arm fitted for accuracy, gate read from its margin.
   **(b′) Arm (b) with a temperature-scaled gate**, the temperature fitted on a
   held-out split by the Guo et al. (2017) procedure (§8.9 D6). (c) The same
   sparse family fitted with a **deferral-quality objective** — the arm H110 is
   about. Arms (b) and (c) differ **only** in the objective, so the comparison
   isolates the thing being claimed.

   **Arm (b′) exists to try to make H110 unnecessary.** §8.9 D6 records that a
   single post-hoc scalar removes most of the calibration error in deep networks.
   If temperature scaling alone carries the gate from 44.4% *(§2.8.5: not
   reproducible; corrected baseline 30.6%)* to the 60% bar, then the joint
   objective bought nothing and **H110 is refuted by the cheap baseline rather
   than confirmed by the expensive arm**. Registered now, before M102 runs,
   because a plan that tested only its preferred method against a weak baseline
   would be constructing its own confirmation.
2. **Deferral sweep.** Deferral rates {0.25, 0.40, 0.50, 0.60, 0.75}, registered
   now. The 0.50 rate is the pre-registered primary operating point because §2.8
   locates the oracle/full-model crossover there; the rest are reported but do
   not carry the verdict. This is registered in advance so that the operating
   point cannot be chosen after seeing which one flatters an arm.
3. **Nulls, per R5.** Every point carries (i) a **random-deferral null** of
   identical size, (ii) an **oracle** upper bound, and (iii) a **budget-matched
   dense gate** — a small dense confidence head with parameter count matched to
   arm (c). Without (iii), a positive result would only show that *some* gate
   beats a margin, not that a *sparse* one does. Arm (c) versus the dense gate is
   the operand that speaks to Q2's sparsity claim; arm (c) versus arm (b) is the
   operand that speaks to H110's objective claim.
4. **Compute accounting.** Every arm reports the full §5.11 ledger — MACs,
   parameters, and wall-clock — for the cascade *as a whole*, including the gate
   itself and including the deferred calls. **An arm whose gate costs more than
   it saves is reported as a loss**, and §5.11.4 remains the only Q1-facing
   efficiency gate.

**The trunk problem, and how M102 handles it honestly.** §2.8's registered
limitation is that every arm consumed the 384-d feature, so no arm saved
anything. M102 splits into two tiers and reports them separately:

- **Tier A (statistical, unconditional).** Everything above, run on the sealed
  features. It measures *gate quality* and cannot measure saving. Its ledger
  reports MACs **excluding** the trunk, labelled as such, and states in the same
  table that trunk cost is unchanged at 6,065,759,232 MACs per input for every
  arm. Tier A can confirm or refute **H110** and nothing else.
- **Tier B (systems, conditional on Tier A confirming H110).** Stage-1 is
  re-fitted on a cheap input — a downsampled image or an early-exit trunk
  block — so that a non-deferred input genuinely never pays the full trunk. Only
  Tier B can produce a compute number under §5.11.4. It is **explicitly
  conditional**, because running it after a refuted Tier A would spend a
  milestone on a gate already known not to work.

**Kill switch.** If arm (c) fails to exceed 60% oracle-gain recovery at 0.50
deferral, H110 is refuted, Tier B does not run, and the reported finding is that
**abstention on this representation is not improvable by objective choice** —
which, given §2.8.3's record of the same defect appearing as poor calibration
and weak OOD across three prior versions, is a substantive negative result about
sparse geometry and should be reported as one.

**Second kill switch, from §8.9 D6.** If arm (b′) — accuracy-fitted with nothing
but a temperature-scaled gate — reaches the 60% bar, H110 is **refuted by
sufficiency of the baseline**, regardless of how arm (c) performs. The registered
finding in that case is that abstention on this representation was improvable all
along by a single scalar, which is a cheaper and more useful result than H110's
confirmation would have been, and it is reported as such rather than buried.

**Registered honesty condition.** Tier A's numbers are gate-quality numbers.
§11.2 item 20 forbids reporting a Tier A confirmation using the language of
compute saving, and forbids reporting Tier A's excluded-trunk MAC figure without
the accompanying unchanged-trunk statement in the same table.

### 7.8 Seeds and spreads

Every M96 and M97 arm runs at **three seeds** (11, 12, 13), and every operand is
reported as mean and spread. This is registered because of N82.7: M81's headline
I5-8 figure of 0.4022 was a per-seed best-arm maximum misattributed as an arm
mean, and the arm's own mean was 0.3866. **No v15 figure is reported as a
maximum over arms.** Where a best arm is named, its own three-seed mean is
reported beside it.

M100 and M101 inherit the same rule. For M100's timing operands the seed axis is
supplemented by the repeated-timing null of §5.11.2, because process-to-process
variation on a shared machine is a larger error source than seed variation and
the two must not be conflated.

M102 inherits the same rule and adds one of its own: because H110's bar is a
*ratio* (recovered fraction of oracle gain), the three-seed spread is reported on
the ratio itself, not on its numerator and denominator separately. A ratio of
differences is noisier than either difference, and reporting only the component
spreads would understate the uncertainty on the quantity that carries the
verdict.

### 7.9 M103 — is a grown dictionary better than a random one? **[new — registered after M102]** {#m103}

**Question.** §2.9.3 measures a patch representation that is built additively,
without backpropagation, at three orders of magnitude less compute than the v13
trunk — and finds that **learning** the dictionary by k-means is *worse* than
drawing it at random, at every budget tested. M103 asks the question that
follows: **does building the dictionary additively against a discriminative
residual beat random selection at matched size?**

**Why this milestone exists, and why it is not M99.** M99 (§10.2) is a large,
externally-benchmarked milestone gated on a cost ledger that §2.9.2 shows cannot
currently be produced. M103 is the small, unconditional, self-contained question
underneath it: *does atom choice matter at all?* If it does not, M99's entire
premise is unsupported and the program should know that before spending M99. If
it does, M103 supplies the construction procedure M99 would scale. Registering
the cheap discriminating experiment before the expensive one is the same
ordering discipline §4.3 records for arm (b′).

**[recorded after execution — the prior on this milestone has reversed.]** This
section as first committed registered M103 while the program's own scoping
evidence predicted it would fail, on the strength of §2.9.3's random-beats-
k-means ordering. §2.9.4 measures that the ordering had a second reading and
that the second reading holds: a dictionary selected against a discriminative
residual beat random selection in **all six** seed-budget cells tried, at 3.0–
3.7× the null arm's own seed spread. The registered expectation is therefore
reversed, and it is reversed **in place** rather than edited away so that the
record shows the plan predicted failure before it measured otherwise. **Nothing
else in this section changes** — not the design, not either kill switch, not the
sample floor, not the acceptance criterion. A milestone whose bar moved when its
prior moved would not be a milestone.

**This is the plan's central thesis in its most direct form.** §3.1 asks whether
a model *grown* one component at a time can occupy a point no *fitted* model
reaches. M96 and M97 ask that about the head, which §2.6.2 shows is under 0.05%
of inference compute. M103 asks it about the **representation**, which is the
other 99.95%, on a corpus where a published backprop-free bar exists.

**Design.**

1. **Arms, all at matched dictionary size.** (a) **Random patches** — the null,
   drawn from the whitened patch pool, which §2.9.3 measures as the *strongest*
   of the three constructions tried so far. (b) **k-means**, the Coates et al.
   construction, carried so that §2.9.3's ordering is tested under seal rather
   than recalled. (c) **Additive residual-driven selection** — atoms chosen one
   at a time to maximally reduce the residual of a discriminative objective, the
   arm the thesis is about. (d) **Random projections**, not patches, as a second
   null distinguishing "atoms drawn from data" from "atoms that are directionally
   arbitrary".
2. **Budget sweep, registered now:** 64, 128, 256, 512, 1024, 2048 atoms. The
   registered primary operand is **the atom count at which each arm first reaches
   the accuracy arm (a) reaches at 1024** — that is, M103 measures *efficiency at
   matched accuracy*, not accuracy at matched size, because §10.2's registered
   question is about needing **fewer** patches.
3. **Seeds.** Three (11, 23, 37), mean and spread on every reported figure.
   §2.9.3's ordering is single-seed and carries nothing until this is done.
4. **Instrument correctness.** Full 50,000-row CIFAR-10 train split, stride 1,
   head trained to convergence — §2.9.3 relaxed all three and its accuracies are
   therefore below what the family reaches. M103's arm (b) at 1024–2048 atoms
   must land in the region of the **0.796** Coates reference (§8.5) or the
   instrument is reported broken and no arm is read.

   **[corrected after execution — this check as written cannot be satisfied by
   any run, and it violates R7. Retained per §5.10.]** §8.5's Coates figure is
   **0.796 at 4000 features**, and §7.9 restriction 4 — registered in the same
   commit as the sentence above — fixes M103's top *readable* rung at **1024
   atoms**. §2.9.6 measures arm (b) at 1024 atoms at **0.6856**, and the curve
   does not reach 0.796 there and cannot. A check that fails on every possible
   run carries no information. Separately, making an external figure a pass/fail
   gate makes it an **operand**, which R7 forbids outright. The corrected check
   is therefore three internal conditions and one anchor:

   * **(i) Monotonicity.** Arm (b)'s accuracy must rise with atom count across
     the readable rungs. A patch pipeline whose accuracy does not increase with
     dictionary size is broken regardless of where it sits.
   * **(ii) A floor set by this program's own weaker instrument.** Arm (b) at
     1024 atoms must exceed §2.9.3's own 1024-atom k-means reading of
     **0.6223**. M103 relaxes none of the three things §2.9.3 relaxed, so a
     pipeline that fails to beat the probe it supersedes is broken.
   * **(iii) Encode determinism.** The encode must be bitwise repeatable within
     the run, checked by encoding one block twice, because the sweep runs on a
     multi-threaded backend and an irreproducible encode makes every figure
     from it irreproducible too.
   * **(iv) The Coates figure is reported beside the curve as an anchor**, with
     the 4000-versus-1024 mismatch stated in the same table, and it gates
     nothing.

   This correction **tightens** conformance to R7 rather than relaxing a bar,
   and it is registered before M103 runs. What it does not do is change any
   operand, either kill switch, the sample floor, or the acceptance criterion.
5. **Compute ledger.** Every arm reports the full §5.11 ledger. Unlike M102 Tier
   A, this milestone changes the representation itself, so its MAC figures are
   **real** and are not subject to prohibition 20's excluded-trunk restriction.
6. **Candidate-pool matching. [added after execution, from §2.9.4 limitation
   (iii)]** Arm (c) selects from a finite candidate pool, and a larger pool is
   itself an advantage independent of the criterion. **Arms (a) and (c) must
   draw from the same pool at the same size**, so that the operand isolates the
   selection criterion rather than the breadth of the search. The pool size is
   reported beside every arm (c) figure. Without this, a confirmed arm (c) would
   be ambiguous between "the discriminative criterion works" and "looking at more
   atoms works", and §2.9.4 cannot distinguish those.
7. **Training-compute disclosure. [added after execution]** Arm (c) pays a
   selection cost — encoding the whole candidate pool — that arms (a) and (d) do
   not pay at all. §5.11 puts training and inference in separate columns and
   §11.2 item 10 forbids netting them. M103 reports **both** columns for every
   arm, and any statement that arm (c) is more efficient must name which column
   it means in the same sentence.

**Kill switch.** If arm (c) does not reach arm (a)'s 1024-atom accuracy with
**fewer than 1024 atoms**, at three seeds, then additive residual-driven
construction does not beat random selection on this corpus, and the registered
finding is that **for patch dictionaries, atom count dominates atom choice**.
That is a substantive negative result about the plan's central thesis and §11.1
requires it to be reported as the headline, not as a footnote.

**Second kill switch.** If arm (d) — directionally arbitrary projections, not
data patches — matches arm (a), then the effective content of the dictionary is
its *size and geometry* rather than its provenance in the data, and both M103
and M99's premises weaken together. That outcome is registered now so it cannot
be reported as a curiosity later.

**Registered restrictions.**

1. **Corpus isolation, inherited from §10.2 restriction 1.** M103 runs on
   CIFAR-10. **No M103 figure may be compared to any v13, v14 or v15 DomainNet
   figure in either direction** (R7). §2.9.3's note about the coincidental
   proximity of 0.6339 and 0.6322 applies with full force.
2. **No novelty.** Patch dictionaries, random features and greedy dictionary
   construction are established (§8.1, §8.2, §8.4, §8.5). M103's contribution is
   a comparison, not an invention, and §11.2 item 22's form applies here too.
3. **Bars re-verified at execution time**, per §8.6 item 8, exactly as §10.2
   restriction 2 requires of M99.
4. **Sample floor.** §5.3's floor of 10 binds on the linear head: at 50,000 rows
   and `4 × atoms` features, 2048 atoms gives 8,192 features and **6.1 rows per
   feature**, which is **below the floor**. The 2048 rung is therefore registered
   as **expected void** unless pooling is coarsened, and M103 may not read it
   without recording the adequacy figure beside it. This is registered before the
   run because M102's ledger had to correct exactly this error after the fact
   (C102.2).

---

### 7.10 M104 — does sizing an expert to its sub-population's effective rank beat sizing it uniformly? **[new — registered after the M103 prior-art audit]** {#m104}

**Question.** §2.9.7 probe 3 measures a **6.32× spread** in effective rank
across DomainNet's six data types. Every mixture-of-experts in §8.10 — Switch,
Mixtral, DeepSeekMoE — allocates **the same capacity to every expert**;
DeepSeekMoE's contribution was making experts *finer*, and they remain uniform.
A uniform allocation therefore overspends on quickdraw by roughly 5× and starves
the four domains that sit above the mixed control. M104 asks: **at matched total
inference compute, does allocating capacity in proportion to measured effective
rank beat allocating it equally?**

**Why this milestone exists.** M103 confirmed that atom *choice* matters within
one dictionary. M104 asks whether atom *allocation* matters across several. It
is the smallest experiment that can distinguish the new direction from the
published MoE literature, and it is registered before any system is built for
the same reason §7.9 was registered before M99 — the cheap discriminating
experiment comes first.

**Design.**

1. **Arms, all at matched total inference MACs.**
   * **(a) Uniform MoE** — six experts, equal atom count. **The null**, and the
     construction the entire published literature uses.
   * **(b) Rank-sized MoE** — same total atoms, allocated proportional to each
     domain's population effective rank measured on **train rows only**.
   * **(c) Single generalist** — one dictionary at the same total atoms. Tests
     whether *any* partition helps, independent of sizing. **[amended in place
     before execution — see the execution-time amendments below. "The same
     total atoms" and design item 2's MAC match are not the same constraint for
     a generalist; they differ by a factor of six, and BOTH are now run.]**
   * **(d) Random-partition MoE** — same total atoms, allocated by a random
     partition drawn to the same sum and the same per-expert floor. **This is
     the structure-matched null R5 requires**, and it is the arm that decides
     whether the *instrument* is doing the work.
2. **Matched on inference MACs, not parameters**, per §5.11. The allocation
   changes both, and only one of them is the cost. **[amended in place before
   execution: this reduces exactly to matching the ROW-WEIGHTED atom sum
   `Σ_e f_e·A_e`, not the plain atom sum `Σ_e A_e`. See the execution-time
   amendments below. Where this item and design item 1 conflict, THIS ITEM
   GOVERNS.]**
3. **Routing is by oracle domain label in M104**, so that a routing failure
   cannot be mistaken for a sizing failure. **M104 therefore measures an upper
   bound and may not be reported as a system result** — §7.11 measures the
   system. Any statement of an M104 figure must carry the word *oracle* in the
   same sentence.
4. **Seeds.** Three (11, 23, 37), mean and spread on every reported figure,
   per §7.8.
5. **Operand.** Test accuracy, reported **per domain and pooled**. The per-domain
   breakdown is the operand that tests the mechanism; the pooled figure alone
   cannot distinguish the registered prediction from its alternatives.
6. **Sample floor.** §5.3's floor of 10 rows per fitted dimension binds **per
   expert, on that expert's own rows** — an expert serving one sixth of the
   corpus reaches the floor at one sixth the features. The adequacy figure is
   computed for every expert in every arm **before** the run, and any expert
   below the floor makes its arm **void, not negative**, exactly as C102.2
   required after the fact. Arm (b)'s largest expert is capped by this and the
   cap is reported beside the allocation.
7. **Compute ledger.** Full §5.11 ledger for every arm, with training and
   inference in separate columns and no netting (§11.2 item 10). Arm (b) pays a
   rank-measurement cost that arm (a) does not; it is charged in full to the
   training column and reported even though it is small.

**Execution-time amendments, registered before any M104 figure was computed.**
Building the runner forced four questions the registration above did not answer,
and running the instrument once on a deliberately inadmissible smoke corpus
exposed a fifth. All five are recorded here, in the plan,
**before the sealed run was started**, because a design decision taken after
seeing a result is not a design decision.
Each is written so that a reader can see whether it makes the milestone easier
or harder for arm (b) to pass.
The first four make it **harder**; the fifth makes the sample floor
**bind as it was written to bind**, which is what the registration already
claimed it did.

1. **The MAC match is the row-weighted atom sum, and the two readings of design
   item 1 differ by 12.5%.** Under oracle routing an image is encoded by exactly
   one expert, so the system's per-image cost is `whitening + k·A_e` for
   whichever expert serves it. The whitening term is identical in every arm.
   Matching total inference MACs therefore reduces **exactly** to matching
   `Σ_e f_e·A_e`, where `f_e` is domain *e*'s share of rows. This is **not** the
   plain atom sum, because DomainNet's domains differ in size by **3.6×**
   (`real` holds 120,906 train rows, `clipart` 33,525). At 512 atoms, arm (a)
   spends **3,072** atoms and arm (b) spends **3,455** — 12.5% more parameters —
   for the identical inference MACs. Design item 2 governs, so arm (b) is
   matched on the cost and its parameter excess is
   **reported rather than matched away**.
   Reporting it is against interest: a reader who thinks parameters are the
   right currency can see immediately that arm (b) is not matched in that
   currency, and the figure needed to make that objection is supplied here
   rather than withheld.
   **[Provenance of the 3,455, recorded once the sealed run had started and
   before any accuracy existed.]** The figure just quoted is computed from
   §2.9.7 probe
   3's published per-domain ratios, which are an **anchor** under R7 and are
   never an operand. The sealed run does **not** consume them: it re-measures
   effective rank inside itself, under its own seed, whitener and encoder, on
   2,000 train rows per domain, precisely so that arm (b) is self-contained.
   The run's own allocation therefore need not equal it and is not expected
   to; at seed 11 it is close to it but not identical.
   **That figure illustrates the size of the parameter excess the MAC match
   implies; it is not a prediction
   of the run and no result is read against it.** The figure the milestone
   reports is the run's own row-weighted total, taken from the evidence file.
   This note is a statement about where an already-registered number came from.
   It changes no design, no arm, no null and no kill switch, and it was written
   while the run's ranks were visible and none of its accuracies were.
2. **Both generalists are run.** Design item 1(c) says "the same total atoms";
   design item 2 says "matched on inference MACs". For a *mixture* these differ
   by 12.5%, which is a detail. For a *generalist* they differ by a **factor of
   six**, because a generalist pays for all its atoms on every image while a
   mixture pays for one expert's. The registration did not notice. Kill switch 3
   is decided differently under each reading, so choosing between them now —
   after the conflict is visible but before any accuracy is known — would still
   be choosing which question to ask. **Both are run**: arm (c1) at the
   mixture's per-image cost and arm (c2) at the mixture's atom sum, and kill
   switch 3 is evaluated separately against each. This is strictly harder for
   the mixture than running either alone, because the mixture must beat a
   generalist that is allowed six times its inference budget.
3. **Arm (e), traffic-inverse sizing, is added as a second structure-matched
   null, and kill switch 4 with it.** Reading the corpus revealed a confound
   §7.10 does not control. Under a MAC match, capacity on a high-traffic domain
   is expensive and capacity on a low-traffic one is cheap, so **any** rule that
   moves atoms off the big domains buys more atoms in total. Quickdraw holds
   **29.46%** of train rows and has the lowest measured rank, so rank-sizing and
   traffic-avoidance point the same way there, and arm (b) could win for a
   reason with nothing to do with effective rank. **Arm (d) does not control
   this**: a Dirichlet draw is *uncorrelated* with traffic, not *anti-correlated*
   with it. Arm (e) sets `A_e ∝ 1/f_e`, which maximises the confound while
   carrying no rank information whatsoever. Kill switch 4: **if arm (e) matches
   arm (b), the operand is measuring traffic-weighted MAC arbitrage and not
   effective rank**, and the same reading as kill switch 2 applies. Adding a
   *harder* null before measurement tightens the test and is permitted; removing
   one, or adding one after seeing the result, is not.
4. **The sample floor is read per fitted dimension, and the stricter reading is
   reported beside it.** §5.3's floor binds per expert on that expert's own
   rows, per design item 6. The quantity used is `n_e / (4·A_e) ≥ 10`, which is
   what M103 computed. §5.3's phrase "applied per class" was written for
   components fitted *separately per class*, where each class's component sees
   only that class's rows; **this head is a single multi-output linear map**, so
   all 345 outputs share one design matrix and one Gram inverse and fitting 345
   outputs instead of one consumes no additional degrees of freedom per row.
   The reading is nonetheless arguable, so the per-class ratio is **computed and
   reported for every expert of every arm**. On a 345-class corpus that stricter
   reading is met by **no arm at any budget**, and it is met by no arm *equally*,
   so it cannot favour one allocation rule; what it does bound is how well any
   *absolute* accuracy here is estimated, and §7.10's operand is a difference
   between arms, not an absolute.
5. **The reported model is refitted on every row the expert owns, and the
   validation split is used only to choose the constant.** **[added before
   execution, after the smoke run exposed the defect it fixes.]** The atom cap
   in item 4 is computed from an expert's **full** row count `n_e`, but the
   first implementation fitted the reported model on the 90% left after the
   validation split. A capped expert could therefore still fall below the
   floor, and in the smoke run it did: every cap was respected and **two of arm
   (b)'s six experts were voided anyway**. A guard that does not guard is worse
   than no guard, because it is read as one. The fit is now two solves from a
   single accumulator — a **selection** model on the first 90%, which is what
   the held-out rows score, and a **final** model on 100%, which is what test
   rows score. The held-out rows are already encoded and standardised at that
   point, so this costs **no additional encode pass and no additional memory**.
   The constant is still chosen on rows the selection model never saw, and
   `n_e / (4·A_e) ≥ 10` is now enforced **exactly** by the cap rather than
   approximately. Refitting on the full training split after selecting a
   hyper-parameter on a held-out part of it is standard practice and touches no
   test row; what is new here is only that the floor and the cap now agree.
6. **The `seconds` field in M104's evidence is NOT an operand and may never be
   quoted as one.** **[registered during execution, on discovering the
   contamination — not after reading it.]** The M104 process did not have the
   machine to itself: the M107 pixel pre-materialisation ran alongside it for
   roughly two hours, and two M107 smoke runs ran during it as well. Every
   per-arm `seconds` value therefore mixes this milestone's work with another's
   and measures the scheduler, not the allocation rule. This costs M104 nothing,
   because its operand is **test accuracy** and its compute ledger is
   **analytic** — `training_macs` and `rank_measurement_macs` are counted, not
   timed, so neither is affected by what else the machine was doing. §7.14 marks
   its own timing the same way under restriction 5; the difference is that
   M107's runner carries the marker in the payload and M104's does not, so the
   prohibition is recorded here instead. **No efficiency, cost or speed claim
   about any M104 arm may rest on a wall-clock figure.** Had M104's operand been
   wall-clock, the correct response would have been to stop and rerun it alone
   rather than to disclose and continue.
7. **M107 is NOT run concurrently with M104, and the reason is recorded because
   the opposite was nearly done.** **[registered during execution.]** M104 was
   measured occupying only **7.1 of 16** logical cores, which invited filling the
   remainder with §7.14's dense comparator and saving most of a day. It was
   rejected. Both sealed configs pin `torch_threads` (and M107's `onnx_threads`)
   at **16**, so running them together would place over thirty spinning threads
   on sixteen cores and would most likely make the pair finish later than
   running them in sequence. Avoiding that needs the thread counts lowered — and
   they live in each config's **`numerics`** block, which the program treats as
   part of the specification rather than as an execution detail, because thread
   count changes floating-point reduction order. **Editing a sealed numerics
   block for scheduling convenience is not a trade this program makes**, and the
   saving is wall-clock only: it would buy no additional evidence and answer no
   additional question. M107 therefore starts when M104 finishes.

A further consequence follows from item 4 and is recorded because it runs against
the direction a reader would expect. The floor caps clipart at **838** atoms and
infograph at **900**, the two smallest domains — and those are also the two
*highest*-rank domains. Arm (b)'s allocation at 512 atoms hits **no cap**; arms
(d) and (e) both do, and the redistribution pushes them toward high allocations
on exactly the domains arm (b) chooses to favour. **The floor therefore makes
the nulls more like the treatment, not less**, and kill switches 2 and 4 are
harder for arm (b) to survive because of it, not easier.

**Registered prediction, recorded before measurement.** Arm (b) beats arm (a),
**and the margin is concentrated in quickdraw and sketch** — the two domains
whose measured rank sits below the control. §2.9.7 probe 3 places the other four
domains *above* the mixed control, so the mechanism predicts little or nothing
there. **If the margin is uniform across all six domains, the stated mechanism
is wrong even if the aggregate numbers favour arm (b)**, and the plan requires
that to be reported as a mechanism failure rather than an accuracy success.

**Kill switch 1.** If arm (b) does not beat arm (a) by more than arm (a)'s own
seed spread, at three seeds, then **rank-sizing buys nothing at matched compute**
and the direction is refuted. That is the headline under §11.1 and may not be
reported as a footnote.

**Kill switch 2.** If arm (d) — a random allocation with the same heterogeneity
and the same total — matches arm (b), then the win belongs to **heterogeneity
itself and not to effective rank**, the instrument is not doing the work it is
credited with, and the contribution claimed for this direction does not exist.
Registered now so it cannot be reported later as a curiosity.

**Kill switch 3.** If arm (c), the single generalist, matches the best mixture
arm, then partitioning buys nothing at this scale and M105 and M106 are not
worth running. **[amended in place before execution: evaluated separately
against arm (c1), the generalist at the mixture's per-image inference cost, and
arm (c2), the generalist at the mixture's plain atom sum, which costs six times
as much per image. Either one matching the best mixture arm fires this switch.]**

**Kill switch 4.** **[added before execution, with arm (e).]** If arm (e) — a
traffic-inverse allocation carrying **no rank information at all** — matches arm
(b), then the operand is measuring **traffic-weighted MAC arbitrage** rather than
effective rank, and the same reading as kill switch 2 applies: the contribution
claimed for this direction does not exist. The reason this null is necessary is
given in execution-time amendment 3 above.

**Registered restrictions.**

1. **Corpus isolation.** M104 runs on DomainNet. **No M104 figure may be compared
   to any M103 or §2.9.3 CIFAR-10 figure in either direction** (R7, prohibition
   24, §10.2 restriction 1).
2. **No novelty.** Mixtures of experts, conditional computation, effective rank
   and spectral sizing are all established (§8.10). M104's contribution is a
   **comparison of allocation rules**, not an invention of any component, and
   §11.2 item 22's form applies.
3. **The instrument is not this program's.** RankMe is Garrido et al.'s and is
   used unmodified. No M104 document may present effective-rank measurement as a
   contribution of GEODE (prohibition 25).
4. **Oracle routing is stated in every sentence** that quotes an M104 figure, per
   design item 3.
5. **Resolution disclosure.** §2.9.7 probe 3's spread is measured at 32×32 and
   its magnitude is resolution-dependent. If M104 runs at 32×32 it inherits that
   limitation and must state it beside the allocation table.
6. **Bars re-verified at execution time**, per §8.6 item 8.
7. **The head is a multi-output ridge, and that is a change from M103.**
   **[added before execution.]** M103 used a multinomial logistic head. M104 uses
   a single multi-output ridge solved from the normal equations, for three
   reasons recorded before the run: it is **exactly deterministic**, so no arm
   can differ from another by where an optimiser stopped; it is **closed form**,
   so "trained to convergence" is not a judgement call; and on **345 classes** an
   iterative multinomial fit costs `O(n·d·C)` *per iteration*, which would have
   forced a train subsample — and a train subsample lowers the §5.3 cap that
   arm (b)'s largest expert is bound by, which is the one thing this milestone
   cannot afford to lower. The same head family, the same grid and the same
   chosen constant are used by every arm and every expert; the constant is
   chosen **once, on the null arm (a), at the first seed**, so arm (b) never gets
   a constant tuned to itself. Restriction 1 already forbids any comparison to
   M103, so the change costs no comparability that was permitted anyway.

**Result — measured at three seeds, oracle routing. Four of the five kill
switches fired. Rank-sizing is refuted.** **[written after the sealed run;
`logs/results/v15/m104_experts/evidence.json`, corpus digest `81099916e5036d1c`,
409,832 train rows and 100,000 test rows, routing `oracle_domain_label`.]**

Pooled test accuracy under oracle routing, averaged over seeds 11, 23 and 37, at
the 512-atom row-weighted budget:

| arm | pooled accuracy (oracle) | plain atom sum |
|---|---|---|
| (a) uniform | **24.22%** | 3,072 |
| (b) rank-sized | **22.47%** | ~3,416 |
| (c1) generalist, MAC-matched | **18.12%** | 512 |
| (c2) generalist, atom-matched | **24.51%** | 3,072 |
| (d) random-sized | **23.54%** | ~3,813 |
| (e) traffic-inverse | **23.53%** | ~3,870 |

**Kill switch 1 fired.** Arm (b) scored **22.47%** against arm (a)'s **24.22%**
under oracle routing — a margin of **−1.76 pp** against arm (a)'s own seed
spread of **0.16 pp**. Rank-sizing does not merely fail to beat uniform
allocation at matched compute; it loses to it by more than ten times the seed
spread. Under §11.1 this is the headline of M104 and is not reportable as a
footnote: **sizing experts by effective rank buys nothing, and costs
accuracy.**

**Kill switch 2 fired.** Arm (b) at **22.47%** against arm (d), a *random*
allocation with the same heterogeneity and the same row-weighted total, at
**23.54%** under oracle routing — **−1.07 pp**, far inside the 0.50 pp
tolerance and on the wrong side of it. The rank measurement is not merely
uninformative; a random draw with the same shape does better.

**Kill switch 4 fired.** Arm (b) against arm (e), the traffic-inverse
allocation that carries **no rank information whatsoever**, at **23.53%** under
oracle routing — **−1.06 pp**. Arm (e) exists precisely because a heterogeneous
allocation correlated with domain traffic can arbitrage row-weighted MACs
without knowing anything about the data's geometry. It beat the arm that
measured the geometry.

**Kill switch 3 fired on (c2) and did not fire on (c1).** The atom-matched
generalist — one dictionary of 3,072 atoms, no partition, no router — scored
**24.51%** under oracle routing, **above every mixture arm including the
uniform one**. The MAC-matched generalist at 512 atoms scored **18.12%**, so
the atom budget is doing real work and the comparison is not vacuous. The
reading is that **at this scale the partition itself buys nothing**: six
domain experts holding 512 atoms each are beaten by a single generalist holding
3,072, even when the mixture is handed its domain labels for free.

**The registered mechanism is not merely unsupported — it is inverted.** The
prediction, recorded before measurement, was that arm (b)'s margin over arm (a)
would be *concentrated in quickdraw and sketch*, the two domains whose measured
effective rank sits below the mixed control. Under oracle routing the margin on
those two domains averaged **−4.84 pp**, while the four domains where the
mechanism predicted "little or nothing" averaged **+1.10 pp**:

| domain | uniform (oracle) | rank-sized (oracle) | margin | RankMe | atoms given |
|---|---|---|---|---|---|
| clipart | 26.56% | 28.01% | **+1.45 pp** | 77.8 | ~744 |
| infograph | 8.36% | 8.79% | **+0.43 pp** | 72.3 | ~693 |
| painting | 14.67% | 15.43% | **+0.76 pp** | 80.3 | ~767 |
| quickdraw | 37.07% | 29.14% | **−7.93 pp** | **10.6** | **~104** |
| real | 23.25% | 25.02% | **+1.77 pp** | 79.8 | ~761 |
| sketch | 15.02% | 13.26% | **−1.76 pp** | 36.4 | ~348 |

**Why it inverted, stated as a mechanism and not an excuse.** Effective rank
measures how many directions the *input* distribution occupies. It says nothing
about how many directions are needed to separate **345 classes** inside that
subspace. Quickdraw has the lowest effective rank of the six domains — line
drawings are geometrically simple — so the rule handed it **~104 atoms**, about
3.4% of the pool. Those atoms carry `4·A ≈ 416` features to separate 345
classes: barely more than one feature per class. Quickdraw is also **29.5% of
the training corpus** and, under uniform allocation, the **most accurate**
domain at 37.07% under oracle routing. Starving the largest and most learnable
domain cost 7.93 pp on nearly a third of the test traffic, and no gain spread
across the other four recovered it.

The general statement, which is what M104 actually contributes: **effective rank
is a property of the input distribution; the capacity a domain needs is a
property of its label structure.** An allocation rule that reads only the first
is blind to the second, and on DomainNet the two point in opposite directions
— the simplest inputs carry the same 345 labels as the richest. §2.9.7 probe
3's rank spread was real and reproduced here (RankMe 10.6 to 80.3, a factor of
7.6); the error was in believing that spread licensed an allocation.

**Consequences, applied.** §7.11 (M105, the intrinsic router) and §7.12 (M106,
additive growth) are **conditional on M104 surviving its kill switches**. It did
not. **M105 and M106 do not proceed.** Pursuing a router for a partition that
loses to its own absence, or additive growth over experts that lose to one
generalist, would be building on a refuted premise. This is recorded as a
direction closed, not as a result pending.

**What survives.** Nothing in this refutes sparse dictionaries as a
representation — arm (c2), a single 3,072-atom generalist, is itself a sparse
model and is the best arm here under oracle routing. What is refuted is
**rank-guided expert sizing** and, at this scale, **domain partitioning as a
way to spend a fixed atom budget**. The question of whether a sparse dictionary
can match a dense network at matched inference cost is untouched by M104 and is
measured by §7.14 (M107), which is why M107 is registered independently of this
gate.

**Restrictions honoured.** Every figure above is stated under **oracle routing**
(restriction 4); no figure here is compared to any M103 or §2.9.3 CIFAR-10
figure (restriction 1, prohibition 24); no wall-clock figure is quoted
(execution-time amendment 6); RankMe is Garrido et al.'s instrument used
unmodified (restriction 3); and the 32×32 resolution limitation of §2.9.7 probe
3 is inherited here (restriction 5) — quickdraw's low measured rank may be
partly a downsampling artefact on line art, which is disclosed and was not
measured.

---

### 7.11 M105 — does the intrinsic router survive contact with the system? **[new — registered after the M103 prior-art audit]** {#m105}

**Conditional on M104 surviving all three kill switches.** A sizing win measured
under oracle routing is not a system, and §11.2 already forbids reporting one as
though it were.

**Question.** M104 assumes the router. M105 pays for it. Does a router built from
**four training-free scalars** (§2.9.7 probe 4) recover enough of the oracle's
accuracy to leave the sizing win intact **after its own compute is charged**?

**Design.**

1. **Arms, all on M104's best mixture configuration.**
   * **(a) Oracle routing** — the upper bound, carried from M104.
   * **(b) Intrinsic router** — the four spectral scalars, whose *features* are
     never fitted; only a 4-dimensional decision rule is.
   * **(c) Learned router on pooled features** — 2048-dimensional, the
     conventional construction, and the arm that says what the intrinsic router
     costs in accuracy.
   * **(d) Random routing** — **the null**. A router that cannot beat random
     assignment carries no signal regardless of its probe accuracy.
2. **Operand.** End-to-end accuracy **and** total inference MACs **including the
   router**, per §5.11.1. Neither is readable without the other.
3. **The routing tax is reported with the headline.** The gap between (a) and (b)
   is the cost of not having an oracle, and it binds every M105 statement exactly
   as C103.3 binds C103.1. A document that quotes M105's accuracy without the
   (a)−(b) gap misstates the evidence.
4. **Seeds.** Three (11, 23, 37).
5. **Sample floor and compute ledger** as in §7.10 items 6 and 7.

**Registered prediction.** (b) beats (d) decisively and loses to (c) on accuracy
while winning on router compute. §2.9.7 probe 4 measures 0.49278 against 0.69056
on a *probe*, which is not a system figure and does not transfer.

**Kill switch.** If (b) does not beat (d) by more than the seed spread, the
intrinsic fingerprint carries no usable signal at system level and the
architecture reverts to a conventional learned router — which §2.9.7 probe 4
shows costs 2.8× the assignment stability, and §7.12 then measures what that
costs.

**Registered restrictions.** §7.10's restrictions 1, 2, 3, 5 and 6 apply
unchanged. Additionally: **§2.9.7 probe 4's 0.49278 is a probe figure under
prohibition 23** and may not be quoted as M105's expected accuracy.

---

### 7.12 M106 — does the construction actually compose additively? **[new — registered after the M103 prior-art audit]** {#m106}

**Conditional on M105 surviving its kill switch.** This is the milestone that
tests **the plan's thesis** rather than the artifact, and it is the one whose
failure would matter most.

**Question.** §3.1 asks whether a model *grown* one component at a time can
occupy a point no *fitted* model reaches. M103 asked that of one dictionary.
M106 asks it of a system: **can expert K+1 be added without refitting experts
1..K, and without degrading them?**

**Why this is the load-bearing test.** If adding an expert forces the earlier
ones to be refitted, total construction cost is **O(K²)** and the additive
efficiency claim dies at the architecture rather than at any component. §2.9.7
probe 4 measures that a conventional router moves **9.341%** of its assignments
under growth against the intrinsic router's **3.336%**. That is a measurement of
*drift*, not of *cost*. M106 measures the cost.

**Design.**

1. **Procedure.** Build the M105 system on four domains. Then add the fifth and
   sixth **without refitting the first four experts and without re-measuring
   their allocations**. Refit only the router's decision rule.
2. **Operand 1 — backward degradation.** Accuracy on the original four domains,
   before and after growth, per domain and pooled.
3. **Operand 2 — the cost of the addition.** Training MACs spent adding experts
   5 and 6, against the training MACs of building all six from scratch. The
   additive claim is a claim about this ratio.
4. **Arms.** (a) grown incrementally, as above; (b) **built from scratch on all
   six** — the null that says what was lost by growing; (c) grown incrementally
   **with full refitting** — the O(K²) construction, which bounds what arm (a)
   gives up.
5. **Order sensitivity.** Growth is run in **two different domain orders**, both
   registered now: `{clipart, infograph, painting, real} → {quickdraw, sketch}`
   and `{quickdraw, sketch, real, painting} → {clipart, infograph}`. A
   construction whose result depends on presentation order is not additive in the
   sense §3.1 means, and the second order is registered specifically because it
   introduces the two low-rank domains **first**.
6. **Seeds.** Three (11, 23, 37).

**Kill switch.** If accuracy on the original four domains degrades by more than
the seed spread under either order, **the construction is not additive** and the
O(K) claim fails. That is a substantive negative result about the plan's central
thesis and §11.1 requires it as the headline.

**Second kill switch.** If arm (a) falls short of arm (b) — built from scratch —
by more than the seed spread, then growing costs accuracy, and the efficiency of
growth must be quoted against that loss in the same sentence, per §5.11.1.

**Registered restrictions.** §7.10's restrictions 1, 2, 3, 5 and 6 apply
unchanged. Additionally: **no M106 figure may be described as demonstrating
continual learning or the absence of catastrophic forgetting.** Those are
established fields with established benchmarks (van de Ven & Tolias, §8.10), M106
runs none of them, and §11.2 item 22's form applies with full force.

---

### 7.13 The dense comparator, and a gap in the program's own record **[new — registered after the M103 prior-art audit]** {#dense-comparator}

**Recorded as a registration defect, not as a milestone.** Q2 (§3.2) asks whether
a sparse model is more efficient **than the best dense model at matched
accuracy**. M102 measured a head. M103 measured a dictionary against other
dictionaries. **Neither compared anything to a dense network**, and no v15
milestone as registered does. The program has therefore never asked its own Q2 in
the form Q2 is written.

The cache at `GEODE_CACHE_DIR` holds **DINOv2 small, base and large ONNX
exports**, which supply a dense ladder with computable FLOPs on the same corpus.
Registering the comparator is cheap; what is expensive is discovering after
M104–M106 that the comparison was never available.

**Registered consequence.** **No M104, M105 or M106 document may state or imply
an efficiency result against dense networks** until a dense comparator has been
measured on the same corpus at matched accuracy. Until then the admissible claim
is efficiency **relative to a uniform mixture**, which is what M104's null
actually is. This restriction is registered before M104 runs so that it cannot be
relaxed after seeing a favourable number.

**[The comparator is now designed, in §7.14, and registered while M104 was still
running and before any M104 accuracy existed. §7.13 remains the record of the
defect; §7.14 is the measurement that closes it. Prohibition 27 is unchanged and
stays in force until §7.14 has actually been run.]**

---

### 7.14 M107 — the dense comparator: what does a sparse model's compute buy against a dense one's? **[new — registered while M104 was running, before any M104 accuracy existed]** {#m107}

**Question.** §3.2 Q2 asks whether a sparse, inspectable model is more efficient
than **the best dense model at matched accuracy**. §7.13 records that no
milestone in this program has ever asked it. M107 asks it: on one corpus, with
one head, one train split and one test split, **how does test accuracy per
inference MAC compare between a frozen sparse dictionary code and a frozen dense
transformer feature?**

**Why it is answerable now.** The cache holds **DINOv2 small, base and large ONNX
exports** whose input resolution is dynamic (`floor(H/14)·floor(W/14) + 1`
tokens), so the dense side is not one point but a **ladder** — three model sizes
and, for the small model, a resolution sweep. The sparse side is a ladder in atom
budget. Two ladders on the same axes is a comparison; two points is an anecdote.

**The comparison is a representation comparison, and only that.** Both families
are run in the identical protocol: **frozen features → the same multi-output
ridge head → the same penalty grid → the same selection rule → the same test
rows**. The *only* thing that differs between arms is what produced the features.
This is the standard linear-probe protocol DINOv2 is itself evaluated under, so
the dense arm is used at its intended operating point and not handicapped.

**Design.**

1. **Corpus.** DomainNet, six domains, 345 classes, one fixed stratified
   subsample shared by **every** arm: **400 train rows per class** and **100 test
   rows per class**. The subsample exists because DINOv2-large at 224 costs about
   **4.5 images per second** on this machine's sixteen cores, and the full
   409,832-row train split would cost it more than a day. The train size is
   chosen so the §5.3 floor still permits a sparse generalist of
   `138,000 / (4·10)` = **3,450** atoms, which is above the top of the registered
   sparse ladder; the ladder is therefore not truncated by the floor.
2. **Dense arms.**
   * **(d1) DINOv2-small at 224** — 257 tokens, width 384.
   * **(d2) DINOv2-base at 224** — 257 tokens, width 768.
   * **(d3) DINOv2-large at 224** — 257 tokens, width 1024.
   * **(d4) DINOv2-small at 140, 98, 70 and 42** — the resolution sweep, which is
     the only way the dense ladder reaches down into the sparse ladder's MAC
     range at all. **Disclosed as out of distribution**: DINOv2 is trained at 224
     and its position embeddings are interpolated at every other resolution, so
     these points understate what a dense network *designed* for that budget
     would achieve. They are reported as a **lower bound on dense**, never as
     "the dense curve".
   * **(d5) DINOv2-small at 224 on the 32×32 tensors, upsampled** — the
     **information-matched** dense arm, and the one that isolates architecture
     from pixels. See design item 2b.
   * Features are the **CLS token concatenated with the mean of the patch
     tokens**, which is DINOv2's own linear-evaluation protocol, giving `2·width`
     features.
2b. **The resolution asymmetry is real, runs in dense's favour, and is measured
   rather than argued about.** Arms (d1)–(d4) read the **original-resolution**
   DomainNet images and resize them for the transformer. The sparse arms read the
   **32×32** downsample this whole program uses, because that is the pipeline
   §2.9.7's probes and M103 and M104 all measured and changing it would make M107
   incomparable to them. A dense arm that sees more pixels than the sparse arm is
   **not** a controlled comparison of architectures — it is a comparison of
   systems, which is what a practitioner actually chooses between, so it is the
   headline. Arm (d5) supplies the control: the identical DINOv2-small at the
   identical 224 tokens, fed the identical 32×32 tensors the sparse arms see,
   bilinearly upsampled. **(d1) minus (d5) is what the extra pixels are worth;
   (d5) minus the sparse ladder is what the architecture is worth.** Reporting
   only one of the two would be choosing the answer.
3. **Sparse arms.** The M104 pipeline — 6×6 patches, stride 1, ZCA whitening,
   2×2 pooling — as a **single generalist** dictionary at budgets
   **128, 256, 512, 1024, 2048, 3072**, and as the **uniform six-expert mixture
   under oracle routing** at the same row-weighted budgets. The mixture is
   carried because it is M104's null and because oracle routing is an **upper
   bound** that flatters the sparse side; restriction 4 of §7.10 applies to every
   sentence quoting it.
4. **Operand.** Test accuracy against **inference MACs per image**, both
   families on one plot and one table. Neither number is readable without the
   other (§5.11.1). Transformer MACs are computed analytically from tokens,
   width, depth and the MLP ratio, not timed; wall-clock is recorded separately
   and is not an operand.
5. **Head.** Identical to M104's: one multi-output ridge from standardised
   features to one-hot targets, solved from the normal equations, the same
   five-constant grid, the constant chosen **once, on the sparse generalist at
   the smallest budget**, and applied unchanged to every arm. Choosing it on the
   *sparse* side is against interest: it denies the dense arms a constant tuned
   to them.
6. **Sample floor.** §5.3 as amended by §7.10 execution-time amendments 4 and 5,
   read per fitted dimension on the rows each arm actually fits. DINOv2-large's
   2,048 features on 138,000 rows gives **67.4** rows per fitted dimension; the
   sparse generalist at 3,072 atoms gives **11.2**. Both clear the floor of ten,
   and **the sparse side clears it by less**, which is disclosed rather than
   averaged away.

**The caveat that binds every M107 sentence.** DINOv2 is pre-trained on
**LVD-142M**, 142 million curated images. The sparse dictionary is drawn from
**this corpus's own train patches and nothing else**. The two are therefore *not*
matched on training data, by about three orders of magnitude, and no M107
sentence may omit this. It cuts both ways and both directions must be stated:
the dense arm has an enormous data advantage the sparse arm does not, **and**
that advantage is exactly what a practitioner gets for free when they download
the weights, so refusing to count it would be answering a question nobody asked.
Q2 asks about **inference** efficiency at matched accuracy, and that is the only
question M107 answers.

**Registered prediction, recorded before measurement.** The dense ladder
dominates the sparse ladder in accuracy at every MAC budget where the two
overlap, and the sparse ladder does not reach the dense ladder's accuracy at any
budget within reach of this corpus. **This prediction is against the program's
own thesis** and is registered anyway, because a program that only registers
predictions it wants to be true is not registering anything.

**Kill switch 1.** If the dense curve is above the sparse curve at every
overlapping MAC budget, then **§3.2 Q2's efficiency claim is refuted at this
scale on this corpus**, the admissible reading of M104–M106 collapses to
"efficiency relative to a uniform mixture", and that is the headline under §11.1.
It may not be reported as a footnote, a limitation, or future work.

**Kill switch 2.** If the sparse curve is above the dense curve at some budget,
the admissible claim is bounded to **that budget, that corpus and that accuracy
level**, must quote the accuracy achieved alongside the MAC figure, and must
still carry the LVD-142M caveat and the oracle-routing caveat. A crossing at an
accuracy nobody would deploy is not an efficiency result and may not be reported
as one.

**Kill switch 3.** If the sparse generalist beats the sparse mixture under
**oracle** routing at matched MACs on this corpus, then M104's partition is not
buying anything here either, and M107 must say so beside M104's own kill switch
3 rather than leaving the two unreconciled.

**Registered restrictions.**

1. **No novelty.** Linear probing of frozen features is the standard evaluation
   protocol for self-supervised vision models and is not this program's idea.
   DINOv2 is Oquab et al.'s, used unmodified from a published ONNX export, and
   §11.2 item 22's form applies (§8.10).
2. **The resolution sweep is a lower bound on dense, never the dense curve**, per
   design item 2.
3. **Every sentence quoting a sparse mixture figure states oracle routing**,
   per §7.10 restriction 4.
4. **Every sentence quoting any M107 figure states the LVD-142M asymmetry.**
5. **Analytic MACs only**, per design item 4; no wall-clock comparison between
   families, because the two run on entirely different kernels and the ratio
   would measure onnxruntime against numpy rather than the models.
6. **Prohibition 27 is lifted only for the specific comparison M107 measures** —
   this corpus, these budgets, this protocol — and remains in force for every
   other dense-efficiency statement.
7. **Arm (d1) and arm (d5) are reported together or not at all**, per design
   item 2b. Quoting the pixel-matched arm without the information-matched arm
   overstates the dense advantage; quoting the information-matched arm without
   the pixel-matched one understates it. Neither is admissible alone.
8. **The subsample is disclosed with every figure.** M107 fits on 138,000 train
   rows, a third of DomainNet's train split, because DINOv2-large at 224 costs
   about 4.5 images per second on this machine. Both families are affected
   identically, but absolute accuracies are lower than a full-split fit would
   give and no M107 accuracy may be quoted as this program's best.

**Execution-time amendments, all registered before any M107 accuracy existed.**
Recorded here in the form §7.10 uses, so the record shows what the instrument
does rather than what the design item said it would.

1. **The resolution sweep gains 28 and 56.** Design item 2 registers
   {140, 98, 70, 42}. Computing the analytic MACs *before running anything* —
   from geometry read off the ONNX graph, with no accuracy in hand — showed that
   the sparse ladder tops out at **255 million** MACs while the registered
   sweep's cheapest point sits at **216 million**, so exactly **one** sparse
   budget would have had a dense reference at or below it. A comparison with one
   comparable point is an anecdote. Resolution **28**, at **108 million**, makes
   it **two**; resolution **56**, at **368 million**, puts a dense point
   immediately above the sparse ceiling so the gap at the top of the sparse
   ladder is read rather than extrapolated. **What this amendment does not do is
   make the sparse side's job easier at any budget that was already
   comparable**: every sparse budget that had a dense reference before this
   amendment has the same one after it. **The overlap is two sparse budgets wide
   and that is a limitation of M107, disclosed here rather than discovered
   later.** It is set by the corpus, by the §5.3 floor that caps the sparse
   ladder near **3,450** atoms, and by what DINOv2 costs at any resolution it
   can actually use. A dense point cheap enough to widen the overlap further
   would have to sit below one patch of image, and handing the sparse side a
   degenerate opponent is not a measurement.
2. **The §5.3 floor voids an arm rather than being reported beside it.** A
   voided arm contributes to no kill switch and appears on no curve. This is
   **§7.10 execution-time amendment 5's lesson applied in advance**: there, a cap
   computed from a row count the fitted model did not use silently guarded
   nothing. The runner also **aborts** if the arm design item 5 chooses the head
   constant on is itself void, because every other arm inherits that constant.
   The smoke configuration is deliberately sized *below* the floor so the
   voiding path is exercised rather than assumed.
3. **The instrument proves, rather than asserts, that the two families see the
   same images.** Every dense figure M107 produces is a claim about the images
   the sparse arms saw, and that claim rests entirely on the decoded 32×32 cache
   and the parquet stream enumerating rows in the same order. At startup the
   runner decodes the first **64** selected rows of each split straight from the
   parquet and requires them **bitwise** equal to the cached tensors; a mismatch
   aborts. The check was itself negative-controlled: flipping one byte of one
   cached image makes it fire.
4. **One dictionary seed on the sparse side, not M104's three.** M104 spends
   three because its operand is a difference between allocation rules and must
   survive the draw; M107's operand is a difference between *families* expected
   to exceed an order of magnitude, and the dense side has no seed at all
   because its weights are downloaded. Three seeds would triple the cheap half
   of the run and leave the expensive half untouched. **Unlike amendments 1–3
   this does not make the milestone harder**, and it is recorded as a limitation
   of every sparse M107 figure rather than folded in with them.
5. **The mixture ladder runs only at 128 and 256 atoms, and §5.3 is why.**
   Counting the subsample's rows per domain — arithmetic, before anything ran —
   showed `clipart` holds **11,224** of the 138,000 train rows. A mixture expert
   fits on its own domain's rows and spends `4·A` features, so clipart clears
   ten rows per fitted dimension at 256 atoms (**10.96**) and fails at 512
   (**5.48**). Design item 3 registers the mixture at all six budgets; four of
   them would be **voided on arrival**. **Kill switch 3 is therefore decidable
   at two budgets instead of six**, which narrows M107 and is registered here
   rather than discovered in the output — and every sentence reporting kill
   switch 3 must say so. The *generalist* ladder is untouched and still clears
   the floor at 3,072 atoms (**11.23**), so M107's actual question, dense
   against sparse, is measured across its whole range. Capping each expert at
   its own floor and redistributing, as §7.10 does, was considered and
   **rejected**: it would turn a uniform mixture into an allocation experiment,
   which is M104's operand and not M107's. This is the subsample's doing and is
   disclosed under restriction 8; at the full train split the mixture would
   reach 512 and beyond.
6. **An inadmissible configuration can no longer write where the sealed one
   writes.** **[added before the sealed run, after the smoke run did exactly
   that.]** The M107 smoke configuration was executed with `--config` but
   without `--output`, so it wrote a **2,760-row** `evidence.json` straight into
   `logs/results/v15/m107_dense/` — the sealed directory. Nothing had read it
   yet only because §7.14's verifier block does not exist until M107 lands; the
   block would then have recomputed its figures from **smoke numbers** and
   passed, because the verifier reads that path by convention and had no way to
   know what produced the file. Three changes close it. The smoke config now
   carries the same `_smoke_note` self-declaration M104's does. The runner
   **refuses to start** when a config declaring itself inadmissible is pointed
   at the sealed output directory. And every `evidence.json` now records
   `admissible_as_evidence` and the `config_file` that produced it, so a reader
   — human or verifier — can tell what it is holding without inferring it from a
   path. The contaminated directory was deleted. This is registered rather than
   quietly fixed because for several hours the sealed path held numbers that
   were not the milestone's, and §11.2 item 23 exists precisely to stop that.

**Result — kill switch 2 fired, kill switch 1 did not, and the registered
prediction is refuted.** **[written after the sealed run;
`logs/results/v15/m107_dense/evidence.json`, corpus digest
`63f590097008f749`, 138,000 train rows and 34,500 test rows, 345 classes,
`admissible_as_evidence` true.]** §7.14 registered the prediction that the dense
ladder would dominate the sparse ladder at **every** overlapping MAC budget.
That prediction ran against this program's own thesis and was written down
precisely so that it could not be quietly softened afterwards. **It failed.**
The sparse generalist is above the best dense arm at or below its budget at both
of the two budgets where the comparison is decidable.

The ridge penalty **1.0** was chosen once on `s_generalist_128` — the sparse
side, per design item 5 — and applied unchanged to all eighteen arms. No arm was
voided. Every figure in the table below carries the two standing asymmetries of
§7.14: DINOv2 was pre-trained on **LVD-142M**, which this program's dictionaries
never saw, and the dense arms other than `d5` read original-resolution pixels
while every sparse arm reads 32×32. Both asymmetries favour the dense side.

| arm | family | analytic MACs per image | test accuracy |
|---|---|---|---|
| `d4a_small_28` | dense | 107,566,848 | **15.99%** |
| `d4b_small_42` | dense | 215,555,328 | **19.72%** |
| `d4c_small_56` | dense | 367,513,344 | **24.50%** |
| `d4d_small_70` | dense | 564,215,040 | **31.18%** |
| `d4e_small_98` | dense | 1,096,051,968 | **44.76%** |
| `d4f_small_140` | dense | 2,261,456,640 | **49.74%** |
| `d5_small_224_from_32` | dense, information-matched | 6,123,826,944 | **38.86%** |
| `d1_small_224` | dense, pixel-matched | 6,123,826,944 | **53.75%** |
| `d2_base_224` | dense | 23,161,757,184 | **61.30%** |
| `d3_large_224` | dense | 81,012,688,896 | **65.06%** |
| `s_generalist_128` | sparse generalist | 18,757,392 | **11.17%** |
| `s_generalist_256` | sparse generalist | 29,011,728 | **14.00%** |
| `s_generalist_512` | sparse generalist | 49,520,400 | **16.42%** |
| `s_generalist_1024` | sparse generalist | 90,537,744 | **18.64%** |
| `s_generalist_2048` | sparse generalist | 172,572,432 | **20.61%** |
| `s_generalist_3072` | sparse generalist | 254,607,120 | **21.52%** |
| `s_mixture_128` | sparse mixture, oracle routing | 18,757,392 | **16.59%** |
| `s_mixture_256` | sparse mixture, oracle routing | 29,011,728 | **18.71%** |

**Kill switch 1 did not fire.** Under the LVD-142M asymmetry and the resolution
asymmetry, both of which favour dense, the dense curve is **not** above the
sparse curve at every overlapping budget, so §3.2 Q2's efficiency claim is not
refuted at this scale on this corpus. This is the first head-to-head measurement
the program has ever made against a dense opponent it did not construct, and the
opponent did not win it outright.

**Kill switch 2 fired, at both decidable budgets.** Carrying the LVD-142M
asymmetry: the sparse generalist at **172,572,432** MACs scores **20.61%**
against `d4a_small_28`, the best dense arm at or below that budget, at
**15.99%** — a margin of **+4.62 pp**. At **254,607,120** MACs it scores
**21.52%** against `d4b_small_42` at **19.72%** — **+1.80 pp**. Under §11.1 this
is the headline of M107.

**The five bounds that travel with that headline, none of them optional.** The
fifth was added after the run, on recomputing the gate a second way; it is
registered in place rather than omitted, and it is the one that most nearly
undoes the headline.

1. **Two of six budgets are decidable; the other four are void, not won.** The
   sparse ladder's four cheapest rungs sit at 18.8 M to 90.5 M MACs, below
   `d4a_small_28`'s 107,566,848, so no dense arm exists at or below them. §7.14's
   admissibility note is explicit that this is **VOID and not a "not fired"**,
   and it is equally not a win. The program does not know what a dense model
   costing 19 million MACs would score, because it did not build one.
2. **The crossings are at accuracies nobody would deploy.** **20.61%** and
   **21.52%** on 345 classes, under the LVD-142M asymmetry. Kill switch 2's own
   registered text says a crossing at such an accuracy is not an efficiency
   result and may not be reported as one. It is reported here as what it is: a
   crossing, in a regime where both families are bad.
3. **Dense passes the sparse ceiling for 1.44× the cost.** The sparse ladder tops
   out at **21.52%** at **254,607,120** MACs. `d4c_small_56` reaches **24.50%**
   at **367,513,344** MACs, which is **1.44×** the sparse ceiling's budget and
   above everything the sparse family achieved at any budget, under the LVD-142M
   asymmetry. The crossing therefore lives inside a narrow window that closes
   before half an order of magnitude has passed. The sparse family reached no
   accuracy the dense family could not reach for less than half as much again.
4. **The window exists partly because the dense ladder is resolution-starved at
   its bottom end.** `d4a_small_28` and `d4b_small_42` feed DINOv2 28×28 and
   42×42 images — 4 and 9 patch tokens. The sparse arms read 32×32. The
   comparison in the crossing window is therefore not "sparse beats dense", it is
   "a 3,072-atom dictionary on 32×32 beats a 12-layer transformer given 4 to 9
   tokens", under the LVD-142M asymmetry. That is a real and registered
   comparison, and it is also a narrower statement than §3.2 Q2 asks about.
5. **The comparison rule lets the sparse arm outspend its opponent, and the
   margin shrinks when it may not.** **[added after the sealed run, on
   recomputing the gate a second way. Registered here rather than left out
   because it bounds a figure this document already quotes.]** Gate item 4
   compares a sparse point at M MACs against the best dense point **at or below**
   M, and the dense ladder steps by roughly 2× at its bottom end, so no dense arm
   exists between 107,566,848 and 215,555,328 MACs. Both crossings therefore
   compare a sparse arm against a **cheaper** dense arm: `s_generalist_2048`
   spends **1.60×** `d4a_small_28`'s MACs and `s_generalist_3072` spends
   **1.18×** `d4b_small_42`'s. Interpolating the dense curve to the sparse arm's
   own budget instead — which the gate does not do, and which is **not** a
   measured dense arm — the margins fall from **+4.62 pp** to about **+2.4 pp**
   linearly or **+2.1 pp** in log-MACs at 2,048 atoms, and from **+1.80 pp** to
   about **+0.6 pp** linearly or **+0.3 pp** in log-MACs at 3,072. **The crossing
   survives both interpolations at both budgets**, so kill switch 2's verdict is
   unchanged, but the 3,072-atom margin is thin enough that it would not survive
   a dense arm measured at that budget being half a point better than the
   interpolation predicts. No such arm was run. The honest statement is that the
   crossing is **established at the two measured budgets and marginal under
   interpolation at the upper one**, and any successor milestone should place a
   dense arm inside the window rather than argue about the segment between two.

**Kill switch 3 did not fire, and it reconciles with M104 rather than
contradicting it.** At matched inference MACs and under **oracle** routing, the
six-expert mixture beats the single generalist at both budgets amendment 5
leaves decidable: **16.59%** against **11.17%** at 18,757,392 MACs
(**+5.42 pp**), and **18.71%** against **14.00%** at 29,011,728 MACs
(**+4.71 pp**), both under oracle routing and under the LVD-142M asymmetry.
Amendment 5 caps the mixture ladder at 128 and 256 atoms per expert, so this
switch is decidable at **two budgets instead of six**, and no claim is made about
the other four. M104's kill switch 3 fired on exactly the opposite comparison:
there, the atom-matched generalist at 3,072 atoms scored **24.51%** under oracle
routing against the uniform mixture's **24.22%** under oracle routing. **The two
results are consistent, and together they say something neither says alone.**
Partitioning buys nothing per *parameter* — M104 — and a great deal per
*inference MAC* — M107 — because oracle routing makes five of the six experts
free at inference time. What M104 refuted was rank-guided *sizing*; what M107
measures is conditional *execution*, and the subsidy that makes it look good is
the oracle. Both figures are oracle figures and neither survives without a router
that does not yet exist (§7.11–§7.12 remain unrun).

**The information-matched control, reported with its pair per restriction 7.**
`d1_small_224` reads original-resolution pixels and scores **53.75%**;
`d5_small_224_from_32` runs the identical network on the identical 32×32 data the
sparse arms see, upsampled, and scores **38.86%**. The resolution asymmetry is
therefore worth **14.89 pp** to the dense side on this corpus — a measured
figure, not an estimate, and the first time this program has quantified an
asymmetry it had previously only disclosed. Both figures carry the LVD-142M
asymmetry. The control cuts both ways: at information parity the dense side still
reaches **38.86%** against the sparse ceiling's **21.52%**, but it spends
**6,123,826,944** MACs to do it, **24.1×** the sparse ceiling's budget. There is
no information-matched dense arm inside the crossing window, so kill switch 2's
verdict cannot be re-decided at information parity; that is a gap in M107's
design, disclosed here rather than argued around.

**The sparse ladder was truncated by the corpus, not by the method, and this is
the most actionable thing M107 found.** The sparse arms fit 4·A features, so
rows per fitted dimension falls as the ladder climbs: **269.53** at 128 atoms,
**134.77**, **67.38**, **33.69**, **16.85**, and **11.23** at 3,072 atoms. §5.3's
floor is **10**, and 138,000 train rows admit at most **3,450** atoms at that
floor. The ladder stopped one rung short of the floor **while still improving** —
its last step was **+0.91 pp** — whereas every dense arm sits between **67.38**
and **179.69** rows per fitted dimension and never came near the floor, because a
dense arm's feature count is fixed at 2·width no matter how much compute it
spends. **The two families do not pay for capacity in the same currency**: the
sparse side buys capacity with fitted dimensions, which the corpus rations, while
the dense side buys it with depth and resolution, which the corpus does not.
M107's crossing was therefore measured against a sparse ladder held at the
corpus's ceiling rather than the method's, and the honest reading is that M107
does not know where the sparse curve goes next. That is a registered question for
a successor milestone, not a claim.

**What M107 licenses, and what it does not.** Prohibition 27 is discharged for
this comparison and this comparison only, per restriction 6: on DomainNet at
32×32 with 138,000 train rows, a sparse dictionary is not uniformly dominated by
a dense transformer on the accuracy-versus-MACs plane, and there is a bounded
window — 172,572,432 to 254,607,120 MACs, at 20.61% to 21.52% accuracy, under the
LVD-142M asymmetry — where it is ahead. Every other dense-efficiency statement
remains prohibited. M107 does **not** license "sparse models are more efficient
than dense ones", does not license any claim about behaviour above 254,607,120
MACs, does not license any claim at all at the four void budgets, and does not
license a single sentence about deployment. It also does not license the mixture
figures without the word **oracle** beside them.

---

## 8. Prior art and lineage

Registered under §7 of `PRIOR_ART_AUDIT_v13.md` and bound by its §8.4
consequence. **This section makes no novelty claim of any kind and asserts no
absence of prior art.** Its purpose is to name the lineage each v15 milestone
sits inside and to record where the search failed. M88 measured this program's
search-instrument recall at **4 of 7**; §8.6 discloses what that means for what
follows.

### 8.1 Greedy additive approximation — the theory M96 and M97 sit inside

- **Jones, L.K. (1992).** _A simple lemma on greedy approximation in Hilbert
  space and convergence rates for projection pursuit regression and neural
  network training._ Annals of Statistics 20(1). The canonical greedy
  approximation lemma: for a target in the closed convex hull of a bounded
  dictionary, a greedy `T`-term approximation attains squared error `O(v_f²/T)`.
  **Verification status: not verified by fetch.** The theorem is
  existential over an oracle and states nothing about the tractability of
  selecting the next atom.
- **Barron, A.R. (1993).** _Universal approximation bounds for superpositions of
  a sigmoidal function._ IEEE Trans. Information Theory 39(3):930–945,
  DOI 10.1109/18.256500. Translates the greedy rate into neural approximation:
  mean squared error `O(C_f²/n)` for `n` units, with `C_f` the spectral first
  moment, **with no explicit dependence on ambient dimension**.
  **Verification status: not verified by fetch.** This is the source of §6.1 P1.
- **Friedman, J.H. & Stuetzle, W. (1981).** _Projection Pursuit Regression._
  JASA 76(376):817–823, DOI 10.1080/01621459.1981.10477610. The classical
  template for building `f(x) = Σ g_t(a_tᵀx)` one ridge function at a time. The
  paper's own limit is that the projection index optimisation is nonconvex and
  heuristic. **Verification status: not verified by fetch.**
- **Barron, A.R., Cohen, A., Dahmen, W. & DeVore, R.A. (2008).** _Approximation
  and learning by greedy algorithms._ Annals of Statistics 36(1):64–94,
  DOI 10.1214/009053607000000631. Extends the theory to relaxed and orthogonal
  greedy schemes and connects approximation rates to learning guarantees.
  **Verification status: not verified by fetch.**
- **Blum, A. & Rivest, R. (1989).** _Training a 3-node neural network is
  NP-complete._ Cited for the general hardness of globally training small
  networks. **Verification status: not verified by fetch.**

**Search failure disclosed:** we did **not** locate a verified canonical result
proving that the projection-pursuit next-atom oracle step specifically is
NP-hard. §6.1 P2 is therefore registered as an assumption of this program and
not as an inherited theorem.

### 8.2 Greedy sparse selection — the algorithm M96 implements

- **Mallat, S. & Zhang, Z. (1993).** _Matching Pursuits with Time-Frequency
  Dictionaries._ IEEE Trans. Signal Processing, DOI 10.1109/78.258082. Repeatedly
  select the atom most correlated with the residual. **Not verified by fetch.**
- **Pati, Y.C., Rezaiifar, R. & Krishnaprasad, P.S. (1993).** _Orthogonal
  Matching Pursuit._ Asilomar. Adds the refit-on-support step. **M96 step 3 is
  this step.** **Not verified by fetch.**
- **Tropp, J.A. (2004).** _Greed is Good: Algorithmic Results for Sparse
  Approximation._ IEEE Trans. Information Theory,
  DOI 10.1109/TIT.2004.834793. Recovery guarantees for orthogonal matching
  pursuit under coherence conditions. **Not verified by fetch.**
- **Mairal, J., Bach, F. & Ponce, J. (2012).** _Task-Driven Dictionary
  Learning._ IEEE TPAMI, DOI 10.1109/TPAMI.2011.156. Optimises dictionary and
  predictor jointly for a supervised loss rather than reconstruction. **This is
  the closest prior art to M96's discriminative-residual criterion** and the
  reason null (b) in §7.3 is mandatory. **Not verified by fetch.**
- **Jiang, Z., Lin, Z. & Davis, L.S. (2013).** _Label Consistent K-SVD._ IEEE
  TPAMI, DOI 10.1109/TPAMI.2013.88. Adds label consistency and classifier loss
  to K-SVD so atoms become discriminative. **Not verified by fetch.**

### 8.3 Constructive and additive model families — where M97 sits

- **Fahlman, S.E. & Lebiere, C. (1990).** _The Cascade-Correlation Learning
  Architecture._ Adds hidden units one at a time, freezing previous weights and
  training each new unit to correlate with residual error. **The most direct
  historical antecedent of M97.** **Not verified by fetch.**
- **Bengio, Y., Le Roux, N., Vincent, P., Delalleau, O. & Marcotte, P. (2005).**
  _Convex Neural Networks._ NeurIPS. Recasts single-hidden-layer networks as
  convex optimisation over an infinite dictionary of hidden units, with finite
  models as sparse extreme-point solutions. **Not verified by fetch.**
- **Cortes, C., Gonzalvo, X., Kuznetsov, V., Mohri, M. & Yang, S. (2017).**
  _AdaNet: Adaptive Structural Learning of Artificial Neural Networks._ ICML,
  **arXiv:1607.01097 [verified]**. Grows structure and weights together with
  data-dependent generalisation guarantees. Its own reported scope is **binary
  classification tasks extracted from CIFAR-10**, not large-scale vision.
- **Badirli, S. et al. (2020).** _GrowNet: Gradient Boosting Neural Networks_,
  commonly cited as arXiv:2002.07971. Stagewise additive boosting with neural
  weak learners. **Not verified by fetch.**
- **Sivaprasad, S., Singh, A., Manwani, N. & Gandhi, V. (2021).** _The Curious
  Case of Convex Neural Networks_, **arXiv:2006.05103 [verified]**. Reports
  convex networks approaching base convolutional architectures — the abstract's
  claim is **similar performance**, not superiority.
- **Hastie, T. & Tibshirani, R. (1990).** _Generalized Additive Models._ The
  statistical family. **Not verified by fetch.**
- **Lou, Y., Caruana, R., Gehrke, J. & Hooker, G. (2012, 2013).** _Intelligible
  Models for Classification and Regression_ and _Accurate Intelligible Models
  with Pairwise Interactions._ KDD. The GA²M / explainable-boosting-machine
  line. The 2013 paper exists because the purely additive restriction costs real
  accuracy. **Not verified by fetch.**
- **Agarwal, R., Melnick, L., Frosst, N., Zhang, X., Lengerich, B., Caruana, R.
  & Hinton, G.E. (2020).** _Neural Additive Models._ NeurIPS,
  **arXiv:2004.13912 [verified]**. Replaces each additive shape function with a
  neural subnet. The verified abstract claims performance **similar to
  state-of-the-art generalised additive models** and does **not** claim parity
  with unrestricted deep networks.
- **Liu, Z. et al. (2024).** _KAN: Kolmogorov–Arnold Networks_,
  **arXiv:2404.19756 [verified]**. Learnable univariate functions on edges. The
  abstract's strongest evidence is on data fitting and PDE solving, not
  large-scale image classification. Skeptical follow-ups (Shukla et al. 2024;
  Yu et al. 2024, **not verified by fetch**) report that the advantages shrink
  under matched-budget comparison.
- **Chen, T. & Guestrin, C. (2016).** _XGBoost._ KDD,
  DOI 10.1145/2939672.2939785. The canonical strong stagewise additive baseline.
  **Not verified by fetch.**
- **Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022).** _Why do tree-based
  models still outperform deep learning on tabular data?_
  **arXiv:2207.08815 [verified]**. On 45 datasets, tree-based models remain
  state of the art on medium-sized tabular data. **This is why M94 Part B adds
  tree families before v15 concludes anything about additive models.**
- **Frankle, J. & Carbin, M. (2019),** _The Lottery Ticket Hypothesis_,
  arXiv:1803.03635, and **Evci, U. et al. (2020),** _Rigging the Lottery_,
  arXiv:1911.11134. **Not verified by fetch.** Recorded as the contrast case:
  sparse networks discovered **through** dense training, which is the approach
  v15 is deliberately not taking.

### 8.4 Sparse dictionaries, concept models and simulatability — where the acceptance frame sits

- **Makhzani, A. & Frey, B. (2014).** _k-Sparse Autoencoders_,
  **arXiv:1312.5663 [verified]**. The top-k activation M80's dictionary uses.
- **Cunningham, H., Ewart, A., Riggs, L., Huben, R. & Sharkey, L. (2024).**
  _Sparse Autoencoders Find Highly Interpretable Features in Language Models_,
  **arXiv:2309.08600 [verified]**.
- **Bricken, T. et al. (2023).** _Towards Monosemanticity_, Anthropic
  Transformer Circuits Thread **[web verified]**. Introduces feature splitting
  and the "loss recovered" metric; notes that loss recovered is always below
  100% at useful sparsity.
- **Templeton, A. et al. (2024).** _Scaling Monosemanticity_, Anthropic
  Transformer Circuits Thread **[web verified]**.
- **Gao, L. et al. (2024).** _Scaling and Evaluating Sparse Autoencoders_,
  **arXiv:2406.04093 [verified]**. Introduces the top-k SAE and shows it
  dominates ReLU SAEs on both the sparsity–MSE and sparsity–probe-loss
  frontiers. Compares SAE types to each other, **not SAE codes to raw
  representations**.
- **Lieberum, T., Dunefsky, J., Nanda, N. & Conmy, A. (2024).** _Improving
  Dictionary Learning with Gated Sparse Autoencoders_,
  **arXiv:2404.16014 [verified]**. Also reports that the SAE encoder is a weaker
  sparse-approximation algorithm than iterative pursuit methods — **a direct
  argument for M96's pursuit-based selection over an encoder**.
- **Makelov, A., Lange, G. & Nanda, N. (2024).** _Towards Principled Evaluations
  of Sparse Autoencoders_, **arXiv:2405.08366 [verified]**. Argues the evidence
  base for interpretability claims is weaker than usually stated.
- **Engels, J. et al. (2024).** _Not All Language Model Features Are
  One-Dimensionally Linear_, **arXiv:2405.14860 [verified]**. Some features are
  multi-dimensional geometric structures, challenging the one-direction-per-atom
  assumption that M96 and M97 both make. **Registered as a limit on both.**
- **Bushnaq, L. et al. (2025).** _Sparse Autoencoders Do Not Find Canonical
  Units of Analysis_, **arXiv:2502.04878 [verified]**. Larger dictionaries
  contain both novel and composed latents; **no canonical feature set exists and
  granularity is a hyperparameter**. This is directly relevant to §7.4's
  registered caution about reading a dictionary-size effect as a growth effect.
- **Koh, P.W. et al. (2020).** _Concept Bottleneck Models_,
  **arXiv:2007.04612 [verified]**. Reports roughly 1–2 points of accuracy cost
  on CUB-200-2011.
- **Chen, C. et al. (2019).** _This Looks Like That_ (ProtoPNet),
  **arXiv:1806.10574 [verified]**. Prototype-part classification by weighted
  similarity sum.
- **Nauta, M., van Bree, R. & Seifert, C. (2021).** _Neural Prototype Trees_,
  **arXiv:2012.02046 [verified]**.
- **Wan, A. et al. (2021).** _NBDT: Neural-Backed Decision Trees_,
  **arXiv:2004.00221 [verified]**. Its related-work section states of
  hierarchical surrogate losses that "these methods quickly suffer from major
  accuracy loss with more classes or higher-resolution images (e.g. beyond
  CIFAR10)". **This is published corroboration of §2.4 observation A** and part
  of the basis for §11.2 item 9.
- **Morin, F. & Bengio, Y. (2005).** _Hierarchical Probabilistic Neural Network
  Language Model._ AISTATS. **Not verified by fetch.** The origin of
  hierarchical softmax; prioritised speed over accuracy.
- **Doshi-Velez, F. & Kim, B. (2017).** _Towards A Rigorous Science of
  Interpretable Machine Learning_, **arXiv:1702.08608 [verified]**. The
  application-grounded / human-grounded / functionally-grounded taxonomy, and
  the argument that model classes assumed interpretable must be validated
  empirically.
- **Hase, P. & Bansal, M. (2020).** _Evaluating Explainable AI_,
  **arXiv:2005.01831 [verified]**. Establishes that forward-simulation studies
  require a no-explanation control and non-overlapping explanation and test
  items, and reports that properly controlled gains over the no-explanation
  baseline are **modest**. **The I5 protocol's example-leakage prohibition and
  shuffled null sit in this lineage.**
- **Shazeer, N. et al. (2017),** _Outrageously Large Neural Networks_,
  **arXiv:1701.06538 [verified]**, and **Fedus, W., Zoph, B. & Shazeer, N.
  (2022),** _Switch Transformers_, **arXiv:2101.03961 [verified]**. Recorded
  because this program is named CG-MoE and the naming invites the comparison:
  **neither paper claims routing sparsity as an interpretability benefit.** Both
  frame sparsity as a compute-efficiency mechanism. Switch-XXL reports SQuAD
  89.7 against T5's 91.3 and SuperGLUE 87.5 against 89.3.

### 8.5 Representations built without end-to-end backpropagation — the M99 bars

Every figure in this subsection was **verified by fetching the source** and is
the basis for §10.2. No M99 bar may be set against anything not in this table.

| method                                                       | corpus     | accuracy                | representation learned by backpropagation? | head linear? |
| ------------------------------------------------------------ | ---------- | ----------------------: | ------------------------------------------ | ------------ |
| Coates, Lee & Ng (2011), k-means triangle, 4000 features      | CIFAR-10   | **0.796**               | no                                          | yes          |
| Oyallon & Mallat (2015), roto-translation scattering          | CIFAR-10   | **0.823**               | no                                          | no (Gaussian SVM) |
| Li et al., enhanced CNTK, plain                               | CIFAR-10   | **0.8140**              | no                                          | no (kernel)  |
| Li et al., enhanced CNTK + patch preprocessing                | CIFAR-10   | **0.8836**              | no                                          | no (kernel)  |
| **Thiry et al. (2021), SimplePatch, linear head**             | CIFAR-10   | **0.869** *(unconfirmed — §2.9.5)* | **no**                   | **yes**      |
| Thiry et al. (2021), SimplePatch, 1-hidden-layer head         | CIFAR-10   | 0.885                   | no                                          | no           |
| Shankar et al. (2020), compositional kernel                   | CIFAR-10   | **0.900**               | no                                          | no (kernel)  |
| Shankar et al. (2020), paired learned network                 | CIFAR-10   | 0.960                   | yes                                         | no           |
| Zarka et al. (2020), scattering + linear probe                | ImageNet   | 0.261 top-1 / 0.447 top-5 | no                                        | yes          |
| **Thiry et al. (2021), SimplePatch, linear head**             | ImageNet   | **0.360 top-1 / 0.576 top-5** | **no**                                | **yes**      |
| Thiry et al. (2021), SimplePatch, 1-hidden-layer head         | ImageNet   | 0.394 top-1 / 0.621 top-5 | no                                        | no           |
| Hinton (2022), Forward-Forward                                | CIFAR-10   | 0.59 (backprop baseline 0.63 in the same table) | no          | no           |
| Bartunov et al. (2018), feedback alignment                    | ImageNet   | **0.0692 top-1**        | no                                          | no           |
| Bartunov et al. (2018), backpropagation convnet baseline      | ImageNet   | 0.3607 top-1            | yes                                         | no           |
| Petersen et al. (2024), convolutional logic gate networks     | CIFAR-10   | 0.8629, 61M gates       | **yes, via a differentiable relaxation**    | no           |
| Belilovsky et al. (2019), greedy layerwise VGG-11             | ImageNet   | 0.676 / 0.880           | **yes, but not end-to-end**                 | no           |
| Belilovsky et al. (2019), VGG-11 end-to-end                   | ImageNet   | 0.679 / 0.880           | yes                                         | no           |
| Belilovsky et al. (2019), greedy layerwise SimCNN k=3 ensemble | ImageNet  | 0.716 / 0.898           | yes, but not end-to-end                     | no           |
| **Oquab et al. (2023), DINOv2 ViT-S/14 linear probe**         | ImageNet   | **0.811 top-1**         | yes                                         | yes          |
| Oquab et al. (2023), DINOv2 ViT-S/14 k-NN                     | ImageNet   | 0.790 top-1             | yes                                         | n/a          |
| Oquab et al. (2023), DINOv2 ViT-g/14 linear probe             | ImageNet   | 0.865 top-1             | yes                                         | yes          |
| Oquab et al. (2023), DINOv2 ViT-S/14 linear probe             | CIFAR-10   | 0.977                   | yes                                         | yes          |
| Oquab et al. (2023), DINOv2 ViT-g/14 linear probe             | CIFAR-10   | 0.995                   | yes                                         | yes          |
| absolute ImageNet state of the art, as cited by DINOv2        | ImageNet   | 0.911 top-1             | yes                                         | no           |

Sources: Coates et al., PMLR v15; Oyallon & Mallat, CVPR 2015 open access;
Li et al., arXiv:1911.00809; Thiry, Arbel, Belilovsky & Oyallon,
arXiv:2101.07528; Shankar et al., arXiv:2003.02237; Zarka et al.,
arXiv:1910.03561; Hinton, arXiv:2212.13345; Bartunov et al., arXiv:1807.04587;
Belilovsky, Eickenberg & Oyallon, arXiv:1812.11446; Oquab et al.,
arXiv:2304.07193 Tables 4 and 8.

**The size of the gap, stated plainly, because the plan's ambition has to be
calibrated against it.**

- **CIFAR-10.** The best verified representation built without learned filters
  reaches **0.900** (compositional kernel). With a strictly linear head on fixed
  patches it reaches **0.869**. A frozen DINOv2 ViT-g/14 with a linear probe
  reaches **0.995** on the same corpus. The gap is roughly **9.5 to 12.6
  points**.
- **ImageNet.** The best verified representation built without learned filters
  reaches **0.394 top-1**, and **0.360** with a strictly linear head, against
  **0.811** for a frozen DINOv2 ViT-S/14 linear probe and **0.911** for the
  absolute state of the art. The gap is roughly **42 to 55 points**.

**This is registered as a constraint on v15's ambition, not as a target.** A
forty-point gap at ImageNet scale is not closed by a plan of this size. §10.2's
M99 is scoped to CIFAR-10 against the **0.869** fixed-patch linear-head bar for
exactly this reason, and even that is conditional and non-gating.

**Search failure disclosed:** the **current best published CIFAR-10 test
accuracy was not verified** in this search and no figure for it appears in this
plan. The 0.995 figure above is a DINOv2 frozen-feature linear probe and is
**not** claimed to be the state of the art.

### 8.6 Disclosed search failures

Registered under `PRIOR_ART_AUDIT_v13.md` §8.4 and §6.3 A7.

1. Of the sources listed in §8.1–§8.4, **twenty-one were not verified by
   fetching the source** and are marked as such inline. Their bibliographic
   details, and in some cases their reported figures, may be wrong. **Every
   figure in §8.5 was verified by fetch**, which is why §8.5 is the only
   subsection from which a bar may be set (§11.2 item 12).
2. No verified hardness theorem was located for the projection-pursuit oracle
   step specifically (§8.1).
3. Exact accuracy tables were **not** verified for Koh et al., Chen et al.,
   Nauta et al., Lou et al., or Agarwal et al.; only abstract-level claims were
   verified where marked. **No v15 milestone may register a bar against an
   unverified external number.**
4. The **current best published CIFAR-10 accuracy was not verified** and does not
   appear in this plan (§8.5).
5. The claim that the **encoder** matters more than the dictionary-learning
   algorithm, sometimes attributed to Coates & Ng, was **not** verified in its
   exact wording. What was verified from Coates, Lee & Ng (2011) is the weaker
   statement that feature count, whitening and dense extraction "can, in fact,
   be as important as the unsupervised learning algorithm itself".
6. Exact small-benchmark figures from the original feedback-alignment and
   direct-feedback-alignment papers (Lillicrap et al.; Nøkland) were **not**
   verified; only Bartunov et al.'s scaling result was.
7. M88's measured search recall of 4 of 7 applies to this section. The lineage
   above is **not** a coverage claim, and the absence of a source from it is not
   evidence that the source does not exist.
8. §9 and §10 are conditional partly because their bars depend on external
   numbers, and §10.2 may not open until §8.5's bars are re-verified at
   execution time.
9. In §8.7, exact end-to-end application latency for NVIDIA's 2:4 workflow,
   paper-side inference-latency numbers for Switch/GShard/Mixtral, SkipNet's
   wall-clock figure, Hinton et al.'s distillation compression ratio, and direct
   energy measurements for sparsity were **not** verified and are marked inline.
10. In §8.8, the official M4 winning score and the official M5 Accuracy winner
    and score were **not** verified, and no verified head-to-head sample-
    efficiency comparison between additive models and deep networks was located
    at all. The §8.8 gap statement therefore rests on forecasting and language
    benchmarks only.
11. §8.7 and §8.8 were produced by a single search pass each. Item 7's recall
    caveat applies to them with more force than to §8.1–§8.5, which were
    assembled over several passes.
12. **§8.9's inaccessible sources.** Cortes, DeSalvo & Mohri (2016), _Learning
    with Rejection_ (ALT 2016, LNAI 9925) is not on arXiv and was not opened, so
    whether it trains the rejector jointly is unknown to this plan. Chow (1970)
    was not opened. For Geifman & El-Yaniv (2017), SelectiveNet (2019), Madras
    et al. (2018), Mozannar & Sontag (2020) and Papernot & McDaniel (2018) the
    **formulations were read but the experimental numbers were not**; no bar may
    be set from any of them (§11.2 item 12).
13. **§8.9's unresolved efficiency figures.** Shallow-Deep Networks' "up to ~75%
    inference cost" could not be resolved to FLOPs or wall-clock, and DeeBERT's
    "~40% inference time" was read from an abstract only. Both are inadmissible
    as bars under §11.2 item 16.
14. **The H111 regime gap.** No paper was located reporting kNN ≥ linear probing
    in any regime — low-shot, distribution-shifted or fine-grained. DINOv2's
    Tables 5–9 cover those regimes and **were not read**; only Table 4 was. §8.9
    D4's negative prior therefore covers **full-shot ImageNet-1k only**, and
    H111's low-rung behaviour has no located prior in either direction. This is
    the single most consequential gap in §8.9, because H111's interesting claim
    lives precisely in the regimes that were not searched.
15. **The §8.9 D5 framing.** No paper was located reporting the oracle-versus-
    confidence deferral gap as a recovered fraction of oracle gain. Given item
    7's 4/7 measured recall, **this plan does not assert that the framing is
    absent from the literature**, only that this search did not find it.

### 8.7 Sparsity and efficiency — the literature behind P6 **[new in v15]**

Every figure below was verified by fetching the source unless marked otherwise.
Under §11.2 item 12 these verified figures may be used to set bars; the marked
ones may not.

**A1 — unstructured sparsity does not accelerate dense hardware.** Mishra et al.
(2021), _Accelerating Sparse Deep Neural Networks_ (**arXiv:2104.08378,
verified**). Fine-grained sparsity "maintains accuracy but poorly utilizes
memory accesses and fails to take advantage of modern vector and matrix math
pipelines, thus it does not outperform traditional dense models on processor
architectures such as GPUs". Its survey section: the performance benefit of
unstructured pruning "is negligible and at times negative, even when pruning
rate is high (e.g. 95%)"; parameters can be pruned "nearly 13× with no loss in
accuracy" in a pattern "not conducive to hardware acceleration".

**A2 — the crossover point is high.** Gale et al. (2020), _Sparse GPU Kernels for
Deep Learning_ (**arXiv:2006.10901, verified**). Vendor sparse kernels target
99%+ scientific sparsity; at the moderate sparsity found in neural networks they
are "not able to outperform their dense counterparts". Purpose-built kernels
beat dense from **71% sparsity**, giving **3.58×** and **2.19×** kernel-level
speedups over cuSPARSE and **1.2–2.1×** end-to-end with **12.8×** memory
reduction at matched accuracy. **The efficiency win is a kernel co-design
result, not a property of the zeros.**

**A3 — simple pruning is as good as elaborate pruning.** Gale, Elsen & Hooker
(2019), _The State of Sparsity in Deep Neural Networks_ (**arXiv:1902.09574,
verified**): variational dropout and ℓ₀ regularisation "perform inconsistently
for large-scale tasks" while "simple magnitude pruning can achieve comparable or
better results for a reduced computational budget".

**A4 — where hardware sparsity does work: N:M.** Mishra et al. (2021), same
source. Sparse Tensor Cores give **2×** math throughput, and large GEMMs achieve
"nearly a 2× speedup". Accuracy recovers after retraining: ResNet-50 top-1
**76.1 → 76.2** dense-FP16 to sparse-FP16; EfficientNet-B0 **77.25 → 77.29**
dense-FP16 to sparse-INT8 after the permutation workflow. Exact end-to-end
application latency: **unverified**. §5.11.7 and §11.2 item 14 forbid v15 from
borrowing this result.

**A5 — the framing.** Hooker (2020), _The Hardware Lottery_ (**arXiv:2009.06489,
verified**): "We may very well be in the midst of a present day hardware
lottery."

**A6 — conditional compute, where the win is training rather than latency.**
Fedus et al. (2021), Switch Transformer (**arXiv:2101.03961, verified**):
Switch-C is "4× faster to a fixed perplexity" than T5-XXL — a *training*
efficiency claim. Lepikhin et al. (2020), GShard (**arXiv:2006.16668,
verified**): 600B parameters on 2048 TPU v3 in 4 days, **22 TPU v3 core-years**
against **235.5** for the best dense baseline. Jiang et al. (2024), Mixtral
(**arXiv:2401.04088, verified**): **13B active of 47B resident**, 5× fewer
active parameters than Llama 2 70B — but "the memory costs for serving Mixtral
are proportional to its sparse parameter count, 47B". Inference latency figures:
**unverified**. **The MoE literature's verified wins are in training cost and
active-parameter count, not measured inference latency**, which is exactly the
distinction §5.11.5 forces v15 to keep separate.

**A7 — adaptive compute, where measured speedups exist.** Teerapittayanon et al.
(2017), BranchyNet (**arXiv:1709.01686, verified**): exits 94% of LeNet samples
at the first branch, with CPU/GPU speedups of **5.4×/4.7×** (LeNet),
**1.5×/2.4×** (AlexNet), **1.9×/1.9×** (ResNet). Wang et al. (2017), SkipNet
(**arXiv:1711.09485, verified abstract**): "reduces computation by 30–90% while
preserving the accuracy"; wall-clock figure **unverified**.

**A8 — the cheaper alternatives v15 is competing against.** Sanh et al. (2019),
DistilBERT (**arXiv:1910.01108, verified**): **40% smaller, 60% faster, 97%
retained**. Yao et al. (2022), ZeroQuant (**arXiv:2206.01861, verified**): INT8
gives up to **5.19×/4.16×** speedup with minimal accuracy impact and **3×**
memory reduction. Jacob et al. (2017) (**arXiv:1712.05877, verified**): "many
quantization approaches do not deliver verifiable efficiency improvements on
real hardware" — the earliest clean statement of the FLOPs-are-not-latency point
that §5.11.2 encodes as a rule.

**A9 — the trunk.** DINOv2 model card (**verified**): ViT-S **21M parameters**,
patch 14, embedding 384, and at 224×224 "1 class token + 256 patch tokens". timm
card for `vit_small_patch14_dinov2.lvd142m` (**verified**): **22.1M parameters,
46.8 GMACs at 518×518**. §2.6.2's derived **6,065,759,232 MACs at 224×224**
reproduces the timm figure under rescaling and is the number v15 uses.

**A10 — the comparison that should embarrass the trunk.** torchvision
(**verified**): MobileNetV3-Large, top-1 **75.274%**, **5,483,032** parameters,
**0.22 GFLOPs**; EfficientNet-B0, top-1 **77.692%**, **5,288,548** parameters,
**0.39 GFLOPs**. Against ≈12.13 GFLOPs for the frozen ViT-S/14 trunk at the same
resolution. Recorded because A5's efficiency obligation is about the whole
system, and the whole system's dominant cost is a backbone chosen for feature
quality with no efficiency constraint ever applied to it.

**What §8.7 licenses.** P6. And one structural conclusion that §7.5 M100 is
built to establish by measurement: on this program's architecture, efficiency is
a property of the trunk, and no head-side intervention can reach it.

### 8.8 Sequence modelling — the literature behind P7 and M101 **[new in v15]**

**C1 — simple beats deep on long-horizon forecasting.** Zeng et al. (2023),
_Are Transformers Effective for Time Series Forecasting?_ (**arXiv:2205.13504,
verified, Table 2**): LTSF-Linear outperforms Transformer-based LTSF models on
all nine datasets, often "by a large margin".

| corpus, horizon 96 | arm | MSE | MAE |
| --- | --- | ---: | ---: |
| Electricity | `Linear` | **0.140** | **0.237** |
| Electricity | FEDformer | 0.193 | 0.308 |
| Exchange | `DLinear` | **0.081** | **0.196** |
| Exchange | FEDformer | 0.148 | 0.278 |

These are the registered external anchors for M101 (§7.6), recorded under R7 as
anchors and never as operands.

**C2 — and loses badly on language.** Chelba et al. (2013), One Billion Word
(**arXiv:1312.3005, verified**): unpruned Kneser–Ney 5-gram perplexity **67.6**.
Against it on the same benchmark: Jozefowicz et al. (2016)
(**arXiv:1602.02410, verified**) **30.0** single model and **23.7** ensemble;
Baevski & Auli (2018) (**arXiv:1809.10853, verified**) **23.02**; Dai et al.
(2019), Transformer-XL (**arXiv:1901.02860, verified**) **21.8** on One Billion
Word and **18.3** on WikiText-103. The gap from the best sparse count-based
model to the best Transformer is **45.8 perplexity, a factor of 3.10**.

**This is the single most important entry in §8 for scoping Q3.** It is why
§3.3 asks about transfer rather than competitiveness, why M101 runs on
forecasting, and why §11.2 item 13 exists.

**C3 — trees remain strong on tabular and boosting on forecasting.** Elsayed et
al. (2021) (**arXiv:2101.02118, verified**): a gradient-boosted regression tree
setup "outperform[s] all state-of-the-art DNN models evaluated in this paper",
against eight deep models on nine datasets. Grinsztajn et al. (2022)
(**arXiv:2207.08815, verified**): "tree-based models remain state-of-the-art on
medium-sized data (~10K samples) even without accounting for their superior
speed", over 45 datasets, with a 20,000 compute-hour hyperparameter search.

**C4 — sparse symbolic identification works, within a stated envelope.** Brunton,
Proctor & Kutz (2016), SINDy (**arXiv:1509.03580, verified**): sparse regression
over a candidate library recovers governing equations, *conditional on the
dynamics being sparse in the chosen library*. The follow-up literature states
the envelope rather than hiding it: Fasel et al. (**arXiv:2111.10992,
verified**) call standard sparse model discovery "sensitive to noise, especially
in the low-data limit"; Messenger & Bortz (**arXiv:2005.04339**,
**arXiv:2007.02848, verified**) improve noise robustness "by orders of
magnitude"; Kaheman et al. (**arXiv:2004.02322, verified**) call prior implicit
variants "extremely sensitive to noise"; Bakarji et al.
(**arXiv:2201.05136, verified**) need a deep delay autoencoder for partial
observability; Fukami et al. (**arXiv:2010.12177, verified**) need a CNN
autoencoder to reach low-dimensional coordinates before the library is usable;
Delgado-Cano et al. (**arXiv:2507.00747, verified**) report the regression
becoming "computationally intractable and ill-conditioned" in high dimensions.

**SINDy is the closest existing relative of what v15 proposes, and its failure
modes are v15's failure modes**: library design is the bottleneck, noise is the
enemy, and high dimension forces a learned encoder in front of the sparse part.
That last point is the same conclusion §2.6.2 reaches from the cost side, and it
is why A5's permission for dense components is load-bearing rather than
cosmetic.

**C5 — non-attention deep sequence models are strong, and are not sparse.** Gu et
al. (2021), S4 (**arXiv:2111.00396, verified**): 91% on sequential CIFAR-10,
generation 60× faster, SOTA on every Long Range Arena task including Path-X.
Gu & Dao (2023), Mamba (**arXiv:2312.00752, verified**): 5× higher throughput
than Transformers, linear scaling in sequence length, and Mamba-3B "outperforms
Transformers of the same size and matches Transformers twice its size". Mamba's
own abstract records that prior subquadratic architectures "have not performed
as well as attention on important modalities such as language". Recorded so that
v15 does not mistake "not a Transformer" for "sparse and inspectable": these
models are neither.

**C6 — sparse attention is an efficiency technique, not an interpretability
one.** Child et al. (**arXiv:1904.10509, verified**) O(n√n); Beltagy et al.,
Longformer (**arXiv:2004.05150, verified**) linear scaling; Zaheer et al.,
BigBird (**arXiv:2007.14062, verified**) linear, 8× longer sequences. None of
these papers claims inspectability, and §11.2 item 15 forbids citing them as
though they did.

**C7 — attention weights are not an explanation.** Jain & Wallace (2019)
(**arXiv:1902.10186, verified**): attention weights are often uncorrelated with
gradient-based importance, and very different attention distributions can yield
equivalent predictions. Wiegreffe & Pinter (2019) (**arXiv:1908.04626,
verified**) contest the framing and propose four alternative tests. Recorded
because it is the sequence-domain instance of the standard v15 already holds
itself to: the explanation must be the computation (§5.11.4, D7).

**C8 — competition evidence, partly unverified.** Redd et al.
(**arXiv:1907.03329, verified**) state ES-RNN "achieved a 9.4% sMAPE improvement
in the M4 competition"; the official M4 winning score is **unverified**. Anderer
& Li (**arXiv:2103.08250, verified**) took second in M5 Accuracy with N-BEATS at
upper levels and LightGBM at the bottom; the official M5 winner and score are
**unverified**. No bar may be set from C8 (§11.2 item 12).

**C9 — a verified gap in the literature.** No verified head-to-head quantitative
comparison of **sample efficiency** between additive/sparse models and deep
networks was located. H108 is therefore registered without a literature prior in
either direction, which is unusual in this plan and is stated so that H108's
result is not later described as confirming or contradicting a body of work that
was never found.

### 8.9 Abstention, deferral and cascades — the literature behind H110 and M102 **[new in v15]**

Registered under the same rule as §8.7 and §8.8: a figure may be used as a bar
only if its source was fetched and the number read from it (§11.2 item 12).
Every claim below is marked **verified** or **unverified**, and §8.6 carries this
section's search failures.

**D1 — abstention is a mature field, and M102 is a late entrant to it.** This is
stated first and plainly, because §2.8's framing ("the program has never
optimised abstention") is a statement about *this program's history*, not about
the literature, and the two must not be confused.

- Chow (1970), _On optimum recognition error and reject tradeoff_
  (**DOI 10.1109/TIT.1970.1054406, source not opened — unverified**): the
  founding error/reject tradeoff. Post-hoc threshold on the posterior.
- Geifman & El-Yaniv (2017), _Selective Classification for Deep Neural Networks_
  (**arXiv:1705.08500, formulation verified; experimental numbers not read**):
  the selection function is the max softmax of an already-trained network and
  only the threshold is chosen post-hoc. **Confirms the post-hoc paradigm.**
- Geifman & El-Yaniv (2019), _SelectiveNet_ (**arXiv:1901.09192, formulation
  verified from Eq. 2; experimental numbers not read**): predictor `f` and
  selection function `g` **share parameters and are trained simultaneously**
  under a coverage constraint.
- Cortes, DeSalvo & Mohri (2016), _Learning with Rejection_ (ALT 2016, LNAI
  9925): **not on arXiv, source not opened — unverified**, see §8.6.

**D2 — learning to defer already trains the gate jointly, and this is the
finding that most constrains H110.**

- Madras, Pitassi & Zemel (2018), _Predict Responsibly / Learning to Defer_
  (**arXiv:1711.06664, formulation verified**): a joint deferral loss trains the
  predictor and the deferral decision together.
- Mozannar & Sontag (2020), _Consistent Estimators for Learning to Defer_
  (**arXiv:2006.01862 / PMLR 119:7076–7087, abstract verified; PDF not read**):
  a consistent surrogate loss for joint training of classifier and rejector.

**Registered consequence, stated against this plan's interest.** §4.3 motivates
H110 by observing that this program has only ever read abstention off an
accuracy-fitted model. That observation stands for *this program*. It does
**not** stand for the field: SelectiveNet (2019), Madras et al. (2018) and
Mozannar & Sontag (2020) all train the gate jointly with the task, and the last
supplies consistency guarantees. **H110's "train the gate directly" is therefore
a known and theoretically grounded idea, and no v15 document may present it as
otherwise** (§11.2 items 7 and 22). What remains open for M102 is narrower and
is stated in D5.

**D3 — cascades do produce measured savings, and the honest comparators are
wall-clock.**

- FrugalGPT, Chen, Zaharia & Zou (2023) (**arXiv:2305.05176, Table 3 and Fig. 3c
  verified**): up to **98.3%**, **73.3%** and **59.2%** cost savings on its three
  benchmarks, and a reported **80% cost reduction with a 1.5-point accuracy
  gain**. That an accuracy *gain* accompanies a cost cut is the same qualitative
  effect §2.8's oracle shows and is the strongest external support for Reading 1.
  It is an **anchor, never an operand** (R7): different modality, different
  stack, cost measured in dollars.
- BranchyNet, Teerapittayanon et al. (2017) (**arXiv:1709.01686, verified,
  wall-clock**): **5.4×/4.7×** (B-LeNet), **1.5×/2.4×** (B-AlexNet),
  **1.9×/1.9×** (B-ResNet), CPU/GPU.
- CALM, Schuster et al. (2022) (**arXiv:2207.07061, verified, wall-clock on
  TPUv3**): **×3.53** on CNN/DM at δ=0.05 and **×2.83** on WMT at δ=0.25.
- Shallow-Deep Networks, Kaya et al. (2019) (**arXiv:1810.07052, introduction
  only**): "up to ~75% inference cost" — **whether this is FLOPs or wall-clock
  was not confirmed, and it is therefore inadmissible as a bar** under §11.2
  item 16.
- DeeBERT (2020) (**DOI 10.18653/v1/2020.acl-main.204, abstract only**): "~40%
  inference time" — **abstract claim only, unverified**, inadmissible as a bar.

**D4 — retrieval versus fitting on frozen features, the H111 evidence.** All four
rows verified by fetch.

| model | kNN | linear | source |
| --- | ---: | ---: | --- |
| DINOv2 ViT-S/14 | 0.790 | 0.811 | arXiv:2304.07193v2 Table 4, **verified** |
| DINO ViT-B/8 | 0.783 | 0.801 | arXiv:2104.14294, **verified** |
| iBOT ViT-B/16 | 0.771 | 0.795 | arXiv:2111.07832, **verified** |
| MoCo v3 ViT-B | — | 0.767 | arXiv:2104.02057, **verified**; kNN not reported |

**Linear beats kNN in every verified row**, by 1.8 to 2.4 points. This program's
sealed measurement runs the other way — `knn` **0.661255** over
`mlp_integrated_gradients` **0.660522** — and §2.5 already records the ordering
inversion. **The literature prior for H111 is therefore negative**, and H111 is
registered as expected-refuted at the upper rungs of the §5.11.5 ladder. If it
survives, the interesting quantity is the *rung at which the ordering flips*,
not the flip itself.

**Registered search failure bearing directly on H111.** No paper was located
reporting kNN ≥ linear in any regime — low-shot, distribution-shifted or
fine-grained. DINOv2's Tables 5–9 cover those regimes and **were not read**; only
Table 4 was. So the negative prior above is established for **full-shot
ImageNet-1k only**, and H111's low-rung behaviour has **no located prior in
either direction**. §8.6 item 14.

**D5 — the one framing that was searched for and not found.** No paper was
located that reports the **gap between an oracle deferral rule and a
confidence-based deferral rule as a recovered fraction of oracle-available
gain**. Selective-prediction work reports risk–coverage curves and AURC;
learning-to-defer work reports accuracy under a deferral budget; cascade work
reports cost savings. The oracle upper bound as an explicit denominator was not
found in any of them.

**What this does and does not license.** It licenses reporting M102's recovered
fraction as this program's chosen instrument, and it obliges §8.6 to record that
the search may simply have missed it — M88 measured this program's search recall
at **4/7**, and absence of a located result is not absence of the result. **No
v15 document may state that this framing is new** (§11.2 item 7 already forbids
it; item 22 makes it specific). The honest statement of M102's contribution is:
a known family of methods, evaluated against an oracle denominator this program
could not find reported, on a sparse geometric model family that the
selective-prediction literature does not appear to cover.

**D6 — calibration, the mechanism §2.8.3 blames.**

- Guo et al. (2017), _On Calibration of Modern Neural Networks_
  (**arXiv:1706.04599, Table 1 verified**): ECE for ResNet-152 on ImageNet falls
  **5.48% → 1.86%** under temperature scaling, and ResNet-110 on CIFAR-100
  **16.53% → 1.26%**. Modern networks are badly calibrated and a **single
  post-hoc scalar** fixes much of it.
- Papernot & McDaniel (2018), _Deep k-Nearest Neighbors_ (**arXiv:1803.04765,
  introduction and §VI read; calibration tables not reached — unverified**):
  claims improved calibration and credibility from kNN over representations.

**Registered consequence for M102's design.** Guo et al. is why M102 must include
temperature scaling of the confidence gate as an arm. A cheap post-hoc scalar
that closes much of the 44.4%→60% distance *(§2.8.5: 44.4% is not reproducible;
the corrected baseline is 30.6%, so the distance was larger than written here)*
would mean H110's expensive joint objective was never needed, and a plan that
omitted the cheap comparator would be crediting its preferred method with a win
the baseline could have had. This is registered in §7.7 as arm (b′).
**[recorded after execution: the scalar did not close it — temperature scaling
read 27.7% against the untreated margin's 30.6%, ledger C102.3.]**

---

### 8.10 Effective rank, random features and mixtures of experts — the lineage behind §2.9.7 and M104–M106 **[new — recorded after the M103 prior-art audit]**

**This section exists because an audit went against this program.** It is
registered under the same §8 rule as every other lineage section: **no novelty
claim of any kind, and no assertion of absence of prior art.**

#### 8.10.1 The random-features result that subsumes C103.1's headline

M103's confirmed result (C103.1) is that a dictionary chosen against a
discriminative residual reaches a random dictionary's accuracy at **half the
atoms**. The audit found this phenomenon **already published, in a stronger
form, by a label-free mechanism, with matching lower bounds**.

- **Avron, H., Kapralov, M., Musco, Cameron, Musco, Christopher, Velingker, A. &
  Zandieh, A. (2017).** _Random Fourier Features for Kernel Ridge Regression:
  Approximation Bounds and Statistical Guarantees._ ICML. **arXiv:1804.09893,
  abstract verified by fetch.** Uniform random Fourier features require
  `O(n_λ · log s_λ)` samples; sampling proportional to the **ridge leverage
  scores** requires `O(s_λ · log s_λ)`, where `s_λ ≤ n_λ`, **with a matching
  lower bound**. This is an asymptotic separation, not a constant factor.
- **Li, Z., Ton, J.-F., Oglic, D. & Sejdinovic, D. (2019).** _Towards a Unified
  Analysis of Random Fourier Features._ ICML. **arXiv:1806.09178, abstract
  verified by fetch.** Uniform sampling needs `Ω(√n · log d_eff)` features while
  leverage-weighted sampling needs `Ω(d_eff)`; under favourable noise conditions
  the requirement falls to `Ω(log n log log n)` or, in the authors' words, *"even
  a constant number of features in some benign cases."*
- **Rudi, A. & Rosasco, L. (2017).** _Generalization Properties of Learning with
  Random Features._ NeurIPS. **arXiv:1602.04474, abstract verified by fetch.**
  `O(√n log n)` features suffice for `O(1/√n)` learning rates; faster rates
  require *"a possibly problem dependent"* sampling distribution.
- **Sinha, A. & Duchi, J. (2016).** _Learning Kernels with Random Features._
  NeurIPS. The supervised analogue — features selected against labels rather than
  drawn uniformly. **Exact figures not verified by fetch**; NeurIPS serves this
  paper as a PDF behind an iframe that defeated every extraction route attempted.
  Recorded as a search failure under §8.6.

**Three consequences, all against this program's interest.**

1. **C103.1's phenomenon is not new.** A 2× constant on CIFAR-10 sits inside a
   published asymptotic separation. §11.2 item 22's form applies, and the
   disclosure is recorded in `analysis/CLAIM_LEDGER_v15.md` as an amendment to
   C103.1.
2. **C103.3's narrowing margin is *predicted*, not merely volunteered.** The
   improvement factor is `n_λ / s_λ`, and it contracts toward 1 as the budget
   grows past the problem's effective dimension. The program reported the
   narrowing honestly and against interest; the theory says it was the expected
   observation.
3. **The strongest published results are label-free.** Ridge-leverage sampling
   uses no labels. M103 arm (c) uses them. The mechanism this program built is
   therefore **more restrictive and less powerful** than the published one, which
   is the opposite of the reading a casual statement of C103.1 would invite.

**What survives.** M103's figures, its nulls, its seed spreads and its
instrument checks are unaffected — the audit changes the claim's **standing as a
contribution**, not its validity as a measurement. And the theory hands the
program something it never had: a **computable target**, the effective dimension
`d_eff`, which §2.9.7 and M104 act on directly.

#### 8.10.2 The instrument

- **Garrido, Q., Balestriero, R., Najman, L. & LeCun, Y. (2023).** _RankMe:
  Assessing the Downstream Performance of Pretrained Self-Supervised
  Representations by Their Rank._ ICML. **arXiv:2210.02885, abstract verified by
  fetch.** Defines the **effective rank** of a representation as the exponentiated
  Shannon entropy of its normalised singular-value spectrum, and states it *"does
  not have any training or hyper-parameters to tune."* **This is the instrument
  §2.9.7 uses, unmodified.** It was not built here and no document of this program
  may present it as this program's.
- **Agrawal, K.K., Mondal, A.K., Ghosh, A. & Richards, B. (2022).** _α-ReQ:
  Assessing Representation Quality in Self-Supervised Learning by Measuring
  Eigenspectrum Decay._ NeurIPS. The power-law decay exponent used as §2.9.7
  probe 4's second scalar. **Not verified by fetch.**
- **Jing, L., Vincent, P., LeCun, Y. & Tian, Y. (2022).** _Understanding
  Dimensional Collapse in Contrastive Self-Supervised Learning._ ICLR.
  **arXiv:2110.09348, abstract verified by fetch.** Establishes that
  representations collapse to a subspace far smaller than their ambient
  dimension. §2.9.7 probe 1's falling useful fraction is an instance of the
  phenomenon this paper names.

#### 8.10.3 Constructive and white-box networks — the line M104–M106 must not re-run

- **Chan, K.H.R., Yu, Y., You, C., Qi, H., Wright, J. & Ma, Y. (2021).**
  _ReduNet: A White-box Deep Network from the Principle of Maximizing Rate
  Reduction._ JMLR. **arXiv:2105.10446, abstract verified by fetch.** The
  *"parameters of the network are all explicitly constructed layer-by-layer via
  forward propagation."* **This is constructive network building, already done.**
- **Yu, Y., Buchanan, S., Pai, D., Chu, T., Wu, Z., Tong, S., Haeffele, B.D. &
  Ma, Y. (2023).** _White-Box Transformers via Sparse Rate Reduction._ NeurIPS.
  **arXiv:2306.01129, abstract verified by fetch.** CRATE is *"mathematically
  fully interpretable"*, is evaluated on ImageNet, and reaches *"performance very
  close to"* a ViT. Code at `github.com/Ma-Lab-Berkeley/CRATE`. **A sparse,
  inspectable architecture at scale already exists and is public.**
- **Yu, Y., Chan, K.H.R., You, C., Song, C. & Ma, Y. (2020).** _Learning Diverse
  and Discriminative Representations via the Principle of Maximal Coding Rate
  Reduction._ NeurIPS. **arXiv:2006.08558.** The MCR² objective underneath both.
- **Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M.,
  LeCun, Y. & Ballas, N. (2023).** _Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture._ CVPR. **arXiv:2301.08243, abstract
  verified by fetch.** I-JEPA — the predictive-embedding programme.

**Registered consequence.** **This program may not build a constructive
white-box network, a rate-reduction objective, or an effective-dimension
instrument.** All three exist. M104–M106 are restricted to the one question
§8.10.4 shows is open.

#### 8.10.4 Mixtures of experts — and the one thing the audit did not find

- **Fedus, W., Zoph, B. & Shazeer, N. (2021).** _Switch Transformers._
  **arXiv:2101.03961, abstract verified by fetch.** Top-1 routing to
  **identically sized** experts.
- **Dai, D., Deng, C., Zhao, C., Xu, R.X. et al. (2024).** _DeepSeekMoE: Towards
  Ultimate Expert Specialization in Mixture-of-Experts Language Models._
  **arXiv:2401.06066, abstract verified by fetch.** DeepSeekMoE 16B reaches
  LLaMA2 7B's performance with *"only about 40% of computations"*, and 145B
  approaches DeepSeek 67B with **28.5%**. Its contribution is **finer-grained
  segmentation** of experts — *"finely segmenting the experts"* — and the experts
  remain **uniformly sized**.
- **Jiang, A.Q. et al. (2024).** _Mixtral of Experts._ **arXiv:2401.04088.**
  Eight experts, all the same size. **Not verified by fetch.**
- **Shazeer, N. et al. (2017).** _Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts Layer._ ICLR. **arXiv:1701.06538.** The
  modern origin. **Not verified by fetch.**

**The disclosed search, and what it did not turn up.** Searched: `mixture of
experts heterogeneous capacity`, `expert size allocation effective rank`,
`variable capacity experts MoE`, `intrinsic dimension expert allocation`,
`spectral sizing mixture of experts`, on arXiv listing and full-text search,
August 2026. **No paper was found that sizes each expert to the measured
effective rank of the sub-population it serves.** This is a **search failure
disclosure under §8.6 and not a novelty claim** — §8.4's consequence governs,
the search was neither exhaustive nor systematic, and the correct reading is
*"this program did not find it"*, never *"it does not exist."* If it is found
later, the finding is recorded here and M104's contribution reduces to a
replication, which §11.2 item 22 already requires the program to accept without
argument.

#### 8.10.5 The routing question, and a citation this plan had wrong

- **Fang, Z., Li, Y., Lu, J., Dong, J., Han, B. & Liu, F. (2022).** _Is
  Out-of-Distribution Detection Learnable?_ NeurIPS. **arXiv:2210.14707, abstract
  verified by fetch.** A **PAC-learnability** analysis. It proves impossibility
  in some settings but states those conditions *"may not hold in some practical
  scenarios"* and supplies *"necessary and sufficient conditions"* under which
  OOD detection **is** learnable.
- **van de Ven, G.M. & Tolias, A.S. (2019).** _Three Scenarios for Continual
  Learning._ **arXiv:1904.07734, abstract verified by fetch.** The taxonomy M106
  restriction on continual-learning language refers to.

**Correction to §6.1 P2, recorded here and in P2 itself.** P2 cited Blum &
Rivest's NP-completeness result in a context that implied routing or
distribution assignment is intractable. **That result is about *training* a
3-node network, it was already flagged unverified in §8.6, and it does not bear
on routing at all.** The result that does bear on the question is Fang et al.,
and its own abstract points the other way.

**The distinction M104–M106 depend on, and the program's own evidence for it.**
v14 M90.2 (sealed `0120ccc0…`) measured a **closed-set** domain probe at
**0.8946** on DomainNet, and measured **open-set** rejection at AUROC **0.5839**
against a 0.5851 baseline — chance. M90.2 was filed as a *failure* because H94
wanted domain information gone. **A router among known experts solves the
closed-set problem, at which this program already measures 0.8946; every v13 and
v14 rejection failure was on the open-set problem, which Fang et al. show is
conditionally unlearnable and which M104–M106 do not attempt.** This is a re-read
of sealed evidence and not a new measurement, and **the 0.8946 figure is a v14
DomainNet figure that may not be compared to any CIFAR-10 figure** (R7,
prohibition 24).

---

## 9. What v15 does differently from the lineage in §8

Stated as **differences**, not as advantages, and not as novelty. Every item
below is a difference in **experimental frame**, and each one is a place where a
v15 result and a published result are **not comparable** rather than a place
where v15 is better.

**D1 — the budget is on the fired set per decision, tied to a registered
deployment context.** §8.4's sparse-dictionary lineage measures sparsity as
`L0` of the code, and §8.3's additive lineage measures it as model size or
component count. V15's currency is **the number of atoms a single decision
cites**, capped at 10, derived from `ACCEPTANCE_CRITERIA_v13.md` §2's assisted-
triage setting with a ≤30 s read. Under this currency an 8192-atom dictionary
citing 6 atoms per decision is inside the budget and a 32-atom head citing all
32 is outside it — which inverts the usual ordering and is why §2.1's table
looks unlike a published sparsity table.

**D2 — accuracy and forward simulability are reported jointly, each against a
structure-matched null.** §8.4's simulatability lineage (Doshi-Velez & Kim;
Hase & Bansal) establishes the protocol; §8.3's additive lineage reports
accuracy. V15 reports both axes for the same arm at the same budget, with a
shuffled-label null on the accuracy axis and a shuffled-explanation null on the
simulatability axis. **Hase & Bansal's own finding — that properly controlled
gains are modest — is the reason this is registered as a frontier report rather
than as an expected win.**

**D3 — atoms are grown against a discriminative residual in a learned
overcomplete basis.** §8.2's pursuit lineage grows against a **reconstruction**
residual; §8.3's boosting lineage grows against a discriminative residual but in
**axis-aligned** or tree-structured components. M97 combines the discriminative
residual with a learned dense direction. **§8.2's task-driven dictionary
learning (Mairal et al.) and label-consistent K-SVD (Jiang et al.) are the
closest prior art, and §7.3's null (b) exists specifically so that any v15 result
is reported against that lineage rather than around it.**

**D4 — the free control is a gating bar, not a footnote.** L6 (§5.7) makes
`knn` at `(0.661255, 6.72, 1.0000)` a bar that any progress claim must clear on
**both** axes. §8's lineage papers generally compare against unconstrained deep
baselines on the accuracy axis and against other interpretable methods on the
interpretability axis; comparing against the cheapest composable baseline on
both axes at once is what produced §2.2 Reading 2, and it is uncomfortable
enough that registering it in advance is the only way it survives contact with a
result.

**D5 — every operand carries a structure-matched null and every instrument is
validated at both ends.** R5 (§5.4) and §5.5. An arm that fails its instrument
is void rather than negative. This is why v15 can register outcomes in which
nothing wins as successes (§10).

**D6 — the artifact supports exact edit and rollback.** The program's
`src/model_editor.py` supports removing or replacing an individual component and
recomputing the decision exactly. This is a property of the additive form and is
recorded as a capability of the artifact rather than as a measured result;
**no v15 milestone measures it and no v15 claim rests on it.**

**D7 — the components are fitted in closed form, not by gradient descent on an
auxiliary objective.** §8.5's most encouraging verified result — Belilovsky et
al. (2019), greedy layerwise VGG-11 at 0.676 top-1 against 0.679 end-to-end on
ImageNet — removes the **end-to-end** gradient path while still training each
layer by gradient descent on a shallow auxiliary problem. M96 and M97 adopt the
stricter constraint that each added component is fitted by closed-form or convex
means. **This is a difference that makes v15 harder, not better**: §6.1 P4
records that Belilovsky et al.'s result bounds v15's ambition from above rather
than predicting its outcome, and §6.1 P5 records that methods abandoning
gradients altogether have a large measured cost at scale (feedback alignment at
0.0692 top-1 on ImageNet against a 0.3607 backpropagation baseline).

**D8 — the efficiency accounting includes the backbone, and the plan expects the
answer to be unflattering. [new in v15]** §8.7's efficiency literature is
overwhelmingly reported at the level of the component being sparsified: pruning
papers report on the pruned layers, MoE papers report active-versus-resident
parameters, sparse-kernel papers report kernel-level speedups. §5.11.1 instead
makes the **whole-system** figure primary and the component figure a disclosed
secondary, which is why §2.6.2 can state that head sparsity controls under 0.05%
of this system's inference compute — a fact that a component-level accounting
would never surface. **This is a difference that makes v15 look worse, not
better**, and P6 registers the expected negative before M100 runs.

The nearest prior art for the framing is Hooker (2020) and the FLOPs-versus-
latency literature at §8.7 A1–A3, which argue the general point. §5.11.2's
prohibition on FLOP-only claims and §5.11.6's confirmation replay for an
unexpected positive are the operational form. §2.6.3 records that this program
already had one efficiency story fail on exactly this point and wrote it down.

**D9 — the sequence milestone is scoped to transfer, on the arena where the
method could win, with the losing arena named. [new in v15]** The common pattern
in interpretable-sequence work is to demonstrate on a task where the
interpretable model does well and leave the harder arena unmentioned. §3.3
instead registers **both** halves of P7 in advance: forecasting, where Zeng et
al. and Elsayed et al. verify that simple models beat deep ones, is chosen
because a loss there is informative about the method; and language modelling,
where §8.8 C2 verifies a 3.10× perplexity gap, is named as the arena v15 is
**not** entering and may not claim anything about (§11.2 item 13).

The closest relative of the construction itself is SINDy (§8.8 C4), and its
disclosed failure modes — library design as the bottleneck, noise sensitivity,
and the need for a learned encoder in high dimensions — are registered as v15's
own expected failure modes rather than as someone else's problem. That last
failure mode is the same conclusion §2.6.2 reaches from the cost side, and it is
what makes A5's permission for dense components load-bearing.

None of D1–D9 is claimed to be absent from the literature. §8.6 item 7 applies,
and applies with extra force to D8 and D9, whose supporting sections were each
assembled in a single search pass (§8.6 item 11).

---

## 10. Conditional milestones

### 10.1 M98 — the oracle scaffold (conditional; A5 settled, no longer blocked)

**Opens only if** H102 is refuted by both M96 and M97. The A5 design decision
that previously also blocked this milestone was **settled in favour of
admitting dense components** (§6.2 A5), so M98's only remaining condition is the
H102 refutation.

If growth underperforms, P2 (§6.1) says the cause is the search step. M98 tests
that by replacing the closed-form direction proposer with a **dense multilayer
perceptron used solely as a proposer**: the teacher suggests candidate
directions, the additive model selects and refits, and **the teacher is
discarded**. The delivered artifact contains no dense network.

**Null (R5).** The identical growth procedure with a random-direction proposer,
identical step count, identical refit — this isolates "the oracle was the
constraint" from "more steps helped".

**Kill switch.** H104 is refuted if the teacher-proposed arm does not beat the
random-proposer arm by more than the seed spread at matched budget.

**Registered under A5.3.** The teacher is a **training-time** cost and is
reported in §5.11.5's training column, never summed into inference cost, because
it is discarded before delivery. M98's delivered arm carries no dense component
at inference and therefore does not run the §5.11.4 mechanism ablation — there is
nothing dense left to ablate. An M98 variant that **retained** the teacher at
inference would be a different arm, would run the ablation, and would have to
survive it to count toward Q1.

**Registered restriction, retained for the record.** The pre-settlement fallback
— running M98 with random and closed-form proposers only — is no longer in
force. It is left recorded here because §5.10 and the program's practice require
that a superseded registration be contradicted in place rather than deleted.

### 10.2 M99 — growing the representation (conditional, and isolated)

**Opens if M100 confirms H106 — regated in v15.** The original gate was *"H102
refuted and H104 confirmed"*, and §5.10 requires that a superseded registration
be contradicted in place rather than deleted, so it is stated here and then
replaced.

**Why the original gate was wrong, and this is the plan correcting itself.** The
original gate placed M99 behind two **head** milestones. §2.6.2 establishes that
the head is under 0.05% of inference compute and the representation trunk is the
other 99.95%. M99 is therefore **the only milestone in v15 that touches the part
of the system where the compute actually is** — and it was gated behind
milestones that P6 (§6.1) already registers as unable to produce an efficiency
result. That ordering conditions the one compute-relevant milestone on outcomes
that are irrelevant to compute, and it would have been an error of the plan's own
making rather than of the evidence's.

**The replacement gate, and why it is tied to evidence rather than loosened.**
M99 opens when **M100 confirms H106** — that is, once the cost ledger has
actually demonstrated that trunk cost dominates on this hardware, rather than
merely being calculated to. This keeps M99 conditional on a measurement, which
is what a gate is for, while conditioning it on a measurement that is *about the
thing M99 addresses*. §3.4.4's non-interference contract is preserved: M99
remains a Q1 milestone, its confirmation still cannot rescue a refuted H102, and
H106's confirmation gates only M99's **execution**, never any Q1 verdict.

**Registered consequence, stated against interest.** The new gate makes M99
easier to reach, and easier-to-reach milestones are exactly the multiplicity
risk §3.4.3 exists to control. Two things hold it: M99's own acceptance
criterion is unchanged and unchanged in strictness (the four restrictions below),
and M99 continues to report against an **external** bar on a **separate corpus**,
so it cannot contribute an operand to any v15 comparison. Reaching M99 more
often does not create a new way for v15 to declare success.

**Second regate. [recorded after execution — supersedes the replacement gate
above, which is retained per §5.10]** The replacement gate conditions M99 on
M100 confirming H106. §2.9.2 measures that **M100 cannot currently supply that
confirmation**: the only multi-scale backbone features the program holds are
INT8 exports whose quantisation damage is confounded with model size and is not
monotone in it, one of the three is void outright, and the v13 source images are
no longer on disk to re-extract from. A gate conditioned on a measurement that
cannot presently be made is not a gate but a block, and §3.2.1 registers M99 as
one of only two remaining live routes to Q2.

**The operative gate is therefore: M99 opens when _either_ M100 confirms H106
_or_ M103 (§7.9) confirms that additive atom selection beats a random dictionary
at matched size.** The second disjunct is the substantive one. M103 is
unconditional, runs on CIFAR-10 — M99's own corpus — needs no image
re-download, and tests M99's premise directly rather than by proxy: if atom
choice does not beat random selection at 1024 atoms, M99's registered question
("does additive construction reach the bar with *fewer* patches") is already
answered negatively and M99 should not run at all.

**Why this is not gate-loosening.** The disjunct added is **harder to satisfy
than the one it joins**, not easier. H106 asks whether trunk cost dominates,
which §2.6.2 already calculates at 99.95% and which M100 would essentially
confirm; M103 asks whether the plan's central thesis holds on the
representation, and §2.9.3 supplies a single-seed observation pointing the
**wrong way** — random beat k-means at all five budgets. The program is
conditioning its largest remaining milestone on a result its own scoping
evidence currently predicts will fail. That is registered here so that an M103
confirmation cannot later be described as expected.

**[recorded after execution — the prediction in the paragraph above did not
survive, and the paragraph is retained per §5.10.]** §2.9.4 measures the second
reading of §2.9.3 and finds it holds: discriminative selection beat random in
all six seed-budget cells at 3.0–3.7× the null's seed spread. **The claim that
this disjunct is harder than H106's still stands** — M103 remains a test of the
central thesis against a null the program has measured to be strong, and its
kill switches are unchanged. What no longer stands is the accompanying
prediction of failure. The consequence registered above is now the operative
one in reverse: an M103 **refutation** cannot be described as expected either,
because the program's scoping evidence now points the other way. Both directions
are on the record, dated, before M103 runs.

M99 would build the **representation** additively from image patches with no
end-to-end backpropagation, and compare against the published patch-based bars
verified in §8.5.

**Registered bars, from §8.5, verified by fetch.** M99's instrument is
**CIFAR-10**, and its bar is **Thiry et al. (2021) SimplePatch with a linear
head at 0.869**, because that is the closest published setting to the one M99
would occupy: a fixed patch dictionary, no representation learning, a linear
classifier. The secondary reference points are Coates, Lee & Ng (2011) at
**0.796** with 4000 k-means triangle features and a linear support vector
machine, and Shankar et al. (2020) at **0.900** for a compositional kernel,
which is the best verified figure in the family and is **not** a linear-head
result.

**Registered in advance: what would count.** Reaching 0.869 would be a
reproduction, not a result. The registered question for M99 is narrower and is
about the **construction procedure**, not the accuracy: does building the patch
dictionary **additively against a discriminative residual** reach the
fixed-dictionary bar with **fewer patches**, at a stated patch budget, against a
random-patch null of identical size. An arm that needs more patches to reach the
same accuracy has answered the question negatively, and that is reportable.

**Registered in advance: what M99 does not attempt.** §8.5 records that the best
verified backpropagation-free ImageNet representation reaches **0.360 top-1**
with a linear head against **0.811** for a frozen DINOv2 ViT-S/14 linear probe —
a gap of roughly **45 points**. M99 does not attempt ImageNet and no v15
document may suggest that this plan is a route to closing that gap.

**Four restrictions registered now.**

1. **Corpus isolation.** M99 requires a corpus with a published
   backpropagation-free bar. That is not the v13 DomainNet corpus. Under R7 and
   `ACCEPTANCE_CRITERIA_v13.md` §9.7, **no M99 number may be compared to any
   v13, v14 or v15 number**, in either direction. M99 is a separate instrument
   reporting against its own external bar, and its corpus change is registered
   here — before any measurement — for the stated methodological reason that no
   published backpropagation-free bar exists on DomainNet.
2. **Bars re-verified at execution time.** §8.6 item 8. The §8.5 figures were
   verified during registration; they are re-verified by fetching the sources
   again before M99 runs, and any discrepancy is recorded as an amendment to
   this plan rather than absorbed silently.
3. **Null.** A random-patch dictionary of identical size with an identical
   linear head, on identical rows. Additionally, since Thiry et al.'s dictionary
   is itself drawn from data without learning, **their construction is the
   informative null for M99's growth procedure**, not merely a bar.
   **[amended after execution]** §2.9.3 measures that on a subsampled CIFAR-10
   pipeline the random-patch null **beats** the k-means construction at all five
   budgets tried, by a margin that widens with dictionary size. If that ordering
   replicates under M103's seals, then the random-patch dictionary is not merely
   *an* informative null but the **strongest arm the family has produced**, and
   M99 must beat it rather than beating k-means. This restriction is tightened
   accordingly: **M99's acceptance is against the best of {random patches,
   k-means}, at matched size and matched rows**, not against k-means alone.
4. **Determinism.** Patch extraction at scale may require the GPU. §5.8 permits
   that only for upstream extraction sealed by hash, with every gated figure
   computed on CPU. If that is not achievable for M99, M99 reports its figures
   as **non-gating**.

**Fifth restriction, added with the second regate. [recorded after execution]**
M99 may not read any figure at a dictionary size whose implied feature count
puts the linear head below §5.3's sample floor of 10 rows per fitted dimension,
and must report the adequacy ratio beside every accuracy it does read. §7.9
restriction 4 computes that this binds at 2048 atoms on CIFAR-10 under 2×2
pooling. This is registered because C102.2 records the program committing
exactly this error in M102 and having to void a headline figure afterwards.

M99 is registered at this level of detail deliberately. It is the most
interesting milestone in the plan and the one furthest from evidence, and
writing it out further before H102 and H104 report would be planning on
speculation.

---

## 11. Outcomes, and what v15 may not do

### 11.1 Outcome taxonomy

Inherited from `ACCEPTANCE_CRITERIA_v13.md` §10 and applying to v15 only.

| outcome | meaning                                                                                                    |
| ------- | ---------------------------------------------------------------------------------------------------------- |
| **A**   | H102 confirmed: a grown sparse model reaches the accuracy floor inside the explanation budget, and clears L6 |
| **B**   | H102 confirmed with a narrowed claim — the frontier point is reached but L6 is not cleared                  |
| **C**   | Frontier delivered: H102 refuted, but the `(accuracy, budget)` trade for grown models is characterised with intervals, and the mechanism of the shortfall is identified |
| **D**   | Decisive negative with an identified mechanism                                                             |
| **E**   | Decisive negative without an identified mechanism                                                          |
| **F**   | Void — the measurement was invalid and nothing is concluded                                                |

**C is registered in advance as a success.** V15's most likely outcome, given
§2.2 Reading 3's seventeen-point gap and §6.2 A4's corpus restriction, is that
H102 is refuted and the shortfall is characterised. Registering that as a
success is what prevents the failure mode `ACCEPTANCE_CRITERIA_v13.md` §10
identified: continuing to change the setup until a win appears.

**The taxonomy grades Q1 only.** Q2 and Q3 do not have outcome letters and do
not modify v15's outcome letter. This is registered per §3.4.1: adding two
questions must not add two more ways to declare v15 a success. Their results are
recorded as findings in their own right, in their own subsections, under the
reporting order of §11.3.

### 11.1.1 Q2 and Q3 result records **[new in v15]**

Each is reported as a standalone record with its own verdict vocabulary, drawn
so that it cannot be mistaken for a v15 outcome letter.

| question | verdict | meaning |
| --- | --- | --- |
| Q2 | `cost_characterised` | the four currencies are measured for every arm, whole-system and head-only, and H106/H107/H108 are resolved |
| Q2 | `cost_dominance` | H107 refuted — a sparse arm is strictly cheaper in all four currencies at matched accuracy, **and** the §5.11.6 confirmation replay reproduced it |
| Q2 | `not_instrumented` | the zero-cost head null failed to reproduce trunk-only cost; nothing is concluded |
| Q2 | `gate_improvable` | H110 confirmed — the abstention-optimised arm recovered > 60% of oracle gain at 0.50 deferral, beat the budget-matched dense gate, and beat the accuracy-fitted arm |
| Q2 | `gate_not_improvable` | H110 refuted; abstention on this representation did not improve under objective choice. A substantive negative, per §7.7. **This is the recorded M102 Tier A result** — see `analysis/CLAIM_LEDGER_v15.md` |
| Q2 | `training_avoidable` | H111 confirmed — retrieval within 0.01 of every fitted arm at every §5.11.5 rung |
| Q3 | `transfers` | H109 confirmed on the registered corpus and horizon |
| Q3 | `does_not_transfer` | H109 refuted; the additive construction did not beat persistence or its own null |
| Q3 | `void` | an adequacy or leakage check failed, as §2.7 records for the Tier 6 artifact |
| Q3 | `not_run` | budget exhausted before M101; **not** a negative result |

`cost_characterised` is registered in advance as the expected and sufficient Q2
result, on the same principle that makes Outcome C a success: §2.6.2's
arithmetic predicts it, and a plan that only counted `cost_dominance` as a
result would be a plan with an incentive to keep searching for one.

**`gate_improvable` is a Tier A verdict and is not a compute claim.** It records
that the deferral signal improved. It does **not** record that any compute was
saved, because §7.7 Tier A saves none. Only a completed Tier B may be described
in the language of compute, and only through §5.11.4. §11.2 item 20 binds this.
Symmetrically, `gate_not_improvable` is registered in advance as a **publishable
finding**, not a failure: it would establish that the calibration weakness
recorded across three prior program versions (§2.8.3) is a property of sparse
geometry on this representation rather than an artifact of not having tried, and
that is a stronger statement than the program has been able to make so far.

### 11.2 What v15 may not do

1. Reopen, revise or soften any v13 or v14 verdict. Outcome C and the v14
   refutations stand as sealed. **M95 in particular may produce a limit on how
   the sealed figures are read; it may not change them.**
2. Use a threshold-free result to overturn a threshold result (N85.4a).
3. Retouch a sealed evidence file, ever.
4. Waive the sample-adequacy floor (§5.3) for any arm.
5. Substitute a corpus to rescue a failing gate. §10.2's corpus change is
   registered in advance for a stated methodological reason and is isolated from
   every other v15 figure.
6. Select an arm or a budget after seeing the operands, or re-tune a losing arm.
   The M96 and M97 budget sweeps are reported in full.
7. **Claim novelty.** The §8.4 consequence registered in
   `PRIOR_ART_AUDIT_v13.md` binds every v15 write-up: no novelty claim of any
   kind, no assertion that prior art is absent, and the M88 search failures
   disclosed wherever a claim family is discussed. §8.6 and §9's closing
   sentence discharge this for this document; every later document must
   discharge it again.
8. Gate any figure computed on the GPU.
9. **Pursue hierarchical routing.** Registered as a do-not-pursue direction on
   the basis of §2.4 observation A, v14 M89's cell-geometry measurement, v14
   M90.2's erasure result, and the published statement quoted at §8.4 from
   NBDT's related-work section. M94 Part D records the arms under contract and
   closes the direction; nothing further is spent on it.
10. Report a bare accuracy figure for a sparse arm. L1′ (§5.7) makes the joint
    `(accuracy, fraction within budget)` report mandatory.
11. Report a maximum over arms as an arm's result (§7.8, N82.7).
12. Register a bar against an external number that was not verified by fetching
    its source. §8.5, and the figures explicitly marked **verified** in §8.7 and
    §8.8, are the only places in this plan from which a bar may be set (§8.6
    items 1, 9, 10). Every figure marked **unverified** — including the M4 and
    M5 official scores, MoE inference latencies, and end-to-end N:M application
    latency — is inadmissible as a bar.
13. **Extrapolate a forecasting result to language modelling.** §8.8 C2 records a
    verified 3.10× perplexity gap between the best sparse count-based model and a
    Transformer on One Billion Word. No M101 outcome, in either direction,
    licenses any statement about whether sparse additive construction can rival
    autoregressive language models. Q3 is a transfer question (§3.3).
14. **Borrow the structured-sparsity hardware result.** §8.7 A4's verified 2×
    Tensor Core speedup belongs to N:M structured sparsity with a retraining
    workflow. v15's sparsity is unstructured and its measurement is CPU-only
    (§5.11.7). Citing A4 in support of v15 being fast is forbidden.
15. **Cite sparse-attention or state-space work as interpretability evidence.**
    §8.8 C5 and C6 record that these are efficiency and long-context results;
    none of those papers claims inspectability. §8.8 C7 additionally records that
    attention weights are contested as explanations at all.
16. **Describe a FLOP or parameter reduction as an efficiency result without the
    measured wall-clock beside it** (§5.11.2), or report a head-only cost without
    the whole-system cost beside it (§5.11.1). §2.6.3 records that this program
    has already had a routing-efficiency story fail on exactly this point.
17. **Introduce an efficiency threshold as a gate on Q1** (§3.4.2), promote a
    secondary hypothesis to primary (§3.4.6), offer a Q2 or Q3 result in place of
    Q1's verdict (§3.4.1), or use a result on one axis to rescue a refutation on
    another (§3.4.5).
18. **Treat `not_run` as a negative result.** If M101 is not reached within
    budget, Q3 is unanswered, and no document may imply otherwise (§11.1.1).
19. **Extend H111 to representation training.** H111 (§4.3) concerns fitting a
    **head** on a frozen representation. Confirmation licenses no statement about
    the cost of training the representation itself, which is where the field's
    training compute actually goes and which this program never touches. "This
    program shows training is unnecessary" is a forbidden sentence under every
    outcome.
20. **Report a Tier A abstention result in the language of compute saving.**
    §7.7 Tier A runs every arm on the full 384-d feature, so every arm pays the
    full 6,065,759,232-MAC trunk and **no arm saves anything**. Tier A confirms
    or refutes gate quality only. Its excluded-trunk MAC figure may not appear
    without the unchanged-trunk statement in the same table, and `gate_improvable`
    may not be described as an efficiency result (§11.1.1).
21. **Quote §2.8's figures as sealed evidence.** They come from a v15 planning
    probe on a different split and a different kNN configuration; its full-model
    reading is **0.6357** where the sealed M81 arm reads **0.661255**. Under
    §2.4's convention they are scoping observations and are inadmissible as
    operands. **[updated after execution]** This rule was written with an escape
    condition — *"until M102 reproduces them under seal"* — and **that condition
    has now resolved negatively**: M102 ran and did **not** reproduce them
    (§2.8.5). The escape is therefore closed, not pending. §2.8's figures are
    permanently inadmissible and the corrected readings in M102's evidence file
    supersede them. In particular the **44.4%** figure that set H110's bar is
    **not reproducible** and may not be quoted anywhere without §2.8.5 beside it.
    H110's 60% target was registered against it in full knowledge that it was a
    probe number (§4.3); that the anchor later proved wrong does not
    retrospectively change the bar, and the bar was cleared by no arm regardless.
22. **Present joint gate training, selective prediction or cascading as new.**
    §8.9 D1–D3 record a mature literature: SelectiveNet trains predictor and
    selector together, learning-to-defer supplies consistency guarantees, and
    FrugalGPT, BranchyNet and CALM report measured cascade savings. §2.8's
    statement that abstention was never optimised is about **this program's
    history only**, and no v15 document may let it read as a claim about the
    field. M102's contribution is stated in §8.9 D5 and nowhere more strongly.
23. **Quote any §2.9 figure as evidence. [added after execution]** All probes in
    §2.9 are unsealed, unreplicated scoping observations under §2.4.
    Specifically: the ensemble deltas of §2.9.1 may not be reported as a measured
    bound on cascade gains; the backbone curve of §2.9.2 may not be reported
    without the INT8 divergence column and the statement that dinov2-base is
    **void**; §2.9.3's random-beats-k-means ordering is **single-seed**, is
    **Thiry et al.'s published result rather than this program's**, and may not
    be described as a finding of this program in any document; and §2.9.4's
    discriminative-selection gain may not be described as a finding, a result, or
    evidence that the plan's central thesis holds until M103 reports it under
    seal; and §2.9.6's figures are **single-seed**, are drawn from an
    instrumentation run built to size M103 rather than to answer it, and were
    produced under a **truncated regularisation grid whose top value won for
    every arm**, so none of them may be quoted as an accuracy this family
    reaches. **Escape condition:** each figure becomes quotable only through the
    milestone that reproduces it — §2.9.3's, §2.9.4's and §2.9.6's
    through M103 (§7.9), and §2.9.2's through M100.

    **[escape condition discharged in part, after M103 executed.]** M103 has run
    and its result is recorded in `analysis/CLAIM_LEDGER_v15.md` C103.1–C103.8.
    What is now quotable is **M103's own sealed figures**, under the
    restrictions listed in that ledger entry — *not* the probe figures
    themselves. Specifically: §2.9.4's discriminative-selection gain is
    reproduced under seal and may be described as a finding **in the form
    C103.1 states it**, with C103.3's narrowing margin attached; §2.9.3's
    ordering is reproduced under seal (C103.5) but **remains Thiry et al.'s
    result and not this program's**, so that half of the prohibition stands
    unchanged; and §2.9.6's finding 2 is **contradicted** by M103 rather than
    confirmed, so it does not escape at all and is now doubly unquotable.
    §2.9.1's and §2.9.2's figures are untouched by M103 and remain bound.
24. **Compare any CIFAR-10 figure to any DomainNet figure. [added after
    execution]** §2.9.3, M99 and M103 run on CIFAR-10 at 10 classes; every v13,
    v14 and v15 figure is DomainNet at 128 classes. R7 and §10.2 restriction 1
    already forbid this, and it is restated as a standing prohibition because
    §2.9.3's 0.6339 and §2.9.2's 0.6322 are numerically adjacent by coincidence
    and invite exactly this error.
25. **Present effective-rank measurement, constructive white-box networks, or
    rate reduction as this program's. [added after the M103 prior-art audit]**
    RankMe is Garrido, Balestriero, Najman & LeCun's (§8.10.2) and is used
    **unmodified**; α-ReQ is Agrawal et al.'s; ReduNet and CRATE (§8.10.3) are
    constructive white-box networks that already exist, at ImageNet scale, with
    public code. No GEODE document may describe measuring effective rank as an
    instrument this program built, and **this program may not build a
    rate-reduction objective, a constructive white-box network, or an
    effective-dimension instrument** while all three exist. §2.9.7's figures are
    additionally bound by prohibition 23 as unsealed probes.
26. **State C103.1 without its prior art. [added after the M103 prior-art
    audit]** §8.10.1 records that C103.1's phenomenon is published in a stronger,
    label-free form with matching lower bounds (Avron et al. 2017; Li et al.
    2019). Every statement of C103.1, in any document, must carry that
    disclosure **and** C103.3's narrowing margin, which restriction 5 of the
    M103 ledger entry already binds. A statement of C103.1 as a novel efficiency
    finding is a misstatement of the record, not merely an omission.
27. **Claim an efficiency result against dense networks from M104, M105 or M106.
    [added after the M103 prior-art audit]** §7.13 records that **no v15
    milestone as registered compares anything to a dense network**, which is the
    form Q2 (§3.2) is actually written in. Until a dense comparator is measured
    on the same corpus at matched accuracy, the admissible claim from M104–M106
    is efficiency **relative to a uniform mixture** — M104's own null — and
    nothing wider. Registered before M104 runs so it cannot be relaxed after a
    favourable number.

### 11.3 Reporting order **[new in v15]**

Registered so that the write-up cannot lead with whichever question came out
best.

1. Q1's outcome letter and H102's verdict, first, in every summary, abstract,
   ledger entry and README line.
2. Q2's record, second, with the whole-system figure before the head-only figure.
   Where an M102 result is reported, the **tier** is named in the same sentence
   as the result (prohibition 20), and Tier A results are stated as gate-quality
   findings before any Tier B compute figure is given.
3. Q3's record, third, with its corpus isolation restated.
4. The §5.10 disclosure, §6.2 A4's corpus restriction, and §8.6's search
   failures, in every document that carries a claim.

No document may report Q2 or Q3 without reporting Q1's verdict in the same
document. No document may report a head-only efficiency figure without the
whole-system figure beside it (§5.11.1). No document may report §2.8's or M102
Tier A's figures without stating that the trunk cost is unchanged across every
arm being compared.

