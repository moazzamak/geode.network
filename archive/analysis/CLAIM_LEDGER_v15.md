# GEODE Claim Ledger v15

Every claim v15 is entitled to make, with the evidence file that carries it and
the restriction that binds it. A claim absent from this ledger is not a v15
claim. Registered under `analysis/RESEARCH_IMPLEMENTATION_PLAN_v15.md`.

**Reporting order (plan §11.3).** Q1's verdict leads every document. Q2 and Q3
are separate axes that do not carry outcome letters. No head-only efficiency
figure appears without the whole-system figure beside it.

**Status of v15 as a whole:** Q1 not yet answered — M94–M98 have not run. The
executed milestones are **M102** and **M103**, both of which are **Q2**
milestones. Under §3.4.1 this ledger therefore records **no v15 outcome
letter**, and nothing below may be read as one.

**M103 is the first affirmative sealed result this program has produced on its
own central thesis.** It is also a Q2 result on a single corpus with a
narrowing margin, and C103.3 states the limitation that binds it. Both
sentences are part of the finding.

---

## M102 — abstention as the objective (Tier A)

| field | value |
| --- | --- |
| hypothesis | **H110** |
| question | **Q2** (efficiency) — not Q1, and carries no outcome letter |
| tier | **A** — gate quality only |
| verdict | **refuted** |
| evidence | `logs/results/v15/m102_abstention/evidence.json` |
| runner | `experiments/tier4/eval_v15_m102_abstention.py` |
| configuration | `experiments/configs/v15/m102_abstention.json` |
| corpus | sealed v13 DomainNet, index `a6485f90…41bbf85` |
| seeds | 11, 23, 37 |
| primary operating point | 32 PCA dimensions, 50% deferral — both registered before the run |

### C102.1 — H110 is refuted, and the joint objective made the gate worse

At the registered operating point, the mean recovered fraction of
oracle-available gain across three seeds:

| arm | recovered | spread |
| --- | ---: | ---: |
| (b) sparse linear, gate = logit margin | **30.6%** | 5.5 |
| (b′) same model, temperature-scaled gate | 27.7% | 5.7 |
| (a) nearest class mean, gate = distance margin | 22.5% | 6.2 |
| (c) **sparse selective, gate trained with the task** | **13.6%** | 3.3 |
| (d) budget-matched dense gate, supervised on correctness | *(−1.9%)* **void — see C102.2** | 8.0 |

Arms (a), (b), (b′) and (c) are fitted on the 57,344-row fit split and are
sample-adequate at this width. **Arm (d) is fitted on the 8,192-row calibration
split and is not**, so its cell is void rather than negative and is excluded
from H110's verdict. H110's verdict rests on arm (c) against arm (b), both
adequate, and is unaffected.

H110's registered bar was **60%**, fixed in the plan before the milestone was
written. The best arm reaches **30.6%**, and the arm H110 is about — (c), the
one whose objective *is* deferral quality — reaches **13.6%**, less than half
the accuracy-fitted comparator it was supposed to beat. Arms (b) and (c) share
trunk, width, L1 penalty, epochs, optimiser and seed, and differ only in the
loss, so the comparison isolates the objective.

**Entitled claim.** On this corpus and representation, training the gate jointly
with the task under a SelectiveNet objective produced a *worse* deferral signal
than reading a margin off an accuracy-fitted model.

**Not entitled.** Any statement that joint gate training does not work in
general. §8.9 D2 records that the method is established and carries consistency
guarantees; this is one corpus, one representation, one sparse family, one
hyper-parameter setting.

### C102.2 — a directly supervised gate does not work, but the reading is narrower than first written

**[corrected after execution — the first version of this entry breached the plan's
own sample floor, and the correction is recorded rather than substituted.]**

Arm (d) is a dense MLP given the **exact** supervised signal a gate wants — "was
stage-1 right on this row?" Unlike every other arm it must be fitted on the
**calibration** split, because the fit split is what stage-1 was trained on and
its correctness labels there are not honest. That leaves it 8,192 rows:

| dims | gate parameters | calibration rows per parameter | §5.3 floor of 10 |
| ---: | ---: | ---: | :--- |
| 8 | 641 | **12.78** | passes |
| 16 | 1,153 | 7.10 | **void** |
| 32 | 2,177 | **3.76** | **void** |
| 64 | 4,225 | 1.94 | **void** |

**The first version of this entry quoted −1.9% at 32 dimensions, which is a void
width.** That is the same defect this ledger records in C102.6 against the plan's
own probe, committed one entry later by this ledger. It is corrected here and
left visible per §5.10.

**On the one adequate width**, 8 dimensions, arm (d) reads **+2.6%** at 50%
deferral (spread 3.3), against a random-deferral null of 0. It is the **worst of
the five arms** at that width at every deferral rate measured.

**Entitled claim.** A dense gate given direct correctness supervision, trained on
the honest data available for that supervision, does not produce a useful
deferral signal at the one width where it is sample-adequate.

**Not entitled — and this is where the first version overreached.** The original
entry read this as evidence that *the representation does not carry the signal*,
and pointed at M99. It does not support that. At the same 8 dimensions arm (c)
recovers **25.1%**, so a substantial deferral signal **is** extractable from an
8-dimensional representation — by a model fitted on 57,344 rows rather than
8,192. The distinction between arm (c) and arm (d) is confounded by training-set
size and cannot be attributed to sparsity, to architecture, or to the
representation. The honest reading is that **direct correctness supervision is
sample-expensive**, not that the information is absent.

### C102.3 — temperature scaling does not help, and the baseline kill switch did not fire

Arm (b′) applies Guo et al. (2017) temperature scaling, fitted on the
calibration split, to arm (b)'s logits. It reads **27.7%** against arm (b)'s
**30.6%** — very slightly *worse*. Plan §7.7's second kill switch (H110 refuted
by sufficiency of the baseline) did **not** fire.

**Entitled claim.** The gate's weakness here is not a calibration-of-confidence
problem in the Guo et al. sense; a single post-hoc scalar does not address it.

### C102.4 — Reading 1 survives, corrected, and is larger than the probe claimed

The corrected oracle at 32 dimensions reaches **0.6976** balanced accuracy
against the full weighted-kNN model's **0.6322**, and it does so at **every**
deferral rate measured, down to **25%**.

**Entitled claim.** On this corpus, a cascade that sends only a quarter of
inputs to the expensive model can, given a perfect deferral rule, exceed the
expensive model applied to everything. The difficulty skew is real and larger
than plan §2.8 estimated.

**Not entitled — and this is the binding restriction on the whole milestone.**
**No compute saving is claimed or measured.** Every Tier A arm consumes the full
384-dimensional DINOv2 feature and therefore pays the full
**6,065,759,232-MAC** trunk on every input, including the inputs it declines to
escalate. The trunk term is identical across all arms; the head-and-gate MACs
excluding it are a rounding error (plan §2.6.2). Plan §11.2 item 20 forbids
reporting any figure in this ledger entry in the language of compute saving.
Tier B, which would refit stage-1 on a cheap input so a non-escalated row
genuinely never pays the trunk, is conditional on H110 and **does not open**.

### C102.5 — an instrument defect was found and is recorded, not absorbed

The first implementation of the oracle deferred stage-1's errors first, chosen
at random among them. That is an upper bound only when errors fit inside the
deferral budget, which they did not: every stage-1 arm was wrong on more rows
than the budget allowed it to defer. The defect surfaced as arm (c) reading
**141.4%** of a quantity that was supposed to cap it.

The corrected oracle ranks rows by deferral benefit (+1 where stage-1 is wrong
and stage-2 right, 0 where deferring changes nothing, −1 where stage-1 is right
and stage-2 wrong) and spends the budget highest-benefit first. On an exactly
balanced evaluation split this is optimal for balanced accuracy, so it is a true
bound. The runner now **raises** on any point where an arm beats the oracle,
rather than reporting it.

**Consequence for the plan.** Plan §2.8's figures were computed on the defective
oracle and are all too high; §2.8.5 contradicts them in place per §5.10. In
particular the **44.4%** that anchored H110's bar is not reproducible — the
corrected reading at the same operating point is **30.6%**. The measured gap
between what a cascade could reach and what a gate delivers is **wider** than
the plan claimed.

### C102.6 — which widths are void, and for which arms

Plan §2.8 took its headline from a 64-dimensional stage-1. At M102's 448 fit
rows per class that is **7 fit samples per fitted dimension**, below the floor
of 10 that plan §5.3 never waives. The arm is reported **void** under
M83.1/N83.8 and carries no verdict. The primary operating point is 32
dimensions at 14 fit samples per dimension.

**Sample adequacy is per arm, not per width, because the arms are not fitted on
the same split.** Arms (a), (b), (b′) and (c) are fitted on the 57,344-row fit
split and are adequate at 8, 16 and 32 dimensions. Arm (d) is fitted on the
8,192-row calibration split and is adequate **only at 8 dimensions** (C102.2).
The 64-dimension column is void for every arm.

**Registered consequence.** M102's design did not state a sample floor for the
gate arms separately from the stage-1 arms, and this ledger initially read a
void arm-(d) cell as a finding. Any future milestone with arms fitted on
different splits must check adequacy per arm before any cell is read.

### C102.7 — a registered secondary observation, not an operand

Arm (c) is the **best** arm at 8 and 16 dimensions (35.6% and 35.9% at 40%
deferral) and the **worst** at 32 and 64. The joint objective appears to help at
low stage-1 capacity and hurt at higher capacity.

**Not entitled.** This is a post-hoc pattern across a sweep that was registered
for a different purpose, at widths that are not the primary operating point. It
is recorded so that it is not rediscovered and presented as a finding later, and
it is **not** evidence for or against H110. Testing it would require its own
registration.

### Restrictions binding every claim above

1. **Q2 only.** M102 answers no part of Q1 and produces no operand admissible in
   any H100–H105 comparison (plan §3.4.7 item 2).
2. **No compute language** (plan §11.2 item 20). See C102.4.
3. **No novelty.** Joint gate training, selective prediction and cascading are
   established prior art (plan §8.9 D1–D3, §11.2 item 22). M102's framing —
   an oracle denominator — was searched for and not located, which plan §8.6
   item 15 records as a **search limit and not an absence**.
4. **One corpus, one representation.** No M102 figure transfers to any other
   corpus, and none may be compared to any published number as an operand (R7).
5. **The probe is not evidence.** Plan §2.8's figures remain inadmissible
   (§11.2 item 21). M102's evidence file supersedes them.

---

## M103 — is a grown dictionary better than a drawn one?

| field | value |
| --- | --- |
| registered in | plan **§7.9**, unconditional, before any M103 figure existed |
| question | **Q2** (efficiency), on the M103 → M99 chain — not Q1, and carries no outcome letter |
| verdict | **confirmed** |
| evidence | `logs/results/v15/m103_atoms/evidence.json` |
| payload hash | `65972da7…17ba61` |
| runner | `experiments/tier4/eval_v15_m103_atoms.py` |
| configuration | `experiments/configs/v15/m103_atoms.json` |
| corpus | CIFAR-10, full 50,000/10,000 splits, sha256 `e4c49989…6879937` |
| seeds | 11, 23, 37 |
| budgets | 64, 128, 256, 512, 1024 readable; **2048 void** (§5.3 floor) |
| primary operand | atom count at which each arm first reaches arm (a)'s accuracy at 1024 atoms — registered in §7.9 design item 2 |

### C103.1 — a discriminatively grown dictionary reaches the null's accuracy with half the atoms

Mean test accuracy over three seeds, all five sample-adequate rungs:

| atoms | (a) random patches | (b) k-means | (c) **discriminative** | (d) random projections | (c) − (a) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.5730 | 0.5746 | **0.6065** | 0.5349 | **+0.0335** |
| 128 | 0.6211 | 0.6143 | **0.6414** | 0.5765 | **+0.0203** |
| 256 | 0.6520 | 0.6477 | **0.6746** | 0.6167 | **+0.0226** |
| 512 | 0.6720 | 0.6705 | **0.6908** | 0.6476 | **+0.0188** |
| 1024 | 0.6879 | 0.6839 | **0.6983** | 0.6684 | **+0.0104** |

The registered operand is efficiency at matched accuracy. Arm (a) reaches
**0.6879** at 1024 atoms. Arm (c) passes that at **512 atoms** — a **2.0×**
reduction by rung, **2.2×** by linear interpolation between the 256 and 512
rungs (**465.8** atoms). Arms (b) and (d) never reach it at any readable rung.

**This holds per seed, not only on means.** All three seeds reach the reference
at the 512 rung, and each seed's own arm (c) at 512 beats *that same seed's*
arm (a) at 1024: 0.6862 vs 0.6818 (seed 11), 0.6950 vs 0.6919 (seed 23),
0.6913 vs 0.6900 (seed 37).

**Kill switch 1 did not fire.** §7.9 registered that if arm (c) failed to reach
the reference with fewer than 1024 atoms at three seeds, the finding would be
that *atom count dominates atom choice*. Arm (c) beat arm (a) in **15 of 15**
seed-budget cells.

**Entitled claim.** On CIFAR-10 with this patch pipeline, selecting dictionary
atoms one at a time against the residual of a discriminative objective reaches
the accuracy of a randomly drawn dictionary of 1024 atoms using **512** atoms,
at three seeds, on a sealed pre-registered operand.

**Not entitled.** That this transfers to any other corpus, representation or
task; that it is a novelty (see restriction 3); or that it is a Q1 result. It
is one corpus, one pipeline, one selection criterion.

**[Amendment, recorded after a prior-art audit. The claim above is not edited;
this is added beside it per §5.10.]**

**The phenomenon C103.1 measures is already published, in a stronger form, by a
label-free mechanism, with matching lower bounds.** The audit is recorded in
plan §8.10.1 and its sources were verified by fetch:

| source | what it establishes |
| --- | --- |
| Avron et al., ICML 2017 (arXiv:1804.09893) | uniform random features need `O(n_λ·log s_λ)`; **ridge-leverage** sampling needs `O(s_λ·log s_λ)` with `s_λ ≤ n_λ`, **and a matching lower bound** |
| Li, Ton, Oglic & Sejdinovic, ICML 2019 (arXiv:1806.09178) | uniform `Ω(√n·log d_eff)` versus leverage `Ω(d_eff)`; under low noise `Ω(log n log log n)`, or *"even a constant number of features in some benign cases"* |
| Rudi & Rosasco, NeurIPS 2017 (arXiv:1602.04474) | `O(√n log n)` suffices; faster rates need *"a possibly problem dependent distribution"* |
| Sinha & Duchi, NeurIPS 2016 | the supervised analogue. **Figures not verified by fetch** — recorded as a search failure under plan §8.6 |

**Three readings, all against this entry's interest.**

1. **The 2.0× is a constant inside a published asymptotic separation.** C103.1
   is a replication of a known effect on a new pipeline, not a new effect.
2. **C103.3 is *predicted*, not merely volunteered.** The improvement factor is
   `n_λ/s_λ` and contracts toward 1 once the budget passes the problem's
   effective dimension. This entry reported the narrowing honestly and against
   interest before knowing the theory said to expect it. That is to the
   program's credit as process and **to the claim's cost as a contribution**.
3. **The stronger published mechanism is label-free.** Ridge-leverage sampling
   uses no labels; arm (c) does. The mechanism built here is therefore **more
   restrictive and less powerful** than the published one.

**What does not change.** No figure in this entry is withdrawn, no arm is
revoided, no kill switch is re-evaluated, and the seed-level result stands
exactly as measured. What changes is the claim's **standing as a
contribution** — which restriction 3 below already governed, and which
plan prohibition 26 now makes explicit: **no statement of C103.1, in any
document, may omit this disclosure or C103.3's table.**

**What the audit gives back.** The same theory names a **computable target** the
program never had — the effective dimension `d_eff` — and plan §2.9.7 and
milestones M104–M106 (§7.10–§7.12) are registered on it.

### C103.2 — provenance in the data matters, and kill switch 2 did not fire

Arm (d) draws isotropic Gaussian directions and then **resamples the patch
pool's empirical norm distribution**, so it differs from arm (a) in *direction
only*. It is below arm (a) at every readable rung, by 0.0195 to 0.0446, far
outside the null's own seed spread (0.0053–0.0131).

**Entitled claim.** The content of these dictionaries is not reducible to their
size and norm geometry. Atoms drawn from the data outperform directionally
arbitrary atoms carrying the same norm distribution. §7.9's second kill switch,
which would have weakened both M103's and M99's premises, did not fire.

### C103.3 — the margin narrows as the budget grows, and this is the limitation that binds C103.1

Stated against interest, and prominently, because it is the strongest reading
available *against* the result:

| atoms | (c) − (a) | arm (a) seed spread | ratio |
| ---: | ---: | ---: | ---: |
| 64 | +0.0335 | 0.0097 | **3.45** |
| 128 | +0.0203 | 0.0131 | 1.55 |
| 256 | +0.0226 | 0.0053 | **4.26** |
| 512 | +0.0188 | 0.0073 | 2.58 |
| 1024 | +0.0104 | 0.0101 | **1.03** |

At the **top readable rung the margin is barely one seed spread**. The trend
across the sweep is downward, and the honest extrapolation is that the advantage
may continue to shrink and could vanish at budgets M103 cannot read. The
efficiency claim in C103.1 rests on the 512 rung, where the ratio is 2.58, and
on a reference taken at 1024, where it is 1.03.

**What this does not do.** It does not void C103.1. The registered operand is
efficiency at matched accuracy, not margin at matched size, and that operand is
met at three seeds individually. But any statement of C103.1 that omits this
table is a misstatement of the evidence, and plan §11.2 binds it.

**Registered consequence for M99.** M99 would scale this construction. The
direction of travel in this table is the single most important thing M99 must
measure, and M99 must read budgets above 1024 on a corpus large enough to keep
them sample-adequate — which CIFAR-10 at 50,000 rows is not.

### C103.4 — five fits at a readable rung did not converge, and all five are in the arms that lose

Ten fits hit the 1000-iteration lbfgs limit. Five are at the void 2048 rung and
carry nothing. **Five are at the readable 256 rung**, and their distribution is
against the null:

| rung | arm (a) | arm (b) | arm (c) | arm (d) |
| --- | :---: | :---: | :---: | :---: |
| 256 | seed 11 | seeds 11, 23, 37 | **none** | seed 23 |

**Arm (c) converged at every readable rung; arms (a), (b) and (d) did not.** A
non-converged fit understates its arm's accuracy, so the 256 row of C103.1's
table **flatters arm (c)** by an unmeasured amount.

**What this does and does not touch.** The 256 rung feeds the interpolated
465.8-atom figure, through arm (c) at 256 and 512 and arm (a) at 1024 — **all
three converged**, so the interpolation is not contaminated. The rung answer
(512) does not use the 256 row at all. C103.1's headline is therefore intact,
and the contaminated quantity is the +0.0226 margin at 256, which is not an
operand. It is recorded rather than repaired because repairing it after seeing
the result would be a post-hoc change to a sealed run.

### C103.5 — the instrumentation run's k-means reversal was a single-seed artifact

Plan §2.9.6 finding 2 recorded that §2.9.3's random-beats-k-means ordering
*reverses* at full scale, on the strength of one seed at 1024 atoms
(k-means 0.6856 > random 0.6818). Under seal at three seeds, **arm (a) beats
arm (b) at four of the five readable rungs**, including 1024 (0.6879 vs 0.6839).

The sealed run reproduces seed 11 at 1024 **exactly** — 0.6818 / 0.6856 /
0.6951 / 0.6760, to four decimals, across all four arms — so the disagreement is
not an implementation difference. It is that §2.9.6 read one seed. §2.9.6's
finding 2 is contradicted in place in the plan per §5.10.

**Entitled claim.** §2.9.3's ordering — random patches beat k-means centroids —
**survives under seal** at full scale and three seeds. Under restriction 3 below
this remains **Thiry et al.'s published result rather than this program's**, now
independently reproduced rather than merely recalled.

**Recorded consequence.** A single-seed reading was published inside this
program's own plan and was wrong within nine days. §7.9 design item 3 required
three seeds for exactly this reason, and it was the seeded milestone, not the
probe, that caught it.

### C103.6 — the compute ledger, with training charged in full

Inference, per image, at the matched-accuracy comparison (plan §5.11; patch
extraction and per-patch contrast normalisation are excluded and that exclusion
is stated, per §2.9.3 limitation iv):

| | encoding | whitening | head | **total** |
| --- | ---: | ---: | ---: | ---: |
| arm (a) @ 1024 atoms | 80,621,568 | 8,503,056 | 40,960 | **89,165,584** |
| arm (c) @ 512 atoms | 40,310,784 | 8,503,056 | 20,480 | **48,834,320** |

At matched accuracy arm (c) costs **1.83× fewer inference MACs per image**.

**Training is reported separately and is not netted against inference** (§11.2
item 10). Arm (c) pays a selection cost arms (a) and (d) do not pay at all:
**5.334 TMAC** (133 s) per seed. Because the selection order is nested, that
cost is paid **once for the whole sweep**, but it is charged **in full to every
budget** rather than amortised — the reading against this arm. Arm (b) pays
0.679 GMAC; arms (a) and (d) pay zero.

**Break-even, stated plainly: 132,244 inferences.** Below that, arm (c) costs
more end to end than arm (a) at 1024 atoms. That is **13.2 passes over the
CIFAR-10 test split**, so on a single evaluation pass the construction does
**not** pay for itself.

**Entitled claim.** A saving at inference, of 1.83×, purchased with a
one-off training cost that repays after ~132k inferences. **Not entitled:** any
statement of the inference figure without the training figure beside it.

### C103.7 — the instrument check, as corrected before the run

Plan §7.9 design item 4 as corrected in §2.9.6 — three internal conditions, no
external gate:

| condition | required | measured | |
| --- | --- | --- | :--- |
| monotonicity | arm (b) rises with atom count | 0.5746 → 0.6143 → 0.6477 → 0.6705 → 0.6839 | **pass** |
| internal floor | arm (b) @ 1024 > §2.9.3's 0.6223 | **0.6839** | **pass** |
| encode determinism | bitwise repeatable within the run | `True` | **pass** |

The **Coates 0.796 at 4000 features** is reported here as an **anchor and not an
operand** (R7). M103's top readable rung is 1024 atoms; the two are not
comparable and the anchor gates nothing. §2.9.5 separately records that the
0.869 Thiry bar did not re-verify, and **no M103 arm is reported against it**.

A float64 control re-encoded one cell in double precision: maximum absolute
difference **1.22e-4**, maximum relative difference **3.24e-7**. The float32
encode is not a source of error at the scale of any margin above.

### C103.8 — which rung is void, and the regularisation disclosure

The **2048 rung is void, not negative**: 6.1 rows per fitted dimension against
§5.3's floor of 10, which is never waived. It was registered as expected-void
*before* the run. Its figures (arm (c) 0.7079–0.7111) are recorded in the
evidence file and **are not read**, and they may not be quoted as M103's best
accuracy.

The head's inverse-regularisation constant was chosen once per budget, on the
**null arm (a)** at the first seed — deliberately against interest — and applied
unchanged to every arm and seed. **At 64, 128 and 256 atoms the top of the
extended grid (0.3) was selected**, which means the grid is still truncated at
those budgets and all four arms there are regularised more strongly than their
optimum. This is disclosed rather than re-run, because the grid had already been
extended once after seeing §2.9.6's data and extending it again after seeing
sealed data would be fitting the instrument to the result. It affects all four
arms identically at each budget, so the comparison remains matched.

### Restrictions binding every claim above

1. **Q2 only.** M103 answers no part of Q1 and carries no outcome letter. Q1
   remains unanswered and v15 still has no outcome letter (§3.4.1).
2. **No CIFAR-to-DomainNet comparison** (plan §11.2 item 24, §7.9 restriction 1).
   No M103 figure may be compared to any v13, v14 or v15 DomainNet figure in
   either direction, including M102's.
3. **No novelty.** Discriminative dictionary learning, supervised dictionary
   selection and orthogonal matching pursuit are established prior art, and
   §2.9.3's random-beats-learned ordering is **Thiry et al. (2021)'s published
   result**. M103's contribution is a sealed, seeded, sample-floored measurement,
   not a new idea. Plan §11.2 item 22 binds this.
4. **One corpus, one representation, one criterion.** The selection criterion is
   group OMP against a centred one-hot residual. Nothing here says other
   discriminative criteria behave the same way.
5. **C103.3 travels with C103.1.** Any statement of the 2× efficiency result
   that omits the narrowing margin is a misstatement of this evidence.
6. **Training and inference are never netted** (§11.2 item 10). See C103.6.
7. **The 2048 rung is void** and carries no claim in either direction.
8. **The C103.1 prior-art disclosure travels with C103.1. [added after the
   prior-art audit]** Restriction 3 disclaimed novelty generically; the
   amendment to C103.1 now names the specific published results that subsume the
   phenomenon, and plan prohibition 26 binds every document to carry them.

---

## Milestones not yet run

M94, M95, M96, M97, M98, M99, M100, M101, and M102 Tier B. **Q1 is unanswered**,
so v15 has no outcome letter. H106–H109 and H111 are unmeasured. **M103 has run
and confirmed** (see above); M99, the milestone it was registered to unblock, is
still gated on §10.2's regate and on the cost ledger §2.9.2 shows cannot
currently be produced.

**Registered after the M103 prior-art audit:** **M104** (plan §7.10), **M105**
(§7.11) and **M106** (§7.12) are registered and **not yet run**. M104 is
unconditional; M105 is conditional on M104 surviving all three of its kill
switches; M106 is conditional on M105 surviving its kill switch. §7.13 records a
registration defect — **no v15 milestone as registered compares anything to a
dense network**, which is the form Q2 is written in — and prohibition 27 binds
M104–M106 accordingly. None of these milestones carries an outcome letter; all
three are Q2. **[superseded in place, §5.10: M104 has since run to completion.
Its result is the next entry. "Not yet run" is true of M105 and M106 only, and
they are now closed rather than pending, because M104 did not survive its kill
switches.]**

**M104 result (plan §7.10). Outcome letter: R — refuted. Four of five kill
switches fired.** Measured at three seeds under **oracle** routing on DomainNet,
409,832 train and 100,000 test rows, evidence at
`logs/results/v15/m104_experts/evidence.json`, corpus digest `81099916e5036d1c`.
**Rank-sized allocation scored 22.47% against uniform allocation's 24.22%**
under oracle routing — a **−1.76 pp** margin against uniform's own **0.16 pp**
seed spread, so **kill switch 1 fired** and under §11.1 that is M104's headline.
Rank-sizing also lost to its two nulls: **random-sized 23.54%** (kill switch 2
fired) and **traffic-inverse 23.53%** (kill switch 4 fired), the latter carrying
no rank information at all. **Kill switch 3 fired on the atom-matched
generalist**, which scored **24.51%** under oracle routing — a single 3,072-atom
dictionary with no partition and no router beat **every** mixture arm — while
the MAC-matched generalist at 512 atoms scored **18.12%**, so the atom budget is
doing real work and the comparison is not vacuous.

**The registered mechanism inverted.** The prediction was that the margin would
concentrate in quickdraw and sketch, the two low-rank domains. Under oracle
routing the margin on those two averaged **−4.84 pp** and on the other four
**+1.10 pp**. Quickdraw, lowest in effective rank (RankMe **10.6**) and
therefore handed only **~104 of 3,072 atoms**, is **29.5%** of the training
corpus and the *most* accurate domain under uniform allocation (**37.07%**
oracle); starving it cost **−7.93 pp** there. The contribution M104 actually
makes is the negative one: **effective rank is a property of the input
distribution, while the capacity a domain needs is a property of its label
structure**, and an allocation rule that reads only the first is blind to the
second. §2.9.7 probe 3's rank spread reproduced (10.6 to 80.3); the error was
believing that spread licensed an allocation. Restriction 5 is honoured: the
32×32 resolution may itself depress quickdraw's measured rank on line art, which
is disclosed and unmeasured.

**M105 and M106 are closed, not pending.** Both were registered conditional on
M104 surviving its kill switches. It did not. Building a router for a partition
that loses to its own absence, or additive growth over experts that lose to one
generalist, would extend a refuted premise. **Nothing here refutes sparse
dictionaries as a representation** — the best arm in M104, the atom-matched
generalist, *is* a sparse model. What is refuted is **rank-guided expert
sizing**, and at this scale **domain partitioning as a way to spend a fixed
atom budget**. Whether a sparse dictionary can match a dense network at matched
inference cost is untouched by M104 and is measured by M107, which was
deliberately registered independent of this gate.

**M107 (plan §7.14), registered while M104 was running and before any M104
accuracy existed.** M107 is the measurement §7.13 says the program has never
made: **two ladders on one pair of axes**, accuracy against inference MACs per
image, with **frozen DINOv2 features** on one side and **frozen sparse
dictionary codes** on the other, run through the **same head, the same penalty
grid, the same 138,000 train rows and the same 34,500 test rows**. The only
thing that differs between arms is what produced the features. M107 is
**unconditional** — it does not depend on M104's outcome, which is the point of
registering it while M104 is still running. Its **registered prediction is that
the dense ladder dominates the sparse ladder at every overlapping MAC budget**,
which is against this program's own thesis and is registered anyway. Three
asymmetries are registered as binding on every M107 sentence: DINOv2 is
pre-trained on **LVD-142M** and the sparse dictionary on this corpus alone; the
dense arms read **original-resolution** images and the sparse arms the **32×32**
downsample the rest of the program uses, with arm **(d5)** supplying the
information-matched control that separates pixels from architecture; and the
sparse mixture is scored under **oracle** routing. Prohibition 27 stays in force
until M107 has actually been run, and is then lifted only for the specific
comparison M107 measures. **[resolved in place, §5.10: M107 has since run to
completion. Prohibition 27 is now discharged for that one comparison and for
nothing else; the M107 result entry below records exactly what it licenses. The
registered prediction quoted three sentences above was refuted.]**

**M107 execution-time amendments, recorded before the run and before any M107
figure existed.** Building the M107 instrument forced four decisions §7.14 did
not answer, all registered in §7.14 in place. The resolution sweep gains **28
and 56**, because the analytic MACs — computed from the ONNX graph, before
anything ran — showed the registered sweep would have left the two ladders
overlapping at **one point**, and one point is an anecdote; the addition puts
more dense points inside the sparse ladder's own MAC range and therefore runs
**against the sparse side**. The §5.3 floor **voids** an arm instead of being
reported beside it, and the run **aborts** if the arm the head constant is
chosen on is itself void — §7.10 amendment 5's lesson applied in advance, and
exercised by a smoke configuration deliberately sized below the floor. The
instrument **proves bitwise** that a 32×32 row index addresses the same picture
in the parquet as the dense arms read, rather than asserting it, and that check
was itself negative-controlled by flipping one byte. The sparse side runs
**one** dictionary seed rather than M104's three; unlike the first three this
does **not** make the milestone harder and is recorded as a limitation of every
sparse M107 figure. A **fifth** amendment came from counting the subsample's
rows per domain: `clipart` holds **11,224** of the 138,000 train rows, so a
mixture expert clears the §5.3 floor at 256 atoms and fails at 512, and the
**mixture ladder therefore runs only at 128 and 256**. **Kill switch 3 is
decidable at two budgets instead of six** and every sentence reporting it says
so. The generalist ladder is untouched, so the dense-against-sparse question is
still measured across its whole range. A **sixth** closes a contamination the
smoke run caused: run with `--config` but without `--output`, it wrote a
**2,760-row** `evidence.json` into the **sealed** M107 directory, where the
§7.14 verifier block would later have read it as the milestone. The smoke config
now declares itself inadmissible, the runner **refuses** to let such a config
write to the sealed path, and every `evidence.json` now carries
`admissible_as_evidence` and the `config_file` that produced it.

**M107 result (plan §7.14). Outcome letter: P — the registered prediction is
refuted, and the refutation favours this program's thesis.** Evidence:
`logs/results/v15/m107_dense/evidence.json`, corpus digest `63f590097008f749`,
138,000 train rows and 34,500 test rows over 345 classes, eighteen arms, none
voided, `admissible_as_evidence` true. Recomputed independently of the runner's
own gate by `experiments/tier4/report_v15_m107_gate.py`, which agrees with it on
all three switches and on the crossing budgets. The registered prediction — the
dense ladder dominating at **every** overlapping MAC budget — **failed**. **Kill
switch 1 did not fire. Kill switch 2 fired at both decidable budgets**, and under
§11.1 that is M107's headline: the sparse generalist scores **20.61%** at
**172,572,432** MACs against `d4a_small_28`'s **15.99%** (**+4.62 pp**), and
**21.52%** at **254,607,120** MACs against `d4b_small_42`'s **19.72%**
(**+1.80 pp**), both under the LVD-142M asymmetry. **This claim is admissible
only with its five registered bounds**, all recorded in §7.14: **four of the six**
sparse budgets are **void** rather than won, because no dense arm exists at or
below 18.8 M to 90.5 M MACs; the crossings sit at accuracies **nobody would
deploy**, which kill switch 2's own text says is not an efficiency result;
`d4c_small_56` passes the entire sparse ladder at **24.50%** for **1.44×** the
sparse ceiling's MACs, so the window closes inside half an order of magnitude;
and the window exists partly because `d4a` and `d4b` feed DINOv2 only **4 and 9
patch tokens**. Prohibition 27 is discharged **for this comparison alone** per
§7.14 restriction 6 and remains in force everywhere else.

**M107's fifth bound was found after the run and is the one that most nearly
undoes the headline.** The gate compares a sparse point against the best dense
point **at or below** its MACs, and the dense ladder steps by roughly 2×, so both
crossings pit a sparse arm against a **cheaper** dense arm — `s_generalist_2048`
outspends `d4a_small_28` by **1.60×** and `s_generalist_3072` outspends
`d4b_small_42` by **1.18×**. Interpolating the dense curve to the sparse arm's own
budget, which is not a measured arm and is recorded as arithmetic rather than
evidence, the margins fall from **+4.62 pp** to roughly **+2.4 pp** linear or
**+2.1 pp** log-MACs at 2,048 atoms, and from **+1.80 pp** to roughly **+0.6 pp**
linear or **+0.3 pp** log-MACs at 3,072. **The crossing survives every one of
those readings**, so kill switch 2 stands, but the upper crossing is thin and the
correct successor experiment places a **measured** dense arm inside the window
rather than interpolating across it.

**M107 kill switch 3 did not fire, and it reconciles M104 rather than
contradicting it.** At matched inference MACs and under **oracle** routing the
six-expert mixture beats the single generalist at both budgets amendment 5
leaves decidable — **16.59%** against **11.17%** and **18.71%** against
**14.00%** — while M104's kill switch 3 fired on the atom-matched comparison,
where the generalist scored **24.51%** under oracle routing against the uniform
mixture's **24.22%** under oracle routing. Both readings are correct because the
budgets differ: **partitioning buys nothing per parameter and a great deal per
inference MAC**, since oracle routing makes five of six experts free at inference.
Every figure in this entry is an **oracle** figure, and none of them survives
without the router §7.11 and §7.12 have not yet built.

**M107 measured the resolution asymmetry it had previously only disclosed.**
Reported as a pair per §7.14 restriction 7: `d1_small_224` on original-resolution
pixels scores **53.75%** and `d5_small_224_from_32` on the same 32×32 data the
sparse arms see scores **38.86%**, so the asymmetry is worth **14.89 pp** to the
dense side, both under the LVD-142M asymmetry. At information parity the dense
side is still far ahead of the sparse ceiling's **21.52%**, but spends **24.1×**
the MACs to get there. **No information-matched dense arm exists inside the
crossing window**, so kill switch 2 cannot be re-decided at information parity;
that is a disclosed gap in M107's design, not an argument.

**M107's sparse ladder was truncated by the corpus, not by the method — a
registered question, not a claim.** Rows per fitted dimension falls from
**269.53** at 128 atoms to **11.23** at 3,072 against §5.3's floor of **10**, and
138,000 train rows admit at most **3,450** atoms; the ladder stopped one rung
short of the floor **while still improving**, its last step being **+0.91 pp**.
Every dense arm sits between **67.38** and **179.69** rows per fitted dimension.
The two families do not pay for capacity in the same currency — the sparse side
buys it with fitted dimensions, which the corpus rations, and the dense side with
depth and resolution, which it does not. **Where the sparse curve goes above
3,072 atoms is unmeasured**, and no sentence in this program may assume it
continues, flattens or crosses again.

**M104 execution-time amendments, recorded before the sealed run started.**
Building the M104 runner forced four design questions §7.10 did not answer, and
all four are registered in §7.10 **in place, before any M104 figure existed**:
the MAC match is the **row-weighted** atom sum `Σ_e f_e·A_e` and not the plain
sum, which at 512 atoms leaves arm (b) spending **3,455** atoms against arm (a)'s
**3,072** for identical inference cost; **both** generalists are run, because
design item 1(c)'s "same total atoms" and design item 2's MAC match differ by a
factor of **six** for a generalist; **arm (e)** and **kill switch 4** are added
because a MAC match rewards moving capacity off high-traffic domains and
quickdraw holds **29.46%** of train rows with the lowest measured rank, a
confound arm (d) does not control; and the §5.3 floor is read per fitted
dimension with the per-class reading **reported beside it** for every expert of
every arm. §7.10 restriction 7 records the head change from M103's multinomial
logistic to a multi-output ridge and why. **Each of the four makes M104 harder
for arm (b) to pass, not easier**, and the §5.3 cap binds against the nulls
rather than against arm (b) — the reasoning is spelled out in §7.10 so a reader
can check the direction of every one of them rather than take it on trust.

A **fifth** amendment was forced by running the instrument once on a smoke
corpus that is marked inadmissible as evidence in the config file itself. The
atom cap is computed from an expert's full row count, but the reported model was
fitted on the 90% left after the validation split, so a **capped** expert could
still fall below the floor — and two of arm (b)'s six experts were voided that
way while every cap was respected. §7.10 execution-time amendment 5 fixes it:
the selection model is fitted on the first 90% and scored on the held-out rows,
and the reported model is refitted on **every row the expert owns**, at no extra
encode pass, so `n_e / (4·A_e) ≥ 10` is now enforced **exactly** by the cap. The
defect is recorded rather than quietly repaired because the guard it broke is
the one §5.3 relies on, and a reader is entitled to know that it was broken for
as long as one smoke run.

A **sixth** amendment was registered *during* the sealed run, on noticing the
contamination rather than on reading the result. The `seconds` field in M104's
evidence **is not an operand and may never be quoted as one**: the M107 pixel
pre-materialisation ran alongside it for roughly two hours and two M107 smoke
runs ran during it, so those figures measure the scheduler, not the allocation
rule. M104 loses nothing by it, because its operand is **test accuracy** and its
compute ledger is **analytic** — `training_macs` and `rank_measurement_macs` are
counted rather than timed. Had the operand been wall-clock, the correct response
would have been to stop and rerun M104 alone rather than to disclose and
continue. A **seventh** records the near-miss that followed: M104 occupies only
**7.1 of 16** cores, and filling the rest with M107 was rejected because both
sealed configs pin their thread counts at **16** in a **`numerics`** block, and
editing sealed numerics to buy wall-clock is not a trade this program makes.

**Provenance of the 3,455.** Recorded once the sealed run had started and before
any accuracy existed. The **3,455** above is computed from §2.9.7 probe 3's
published ratios, which are an **anchor** under R7 and never an operand. The
sealed run does not consume them; it re-measures effective rank inside itself.
The figure therefore illustrates the size of the parameter excess a MAC match
implies and is **not a prediction of the run**, and no M104 result is read
against it. The number M104 reports is the run's own row-weighted total, taken
from its evidence file.

