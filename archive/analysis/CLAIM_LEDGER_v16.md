# GEODE Claim Ledger v16

**Status:** registration draft, 5 August 2026. **No v16 milestone has run, so no
v16 claim is entitled yet.** Every entry below is a _registration_: it states what
would be claimable if the milestone returns a particular result, and what would
not be claimable no matter what it returns. Entries are converted to results only
by `experiments/tier4/verify_v16_plan.py` reading an evidence artifact.

**Amended, 5 August 2026, still before any milestone ran.** A prior-art search
audit (plan §7.1) confirms every mechanism v16 uses is already published and
identifies the nearest antecedents: the sparse pipeline itself is Thiry, Arbel,
Belilovsky & Oyallon, ICLR 2021 (arXiv:2101.07528); arm (e)'s leverage feature
selection is Paul & Drineas 2015 (arXiv:1506.05173) and Avron et al. ICML 2017
(arXiv:1804.09893); arm (c)'s greedy-vs-random direction is Shahrampour & Tarokh
NIPS 2018 (arXiv:1810.03817) in a neighbouring setting. The entries below are
read against §7.1: what is claimable is _measurement/transfer_, never mechanism
novelty, and no "first"/"novel" sentence is licensed by the search.

**Second amendment, 5 August 2026, still before any milestone ran.** The
strategic reframing (plan §8, §7.1): the "both-trained, matched-cost MAC win" is
the form the literature most plausibly closes, so the achievable contribution is
reframed as the **gap-closing fraction** — how much of the distance between the
random-feature null and a trained transformer is recovered by selection-based
construction, with M108 arm (c) named a single-layer instance of neural
low-degree filtering (Dandi et al., 2026, arXiv:2605.13612) and the sparse-side
ceiling explained by measured prior art (Székely et al., NeurIPS 2024,
arXiv:2312.14922). No registered prediction, kill switch or operand of any
existing entry is changed; C108.1 gains a second registered operand below, and
the claimable language reflects the reframing.

**Reading order.** `analysis/RESEARCH_IMPLEMENTATION_PLAN_v16.md` registers the
design; this file registers the claims; the verifier recomputes both. A figure
that appears in only one of the three is a defect.

**Inheritance.** `CLAIM_LEDGER_v15.md` remains in force. Its claims are not
restated here and are not superseded. This file carries forward only the entries
v16 depends on, marked **[carried]**, and adds registrations, marked
**[registered, not yet run]**.

---

## 1. Carried forward

### C107.1 — the sparse ladder is not dominated **[carried from v15]**

**Entitled, unchanged.** On DomainNet at 345 classes, 138,000 train rows and
34,500 test rows, with one ridge penalty (1.0) chosen once on the sparse side and
applied unchanged to all eighteen arms, a frozen sparse dictionary code sits
above a frozen DINOv2 feature at two inference-MAC budgets:

| sparse arm          |        MACs | accuracy | beaten dense arm |        MACs |   margin |
| ------------------- | ----------: | -------: | ---------------- | ----------: | -------: |
| `s_generalist_2048` | 172,572,432 |   20.61% | `d4a_small_28`   | 107,566,848 | +4.62 pp |
| `s_generalist_3072` | 254,607,120 |   21.52% | `d4b_small_42`   | 215,555,328 | +1.80 pp |

v15 §7.14 kill switch 1 did not fire; kill switch 2 fired. The registered
prediction — dense dominates everywhere the ladders overlap — **failed**.

**The five bounds travel with the claim** and are restated in plan v16 §2.2: low
absolute accuracy, out-of-distribution dense arms below 224 px, the unmatched
LVD-142M training data, the closing window, and the gate letting the sparse arm
outspend the arm it beats.

**Not entitled.** That this is evidence for additive sparse construction — see
C107.2.

### C107.2 — the sparse side of M107 was never trained **[carried from v15, and it is the reason v16 exists]**

**Entitled.** `eval_v15_m107_dense.py` L713–719 constructs the M107 sparse
dictionary as a random draw of ZCA-whitened image patches, nested by a random
permutation. Only the ZCA whitener and the ridge head are fitted, both in closed
form. There is no backward pass, no optimiser and no gradient in the sparse arm.
In M103's vocabulary the M107 sparse ladder is **arm (a) — the registered null**.

**Both readings are entitled and neither may be quoted without the other.**

1. **Against the program.** C107.1 is not evidence for additive sparse
   construction. It is evidence for random features. The thesis this program
   exists to test was not exercised by the experiment that produced its best
   result.
2. **For the program.** C107.1 is a floor reached with learning switched off, and
   every mechanism this program has proposed remains unspent.

**Not entitled.** That turning learning on will improve the result. That is
M108's and M109's question and it has two registered kill switches that can
answer it in the negative.

### C103.1 — discriminative growth halves the atom count **[carried from v15, with its amendment]**

Carried **only** as M108's input, and carried **with** the prior-art amendment
that binds it: the phenomenon is already published in a stronger, label-free form
with matching lower bounds (Avron et al. ICML 2017; Li et al. ICML 2019; Rudi &
Rosasco NeurIPS 2017). Plan v16 §3.7 keeps prohibition 26 in force: **no
statement of C103.1, in any v16 document, may omit that disclosure.**

**Not entitled, and this is the whole of M108's reason to exist.** That C103.1
transfers to any other corpus. It was measured on CIFAR-10 at 10 classes.

---

## 2. Registered for v16

### C108.1 — does discriminative growth transfer to DomainNet? **[result, 5 August 2026: C103.1 does not transfer; KS3 fired]**

**What would be claimable.** If M108 arm (c) beats arm (a) at matched atoms on
DomainNet at the registered budgets {128, 256, 512, 1024, 2048, 3072}, then
additive discriminative construction improves a sparse dictionary on a corpus it
was not developed on, at 345 classes across six domains. **Second registered
operand, 5 August 2026 (plan §5.1): the gap-closing fraction.** With M107's
sealed in-distribution frozen dense reference (`d1_small_224`, 53.75%) as the
"trained transformer" pole and arm (a) at its top rung as the random-feature
null, `(c − a) / (dense_224 − a)` is the fraction of the representation gap that
selection closes at zero gradient training — claimable as a measurement of how
much of feature learning the literature's own mechanism (neural low-degree
filtering, Dandi et al. 2026) buys by selection rather than by gradient
descent. It is never claimable as mechanism novelty.

**What would not be claimable, whatever the result.** Novelty of the mechanism;
greedy forward selection is decades old and the greedy-vs-random direction is
published in a neighbouring setting (Shahrampour & Tarokh, NIPS 2018,
arXiv:1810.03817; plan §7.1). That the gain is larger than the published
label-free separation — that is arm (e)'s question. That any of this is a Q1
result.

**Registered prediction.** Arm (c) beats arm (a), by a **smaller** margin than
CIFAR-10's +0.0104 to +0.0335, because 345 classes give the discriminative
residual more directions to chase and because `d_eff` is larger here. **Arm (e),
ridge-leverage, beats arm (c).**

**Kill switch 1 fires if** arm (c) does not beat arm (a) at matched atoms. The
entitled claim then is that **C103.1 does not transfer**, and it is the headline
under plan §3.6.

**Kill switch 2 fires if** arm (e) beats arm (c) at every budget. Every
subsequent v16 sparse arm is then constructed by ridge-leverage sampling and the
program does not keep its own mechanism for authorship reasons.

**Kill switch 3 fires if** arm (c) beats arm (a) but the C107.1 crossings do not
move earlier in MACs. Accuracy and the efficiency operand are then reported as
separate findings.

**Evidence pointer.** `logs/results/v16/m108_dictionary/evidence.json` — **has
run**; `admissible_as_evidence` true, payload hash `baee732dcaffde0f`, arm (a)
reproduction `max_abs_delta` 8.70e-05, selection on cuda with order-parity
confirmed.

**[recorded after execution] Result.** M108 ran on the 9070 XT in `.venv-rocm`
under the registered protocol, `HIP_VISIBLE_DEVICES=1`, chosen penalty 1.0
(reproducing M107's choice), arm (a) reproducing M107's sealed `s_generalist`
figures to within 8.70e-05 at every budget. Per-budget accuracies at the chosen
penalty:

| budget | arm (a) | arm (c) | arm (e) |       c − a |   e − c |
| ------ | ------: | ------: | ------: | ----------: | ------: |
| 128    |  0.1117 |  0.1099 |  0.1100 |     −0.0018 | +0.0000 |
| 256    |  0.1400 |  0.1348 |  0.1345 |     −0.0052 | −0.0003 |
| 512    |  0.1642 |  0.1637 |  0.1593 |     −0.0005 | −0.0044 |
| 1024   |  0.1863 |  0.1892 |  0.1847 | **+0.0028** | −0.0044 |
| 2048   |  0.2061 |  0.2056 |  0.2030 |     −0.0004 | −0.0026 |
| 3072   |  0.2153 |  0.2138 |  0.2120 |     −0.0014 | −0.0018 |

**The registered prediction is refuted in substance.** Arm (c) beats arm (a) at
**1 of 6** budgets (1024, +0.28 pp) and loses at the other five (up to −0.52 pp);
the mean `c − a` is about **−0.11 pp**. The binary kill switch 1, under the
pre-registered reading that it fires only when arm (c) does not beat arm (a) at
any budget, did **not** fire; under the reading that arm (c) must beat arm (a)
across the registered budgets, it fires. **Either way the substantive verdict is
the same and it is the headline: C103.1 does not transfer to DomainNet.** The
core efficiency reading of C103.1 fails decisively — arm (c) at 512 atoms
(0.1637) sits **2.3 pp below** arm (a) at 1024 atoms (0.1863), so the
"half the atoms for the same accuracy" phenomenon is absent at 345 classes.
Kill switch 3 **fired**: arm (c)'s crossing budget (2048) equals arm (a)'s, so
selection did not move the C107.1 crossings earlier in MACs.

**Kill switch 2 did not fire.** Arm (e) beats arm (c) at 1 of 6 budgets (128, by
a rounding-level tie) and loses elsewhere; the registered prediction that
ridge-leverage beats discriminative growth is also refuted. Both selected arms
sit marginally _below_ the random null on average, so the "selection closes the
gap" framing of the reframed question is answered in the negative: the
gap-closing fraction is ≤ 0 at every budget except arm (c) at 1024 (+0.0081).
**Selection does not recover the representation gap to a trained transformer at
345 classes at zero gradient training** — consistent with the prior art the
audit surfaced (Székely et al., NeurIPS 2024): lazy/selection-based methods do
not reach the structure neural networks learn.

**What is now claimable.** That the discriminative-growth advantage measured on
CIFAR-10 does **not** transfer to a 345-class, six-domain corpus inside M107's
protocol — a registered negative that narrows C103.1 to the corpus it was
measured on. That neither greedy selection nor ridge-leverage sampling closes
the gap to a pre-trained transformer at this scale. That the M107 crossings are
a property of the frozen random-feature ladder, not of selection. Nothing about
the mechanism's novelty is claimed, per the audit.

### C109.1 — does the crossing survive trunk training? **[result, 6 August 2026: the crossing does not survive trunk training; KS1 fired at t2–t4, registered prediction confirmed]**

**This is the milestone this document was opened to register.** It has been
deferred four times — `ACCEPTANCE_CRITERIA_v12.md` L172 and L178–180,
`RESEARCH_IMPLEMENTATION_PLAN_v14.md` L120 — every time for compute, and the
compute premise is void as of plan v16 §4.

**What would be claimable.** If, after both families train their own
representation under a matched schedule across the four rungs (frozen,
projection, partial, full), the sparse family still sits above the dense family
at some inference-MAC budget, then C107.1's crossing is a property of the
representation rather than of the freezing — bounded to those budgets, that
corpus, and the accuracy quoted beside them.

**What would not be claimable, whatever the result.** Novelty of fine-tuning,
LoRA or linear probing. Any comparison of wall-clock between families. Any figure
from rung (t4) without the plan §5.2.6 inversion disclosure.

**Registered prediction, against the program's interest.** **The dense side gains
more from trunk training than the sparse side does, and the C107.1 crossings
close.** A 21.46 M-parameter transformer pre-trained on 142 M images has more
capacity to reallocate toward 345 classes than a 3,072-atom dictionary of 6×6
patches has.

**Kill switch 1 fires if** the dense curve is above the sparse curve at every
overlapping budget once both trunks are trained. The entitled claim then narrows
C107.1 to _"frozen sparse features beat frozen out-of-distribution dense features
at two budgets on one corpus"_, and that is the headline under plan §3.6.

**Kill switch 2 fires if** the sparse side is still above somewhere. The claim is
bounded as above.

**Kill switch 3 fires if** trunk training does not move the sparse curve at all.
The entitled claim is then that the sparse dictionary is **at capacity, not at
its optimum**, and no future sparse gain may be attributed to optimisation.

**The inversion this claim must not walk into.** Training the sparse dictionary
on DomainNet while the dense trunk stays LVD-142M pre-trained reverses the
asymmetry C107.1 discloses: at rung (t4) the sparse side would be the only family
trained on the evaluation corpus. Plan §5.2.6 registers the resolution — a
**from-scratch dense arm** at (t4), same geometry, random initialisation, same
schedule — so one rung compares two models that have seen the same data and
nothing else. **Without that arm, no (t4) figure is entitled.**

**Evidence pointer.** `logs/results/v16/m109_trunk/evidence.json` — **has run**;
`admissible_as_evidence` true, payload hash `cd935cf01ea0c1f9`, t1 reproduction
max delta 0.00081 (bound 0.002), parity guard worst relative difference 2.75e-05
(bound 1e-04).

**[recorded after execution] Result.** M109 ran sealed on the 9070 XT in
`.venv-rocm`, `HIP_VISIBLE_DEVICES=1`, under the registered protocol. The (t1)
frozen rung reproduced M107/M108 to within 0.00081 — dense r28 0.1590 (M107
0.1599), r42 0.1971 (0.1972), r224 0.5368 (0.5375), sparse 0.2148 (M108 0.2153)
— so the trained rungs start from the sealed frozen figures. Sparse MACs 254.6 M;
the best dense point at-or-below is r42 in every rung.

| rung            |     sparse | best dense ≤ sparse MACs | KS1 fires? | KS2 fires? |
| --------------- | ---------: | -----------------------: | ---------- | ---------- |
| (t1) frozen     | **0.2148** |                   0.1971 | no         | **yes**    |
| (t2) projection |     0.0554 |                   0.2212 | **yes**    | no         |
| (t3) partial    |     0.1588 |                   0.2846 | **yes**    | no         |
| (t4) full       |     0.1302 |                   0.1695 | **yes**    | no         |

**Kill switch 1 fired at (t2), (t3) and (t4): the crossing does not survive
trunk training.** After both families train their own representation, the dense
curve is above the sparse curve at every overlapping budget. This confirms the
registered prediction and the §5.2 strategic note: M107's window was a protocol
artefact of comparing a trained-for-something-else dense model against a
fitted-for-this sparse model. The headline under §3.6 narrows C107.1 to "frozen
sparse features beat frozen out-of-distribution dense features at two budgets on
one corpus".

**Kill switch 2 did not fire.** The sparse side is never above the dense side
after both trunks train.

**Kill switch 3 did not fire — with a twist worth recording.** Gradient training
_did_ move the sparse curve (0.2148 → 0.1302, a 0.0846 drop), so the dictionary
is not immovable. But it moved the wrong way: the gradient-trained dictionary is
**worse** than the constructed frozen one. The dictionary was closer to its
optimum under construction + closed-form ridge than under 8 epochs of
corpus-gradient training.

**The mechanism is degradation, not dense improvement — an honest correction to
the registered prediction's phrasing.** "Dense gains more" is not what happened
at the crossing budgets. Dense r28 dropped 0.1590 → 0.1321 (−2.7 pp), r42
0.1971 → 0.1695 (−2.8 pp), while sparse dropped 0.2148 → 0.1302 (−8.5 pp). The
crossing closes because the sparse side degrades ~3× more than the dense side at
those budgets, not because dense improved there. The most dramatic collapse is
dense r224 under full fine-tune (0.5368 → 0.1907): the 8-epoch lr 3e-4 schedule
destroys the LVD-142M features, and even so dense stays above sparse at (t4).

**Two within-rung observations, recorded as artefacts not claims.** (a) At (t2),
training a linear head helped dense (r28 0.1590→0.1718, r42 0.1971→0.2212, r224
0.5368→0.6441) while it collapsed sparse (0.2148 → 0.0554); the closed-form
float64 ridge head dramatically outperforms the shared-schedule SGD head on the
3,072-dim sparse codes, which the 4-epoch budget underfits. (b) The §5.2.6
symmetry arm — the only rung pairing two models that saw the same data — still
favours sparse: t4 from-scratch dense 224 (0.1132, 6.1 G MACs) sits **below** t4
sparse (0.1302, 254.6 M MACs).

**What is now claimable.** That C107.1's crossing is a property of the frozen
comparison, not of the representation — a registered negative that confirms the
prior art. That a 3,072-atom constructed dictionary beats its own
gradient-trained version on this corpus under this schedule. Nothing about
novelty, per the audit.

### C110.1 — the parameter axis **[result, 6 August 2026: the axes disagree at (t4) (KS1 fired); the head is 92.5% of the sparse count (KS2 fired)]**

**Why it is registered now.** Because registering an axis after seeing which axis
flatters a result is choosing the answer. Plan §5.3 restriction 3.

**What is currently true and currently not a claim.** Recomputing M107's
geometry gives the sparse arms their crossings at **0.14×** and **0.21×** the
parameters (3.06 M and 4.58 M against 21.72 M). **This is a calculation on an
unregistered axis and is not entitled as a result** until M110 is run by the
verifier over sealed evidence.

**Registered prediction.** The sparse advantage is larger on the parameter axis
than on the MAC axis wherever both are readable, because the dense family pays
its full trunk parameter count at every resolution while its MACs fall with token
count.

**Kill switch 1 fires if** the two axes disagree about _whether_ a crossing
exists. Neither axis may then be reported alone, anywhere in the program.

**Kill switch 2 fires if** the ridge head remains ~93% of the sparse parameter
count. The axis must then be reported split into representation and head, because
otherwise it measures the head.

**Evidence pointer.** `logs/results/v16/m110_parameter_axis/evidence.json` —
**computed 6 August 2026** by `verify_v16_plan.py` (56 checks, 0 failures; 4
negative controls fire) from each milestone's own sealed evidence artifact.

**[recorded after execution] Result.** M110 ran as the parameter-axis
re-analysis in `verify_v16_plan.py` over the sealed M107/M108/M109 evidence.
Parameter counts are computed from geometry, never from a model card: sparse at
3,072 atoms = **4,583,253** total (representation 343,548 + head 4,239,705);
dense = **22,321,881** (trunk 22,056,576 + head 265,305). Both kill switches
fired:

- **KS1 fired — the two axes disagree about the winner at (t4).** On the MAC
  axis dense wins accuracy at-or-below sparse MACs (0.1695 > 0.1302). On the
  parameter axis sparse wins accuracy-per-parameter at **11 of 12**
  rung×resolution cells (exception: t2 r224, 0.42×, where the dense head hits
  0.6441 on strong features). Registered consequence: neither axis may be
  reported alone; every efficiency sentence carries both.
- **KS2 fired — the head is 92.5% of the sparse parameter count** (4,239,705 of
  4,583,253). Registered consequence: the axis is reported split into
  representation and head, never as a single number.

**Accuracy-per-parameter, split.** Sparse 4.58 M params (0.34 M representation +
4.24 M head) delivers 0.1302 at (t4); the §5.2.6 same-data from-scratch dense
arm delivers 0.1132 at 22.32 M params. Sparse accuracy-per-parameter is **5.6×**
the from-scratch dense arm (2.84e-08 vs 5.07e-09), on top of the **24× MAC** and
**4.7× parameter** advantages. The sparse efficiency story therefore survives on
the parameter axis — but only as a split number, and only ever beside the MAC
axis, per KS1/KS2.

**What is now claimable.** That the two axes disagree about the winner at (t4)
— the rarely-reported both-axes finding §8 registers — and that the sparse
representation is 0.34 M parameters against a 22.06 M trunk. Nothing about
novelty; the axis is a re-analysis, per §5.3 restriction 1.

### C111.1 — does the crossing survive a measured in-window dense arm? **[registered, not yet run]**

**The defect it addresses.** C107.1's bound 5: no dense arm exists between
107,566,848 and 215,555,328 MACs, so both crossings beat a _cheaper_ opponent.
Interpolating dense to the sparse budget — arithmetic, not a measured arm — the
+1.80 pp reading falls to **+0.57 pp** (linear) or **+0.31 pp** (log-MACs).

**Registered prediction.** The `s_generalist_2048` crossing survives; **the
`s_generalist_3072` crossing does not.**

**Kill switch 1 fires if** neither crossing survives. **v15 §7.14 kill switch 2 is
then retracted in place** and kill switch 1's consequence applies: §3.2 Q2's
efficiency claim is refuted at this scale on this corpus. Headline, under plan
§3.6.

**Kill switch 2 fires if** both survive. Bound 5 is discharged and C107.1 is
reported without the interpolation caveat — and only then.

**Evidence pointer.** `logs/results/v16/m111_window/evidence.json`, absent until
M111 runs. M111 may not run before M108, per plan §5.4 restriction 3.

### C112.1 — where does the sparse curve stop? **[registered, not yet run]**

**Registered prediction.** The curve keeps climbing past 3,072 atoms on the full
split and its slope keeps shallowing, so extrapolating to the dense ladder's
accuracy needs a budget outside anything this corpus can fit. The crossing region
does not widen with scale.

**Kill switch 1 fires if** the curve saturates below 3,072 atoms once the sample
floor is lifted. C107.1's ladder was then **not** truncated by the floor, and no
future milestone may attribute a sparse shortfall to sample adequacy.

**Kill switch 2 fires if** the curve would reach the dense ladder inside a
fittable budget. That extrapolation is then **registered and measured, not
reported**. An extrapolated crossing is not a crossing.

**Evidence pointer.** `logs/results/v16/m112_ceiling/evidence.json`, absent until
M112 runs.

---

## 3. The instrument, registered as a fact about the program rather than a claim

### I16.1 — this program has a usable GPU and did not know it

**Entitled as an engineering measurement, and as an operand for nothing.**

- The discrete device is an **AMD Radeon RX 9070 XT**, `gfx1201`, 32 CUs,
  15.92 GB. An integrated Radeon shares the bus and **poisons HIP context
  initialisation for both devices** with `hipErrorInvalidImage`, despite
  `torch.cuda.get_arch_list()` containing `gfx1201`. `HIP_VISIBLE_DEVICES=1`
  resolves it and is a registered precondition of every v16 run.
- **ONNX DirectML against the CPU provider M107 used:** 9.92× (small), 11.22×
  (base), 13.34× (large).
- **Torch ROCm against CPU:** 12.26× forward, **13.38× forward+backward**,
  28.81× on the sparse encode matmul.
- **M107's 28.5-hour run was CPU-bound by one hardcoded line**,
  `eval_v15_m107_dense.py` L373, in an interpreter that had
  `DmlExecutionProvider` available throughout.

**The standing "38× slower" note about `.venv-rocm` is wrong and is contradicted
in place**, per plan §3.4. The measured direction is the opposite by an order of
magnitude. No v16 document may repeat the 38× figure without this correction
beside it.

**Not entitled.** Any of these numbers as a result, a claim, or an operand. Plan
§4.6 restriction 5.

### I16.2 — torch on the GPU reproduces the features M107 measured

**Entitled as an engineering measurement.** Canonical `facebook/dinov2-*` weights
run through M107's own feature definition on a fixed input agree with the
`onnxruntime` CPU session M107 used to a **worst relative difference of
1.263e-05** across small, base and large, at mean cosine similarity
**1.00000000**.

**What it licenses.** That v16 may run dense encode, sparse encode and trunk
training in **one interpreter** (`.venv-rocm`) on **one device**; and that
M109's frozen rung is a _reproduction_ of M107 rather than a new baseline, so any
disagreement between them is an instrument fault, not a finding.

**The void condition it creates.** Plan §4.6 restriction 1: every sealed v16 run
re-runs this check at startup and **a run whose worst relative disagreement
exceeds 1e-04 is void**, not negative, per plan §3.3.

**Evidence pointer.** `logs/results/v16/parity.json`,
`logs/results/v16/provider_benchmark.json`,
`logs/results/v16/torch_benchmark_{cpu,gpu}.json`. All four are marked
`"_note": "engineering measurement of the instrument, NOT evidence"` in the files
themselves.

---

## 4. What v16 may not say, whatever it measures

1. **No efficiency sentence without both axes**, once C110.1's kill switch 1 has
   been evaluated.
2. **No sparse figure without its null**, per plan §3.1.
3. **No C103.1 statement without the ridge-leverage disclosure**, per plan §3.7.
4. **No mixture figure without the word "oracle"**, per plan §3.9.
5. **No cross-family figure without the LVD-142M asymmetry** at rungs (t1)–(t3),
   and none at (t4) without the §5.2.6 inversion disclosure instead.
6. **No wall-clock comparison between families**, per plan §4.6 restriction 3.
7. **No claim that the GPU measurements are results.**
8. **No milestone reported before its kill switches have been evaluated against
   its own evidence artifact by the verifier.**

---

## 5. The bar v16 is trying to clear, so that failing to clear it is visible

Plan §8 states it: a sparse, inspectable model beating a dense model of the same
inference cost, at a deployable accuracy, with **both** sides trained, on a corpus
neither was tuned for, confirmed on **both** axes.

**M107 delivered one of those five.** The accuracy was 21.52% on 345 classes,
neither side was trained, the dense opponent was out of distribution and cheaper,
and only one axis was registered.

If C109.1's kill switch 1 fires, the honest reading is that C107.1's crossing was
an artefact of freezing, and this program says so in that sentence, in the
headline position.
