# GEODE Research Implementation Plan v16

## Turning the Learning On: Trunk Training, Grown Dictionaries, and the Window M107 Left Open

**Status:** registration draft, 5 August 2026. **No v16 milestone has run.** Every
section below was written before any v16 accuracy existed. Text added after
execution will be marked `[recorded after execution]` and will never replace a
registration, per the supersede-in-place rule carried forward as §3.4.

**Amended, 5 August 2026, still before any milestone ran.** A prior-art search
audit was executed and registered (§7.1). It confirms that **every mechanism this
program uses is already published** — including the sparse pipeline itself
(Thiry et al., ICLR 2021) — and it identifies the nearest published antecedents
of M108 arm (c) and arm (e). No registration, operand, kill switch or
prediction in this document is changed by the audit; what changes is the
standing of what would be claimable, which §7.1 states.

**Second amendment, 5 August 2026, still before any milestone ran.** The
strategic reframing registered in §7.1 and §8. The literature most plausibly
closes the "both-trained, matched-cost, deployable-accuracy MAC win" reading of
the breakthrough (Ghorbani et al. 2020; Székely et al. 2024), so the achievable
breakthrough is reframed as the **gap-closing fraction** — how much of the
distance between the random-feature null and a trained transformer is recovered
by selection-based construction (M108 arm (c), which the 2026 literature names
a single-layer instance of neural low-degree filtering, Dandi et al. 2026) at a
small fraction of the training cost — plus the both-axes (parameter-axis) story
(§5.3) and the crossover map (§5.5/§6). No existing registration, operand, kill
switch or prediction is changed; §8's breakthrough definition and M108's operand
are extended in place, and the new sections are marked as registered 5 August 2026.

**This document is a successor, not a replacement.** `RESEARCH_IMPLEMENTATION_PLAN_v15.md`
and `CLAIM_LEDGER_v15.md` remain in force as the record of M102–M107. v16 opens
because the M104–M107 branch closed: M104 was refuted, M107 answered its question
and returned a result that **contradicted its own registered prediction in the
program's favour**, and the two facts together point at one experiment that has
never been run in fifteen plan versions. v16 exists to run it.

---

## 1. The one-sentence question

**M107 measured a sparse model that was never trained. If the learning is turned
on — on the dictionary, and on the trunk, on both sides of the comparison — does
the window in which sparse beats dense widen, or close?**

Everything below is bookkeeping in service of that sentence.

---

## 2. What M107 established, stated against interest

### 2.1 The result

M107 (v15 §7.14) put a frozen sparse dictionary code and a frozen DINOv2 feature
through the identical ridge-probe protocol on DomainNet — 345 classes, 138,000
train rows, 34,500 test rows, one penalty (**1.0**) chosen once on the sparse
side and applied unchanged to all eighteen arms.

**Kill switch 1 did not fire. Kill switch 2 fired.** The registered prediction —
that dense dominates at every overlapping budget — **failed**. Two sparse arms
sit above the dense ladder:

| sparse arm          |        MACs | accuracy | beats          |     at MACs |       margin |
| ------------------- | ----------: | -------: | -------------- | ----------: | -----------: |
| `s_generalist_2048` | 172,572,432 |   20.61% | `d4a_small_28` | 107,566,848 | **+4.62 pp** |
| `s_generalist_3072` | 254,607,120 |   21.52% | `d4b_small_42` | 215,555,328 | **+1.80 pp** |

### 2.2 The five bounds that already restrict it

Carried forward verbatim in force from v15 §7.14. They are restated here because
a successor plan that quotes the headline without the bounds is doing the thing
this program exists to prevent.

1. **Absolute accuracy is low.** 21.52% on 345 classes. A crossing at an accuracy
   nobody would deploy is not an efficiency result.
2. **The dense arms below 224 px are out of distribution.** DINOv2 is trained at
   224; its position embeddings are interpolated everywhere else. The resolution
   sweep is a **lower bound on dense**, never the dense curve.
3. **The training data is not matched, by three orders of magnitude.** DINOv2 is
   pre-trained on LVD-142M; the sparse dictionary is drawn from this corpus's own
   train patches and nothing else.
4. **The window closes.** The margin falls from +4.62 pp to +1.80 pp across one
   ladder step, and the sparse ladder is rationed by the §5.3 sample floor at
   3,450 atoms on this subsample.
5. **The gate let the sparse arm outspend the arm it beat.** No dense arm exists
   between 107,566,848 and 215,555,328 MACs, so both crossings beat a _cheaper_
   opponent. Interpolating dense to the sparse budget — **arithmetic, not a
   measured arm** — the +1.80 pp reading falls to +0.57 pp (linear) or +0.31 pp
   (log-MACs). The crossing survives all four readings; the upper margin is thin.

### 2.3 The finding that motivates this entire document

**The sparse side of M107 was not trained. Nothing was learned.**

`eval_v15_m107_dense.py` L713–719 draws the dictionary as a random sample of
ZCA-whitened image patches and nests the budgets by a random permutation. Only
the ZCA whitener and the ridge head are fitted, and both are closed-form. There
is no backward pass, no optimiser, no epoch, and no gradient anywhere in the
sparse arm.

In M103's own vocabulary this is **arm (a) — the registered null.**

This cuts in both directions and both must be stated in every sentence that
quotes it:

- **Against the program.** The crossing is _not_ evidence for additive sparse
  construction. It is evidence for random features. The thesis this program
  exists to test was not exercised by the experiment that produced its best
  result.
- **For the program.** The crossing is a **floor achieved with learning switched
  off**. Every mechanism this program has ever proposed is still unspent.

**v16's entire premise is that second reading.** M103 already showed, on CIFAR-10
at three seeds, that discriminative growth beats the random null in **15 of 15**
seed-budget cells and reaches the null's 1024-atom accuracy at **512 atoms**
(C103.1). M107 then ran the null — and the null was already competitive. Nobody
has yet run the two together.

### 2.4 The unregistered result that may be the stronger one

M107 registered accuracy against **inference MACs**. Computing parameters after
the fact — from the geometry the verifier already re-reads off the ONNX graph —
gives a second axis that was never registered and is not admissible until it is:

| pair                |            MACs | parameters | accuracy |
| ------------------- | --------------: | ---------: | -------: |
| `s_generalist_2048` | 172.6 M (1.60×) | **3.06 M** |   20.61% |
| `d4a_small_28`      |         107.6 M |    21.72 M |   15.99% |
| `s_generalist_3072` | 254.6 M (1.18×) | **4.58 M** |   21.52% |
| `d4b_small_42`      |         215.6 M |    21.72 M |   19.72% |

The sparse arms win their crossings at **0.14×** and **0.21×** the parameters.
Every dense arm below 224 px carries the full 21.46 M trunk regardless of
resolution, which is why the dense parameter count is flat while its MACs fall.
About 93% of the sparse parameter count is the ridge head; the dictionary itself
is 0.33 M at 3,072 atoms.

**This is currently a calculation, not a result.** M110 (§5.3) registers it so it
can become one.

---

## 3. Discipline carried forward, binding on every v16 sentence

These are not restated for ceremony. Each one has already caught a specific error
in this program, and each is carried into v16 unchanged.

**§3.1 — R5: an operand without its null is not evidence.** Every v16 arm ships
with the arm it is supposed to beat, measured in the same run.

**§3.2 — R7: external figures are anchors, never operands.** No published number
may be a pass/fail condition. Where v16 compares against prior art, it
**re-implements and re-measures** the mechanism inside the same protocol.

**§3.3 — a failing arm is void, not negative.** An arm that fails its sample
floor, crashes, or violates its own precondition produces no figure. It may not
be reported as a low score.

**§3.4 — superseded text is contradicted in place, never deleted.** v15 §5.10.

**§3.5 — the sample floor is ten rows per fitted dimension and is never waived.**
v15 §5.3. It is read per fitted dimension on the rows each arm actually fits.

**§3.6 — a fired kill switch is the headline, never a footnote.** v15 §11.1.

**§3.7 — prohibition 26 remains in force.** No statement of C103.1, in any
document, may omit the ridge-leverage prior-art disclosure (§7).

**§3.8 — prohibition 27 remains in force** except for the specific comparison
M107 measured, and is **not** extended to any v16 arm until that arm has run.

**§3.9 — the oracle-routing disclosure.** Every sentence quoting a mixture figure
states that routing is oracle. v15 §7.10 restriction 4.

**§3.10 — the LVD-142M asymmetry is disclosed with every cross-family figure.**
v15 §7.14 restriction 4, and see §5.2.6 for the way trunk training threatens to
invert it.

---

## 4. The instrument: this program now has a GPU, and that is a registration-worthy change

### 4.1 Why this is in the plan and not in a README

Every deferral of trunk training in this program's history has the same cause,
and it is recorded in writing. `ACCEPTANCE_CRITERIA_v12.md` L172: _"torch
2.13.0+cpu with no GPU backend. Full ViT-S/14 fine-tuning is [infeasible]."_
L178–180 registers a staged ladder — frozen, learned projection, LoRA/last-k,
full fine-tune — that was never climbed. `RESEARCH_IMPLEMENTATION_PLAN_v14.md`
L120 ordered _"backbone capacity third, trunk training last."_ It was last, and
last never arrived.

**The premise of that deferral is now void, and it was void for longer than
anyone noticed.** M107's 28.5-hour run was CPU-bound because
`eval_v15_m107_dense.py` L373 hardcodes `providers=["CPUExecutionProvider"]`
while the interpreter running it had `DmlExecutionProvider` available the whole
time. A one-line omission cost roughly a day per experiment for an unknown number
of experiments.

A change that turns a 33-hour training run into a 2.5-hour one is not a
convenience. It changes which milestones are registrable, so it is registered.

### 4.2 What was measured

All figures below are **engineering measurements of the instrument**. They are
not evidence, they touch no corpus, they produce no accuracy, and no v16 claim
may cite them as an operand. They are recorded because the milestone budget in §6
depends on them and a budget from unmeasured assumptions is a guess.

Hardware: **AMD Radeon RX 9070 XT**, RDNA4, `gfx1201`, 32 CUs, 15.92 GB, plus an
integrated Radeon that must be hidden (§4.4).

**ONNX inference, `DmlExecutionProvider` against the `CPUExecutionProvider` M107
actually used** (`experiments/tier4/bench_v16_providers.py`, batch 32 at 224 px,
batch 16 for large):

| model        | CPU img/s | DirectML img/s |    speedup |
| ------------ | --------: | -------------: | ---------: |
| DINOv2-small |     33.92 |         336.47 |  **9.92×** |
| DINOv2-base  |     11.34 |         127.16 | **11.22×** |
| DINOv2-large |      3.05 |          40.71 | **13.34×** |

**Torch, ROCm against CPU** (`experiments/tier4/bench_v16_torch.py`, ViT-S
geometry, batch 32, 257 tokens):

| operation                    | CPU img/s | ROCm img/s |    speedup |
| ---------------------------- | --------: | ---------: | ---------: |
| ViT-S forward only           |     46.79 |     573.61 | **12.26×** |
| **ViT-S forward + backward** | **11.70** | **156.50** | **13.38×** |
| sparse encode matmul         |    460.08 |  13,255.68 | **28.81×** |

The middle row is the one that matters. Trunk training a ViT-S over 138,000
images costs **3.28 hours per epoch on the CPU and 14.7 minutes on the GPU**.
Ten epochs: 32.8 hours becomes 2.45 hours. That is the difference between a
milestone that cannot be registered and one that can.

### 4.3 The correction to a standing instruction

This program carried a standing note that `.venv-rocm` must never be used because
it was **"38× slower"**. That note is **wrong, and it is contradicted in place
here rather than deleted**, per §3.4.

The measured direction is the opposite: ROCm is **12–29× faster** than the CPU on
every operation v16 needs. The likely origin of the error is §4.4 — a benchmark
that landed on the integrated GPU, or that fell back after the discrete card
failed to initialise. No v16 document may repeat the 38× figure without this
correction beside it.

### 4.4 The failure that hid the GPU, recorded so it is not rediscovered

`torch.cuda.device_count()` returns **2**: index 0 is the integrated Radeon
(reporting 35.87 GB of shared system memory and 1 compute unit), index 1 is the
RX 9070 XT. Any allocation on **either** device fails with:

```
torch.AcceleratorError: CUDA error: device kernel image is invalid   (hipErrorInvalidImage)
```

even though `torch.cuda.get_arch_list()` returns `['gfx1200', 'gfx1201']` and the
discrete card reports `gcnArchName = gfx1201`. The kernels are present and the
architecture matches; the integrated GPU poisons context initialisation for both.

**The fix is one environment variable**, and it is a registered precondition of
every v16 run:

```
HIP_VISIBLE_DEVICES=1
```

With the integrated GPU hidden, `device_count()` is 1, the visible device is
`gfx1201`, and every operation in §4.2 runs.

### 4.5 One venv, one device — and the parity measurement that permits it

Trunk training needs gradients. `onnxruntime` will not give them, so the trunk
arm must be **torch**. That is only sound if the torch model _starts_ from
M107's exact frozen operating point — otherwise the frozen and trained arms
differ in two things at once and neither can be read.

`experiments/tier4/bench_v16_parity.py` measures it. Canonical
`facebook/dinov2-{small,base,large}` weights were fetched to
`_cache_root()/torch/`, run through M107's own feature definition (CLS token
concatenated with the mean of the patch tokens) on a fixed input, and compared
against the `onnxruntime` CPU session M107 used:

| model        | onnx/CPU vs onnx/DirectML | onnx/CPU vs torch/ROCm | mean cosine |
| ------------ | ------------------------: | ---------------------: | ----------: |
| DINOv2-small |                 1.448e-06 |              1.520e-06 |  1.00000000 |
| DINOv2-base  |                 2.974e-06 |              2.473e-06 |  1.00000000 |
| DINOv2-large |                 9.526e-06 |          **1.263e-05** |  1.00000000 |

**Worst relative disagreement across all pairs: 1.263e-05**, cosine similarity
1.00000000 to eight decimal places. This is fp32 accumulation-order noise, not a
different model.

Three consequences, all registered:

1. **v16 runs entirely in `.venv-rocm`.** It already carries torch 2.11.0+rocm,
   torchvision, transformers 4.57.1, safetensors, numpy and pillow. Dense encode,
   sparse encode and trunk training all run in one interpreter on one device. The
   dual-venv design the dense/sparse framework split seemed to force is not
   needed.
2. **The trunk arm provably starts where M107's frozen arm sat.** The frozen rung
   of the trunk ladder (§5.2) is therefore a _reproduction_ of M107, not a new
   baseline, and any disagreement between them is an instrument fault rather than
   a finding.
3. **Parity is a startup guard with a void condition** (§4.6).

### 4.6 Registered instrument restrictions

1. **Every sealed v16 run re-runs the parity check at startup** and writes the
   measured disagreement into its evidence artifact. **A run whose worst relative
   disagreement against the M107 ONNX reference exceeds 1e-04 is void**, not
   negative, per §3.3. The bound is a factor of ~8 above the measured 1.263e-05,
   which leaves room for driver and kernel variation without leaving room for a
   different model.
2. **`HIP_VISIBLE_DEVICES=1` is a precondition.** A run that finds
   `torch.cuda.device_count() != 1`, or a visible device whose `gcnArchName` is
   not `gfx1201`, aborts before producing a figure rather than silently landing
   on the integrated GPU.
3. **Wall-clock remains not an operand**, per v15 §7.14 restriction 5, and the
   reason is now stronger rather than weaker: the families no longer even differ
   in framework, so a timing ratio would measure kernel maturity on one vendor's
   driver. Analytic MACs and parameter counts are the efficiency operands.
4. **The GPU may not change any arm's arithmetic.** MACs, parameter counts,
   sample floors and the ridge solve are computed as v15 computed them. The
   device is permitted to change how long a figure takes and nothing else.
5. **§4.2's speedups are never cited as a result.** They size the budget in §6
   and they do nothing else.

---

## 5. Milestones

Registered in dependency order. Each carries a prediction written before
measurement, at least one kill switch that can fire against the program, and its
own restrictions.

### 5.1 M108 — grow the dictionary instead of drawing it {#m108}

**Question.** M107's sparse ladder is M103's arm (a): a random draw. M103 showed
on CIFAR-10 that arm (c), additive discriminative growth, reaches arm (a)'s
1024-atom accuracy at 512 atoms. **Does that 2.0× survive transfer to DomainNet
at 345 classes, inside M107's own protocol?**

C103.1 explicitly does **not** entitle transfer. This asks for it.

**Design.** M107's rig, one variable changed. Same corpus, same 138,000/34,500
subsample and digest, same patch pipeline, same ZCA whitener, same ridge head,
same penalty grid, same selection rule, same budgets {128, 256, 512, 1024, 2048,
3072}. The dictionary construction is the only difference.

**Arms.**

- **(a) random patches** — M107's exact construction, re-run rather than quoted,
  per §3.1. It is the registered null.
- **(c) discriminative growth** — `select_discriminative()` from
  `eval_v15_m103_atoms.py` L227–280: build a one-hot centred residual, select the
  atom maximising `work.T @ residual`, least-squares-project it out, repeat.
  **This is additive greedy construction and it is the mechanism this whole
  program is about.** The §7.1 audit adds a modern name for what this is: a
  **single-layer, selection-based instance of neural low-degree filtering**
  (Dandi et al., 2026, arXiv:2605.13612) — the mechanism the current literature
  identifies as what deep feature learning _is_. M108 therefore measures, in a
  cheap form, a mechanism the current literature treats as the heart of
  representation learning. That framing is what the second operand below
  quantifies.
- **(e) ridge-leverage sampling** — Avron et al., ICML 2017. **New in v16, and
  required.** §3.7 binds every statement of C103.1 to disclose that a stronger,
  **label-free** mechanism with matching lower bounds is already published.
  Disclosing it in prose while never running it would be the weaker half of
  compliance. Arm (e) re-implements it inside this protocol so the comparison is
  measured rather than cited, per §3.2.
- **(b) k-means** is **not** carried. M103 measured it, it lost to random at
  small scale and reversed at full scale, and it tests reconstruction rather than
  discrimination. Its absence is a registered scope decision, not an oversight.

**Operand.** Test accuracy against atom budget, and against inference MACs, all
arms on one table. The efficiency operand is **atoms to reach arm (a)'s
top-rung accuracy**, exactly as C103.1 defined it.

**Second operand, registered 5 August 2026 — the gap-closing fraction.** §8's
reframing makes M108's question not only "does arm (c) beat arm (a)" but
"**how much of the gap between the random-feature null and a pre-trained
transformer does selection close, at zero gradient training**". Registered
definition: with the in-distribution frozen dense reference `d1_small_224` from
M107's sealed evidence (53.75%) as the "trained transformer" pole and arm (a)
at its top rung as the "random features" pole, the fraction recovered by arm (c)
(and arm (e)) at the same budget is `(c − a) / (dense_224 − a)`, reported at
every budget where both are readable. It is a measurement, never a novelty
claim: both mechanisms are published. Its content is that it quantifies, at 345
classes, how much of feature learning the literature's own mechanism buys when
done by selection rather than by gradient descent. The same fraction is
re-measured at every M109 rung with the trained dense side, where the theory
(Ghorbani et al. 2020; Székely et al. 2024) predicts it will shrink — which is
itself a registered prediction of M109's reading.

**Registered prediction.** Arm (c) beats arm (a) at matched atoms on DomainNet,
**but by a smaller margin than the +0.0104 to +0.0335 measured on CIFAR-10**,
because 345 classes across six domains give a discriminative residual far more
directions to chase than 10 classes do, and because the effective dimension
`d_eff` the prior art identifies as the controlling quantity is larger here.
**Arm (e) beats arm (c)**, because the published separation is stronger than the
one this program built and there is no reason to expect otherwise.

**Kill switch 1.** If arm (c) does not beat arm (a) at matched atoms on DomainNet
at the registered budgets, **C103.1 does not transfer**, and the additive-growth
thesis has failed its first test outside CIFAR-10. That is the headline under
§3.6. It may not be reported as a limitation or as future work.

**Kill switch 2.** If arm (e) beats arm (c) at every budget, then the label-free
published mechanism dominates the one this program built, and **every subsequent
v16 sparse arm is constructed by ridge-leverage sampling, not by
`select_discriminative()`**. The program does not get to keep its own mechanism
for authorship reasons.

**Kill switch 3.** If arm (c) beats arm (a) but the M107 crossings do **not**
move — that is, the dense arms are not overtaken any earlier in MACs — then
dictionary learning improves accuracy without improving the efficiency operand,
and the two must be reported as separate findings rather than as one.

**Restrictions.**

1. **Prohibition 26 applies in full**: no M108 sentence quoting arm (c) omits the
   ridge-leverage disclosure or C103.3's narrowing table.
2. **`select_discriminative()` is not novel and is not claimed as such.** It is a
   greedy forward-selection procedure; the family is decades old.
3. **The §3.5 floor is read per arm.** At 3,072 atoms on 138,000 rows the sparse
   generalist gives 11.23 rows per fitted dimension. The floor is not waived for
   any arm that grows past it; the arm is void.
4. **Arm (a) is re-measured, never quoted from M107.** If it disagrees with
   M107's figure beyond the §4.6 parity bound, M108 is void and the instrument is
   at fault.

**[recorded after execution] M108 result, 5 August 2026.** M108 ran sealed on
the 9070 XT; evidence `logs/results/v16/m108_dictionary/evidence.json`, arm (a)
reproducing M107 to 8.70e-05 at every budget, chosen penalty 1.0, selection on
the GPU with the registered order-parity check confirmed. The registered
prediction is **refuted in substance**: arm (c) beats arm (a) at 1 of 6 budgets
(1024, +0.28 pp) and loses at the other five; the mean `c − a` is about
−0.11 pp; arm (c) at 512 atoms sits 2.3 pp below arm (a) at 1024, so the
C103.1 efficiency reading fails decisively. Kill switch 1's substantive verdict
is **C103.1 does not transfer** (its binary flag is false only under the
pre-registered "any budget wins" reading; see the ledger for the fork). Kill
switch 2 did not fire — arm (e) does not beat arm (c) at every budget. **Kill
switch 3 fired**: the crossings did not move earlier in MACs (both arms cross at
2048 atoms). The gap-closing fraction (second operand) is ≤ 0 at every budget
except arm (c) at 1024 (+0.0081): **selection does not recover the
representation gap to a trained transformer at 345 classes**, consistent with
the prior art (Székely et al., NeurIPS 2024). This narrows C103.1 to the corpus
it was measured on and is the program's first registered negative outside
CIFAR-10.

### 5.2 M109 — trunk training, and it is not last {#m109}

**Question.** Both families in M107 were frozen. The dense trunk was frozen
because it was pre-trained on LVD-142M and freezing is DINOv2's own evaluation
protocol; the sparse trunk was frozen because there was no compute to train it.
**What happens to the crossing when each family is allowed to train its own
representation?**

**Why it is registered as a primary milestone.** Because it has been deferred
four times, always to the end of an ordering, always for compute, and the compute
premise is now void (§4). Registering it last again would be registering it never.
**M109 runs immediately after M108 and before M110–M112.**

**The v12 ladder, revived.** `ACCEPTANCE_CRITERIA_v12.md` L178–180 registered it
and it was never climbed. It is climbed here, on both sides, in this order:

| rung                | dense side                             | sparse side                                  |
| ------------------- | -------------------------------------- | -------------------------------------------- |
| **(t1) frozen**     | M107's arm, reproduced via §4.5 parity | M108's best arm, frozen                      |
| **(t2) projection** | learned linear map on frozen features  | learned linear map on frozen codes           |
| **(t3) partial**    | LoRA, and last-_k_ blocks unfrozen     | dictionary gradient-trained, whitener frozen |
| **(t4) full**       | full fine-tune of the trunk            | dictionary and whitener both trained         |

Each rung is measured on both families in the same run, with the same optimiser
family, the same schedule length, and the same early-stopping rule, so that a
difference between families is not a difference in training budget.

**Operand.** Test accuracy against inference MACs **and** against trainable
parameters. The second axis is registered here because at (t3) and (t4) the
number of parameters that receive a gradient is the quantity that separates the
rungs, and reporting only MACs would make (t2) and (t4) look identical.

**Registered prediction.** Written against the program's interest, as v15 §7.14's
was, and for the same reason.

**The dense side gains more from trunk training than the sparse side does, and
the M107 crossings close.** A 21.46 M-parameter transformer pre-trained on 142 M
images has far more capacity to reallocate toward 345 DomainNet classes than a
3,072-atom dictionary of 6×6 patches has. If that is right, M107's window is an
artefact of comparing a trained-for-something-else dense model against a
fitted-for-this sparse model, and it will not survive both sides being fitted for
this.

**Strategic note, registered 5 August 2026.** This prediction is no longer
written only against the program's own interest; the literature now predicts the
same direction. Ghorbani et al. (NeurIPS 2020) show neural networks beat kernel
methods when they learn a low-dimensional representation, and Székely et al.
(NeurIPS 2024) measure that random features cannot reach the higher-order
structure neural networks learn. §8 therefore reads M109 as the **decisive test
of whether M107's crossing was a protocol artefact** (out-of-distribution dense,
corpus-matched sparse), and expects the registered prediction to be confirmed.
If it is, that is a confirmation of the prior art, reported as such — and the
program's measurable content moves to the gap-closing fraction, the parameter
axis and the crossover map (§8).

**Kill switch 1.** If the dense curve is above the sparse curve at every
overlapping MAC budget once both trunks are trained, then **M107's crossing does
not survive trunk training**, and the admissible reading of M107 narrows to
"frozen sparse features beat frozen out-of-distribution dense features at two
budgets on one corpus". That is the headline under §3.6.

**Kill switch 2.** If the sparse side is _still_ above the dense side somewhere
after both trunks are trained, then the crossing is a property of the
representation rather than of the freezing, and the claim is bounded to those
budgets, that corpus and the accuracy quoted beside them.

**Kill switch 3.** If trunk training does not move the sparse curve at all — the
gradient-trained dictionary matches the constructed one — then **the sparse
dictionary is at capacity, not at its optimum**, and every future sparse gain
must come from more atoms or better construction rather than from optimisation.
This would also mean the program's own mechanism (M108 arm (c)) is doing all the
work available, which is a positive finding stated in a form that can fail.

**§5.2.6 — the symmetry trap, registered before it can be walked into.** Training
the sparse dictionary on DomainNet while the dense trunk stays LVD-142M
pre-trained **inverts the asymmetry §3.10 discloses**. M107's caveat was that
dense had three orders of magnitude more training data; at rung (t4) the sparse
side would be the only one trained on the evaluation corpus. Both directions
would then be defensible and the choice of which to report would be the choice of
the answer. **Registered resolution:** rung (t4) carries a **from-scratch dense
arm** — the same DINOv2-small geometry, randomly initialised, trained on
DomainNet under the same schedule — so that at least one rung compares two models
that have seen exactly the same data and nothing else. It is expected to be poor;
it is carried because without it (t4) is uninterpretable.

**Restrictions.**

1. **The schedule is fixed before the first run and is identical across
   families.** Optimiser family, learning-rate schedule shape, epoch count and
   early-stopping rule are registered in the config and are not tuned per arm.
   Per-family learning-rate _scale_ is permitted and must be disclosed, because a
   single scale across a pre-trained transformer and a freshly-initialised
   dictionary would handicap one of them.
2. **Every rung reports its frozen predecessor beside it.** A trained figure
   without its frozen figure is not readable, per §3.1.
3. **§3.10's LVD-142M disclosure applies to rungs (t1)–(t3) unchanged, and is
   replaced at (t4) by the §5.2.6 statement**, which discloses the inversion
   rather than the original asymmetry.
4. **No wall-clock comparison**, per §4.6 restriction 3.
5. **Trunk training is not novel and is not claimed as such.** Fine-tuning,
   LoRA (Hu et al., 2021) and linear probing are standard; only the _comparison_
   between families under a matched schedule is this program's.

**[recorded after execution] M109 result, 6 August 2026.** M109 ran sealed on
the 9070 XT; evidence `logs/results/v16/m109_trunk/evidence.json`, parity guard
worst relative difference 2.75e-05 (bound 1e-04), (t1) frozen reproduction max
delta 0.00081 (bound 0.002). **Kill switch 1 fired at (t2), (t3) and (t4): the
M107 crossing does not survive trunk training.** Once both families train their
own representation under the matched schedule, the dense curve is above the
sparse curve at every overlapping budget (sparse 254.6 M MACs vs best dense
at-or-below): (t1) sparse 0.2148 vs dense 0.1971 — crossing reproduced; (t2)
sparse 0.0554 vs dense 0.2212; (t3) sparse 0.1588 vs dense 0.2846; (t4) sparse
0.1302 vs dense 0.1695. The registered prediction and the §5.2 strategic note
are confirmed: M107's window was a protocol artefact (frozen sparse vs
pre-trained-for-elsewhere dense), exactly as the prior art (Ghorbani et al. 2020;
Székely et al. 2024) predicts, and it is reported as such. Kill switch 2 did not
fire; kill switch 3 did not fire but moved the sparse curve **down** 0.0846 —
the constructed frozen dictionary (0.2148) beats its own gradient-trained
version (0.1302). The §5.2.6 symmetry arm (same data, nothing else) still
favours sparse: t4 from-scratch dense 224 (0.1132, 6.1 G MACs) below t4 sparse
(0.1302, 254.6 M MACs). Full figures, the degradation-not-improvement correction
and the two within-rung artefacts are in the ledger C109.1.

### 5.3 M110 — register the parameter axis {#m110}

**Question.** §2.4's table shows the sparse arms winning their M107 crossings at
0.14× and 0.21× the parameters — a wider margin than the MAC axis gives. **Is
accuracy-per-parameter the operand this program should have been using?**

**Strategic note, registered 5 August 2026.** §8 identifies the parameter axis
as the most likely survivor of M109: even if trunk training closes the MAC
window (the registered prediction), the dense family pays its full 21.46
M-parameter trunk at every resolution while the sparse family's parameter count
is dominated by the ridge head both families pay. M110 is therefore not a
footnote axis; it is the axis on which the program's sparse story is expected to
survive. The §5.2.6 (t4) from-scratch dense arm must be reported on it like
every other arm, because at (t4) it is the only dense arm whose parameter count
is not inherited from LVD-142M pre-training.

**Design.** No new run. M110 is a **re-analysis of M107, M108 and M109 evidence
on a second registered axis**, computed by the same verifier that recomputes the
MAC axis, from the same geometry.

**Registered prediction.** The sparse advantage is larger on the parameter axis
than on the MAC axis at every budget where both are readable, because the dense
family pays its full trunk parameter count at every resolution while its MACs
fall with token count, and the sparse family's parameter count is dominated by
the ridge head, which both families pay.

**Kill switch 1.** If the parameter axis and the MAC axis disagree about
_whether_ a crossing exists — not merely about its size — then **neither axis may
be reported alone**, and every efficiency sentence in the program carries both.

**Kill switch 2.** If ~93% of the sparse parameter count remains the ridge head,
then the sparse _representation_ is not what the parameter axis is measuring, the
head is, and the axis must be reported split into representation and head rather
than as a single number.

**Restrictions.**

1. **M110 registers an axis; it does not license a claim by itself.** Its figures
   are readable only as re-analyses of the milestone that produced them.
2. **Parameter counts are computed from geometry by the verifier**, never quoted
   from a model card, per §3.2.
3. **The axis is registered before M108 and M109 run**, so it cannot be chosen
   after seeing which axis flatters the result. This is the only reason M110 is
   registered here rather than after M109.

**[recorded after execution] M110 result, 6 August 2026.** M110 ran as a
re-analysis in `verify_v16_plan.py` over the sealed M107/M108/M109 evidence (56
checks, 4 negative controls fire). Parameter counts computed from geometry, never
from a model card: sparse at 3,072 atoms = 4,583,253 total (representation
343,548 + head 4,239,705); dense = 22,321,881 (trunk 22,056,576 + head
265,305). **Kill switch 1 fired**: at (t4) the MAC axis winner is dense
(accuracy) while the parameter axis winner is sparse (accuracy-per-parameter) —
the two axes disagree, so neither may be reported alone and every efficiency
sentence carries both. **Kill switch 2 fired**: the head is 92.5% of the sparse
parameter count, so the axis is reported split into representation and head.
Split accuracy-per-parameter at the §5.2.6 same-data arm: sparse 0.1302 @
4.58 M params (2.84e-08) vs from-scratch dense 0.1132 @ 22.32 M params
(5.07e-09) — a 5.6× sparse advantage, alongside the 24× MAC and 4.7× parameter
advantages. This is the axis on which the sparse story survives (§8 contribution
2), reported exactly as KS1 and KS2 require. Evidence:
`logs/results/v16/m110_parameter_axis/evidence.json`.

### 5.4 M111 — put a measured dense arm inside the window {#m111}

**Question.** §2.2 bound 5: both M107 crossings beat a _cheaper_ dense arm,
because no dense arm exists between 107.6 M and 215.6 M MACs. The interpolated
readings are arithmetic, not measurements. **Does the crossing survive a dense
arm measured at the sparse arm's own budget?**

**Design.** Dense arms placed deliberately inside the window: DINOv2-small at the
resolutions whose analytic MACs bracket **172,572,432** and **254,607,120** —
computed from ONNX geometry before running anything, exactly as v15 §7.14's
execution-time amendment 1 placed 28 and 56. Same protocol, same subsample, same
head, same penalty.

**Registered prediction.** The crossing at `s_generalist_2048` survives, because
+4.62 pp against a 1.60× cheaper arm has room to absorb the interpolation. **The
crossing at `s_generalist_3072` does not survive**, because +1.80 pp against a
1.18× cheaper arm falls to +0.31 pp on the log-MACs reading and a measured arm is
unlikely to sit above that interpolation.

**Kill switch 1.** If neither crossing survives a measured in-window dense arm,
**M107's kill switch 2 is retracted in place** and kill switch 1's consequence
applies instead: §3.2 Q2's efficiency claim is refuted at this scale on this
corpus. This is the headline under §3.6.

**Kill switch 2.** If both survive, bound 5 is discharged and the M107 window is
reported without the interpolation caveat — and _only_ then.

**Restrictions.**

1. **The bracketing resolutions are computed and registered before any M111
   accuracy exists**, from ONNX geometry, and are written into the config.
2. **The in-window arms are out of distribution** and remain a lower bound on
   dense, per §2.2 bound 2. M111 tightens bound 5; it does not touch bound 2.
3. **M111 may not be run before M108.** If M108 moves the sparse curve, the
   window moves with it and M111's resolutions are recomputed against the new
   curve. Registering the arithmetic first and the arm second is what makes this
   a test rather than a search.

### 5.5 M112 — find where the sparse curve actually stops {#m112}

**Question.** M107's sparse ladder stops at 3,072 atoms because the §3.5 floor
admits at most 3,450 on a 138,000-row subsample. The curve was still climbing.
**Where does it stop when the floor is not the thing stopping it?**

**Design.** Two independent ways to buy fitted dimensions, run as separate arms
because they trade different things:

- **(f1) more rows** — the full DomainNet train split, 409,832 rows, which raises
  the ceiling to 10,245 atoms. Affordable only because of §4; on the CPU this arm
  alone would have cost days.
- **(f2) fewer classes** — the same 138,000 rows on a registered subset of
  classes, which lowers the fitted dimension per atom. This changes the task and
  is therefore **not** comparable to M107 or to (f1); it is carried only to
  separate "the curve saturates" from "the floor bound it".

**Registered prediction.** The sparse curve continues to climb past 3,072 atoms
on (f1) and its slope continues to shallow, so that extrapolating it to the dense
ladder's accuracy requires a budget outside anything this corpus can fit. The
crossing region does not widen with scale.

**Kill switch 1.** If the sparse curve saturates below 3,072 atoms once the floor
is lifted, then M107's ladder was **not** truncated by the floor, the sparse
representation is at capacity, and no future milestone may attribute a sparse
shortfall to sample adequacy.

**Kill switch 2.** If the curve keeps climbing at a slope that would reach the
dense ladder's accuracy inside a budget this corpus can fit, the extrapolation is
**registered and then measured**, not reported. An extrapolated crossing is not a
crossing.

**Restrictions.**

1. **(f2) is never compared to any arm outside M112.** A different class set is a
   different task.
2. **(f1) supersedes M107's subsample disclosure** for M112 figures only. Every
   other milestone's figures keep v15 §7.14 restriction 8.
3. **M112 runs last** because it is the most expensive and the least diagnostic:
   it sizes a curve rather than testing a mechanism.

---

## 6. Execution order and budget

**Order is registered and is not a preference.** M108 → M109 → M111 → M112, with
M110 computed over each as it lands.

- **M108 first** because every later sparse arm should be built the best way
  known, and because it is the first out-of-corpus test of C103.1.
- **M109 second** because it is the user-facing question this document exists to
  answer, and because deferring it again would repeat the failure recorded in §4.1.
- **M111 third** because its resolutions depend on where M108 leaves the sparse
  curve.
- **M112 last**, per §5.5 restriction 3.
- **M110 continuously**, because it is a verifier axis rather than a run.

**Budget, from §4.2 and stated as an estimate rather than a measurement.**
M107 cost 28.5 hours CPU-bound. The same dense work at DirectML/ROCm rates is
roughly 2–2.5 hours. Trunk training a ViT-S for ten epochs over 138,000 images is
about 2.45 hours at the measured 156.50 img/s, and rungs (t2) and (t3) are
cheaper than (t4). **The whole v16 programme is estimated at well under a week of
device time, against roughly two months at M107's rate.** If any milestone
exceeds 12 hours, that is an instrument fault to investigate under §4.6 before it
is an experiment to wait for.

**Registered extension, 5 August 2026 — the crossover as a function of data
size.** §8's crossover map is served by M111 and M112 and by one further
measurement registered here: a **train-size sweep** of the sparse ladder against
the frozen dense reference, over registered fractions of the shared subsample
(12.5%, 25%, 50%, 100%), at the registered budgets each row-count's §3.5 sample
floor admits. The literature says kernel / random-feature methods win in the
low-data regime; this sweep maps where the sparse-wins / dense-wins boundary
sits as data grows, and is the honest place the M107 window is shown to live
rather than to refute anything. It is not a new milestone: it runs inside
M108's rig as an additional arm set after M108's main arms, costs a small
fraction of M108 itself, and nothing in it may be compared across row-counts
(the task's train set differs, so each fraction is its own corpus with its own
floor).

---

## 7. Prior art v16 builds on rather than rediscovers

Carried from v15 §8.10.1–§8.10.5 and binding under §3.2 and §3.7.

| source                                                                           | what it establishes                                                                                                                                                                                                         | what v16 does with it                                                                                                                                                      |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Avron, Kapralov, Musco, Musco, Velingker & Zandieh, ICML 2017 (arXiv:1804.09893) | ridge-leverage sampling needs `O(s_λ log s_λ)` features against uniform's `O(n_λ log s_λ)`, **with a matching lower bound**; citation confirmed by the §7.1 search                                                          | **M108 arm (e) implements it** as the published comparator                                                                                                                 |
| Paul & Drineas, Neural Computation 2015 (arXiv:1506.05173)                       | leverage-score sampling as **feature selection for ridge regression / regularized least-squares classification** with risk bounds; the closest published antecedent of arm (e)'s construction                               | names arm (e) a re-implementation of a published feature-selection mechanism, never a new one                                                                              |
| Shahrampour & Kolouri, 2019 (arXiv:1903.08329)                                   | empirical leverage-score sampling of random features beats vanilla Monte Carlo and approaches supervised data-dependent kernels **without using labels**                                                                    | strengthens arm (e)'s standing: the label-free leverage advantage is already measured in the literature                                                                    |
| Li, Ton, Oglic & Sejdinovic, ICML 2019 (arXiv:1806.09178)                        | uniform `Ω(√n log d_eff)` vs leverage `Ω(d_eff)`                                                                                                                                                                            | names `d_eff` as the controlling quantity behind M108's prediction                                                                                                         |
| Rudi & Rosasco, NeurIPS 2017 (arXiv:1602.04474)                                  | `O(√n log n)` suffices; faster rates need a problem-dependent distribution                                                                                                                                                  | why M108 predicts a _smaller_ gain on 345 classes than on 10                                                                                                               |
| Shahrampour & Tarokh, NIPS 2018 (arXiv:1810.03817)                               | greedy sequential selection of explicit features by a correlation metric beats data-dependent random features at a fixed feature count, with learning bounds                                                                | the closest published antecedent of M108 arm (c) vs arm (a); arm (c) is an empirical transfer of this published phenomenon to patch dictionaries                           |
| Thiry, Arbel, Belilovsky & Oyallon, ICLR 2021 (arXiv:2101.07528)                 | **this program's sparse pipeline itself**: a whitened patch dictionary + linear classifier reaches 87–90% on CIFAR-10 and an ImageNet "baseline without representation learning", with low-dimensional dictionary structure | M108 arm (a); the reason the sparse side is a published strong baseline, not a straw man                                                                                   |
| Coates, Ng & Lee, AISTATS 2011                                                   | origin of the patch-dictionary + triangle-encoding + linear-head pipeline; not on the arXiv index (PMLR), cited via Thiry et al.                                                                                            | the pipeline's provenance; arm (a)'s ancestry                                                                                                                              |
| Rahimi & Recht, 2007/2008                                                        | random features / random kitchen sinks; not on the arXiv index (NIPS), cited throughout the kernel literature                                                                                                               | the random-feature ancestry of the sparse side                                                                                                                             |
| Oquab et al., DINOv2                                                             | the dense trunk, used unmodified at its own linear-probe protocol                                                                                                                                                           | M109 rungs (t1)–(t4)                                                                                                                                                       |
| Hu et al., 2021, LoRA                                                            | low-rank adaptation                                                                                                                                                                                                         | M109 rung (t3), dense side                                                                                                                                                 |
| Ghorbani, Mei, Misiakiewicz & Montanari, NeurIPS 2020 (arXiv:2006.13409)         | neural networks outperform kernel/RKHS methods when they learn a low-dimensional representation; hypothesised in image classification                                                                                       | **the theoretical frame M109's registered prediction is written against**: it predicts the dense side gains more from trunk training, which is exactly what M109 registers |
| Lee, Shen, Song, Wang & Yu, 2020 (arXiv:2009.09829)                              | generalised leverage-score sampling for neural networks and the NTK                                                                                                                                                         | confirms leverage sampling is an active, established technique beyond kernel methods                                                                                       |
| Han, Avron, et al., 2021 (arXiv:2104.01351, 2106.07880)                          | leverage-based feature sampling scales NTK/CNTK features                                                                                                                                                                    | the same leverage machinery used at scale                                                                                                                                  |
| Dandi, Vilucchio, Arnaboldi, Tabanelli & Krzakala, 2026 (arXiv:2605.13612)       | feature learning = iterative selection of directions with maximal low-degree correlation to the label ("Neural LoFi")                                                                                                       | names M108 arm (c) a single-layer, selection-based instance of the published feature-learning mechanism                                                                    |
| Székely, Bardone, Gerace & Goldt, NeurIPS 2024 (arXiv:2312.14922)                | random features cannot learn the higher-order structure neural networks learn (spiked-cumulant model)                                                                                                                       | the mechanism behind M109's registered prediction and §8's honest reading of M107's crossing                                                                               |

**The standing instruction this table serves:** do not re-derive existing ideas
from scratch. Where a mechanism is published, v16's job is to **measure against
it**, not to rebuild it and claim the difference.

### 7.1 Prior-art search audit, 5 August 2026

**Why it was run.** The user directed a literature search before any v16
milestone ran, so that nothing this program might produce would claim credit
for an idea others have already published. This section records the search as
an instrument: registration first, then execution, then the limits, then what
the results do and do not license.

**Registered before searching** (claims checked, displacement criteria, anchor
queries, topic queries — `/memories/session/v16-prior-art-search.md`):

1. **Claim checked 1** — a patch dictionary grown by greedy group-OMP against a
   class residual beats a random draw at matched atoms on image classification
   (M108 arm (c)). The greedy family is disclosed as old; what was checked is
   whether the specific empirical transfer is published.
2. **Claim checked 2** — ridge-leverage sampling as the published comparator
   (M108 arm (e)); and whether a stronger direct antecedent exists.
3. **Claim checked 3** — the efficiency comparison: a sparse patch-dictionary
   code vs a frozen _and_ fine-tuned dense ViT/DINOv2, both trained under a
   matched schedule, accuracy per inference MAC and per parameter (M109).
4. **Claim checked 4** — the pipeline itself (ZCA whitening + patch dictionary +
   triangle encoding + linear head) and the strongest form of the
   "random patches beat learned" baseline.

**Instrument facts, recorded so the search is not over-read.** The arXiv API
was the index; two-stage exact-phrase anchors then topic queries. Anchor recall:
`Faster Kernel Ridge Regression Using Sketching and Preconditioning` (Avron et
al., arXiv:1611.03220) and `Patches Are All You Need?` (Trockman & Kolter,
arXiv:2201.09792) were found by exact phrase. `Coates, Ng & Lee 2011` and
`Rahimi & Recht 2007` were **not** found by exact phrase because those papers
are not on the arXiv index (PMLR / NIPS); their families were found via
citations (Fastfood arXiv:1408.3060; the random-kitchen-sinks literature). The
Semantic Scholar API rate-limited (HTTP 429) on every attempt; those queries are
recorded as **rate-limited, not empty**, and were not repeated to completion.
A failed anchor is a broken search, not absence of prior art; a rate-limited
query is not a zero-hit query. The nearest papers below were found by multiple
independent queries, which is the strongest this index licenses.

**Findings, per claim.**

- **Claim 4 — the pipeline is published, in this program's own terms.** Thiry,
  Arbel, Belilovsky & Oyallon, ICLR 2021 (arXiv:2101.07528), "The Unreasonable
  Effectiveness of Patches in Deep Convolutional Kernels Methods", reports
  exactly the frozen sparse side: a _whitened dictionary of patches_ followed by
  a linear classifier, reaching 87–90% on CIFAR-10 and an ImageNet "new
  baseline for object recognition without representation learning", with
  ablations showing the dictionaries are low-dimensional. Coates, Ng & Lee
  (AISTATS 2011) is the pipeline's origin. **M107's sparse ladder and M108's
  arm (a) are re-measurements of a published model family.**
- **Claim 2 — arm (e) has a direct published antecedent.** Paul & Drineas,
  "Feature Selection for Ridge Regression with Provable Guarantees" (Neural
  Computation 2015, arXiv:1506.05173), studies leverage-score sampling as
  **feature selection for ridge regression / regularized least-squares
  classification** with risk bounds. Shahrampour & Kolouri (2019,
  arXiv:1903.08329) measure empirical leverage-score sampling of random
  features beating vanilla Monte Carlo **without labels**. The plan's citation
  of Avron et al. ICML 2017 (arXiv:1804.09893) is confirmed correct. Arm (e)
  is therefore not merely a cited comparator: it is a re-implementation of a
  mechanism with a published empirical and theoretical record.
- **Claim 1 — arm (c)'s phenomenon is published in a neighbouring setting.**
  Shahrampour & Tarokh, NIPS 2018 (arXiv:1810.03817), "Learning Bounds for
  Greedy Approximation with Explicit Feature Maps from Multiple Kernels",
  shows greedy sequential selection of explicit features by a correlation
  metric beats data-dependent random features at a fixed feature count, with
  learning bounds. That is the same direction as M108's registered prediction
  (arm (c) beats arm (a)) in the kernel-feature-approximation setting. The
  discriminative-dictionary-learning literature (e.g. Ghanem & Ahuja 2011,
  arXiv:1109.2389) optimises dictionaries with discriminative terms and is a
  separate, large family.
- **Claim 3 — the M109 comparison was not found in the searched indexes.** No
  paper was found that measures a sparse patch-dictionary code against a frozen
  _and_ fine-tuned DINOv2/ViT at matched inference MACs and parameters under a
  matched training schedule on one corpus. The "training-free" ViT literature
  found by query (82 hits) is architecture search and token compression, a
  different meaning of the term. The theoretical backdrop that _does_ bear on
  the question is published: Ghorbani, Mei, Misiakiewicz & Montanari, NeurIPS
  2020 (arXiv:2006.13409), "When Do Neural Networks Outperform Kernel
  Methods?", shows neural networks can beat RKHS methods when they learn a
  low-dimensional representation, and hypothesises this structure in image
  classification — the same direction as M109's against-the-program's-interest
  registered prediction.
- **Claim 5 — the mechanism arm (c) implements is now formalised in the
  literature.** Dandi, Vilucchio, Arnaboldi, Tabanelli & Krzakala, 2026
  (arXiv:2605.13612), "Deep Learning as Neural Low-Degree Filtering", define
  feature learning as an iterative spectral procedure in which each layer
  selects directions with maximal low-degree correlation to the label. M108
  arm (c)'s group-OMP against a one-hot residual is a single-layer,
  selection-based instance of exactly this. This is the strongest statement the
  search found that the program's mechanism is a published mechanism, and it
  gives the reframed M108 question (§5.1) its modern name.
- **Claim 6 — the reason the sparse side is expected to lose once both sides
  are trained is now measured, not only conjectured.** Székely, Bardone,
  Gerace & Goldt, NeurIPS 2024 (arXiv:2312.14922) show that in a
  spiked-cumulant model random features are "not better than random guessing"
  where neural networks learn, because learning higher-order structure needs
  moments that lazy methods cannot reach. This is the mechanism behind M109's
  registered prediction and behind §8's honest reading of M107's crossing: the
  crossing is a protocol artefact (out-of-distribution dense, corpus-matched
  sparse), not a refutation of the prior art.

**What the search licenses, and what it does not.**

- It **licenses** that M108's arms (a), (c) and (e) are measurements _of_ or
  _against_ published mechanisms, so the milestone's claims must be framed as
  transfer/measurement results, never as mechanism novelty. Prohibition 26 and
  §7's standing instruction already say this; the audit gives them concrete
  citations.
- It **does not license** any novelty claim for M109's comparison. Absence from
  the arXiv index (with the Semantic Scholar rate-limit) is not evidence of
  absence from the literature. Per v15 §8.4's consequence and this program's
  standing rule, **no sentence in any v16 document may claim "first" or
  "novel" on the strength of this search.**

**What is different from the prior art, stated honestly.** The mechanisms are
all published. What v16 adds, if the measurements land, is not a mechanism but
a set of _comparisons under one protocol_: (i) the greedy-vs-random and
leverage-vs-greedy atom comparisons transferred to a 345-class, six-domain
corpus inside M107's exact protocol (M108), where the nearest published
antecedent is a different setting (kernel-feature approximation) and a
different scale; and (ii) the matched-schedule, both-families-trained
efficiency comparison on both the MAC and the parameter axis (M109), which
was not found in the searched indexes. Even these are not claimed as novel;
they are registered as measurements whose outcome letters the kill switches
define.

---

## 8. What would count as the breakthrough, stated so it can fail

A sparse, inspectable model is a breakthrough for this program's purposes if it
sits **above** a dense model of the same inference cost, at an accuracy someone
would deploy, with **both** sides trained, on a corpus neither was tuned for, and
with the advantage confirmed on **both** the MAC and the parameter axis.

M107 delivered exactly one of those five: above a dense model of the same
inference cost. The accuracy was 21.52% on 345 classes, neither side was trained,
the opponent was out of distribution and cheaper, and only one axis was
registered.

**First, the honest reading of M107, registered 5 August 2026.** M107's crossing
is **not** counter to prior art. Every element is explained by the protocol: the
crossed dense arms are out of distribution (DINOv2 with interpolated position
embeddings at 4–9 tokens), the sparse code is corpus-matched while DINOv2 is
LVD-142M pre-trained, the gate compares against cheaper dense arms (bound 5),
and nothing on either side was trained. The prior-art-compatible reading is
stronger than the counter-to-prior-art one: when a pre-trained transformer is
taken far out of distribution and MACs are matched, a corpus-matched fitted code
wins. The reason this should close once both sides are trained is now measured,
not merely conjectured (Székely et al., NeurIPS 2024). M109 and M111 are the
registered tests of this reading, and their registered predictions expect it to
be confirmed.

**The achievable breakthrough, reframed, registered 5 August 2026.** The
"both-trained, matched-cost, deployable-accuracy MAC win" is the form the
literature has most reason to believe is closed. The form that remains open,
stated so it can fail, is:

> **A selection-trained sparse model captures a measurable fraction of the
> feature-learning gap to a trained transformer — against the random-feature
> null — at a small fraction of the training cost, and this program can say
> where that share comes from and where it stops.**

Three measured contributions sit under this sentence, each with its own kill
switch above:

1. **The gap-closing fraction** (M108, §5.1): how much of the distance between
   the random-patch null and a pre-trained transformer does greedy / leverage
   selection recover at zero gradient training, on 345 classes? M108 arm (c) is
   a single-layer, selection-based instance of the published feature-learning
   mechanism (Dandi et al., 2026). The fraction is re-measured at every M109
   rung, where the theory predicts it shrinks.
2. **The both-axes story** (§5.3, C110.1): the parameter axis is where the
   sparse side can plausibly still win after M109 closes the MAC window — the
   dense family pays its full trunk parameter count at every resolution, the
   sparse side does not. A result in which the two axes disagree about the
   winner is itself a clean, rarely-reported finding.
3. **The crossover map** (§5.5 and §6): where the sparse-wins / dense-wins
   boundary lives as a function of data size and budget. The literature says
   kernel / random-feature methods win in the low-data regime; mapping that
   boundary for patch codes vs ViT is unclaimed and is the honest place the
   program's window can be shown to live rather than to refute anything.

None of these claims mechanism novelty — the §7.1 audit forbids it. They claim
_measurements under one protocol_ of a mechanism the current literature has just
formalised.

If M109 kill switch 1 fires — the dense side gains more from trunk training and
the window closes — then the honest reading is that M107's crossing was an
artefact of freezing, and this program will say so in that sentence, in the
headline position, under §3.6 — and the three reframed contributions above
remain the measurable content.

---

## 9. Files

| path                                              | role                                                                                                     |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `analysis/RESEARCH_IMPLEMENTATION_PLAN_v16.md`    | this document                                                                                            |
| `analysis/CLAIM_LEDGER_v16.md`                    | every v16 claim with its evidence pointer                                                                |
| `experiments/tier4/verify_v16_plan.py`            | recomputes every figure quoted above from evidence, then corrupts each document to prove the checks fire |
| `experiments/tier4/bench_v16_providers.py`        | §4.2 ONNX provider measurement                                                                           |
| `experiments/tier4/bench_v16_torch.py`            | §4.2 torch CPU/ROCm measurement                                                                          |
| `experiments/tier4/bench_v16_parity.py`           | §4.5 parity measurement and §4.6 startup guard                                                           |
| `logs/results/v16/provider_benchmark.json`        | §4.2 artifact                                                                                            |
| `logs/results/v16/torch_benchmark_{cpu,gpu}.json` | §4.2 artifact                                                                                            |
| `logs/results/v16/parity.json`                    | §4.5 artifact                                                                                            |

Every artifact under `logs/results/v16/` produced before a milestone runs is
marked `"_note": "engineering measurement of the instrument, NOT evidence"` in
the file itself, and none of them is an operand.
