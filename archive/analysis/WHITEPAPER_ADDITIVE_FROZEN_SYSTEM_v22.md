# Whitepaper: The Additive Frozen System

## A staged, fixed-feature architecture for learning tasks at low cost — sealed measurements of what works, what doesn't, and what ships

Date: 15 August 2026. Status: all cited figures are sealed evidence under
`logs/results/`; the source of truth for verdicts is
`analysis/RESEARCH_IMPLEMENTATION_PLAN_v22.md` section 12, and the prior-art
claim ledger is `analysis/PRIOR_ART_V22_FINAL.md` (survey M148).

---

## 1. Abstract

We measured, on a fixed corpus at matched cost, whether a system built from
**additive, non-trained constructions** — fixed feature encoders plus
closed-form fits — can learn tasks and adapt, and whether it is more efficient
than the industry default of training a large network and pruning it down.
The finding, at the measured scale: the **frozen system wins**. A fixed
whitener, a seeded patch dictionary, triangle coding, 21-bin spatial-pyramid
pooling, signed-square-root power normalisation, and a closed-form ridge head
reach **27.86% top-1 on 345-way DomainNet at ~175M MACs/image** — beating
unpruned DINOv2-small at comparable resolution by 3.4 points at roughly half
the compute, and beating the previous sealed frontier of the program at
~2.9× fewer MACs. Channel-pruning DINOv2-small to half its parameters
collapses it to 10.8%. Every _adaptive_ mechanism we measured against the
frozen baseline — late fusion with competence routing, residual-targeted
growth, group splitting, trained heads, and gradient co-adaptation of the
code itself — either lost or tied. The end-to-end arbiter therefore selects
the shipping mode: **frozen system, closed-form heads, contract gate in
front**. Nothing in this paper is claimed as a new mechanism; every component
is prior art, and our contribution is the sealed, matched-cost measurement of
which combination works.

---

## 2. The question

Two industry defaults dominate low-cost inference today: train a large
network and **prune** it down, or **route** across specialised models. Both
are gradient-based at their core. The alternative we re-explore is the
pre-deep-learning one, assembled stage by stage: fixed feature constructions
plus **closed-form** readouts, with components added one at a time only where
the data shows they pay.

Three specific questions structure the program:

1. **Construction.** Which combination of classical coding constructions
   (patch dictionaries, spatial-pyramid pooling, power normalisation,
   multi-scale patches) holds the cost–accuracy frontier, at matched cost?
2. **Additive vs pruning.** At the same parameter/cost scale, does building
   up additively beat training-big-then-pruning — and by how much?
3. **The price of freezing.** What would gradients _through_ the additive
   code buy? If they pay, ship a hybrid; if they don't, ship the frozen
   system.

---

## 3. The system

The architecture is eight stages, with one invariant stated once:
**everything the system can know exactly must be checked before anything it
knows fuzzily** (exact contract → cheap construction → fuzzy router →
learned fallback only where it pays).

```
input
  → S1 fingerprint & contract check (exact, zero learned cost)
  → S2 additive embedding (fixed construction; the "code")
  → S3 task identifier (fuzzy, cheap linear router)
  → S4 dispatcher (contract + identity + state → model group)
  → S5 task models (global model + specialist registry, closed-form fits)
  → S6 output contract (typed validation + confidence → accept/reject)
  → output
  S7 state & memory (task history, reject log, performance ledger)
  S8 adaptation loop (task registration → closed-form fit/extension →
     acceptance gate → promotion; growth and splitting live here)
```

**The static branch (S2→S5).** The promoted construction, as sealed:

1. **Whitening.** 6×6 patches, stride 1, contrast-normalised (ε = 10), then
   ZCA-whitened from a 400,000-patch fit (seed 11). Fixed forever.
2. **Dictionary code.** 1,923 atoms drawn as a seeded prefix of an 8,192-
   candidate patch pool; each image's 729 patches are scored against every
   atom with the soft "triangle" code
   `max(mean-distance − distance, 0)`.
3. **Pooling.** 21-bin spatial pyramid (1×1 + 2×2 + 4×4 sum pooling over the
   27×27 patch grid).
4. **Power normalisation.** Signed square root, then per-row L2 — the
   classical Fisher-vector post-processing.
5. **Readout.** A 345-way **closed-form ridge head** (exact solve, no
   epochs), regularisation λ = 0.1.

The same code is the substrate for everything downstream: identity routing
(E11), specialist fits, fusion (M143), growth (M145), and the end-to-end
arbiter (M146) all consume the identical cached codes.

**The temporal branch.** For tasks that unfold over time, the S2 code at each
step feeds one of three memory arms into the same closed-form readout: a
**fixed random reservoir** (delay + feedback; echo-state property verified
before any readout is trusted), an **additive tap-delay line** (concatenated
last-k codes, no recurrence), or **programmatic primitives** (plain Python
ring buffers, counters, running statistics). Trained sequence baselines are
measured alongside.

**Adaptation (S8).** Two first-class operations, both transactional and
lock-by-construction: **residual-targeted growth** (fit a new closed-form
component on the rows the fused system currently gets wrong, append, re-solve
fusion) and **split-and-rebuild** (when a task group shows two
subpopulations, split into two specialists and promote only if the fused
pair beats the incumbent).

---

## 4. Method and epistemic discipline

Every number in this paper carries the same guarantees:

- **Sealed splits and artifacts.** One fixed corpus digest; every milestone
  reproduces its sealed inputs bit-exactly before measuring anything new.
- **Anchors first.** Each cell begins with t1-style anchor reproductions of
  the sealed figures it builds on; an anchor outside tolerance voids the run,
  and the void is _recorded_, not repaired in place. (This discipline caught
  two real instrument defects in this program: a pixel-pipeline mismatch in
  the dense baseline, and an anchor protocol misregistration in the arbiter.)
- **Premise gates before dispatch.** Cell feasibility is checked arithmetically
  first — e.g. the section 5.3 floor (≥10 fit rows per fitted dimension)
  gates every fit, and budgets below it are void, not negative.
- **Smoke runs refuse the sealed output directory.** Inadmissible
  configurations cannot write where sealed figures are read from.
- **Controls and dual reads.** Every construction cell carries a trained-head
  read alongside the ridge read; every growth/split claim carries a blind
  control (random split, blind-greedy selection) so capacity-only gains are
  exposed.
- **Scope-bound citation.** A negative result may be cited only against
  designs inside its measured scope; on an axis change it becomes a
  registered prior to be re-tested, never a pre-emptive rejection.
- **Void is not negative, negative is not void.** A failed instrument voids
  a cell; a failed gate is a scoped negative. The two are never conflated.

---

## 5. Results (all sealed)

### 5.1 The construction factorial (M142) — the winner

At matched ~175.2M MACs/image, full data, ridge readout:

| Construction                                  | Top-1 (full data) |
| --------------------------------------------- | ----------------- |
| Single 6×6 patch pool (2×2 pooling)           | 0.2275            |
| Multi-scale 3/5/7 patches (one 2×2 pool each) | 0.2421            |
| Multi-scale + signed sqrt + L2                | 0.2507            |
| 21-bin spatial-pyramid pooling                | 0.2605            |
| **SPM + signed sqrt + L2 (promoted)**         | **0.2786**        |

Diagnostics: the fine 4×4 level carries the pyramid (1×1 → 0.154, 2×2 →
0.224, 4×4 → 0.260); every scale loses alone to the single 6×6 pool, and the
concatenation is additive; the power-norm gain is entirely the square root
(p=1.0, L2 alone, _hurts_ slightly). The trained-head read collapsed
(0.003–0.010) on every cell — E5-consistent; the closure rule required both
reads to fail, so the cells stand on the ridge read.

The promoted recipe reaches **0.2786 at ~175.2M MACs**: +1.7 points over the
program's sealed frontier (0.2614 at 6,144 atoms, ~500.7M MACs) at ~2.9×
fewer MACs, and +3.4 points over dense r56 (0.2450 at 367.5M) at roughly
half the cost. The dense ladder still leads above ~0.31 (r70 0.3118 at
564.2M; r224 0.5368 at ~6.1G).

### 5.2 What the earlier ladder established (E1–E13, condensed)

| #   | Finding (sealed)                                                                                                                                                                                                                                                                                                                                            |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E1  | A never-trained dictionary code holds the frontier up to a point: 0.2614 @ 500.7M beats dense r56 (0.2450 @ 367.5M) on accuracy at 1.36× the cost; the 3,072-atom construction (0.2153 @ 254.6M) beats dense r42 (0.1972 @ 215.6M) by 1.8 points at 1.18× the cost, both at 138k rows. The clean both-axes win is the promoted SPM+sqrt construction (5.1). |
| E2  | Width saturates: beyond ~6,144 atoms accuracy stops rising; the code lives in ~8 effective dimensions.                                                                                                                                                                                                                                                      |
| E3  | Data is the measured lever: 0.2246 @ 138k → 0.2614 @ 410k rows, gain accelerating at the end.                                                                                                                                                                                                                                                               |
| E4  | Deeper closed-form heads are flat: one linear readout is enough.                                                                                                                                                                                                                                                                                            |
| E5  | Trained heads _lose_ on the sparse code (~15% vs 21.5%) and _win_ on dense features — heads and features must match.                                                                                                                                                                                                                                        |
| E6  | Head objective is flat (λ across two orders of magnitude); the bottleneck is the code, not the head.                                                                                                                                                                                                                                                        |
| E7  | Dictionary draw doesn't matter: two draws averaged 22.1% vs one wider pool's 22.5%; rank stays ~8.                                                                                                                                                                                                                                                          |
| E8  | Learned dictionaries (k-means, discriminative/greedy selection) don't transfer to this scale.                                                                                                                                                                                                                                                               |
| E9  | Binary codes lose ~3 points; a cost tool, not a quality tool.                                                                                                                                                                                                                                                                                               |
| E10 | Specialists win locally but lose assembled: hard routing 18.8%, oracle 20.5%, below the global 22.5%.                                                                                                                                                                                                                                                       |
| E11 | Task identity is cheaply inferable at coarse grain only: domains 75.6%, classes 22.5%.                                                                                                                                                                                                                                                                      |
| E12 | (a) A contract gate rejects out-of-contract inputs at zero learned cost with in-contract accuracy preserved exactly. (b) A programmatic count memory works at KB scale (optimum window 4). (c) Tiny trained transformers beat every fixed construction ~10× on a language task — learned components are used where the measured price of learning pays.     |
| E13 | The code's spectrum cannot predict the dense/sparse crossing; kept as diagnostics only.                                                                                                                                                                                                                                                                     |

### 5.3 Dense and pruned-dense baselines (M107 ladder, M144)

The dense ladder (DINOv2-small, closed-form ridge on the CLS+mean-patch
feature, sealed M107/M109): r42 **0.1972** @ 215.6M; r56 **0.2450** @
367.5M; r70 **0.3118** @ 564.2M; r224 **0.5368** @ ~6.1G.

The pruned-dense curve (M144, channel-magnitude pruning of attention+MLP,
no retraining, at the sealed r56 level):

| Keep fraction | Params nonzero | Accuracy   | MACs   |
| ------------- | -------------- | ---------- | ------ |
| 1.0           | 100%           | **0.2450** | 367.5M |
| 0.5           | 51.8%          | 0.1076     | 185.0M |
| 0.25          | 30.4%          | 0.0695     | 104.4M |

Halving the channels costs **13.7 points**; quartering costs **17.6**. The
additive recipe at _fewer_ MACs than the pruned arms (175.2M) holds 0.2786.
On this corpus, pruning down to additive-scale parameter counts is not a
path to this frontier; the additive side wins the comparison.

### 5.4 Integration, growth, splitting — the adaptive machinery, measured

- **Late fusion + competence routing (M143): scoped negative.** Fusion of
  six per-domain specialists plus the global model, fit on held-out rows:
  fused **0.1463** vs global **0.2251**. Refit on the arms' own train scores
  (M143b), fusion ties the global arm exactly (0.2243 vs 0.2246) but never
  exceeds it, and competence routing (0.1826) loses to plain identity
  routing (0.1877). The interface itself is the finding: on this corpus the
  integration layer adds no measured value over one global head.
- **Residual-targeted growth (M145): scoped negative.** Growing a
  floor-sized specialist on the fused system's 2,760 error rows lifts
  nothing: growth 0.1453/0.1451 vs static 0.1463 at both budgets, and the
  **blind-greedy control matches it** (0.1467/0.1469) — blind dictionary
  selection explains the null effect as well as residual targeting does.
- **Split-and-rebuild (M149/M149b): narrow pass.** Exactly one domain (of
  six) shows real two-subpopulation structure: fused 2-means split **0.1276**
  beats the real incumbent 0.1111 _and_ the random-split control 0.1242.
  Elsewhere splits gain capacity without structure — the random-split
  control caught it. Splitting is a promotable registry operation only
  where the data shows the structure.

### 5.5 The end-to-end arbiter (M146) — the price of freezing

One cell with gradients through the additive code, at the sealed 138k
level, under the program's single shared training schedule (AdamW, cosine
3e-4, batch 64, patience 2):

| Rung                                         | Accuracy   | Params |
| -------------------------------------------- | ---------- | ------ |
| r1 frozen codes + closed-form ridge (anchor) | **0.2274** | 0      |
| r2 frozen codes + trained linear head        | 0.0426     | 13.9M  |
| r3 trainable dictionary + trained head       | 0.1060     | 14.1M  |

Co-adapting the dictionary lifts the trained-head collapse (0.043 → 0.106)
but remains **12.1 points below the frozen closed-form read**. The price of
freezing is _negative_: gradients through the code do not pay. **The frozen
system ships**, with learned components confined to the places the evidence
already says they pay (E12c).

### 5.6 The temporal branch (M147 + the M134 prior)

One-step-ahead Mackey–Glass (τ=17), NRMSFE: no-memory ridge 0.1459;
tap-delay k=8 0.0181; programmatic primitives 0.0032; reservoir best per
seed **0.0022–0.0027** (u=1024, ρ 0.9–0.99), beating the best non-recurrent
arm by ≥5% relative on all three seeds; echo-state property verified per
run. Feedback earns its keep on the chaotic-series axis. The registered
prior stands: on the DSL token task the reservoir _loses_ (r128 ppl 29.72,
r512 27.54 vs count-memory w4 3.32 and transformer 2.80) — the reservoir's
value is axis-dependent, and programmatic primitives remain strong on both.

### 5.7 A second dataset (M103, CIFAR-10)

The same recipe family, fit from scratch on CIFAR-10: **0.62 → 0.69** top-1
across the 128 → 1,024-atom ladder (three seeds, per-arm spread ≤ 0.0131). The
frozen pipeline therefore has two measured from-scratch successes; no
transfer claim is made beyond those two points.

---

## 6. Prior art and claims

Every mechanism in this program is prior art. The program's survey (M148,
`PRIOR_ART_V22_FINAL.md`) registered the claim ledger before searching, and
reports its instruments honestly: the positive control (named papers that
certainly exist, queried by topic) retrieved only **1 of 6** registered
papers — so absence statements are **UNRESOLVED, never "first"** — while
found hits are decisive. The contribution claimed is narrow and explicit:
**the sealed, matched-cost, same-corpus measurement of combinations.**

### 6.1 Mechanism-by-mechanism claim ledger

| Mechanism the program uses                                        | Prior art and its claim                                                                                                                                                                                                                                                    | Our position                                                                                                                    |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Fixed-code geometry (triangle code, VLAD/Fisher, SPM, power-norm) | Perronnin & Sánchez 2013 (Fisher Vectors); Jégou et al. 2012 (VLAD); spatial-pyramid VLAD 2016; square-root normalisation 2015 — the pre-deep standard code, fully concluded.                                                                                              | Concluded, not redone. We re-measure _combinations_ at matched cost — e.g. SPM + sqrt + L2 on one corpus.                       |
| Bilinear / second-order pooling                                   | Lin et al. 2015 (Bilinear CNNs); Gao et al. 2016 (compact bilinear) — concluded.                                                                                                                                                                                           | Same: option-space only; not measured in this round.                                                                            |
| Scattering / fixed filter banks                                   | Mallat 2012; Bruna & Mallat 2013 — concluded.                                                                                                                                                                                                                              | Same.                                                                                                                           |
| Shallow learned patch encoders                                    | Mairal et al. 2014 (CKNs); Thiry et al. ICLR 2021 — the program's registered direct antecedent.                                                                                                                                                                            | Acknowledged antecedent; the program measures the frozen side of this family.                                                   |
| Untrained networks as frontiers                                   | "Untrained CNNs match backpropagation at V1" 2026; "Contrasting random and learned features" 2022; Rahimi & Recht 2019 critique — an active research program.                                                                                                              | The program is an instance of this line: our contribution is the specific sealed frontier.                                      |
| Per-domain specialists + routing                                  | Rebuffi et al. 2017 (incremental experts); Med-MoE 2024; DA-MoE 2025; AnchorMoE 2026; ViMoE; Union of Experts; MDViT — active concluded line.                                                                                                                              | Measured, not claimed: on this corpus routing lost (E10) and fusion tied (M143b) — a finding about the interface at this scale. |
| Reservoir computing / ESNs                                        | Jaeger 2001, 2002; Jaeger & Haas 2004; Maass et al. 2002 (liquid state machines); Integer ESNs 2017; Deep ESN topology 2019; Reservoir Memory Machines 2020; evolutionary ESNs 2022 — fully claimed.                                                                       | Re-measured on one new axis (M147) with additive and programmatic controls; never claimed.                                      |
| Delay-line / AR memory                                            | Takens 1981 (delay embedding); Waibel et al. 1989 (TDNN); the NARX line; AR-with-slack 2023 — fully claimed.                                                                                                                                                               | The M147 tap-delay arm is classical; only its measurement inside the staged system is ours.                                     |
| Growing / incremental reservoirs                                  | RECAP 2026; clustered ESNs 2025; online residual RC for quadrotor control 2024; RC cost-of-training 2025 — active.                                                                                                                                                         | M145's append-lock-and-re-solve is a measurement of this idea, with a blind-greedy control.                                     |
| Stacking / score fusion / classifier selection                    | Wolpert 1992 (stacked generalization); META-DES 2015; FIRE-DES++ 2018; late-fusion multimodal lines through 2026 — fully claimed. The "fused ≥ any single arm" property is a mathematical fact, not a contribution.                                                        | Measured on this corpus (M143/M143b): fusion ties the global arm, competence routing loses to identity.                         |
| Residual growth over base learners                                | Friedman 2001 (gradient boosting); Random Feature Representation Boosting 2025; ANOVA-boosting for RFF 2024; "Ridge Boosting is Both Robust and Efficient" 2025; Boosted Kernel Ridge Regression 2019 — **this claims the M145 idea itself in the random-feature regime.** | M145 is a protocol-identical measurement with a pruned-dense context — not a new idea.                                          |
| Pruning and sparse training                                       | Structured pruning of CNNs 2015; the lottery-ticket / IMP line 2019–2024; Hoefler et al. 2021 (sparsity: pruning _and_ growth); pruning-vs-efficiency benchmarks through 2026 — fully claimed.                                                                             | M144 is a baseline measurement at the program's budgets, nothing more.                                                          |
| Dense comparator                                                  | Oquab et al. 2023 (DINOv2), used unmodified (published linear-eval feature).                                                                                                                                                                                               | Used as the frozen dense ladder; no claim on DINOv2 itself.                                                                     |
| Chaotic-series benchmark                                          | Mackey & Glass 1977.                                                                                                                                                                                                                                                       | The M147 axis.                                                                                                                  |
| Data-axis scaling of frozen codes                                 | Bordelon–Canatar–Pehlevan; Defilippis–Loureiro–Misiakiewicz; ParK 2021; ASkotch 2024 — theory concluded.                                                                                                                                                                   | The Q(n) curve (E3) is ours to measure, not to claim.                                                                           |

### 6.2 What is explicitly NOT claimed

- **No novelty of mechanism.** Every construction, the reservoir, the fusion
  layer, and the growth loop are prior art; the survey's failed positive
  control additionally forbids any "first" or "open" statement.
- **No transfer claim.** The recipe's generalisation is demonstrated on two
  corpora (CIFAR-10, DomainNet); its hyperparameters were developed on
  DomainNet, and every other corpus remains unmeasured.
- **No continual-learning claim.** Keeping task A's knowledge while adding
  task B is exactly the machinery that measured negative here (growth,
  splitting, routing); the system as shipped refits per problem.
- **No high-compute claim.** The frozen family's best (0.2786) sits below
  the dense ladder above r70; the claim is confined to the low-cost regime.

---

## 7. What ships

The measured shipping mode (M146's selection): the **frozen system**.

- Whitener + dictionary + triangle code + 21-bin SPM + signed sqrt + L2
  (all fixed), closed-form ridge head (λ=0.1), per task, fit from scratch.
- The **contract gate in front** (E12a): typed contracts reject
  out-of-contract inputs at zero learned cost with in-contract accuracy
  preserved exactly — the one clean integration success, and the pattern
  every other interface must follow.
- **Learned components only where the measured price of learning pays**
  (E12c: tiny trained transformers on sequences; M147: reservoirs only on
  the axis where they beat both plain sums and hand-written state).
- For two problems at once, the measured best is **one head over the
  concatenated label space** fit on the union of rows — decomposition
  (specialists + routing/fusion) measured worse or tied.

---

## 8. Scope and limitations

- One main corpus (DomainNet 32×32, six domains, 345 classes), plus one
  second dataset (CIFAR-10) and one temporal axis (Mackey–Glass).
- The trained-head blind spot on these codes is consistent (E5, M142 dual
  reads, M146 r2/r3) but the opposite holds on dense features — heads and
  features must match, and that rule is construction-specific.
- The code's information lives in ~8 effective dimensions; depth is flat
  (E4); binary compression costs ~3 points (E9).
- M148's search instrument failed its positive control; absence statements
  are unresolved, found hits decisive.
- Growth/splitting were measured against a base system (M143's fused
  stack) that is itself a scoped negative; the growth finding is about the
  interface at this scale, not growth in general.
- The separability assumption (stages combine independently) is registered
  as tested-not-trusted.

---

## 9. Open questions

1. Does the promoted recipe transfer to other corpora (its two measured
   successes are not a law)?
2. Can the contract-gate pattern extend to learned contracts, and to the
   temporal branch's outputs?
3. Is there a healthier base system on which residual-targeted growth and
   split-and-rebuild pay — or is the closed-form global head simply the
   ceiling for this family?
4. Where exactly on the data axis does the frozen family stop being the
   right tool (E3's curve was still accelerating at the full corpus)?

---

## 10. References (prior art and their claims, as registered in M148)

1. Perronnin, F. and Sánchez, J. (2013). _Fisher Vectors: beyond
   bag-of-visual-words._ — the signed-power + L2 post-processing and the
   high-dimensional Gaussian mixture code.
2. Jégou, H., Perronnin, F., Douze, M., Sánchez, J., Pérez, P., and Schmid,
   C. (2012). _Aggregating local image descriptors into compact codes_
   (VLAD). — the pooled residual-vector image code.
3. Lin, T.-Y., RoyChowdhury, A., and Maji, S. (2015). _Bilinear CNN models
   for fine-grained visual recognition._ — second-order pooling.
4. Gao, Y., Beijbom, O., Zhang, N., and Darrell, T. (2016). _Compact
   bilinear pooling._ — the low-rank approximation of the above.
5. Mallat, S. (2012). _Group invariant scattering._ — fixed wavelet filter
   banks as deep-like representations.
6. Bruna, J. and Mallat, S. (2013). _Invariant scattering convolution
   networks._ — the scattering network.
7. Mairal, J., Koniusz, P., Harchaoui, Z., and Schmid, C. (2014).
   _Convolutional kernel networks._ — the learned-patch-kernel antecedent.
8. Thiry, L., Arbel, M., Belilovsky, E., and Oyallon, E. (2021, ICLR).
   _The unreasonable effectiveness of patches in deep convolutional
   kernels._ — the program's registered direct antecedent.
9. Rahimi, A. and Recht, B. (2019, critique of 2007). _Random features for
   large-scale kernel machines_ — the random-feature regime this program
   measures inside.
10. Bordelon, B., Canatar, A., and Pehlevan, C. (2020). _Spectrum dependent
    learning curves in kernel regression and wide neural networks._ —
    the data-axis theory; our Q(n) curve is a measurement, not a claim.
11. Rebuffi, S.-A., Kolesnikov, A., Sperl, G., and Lampert, C. H. (2017).
    _iCaRL: incremental classifier and representation learning._ —
    the expert/registry line for specialists.
12. Med-MoE (2024), DA-MoE (2025), AnchorMoE (2026) — the medical/domain
    mixture-of-experts line: per-domain specialists with learned routing,
    all trained end-to-end.
13. Oquab, M. et al. (2023). _DINOv2: learning robust visual features
    without supervision._ — the dense comparator, used unmodified with its
    published linear-evaluation feature.
14. Frankle, J. and Carbin, M. (2019). _The lottery ticket hypothesis_; the
    iterative magnitude pruning line (2019–2024). — the pruning baseline's
    method family.
15. Hoefler, T., Alistarh, D., Ben-Nun, T., Dryden, N., and Peste, A.
    (2021). _Sparsity in deep learning: pruning and growth for efficient
    inference and training in neural networks._ — the survey covering
    pruning AND growth.
16. Wolpert, D. H. (1992). _Stacked generalization._ — late fusion's
    mathematical basis; "fused ≥ any single arm" is a fact, not a
    contribution.
17. Friedman, J. H. (2001). _Greedy function approximation: a gradient
    boosting machine._ — residual-targeted growth over base learners, the
    M145 mechanism.
18. "Ridge Boosting is Both Robust and Efficient" (2025); Random Feature
    Representation Boosting (2025); ANOVA-boosting for Random Fourier
    Features (2024); Boosted Kernel Ridge Regression (2019) — the
    random-feature boosting line that claims the M145 idea itself.
19. Jaeger, H. (2001). _The "echo state" approach to analysing and training
    recurrent neural networks_; Jaeger (2002) memory capacity; Jaeger &
    Haas (2004). — the reservoir's construction and its echo-state
    property.
20. Maass, W., Natschläger, T., and Markram, H. (2002). _Real-time
    computing without stable states: a new framework for neural
    computation based on perturbations_ (liquid state machines). —
    the parallel reservoir line.
21. Takens, F. (1981). _Detecting strange attractors in turbulence._ —
    delay embedding, the tap-delay-line arm's basis.
22. Waibel, A. et al. (1989). _Phoneme recognition using time-delay neural
    networks._ — the TDNN/delay-line line.
23. Mackey, M. C. and Glass, L. (1977). _Oscillation and chaos in
    physiological control systems._ — the M147 benchmark.
24. FIRE-DES++ (2018); META-DES (2015) — dynamic classifier selection and
    base-classifier pruning: the competence-routing line.
25. "Untrained CNNs Match Backpropagation at V1" (2026); "Contrasting
    random and learned features" (2022) — the frontier-mapping research
    program this work instantiates.

_All references are listed exactly as registered in the M148 claim ledger;
survey instrument limitations (rate-limited queries recorded as failures,
the failed positive control) are disclosed there and apply to every absence
statement in this paper._
