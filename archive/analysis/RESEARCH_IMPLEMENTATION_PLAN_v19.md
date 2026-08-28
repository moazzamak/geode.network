# RESEARCH IMPLEMENTATION PLAN v19

**The joint-budget frontier: data-elastic compute and the 2D scaling surface.**

Date: 8 August 2026.
Status: v18's sealed evidence stands. v19 consolidates the exploitation follow-ups
(M117-M121, all sealed 8 Aug) into a re-framed programme: the quality axis is not
closed — it is flat only on single slices, and the sealed 2D surface
$Q_S(\text{atoms}, n)$ is **super-additive**. The programme's measured design axis is
**data-elastic compute**: wider closed-form heads consume data productively (float,
binary, and per-domain-specialist codes alike). v19 registers the next milestones on
that axis and the refreshed prior-art position.

---

## 0. Executive summary

Sealed facts (all admissible, all committed):

1. **The single-stage accuracy "ceiling" was a slice artifact.** M117 measured the
   atoms × data surface: super-additive (excess +0.0097 / +0.0101 over the +0.005
   margin). Doubling atoms alone at small n is flat-to-negative (0.1028 → 0.1004);
   doubling data alone helps (0.1028 → 0.1457); doubling BOTH gives 0.1530, beating
   the sum of the axes. At full data, 6144 atoms reach **0.2249** vs 3072 atoms'
   0.2153. **Quality scales with the joint budget** — the first positive on the
   quality axis.
2. **Head-width data scaling (M120, folded into M117) is confirmed:** per-atom Q(n)
   steepness rises 0.094 → 0.115 → 0.134 (1536 → 3072 → 6144 atoms). The wider the
   closed-form head, the more productively it consumes data. **This is the
   "data-elastic compute" mechanism, measured.**
3. **The mechanism transfers to binary codes (M118 KS1):** the 108-bit Hamming code
   inherits the steep Q(n) (gain ratio 0.94 random / 0.92 ITQ vs float). But the
   ~3.3-pt bit loss does **not** shrink with data (KS2 fired: gap 0.020 → 0.031-0.033)
   — the bit-quantisation loss is intrinsic, not a small-n artifact.
4. **The mechanism transfers to per-domain specialists (M119):** every domain's
   specialist Q_d(n) clears the 0.5× steepness threshold (ratios 0.66-0.99), and at
   full data the 512-atom specialists match/beat the global 3072-atom arm per domain
   on 4/6 domains at ~5.6× fewer per-image MACs. A5's per-domain cost win holds at
   scale.
5. **Negatives that stand:** B1 (closed-form depth) false — M115; the MSE-proxy
   spectral certificate (M121) does not predict the accuracy crossing — but the
   surviving spectral fact is real: the sparse span holds **3.4× more one-hot label
   power** than the trunk (11.8% vs 3.5%); M114's MAC-axis breakthrough still fails on
   accuracy preservation; the ≥10× training-cost claim fails symmetric accounting
   (1.11×).

**The v19 thesis.** The family's quality is a function of the _joint_ (compute × data)
budget, not a fixed ceiling; and its compute is _data-elastic_ (head width converts
data into accuracy). The buy-back condition of v18,
$Q_S(C \cdot R) \ge Q_T(C)$, is therefore tested on a 2D surface, not a 1D slice. The
programme's honest claims remain measurement-level: the theory (RF scaling laws) is
prior art; the sealed measurement under this protocol is not.

---

## 1. The success criterion (v19)

- **Mechanism gates keep accuracy as the instrument** (unchanged).
- **Success gates are frontier gates on the surface:**
  - **Joint-budget non-domination** — a point $Q_S(C, n)$ that beats dense at matched
    per-image cost AND matched or larger data is a frontier win (M116's overtaking of
    dense r42 at matched cost, now extended to the atoms axis).
  - **Super-additivity** — the joint gain exceeding the sum of the single-axis gains
    (M117's gate). This is the quantitative signature that the "ceiling" is a slice
    artifact and that compute and data are _complementary_ resources for this family.
  - **Data-elasticity** — the Q(n) steepness as a function of head width (M120's
    measurement), the design rule: "when data is the limiting resource, put compute
    into head width."

**Revisits of v18 consequences (sealed):**

| v18 statement                                                         | v19 reading                                                                                                                            |
| --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| "The quality axis is closed for the sparse family as currently built" | **Rescinded** to "flat only on single slices". M117's surface is super-additive; M116/M117/M119 show rising quality with joint budget. |
| "No LLM path; role scoped to per-domain + training-cost wins"         | Stands for the **depth** route (B1 false) and for **accuracy-dominance**; the joint-budget route to the buy-back is measured-open.     |
| M114 "the MAC axis is not won at preserved accuracy"                  | Stands; M118 confirms the bit loss is intrinsic at every data scale.                                                                   |
| M121 "the MSE certificate does not predict the crossing"              | Stands; the classification-aware certificate is registered as M123.                                                                    |

---

## 2. The breakthrough requirement (updated)

- **B1 — depth that raises degree: FALSE (sealed, M115).** Closed-form depth via the
  registered layer rule does not lift the single-stage ceiling. Not re-opened.
- **B2 — per-layer compute constant: PARTIAL (sealed, M114/M118).** The ~8.4× op
  reduction is real; the accuracy preservation is not, at any data scale.
- **B3 — data scaling: POSITIVE (sealed, M116).** The frozen family's Q(n) is steeper
  than the cost-matched dense trunk's (ratio 1.35).
- **B6 — head-width scaling (NEW, sealed via M117/M120):** Q(n) steepness rises with
  head width. **Data-elastic compute is the measured mechanism.**
- **B7 — the joint-budget surface (NEW, sealed via M117):** $Q_S(C, n)$ is
  super-additive in (atoms, data); the ceiling was a slice artifact.
- **B8 — specialist scaling (NEW, sealed via M119):** per-domain specialists inherit
  per-domain data scaling (ratios 0.66-0.99) at ~5.6× lower per-image cost.
- **B5 — certificate: PARTIAL.** The MSE-proxy certificate failed (M121); a
  classification-aware certificate is registered (M123); the spectral asymmetry
  (3.4× label power in the sparse span) stands as an explanatory fact.
- **B4 — sequence transfer: stays closed** (contingent on B1, which is false).

**Measured status of the conjunction:** the long-term thesis $Q_S(C \cdot R) \ge Q_T(C)$
is neither confirmed nor refuted — it is now a 2D surface measurement with the
atoms axis open (M117) and the data axis won at matched cost (M116).

---

## 3. The theoretical foundation (updated)

- **RF/ridge learning curves and scaling laws** (prior art, verified): Bordelon-
  Canatar-Pehlevan 2002.02561; Canatar-Bordelon-Pehlevan 2006.13198; Defilippis-
  Loureiro-Misiakiewicz 2405.15699; Xiao et al. 2205.14846; Mei-Montanari 1908.05355;
  Bahri et al. 2102.06701; Atanasov et al. 2405.00592. **The theory predicts the
  family's Q(n) shape from the feature Gram spectrum; M116 measured the shape, M121
  measured the spectrum. The two agree on the spectral ASYMMETRY (sparse span holds
  3.4× more label power) but not on the accuracy crossing (the MSE proxy fails on
  argmax).**
- **Overparameterization optimality** (new anchor, verified): Simon-Karkada-Ghosh-
  Belkin 2311.14646 ("More is Better in Modern Machine Learning", ICLR 2024) — RF
  models can be optimal with infinite overparameterization; frames M117's atoms-axis
  finding. Chen-Schaeffer 2110.11477 — RF conditioning and the N/m complexity ratio.
- **Linear-probe sample complexity** (new anchor): Shi et al. 2303.00106 — label
  efficiency of contrastive representations under a linear probe.
- **MoE/specialist scaling** (verified): Fedus 2101.03961; Clark 2202.01169;
  Chowdhury et al. 2306.04073 (patch-level routing sample efficiency).

**What the program adds (measurement, never novelty):** the sealed per-domain,
matched-cost, joint-budget surface of a frozen patch-dictionary family on a
345-class/6-domain benchmark — no prior work measures this object (see §4).

---

## 4. Prior art (refreshed 8 Aug 2026)

Two web-UI sweeps (export API remained 429-limited; residuals recorded, not treated
as absence). **Verified anchors (new):** Simon et al. 2311.14646; Chen-Schaeffer
2110.11477; Shi et al. 2303.00106; Chowdhury et al. 2306.04073. (Earlier verified:
2002.02561, 2006.13198, 1905.10843, 2010.02681, 2405.15699, 2412.05418, 2205.14846,
2403.08160, 2405.00592, 2402.01092, 2102.06701, 2210.16859, 2303.00564, 2602.23039,
2603.14578, 2602.04774, 2603.05691, 2204.02678, 2001.08361, 2203.15556, 1908.05355,
2101.03961, 2202.01169.)

**Recall flags (8 Aug):** `"head width" scaling` 0 hits; `"wide linear head" frozen
scaling` 0 hits; `"more data" "more parameters" "interaction" scaling` 0 hits;
`"random features" "more features" accuracy` 0 hits; `"feature dimensionality"
"learning curves" regression` 0 hits; `"random features" "sample complexity" width`
0 hits. Dictionary-estimation sample complexity (Shakeri-Bajwa-Sarwate 1608.02792,
1605.05284; Schnass 1503.07027) is about RECOVERING generating dictionaries, not the
atoms × data accuracy surface.

**What is NOT done (the open space, now specific):** no published work measures
(1) the joint atoms × data accuracy surface of a frozen random-feature/patch-
dictionary code (M117's object) and its super-additivity; (2) the "data-elastic
head-width" mechanism — Q(n) steepness as a function of frozen head width (M120's
object); (3) per-domain specialist data-scaling curves on a multi-domain benchmark at
matched cost (M119's object). The theory that PREDICTS these shapes is prior art; the
sealed measurement under this protocol is not. The programme's niche is unchanged:
**measurement, never novelty.**

---

## 5. The plan

### 5.1 M122 — the binary joint surface (atoms × data × bits) — SEALED

**Question.** M114's 108-bit loss (3.3 pts) is intrinsic at fixed atoms (M118). Does
the loss shrink when atoms and bits rise JOINTLY with data? M117 showed atoms help at
full data; M114's bit cap (108 = PATCH_DIM for a linear projection) means more bits
need a different code (e.g., per-patch multi-code or a wider projection). Test: the
M117 2×2 surface (atoms 3072/6144 × n 69000/138000) at bits {108, 216} where 216 bits
come from a second independent projection (disclosed: not a linear-cap violation —
two independent 108-bit projections concatenated; the plan heading's "256" is
superseded by its own construction sentence, 216 = 2 × 108). **Gate:** does the
binary-vs-float gap at (6144, 138000, 216 bits) narrow by >= 0.01 vs (3072, 138000,
108)? If yes, the joint budget buys back part of the bit loss; if no, the binary axis
is closed as a quality route (cost route only).

**RESULT (sealed 9 Aug, `logs/results/v16/m122_binary_joint/evidence.json`,
admissible).** t1 +0.00000 for both arms (the (3072,108) cells reused M118's sealed
code memmaps and reproduced 0.1842 / 0.1820 exactly — the reuse itself is verified).
Binary accuracies at n=138000 (float references: 0.2153 at 3072, 0.2249 at 6144):

| cell       | RANDOM-108 | ITQ-108 | RANDOM-216 | ITQ-216 |
| ---------- | ---------- | ------- | ---------- | ------- |
| atoms 3072 | 0.1842     | 0.1820  | 0.1968     | 0.1907  |
| atoms 6144 | 0.1828     | 0.1814  | **0.1986** | 0.1907  |

- **KS1 FIRED (both arms): the joint budget does NOT narrow the binary-vs-float gap
  by 0.01.** b_random: gap 0.0311 → 0.0262 (narrowing +0.0048); c_itq: gap 0.0332 →
  0.0342 (narrowing −0.0009, the gap slightly GREW). The binary axis is **cost-only**;
  M114/M118 verdicts stand unchanged. v19 decision rule resolved to the "M122 fails"
  branch.
- **Reported decomposition (why it fails):** the 108-bit code CAPS the wider head.
  head-only (6144 atoms, 108 bits) makes the gap WORSE (0.0311 → 0.0420 random,
  0.0332 → 0.0434 ITQ) because the float wide-head advantage (+0.0096) is
  inaccessible to the 108-bit code (0.1842 → 0.1828, flat-to-down). bits-only
  (3072 atoms, 216 bits) narrows it (0.0311 → 0.0185 random, 0.0332 → 0.0246 ITQ).
  Joint (6144, 216) recovers the head-only regression but cannot close the gap.
- **Secondary positive (informative, not a gate):** 216 > 108 buys real quality
  WITHIN the binary family at both head widths and on every domain (+0.009–0.016
  overall; e.g. random: 0.1842 → 0.1968 at 3072, 0.1828 → 0.1986 at 6144 — and the
  bits gain is larger at the wider head). The binary quality ceiling is a
  CODE-WIDTH phenomenon, not a head-width one. But the float reference also rises
  with the wider head, so the gap never closes. Interesting secondary: ITQ-216 <
  RANDOM-216 (0.1907 vs 0.1968/0.1986) — a second learned ITQ rotation adds less than
  a second random one (the first ITQ already concentrates the variance).
- **Cost note:** 216 bits ≈ 2× the 108-bit encode cost (~8.4× float → ~4.2× float
  for 216-bit), buying at most ~0.013 — the Hamming encode is a cost route only,
  permanently.

### 5.2 M123 — classification-aware certificate (fix M121) — SEALED

**Question.** M121's MSE proxy failed to predict the accuracy crossing. Replace the
proxy with the object the accuracy actually depends on: the per-class score margins
(argmax is a margin threshold). **Plan:** predict, from the Gram spectrum and label
projections, the variance of each class's score (Canatar's modal machinery applied
per class), and the _margin_ distribution; predict the crossing where the sparse
margin distribution overtakes the dense's. **Gate:** predicted crossing within
[1/3, 3] of the measured M116 crossing; if it still fails, the certificate is closed
as a prediction tool and kept as an explanatory diagnostic only.

**RESULT (sealed 9 Aug, `logs/results/v16/m123_margin_certificate/evidence.json`,
admissible).** First-principles Gaussian margin model from the Gram spectrum +
label projections + class counts (model disclosed in the config): class-conditioned
score mean μ^(c) = S/n*c with S_cc' = Σ a*ρ,c a*ρ,c'/(vals+κ), score covariance
Σ_cc' = Σ E*ρ(P) a*ρ,c a*ρ,c'/vals (Canatar's modal error per class and pairwise),
predicted accuracy = (1/C) Σ*c P(f_c > max*{c'≠c} f_c') by Monte Carlo.

- **KS1 FIRED and KS2 FIRED — the certificate is CLOSED as a prediction tool.**
  No predicted crossing at ANY lambda (0.1/1.0/10.0). The model predicts sparse ≈
  chance (0.0030–0.0046, chance = 1/345 ≈ 0.0029) at every ladder point and every
  lambda, and dense → ~1.0 (λ=1.0: 0.15→0.60→0.96→1.0 as n rises). Measurement says
  the opposite: 0.2153 vs 0.1972 at n_max, sparse slightly better.
- **The failure is informative, not empty.** The spectral facts reproduce M121
  exactly (sparse eff-rank 7.9, tail −2.26, captures 11.8% label power; dense
  eff-rank 30.3, tail −1.39, 3.5%) — the measurement instrument is consistent. The
  margin model's dense prediction (→1.0 vs measured 0.1972) shows the Canatar
  per-class score variance makes dense's score noise vanish (its few captured modes
  are learned instantly), but the real 345-channel argmax is vastly more sensitive:
  the population-spectrum Gaussian model cannot reproduce the argmax behavior of
  either family. M121's conclusion is thereby CONFIRMED at the per-class level: the
  accuracy crossing is an argmax phenomenon that no spectral MSE/margin proxy built
  from the Gram spectrum + label projections alone tracks.
- **Per v19 §6: M123 fails — certificate closed as a prediction tool; kept as an
  explanatory diagnostic.** The surviving spectral fact remains the only
  spectrum-level quantity consistent with sparse's measured edge: 3.4× more one-hot
  label power captured (11.8% vs 3.5%). The programme stops investing in
  spectral-only certificates.

### 5.3 M124 — specialist joint surface (per-domain atoms × data) — SEALED

**Question.** M119 showed specialists inherit per-domain data scaling at 512 atoms.
Does the per-domain specialist's OWN atoms × data surface rise super-additively like
the global one? **Gate:** per-domain super-additivity at the (atoms 256/512 × data
0.4/1.0) cells, mirroring M117. If specialists are super-additive per domain, the
specialist route to the buy-back is strongest (cheap + scales).

**RESULT (sealed 9 Aug, `logs/results/v16/m124_specialist_joint/evidence.json`,
admissible).** t1 +0.00000 on all 12 anchors (both n points × 6 domains reproduce
M119's sealed accuracies exactly). Per-domain 2×2 surfaces (atoms 256/512 ×
0.4·n_d/n_d), A5-exact nested dictionaries. Super-additivity test at cell (256, 0.4·n_d),
excess = joint gain − single-axis gains (margin +0.005):

| domain    | base   | atoms axis | data axis | joint   | excess      | super |
| --------- | ------ | ---------- | --------- | ------- | ----------- | ----- |
| clipart   | 0.1104 | +0.0017    | +0.0571   | +0.0832 | **+0.0243** | yes   |
| infograph | 0.0323 | +0.0051    | +0.0247   | +0.0316 | +0.0017     | no    |
| painting  | 0.0539 | +0.0046    | +0.0477   | +0.0573 | +0.0049     | no    |
| quickdraw | 0.2574 | +0.0157    | +0.0423   | +0.0671 | **+0.0091** | yes   |
| real      | 0.1049 | +0.0070    | +0.0690   | +0.0899 | **+0.0140** | yes   |
| sketch    | 0.0662 | +0.0072    | +0.0386   | +0.0578 | **+0.0120** | yes   |

- **KS NOT fired: 4/6 domains super-additive** (threshold 4/6), and ALL six have
  positive excess — the per-domain joint surface mirrors M117's global
  super-additivity (excess range +0.0017…+0.0243, comparable to M117's
  +0.0097/+0.0101). Infograph/painting are marginally below the margin (+0.0017,
  +0.0049).
- **v19 decision rule resolved to the M124 PASS branch: the specialist route is the
  primary buy-back candidate** — cheap (~45M MACs at 512 atoms/domain vs 254.6M
  global) AND per-domain super-additive (its own atoms and its own data interact
  positively). Second positive on the quality axis after M117.
- Honest scope: the atoms-axis gains at 0.4·n_d are small (≤ +0.016); the
  super-additivity is mostly the data axis being used more productively at the
  wider head. Absolute accuracies still modest (best domain quickdraw 0.3245).

### 5.4 M125 — head-width exponent (quantify the design rule) — SEALED

**Question.** M117/M120 measured steepness rising with width (0.094/0.115/0.134).
Fit the width dependence: does Q(n) steepness ~ W^gamma? **Plan:** extract per-atoms
Q(n) exponents (power-law fit) from M117's sealed surface; regress exponent vs log
width. **Gate:** a monotone, disclosed fit (no threshold); the fitted gamma is the
"data-elasticity" constant the programme reports as its design rule. Cheapest (pure
analysis of sealed data).

**RESULT (sealed 8 Aug, `logs/results/v16/m125_head_exponent/evidence.json`).** Pure
analysis of M117's sealed surface (few-point disclosure recorded). Per-atoms
n-axis gain exponents: beta 1.136 (width 6144), 1.128 (12288), 1.117 (24576);
absolute steepness: 0.094 / 0.115 / 0.134. **The data-elasticity is in the ABSOLUTE
gain, not the rate: beta ~ width^gamma with gamma ≈ -0.012 (width-independent), while
the absolute steepness rises with width.** "Data-elastic compute" is therefore a
larger constant-rate data gain from a wider head, not a faster learning rate. Betas
are NOT monotone in width (slightly decreasing), so the rule is reported as a
trend-level statement, not a law.

### 5.5 M126 — push the joint frontier (extend atoms beyond the 8192 pool) — SEALED

**Question.** M117's atoms axis is capped by the 8192-patch candidate pool. Does the
surface keep rising past 6144 atoms? **Plan:** extend the dictionary by re-sampling
the whitened-patch distribution (new seeded draws from the M108 patch stream,
registered; construction identity disclosed), atoms {8192, 16384}; extend n to the
full 138000. **Gate:** Q(16384, 138000) - Q(6144, 138000) >= +0.005 (the atoms axis
still pays at full data), and the per-image cost at 16384 atoms (~1.4B MACs) vs the
dense curve (r70 0.3118 @ 564.2M; r98 0.4476 @ 1096M) reported for the frontier
table. **Honest note:** dense r224 (0.54) remains far ahead in absolute accuracy; the
frontier claim is cost-matched non-domination, not dominance.

**RESULT (sealed 11 Aug, `logs/results/v16/m126_atoms_extension/evidence.json`,
admissible).** t1 delta −0.00026 (6144 cell reproduces M117's sealed 0.2249 within
tolerance; the residual is disclosed and attributed to the failing-D: code copy).
Atoms axis at full data, n = 138000:

| atoms | width | Q(138000)  | per-domain (cl, in, pa, qu, re, sk) |
| ----- | ----- | ---------- | ----------------------------------- |
| 6144  | 24576 | **0.2246** | .2421 .0696 .1101 .3278 .2389 .1350 |
| 8192  | 32768 | 0.2228     | .2389 .0740 .1104 .3298 .2309 .1332 |
| 12288 | 49152 | 0.2235     | .2364 .0710 .1067 .3373 .2252 .1424 |
| 16384 | 65536 | 0.2205     | .2329 .0747 .1038 .3378 .2170 .1385 |

- **KS FIRED: Q(16384) − Q(6144) = −0.0041 < +0.005. The atoms axis does NOT pay
  past the 8192-pool cap at full data.** The full-data atoms surface is a FLAT RIDGE
  peaking at 6144 atoms and slowly declining beyond (0.2246 → 0.2228 → 0.2235 →
  0.2205). The joint frontier's atoms dimension is SATURATED at ~6144 atoms; the
  remaining joint-budget lever is DATA, not atoms (per M116/M117/M125 the n axis is
  the steep one). M126's decision rule resolves to the FAIL branch: the honest
  ceiling for this family is now measured on both axes.
- **Head-fit engineering (disclosed in the evidence):** the 12288/16384 ridges were
  fit via a memmap-backed Gram (identical arithmetic; the in-RAM accumulator OOMs at
  > = 49152 features on this 66 GB machine), with chunked-output-row accumulation (the
  > bundled scipy-openblas abort-crashes on the single M=49152 gemm) and an F-order
  > in-place solve with a working-set trim before the solve (the C-order solve's
  > internal copy OOM'd). Frontier: 16384 atoms @ 0.2205 at ~1.32B per-image MACs vs
  > dense r70 0.3118 @ 564M / r98 0.4476 @ 1096M — cost-matched non-domination only.
- **Hardware note (registered in the config):** this milestone also surfaced that
  the compute GPU (RX 9070 XT) drives the display, that the page file was on a
  failing D: HDD (bad blocks), and that the D: copy of f12288_train was corrupt
  (CRC error). The working set was relocated to the healthy F: SSD, the page file
  moved to C:/F:, and the GPU encode throttle (0.05 s/batch) added. All three were
  required for the measurement to complete.

### 5.6 Diagnostics (ride along) — EXECUTED (M128, sealed 12 Aug 2026)

Explanatory only, never gates. Evidence: `logs/results/v16/m128_diagnostics/evidence.json`
(exact at 6144/8192 via M121 full-eigh; truncated at 12288/16384 — eff-rank exact,
top-1 via power iteration, tail/captured NOT computed there, disclosed).

- **Margin statistics** on M122/M126 codes (the argmax object), 6144/8192:
  accuracy 0.2246/0.2228; margin mean −0.036/−0.040, median −0.041/−0.045, q25
  −0.075/−0.084, **q75 −0.005/−0.006 (still negative)**, q95 +0.085/+0.095. The
  correct class loses the argmax for ~78% of test samples; the model's accuracy
  lives entirely in a thin positive-margin tail. This is the mechanism behind the
  dense-vs-sparse gap (0.22 vs 0.54): the sparse argmax object is wrong on most
  samples, and its 22% comes from the tail where the true class wins by a small
  margin. Margin mass below zero at q75 means accuracy is structurally capped for
  this code geometry, independent of the ridge solver.
- **Spectrum of the extended atoms codes** (does the tail lengthen with atoms?):
  NO. Tail index −2.233 (6144) / −2.215 (8192) — flat, indistinguishable from the
  M121 value (−2.26 at 3072 atoms). The spectral asymmetry is intrinsic to the
  whitened code space, not the atom count.
- **BBP/effective-rank vs atoms** (does eff. rank track atoms?): NO. Eff-rank
  7.84 / 7.75 / 7.83 / 7.77 at 6144/8192/12288/16384 — flat at ~7.8 (M121: 7.9 at
  3072 atoms). Top-1 share 0.309/0.312/0.309/0.311, top-10 share 0.735/0.739 (exact
  cells); captured target power 0.192 (6144) / 0.235 (8192) vs 0.118 (M121, 3072
  atoms) — grows slowly with atoms, far from saturating the target signal.
- **Reading:** the code space is dominated by a fixed ~8-dimensional structure that
  adding atoms does not change. This is the spectral corroboration of the M126
  verdict (atoms saturated at ~6144): the low-rank geometry of the whitened
  dictionary, not the atom count, bounds the family's accuracy. Consistent with the
  M123 finding that the margin object is not separable by a spectral certificate.

---

## 6. Decision rules

- **M122 passes** (joint budget narrows the bit loss): the binary axis re-opens as a
  quality route — cheap encode AND data-elastic; the strongest combined claim.
  **RESOLVED: FAILED** (KS1 fired both arms; narrowing +0.0048 random / −0.0009 ITQ
  vs +0.01). The binary axis is cost-only; M114/M118 verdicts stand. Reported: 216
  bits buy within-family quality (+0.009–0.016) but the gap never closes — the
  binary ceiling is a code-width phenomenon.
- **M123 passes** (margin certificate predicts the crossing): the programme can
  predict the family's data-scaling frontier from spectra + labels alone — a real
  tool; spectral _shaping_ becomes licensed.
  **RESOLVED: FAILED** (KS1 + KS2 fired; no crossing at any lambda; the model
  predicts sparse ≈ chance and dense → 1.0, both contradicted by measurement).
  Certificate closed as a prediction tool, kept as an explanatory diagnostic.
  The programme stops investing in spectral-only certificates.
- **M124 passes** (specialists super-additive per domain): the specialist route is
  the primary buy-back candidate (cheap + scales + per-domain).
  **RESOLVED: PASSED** (KS not fired; 4/6 domains super-additive, all 6 positive
  excess, t1 +0.00000 on 12 anchors). The specialist route is the primary
  buy-back candidate: cheap (~45M MACs) AND per-domain super-additive.
- **M125:** the fitted data-elasticity gamma is reported as the design rule.
- **M126 passes** (atoms still pay at full data): the joint frontier is
  budget-unbounded within the family; the honest ceiling is cost, not accuracy.
  **RESOLVED: FAILED** (KS fired; Q(16384) − Q(6144) = −0.0041). The atoms axis is
  saturated at ~6144 atoms at full data (flat ridge, slowly declining past it).
  The remaining joint-budget lever is DATA, not atoms. v19 is now fully resolved:
  M125/M122/M123/M126 fired, M117/M124 passed.

---

## 7. What is revisited vs v18

- The "quality axis closed" statement rescinded (M117); the depth route stays closed
  (M115); the binary quality route is re-tested jointly (M122); the certificate is
  re-cast classification-aware (M123); specialists promoted to a primary route
  (M124); the frontier extended (M126).
- The training-cost ratio framing stays with the symmetric-accounting disclosure
  (M115 KS2); no headline uses it.

## 8. What is not changed

All sealed v16/v17/v18 evidence stands. The shared subsample digest, M108 exact
whitener/dictionary arithmetic, the device precondition (HIP_VISIBLE_DEVICES=1,
gfx1201), and the protocol rules (registered before measurement; no novelty claims;
per-domain reporting; matched-cost comparison) are unchanged.

## 9. References

**Added 8 Aug (refresh):** 2311.14646 (more is better / infinite overparameterization,
ICLR 2024), 2110.11477 (RF conditioning, N/m scaling), 2303.00106 (linear-probe label
efficiency), 2306.04073 (patch-level MoE routing sample efficiency), 1608.02792 +
1605.05284 (dictionary-learning minimax sample complexity), 1503.07027 (ITKM
dictionary sample complexity).

**Added 12 Aug (HTN routing survey, M127):** ChatHTN (2505.11814), Online Learning of
HTN Methods for integrated LLM-HTN Planning (2511.12901), HTN Planning with
LLM-Generated Heuristics (2605.07707), Neural Bandit Optimal LLM Selection for a
Pipeline of Subtasks (2508.09958), RouteLLM/cascades, THOR-MoE, HAPS, LogRouter. Full
survey + gap analysis: `analysis/HTN_ROUTING_LITERATURE_REVIEW.md`.

**Added 12 Aug (programmatic-primitives survey, M129):** ART (tool-augmented reasoning),
PORTS + RaTA-Tool (retrieval-based tool selection), Unified Tool Integration for LLMs
(protocol-agnostic function calling), Chameleon + HYDRA (LLM-as-controller module
composition), SelectiveNet + CascadeDebate (reject option / cost-aware cascades),
Stable-MoE / SiftMoE / SpaceMoE (energy-aware MoE routing), Uni-Skill + Logic-Skill
Programming (skill libraries + planner), MathDSL (DSL via program synthesis). Full
survey + technique table: `analysis/PROGRAMMATIC_PRIMITIVES_LITERATURE_REVIEW.md`.

## 10. Future work — HTN-style hierarchical routing (registered 12 Aug 2026, NOT a plan)

**Idea (noted for the future, not committed):** use an HTN-style task vocabulary and
methods (tasks with input requirements and effects, chained into a plan) as the router
of the GEODE/CG-MoE specialist system — the routing decision is an HTN plan whose tasks
are specialist models, with the lightweight LLM maintaining the vocabulary/methods
OFFLINE (never at inference) and the plan executed symbolically. Registered survey M127
(`logs/results/v16/m127_htn_routing_litsearch/evidence.json`) found every component
exists separately (classic HTN; LLM x HTN hybrids; LLM routing; hierarchical MoE) but
not the exact combination. **No novelty claim is licensed by that absence.** If
re-opened, it must be a MEASUREMENT milestone: does HTN-structured dispatch beat the
flat router on robustness/cost on the sealed corpus, with the displacing neighbours
(above) registered as such. Not on any active plan.

## 11. Future work — programmatic primitives + hybrid router (registered 12 Aug 2026,

    ACTIVE via `analysis/ENGINEERING_PLAN_v20.md`, NOT a research claim)

**Idea (engineering track):** treat well-defined computations (math, transforms,
shape/range checks) as PROGRAMMATIC primitives sharing the fingerprint interface with
learned primitives; a contract-gated router (fingerprint match → cheapest correct
primitive, rule-based, no LLM at inference) dispatches inputs with a reject/cascade
fallback; a learned model is used only where no programmatic primitive exists. Goal:
win on footprint and energy at a modest accuracy cost. Registered survey M129
(`logs/results/v16/m129_programmatic_primitives_litsearch/evidence.json`) found every
component exists separately (tool use, neuro-symbolic hybrids, typed tool schemas,
LLM-as-controller, reject/cascade, energy-aware MoE routing, skill libraries, program
synthesis) but not the exact combination — a no-LLM-at-inference contract-gated
dispatch over non-LLM specialists, measured for footprint/energy. **No novelty claim is
licensed by that absence.** Techniques per issue (interface contract → typed schemas,
router → rule/retrieval-based, fallback → reject-option/selective-classification
thresholds + cascades, energy → matched-cost measurement): see
`analysis/PROGRAMMATIC_PRIMITIVES_LITERATURE_REVIEW.md`. The engineering plan
(`analysis/ENGINEERING_PLAN_v20.md`) makes it a MEASUREMENT milestone on the sealed
corpus (out-of-contract rejection with zero learned forward passes, in-contract
accuracy preserved).

**Earlier (unchanged):** all v18 §9 references, plus 2002.02561, 2006.13198,
1905.10843, 2010.02681, 2405.15699, 2412.05418, 2205.14846, 2403.08160, 2405.00592,
2402.01092, 2102.06701, 2210.16859, 2303.00564, 2602.23039, 2603.14578, 2602.04774,
2603.05691, 2204.02678.
