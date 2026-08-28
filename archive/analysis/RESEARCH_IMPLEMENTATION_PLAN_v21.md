# RESEARCH IMPLEMENTATION PLAN v21

**The head-objective and geometry-construction re-measurements: unspent cells
after the M135 prior-art adjudication.**

Date: 13 August 2026.
Status: registration. All hypotheses below are written **before** any measurement.
The M135 survey (`analysis/PRIOR_ART_M135_BREAKTHROUGH_DIRECTIONS.md`) established
that every mechanism named here is prior art; the programme's contribution is the
sealed matched-cost measurement on the DomainNet corpus, never the mechanism.

---

## 0. Evidence basis (all sealed)

- The correct class loses the argmax for ~78% of test samples; accuracy lives in a
  thin positive-margin tail (M128).
- The ridge penalty axis ≥ 1.0 is sealed at M108 (a_random_3072, full data):
  1.0 = 0.21528 > 10.0 = 0.21380 > 100.0 = 0.20693 — monotone decline. The
  **λ < 1.0 direction of the frozen head was never measured**.
- The M117 surface seals Q(6144, 138000) = 0.22487 and Q(6144, 34500) = 0.09058;
  the sealed 6144-atom code memmaps exist in the external cache.
- Per-domain specialists are super-additive (M119/M124) but the assembled routed
  system was never run end-to-end.
- Atoms-axis saturation is structural for ONE pool (M126/M128); different seeded
  pools are new random directions and were never ensembled.
- The data axis is the steep one (M116); Q(n) past 138,000 rows was never measured.

## 1. Milestones

### M136 — the head objective axis (penalty sweep λ < 1.0, smoothed targets, batch hinge) — SEALED (13 Aug 2026)

**RESULT (sealed, `logs/results/v16/m136_margin_head/evidence.json`, admissible).**
t1 anchor delta −0.000261 (≤ 0.002; the F:-copy 0.22461 vs sealed 0.22487, same
as M126's registered note) and t2 anchor delta exactly 0.000000 — both pass.

- **Kill switch FIRED: the head-objective axis is CLOSED; λ = 1.0 ridge remains
  the head.** Best λ < 1.0 cell: 0.22322 (λ = 0.3). Best hinge cell: 0.00580
  (vs same-cell ridge 0.09058). Best margin over same-cell ridge: −0.001391
  (required ≥ +0.003). The exact-solve cells make the verdict: under-regularising
  the closed-form head never beats λ = 1.0.
- **The λ axis is flat, not peaked.** Full-data ladder: 0.22017 (0.01) / 0.22186
  (0.1) / 0.22322 (0.3) / 0.22461 (1.0) / 0.22528 (10.0) — a total range of
  0.005 across two orders of magnitude. The head is insensitive to
  regularisation at 6144 atoms (secondary measured fact; λ = 10.0's +0.00067 is
  below the gate and was not the registered direction).
- **The smoothed-target lemma verified exactly:** every one of the 34,500 test
  predictions of the ε = 0.1 smoothed-target ridge is identical to the λ = 1.0
  ridge (match share 1.000000). Label smoothing on a standardised ridge with
  intercept is a reparametrisation of the penalty, not a new head.
- **Hinge arm disclosure (unconverged instrument, one-sided):** under the
  registered schedule (8 epochs, step 1/(λt), shrinking) the batch hinge did not
  converge — final objective ~1.0e6 (λ = 1e-4), accuracy 0.00580 ≈ 2× chance,
  margins ~−1e8. The closure rests on the exact-solve λ axis, not on this arm;
  the hinge cell is kept as a registered one-sided negative, not a refutation of
  margin objectives in general. A properly converged margin head (S-SVM solver
  or averaged SGD with a fixed schedule) remains a separable future question.
- Same-cell margin object reproduced: positive-margin share 0.0906 = accuracy
  0.0906 (the M128 identity holds at n = 34,500).

**Question.** Accuracy lives in margins (M128). The head is a closed-form ridge at
λ = 1.0 chosen once (M107). Does the accuracy of the frozen 6144-atom codes depend
on the head objective — under-regularised ridge, smoothed targets, or a true margin
objective (multi-class hinge) — at matched head MACs?

**Cells (all on the sealed f6144 codes, no re-encode).**

1. Ridge penalty ladder λ ∈ {0.01, 0.1, 0.3, 1.0, 10.0} at full data (n = 138,000);
   one Gram, `solve_many`. t1 anchor: λ = 1.0 must reproduce the sealed 0.22487
   (tolerance 0.002).
2. Smoothed-target ridge (ε = 0.1, full data). **Registered analytic expectation:**
   for a standardised ridge with intercept, Y_s = (1−ε)Y + ε/345 implies
   cross_s = (1−ε)·cross (the ε/345 term cancels the centring exactly) and
   intercept_s = (1−ε)p̂ + ε/345 — scores shift by a constant per row, so
   **argmax predictions are identical to λ = 1.0 ridge**. The cell verifies the
   lemma numerically rather than discovering it.
3. Batch hinge (multi-class margin objective, projected-free subgradient,
   step η_t = 1/(λ t), gradient averaged over rows, shrinking (1 − 1/t), 8 epochs,
   corpus order, deterministic) at n = 34,500, λ ∈ {1e-4, 1e-3}. Same-cell ridge
   control refit; t2 anchor: the n = 34,500 ridge must reproduce the sealed
   0.09058 (tolerance 0.002).

**Gates.**

- t1/t2 anchors as above (a failure voids the run).
- **KS (kill switch):** if neither any λ < 1.0 cell nor any hinge cell beats its
  own same-cell ridge by ≥ +0.003, the head-objective axis is CLOSED and λ = 1.0
  ridge remains the head (the negative is sealed and reported). If a cell beats
  the margin, the objective is a measured lever and the winning head escalates to
  full data as a follow-up milestone.
- Margins reported for the hinge arm and its ridge control (q25/50/75/95 of
  f_true − max_other and the positive-margin share) to test whether a margin
  objective widens the M128 thin tail.

**Cost.** Head MACs identical for all arms (linear readout). Hinge at n = 34,500
is the registered compute ceiling for this milestone (full-data hinge is the
escalation cell only).

### M138 — seed-ensemble dictionaries (subspace bagging) — REGISTERED (amended 13 Aug, before measurement)

**Amendment (13 Aug, before any M138 measurement).** M126 already sealed the
CONCATENATION variant: appending freshly seeded atom draws past the pool cap is
flat in accuracy and flat in effective rank (~7.8). M138 therefore measures only
the unsealed variant: **score-level ensembling** — k independently seeded
(3072-atom pool, ridge head) pairs whose 345-class scores are averaged, at
matched total atoms (2 × 3072 = 6144, matched per-image MACs to the sealed
6144-arm). The k=2 cell's parts reproduce sealed values first: each member head
must reproduce M117's Q(3072, 138000) = 0.21528 within 0.002 before its scores
enter the ensemble.

**Question.** M126 closed concatenation (flat rank). Does averaging k independent
seeded heads at the score level — the bagging variant, never measured — beat
Q(6144, 138000) = 0.22487 at matched MACs?

**Gate.** Ensemble accuracy − 0.22487 ≥ +0.005, with the ensemble code's
effective rank reported. Fired → the rank-8 ceiling is a property of the
whitened patch space, not of the pool draw; the seed direction closes.

**RESULT (sealed 13 Aug, `logs/results/v16/m138_seed_ensemble/evidence.json`,
admissible).** t1 delta exactly 0.0 (member 1 reproduces the sealed
Q(3072,138000) = 0.21528). Member 2 (seed 22): 0.21351.

- **Kill switch FIRED: ensemble 0.22110 vs the single 6144-pool 0.22487
  (gain −0.00377, required +0.005).** Two independently seeded 3072-atom heads
  averaged at the score level reach 0.22110 — +0.0058 over their members but
  below ONE 6144-atom pool at matched MACs.
- **Joint effective rank 7.76** — indistinguishable from every single-pool
  measurement (M128: 7.84/7.75/7.83/7.77). The two pools span nearly the same
  ~8-dimensional structure; the rank-8 ceiling is a property of the whitened
  patch space, not of the pool draw. The seed direction is CLOSED from both
  sides now: concatenation (M126, flat) and score ensembling (M138, below the
  single pool).
- Secondary measured fact: score averaging DOES help over its members
  (+0.0058/+0.0076) — bagging is real but smaller than widening one pool
  (+0.0096 from 3072 to 6144).

### M139 — specialist buy-back assembled end-to-end + routing slack — REGISTERED

**Amendment (13 Aug, before any M139 measurement):** M139 is split into two runs.
M139a measures the routing slack (CPU-only, sealed f6144 codes); M139b measures the
assembled buy-back (requires a specialist code re-encode — the M124 specialist
memmaps are not in the external cache).

**M139a — the routing slack (SEALED 13 Aug 2026).** A 6-way linear domain router (ridge,
λ = 1.0, the M136 conventions) fit on the sealed 6144-atom codes at full data.
Operands: overall domain-routing accuracy, the per-domain confusion profile, and
the router's per-row confidence distribution. No kill switch: this is a
prerequisite diagnostic; the buy-back gate lives in M139b. t1 anchor: the same
codes must reproduce the sealed class-ridge accuracy (0.22487 within 0.002) to
validate the codes-to-labels pairing.

**RESULT (sealed, `logs/results/v16/m139a_routing_slack/evidence.json`,
admissible).** t1 anchor delta −0.000261 (≤ 0.002) — codes-to-labels pairing
verified. **Domain router accuracy 0.7559** overall; per-domain recall: clipart
0.4296, infograph 0.6601, painting 0.3945, quickdraw 0.9979, real 0.7826,
sketch 0.7237.

- **The code separates domain (6-way) at 75.6% while separating class (345-way)
  at 22.5%** — a direct linear readout of the M89 fact that domain is the
  dominant structure of the code space.
- **The routing slack is concentrated in style-adjacent confusions.** The
  confusion matrix shows painting → real 47.4% (1952/4122) and clipart → real
  32.7% (915/2800); quickdraw is nearly perfectly identified (99.8% of 10,497).
  The misrouted rows land on style neighbours, not random domains — consistent
  with M89's "the same object in two domains is further apart than two different
  objects in one" and M85a's real-photo recall pattern.
- Router confidence: median margin 0.50 (q25 0.20, q95 1.03) — a registered
  confidence-gated fallback to the global head is available for M139b.
- **Consequence for M139b:** code-routed dispatch costs misrouting on ~24% of
  rows, concentrated where painting/clipart look like photos. The buy-back gate
  must be measured on the ROUTED system (oracle-domain dispatch is the ceiling
  arm, reported alongside). A confidence-gated arm (router margin below a
  threshold → global 6144 head) is registered here, before M139b runs.

**M139b — the assembled buy-back (ACTIVE).** Re-encode the 512-atom per-domain specialist
codes (M124 construction, display-GPU throttle constraint), refit the specialist
ridges, and measure oracle-routed vs code-routed specialist accuracy against the
sealed dense ladder at matched per-image MACs.

**Registered arms (13 Aug, before measurement).** All arms score the full
34,500-row shared test set with 345-class predictions:

1. `global` — the sealed f6144 global ridge head (fallback reference).
2. `oracle` — per row, the specialist of the row's TRUE domain (the ceiling arm).
3. `routed` — per row, the specialist chosen by the M139a router's argmax (the
   deployed arm, no domain labels at inference).
4. `gated(τ)` — routed if the router's margin ≥ τ, else the global head;
   τ ∈ {0.0, 0.2046}, where 0.2046 is the SEALED M139a router-margin q25
   (registered here, before this run).

**t1 anchors:** (a) each domain's specialist accuracy on its OWN test rows
reproduces the M119 sealed 512-atom full-n value (tolerance 0.002) — validates
the A5-exact specialist re-encode; (b) the class head reproduces M117's sealed
0.22487; (c) the domain router reproduces M139a's sealed 0.7559.

**Kill switch (the buy-back gate):** the ROUTED arm's per-domain accuracy must
BEAT the sealed dense r28 arm (d4a_small_28, per-domain
[0.1575, 0.0431, 0.1014, 0.1968, 0.2137, 0.0711] at 107.6M MACs) on ≥ 4 of 6
domains, at ~2.4× fewer per-image MACs (specialist 512-atom inference ≈ 49.6M).
Fewer than 4 domains → the assembled buy-back fails the A5 pattern and the
negative is sealed and reported. The oracle arm is the ceiling (reported
alongside); the gated arms are reported against routed.

**RESULT (sealed 13 Aug, `logs/results/v16/m139b_specialist_buyback/evidence.json`,
admissible).** Every anchor exact: all six specialist own-domain accuracies
reproduce the M119 sealed values with delta exactly 0.0; class head delta
−0.000261; router delta +0.000042 (reproduces M139a's 0.7559).

| arm           | global acc | per-domain                                          |
| ------------- | ---------- | --------------------------------------------------- |
| global        | 0.2246     | 0.2421 / 0.0696 / 0.1101 / 0.3278 / 0.2389 / 0.1350 |
| oracle        | 0.2050     | 0.1936 / 0.0638 / 0.1111 / 0.3245 / 0.1948 / 0.1240 |
| routed        | 0.1877     | 0.1514 / 0.0499 / 0.0786 / 0.3238 / 0.1720 / 0.1076 |
| gated(0.2046) | 0.2062     | 0.2061 / 0.0584 / 0.0973 / 0.3239 / 0.2026 / 0.1263 |

- **Kill switch FIRED (wins 3/6: infograph, quickdraw, sketch).** The assembled
  buy-back FAILS the A5 pattern: code-routed specialists beat dense r28 per
  domain on only 3 of 6 domains at 49.5M vs 107.6M MACs.
- **The oracle ceiling (0.2050) sits BELOW the global 6144 head (0.2246).**
  The specialists are super-additive per domain (M119/M124) but each is trained
  on only its domain's rows (11k-42k vs the global 138k); assembled, the data
  deficit outweighs the specialisation. The A5 parts-level win does not survive
  assembly.
- Routing slack flips one domain: clipart wins at the oracle level (0.1936 >
  0.1575) and loses when routed (0.1514); real loses even at the ceiling.
- The confidence gate recovers +0.0185 over routed (0.2062) but stays below the
  global head. The specialist buy-back as a deployable accuracy route is
  CLOSED; its remaining value is the measured per-domain property and the cost
  story, never pooled accuracy.

**Question.** The parts are sealed (M119/M124: per-domain super-additive at ~5.6×
fewer MACs). The assembled routed system was never run. Two operands:

1. **Routing slack:** a 6-way linear domain head on the codes vs oracle domain
   labels — how much accuracy is lost by routing on the code instead of the
   known domain?
2. **Assembled buy-back:** domain-routed 512-atom specialists vs the dense ladder
   at matched per-image MACs.

**Gate.** ≥ 4/6 domains beat their dense arm at matched cost (the A5/M124
pattern) with the routing slack reported. This is the strongest measured-positive
candidate for the programme's buy-back condition Q_S(C·R) ≥ Q_T(C).

### M140 — data-axis extension past 138,000 rows — REGISTERED (amended 13 Aug, before measurement)

**Amendment (13 Aug, before any M140 measurement).** Registered cells and rules:

- **Extension rule.** Per class, the extension rows are the first
  `cap − 400` raw-train rows (raw array order) of that class NOT already in the
  sealed 400-row subsample. Caps {450, 600} → n ∈ {155,250, 207,000}. The
  whitener and 6144-atom dictionary are the SEALED M117 ones (no re-fit); only
  rows are added.
- **Premise gates (GATING, before any operand):** per-class raw availability
  must cover the cap; infeasible cells are recorded as void cells, never
  silently dropped. And the fresh encoder must reproduce the sealed f6144
  memmaps on a 256-row subsample (max-abs delta ≤ 1e-5) before any new-row
  encode is trusted.
- **t1 anchor:** fitting the sealed f6144 codes at n = 138,000 must reproduce
  0.22487 within 0.002.
- **Kill switch:** Q(6144, 207,000) − 0.22487 ≥ +0.005 required; fired → the
  data axis saturates at the corpus cap (negative sealed). The crossover margin
  vs dense r42 (0.1972) is reported at both extended cells.

**RESULT (sealed 13 Aug, `logs/results/v16/m140_data_extension/evidence.json`,
admissible).** Encoder instrument check bit-exact (delta 0.0 on 256 rows);
t1 −0.000261 ✓; premise gates clean (no class short of any cap).

| n       | cap | accuracy   | gain vs 138,000 | crossover vs dense r42 |
| ------- | --- | ---------- | --------------- | ---------------------- |
| 138,000 | 400 | 0.2246     | —               | +0.0274                |
| 155,250 | 450 | 0.2309     | +0.0063         | +0.0337                |
| 207,000 | 600 | **0.2407** | **+0.0161**     | **+0.0435**            |

- **Kill switch NOT FIRED — the FIRST PASS of the v21 plan.** Adding 50% more
  rows (+69,000) buys +1.61 accuracy points, 3.2× the registered +0.005. The
  data axis does NOT saturate at the corpus cap.
- The crossover vs dense r42 widens from +0.0274 to +0.0435: the sparse family's
  data-steepness advantage (M116) is confirmed at scale — dense's frozen trunk
  cannot consume the added rows; the sparse head can.
- Per-domain at 207,000: quickdraw 0.3918 (+0.064 over the 138,000 cell — the
  extra rows land hardest where the corpus is densest, 61% quickdraw),
  clipart 0.2389, real 0.2344, sketch 0.1230, painting 0.1080, infograph 0.0676.
  The per-domain crossover on quickdraw (0.3918) beats dense r98's quickdraw
  (0.2855).
- **The lever is data.** Registered consequence: escalate — the next cell on
  this axis (further caps, or full available data) is the follow-up milestone.

## 2. Prior art and standing

Every mechanism above is claimed in the literature (M135 audit): ridge penalty
sweeps and SVM/hinge heads on fixed features are classical; label smoothing is
Inception-v3 prior art; subspace bagging and dictionary ensembling are classical;
domain-routed MoE is an active line (Med-MoE, DA-MoE, AnchorMoE); data-scaling of
fixed codes is owned by the RF learning-curve theory (M132). **No novelty claim is
made anywhere in this plan.** The measurements are the object.

## 3. Protocol (unchanged, carried forward)

Registered-before-measurement configs; t1/t2 anchor reproduction (0.002) before
any new cell is trusted; kill switches resolve negatives; matched-cost reporting
(MACs, never wall-clock); per-domain accuracy on every arm; sealed evidence with
payload hashes; smoke runs refuse the sealed output directory; no novelty claims.

## 4. Execution order (registered 13 Aug 2026, completed same day)

1. M136 (head objective) — SEALED: kill switch fired (negative).
2. M139a (routing slack) — SEALED diagnostic; M139b (assembled buy-back) —
   SEALED: kill switch fired (negative).
3. M138 (seed ensemble) — SEALED: kill switch fired (negative).
4. M140 (data extension) — SEALED: kill switch NOT fired (POSITIVE, +0.0161).
5. M141 (data escalation, below) — the registered follow-up.

## 5. M141 — the data-axis escalation (registered 13 Aug 2026, before measurement)

Registered consequence of M140's PASS: "escalate — the next cell on this axis".
Premise (measured before dispatch, `data/raw` counts): per-class raw train
counts min 612, median 1,193, total 409,832 rows; no class below 612.

**Cells (same sealed M117 construction, rows only):**

1. n = 211,140 — the uniform cap 612 (every class gets 212 more rows).
2. n = 409,832 — ALL available raw rows per class (non-uniform schedule,
   disclosed). The first 69,000 extension rows are exactly the sealed M140
   ext600 selection, REUSED from the sealed ext600 memmap behind a t2 anchor.

**Anchors:** t1: Q(6144, 138000) = 0.22487 (0.002); t2: the reused-ext600 fit
must reproduce the sealed M140 Q(6144, 207000) = 0.240667 (0.002); encoder
check bit-exact (≤ 1e-5) on 256 subsample rows.

**Kill switch:** Q(6144, 409832) − 0.240667 ≥ +0.005. Fired → the data axis
saturates between 207k and the full corpus (negative sealed). Passed → the
data lever holds to the corpus's full extent; the follow-up question becomes
whether the curve bends before the end.

**Cost note:** the all-available cell encodes 202,832 new rows (~60 min,
display-GPU throttle); the reused ext600 rows are never re-encoded.

**RESULT (sealed 13 Aug, `logs/results/v16/m141_data_full/evidence.json`,
admissible).** Encoder check bit-exact (0.0 on 256 rows); t1 −0.000261; **t2
delta exactly 0.0** (the reused ext600 reproduces the sealed M140
Q(6144, 207000) = 0.240667 — the reuse is verified, never asserted).

| n       | schedule                               | accuracy   | gain vs 138,000 | crossover vs dense r42 |
| ------- | -------------------------------------- | ---------- | --------------- | ---------------------- |
| 211,140 | uniform cap 612                        | 0.2411     | +0.0165         | +0.0439                |
| 409,832 | all available (non-uniform, disclosed) | **0.2614** | **+0.0368**     | **+0.0642**            |

- **Kill switch NOT FIRED — the SECOND consecutive PASS.** The last 202,832
  rows buy +0.0207 over the 207,000 cell (4.1× the +0.005 gate): the data
  axis does NOT bend before the corpus ends. The curve from M116/M140/M141:
  0.2246 (138k) → 0.2309 (155k) → 0.2407 (207k) → 0.2411 (211k) → 0.2614 (410k).
- **The sparse family clears the next dense rung:** 0.2614 @ 254.6M MACs beats
  dense r56 (0.2450 @ 367.5M) by +0.0164 at 31% fewer MACs — a measured
  frontier move beyond the r42 comparison this programme has used since M116.
- Per-domain at full data: real 0.2856 (+0.051 over the 207k cell — extra data
  helps photographs most), clipart 0.2836, quickdraw 0.3627, sketch 0.1600,
  painting 0.1460, infograph 0.0910.
- **The data axis is the programme's measured lever, end to end.** The corpus
  is exhausted; further data would require a different corpus, registered as
  future work.
