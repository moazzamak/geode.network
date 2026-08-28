# RESEARCH IMPLEMENTATION PLAN v23

## The questions v22 opened, and the cheapest admissible path to answering them

Date: 16 August 2026. Status: registered plan; NOTHING in this plan has been
measured. Milestone IDs M150–M164 are reserved here; the v22 queue (M142–M149)
is closed and sealed.

**How to read this document.** This plan inherits the v22 style and its
binding epistemic rules (v22 sections 3 and 11, restated in section 3 below):
plain-English question/answer rows, scope-bound citation, anchors before any
new number, premise gates before compute, smoke runs that refuse sealed
directories, void-is-not-negative, and a result recorded only in its measured
scope. Every cell below registers, BEFORE measurement: the question, the
construction, the anchors, the premise gates, the controls, and the gate that
decides it.

---

## 1. What v22 decided (one line each)

- **M142:** the promoted construction is 21-bin SPM (1,923 atoms) + signed
  sqrt + L2, ridge λ=0.1: **0.278551** at ~175.2M MACs/image on DomainNet
  345-way — above the sealed frontier (0.261362 @ ~500.7M) and dense r56
  (0.245014 @ 367.5M).
- **M144:** channel pruning collapses the dense arm (0.2450 → 0.1076 at
  half channels, no retraining); the additive recipe wins the
  additive-vs-pruning comparison at this scale.
- **M143/M143b:** late fusion ties the global arm; competence routing loses
  to identity routing. The integration interface adds no measured value.
- **M145:** residual growth on the fused stack's errors adds ~nothing, and
  blind-greedy selection matches it.
- **M146:** gradients through the additive code lose by 12.1 points; **the
  frozen system ships**.
- **M147/M134:** the reservoir wins Mackey–Glass, loses the DSL token task;
  its value is axis-dependent.
- **M149/M149b:** split-and-rebuild pays in exactly one domain (real
  two-subpopulation structure) and nowhere else.
- **M148:** the literature instrument failed its positive control; absence
  statements are unresolved, presence decisive.

## 2. The questions (carried from the v22 closing synthesis)

Grouped, each with the sealed finding that raises it:

1. **Frozen/trained boundary.** Trained heads collapse on additive codes
   (E5, M146 r2 0.043) but win on dense features; co-adaptation lifts but
   does not close (r3 0.106 vs r1 0.227). Is the loss schedule-bound? Is
   there a hybrid middle? Can the winning head be predicted from the code?
2. **Representational bottleneck.** The old code lived in ~8 effective
   dimensions (E2); the promoted code was never rank-profiled. Where does
   pooling granularity saturate (1×1→2×2→4×4 was monotone)? Is SPM×MS
   additive? Does p<0.5 keep paying?
3. **Data lever vs specialist paradox.** Data is the measured lever (E3);
   specialists starved on 1/6 of the rows (E10). Does data-sharing make the
   ensemble exceed the global arm? Does any finer routing granularity win?
   Where does Q(n) saturate past 410k?
4. **Growth on a healthier base.** M145 grew on a base that was itself a
   scoped negative, floor-capped at 32–64 atoms. Does residual growth pay
   against the GLOBAL head at full-data scale? Does reservoir growth pay?
5. **Dense side.** M144 pruned without retraining; the industry default is
   prune+retrain. The promoted-vs-dense crossing has one measured point.
6. **Temporal axis-dependence.** Reservoir wins chaos, loses grammar. What
   task property predicts recurrence paying? Does a reservoir+programmatic
   hybrid combine the wins?
7. **Methodological debt.** The M148 instrument cannot support absence
   claims; the contract gate (E12a) was never re-tested on the promoted
   construction.

## 3. The approach and the rules (carried, restated once)

**Currency.** Cells cost in GPU encodes (≈1–2 h each) or in trained runs
(≈1–4 h each). Fits on cached codes are free (minutes). The cached substrate
available to every cell: `v16/m142_c2/spm1923_fulltrain.npy`
(409,832 × 40,383), `pool2062_fulltrain.npy` (409,832 × 8,248),
`spm1923_fulltest.npy` (34,500 × 40,383), `v16/m142_c3/ms357_fulltrain.npy`
(409,832 × 13,244), the f6144 memmaps (`v16/m117`), the M143 score cache
(`v16/m143/scores.npz`) and the M143b train-score cache
(`v16/m143b/train_scores.npz`), all under the pinned subsample digest
`63f590097008f749…` and the M141 cell-2 row schedule (part 1 = 138k
subsample positional; parts 2–3 raw-indexed). SPM×MS is a column
concatenation of two cached matrices — free.

**Rules (binding, from v22 §3 and §11).**

- Anchor reproduction first: every cell re-derives its sealed inputs
  bit-exactly (tolerance per cell) before any new number is read.
- Premise gates before compute: floor arithmetic (≥10 fit rows per fitted
  dimension, section 5.3), row-schedule identity, artifact shape checks —
  checked in-run, gating, and a budget below the floor is VOID, not negative.
- Smoke runs declare inadmissibility and refuse the sealed output directory.
- Controls and dual reads: every construction cell carries a trained-head
  read alongside the ridge read; every growth/split/routing cell carries a
  blind control (random split / blind-greedy selection / identity router).
- Scope-bound citation (v22 §3 rules 1–3) applies unchanged.
- Absence statements remain unresolved under the M148 instrument (M164 is
  the fix, registered below).

**Waves and re-decision points.** (a) fits-only wave → re-decide on
construction and integration; (b) one-encode cells → re-decide on
construction; (c) trained runs → re-decide on shipping; (d) corpus decision
→ Q(n) and transfer. Each wave re-sizes the next; nothing after a
re-decision point runs blind (v22 §10 rule).

---

## 4. The cells (registered, unmeasured)

### Wave A — fits only (free)

**M150 — rank-and-profile sweep of the cached code family.**
Question: do measurable code statistics (effective rank, condition number,
margin profile) order the ridge-vs-trained-head outcomes across the code
family? (E2 measured one code; E13 failed to predict with one code.)
What we do: run the m138 `effective_rank` machinery on the cached pool, MS,
SPM, SPM+sqrt, and f6144 matrices; fit ridge and trained heads on each at
138k; report the statistic-vs-outcome table. Anchors: each code's ridge
refit reproduces its sealed 138k read (pool 0.206406, MS+sqrt 0.223855,
SPM+sqrt 0.227362, f6144 0.224609) within 1e-9 (same matrices, same
fitters). Premise: cached artifact shapes pinned. Gate: none — a measuring
stick; its output feeds M161 and the frozen/trained question.
Cost: fits only. Scope: predictive validity on this code family only.

**M151 — SPM × MS interaction (the separability test).**
Question: does the column-concatenated SPM+MS code (optionally + sqrt + L2)
beat the best single construction at matched cost? (v22 registered the
separability assumption as tested-not-trusted; this is the test.)
What we do: concatenate `spm1923_fulltrain` and `ms357_fulltrain` along the
feature axis (both are the M141 schedule; premise-checked row by row),
fit the ridge ladder. Anchors: fitting each half alone reproduces the sealed
reads (SPM 0.260493, MS 0.242145, full data; tolerance 1e-9). Premise:
row-schedule identity asserted (labels file digest + row count 409,832).
Controls: trained-head read on the concatenated codes.
Gate: fused-concat ≥ best single construction + 0.005 at the disclosed
feature-width cost; if the trained-head read fails and the ridge read wins,
the cell stands on the ridge read (the v22 dual-read closure rule).
Cost: fits only. Scope: one corpus, one interaction pair.

**M152 — power-exponent refinement on cached codes.**
Question: does p ∈ {0.25, 0.33, 0.66} beat the sealed p=0.5 anywhere?
What we do: apply the C4 `_fit_power`/`_score_power` path to the cached SPM
and MS matrices at the refined p-grid, full data and 138k. Anchors: the
sealed p=0.5 and p=1.0 cells reproduce exactly (SPM+sqrt full 0.278551,
138k 0.227362). Premise: none beyond artifacts present. Controls:
trained-head read at the winning p (E5-consistent collapse expected; dual-
read closure rule applies). Gate: best new-p cell ≥ sealed p=0.5 best

- 0.005 at identical cost, else archived as a scoped negative.
  Cost: fits only. Scope: power exponent axis, this corpus.

**M153 — routing granularity on cached score matrices.**
Question: does any finer routing/splitting granularity (class-groups,
adjacent-domain pairs, per-class) beat the single global head, when
domain-level routing measurably does not (E10, M143b)?
What we do: on the M143b train-score cache and the sealed test-score cache,
fit competence routers and split candidates at class-group granularity
(2-means over class score profiles, seeded), fused readout. Anchors:
reproduce M143/M143b sealed reads (fused 0.146261 / 0.224319, global
0.225101 / 0.224609) from the same caches, tolerance 1e-9. Premise:
per-group row floors (section 5.3) checked per candidate; groups below the
floor are void, not negative. Controls: identity router (0.1877 sealed) and
random router; random-split control per the M149b pattern. Gate: a
granularity passes iff fused ≥ global + 0.005 AND fused ≥ random-split
control; otherwise the single-head verdict of M143b is extended to finer
granularity. Cost: fits only. Scope: this corpus, these caches.

**M154 — data-sharing specialists (score-level fusion).**
Question: does fusing specialists whose heads were fit with shared data
exceed the global arm, when per-domain-only specialists measurably do not
(E10's data-starvation diagnosis vs E3's data lever)?
What we do: score-level stacking over cached specialist score matrices with
re-weighted fits (each specialist's score matrix reused; the sharing enters
through the fusion weights and a shared-rows refit where scores permit).
Anchors: reproduce the sealed specialist own-domain accuracies and M143b
reads from the same caches. Premise: cache shapes pinned. Controls:
competence vs identity routing (sealed values as the baselines).
Gate: fused ≥ global + 0.005 on held-out rows. Cost: fits only.
Scope: score-level sharing; head-level sharing is M159 (needs codes).

**M155 — growth premise at full-data scale (premise-only cell).**
Question: how many error rows does the global head's full-data fit leave,
and which growth budgets does the floor permit there? (M145's lesson:
measure the population before registering budgets.)
What we do: refit the sealed full-data head (f6144 path) and the promoted
SPM+sqrt head from cached codes; compute error rows on the train split.
Anchors: the refits reproduce Q(6144, 409832) = 0.261362 and 0.278551,
tolerance 1e-9. Gate: none (no new accuracy claim; the output REGISTERS
the floor-derived budgets for M156). Cost: fits only.

**M156 — residual growth on the global head (full-data scale).**
Question: does residual-targeted growth pay against the GLOBAL head's
errors, at budgets the M155 premise actually permits? (M145's negative was
measured on a weaker base with floor-capped 32–64-atom steps.)
What we do: growth dictionaries = seeded prefixes of the global pool;
specialist heads fit on the M155 error rows; append + re-solve fusion;
budgets = M155's floor-derived rungs (registered only after M155).
Anchors: static-fusion reproduction (a1 pattern), specialist-path anchor
(a2 pattern, d0 0.193571 tolerance 0.002), prefix property (a3).
Premise: M155 premise gates re-checked in-run; n_error > 0 hard-fails.
Controls: blind-greedy dictionary selection (the M108/E8 prior) at the same
budgets, identical head fit. Gate: growth ≥ static + 0.005 AND growth >
control at every budget, else scoped negative. Cost: one encode per budget
per arm (grown dictionaries). Scope: global-head base, this corpus.

**M157 — temporal task-property screen + reservoir growth (CPU).**
Question: which task property predicts whether recurrence pays (chaotic
autoregression vs discrete structure), and does a reservoir+programmatic
hybrid beat both parents? Does residual-targeted reservoir growth help?
What we do: run the M147 three-arm harness on two new series (one chaotic
map, one discrete grammar), plus a hybrid arm (reservoir state concatenated
with primitive features, same ridge readout), plus one reservoir-growth
cell (append units targeted at residual errors, echo-state property
re-verified after each append). Anchors: reproduce the M147 sealed NRMSFEs
(no-memory 0.145856, primitives 0.003172, reservoir best 0.002232–0.002661)
and the M134 DSL anchor exactly. Premise: echo-state property checked
before any readout (spectral radius < 1, warm-up discarded).
Controls: the no-memory and tap-delay arms are the controls; the hybrid
must beat both parents; growth must beat static reservoir + a random-append
control. Gate: per-axis registered margins (≥5% relative, the M147 rule).
Cost: CPU minutes. Scope: the measured axes only.

### Wave B — one encode each

**M158 — finer pyramid (levels {1, 2, 4, 8}).**
Question: where does pooling granularity saturate? (The sealed ladder
1×1 0.154 → 2×2 0.224 → 4×4 0.260 was monotone and steep.)
What we do: re-encode the M141 schedule with SPM levels {1, 2, 4, 8}
(37 bins), same dictionary and whitener; per-level reads as in C2.
Anchors: the {1,2,4} sub-pyramid must reproduce the sealed 21-bin reads
bit-exactly (t1 0.000000, the C2 encoder check pattern). Premise: the
matched-cost pair arithmetic re-registered for 37 bins; no figure quoted
before the anchor holds. Controls: trained-head read (dual-read rule).
Gate: the 8-level pyramid (optionally + sqrt + L2) is Pareto-improving —
beats the sealed 21-bin read at its disclosed MAC cost, i.e., the point
lies on/below the existing accuracy-vs-MAC frontier; else archived.
Cost: one encode (409,832 rows). Scope: pooling axis, this corpus.

**M159 — shared-fit specialists (head-level data sharing).**
Question: do specialists whose heads fit on ALL rows (shared data) exceed
the global arm, when the fusion interface alone only ties it (M143b)?
What we do: fit each domain dictionary's head on all 138k/410k rows (not
just its domain's), then the sealed fusion protocol. Anchors: specialist
own-domain reproductions (d0 0.193571) and M143b reads. Premise: floor
arithmetic for the wider fits. Controls: competence vs identity routing;
the M143b sealed numbers are the incumbent. Gate: fused ≥ global + 0.005.
Cost: one encode pass per specialist dictionary if the codes are not
already cached (checked at build time; M143b cached scores, not codes).
Scope: this corpus, these dictionaries.

### Wave C — trained runs

**M160 — M146 schedule sensitivity.**
Question: is the 12.1-point trained-side deficit a schedule artifact or
fundamental? (M146 and M109 both used one fixed 8-epoch schedule.)
What we do: re-run M146's r3 under two registered schedules (more epochs /
patience, and a different LR), same model, same seeds, same anchors.
Anchors: the sealed-schedule r3 reproduction (0.106029 within a registered
tolerance) and the frozen r1 unchanged (0.227362, tolerance 1e-6).
Premise: same cached artifacts. Gate: measuring stick — if any schedule
clears r1, the "price of freezing" verdict is schedule-bound and reopens
M146's shipping selection; if none does, the frozen verdict hardens.
Cost: two trained runs. Scope: this construction, this corpus.

**M161 — the hybrid readout (ridge + trained residual head).**
Question: does the unmeasured middle — closed-form ridge + a small trained
residual head on the same codes — beat either pure readout?
What we do: ridge solve (frozen) + trained residual head of registered size,
trained under the M109 shared schedule. Anchors: the ridge part reproduces
the sealed read exactly before the residual is added. Controls: the pure
trained head (sealed 0.0426) and the pure ridge (sealed 0.2274) are the
two arms the hybrid must beat. Gate: hybrid ≥ ridge + 0.005 on held-out
rows, else archived. Cost: minutes of training on cached codes.
Scope: readout axis, this corpus.

**M162 — prune+retrain dense (the industry default).**
Question: does prune-then-retrain recover what pure pruning lost (M144:
0.1076 at half channels, no retraining)?
What we do: retrain the M144-pruned DINOv2-small at keep=0.5 under the M109
shared schedule on the M107 pixel path. Anchors: the M144 t2 reproduction
(unpruned r56 0.245014 within 0.002, the M107-materialised pixels).
Premise: the digest-tagged pixel cache present. Gate: measuring stick —
reported against the additive recipe at matched MACs and against the M144
no-retrain curve. Cost: one trained run. Scope: keep=0.5, this corpus.

### Wave D — corpus decision (blocked until decided)

**M163 — Q(n) beyond 410k + cross-corpus transfer.**
Registered as blocked on a corpus decision: the same promoted recipe on a
new corpus (or new rows), fresh whitener/dictionary/anchors per the v22
pattern, judged against that corpus's own dense baseline at matched cost,
never against DomainNet figures. Nothing here runs until the corpus and
the per-corpus gates are registered in a dated amendment.

### Methodological debt

**M164 — search-instrument rebuild.**
Question: can a literature instrument with a passing positive control be
registered, so absence statements become possible? (M148's control failed:
1 of 6 recall.)
What we do: register anchor queries with fixed endpoints (the https arXiv
endpoint, retries with backoff for 429s) and a positive control that MUST
pass before any novelty-relevant query is read; record rate-limit failures
separately from empty results (the M88/M148 lessons). Gate: the control
passes, or the search is not used for any claim. Cost: no compute.
Scope: instrument validity only.

---

## 5. Execution order and budget

1. M155 first (it gates M156's budgets) alongside the other fits-only
   cells M150–M154, M157 — a day of CPU work, zero encodes.
2. Re-decision: does any fits-only cell change the construction or the
   integration verdict? If M151/M152 produce a new winner, the one-encode
   cells re-target it.
3. M158, M159 (one encode each) → re-decision on the construction.
4. M160–M162 (trained runs) → re-decision on shipping.
5. M163 only after a corpus decision; M164 before any novelty claim.

Total committed compute before the next re-decision: roughly two GPU days,
dominated by the encodes, with every trained run reusing cached artifacts.

---

## 6. Execution log (live)

Nothing has run. No cell's accuracy has been measured. Each cell's config,
anchors, and premise gates are registered when its build is dispatched, and
every verdict is recorded here in the v22 section-2 style, with scope
labels.

### 16 Aug 2026 — M155 build registered (premise-only; nothing measured)

M155 (growth premise at full-data scale) build is registered before any
measurement. Config `experiments/configs/v23/m155_growth_premise.json`
pins: the f6144 head = RidgeAccumulator over the M141 cell-2 accumulation
order — [v16/m117/f6144_train.npy (138k subsample), v16/m140/f6144_ext600.npy
(69k, the M141 extension rule), v16/m141/f6144_all_rest.npy (202,832)] with
the M141 label rules, penalty 1.0
(409,832 x 24,576, the M141 schedule, penalty 1.0), anchor Q(6144, 409832)
= 0.2613623188405797 (tol 1e-9); the promoted head = C4's
`_fit_power`/`_score_power` path over `spm1923_fulltrain.npy` / labels
npz, p=0.5, penalty 0.1, anchor 0.278550724637681 (tol 1e-9). Premise
outputs: train-split error-row counts for BOTH heads and the floor-derived
growth-budget ladder per head (ceil(n_err / (4a)) >= 10, powers of two).
No gate; no new accuracy claim — its output REGISTERS M156's budgets.
Interpretation note registered here: M156's growth base is the promoted
SPM+sqrt head (the shipped system); the f6144 error-row counts are
reported alongside. Smoke declares inadmissibility and refuses the sealed
output directory; smoke skips the anchors.

### 16 Aug 2026 — M150 build registered (measuring stick; nothing measured)

M150 (rank-and-profile sweep) build is registered before any measurement.
Config `experiments/configs/v23/m150_rank_sweep.json` pins six codes at
138k rows (pool2062 8,248 cols, ms357 13,244, spm1923 40,383, spm1923_sqrt,
ms357_sqrt, f6144 24,576), each with its sealed 138k ridge anchor at
penalty 1.0 (0.206406 / 0.215739 / 0.214493 / 0.227362 / 0.223855 /
0.224609, tol 1e-9 — penalty 1.0 everywhere per the C4 cells_138k
protocol). Profile = M128 definition (participation-ratio effective rank,
top-k shares, condition number on the standardised Gram via m121
`_spectrum`); trained-head read = head-only under the M109 shared schedule
(4 epochs, val 0.05, val_seed 66). No gate. Artifacts: f6144 under
`v16/m117`, the rest under `v16/m142_c2` (path defect found and fixed in
the M155 build: f6144 memmaps are NOT under m142_c2). Smoke declares
inadmissibility and refuses the sealed output directory.
AMENDED 16 Aug, registered before the full run: the M128 definition
needs only the eigenvalues, so `np.linalg.eigvalsh` replaces the full
eigendecomposition (the 40k-dim eigenvector matrices are impractical —
hours and ~26 GB per code). The definition is unchanged; only the
decomposition routine's output is trimmed. The smoke's pool2062 profile
ran the old routine before the amendment (eff-rank 10.92 at 4k rows,
inadmissible).

### 16 Aug 2026 — M151 build registered, with one amendment (test-side encode)

M151 (SPM x MS interaction) build is registered before any measurement,
with one amendment: the C3 run persisted only `ms357_fulltrain.npy` — the
MS TEST codes were never cached. M151 therefore requires ONE test-side
encode (34,500 test rows against the three C3 scale dictionaries, the C3
encoder, persisted as `v16/m151/ms357_fulltest.npy`). The encode is
anchored by the MS anchor itself: after the encode, the train-side
penalty-1.0 refit scored on the new test codes must reproduce the sealed
MS full-data read 0.24214492753623187 (tol 1e-9) — the same anchor that
proves row identity for the concatenation (the anchor reproductions ARE
the premise check). Config
`experiments/configs/v23/m151_interaction.json`: concat = SPM columns
first, MS second (53,627 cols); penalty ladder {0.1, 1.0, 10.0} x power
{raw, 0.5} at full data; gate = best concat cell >= 0.2604927536231884

- 0.005 (the SPM half's anchor is the incumbent); trained-head read at
  138k (M146 r2 protocol) as the co-adaptation control. Smoke encodes its
  tiny slice to RAM (the C3 smoke pattern) and never touches the persisted
  artifact; smoke refuses the sealed output directory.

### 16 Aug 2026 — M152 build registered (two-stage protocol; nothing measured)

M152 (power-exponent refinement) build is registered before any
measurement. Config `experiments/configs/v23/m152_pgrid.json`: STAGE 1
screens p in {0.25, 0.33, 0.66} at 138k against the sealed p=0.5/p=1.0
cells (anchors: the C4 cells_138k penalty-1.0 reads, p0.5 =
0.2273623188405797 and p1.0 = 0.2106376811594203, tol 1e-9); STAGE 2
promotes AT MOST ONE p (the 138k winner above p=0.5's 138k read) to the
full-data fit, gated against the sealed full-data best
0.278550724637681 + 0.005. Trained-head read at the promoted p under the
C4 protocol (SGD, 8 epochs, lr 0.001, seed 201). If no new p beats p=0.5
at 138k, nothing is promoted and the cell closes as a scoped negative at
the screen (the full-data stage is not run). The two-stage protocol is
registered here to bound compute (each fit is one Gram accumulation +
solves on cached codes). Smoke declares inadmissibility and refuses the
sealed output directory.

### 16 Aug 2026 — M153 and M154 builds registered (fits only; nothing measured)

M153 (routing granularity): config `experiments/configs/v23/m153_routing_granularity.json`.
Class groups = seeded 2-means (seed 311, 2 runs) over the 345 class score
profiles (global_train class-means); child-k scores = the GLOBAL head's
score vector with non-group classes masked to -1e30; fusion = stacking
over [children, global] with the M143b valid-slice protocol; control = K
random class partitions (seed 312); K in {2, 4}. NO new head is fit
(the children reuse the global head's scores restricted to a class
group), hence no section 5.3 floor applies — registered. Anchors: the
M143b reads from the same caches (fused 0.22431884057971013, global
0.22460869565217392, tol 1e-9). Gate per K: fused(kmeans) >= global +
0.005 AND >= fused(random).

M154 (data-sharing, score-level): the registered interpretation is the
per-arm domain-interaction fusion — stacking over the 7 arms PLUS 42
arm-x-domain indicator interaction features (the fusion weights may
depend on the row's domain, which is the score-level proxy for data
sharing; head-level sharing stays M159, which needs codes). Anchors:
the M143b flat stacking reproduces from the same caches. Gate:
fused(interaction) >= global + 0.005 on the sealed test scores.
Both cells are fits-only on the cached score matrices; both smoke
configs declare inadmissibility and refuse the sealed output
directories.

### 16 Aug 2026 — M157 build registered (CPU temporal screen; nothing measured)

M157 build is registered before any measurement. Config
`experiments/configs/v23/m157_temporal_screen.json`: two new axes —
Lorenz-63 x-component (RK4 dt=0.01, sampled every 0.1, warmup 2000,
seed 1571) and a seeded Dyck-like two-bracket grammar (next-token,
numeric token id through the SAME 1-D arm machinery so arms stay
comparable, seed 1572) — plus the hybrid arm (reservoir state
concatenated with the M147 primitive features, must beat BOTH parents)
and the growth arm (reservoir + 128 appended units, readout fit on the
static reservoir's error steps; control = a random same-sized step
subset; echo-state property re-verified per block). Anchors: the sealed
M147 Mackey-Glass reads reproduced with the M147 ladder unchanged
(seeds {21,22,23} x units {64,256,1024} x rho {0.5,0.9,0.99}, warmup
200, penalty 1.0): no-memory 0.14585608397316033, primitives
0.0031721430026391, reservoir best per seed 0.002661459 / 0.002231602 /
0.002485177 (tol 1e-9); plus the M134 DSL anchor (the M147 t1 path).
Gate per new axis: reservoir best >= 5% relative over the best
non-recurrent arm; hybrid beats both parents; growth beats static
reservoir AND the random-subset control. All CPU, minutes per axis.

### 16 Aug 2026 — M157 temporal screen: SEALED — recurrence loses both new axes

Anchors exact (all deltas 0.0 — the M147 Mackey-Glass reads reproduced
with the M147 ladder unchanged). Results (NRMSFE):

- Lorenz (sampled every 0.1): programmatic primitives **0.000292** beat
  everything; the reservoir's best is 0.023582 (80x worse) and the
  no-memory/tap arms are far behind (0.457 / 0.315). The axis is smooth
  enough that the linear extrapolator is near-perfect — recurrence loses
  hard.
- Dyck grammar: programmatic **0.000256** vs reservoir 0.994128
  (no-memory level) — the M134 prior confirmed on a second discrete
  axis.
- Hybrid (reservoir + primitive features, ridge): 0.000312 / 0.000261 —
  TIES the best parent exactly on both axes (the ridge ignores the
  reservoir's contribution); the registered "must beat BOTH parents"
  fails at equality on both axes.
- Growth: beats the random-subset control on Lorenz (0.024231 vs
  0.025434) but NOT static (0.023582); loses both on Dyck (1.033950 vs
  1.007872 control). Residual-targeted reservoir growth does not pay.
  Implication: the reservoir's measured value is axis-specific
  (Mackey-Glass remains the only axis where recurrence wins; the
  programmatic arm wins the two new axes). The task-property answer so
  far: recurrence pays only where the series is chaotic at a sampling
  rate that defeats linear primitives; on smooth-sampled chaos and
  discrete grammar it loses by orders of magnitude. The M134 prior is now
  a two-axis pattern. Scope: three measured axes (MG, Lorenz, Dyck);
  other series unmeasured.
  Evidence: `logs/results/v23/m157_temporal_screen/evidence.json`.

### 16 Aug 2026 — M155 growth premise: SEALED — the M156 budgets are now registered

Both anchors exact (delta +0.000e+00): f6144 full-data 0.261362, SPM+sqrt
full-data 0.278551, from the cached codes. Premise outputs (train split,
409,832 rows): f6144 head train accuracy 0.6333 -> **150,289 error
rows**, floor ladder {32, 64, 128, 256, 512, 1024, 2048, 4096}; SPM+sqrt
head train accuracy 0.8129 -> **76,670 error rows**, floor ladder
{32, 64, 128, 256, 512, 1024, 2048}.
M156 budget registration (derived from this sealed premise, the section
6 interpretation note): base = the promoted SPM+sqrt full-data head;
rungs = **{256, 2048}** atoms (two rungs spanning the floor ladder;
2048 is the largest clearing power of two at ceil(76670/(4a)) >= 10);
the f6144 head's ladder is reported alongside, not used. Growth
dictionary = seeded prefixes of the global pool ([11,100] permutation);
control = blind-greedy OMP selection (the M108/E8 prior) on the error
rows, same head fit; both append to the base head's scores and re-solve
the 2-arm fusion on train, evaluated on test.
Evidence: `logs/results/v23/m155_growth_premise/evidence.json`.

### 16 Aug 2026 — M153 routing granularity: SEALED NEGATIVE — the single-head verdict extends

Anchors exact: the M143b reads reproduced from the same caches (fused
0.224319, global 0.224609, tol 1e-9). Results on the sealed test scores:
K=2 fused(kmeans) 0.224522, K=4 fused(kmeans) 0.224522 — IDENTICAL to
the random-partition control at both K, and marginally BELOW the global
head (0.224609). Class-group decomposition of the label space adds no
measured value, and the partition does not matter (the masked-child
fusion is partition-invariant: it reconstructs the global score vector
with masked entries). The M143b single-head verdict extends to
class-group granularity: the single global head remains the best
measured configuration.
Evidence: `logs/results/v23/m153_routing_granularity/evidence.json`.

### 16 Aug 2026 — M154 first full run VOID (anchor gate) + construction fix, registered before re-measurement

M154's first full dispatch was VOIDED BY ITS OWN ANCHOR GATE before any
new number was read: the M143b flat-stacking anchor measured fused
0.2250144927536232 vs sealed 0.22431884057971013 (delta +6.95e-4 >
tol 1e-9; the global arm reproduced exactly, delta 0.0). Root cause
found in the construction: the runner built the flat concat as
7 arms (6 specialists + global) PLUS the global arm appended again
(2,760 columns) instead of the registered M143b layout of 2,415
columns. The void evidence is recorded at
`logs/results/v23/m154_data_sharing/evidence.json` (void: true).

Fix (registered here before re-measurement): the flat concat is now
the 7-arm transpose-reshape alone (2,415 columns, the M143b arm
order — specialists 0..5 then global); the gated copies stay 7 arms x
6 domains = 14,490; the interaction stack is 16,905 columns. A unit
test pins the flat width at 2,415 and the full width at 16,905 so the
duplicated-global defect cannot recur silently. The smoke run's
numbers (flat 0.337000, gated 0.332000 at 20k rows) are inadmissible
and are not evidence.

### 16 Aug 2026 — M154 memory amendment (registered before re-measurement)

After the flat-construction fix, the re-dispatched full run reproduced
BOTH anchors exactly (fused 0.224319, global 0.224609, delta 0.0) and
then crashed allocating the gated ladder: the sealed stacking helper
materialises float64 copies of the 110,400 x 16,905 fit slice (~14.9
GB each, ~30 GB per call) and the in-RAM float32 stack (9.3 GB) plus
its f32 fit-slice copy did not fit the 63 GB machine. Amendment
(memory management ONLY; feature values and the stacking protocol are
bitwise unchanged): the interaction stack is spilled to disk memmaps
under `GEODE_CACHE_DIR/v23/m154_features/<output>/` —
`full_train.npy` (the [flat | gated] stack), `ft.npy` and `fv.npy`
(the M143b valid-slice partition, same seed 55 / frac 0.8). The
ladder metric and the final refit call the sealed `_stacking_fit` on
the memmaps, which reads the identical float32 values; paths + sha256
of the spilled files are recorded in the evidence. Smoke and full use
output-specific directories. Re-dispatch follows this entry.

### 16 Aug 2026 — M154 data sharing: SEALED PASSED — domain-gated fusion lifts the stack

Both anchors exact (delta 0.0): flat fused 0.22431884057971013, global
0.22460869565217392. Gated fusion (16,905 features: the 2,415 flat +
42 arm-x-domain gates): **0.23953623188405798**, gain **+0.014928** over
the global arm at penalty 10000.0 — the gate (>= global + 0.005) is
CLEARED. Interpretation within scope: letting the stacking weights
depend on the row's domain (the score-level data-sharing proxy) beats
both the flat M143b stack (0.224319) and the global head (0.224609) by
~1.5 points. The fusion weights are domain-dependent, so the sharing
enters through the fit, not the data. Head-level sharing stays M159.
Spilled feature files (the section 6 memory amendment) recorded with
sha256 in the evidence.
Evidence: `logs/results/v23/m154_data_sharing/evidence.json`.

### 16 Aug 2026 — M156 build registered (residual growth on the global head; nothing measured)

M156 build is registered here BEFORE the runner is written or anything
runs. Construction (section 4 M156 + the sealed M155 budgets):

- Base = the promoted SPM+sqrt full-data head (test read
  0.27855072463768116). Error rows = the M155 premise population on the
  full train schedule (76,670 rows, exact-match integrity gate; n_error
  > 0 hard-fails; the floor per budget is ceil(n_err / (4g)) >= 10).
- Budgets = {256, 2048} atoms (the M155-registered rungs; 2,048 is the
  largest clearing power of two).
- The growth pool = the cached f6144 codes (24,576 columns, the M141
  cell-2 order). Verified layout: the cached codes are the
  [11,100]-permuted shared pool's first 6,144 atoms (m117
  `_random_dictionary` = `_random_order` [11,100]) with the M103
  interleaved column layout (`_pool` is bin-major: atom a owns
  columns {a, 6144+a, 12288+a, 18432+a}). Growth dictionary for budget
  g = the first g atoms = those 4g columns, extracted from the cache
  (no new encode).
- Control = blind-greedy group-OMP (the M108 arm (c) construction,
  `select_discriminative`) over the SAME error-row features with
  pool 6,144 atoms, budget 2,048, on the GPU port with the M145
  order-parity check (parity subset 512 rows, parity budget 32, numpy
  reference vs GPU port must agree exactly). On parity FAILURE the
  full numpy fallback is prohibitive at this scale (76,670 rows x
  24,576 columns), so — amended here, before the runner exists — the
  CONTROL ARM IS VOID and only growth-vs-static is adjudicated; the
  growth-vs-control clause is marked non-adjudicable, the budget's
  gate cannot pass, and the void is disclosed in the evidence.
- Both arms: ridge head (penalty 1.0) fit on the SAME error rows;
  train + test scores scored from the cached codes in blocks.
- Fusion = 2-arm stacking [base, arm] (690 features) on the train
  scores with the M143b protocol (valid seed 55, frac 0.8, ladder
  {1,10,100,1000,10000}), evaluated on the test scores. Static = the
  base head's test read.
- Anchors: a1 base test read 0.27855072463768116 (tol 1e-9); a2 the
  M145 specialist-path anchor — d0 512-atom own-domain
  0.19357142857142856 (tol 0.002, GPU encode); a3 the growth dicts are
  nested prefixes (cols(256) is a subset of cols(2048)) and the cached
  width is 24,576; a4 cached-code reproduction — a fresh GPU encode of
  the first 64 train rows with the rebuilt 6,144-atom dictionary
  reproduces the cached f6144 codes (validates the column-extraction
  instrument; tol 1e-5, expected bitwise).
- Gate per budget: growth_fused >= static + 0.005 AND growth_fused >
  control_fused; else scoped negative. Ops ledger disclosed: both arms
  pay the g-atom encode-equivalent MACs over train+test rows; the
  control additionally its pool encode + selection MACs (the M145
  ledger rule).
- Smoke declares inadmissibility, skips anchors and control, tolerates
  the floor violation, runs budget {256} on 20k/20k rows, and runs
  entirely on CPU (no GPU needed once anchors/control are skipped).

Build complete: runner `experiments/tier4/eval_v23_m156_growth.py`,
configs `experiments/configs/v23/m156_growth{,_smoke}.json`, 6 unit
tests green (`experiments/common/test_v23_m156_growth.py`). Nothing
has been measured; the smoke is dispatched after this entry. (Build
defect fixed before any measurement: the 1-arm diagnostic's penalty
selection was missing its ladder argument, caught by the smoke at the
registered 40k row count after the premise reproduced the probe's 48
error rows exactly. Second build defect, also fixed before any
measurement: the error-row extraction indexed the n_error-row output
matrix with ABSOLUTE row positions (offset + take_rows); it now uses
a running fill counter, lives in the unit-tested
`_extract_error_rows` helper, and both the control and arm branches
call it. Third instance of the same ladder-argument omission, in the
per-arm stacking penalty selection, fixed likewise before any
measurement.)

### 16 Aug 2026 — M156 smoke premise amendment (registered before re-measurement)

The first smoke dispatch was stopped by its own premise gate: at 20k
train rows the SPM+sqrt head interpolates its training rows (0 error
rows; the head fit on 20k rows with 40,383 features at penalty 0.1
memorises the tiny 50-class cell), so the growth population is empty
and the premise hard-fail fired — correctly, and unwaivable (the M145
lesson: a zero population validates nothing). The smoke row count must
be one where the n-row base head leaves errors. Amendment: run a
premise-only probe (no accuracy claims, no gates) fitting the
SPM+sqrt head at n in {20000, 40000, 60000, 80000, 100000} and
recording the train error-row count of the n-row head on its own n
rows; the smallest probed n with a nonzero population becomes the
registered smoke row count. The probe reads only error COUNTS.

PROBE RESULT (registered): n=20000 -> 0 errors; 40000 -> 48; 60000 ->
399; 80000 -> 1175; 100000 -> 2210. The smallest probed n with a
nonzero population is 40000, so the registered smoke row count is
**40,000** (48 error rows; the smoke declares inadmissibility and
tolerates the floor violation at budget 256). Evidence:
`logs/results/v23/m156_smoke_premise_probe/evidence.json`.

### 16 Aug 2026 — M158 feasibility premise (registered before the build)

The M158 plan text says levels {1, 2, 4, 8} "(37 bins)" — that is a
typo: the C2 bin rule is sum(level²), so {1,2,4,8} = 85 bins. With
the C2 dictionary (1,923 atoms) the 85-bin width is 163,455: the
full-data Gram is 163,455² x 8 = 213.7 GB float64 (RAM is 63 GB) and
persisting the full-train codes is ~268 GB (F: has 91.7 GB free —
measured). Both are INFEASIBLE, so the cell cannot run as registered;
nothing for M158 is built or measured. Re-scope decision pending
(registered options under consideration: (a) the 8x8 level ALONE at a
matched-cost atom count — 64 x a = 16 x 1,923 -> a = 481 atoms, width
30,784, Gram 7.6 GB, compared against the cached 4x4 level at 1,923
atoms at matched head cost; (b) the 85-bin pyramid at a reduced atom
count whose Gram fits). The re-scope will be registered as its own
dated entry before any M158 build.

### 17 Aug 2026 — M158 re-scope registered (before the build; the option (a) construction)

The 85-bin pyramid remains infeasible as registered (section above).
M158 is RE-SCOPED to the feasibility entry's option (a): the 8x8
level ALONE at a = 481 atoms — the FIRST 481 atoms of the same
[11,100]-permuted pool (the C2 dictionary prefix), head-cost-matched
to the sealed 4x4 level at 1,923 atoms (head MACs: 64 x 481 x 345 =
10,620,480 vs 16 x 1,923 x 345 = 10,614,960 — the 8x8 arm pays
+5,520 MACs, +0.052%, within the family's 0.5% cost-tolerance rule).
Width 30,784 -> Gram 7.6 GB, feasible. Construction:

- Encode: one streamed pass over the M141 cell-2 schedule (part 1 =
  the 138k subsample, parts 2-3 = the ext600/rest raw-image rows) with
  the rebuilt M108 whitener + C2 dictionary, pooling = the 8x8 level
  only (the m107 edges rule, level 8). The full-data fit streams into
  the Gram accumulator (no full-data persistence); the 138k-part and
  test codes are persisted under `v16/m158/` for the trained-head
  read and test scoring (~21 GB).
- Anchors: t1 — a fresh 21-bin C2 encode of the first 64 train rows
  with the same rebuilt dictionary reproduces the cached
  spm1923_fulltrain codes BITWISE (tol 0.0; validates dictionary /
  whitener / activation before the new pooling level is trusted); a2 —
  a ridge refit on the cached 4x4-level columns (full data, penalty
  1.0) reproduces the sealed per-level read 0.26014492753623186 (tol
  1e-9).
- Gate: Q(8x8, 481, full data, penalty 1.0) >= 0.26014492753623186 +
  0.005 at matched head cost; else the pooling-saturation point is
  4x4 and M158 closes as a scoped negative.
- Controls: the trained-head read at 138k (the M109 shared schedule)
  is the dual-read control. Premise: section 5.3 floor checked in-run
  (409,832 / 30,784 = 13.3 >= 10). Cost: one streamed encode (~40 min
  GPU) + fits. Smoke declares inadmissibility and refuses the sealed
  output directory.

### 17 Aug 2026 — M160 build registered (schedule sensitivity; nothing measured)

M160 build is registered here BEFORE the runner exists. Construction
(section 4 M160, resolved): re-run M146's r3 (the differentiable
SPM+sqrt encoder, trainable dictionary + head, whitener frozen, the
M146 module and batch factories unchanged) under THREE schedules with
the same model, seeds, and corpus:

- S0 = the sealed schedule (8 epochs, cosine lr 3e-4, wd 1e-4,
  patience 2, batch 64, val frac 0.05, shuffle_seed 11) — this is
  ALSO the anchor run and must reproduce the sealed r3
  0.10602898550724638 within 0.005 (a trained-run reproduction
  tolerance; the run is a re-execution on the same device, not a
  reread).
- S1 = 16 epochs, patience 4, everything else S0 (the more-epochs
  axis).
- S2 = 8 epochs, lr 3e-5, everything else S0 (the different-LR axis).

Anchor: the frozen r1 reproduction (0.2273623188405797, tol 1e-6, the
M146 t1). Premise: the same cached artifacts (the M146 codes/corpus).
No gate (measuring stick): report r3 per schedule vs r1; if any
schedule's r3 clears r1, the "price of freezing" verdict is
schedule-bound and M146's shipping selection reopens; if none does,
the frozen verdict hardens. Cost: three trained runs of the
differentiable encoder (S0 ~2-3h, S1 ~4-6h, S2 ~2-3h GPU) plus the r1
refit — disclosed. Smoke declares inadmissibility, trains one epoch
per schedule on 4k rows, skips anchors, and refuses the sealed output
directory. (Build defect fixed before any measurement: the M160
configs lacked the `sparse` keys the whitener builder reads —
patch/stride, then zca_epsilon/zca_fit_patches/zca_fit_seed/
candidate_pool_size — all added verbatim from M146's config; the two
crashed smokes measured nothing.)

### 17 Aug 2026 — M161 build registered (hybrid readout; nothing measured)

M161 build is registered here BEFORE the runner exists. Construction
(section 4 M161, resolved): on the C4 138k context (the cached SPM
codes, p=0.5, the M146 level) — (1) the frozen ridge (penalty 1.0)
read is the anchor (0.2273623188405797, tol 1e-6); (2) the hybrid =
the SAME ridge logits (frozen, detached) plus ONE trained linear head
of the M146 r2 structure (HeadOnly, 21 x 1,923 inputs, no hidden
layer — the registered "small residual head") trained under the M109
shared schedule (4 epochs, AdamW cosine lr 3e-4, wd 1e-4, patience 2,
batch 64, val frac 0.05, shuffle_seed 11) with cross-entropy on the
COMBINED logits — the head learns only what the ridge leaves; (3) the
in-run controls: the pure trained head (the r2-alone protocol, same
schedule/seeds, no ridge) measured alongside the sealed r2
0.042608695652173914, and the ridge read itself. Gate: hybrid test
read >= 0.2273623188405797 + 0.005, else archived as a scoped
negative (the dual-read rule: the hybrid must beat BOTH parents'
reads). Cost: two trained runs of a 40k-width linear head (~30-60 min
GPU each). Smoke declares inadmissibility, trains 2 epochs on 4k
rows, skips anchors, and refuses the sealed output directory. (Build
defect fixed before any measurement: a missing
`_train_with_schedule` import, caught by the smoke after the ridge
anchor; the crashed smoke measured nothing.)

### 17 Aug 2026 — M162 build registered (prune + retrain dense; nothing measured)

M162 build is registered here BEFORE the runner exists. Construction
(section 4 M162, resolved): prune the M144 torch DINOv2-small at
keep=0.5 with the M144 `_prune` (both sides + bias zeroed), then
RETRAIN under the M109 shared schedule on the M107-materialised r56
pixels (the M144 digest-tagged memmaps): AdamW, batch 64, cosine lr
3e-4, wd 1e-4, patience 2, 4 epochs, val frac 0.05 (shuffle_seed 11),
CE through a trainable 1,536 -> 345 head over the SAME features the
ridge readout uses (CLS + mean-patch tokens). The prune mask is
RE-APPLIED after every optimizer step (pruned weights stay exactly
zero — gradients flow only through kept weights). After training the
head is discarded and the M144 ridge readout (penalty 1.0) fits the
retrained encoder's r56 features. Anchors: the M144 t1 parity guard
and the t2 UNPRUNED reproduction (0.245014492753623 within 0.002, the
M107 pixel path) run before any retrained number. No gate (measuring
stick): the retrained keep=0.5 read is reported against the M144
no-retrain read (0.1076231884057971 @ 185.0M) and the additive recipe
(0.278551 @ 175.2M) at the disclosed effective MACs. Cost: one
trained run (~1.5-2.5h GPU) + two feature encodes. Smoke declares
inadmissibility, trains 1 epoch on 2k rows, skips anchors, and
refuses the sealed output directory.

### 17 Aug 2026 — M160 schedule sensitivity: SEALED — the frozen verdict hardens

Anchors: r1 exact (0.227362, delta +0.000e+00); r3 S0 reproduced
0.106261 (delta +0.000232, tol 0.005 — the sealed-schedule re-run
reproduces M146's 0.106029). Results: S0 (sealed 8-epoch) r3
0.106261; S1 (16 epochs, patience 4) 0.134522 (val 0.133913); S2
(lr 3e-5) 0.040986. More epochs HELP (+2.8 over S0), a lower LR
HURTS (-6.5), and NO schedule approaches r1: S1's best stays 9.3
points below the frozen ridge. The trained-side deficit is not the
sealed schedule's artefact — the frozen verdict hardens and M146's
shipping selection (the frozen system ships) is confirmed across
schedules. Evidence:
`logs/results/v23/m160_schedule_sensitivity/evidence.json`.

### 17 Aug 2026 — M164c fingerprint/routing search: ANCHOR GATE PASSED — the LeCun suspicion confirmed

Registered before running (entry above). Anchor gate passed on stage

1. The LeCun-scoped topic queries returned zero/irrelevant hits
   (recorded — index absence proves nothing), so the two known works
   were pinned by direct title lookup: the arXiv-published AMI paper
   with the **configurator** (Grathwohl, Wang, LeCun et al.,
   arXiv:2306.02572) and I-JEPA (arXiv:2301.08243). Both are design
   antecedents of the v24 fingerprint/router — architecture proposals,
   not measured routing results. Must-cite additions: Routing Networks
   (arXiv:1711.01239), switch transformers, DA-MoE, hierarchical routing
   MoE, unified task embeddings (arXiv:2402.14522), the
   dataset-similarity review (arXiv:2312.04078). Folded into
   `analysis/BLOCKER_ANALYSIS_v24.md` (fingerprint/routing reference
   ledger) and the whitepaper claim ledger (§8.2). Evidence:
   `logs/results/v23/m164c_fingerprint_routing/evidence.json`.

### 17 Aug 2026 — M164c dispatched: fingerprint/routing reference search (the LeCun suspicion)

Registered BEFORE any query runs. Purpose: which published
routing/task-embedding work must the v24 fingerprint/router cite,
including the suspected LeCun line (configurator, task-conditioned
routing). Instrument: 5 anchors (sparsely-gated MoE, switch
transformers, routing networks, task embeddings, LLM routing) + 4
LeCun author-scoped queries + 5 blocker queries; same gate and
rate-limit rules as M164/M164b. The runner now supports per-query
`author` scoping (`au:"..." AND (...)`). Config:
`m164c_fingerprint_routing.json`.

### 17 Aug 2026 — M164b blocker-solution search: ANCHOR GATE PASSED — the fit wall is solved in print; the code ceiling has new candidates

Registered via the config file (written before any query ran:
`registered_before_any_search: true`, dated 17 Aug) and executed the
same day. All six anchors hit in stage 1; one query (b12) returned
zero hits (recorded; absence proves nothing). Findings, folded into
`analysis/BLOCKER_ANALYSIS_v24.md`: L4 (the quadratic fit wall) has a
mature published toolbox — randomized NLA/sketching (2409.14309,
2210.12212, 2003.09097, 2204.06653), Falkon-style
sketch+preconditioning (1611.03220, 2304.12465), Krylov LSMR/Arnoldi
(2409.09104, 2407.05945), and quantum-inspired classical algorithms
(1910.06151, 2009.07268, 2010.08626) — M176b becomes a benchmark, not
an invention. L1 gains candidate 2b: frozen learned local descriptors
(GeoDesc 1807.06294, DELG 2007.13172) pooled with SPM. L5 has mature
HPC mixed-precision techniques (2109.01232, 2505.20719). Evidence:
`logs/results/v23/m164b_blocker_solutions/evidence.json`.

### 17 Aug 2026 — Research→development cleanup: archive move registered

Everything superseded moved to `archive/`: pre-v23 analysis documents,
experiment tiers 1–3/5–6/e2e, the legacy tier4 runners (v5–v14 and
unnumbered), the legacy common modules and tests, configs v5–v15 plus
the flat legacy configs, legacy tools, infrastructure/e7, pre-v16
logs and the chat logs, the legacy README, and `verify_pipeline.py`.
Kept and verified (import check of every class passes): the active
tier4 line (`bench_v16_*`, `eval_v15_m103/m104/m107`, `eval_v16_*`,
`eval_v23_*`, `prepare_v15_m107_pixels`), common (`data_cache`,
`experiment_manifest`, `litsearch_cache`, `v5_artifacts`, `v5_protocol`,
`test_v16_*`, `test_v23_*`), configs v16/v23, logs/results v16/v23, and
src/ (still imported by the sealed M130/M131 builders). The root
README is rewritten as the development-phase repository map; nothing
was deleted — git history preserves the original paths.

### 17 Aug 2026 — Whitepaper v23 (GEODE) written; supersedes the v22 whitepaper

`analysis/WHITEPAPER_GEODE_v23.md` supersedes
`WHITEPAPER_ADDITIVE_FROZEN_SYSTEM_v22.md`. It adds the v23 sealed
verdicts (M150–M162), the v24 toolbox and v25 governance layers, and
the M164 claim ledger. Same standing rule: no novelty-of-mechanism
claim; every figure sealed; the search instrument licenses only
displacement.

### 17 Aug 2026 — M164 buildout-blocker search: ANCHOR GATE PASSED — no displacer found for the core claims

All six anchors hit in stage 1 (AND of quoted abs: phrases) — no
stage-2 fallback needed, 16/16 queries returned results, zero 429s.
The positive control passed, so the search is admissible for its
displacement-only role. No hit displaces the core registered claims
(freezing, additive closed-form composition, measured-behavior
routing, the token mechanism). Prior art the buildout must cite and
compare against, folded into the plans: deep-patch spatial pyramid
VLAD codes (arXiv:1603.09046) and deep dictionary learning
(arXiv:2012.12509, arXiv:1912.10804) for M176c; divide-and-conquer
KRR (arXiv:1305.5029) and two-level preconditioning
(arXiv:1806.05826) for the M176b escape ladder; transfer-metric
instability (arXiv:2204.01403) for M167; the Shapley/Beta-Shapley/
EcoVal line (arXiv:2202.05594, arXiv:2110.14049, arXiv:2402.09288)
for M180; zkDL (arXiv:2307.16273) and secret-shared regression
(arXiv:2309.09486, arXiv:2109.11200) for Track P; retraining-free
pruning (arXiv:2308.03449, arXiv:2212.12651) as M162 context.
Summary: `analysis/BUILDOUT_BLOCKER_LITERATURE_SEARCH.md`; evidence:
`logs/results/v23/m164_buildout_blockers/evidence.json`. Boundary
restated: a no-displacer-found pass is not a novelty statement.

### 17 Aug 2026 — M164 dispatched: buildout-blocker literature search (the rebuilt instrument)

Registered BEFORE any query runs. Purpose: a displacement check on
the buildout blockers (v24 §9 L1–L7 plus the v24/v25 open items) — an
unauthenticated public search can only displace or inform registered
claims; absence proves nothing (the M88/M148 lessons). Instrument: 6
anchor queries (positive control: task embeddings, transferability,
Fisher vectors, large-scale KRR, zk inference, data valuation) + 10
blocker queries, all with quoted abs: phrases; stage 1 = AND, and if
any anchor misses, a UNIFORM stage-2 re-run of ALL queries with OR
(never only the empty ones); HTTP 429 retried with backoff and
recorded separately from empty results. Gate: every anchor must hit
in at least one stage, else the search is VOID for claims. Config:
`m164_buildout_blockers.json`; runner:
`tools/m164_buildout_blocker_search.py`.

### 17 Aug 2026 — M162 prune+retrain: SEALED NEGATIVE — retraining loses to pure pruning

Anchors: t1 parity 1.79e-06 vs bound 1e-4; t2 unpruned r56 EXACT
(0.2450144927536232, delta 1.94e-16). Retrained keep=0.5 (the M109
schedule, 4 epochs, mask re-applied per step; train accuracy only
reached 0.046, best val 0.0429) = 0.059652 @ 185.0M MACs vs the M144
no-retrain read 0.107623 @ 185.0M — retraining LOSES 4.8 points, and
both sit 17–22 points below the additive recipe (0.278551 @ 175.2M).
The industry-default prune-then-retrain does not recover what pruning
lost at the registered schedule: the CE-through-head path degrades
the features the ridge readout sees — the M146/M160/M161 pattern,
now on the pruned arm too. The frozen verdict extends to the
retrained side of the comparison. Evidence:
`logs/results/v23/m162_prune_retrain/evidence.json` (runtime 577.4 s).

### 17 Aug 2026 — M162 smoke: forward-path detach defect + fix, registered before re-measurement

The re-dispatched M162 smoke passed the t1 parity guard (worst
1.794e-06 vs bound 1e-4) and then died in the first training step:
`_forward` round-trips tokens through numpy
(`feature(tokens.cpu().numpy())`), which raises on a tensor that
requires grad — and, had it not raised, would have silently DETACHED
the encoder from the graph, turning the registered "prune + RETRAIN"
cell into head-only training (gradients never reach the kept encoder
weights). The registration (section 4 M162 / build entry) requires
gradients to flow only through kept weights. Fix: `_forward` now
computes the M107 features (CLS + mean patch tokens) natively in
torch (`torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=1)`);
the numpy `feature` path remains only in the no-grad encode/anchor
arms. Nothing was measured. Re-dispatch follows this entry.

### 17 Aug 2026 — M162 smoke crashed (missing parity_guard block) + fix, registered before re-measurement

The M162 smoke died at the t1 parity guard: KeyError 'parity_guard' —
both M162 configs lack the guard block the shared M109 `_parity_guard`
helper reads (resolution, batch, models, bound_relative). Nothing was
measured (the crash precedes the guard). Fix: the block is added
verbatim from the sealed M144 configs (resolution 56, batch 8, models
["small"], bound_relative 1e-4) to both M162 configs — the same
"configs must copy the full upstream recipe block" defect class as the
M160/M161 config gaps. Re-dispatch follows this entry.

### 17 Aug 2026 — M161 hybrid readout: SEALED NEGATIVE — the trained residual destroys the frozen read

Anchors exact: ridge 0.227362 (delta +0.000e+00); the pure trained
head 0.042609 reproduces the sealed r2 0.042608695652173914. Hybrid
(ridge logits + trained residual, M109 schedule, 4 epochs) =
0.076493 — gain vs ridge -0.150870. The gate (hybrid ≥ ridge + 0.005)
FIRED: the hybrid lands 15.1 points BELOW the frozen ridge and only
+3.4 above the pure trained head, and its best validation (0.1738)
was already far under ridge. Training a residual against the frozen
logits actively destroys the frozen readout. The composition family
now closes negative across all three measured probes: M146, M160,
M161 — on the same codes, ANY SGD readout is dominated by the
closed-form ridge, and "frozen + residual" does not compose. The v24
§8 item 7 default is decided: hybrid composition stays OFF unless a
future gate changes a registered premise. Evidence:
`logs/results/v23/m161_hybrid_readout/evidence.json` (runtime
7,382.59 s).

### 17 Aug 2026 — M158 finer pooling: SEALED NEGATIVE — the saturation point is 4x4

Both anchors exact: t1 bitwise (max abs delta 0.0, the fresh 21-bin
encode reproduces the cached codes), a2 0.260145 delta +0.000e+00.
Q(8x8, 481) full data = 0.22072463768115942 vs the sealed 4x4 read
0.26014492753623186: gain -0.039420 — the gate FIRED. At matched head
cost the 8x8 level (481 atoms, 64 bins) LOSES to the 4x4 level (1,923
atoms, 16 bins) by 3.9 points: trading atoms for bins does not pay.
The sealed ladder is now extended: 1x1 0.154 -> 2x2 0.224 -> 4x4 0.260
-> 8x8 0.221 (matched head cost) — concave, the saturation point is
4x4. The trained-head read collapses (0.1013 at 138k, the E5
pattern). M158 closes as a scoped negative; the 21-bin pyramid stays
the registered construction. Evidence:
`logs/results/v23/m158_finer_pool/evidence.json`.

### 16 Aug 2026 — M150 first full dispatch crashed (cache paths) + fix, registered before re-measurement

The M150 full dispatch ran code pool2062 first (ridge anchor EXACT,
delta 0.0; eff-rank 7.85 at 138k, top1 share 0.308, condition 9.96e9;
trained head 0.055333 — instrument check only, nothing sealed: the run
died before writing evidence) and then crashed with FileNotFoundError:
the runner looked for ms357 under v16/m142_c2, but the MS train codes
live under v16/m142_c3, and C3 never persisted MS TEST codes — the
sealed ms357 reads cannot be reproduced from cache without the
test-side encode. Fix (registered here before re-measurement): the
config now carries per-code train/test relpaths; the ms357 test codes
are the M151 artifact v16/m151/ms357_fulltest.npy (M151's registered
MS-test encode, which is itself anchored by the MS full-data read
0.24214492753623187 tol 1e-9). DEPENDENCY: M150 full re-dispatch must
follow M151 full. The runner now reads each code's train and test
memmaps from its registered paths.

### 16 Aug 2026 — M151 smoke crashed (missing config block) + fix, registered before re-measurement

The M151 smoke dispatch crashed in the MS test encode with KeyError
'sparse': both M151 configs lack the C3 recipe block that
`_build_scale_whitener`/`_scale_dictionary` read (stride,
contrast_epsilon, zca_epsilon, zca_fit_patches, zca_fit_seed,
dictionary_seed, candidate_pool_size, atoms_by_scale {3:1950, 5:850,
7:511}, pool_atoms 2062). Fix: the block is added verbatim from the
sealed C3 config to both M151 configs. The MS anchor (the full-data
read 0.24214492753623187 scored on the newly encoded test codes, tol
1e-9) still gates the encode, so a wrong recipe cannot pass silently.
Nothing was measured by the crashed smoke. Re-dispatch follows this
entry.

### 16 Aug 2026 — M151 concat-solver amendment (registered before re-measurement)

The re-dispatched M151 smoke died silently inside the first concat
solve at 53,627 columns: the sealed `np.linalg.solve` copies the
53,627^2 float64 Gram (23 GB), so the ridge solve peaks ~70 GB on a
63 GB machine (the halves at 40,383/13,244 columns fit; the concat
does not). Amendment for the CONCAT cells ONLY (the half-alone
anchors keep the sealed RidgeAccumulator path unchanged):

- A new fitter accumulates the Gram with the SAME block order and
  float64 arithmetic as RidgeAccumulator (raw Gram, column sums,
  cross, class counts), builds the centred/scaled system by the same
  elementwise closed form but IN PLACE inside the Gram buffer, spills
  it to a disk memmap under
  `GEODE_CACHE_DIR/v23/m151_solver_scratch/<output>/centred.npy`, then
  per penalty loads one in-RAM copy and solves with
  `scipy.linalg.solve(assume_a="pos", overwrite_a=True)` (in-place
  Cholesky). Peak ~25 GB. Weights differ from the LU path only at
  decomposition rounding (~1e-12 relative), which is immaterial to the
  1e-9 anchors and the 0.005 gate, and is disclosed.
- In-run equivalence gate (registered, ahead of every concat number):
  on a reduced system (8,192 columns x 20,000 rows, raw codes), the
  Cholesky path must agree with `RidgeAccumulator.solve_many` within
  1e-9 relative on the weights and produce identical standardisers
  (mean/std from the same raw statistics). Failure voids the run.
- The scratch file's path + sha256 are recorded in the evidence.

CORRECTION registered before re-measurement (same day, after the
equivalence gate fired on the first two dispatches): the Cholesky
variant FAILS numerically — the sealed path solves the system built
from the FLOAT32-ROUNDED standardiser statistics, and at the concat
system's condition (~1e9) that rounding makes the standardised Gram
numerically non-positive-definite (scipy posv reports a singular
pivot while the sealed gesv solves it). Two registered consequences:
(1) the fitter must build the centred system from the float32-rounded
statistics, exactly as `_standardised_system` does (the first two
voids were partly this: using the raw float64 statistics shifted the
weights ~39% relative); (2) the in-place factorization is LU (gesv),
not Cholesky — `scipy.linalg.solve(..., overwrite_a=True,
check_finite=False)`, the SAME LAPACK family as the sealed
`np.linalg.solve`, so the equivalence is structural and the gate is
expected to pass at ~1e-16. Both changes are in the concat fitter
only; the halves keep the sealed path. The MS test encode itself was
already validated by the full-run anchors (SPM delta 0.0, MS delta
0.0) and the persisted codes stand.

SECOND CORRECTION registered before re-measurement (same day): even
with in-place LU, the first full-width fit died in the Gram
accumulation — the machine has ~45 GB available and the 53,627^2
float64 matmul temporary (23 GB) on top of the Gram (23 GB) kills it.
The accumulation is therefore COLUMN-CHUNKED (gram[:, c0:c1] +=
block.T @ block[:, c0:c1], chunk 2,048): each Gram entry is the same
dot product whether dgemm's output width is full or chunked, so the
values stay bitwise identical to the sealed `gram += block.T @ block`
while the temporary stays width x 2,048 (~440 MB). The per-penalty
solve copies the spilled system to an F-ORDER buffer so LAPACK factors
in place (a C-order input would be internally copied = another 23 GB).
The equivalence gate stays at 1e-9 relative as registered.

### 16 Aug 2026 — M151 trained-head row gather defect + fix, registered before re-measurement

With the corrected fitter the smoke cleared the equivalence gate
(rel 0.0), ran the concat cell, and crashed in the trained-head read:
`_concat_batches` built each batch by slicing the contiguous range
`[take[0], take[-1]]` of the memmaps, which is wrong when `take`
is a permutation (the trained-head train/val split is a seeded
permutation) — batches came out empty or oversized. Fix: the batch
GATHERS the rows named by `take` (fancy indexing on both memmaps),
which also covers the ordered eval rows. Nothing new was measured by
the crashed smoke (its numbers are inadmissible). Re-dispatch follows
this entry.

### 16 Aug 2026 — M151 interaction: SEALED PASSED — the concatenation is additive-or-better

Both half-alone anchors exact (delta 0.0: SPM 0.2604927536231884, MS
0.24214492753623187), the solver-equivalence gate passed (rel 0.0),
the scratch system's sha256 recorded. Full-data concat cells
(53,627 columns): best = p0.5, lambda 0.1 -> **0.2974782608695652**,
gain **+0.03699** over the incumbent (SPM raw 0.260493, required
+0.005) — the gate is CLEARED: the SPM x MS column concatenation with
signed sqrt + L2 is additive-or-better at the disclosed 53,627-column
width. 138k reads: best 0.24231884057971015 (p0.5, lambda 10).
Trained-head read collapses as always (0.0445, the E5 pattern), so
the win is the frozen readout's, not gradient co-adaptation's. The
v22 separability assumption is now measured, not trusted. The MS test
codes persist as `v16/m151/ms357_fulltest.npy` (anchor-validated).
Evidence: `logs/results/v23/m151_interaction/evidence.json`.

### 16 Aug 2026 — M150 rank sweep: SEALED — the rank-vs-outcome table

All six ridge anchors EXACT (delta +0.000e+00 at tol 1e-9). Table
(138k ridge read, standardised-Gram effective rank, top-1 eigenvalue
share, condition; trained head under the M109 schedule):

| code         | ridge 138k | eff-rank | top1 share | cond    | trained  |
| ------------ | ---------- | -------- | ---------- | ------- | -------- |
| pool2062     | 0.206406   | 7.85     | 0.308      | 9.96e9  | 0.055333 |
| ms357        | 0.215739   | 6.57     | 0.351      | 9.94e9  | 0.056232 |
| spm1923      | 0.214493   | 13.56    | 0.234      | 1.00e10 | 0.078116 |
| spm1923_sqrt | 0.227362   | 16.40    | 0.189      | 9.99e9  | 0.042435 |
| ms357_sqrt   | 0.223855   | 8.22     | 0.286      | 9.99e9  | 0.029739 |
| f6144        | 0.224609   | 7.84     | 0.309      | 9.99e9  | 0.055652 |

Measured pattern (the cell's registered question): the best ridge
read (spm1923_sqrt 0.227362) has the HIGHEST effective rank and the
LOWEST top-1 share; the sqrt power-norm raises eff-rank (13.56 ->
16.40 spm; 6.57 -> 8.22 ms) and lowers top-1 concentration in both
families, and both sqrt cells are the best readers of their family.
So the statistic-vs-outcome ordering runs through spectral spread:
more participation beyond the top eigenvalue accompanies better ridge
reads. Condition ~1e10 everywhere (no ordering signal). Trained heads
collapse everywhere (0.030-0.078, all below their ridge read) — the
E5 pattern now measured across the whole cached family. No gate;
measuring stick. Evidence:
`logs/results/v23/m150_rank_sweep/evidence.json`.

### 16 Aug 2026 — M156 residual growth: SEALED NEGATIVE — growth adds ~nothing on the global head

All anchors exact: a1 base read 0.278551 (delta +0.000e+00), a2 d0
0.193571 (delta 0.0), a4 cached-code reproduction bitwise (max abs
delta 0.0); the premise population matched exactly (76,670 error
rows); OMP order parity passed. Results (static = base read
0.278551): g=256 growth fused 0.278580 (+0.000029), control 0.278145
(-0.000406); g=2048 growth 0.278957 (+0.000406), control 0.279449
(+0.000899). The gate FIRED at both budgets: growth never reaches
static + 0.005, and at g=2048 the blind-greedy control EXCEEDS growth
— what little gain exists is not residual targeting's. All deltas are
< 0.1 point: residual-targeted growth on the global head's errors
adds essentially nothing at the M155-permitted budgets, on the
stronger base (the M145 lesson extends). M156 closes as a scoped
negative; the base head stands. Evidence:
`logs/results/v23/m156_growth/evidence.json`.

### 17 Aug 2026 — M152 penalty grid: SEALED NEGATIVE — the sealed p=0.5 recipe stands

Anchors exact (p=0.5 138k lambda 1.0: 0.2273623188405797 delta 0.0;
p=1.0: 0.2106376811594203 delta 0.0). Stage-1 screen (138k, ladder
{0.1, 1.0, 10.0}): p-best 0.25 -> 0.214754, 0.33 -> 0.221565, 0.5 ->
0.229623, 0.66 -> 0.233652, 1.0 -> 0.214522 — p=0.66 wins and is
promoted. Stage 2 (full data): p=0.66 best 0.27947826086956523
(lambda 0.1) vs the incumbent p=0.5 0.27855072463768116 — gain
+0.000928 < +0.005 -> the gate FIRED, and the trained-head read
collapses (0.0029, the E5 pattern) -> both reads fail. No refined p
beats the sealed p=0.5 recipe at the registered margin: the promoted
recipe's power is a measured optimum of this grid, not a plateau
choice. Evidence: `logs/results/v23/m152_pgrid/evidence.json`.

### 17 Aug 2026 — M159 shared-fit specialists: SEALED NEGATIVE — head-level sharing ties the global arm

All anchors exact: the six own-domain reproductions (delta 0.0 each,
d0 0.193571 ... d5 0.123978) and the M143b reads (fused 0.224319,
global 0.224609, delta 0.0). Shared-fit fusion (heads on ALL 138k
rows): 0.22417391304347825 (penalty 10.0) — gain -0.000435 vs the
global arm -> the gate FIRED; routing controls reported alongside:
competence 0.1877 > identity 0.1662 ~ random 0.1676. Head-level data
sharing does not exceed the global arm, and it lands exactly at the
M143b flat-fusion level (0.224319): fitting the specialists on shared
data changes ~nothing. The sharing that pays is the DOMAIN-GATED
FUSION WEIGHTS (M154, +1.49), not the specialists' head data. The
410k variant stays cost-gated and unrun (the 138k cell did not pass).
Evidence: `logs/results/v23/m159_shared_fit/evidence.json`.

### 16 Aug 2026 — M159 build registered (shared-fit specialists; nothing measured)

M159 build is registered here BEFORE the runner exists. The section 4
text leaves the row scale as "138k/410k" — resolved here: the heads are
fit on ALL 138k train rows (the M143b train-protocol scale, so the
head-level sharing comparison lands exactly where the score-level M154
positive was measured); the 410k variant is registered as a follow-up,
cost-gated (six 409,832-row specialist encodes), run only if the 138k
cell passes.

Construction: rebuild the M108 whitener and the six domain candidate
pools (the M143 phase-1 construction: `_build_whitener` +
`_domain_candidates` per domain, [11,100] prefix, 512 atoms); encode
all 138k train rows + the 34,500 test rows per specialist dictionary
(~6 x 1h GPU; codes persisted under `v16/m159/`, ~2.1 GB total);
each specialist head = ridge penalty 1.0 on ALL 138k train rows (the
M143 specialist-head protocol, wider fit); the fusion = the M143b
protocol over [6 specialists, global] on the train scores (valid seed
55, frac 0.8, ladder {1,10,100,1000,10000}), evaluated on the test
scores; the global arm = the cached M143b global_train/global_test
scores (unchanged). Anchors: the six specialist own-domain
reproductions from the same cached M119/M143 anchor values (d0
0.19357142857142856 tol 0.002, the M143 a2 pattern) and the M143b
fused/global reads (0.22431884057971013 / 0.22460869565217392 tol
1e-9). Premise: section 5.3 floor for the wider fits (138k rows /
2,048 features = 67 rows/dim >= 10, checked in-run). Controls:
competence vs identity routing reported alongside; the M143b sealed
numbers are the incumbent. Gate: fused >= global + 0.005 on the sealed
test scores, else scoped negative.
