# PRIOR ART v22 — FINAL (claim ledger + sanity check)

Date: 14 August 2026. Registered survey M148, executed 14 Aug 2026.
Evidence: `logs/results/v16/m148_v22_final_prior_art_litsearch/evidence.json`
(summary: `logs/results/v16/m148_summary.txt`). Config:
`experiments/configs/v16/m148_v22_final_prior_art_litsearch.json`.

**How to read this document.** Part A lists who claims each mechanism the v22
plan relies on, at what level. Part B is the final sanity check of
`RESEARCH_IMPLEMENTATION_PLAN_v22.md` against the sealed evidence. Nothing here
supports a novelty claim; everything the programme measures is prior art as a
mechanism, and the programme's object is the sealed measurement.

---

## Instrument honesty (registered before search, N95.x)

- Anchors: **144 hits** across both indexes — instrument live.
- Recall probes: **1 of 6 retrieved** (only P-B, Maass' liquid-state-machine
  line, via Semantic Scholar). P-A (Jaeger ESN), P-C (Takens delay embedding),
  P-D (Wolpert stacked generalization), P-E (Friedman GBM), P-F (Waibel TDNN)
  were NOT retrieved. **The positive control failed.** Consequence, per the
  registered rules: absence across R1/R2/R4/R5 is UNRESOLVED — never "first",
  never "open". Found hits remain decisive.
- **28 queries failed** and are recorded as failures, never as empty:
  24 Semantic Scholar HTTP 429 (rate-limit) and 4 arXiv HTTP 301
  ("Moved Permanently" — the http:// endpoint was redirected; future runs
  should register the https:// endpoint). A failure is not a zero-hit result.
- Local cache served 2 queries from disk; 42 fetched live; 316 entries stored
  in `data/litsearch_cache.json`.

---

## Part A — claim ledger

### Carried from M135 (13 Aug 2026, seven image directions) — not re-run here

| Direction                                           | Claimed by                                                                                                                                 | Level                                                              |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| D1 data-axis scaling of frozen codes                | Bordelon–Canatar–Pehlevan et al.; Defilippis–Loureiro–Misiakiewicz; live kernel-ridge-at-scale solver line (ParK 2021, ASkotch 2024, etc.) | Theory claimed; our Q(n) measurement is ours to make, not to claim |
| D2 per-domain specialists + routing                 | Rebuffi et al. 2017; Med-MoE 2024; DA-MoE 2025; AnchorMoE 2026; ViMoE; Union of Experts; MDViT                                             | Claimed (active line)                                              |
| D3 fixed code geometry (Fisher/VLAD/SPM/power-norm) | Perronnin–Sanchez 2013; Jégou 2012; spatial-pyramid VLAD 2016; square-root normalisation 2015                                              | Fully claimed (pre-deep standard code)                             |
| D4 second-order/bilinear pooling                    | Lin et al. Bilinear CNN 2015; Compact/Low-rank Bilinear 2015–2016                                                                          | Fully claimed                                                      |
| D5 scattering/fixed filter banks                    | Mallat 2012; Bruna–Mallat PAMI 2013; learned-scattering hybrids                                                                            | Fully claimed                                                      |
| D6 shallow learned patch encoder                    | CKN (Mairal 2014); Thiry et al. ICLR 2021 (the programme's registered antecedent)                                                          | Claimed (direct antecedent)                                        |
| D7 frontier-map / price of learning                 | "Untrained CNNs Match Backpropagation at V1" 2026; "Contrasting random and learned features" 2022; Rahimi–Recht critique 2019              | Claimed as a research program                                      |

### New families surveyed by M148 (14 Aug 2026)

**R1 — reservoir computing / echo state networks. FULLY CLAIMED.**
Registered: Jaeger 2001 ("The echo state approach…"), Jaeger memory capacity
2002, Jaeger–Haas 2004, Maass et al. 2002 (liquid state machines). Found live:
Integer ESNs for digital hardware (2017), Reservoir Topology in Deep ESNs
(2019), Reservoir Memory Machines (2020), Evolutionary ESN — evolving
reservoirs in the Fourier space (2022). The construction, the echo-state
property, and ridge readouts are all established. The v22 temporal branch
re-measures, never claims.

**R2 — delay-line / autoregressive memory. FULLY CLAIMED.**
Registered: Takens 1981 (delay embedding), Waibel 1989 (TDNN), the NARX line.
Found live: autoregressive-with-slack models (2023), AR-convolutional-RNN
(2019), Hilbert-space embeddings of AR processes (2016), ARIMA–LSTM hybrids
(2024). The tap-delay-line arm of M147 is classical; only its measurement
inside the staged system is the programme's.

**R3 — growing / incremental reservoirs. CLAIMED (active).**
Found live: RECAP — local Hebbian prototype readout for reservoir dynamics
(2026), clustered echo state networks (2025), directed-evolution physical
reservoirs (2025), online residual learning with RC for quadrotor control
(2024), "Reservoir Computing in Real-World Environments: Optimizing the Cost
of Offline and Online Training" (2025). Appending reservoir neurons is a
populated line; the programme's append-lock-and-re-solve-fusion protocol is a
measurement of it.

**R4 — stacking / score fusion / classifier selection. FULLY CLAIMED.**
Registered: Wolpert 1992 (stacked generalization). Found live: META-DES
dynamic ensemble selection (2015), online local pool generation for dynamic
classifier selection (2018), competence measures for dynamic regressor
selection (2019), FIRE-DES++ online pruning of base classifiers (2018),
score-level fusion evaluations (2018), late-fusion multimodal lines through 2026. The v22 integration layer (late fusion + competence routing) is
concluded prior art; the plan's "fusion ≥ any single arm" is a mathematical
fact, not a contribution.

**R5 — residual growth over fixed base learners. FULLY CLAIMED.**
Registered: Friedman 2001 (gradient boosting). Found live: Random Feature
Representation Boosting (2025), ANOVA-boosting for Random Fourier Features
(2024), Landmark-based ensembles with RFF + gradient boosting (2019), "Ridge
Boosting is Both Robust and Efficient" (2025), "Boosted Kernel Ridge
Regression: Optimal Learning Rates and Early Stopping" (2019). **This claims
the M145 residual-growth idea itself, in the random-feature regime.** M145 is
therefore a protocol-identical measurement with a pruned-dense control, not a
new idea.

**R6 — pruned-dense baselines. FULLY CLAIMED.**
Found live: structured pruning survey (2023), structured pruning of CNNs
(2015), the lottery-ticket / iterative-magnitude-pruning line (2019–2024),
"Sparsity in Deep Learning: Pruning and growth for efficient inference and
training" (2021 — note it covers pruning AND growth), pruning-vs-efficiency
benchmarks through 2026. M144 is a baseline measurement at the programme's
budgets, nothing more.

### What survives — and only as measurement

Every mechanism in the v22 plan is claimed by others at the mechanism or
research-program level. The programme's remaining, unclaimed object is the
**sealed, matched-cost, same-corpus measurement of combinations**: the
construction factorial (M142), the fused integration layer with competence
routing (M143), the pruned-dense baseline (M144), residual growth with a
greedy-selection control (M145), the end-to-end arbiter (M146), the
temporal-memory screen with its three arms (M147), and the split-and-rebuild
registry operation (M149). Absence statements everywhere are UNRESOLVED under
the failed positive control; presence statements are decisive.

---

## Part B — final sanity check of RESEARCH_IMPLEMENTATION_PLAN_v22.md

Every cited number was checked against the sealed evidence on disk. Sources are
named per row.

| Plan row                                                        | Sealed evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Verdict                                                                      |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| E1: 26.1% full-corpus; cost claims corrected 14 Aug             | m141 cells `409832` = 0.261362 at the **6144-atom construction, ~500.7M MACs/image** (whitening 8.5M + cdist 483.7M + head 8.5M) — beats dense r56 (0.245014 @ 367.5M) on accuracy at 1.36× cost, NOT fewer. The 254.6M figure belongs to the M113-sealed **3072-atom** construction at 138k rows (0.2153 vs dense r42 0.1972); dense r224 0.5375 @ 6,124M → 12.2× fewer for the 6144-atom best, 24.05× for the cheapest sealed point. Plan E1, Phase A and M142 gates were re-registered accordingly. | ✓ corrected (was: conflated two constructions and overclaimed a 31% MAC win) |
| E2: ~6,144 atoms ceiling; ~8 effective dims                     | m126 ladder [6144..16384], kill switch fired (q_16384 < q_6144 by −0.0041); m138 `effective_rank` = 7.761                                                                                                                                                                                                                                                                                                                                                                                              | ✓ consistent                                                                 |
| E3: 22.5% @ 138k → 26.1% @ 410k                                 | m140 `gain_vs_138000` ⇒ Q(138k) = 0.224609; m141 Q(409832) = 0.261362                                                                                                                                                                                                                                                                                                                                                                                                                                  | ✓ consistent                                                                 |
| E4: closed-form depth flat                                      | m115 lofi: l0 0.215275, l1 0.136087, l2 0.112116 (kill switch fired)                                                                                                                                                                                                                                                                                                                                                                                                                                   | ✓ consistent                                                                 |
| E5: trained head ~15% vs 21.5% on sparse; trained wins on dense | a2_head: sparse SGD converged 0.149015 vs ridge 0.215275; dense SGD 0.645884 vs ridge 0.536754                                                                                                                                                                                                                                                                                                                                                                                                         | ✓ consistent                                                                 |
| E6: penalty flat; smoothing ≡ rescale; margin fails             | m136: ridge ladder 0.01→0.220174 … 1.0→0.224609 (flat); smoothed = 0.224609 exactly; hinge best 0.005797 (not converged)                                                                                                                                                                                                                                                                                                                                                                               | ✓ consistent                                                                 |
| E7: seeds 22.1% vs 22.5%; rank ~8                               | m138: ensemble 0.221101, members 0.215275/0.213507; eff_rank 7.761                                                                                                                                                                                                                                                                                                                                                                                                                                     | ✓ consistent                                                                 |
| E8: learned dictionary doesn't transfer                         | m113 learned / M108 greedy selection verdicts (registered)                                                                                                                                                                                                                                                                                                                                                                                                                                             | ✓ consistent (carried)                                                       |
| E9: binary ~3 points down                                       | m114: binary 0.184203 vs float reference 0.215275 (−3.1)                                                                                                                                                                                                                                                                                                                                                                                                                                               | ✓ consistent                                                                 |
| E10: hard 18.8% / oracle 20.5% / global 22.5%                   | m139b: routed 0.187652, oracle 0.204957, global 0.224609                                                                                                                                                                                                                                                                                                                                                                                                                                               | ✓ consistent                                                                 |
| E11: 75.6% domains / 22.5% classes                              | m139b anchors: router 0.755942; class_head 0.224609                                                                                                                                                                                                                                                                                                                                                                                                                                                    | ✓ consistent                                                                 |
| E12b: KB-scale count memory, optimum window 4                   | m131 per_window: w4 ppl 3.3244 @ 241,344 B is the optimum (w8: 3.447 @ 27.4 MB)                                                                                                                                                                                                                                                                                                                                                                                                                        | ✓ consistent                                                                 |
| E12c: trained transformers ~10× on sequences                    | M134 sealed "price of learning ≈10× on the DSL" (cited by M135 D7)                                                                                                                                                                                                                                                                                                                                                                                                                                     | ✓ consistent (wording points to the sealed note)                             |
| E13: spectrum can't predict the crossing                        | m121 certificate_verdict: MSE-proxy does not predict the crossing; kill switch fired                                                                                                                                                                                                                                                                                                                                                                                                                   | ✓ consistent                                                                 |

**Structural checks.**

- Referenced files exist: `LESSONS_ARCHIVE_v22.md`, `PROGRAMMATIC_PRIMITIVES_
LITERATURE_REVIEW.md`, `HTN_ROUTING_LITERATURE_REVIEW.md`,
  `PRIOR_ART_M135_BREAKTHROUGH_DIRECTIONS.md` — all present in `analysis/`.
- Milestone numbering: M137 (carried), M142–M147 (registered, unsealed),
  M148 (this survey), M149 (group splitting) — no collisions; nothing has run.
- Section cross-references (6, 9, 11) all resolve.
- Epistemic compliance: every evidence row carries a scope column; E12b/E12c
  are treated as priors, not verdicts, in Phase T; the separability assumption
  is registered as tested-not-trusted; every milestone carries anchor
  reproduction, premise gates, smoke refusal, and plain-English recording.
- Plan annexes added this session: section 11 (operational cycle, end-to-end
  protocol) and the M149 split-and-rebuild design (S8, section 7, 11.3).

**Open items carried forward (non-blocking).**

1. Future litsearch runs should register the arXiv **https** endpoint to stop
   the 301s (the instrument's host allow-list needs the change with it).
2. E12c's "~10×" should cite the M134 metric explicitly when the paper is
   drafted (perplexity/footprint ratio on the DSL), so the number is
   reproducible from one sealed file.

**Verdict.** The plan is internally consistent, numerically grounded in sealed
evidence, and prior-art-compliant. It is ready for M142 dispatch (with M143,
M144, M147 running in parallel).
