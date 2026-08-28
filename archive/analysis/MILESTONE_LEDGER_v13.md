# GEODE v13 Milestone Ledger

Program: **Nameable Bases and Absolute Boundaries — Sparse Concept Geometry
with Measured Out-Group Exposure**
Plan: `analysis/RESEARCH_IMPLEMENTATION_PLAN_v13.md`
Opened: 29 July 2026
Status: **open** — final confirmation labels sealed (`final_labels_opened: false`)

This is the authoritative record for the v13 program. `analysis/MILESTONE_RESULTS.md`
remains the cross-program index and carries a one-line pointer per v13
milestone; where the two disagree, this file governs.

## Reporting rules

Inherited unchanged from the v12 protocol, plus two additions specific to v13.

- Compare methods only when corpus, partition hash, feature hash, and seed match.
- Report multi-seed means; state the seed count.
- Test metrics are observational. They never drive gradients, early stopping,
  model selection, or hyperparameter selection.
- Hypotheses, grids, gates, and kill switches are registered **before** execution.
  Any change after registration is recorded as a numbered amendment (R1, R2, ...)
  in the plan, with its justification, before the run.
- **New in v13 (design principle 6).** A cell whose samples-per-fitted-dimension
  ratio is below 10 cannot carry a negative result. Such cells are reported as
  _void_, not as evidence against a hypothesis.
- **New in v13 (design principle 9, added 29 July 2026).** Every measurement
  operand ships with a positive control that fails if the operand is not
  measuring what it names. An operand without one cannot gate a decision.

## Milestone index

| ID  | Milestone                           | Registered question                                                      | Outcome                                                                                       | Evidence                                  |
| --- | ----------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | ----------------------------------------- |
| M77 | Probe-objective forensics           | Was the v12 probe loss training the geometry at all?                     | **H77 confirmed** — the objective was gradient-dead                                           | `logs/results/v13/m77_probe_degeneracy/`  |
| M78 | Sample-adequacy forensics           | Was the v12 DomainNet transfer failure a basis-identifiability artifact? | **H78 confirmed** — the M74 cell is void; transfer reaches parity at low rank                 | `logs/results/v13/m78_sample_adequacy/`   |
| N1  | Open-world detection (negative)     | Can unseen classes be detected at or above the freely composable bar?    | **No, for any method** — representational; amends the M78 open-set finding                    | `logs/results/v13/c3_probe.json`          |
| M79 | Acceptance reframe                  | On what may v13 claim a result, decided before any is measured?          | **Complete** — L2 restated corpus-relative; I5 primary; deployment context specified          | `analysis/ACCEPTANCE_CRITERIA_v13.md`     |
| M80 | Sparse concept dictionary           | Do frozen features admit a sparse basis at low accuracy cost?            | **H80 gate passed** — but the passing cell ties a random dictionary; N80.2 restricts          | `logs/results/v13/m80_sparse_dictionary/` |
| M81 | Sparse head + I5 (**decisive**)     | Are sparse-atom explanations more forward-simulable than dense ones?     | **Task-width artifact** — atoms win at 8-way, no admissible arm at 128-way; dominance blocked | `logs/results/v13/m81_sparse_head/`       |
| M82 | Atom naming                         | —                                                                        | not started                                                                                   | —                                         |
| M83 | Absolute-scale boundary supervision | —                                                                        | not started                                                                                   | —                                         |
| M84 | Out-group exposure ladder           | —                                                                        | not started                                                                                   | —                                         |
| M85 | Confirmation + frontier             | —                                                                        | not started                                                                                   | —                                         |
| M86 | Finalization                        | —                                                                        | not started                                                                                   | —                                         |

---

## M77 — Probe-objective forensics

**Registered hypothesis H77.** The v12 probe/Eikonal open-space objective was
constant with respect to the geometry parameters, so its decreasing loss trace
recorded nothing about open-space rejection.

**Method.** Replay the sealed v12 M73 seed-11 configuration under
instrumentation, and verify the instrumentation is faithful before reading any
diagnostic from it.

**Faithfulness gate (passed).**

| Operand                                          | Value   |
| ------------------------------------------------ | ------- |
| `history_reproduction_delta` vs sealed v12 trace | **0.0** |
| `trained_state_hash_match`                       | true    |

**Result — H77 confirmed.**

| Operand                                          | Epoch 1  | Epoch 24 |
| ------------------------------------------------ | -------- | -------- |
| `own_score__axis_tangent` / `masking` / `normal` | **4.0**  | **4.0**  |
| `own_score__random_direction`                    | 97.554   | 107.872  |
| `probe_target` (detached, adaptive)              | 13.795   | 6.353    |
| `mean_hinge`                                     | 9.649    | 2.318    |
| `own_class_is_minimiser_fraction`                | 0.98694  | 0.98694  |
| `probe_grad_norm_log_tangent`                    | 6.53e-17 | 6.65e-17 |
| `total_grad_norm_log_tangent`                    | 0.2866   | 0.0588   |

- All four trained probe families are **scale-invariant** to below `1e-9`: the
  probes are placed in units of the fitted extent they are meant to constrain,
  so the geometry cannot move relative to them.
- **101.5%** of the probe-loss decrease is explained by the detached target
  falling, not by any probe score rising. The minimum probe score rose 0.078.
- Probe-term gradients are 15–17 orders of magnitude below the total-objective
  gradient. The term was **numerically inert**.

**Consequences.**

1. `analysis/V12_FINAL_CLAIM_LEDGER.md` Amendment A1: the inference "the
   registered probe/Eikonal objectives did not establish generalized open-space
   rejection" is downgraded from a finding to an **untested condition**.
2. v12 Outcome E is **not** overturned. The open-set failures on held-out
   families, real OOD, and transfer stand on independent evidence.
3. Plan design principle 4 becomes mandatory: **no synthetic negative may be
   defined in units of the parameter it constrains.**

**Tests.** `experiments/common/test_v13_probe_forensics.py` — 8 focused tests.

---

## M78 — Sample-adequacy and basis-identifiability forensics

**Registered hypothesis H78.** The M74 DomainNet transfer failure is driven by
rank-32 basis estimation from 60 samples per class, not by a transfer property
of the head.

### Amendments

**R1 — the registered grid was infeasible.** The only DomainNet artifact in the
repository holds **exactly 100 observations per class**. With the M74 partition
contract fixing 20 calibration + 20 evaluation per class, the maximum
`geometry_per_class` is **60**. The registered {60, 200, 600} sweep cannot be
executed without re-acquiring the corpus. This is itself a finding: **v12's
`geometry_per_class: 60` was the ceiling imposed by the extracted array, not a
design choice.** The grid was re-registered as rank-first (Axis A: rank
∈ {2,4,8,16,32} at n=60) with a 3x sample span (Axis B: n ∈ {20,40,60} at rank
∈ {8,32}). The 10x sweep is deferred to M85.

**R2 — the first execution's stability operand was invalid.** See
`logs/results/v13/m78_sample_adequacy_void_r1/VOID.md`. The operand fitted a
separate PCA projection per half and therefore measured projection variance
rather than basis identifiability. Caught by a positive control. The run is
void and retained; this section reports the R2 re-execution.

### Result — H78 confirmed

Seeds 11/23/37. `acc` = transfer known balanced accuracy; `logit` = logistic
control on identical partitions; `ident` = identifiability, where 1.0 means the
two disjoint halves recover the same subspace and 0.0 means the fitted subspace
is indistinguishable from a random one of the same shape.

|      n |  rank | fitted |     n/dim |     acc % |   logit % |       gap | unknown % | logit unknown % | angle° | random° | ident |
| -----: | ----: | -----: | --------: | --------: | --------: | --------: | --------: | --------------: | -----: | ------: | ----: |
|     20 |     8 |      8 |      2.50 |     62.60 |     71.25 |     −8.65 |      2.50 |           19.79 |  59.76 |   72.25 | 0.173 |
|     20 |    32 |     18 |      1.11 |     62.60 |     71.25 |     −8.65 |      2.62 |           19.79 |  59.76 |   72.25 | 0.173 |
|     40 |     8 |      8 |      5.00 |     67.55 |     70.62 |     −3.07 |      4.17 |           24.58 |  52.10 |   72.25 | 0.279 |
|     40 |    32 |     32 |      1.25 |     62.24 |     70.62 |     −8.39 |      4.24 |           24.58 |  48.87 |   61.07 | 0.200 |
|     60 |     2 |      2 |     30.00 |     73.80 |     74.06 |     −0.26 |      0.50 |           20.42 |  49.96 |   82.32 | 0.393 |
| **60** | **4** |  **4** | **15.00** | **74.22** | **74.06** | **+0.16** |      0.52 |           20.42 |  49.70 |   77.97 | 0.363 |
|     60 |     8 |      8 |      7.50 |     70.52 |     74.06 |     −3.54 |      0.61 |           20.42 |  47.25 |   72.25 | 0.346 |
|     60 |    16 |     16 |      3.75 |     71.04 |     74.06 |     −3.02 |      0.73 |           20.42 |  45.60 |   62.99 | 0.276 |
|     60 |    32 |     32 |      1.88 |     65.94 |     74.06 |     −8.13 |      0.80 |           20.42 |  40.36 |   50.03 | 0.193 |

**Findings.**

1. **W2 defect confirmed.** Rank 4 at n=60 beats rank 32 at n=60 by
   **+8.28 accuracy points** (74.22 vs 65.94), above the registered 5-point
   threshold, at **identical sample count**. The binding constraint was rank,
   not data volume.
2. **The v12 transfer deficit disappears.** M74 reported the geometric head
   **7.34 points below** logistic on DomainNet transfer. At rank 4 the gap is
   **+0.16 points** — parity. At rank 2 it is −0.26. Both are inside the
   3.0-point tolerance the v13 plan registers for L1.
3. **The M74 cell is void.** Its samples-per-fitted-dimension ratio is **1.88**,
   far below the registered floor of 10. Under design principle 6 it cannot
   carry a negative result. **7 of 9 cells** fall below the floor; only rank 2
   (30.0) and rank 4 (15.0) at n=60 clear it.
4. **No basis in the grid is identified.** Every cell scores below the 0.5
   identifiability floor. The best is **0.393** at rank 2; the M74 cell is
   **0.193**, barely distinguishable from a random subspace. Identifiability
   falls monotonically with rank (0.393 → 0.363 → 0.346 → 0.276 → 0.193), and
   accuracy tracks it. This is a clean dose–response.
5. **The open-set negative is untouched and is now isolated.** Unknown recall
   is 0.50–0.80% at n=60 across every rank, against 20.42% for logistic. The
   best low-rank gain is **−0.07 points** — the gate does not fire. Sample
   adequacy explains the _accuracy_ failure and explains **none** of the
   open-set failure. Removing the confound makes the open-set negative
   **stronger**, not weaker.

   > **Amendment R3, issued by N1. Finding 5 is withdrawn.** It was reached on a
   > grid whose richest cell held 1.88 samples per fitted dimension at rank 32,
   > so no cell in it was adequately sampled for an open-set measurement, and the
   > conclusion that sample adequacy explains "none" of the open-set failure was
   > not one the grid could support. At 536 samples per class the geometric head
   > reaches 23.86% recall at a matched 10% false-alarm rate, against the 20.42%
   > logistic bar and a 19.86% kNN control. The open-set deficit closes with
   > sampling just as the closed-set deficit does. See N1. Findings 1–4 stand.

**Consequences.**

1. `analysis/V12_FINAL_CLAIM_LEDGER.md` Amendment A2: the L5 transfer
   restriction is amended to **"confounded with basis identifiability at the
   registered rank"**. The v12 claim that the geometric head loses ~7 points
   under corpus transfer is **withdrawn**.
2. **Rank selection becomes sample-dependent by contract** for all later v13
   milestones. No milestone may fit a rank whose samples-per-fitted-dimension
   ratio is below 10 without recording the cell as void.
3. The open-set question is now cleanly separated from the accuracy question
   and passes to M83/M84 unconfounded. **Superseded by N1**, which answered it
   directly: detection is near chance on this corpus for every method, including
   the freely composable controls, so M83/M84 inherit a bounded question rather
   than an open one.
4. M85 must re-extract DomainNet at 640+ per class before any transfer claim.

**Tests.** `experiments/common/test_v13_sample_adequacy.py` — 11 focused tests,
including the positive and negative controls for the stability operand.

**Runtime.** 27 cells (9 configurations x 3 seeds) in 5 min 39 s across 14
worker processes. Each worker trains single-threaded under
`torch.use_deterministic_algorithms(True)`, so parallel and serial execution
agree exactly.

---

## Infrastructure I1 — Compute environment and accelerator qualification

**Question.** The program had never used the GPU. Could the RX 9070 XT carry
v13 work, and if so which parts?

**Environment split.** Two interpreters, deliberately kept apart:

| Environment  | Torch                        | Role                                                                                       |
| ------------ | ---------------------------- | ------------------------------------------------------------------------------------------ |
| `.venv`      | `2.13.0+cpu`                 | **Frozen replay environment.** Reproduces every sealed v12/v13 hash. Must not be modified. |
| `.venv-rocm` | `2.11.0+rocm7.13.0a20260416` | GPU work only. Torch is a _downgrade_, so it cannot reproduce sealed hashes.               |

The ROCm wheel comes from AMD's `rocm.nightlies.amd.com/v2/gfx120X-all/` index.
It is **not** available from `download.pytorch.org`, whose ROCm indices ship
manylinux wheels only.

**Device selection is not optional.** The Ryzen 7800X3D exposes an integrated
`gfx1036` adapter that ROCm enumerates as device 0, ahead of the discrete
`gfx1201` card. The installed wheel ships no kernels for `gfx1036`, and simply
selecting device 1 is **not sufficient**: HIP loads code objects for the whole
visible set, so device-side kernels fail with `hipErrorInvalidKernelFile` even
when the discrete card is selected explicitly. The integrated adapter must be
masked out before the HIP runtime starts. `tools/rocm_device.py` does this by
re-executing the process with `HIP_VISIBLE_DEVICES` pinned.

**Benchmark** (`tools/benchmark_rocm.py`; CPU column measured in `.venv`):

| Workload                               | CPU (torch 2.13, 8 threads) |   RX 9070 XT |   Speedup |
| -------------------------------------- | --------------------------: | -----------: | --------: |
| fp32 matmul 1024³                      |                     0.03 TF |     10.41 TF |         — |
| fp32 matmul 4096³                      |                     0.72 TF |     15.47 TF |     21.5× |
| fp32 matmul 8192³                      |                     0.77 TF | **17.55 TF** | **22.9×** |
| fp64 matmul 2048³                      |                     0.30 TF |      0.68 TF |      2.3× |
| Metric-field shape 1920×384 fp64, ×100 |                      4.45 s |       2.82 s |      1.6× |

Correctness against CPU: max absolute difference **9.9e-05** on fp32 matmul.
`torch.use_deterministic_algorithms(True)` is honoured for the index/scatter
operations the program relies on. fp64 is supported and finite.

**Findings.**

1. **The GPU is transformative for fp32 dense work and marginal for the current
   v13 shapes.** RDNA4 runs fp64 at 1/32 rate, and the metric-field tensors are
   only 1920×384, so launch overhead dominates. Migrating the sealed
   milestones would break every hash for a 1.6× gain and is not worth doing.
   The GPU's value lies in M80's dictionary training, which is fp32 and large.
2. **Never run CPU work in `.venv-rocm`.** Its bundled CPU backend reaches only
   0.02 TF on fp32 8192³, roughly **38× slower** than the CPU build in `.venv`.
3. **DirectML cannot substitute for the CPU provider on the frozen backbone.**
   It disagrees by 16% relative at batch 32 and **11.7% at batch 1**, so the
   divergence is in the quantized kernels themselves, not batch composition. It
   is also only 1.4× faster at batch 1. Feature extraction stays on CPU.

---

## Infrastructure I2 — The frozen backbone is not a per-image function

**Discovery.** Attempting to enlarge the DomainNet corpus, the registered
positive control — "the sealed M70 corpus must be a per-class prefix of any
larger corpus built by the same prefix scan" — **failed**, at a maximum absolute
difference of 2.26 against a feature norm of ~45. Row alignment was verified as
an exact identity permutation, so the same images were being compared.

**Cause.** The frozen `dinov2-small` INT8 graph contains **49
`DynamicQuantizeLinear` operators**. These derive activation scales from the
whole input tensor at run time. Measured directly on the frozen graph:

| Comparison                               | Max absolute difference |
| ---------------------------------------- | ----------------------: |
| One image alone vs. inside a batch of 8  |                   0.851 |
| One image alone vs. inside a batch of 32 |                   1.210 |
| Same batch of 32, reordered              |               **0.000** |

Reordering changes nothing while membership changes a great deal, which
identifies the dependence precisely: features are a function of the _set_ of
images sharing each batch.

**Consequences.**

1. **The sealed M70 corpus is a function of how it was chunked**, not of its
   images alone. It carries a batch-composition perturbation of roughly 15%
   relative magnitude in feature space. Every v12 and v13 result computed on it
   inherits that nuisance term.
2. **M70 cannot be reproduced by any enlarged corpus**, because enlarging the
   corpus necessarily changes the batch layout. The prefix control was not a
   failing test of the new pipeline; it was a _passing_ detector of a defect in
   the old one. It has been replaced by a shard-invariance control and a
   reported (never gated) M70 divergence figure.
3. **All v13 extraction runs at batch size 1**, which makes the corpus a
   well-defined function of `(image, backbone)`. This is also what makes
   extraction fast: because each image is independent, partitioning the work
   across processes provably cannot change the result, so extraction shards
   freely across the CPU.
4. This is a second instance of the M78 lesson. A measurement operand that had
   never been positively controlled was silently wrong for the entire v12
   program.

---

## Infrastructure I3 — Retrospective exposure audit of the whole program

Two defects are now on the table. Both were found in v13, but both are
properties of machinery that older milestones also used, so the honest question
is not "is v12 damaged" but "how far back does the damage reach". This section
answers that from recorded lineage rather than from recollection. The tool
`tools/trace_backbone_lineage.py` walks a configuration's `path` references and
reports every backbone and batch-size declaration it finds.

The two defects have different mechanisms and, crucially, different
consequences. Conflating them would overstate the damage.

### Defect A — sample inadequacy (voids results)

A per-class low-rank basis fitted from too few samples is not identified: the
fitted subspace is dominated by sampling noise, so any measurement of it
measures the noise. M78 fixed the floor at ten samples per fitted dimension.
Exposure requires **per-class low-rank basis fitting** with `n / rank < 10`.

| Milestone group                    | Samples/class |    Rank | `n / rank` | Verdict     |
| ---------------------------------- | ------------: | ------: | ---------: | ----------- |
| v9 M53 bounded tubes               |       **800** | 8/16/32 | 100–**25** | Not exposed |
| v10 M58 affine screen              |       **800** | 8/16/32 | 100–**25** | Not exposed |
| v11 M63 / M65 directional envelope |       **800** | 8/16/32 | 100–**25** | Not exposed |
| v12 M70–M73 DomainNet              |           100 |      32 |  **3.125** | **Void**    |
| v12 M74 transfer                   |            60 |      32 |  **1.875** | **Void**    |

The v9–v11 figure is derived, not assumed. `experiments/configs/v9/m51_surface_diagnostics.json`
declares `geometry_fraction = 0.8`; `data/v5/features/.../extraction_summary.json`
records `train n_samples = 10000` over ten classes, i.e. 1000 per class; so the
`geometry_fit` partition built by `_partition_seed` holds `0.8 × 1000 = 800`
rows per class. This is confirmed independently by the recorded
`score_calibration` count of 1600, which is the complementary `0.2 × 1000 × 8`.

So the v9, v10 and v11 negative outcomes were **not** produced by unidentified
bases. They rest on 25 to 100 samples per fitted dimension, comfortably above
the floor. Those negatives stand.

### Defect B — batch-dependent INT8 features (does not void results)

Exposure requires a **quantized** backbone run at **batch size greater than
one**.

| Backbone                     | Quantization | Extraction batch | Exposed |
| ---------------------------- | ------------ | ---------------: | ------- |
| dinov2-small (v5 m19 native) | INT8, 49 ops |           **32** | Yes     |
| dinov2-small (v12 M70+)      | INT8, 49 ops |           **32** | Yes     |
| siglip-base-patch16-256      | Quantized    |               >1 | Yes     |
| ijepa-vith16-1k (v5 m19 S1)  | INT8         |            **1** | No      |
| MobileNetV2 (tier4/5/6)      | **Float32**  |              n/a | No      |

The dinov2 exposure propagates forward:
`v5 m19_native_dinov2_sphere_support_*` (`extraction_batch_size: 32`) →
`v6 m30_directional_s2` → `v9 m51/m53` → `v10 m58` → `v11 m63/m65`.

**This does not void those milestones**, and the distinction matters. The
perturbation is deterministic, fixed by the batch layout, and _identical for
every method evaluated on the array_ — reordering within a batch was measured at
exactly 0.000. Each frozen array is therefore a self-consistent feature space,
merely a non-canonical one. Method-versus-method comparisons on a single array
remain fair, because both arms see the same features. What is lost is (i)
reproducibility from any other chunking, and (ii) the validity of comparing
values across two arrays built with different layouts.

One caveat is recorded rather than resolved: the perturbation is a small
per-image nuisance term, and low-rank subspace fitting is in principle more
sensitive to it than a nearest-neighbour baseline. The v9–v11 negatives could
therefore be mildly pessimistic toward the geometric method. The effect is
roughly 2–5% of feature norm and is not plausibly the difference between the
recorded outcomes and their gates, so no re-run is scheduled on this basis.

### Not exposed to either defect

These lines of work involve no per-class low-rank basis fitting and no image
backbone at all, so neither defect can reach them:

| Work                              | Why it is out of scope                                                                                            |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Tier1 SDF sphere / ellipsoid      | `EllipsoidExpert` fits explicit shape parameters (center, radii) to 1500 points in 3-D. No subspace, no backbone. |
| Tier1 v6 protocol S0              | Global PCA + LDA + scaler, not per-class fits.                                                                    |
| Tier2 ModelNet10                  | Point-cloud reconstruction, not basis fitting.                                                                    |
| Tier5 CIFAR-100, tier6 refinement | Float32 MobileNetV2 features.                                                                                     |
| Tier6 wikitext103                 | Temporal text prediction, no vision backbone.                                                                     |

### Conclusion

The re-test obligation is **narrow**. Only the v12 DomainNet milestones
(M70–M74) require redoing, which v13 is already doing. The synthetic geometry
results, the MobileNetV2 results, and the v9–v11 directional-envelope negatives
all survive the audit. The program's accumulated negative evidence against the
directional-envelope family is therefore intact, and v13 is not starting from a
blank slate.

---

## Infrastructure I4 — The v13 DomainNet corpus

M78 established that a per-class basis needs at least ten samples per fitted
dimension. The v12 corpus supplied 3.1. This section records the replacement
corpus that clears the floor, built with the batch-size-1 extractor from I2.

### Registered parameters

| Parameter                | Value                                                              |
| ------------------------ | ------------------------------------------------------------------ |
| Classes                  | 128                                                                |
| Samples per class        | 576                                                                |
| Total samples            | 73,728                                                             |
| Output dimension         | 384 (dinov2-small, CLS token)                                      |
| Extraction batch size    | **1**                                                              |
| Native short edge filter | ≥ 256 px                                                           |
| Execution provider       | CPUExecutionProvider                                               |
| Configuration hash       | `812840823f316183597e8264bb21ee9ee5bc31484a25bda23fd027c2ecd717e3` |

### Why 576 and not 640

The first build was registered at 640 per class and **failed closed** at
81,809 of 81,920 images after fifteen minutes, with three classes short:

| Class | Usable train images at ≥256 px |
| ----: | -----------------------------: |
|    65 |                        **579** |
|    53 |                            603 |
|    96 |                            627 |

The native-resolution filter does not have a uniform per-class yield, so the
true uniform ceiling over these 128 classes is 579. The registration was
amended to 576. This is worth recording plainly: the 640 target was a guess made
before the yield was measured, and the fail-closed check caught it rather than
silently emitting a ragged corpus. The cost was fifteen minutes of lost work,
because the extractor has no caching.

### Geometry budget

| Quantity                              |     Value |
| ------------------------------------- | --------: |
| Geometry fit per class                |       536 |
| Calibration per class                 |        20 |
| Evaluation per class                  |        20 |
| Samples per fitted dimension, rank 32 | **16.75** |
| Samples per fitted dimension, rank 60 |      8.93 |
| Registered floor (M78)                |      10.0 |

Rank 32 clears the floor with margin. **Rank is capped at 53** by this corpus;
anything above that falls back below the floor and must not be fitted without
enlarging the corpus again. This is a live constraint on M80–M83, not a
footnote.

### Controls

| Control                                      |                 Result |
| -------------------------------------------- | ---------------------: |
| Shard invariance (1 worker vs. 4, 96 probes) | max abs diff **0.000** |
| Array finiteness                             |             all finite |
| Class-major ordering                         |               verified |
| Per-class count uniformity                   |        min = max = 576 |

The shard-invariance control replaces the M70 prefix control retired in I2. It
is the correct positive control for this pipeline: it asserts the property the
extractor actually claims — that the result is a function of the image alone and
therefore independent of how work was split across processes. It passes at
exactly zero, not merely within tolerance.

### M70 divergence, reported and not gated

| Quantity                    |      Value |
| --------------------------- | ---------: |
| Classes compared            |        128 |
| Prefix samples per class    |        100 |
| Maximum absolute difference |     3.0383 |
| Mean relative difference    | **17.80%** |

This is non-zero **by construction** and is not a failure. M70 was extracted at
batch size 32, where `DynamicQuantizeLinear` makes every feature depend on its
batch neighbours; no batch-size-1 extraction can reproduce it. The figure is
recorded so the size of the v12 nuisance term is on the record, and it is
deliberately not wired to any gate.

### Throughput

Extraction ran 73,728 images in 798 seconds — **92.4 images/second** across 14
worker processes. Per-image extraction is roughly 2.5× slower than batch-32 per
image on one core, but because each image is now provably independent the work
shards freely, and the sharded batch-1 pipeline beats the old single-process
batch-32 path (60.2 img/s) outright. Correctness and speed were not in tension
here; the defect fix paid for itself.

### Test coverage

`experiments/common/test_v13_domainnet_large.py` — 10 tests, all passing. Two
are positive controls for the I2 finding: one asserts that batched extraction
_is_ batch-dependent (so the defect would be detected if it returned), and one
asserts that per-image extraction is _not_ affected by grouping.

### Artifacts

`logs/results/v13/domainnet_large/` — `arrays/features.npy` (73728 × 384
float32), `arrays/labels.npy`, `evidence.json`, `selection_manifest.json`,
`artifact_index.json`. Final labels remain sealed.

### Corpus defect — domain skew, recorded after the fact

Discovered during the N1 diagnosis below, not at build time. The 128 classes were
filled by scanning shards in order, and quickdraw is by far the largest DomainNet
split, so the corpus is **61% quickdraw**:

| Domain    |   Rows | Classes with usable per-class depth |
| --------- | -----: | ----------------------------------- |
| quickdraw | 44,800 | 128, at exactly 350 each            |
| clipart   |  9,939 | 128, median 70                      |
| infograph |  8,029 | 128, median 44                      |
| painting  |  7,195 | 111, median 56                      |
| real      |  3,607 | 55, median 53                       |
| sketch    |    158 | 3                                   |

The corpus is uniform in **class** by construction and badly non-uniform in
**domain** by accident. Quickdraw is the least semantically rich of the six. The
selection is stratified on the wrong axis. Nothing already registered against
this corpus is void — I4's controls all still hold, and the per-class uniformity
that M78 required is intact — but any future milestone that depends on semantic
richness rather than class balance must either restrict to a domain or rebuild
with domain-stratified selection. Recorded here so the next use of this corpus
starts from the defect rather than rediscovering it.

---

## Negative result N1 — Open-world class detection does not work on this corpus, for any method

Produced on the abandoned exploration branch `explore/v13-composability`. The
branch is **not merged**; its charter and ledger stay on the branch. Only the
diagnosis is carried here, as its charter required on abandonment. Evidence:
`tools/probe_c3_detection.py`, `tools/probe_c3_scale.py`,
`logs/results/v13/c3_probe.json`, `logs/results/v13/c3_scale.json`.

### Registered question

The branch asked whether a class can be added after the fact without retraining.
It registered detection of unseen classes as the decisive criterion, to be tested
against the freely composable baselines — kNN and nearest class mean — on
identical features at a matched false-alarm rate, and registered in advance that
a representational failure would be fatal.

### The instrument was validated first

| Control                              | Requirement    | Mixed corpus | Single domain |
| ------------------------------------ | -------------- | ------------ | ------------- |
| Positive: far-field noise, NCM       | AUROC near 1.0 | **1.0000**   | **1.0000**    |
| Positive: far-field noise, kNN       | AUROC near 1.0 | **1.0000**   | **1.0000**    |
| Negative: held-out **known** samples | AUROC near 0.5 | **0.5136**   | **0.5002**    |

### Result — near chance for everything

100 known classes, 28 held out entirely, 536 fitted samples per class. AUROC:

| Score                | Mixed corpus | Single domain |
| -------------------- | ------------ | ------------- |
| Nearest class mean   | 0.5388       | 0.6141        |
| kNN nearest distance | 0.5824       | 0.6228        |
| Geometric, rank 16   | 0.5868       | **0.6570**    |
| Geometric, rank 32   | 0.5898       | 0.6517        |

The geometric head is not the thing that fails: it is marginally the **best** of
the three. The freely composable controls fail with it, on the same features. The
failure is representational — unknown classes lie inside known-class structure in
DINOv2 CLS space on DomainNet — and no distance-based rule recovers them.

### The v12 AUROC record is corpus-specific

v10, v11 and v12 recorded novelty AUROC of 0.902–0.972, which sat in open
contradiction with M78's 0.5–4.24% unknown recall. The obvious reconciliation was
task width: those milestones used eight known classes and two unknown. Sweeping
width on this corpus at rank 16, five independent class draws each, mean [range]:

| Known | Unknown | kNN AUROC               | Geometric AUROC         |
| ----- | ------- | ----------------------- | ----------------------- |
| 8     | 2       | 0.6374 [0.6128, 0.6818] | 0.6327 [0.6112, 0.6686] |
| 16    | 4       | 0.6589 [0.6409, 0.6730] | 0.6518 [0.6226, 0.6740] |
| 32    | 8       | 0.6129 [0.5524, 0.6660] | 0.6150 [0.5674, 0.6536] |
| 64    | 16      | 0.6011 [0.5729, 0.6209] | 0.6106 [0.5783, 0.6370] |
| 100   | 25      | 0.5997 [0.5851, 0.6172] | 0.5964 [0.5825, 0.6206] |

Width is worth about 0.04 AUROC across the whole sweep. It cannot carry 0.637 to
0.95. **The earlier figures measured an easier corpus, not a better detector.**
They stand for what they measured and must not be generalised. Every future
open-set number in this program is to be reported with the corpus and the class
count attached, or it is not interpretable.

### Amendment to M78 — the open-set deficit was a sample-adequacy artifact

M78 recorded geometric unknown recall of 0.50–4.24% against a logistic bar of
19.79–24.58% and concluded the open-set deficit does **not** close as samples per
class rise, in contrast to the closed-set deficit which does. That conclusion is
**withdrawn**. Every M78 cell was sample-starved — at rank 32 the richest cell
had 1.88 samples per fitted dimension against a floor of 10 — so the grid never
observed an adequately sampled open-set cell at all. At 536 samples per class,
recall at a matched 10% false-alarm rate, single domain:

| Score                        | Recall at 10% FA |
| ---------------------------- | ---------------- |
| M78 geometric, rank 32, n=60 | 0.80%            |
| M78 logistic bar             | 20.42%           |
| kNN control, N1 probe        | 19.86%           |
| **Geometric, rank 16**       | **23.86%**       |
| **Geometric, rank 32**       | **22.57%**       |

The geometric head matches and slightly exceeds both the logistic bar and the kNN
composability control. The open-set deficit closes with sampling exactly as the
closed-set deficit does. M78's headline finding — that sample adequacy was the
hidden variable — is **strengthened**; only its open-set exception was wrong, and
it was wrong because no cell in the grid could have detected the exception.

### What this does and does not license

- It does **not** show geometric composition is inferior. The opposite, once
  sampling is adequate.
- It does show that autonomous open-world discovery is not achievable on this
  corpus and feature space by any distance-based rule, which is why the branch
  was abandoned rather than repaired.
- Retrying on an easier corpus was refused. That is the move that produced the
  v12 record shown above to be corpus-specific.

M79–M86 proceed unchanged under the original nameability frame. Nameability does
not depend on detecting unseen classes, so N1 does not constrain it.

---

## M79 — Acceptance reframe

**Registered question.** On what may v13 claim a result, decided before any
result is measured? Unconditional; blocks all architecture work.

**Deliverables.** `analysis/ACCEPTANCE_CRITERIA_v13.md` and
`analysis/CLAIM_LEDGER_v13.md`. Both committed before M80 begins.

### The substantive change — L2 is restated corpus-relative

The v12 frame gated open-set competence at "unknown recall … at least equal to
the strongest support control (currently the v7 low-rank Gaussian at **87.0%**)."
That figure was established on 8-class CIFAR-10 and then carried forward as if it
were a property of detectors.

N1 makes that untenable. On 128-class DomainNet, with the instrument validated at
both ends, **no method reaches 25%** — kNN gets 19.86%, nearest class mean
12.05%, the geometric head 23.86%. Retaining an 87.0% absolute bar would have
closed the program on a corpus mismatch rather than on its hypothesis, and would
have done so while the geometric head was _beating_ every freely composable
control available.

L2 therefore **remains gating**, with the threshold expressed against controls
measured on the identical corpus, features, and class count:

| Component                                                                               | Role                                                                                       |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Unknown recall at or above the strongest freely composable control (kNN, NCM)           | Gating; ties are failures                                                                  |
| AUROC at or above the same controls                                                     | Gating; recall at a threshold cannot separate a bad score from a bad threshold             |
| Positive control (far-field, near 1.0) and negative control (held-out knowns, near 0.5) | **Mandatory. A measurement lacking either is inadmissible.**                               |
| Absolute recall                                                                         | Reported, not gating, so a relative win at 24% is never mistaken for a deployable detector |

Every open-set number is now reported as `(recall, AUROC, corpus, known-class
count, samples per fitted dimension)`. A bare percentage is inadmissible.

### The other registered changes

I5 becomes **primary and gating** at ≥ 2× chance and strictly above kNN. I2 is
demoted to a non-gating structural check — v12 satisfied it at `1.14e-13` while
I5 sat below the kNN control, which is the failure mode the frame must price. I4
is demoted from gating, because v12 established the closed-form counterfactual is
structurally unavailable for multiclass anisotropic quadratics; gating on it
tests the model family rather than the hypothesis. L1's 1.0-point gate becomes a
3.0-point reported tolerance. MLP + SHAP and MLP + Integrated Gradients are
registered as controls, because the program's stated goal is to beat a neural
network on explainability and it has never measured one.

### Two gaps closed that v12 left open

1. **A deployment context now exists.** v12 specified none, which is why its
   1.0-point tolerance could never be justified — a tolerance is meaningless
   without a use for it. v13 registers assisted triage: a domain expert reviewing
   flagged decisions, with a 30-second and 10-active-atom explanation budget. The
   3.0-point L1 tolerance is derived against it, and is licensed only if
   `p(caught | explanation)` is measured. That quantity has never been measured
   by this program, so **every accuracy-for-interpretability trade claim is
   blocked until it is.**
2. **An outcome taxonomy now exists.** Prior versions closed at lettered outcomes
   (v6.1 D, v7 C, v8 D … v12 E) but no surviving document defines what the
   letters mean, so they are treated as opaque historical labels. v13 defines A–F
   for itself, and registers **Outcome C — a characterised frontier without
   dominance — as a success**, so that failing to dominate creates no pressure to
   keep altering the setup until a win appears.

### Prior-art audit — conjunction narrowed, not displaced

Audited for displacement: sparse autoencoders (Cunningham; Bricken), Label-free
CBM, LaBo, SpLiCE, Rudin, Rashomon sets, Outlier Exposure, VOS/NPOS, Fang et al.

The conjunction survives but is **materially narrowed**. M80 and M82 now claim no
methodological novelty — SpLiCE and Label-free CBM occupy that ground directly.
M84's contribution is the quantified exposure ladder, not the existence of the
effect. Rashomon-set results mean M81 must not claim surprise at parity. Fang et
al. bounds the whole open-set line and is consistent with N1.

### Corpus policy registered

ImageNet is **disqualified for any novelty measurement**: DINOv2's LVD-142M was
built by retrieval seeded with ImageNet-22k and ImageNet-1k and contains
ImageNet-1k images, so held-out ImageNet classes are not novel to this backbone
and a favourable result would be uninterpretable. It is **permitted for
nameability** from M82, where the WordNet hierarchy is a genuine asset and no
novelty is claimed. Corpus substitution to rescue a failing gate is prohibited
outright — N1 established that this program's best open-set numbers were corpus
artifacts.

Because M80 and M81 need no new data, the corpus question is **deferred to M82**
rather than settled by an expensive rebuild before the decisive measurement.

---

## M80 — Sparse concept dictionary

**Registered question.** Do frozen DINOv2 embeddings admit an overcomplete
sparse decomposition at a reconstruction fidelity sufficient to preserve
downstream accuracy?

**Registered gate.** "Advance if linear-probe accuracy on codes is within 3.0
points of the raw-feature probe at mean active atoms `<= 64`. Otherwise sweep
`m` and `k` once, then close the arm."

**Artifacts.** `logs/results/v13/m80_sparse_dictionary/`;
`experiments/tier4/eval_v13_m80_sparse_dictionary.py`; 9 focused tests.
Corpus `logs/results/v13/domainnet_large/`, index verified at
`a6485f90…bbf85`, 512 fit and 64 evaluation samples per class over 128
classes, seed 11, final labels sealed.

### Result — the gate passes, and the cell that passes it is inadmissible

| m    | k   | R² held-out | probe       | random-dict probe | margin    | deficit  | atom bits | shuffled bits | dead |
| ---- | --- | ----------- | ----------- | ----------------- | --------- | -------- | --------- | ------------- | ---- |
| 2048 | 16  | 0.6269      | 52.637%     | 40.051%           | +12.59    | 8.67     | 3.87      | 5.13          | 0.0% |
| 2048 | 32  | 0.6846      | 53.516%     | 47.827%           | +5.69     | 7.79     | 4.82      | 5.98          | 0.0% |
| 2048 | 64  | 0.7485      | 55.273%     | 53.564%           | +1.71     | 6.03     | 5.52      | 6.49          | 0.0% |
| 4096 | 16  | 0.6243      | 54.993%     | 45.715%           | +9.28     | 6.31     | 3.07      | 4.13          | 0.9% |
| 4096 | 32  | 0.6793      | 56.152%     | 53.809%           | +2.34     | 5.15     | 4.16      | 5.21          | 0.0% |
| 4096 | 64  | 0.7449      | 57.495%     | 58.435%           | **−0.94** | 3.81     | 5.00      | 5.98          | 0.0% |
| 8192 | 16  | 0.6181      | 58.313%     | 49.634%           | +8.68     | 2.99     | 2.28      | 3.11          | 4.8% |
| 8192 | 32  | 0.6721      | **60.791%** | 57.019%           | **+3.77** | **0.51** | 3.38      | 4.24          | 0.1% |
| 8192 | 64  | 0.7430      | 61.206%     | 61.230%           | **−0.02** | 0.10     | 4.34      | 5.23          | 0.0% |

Raw-feature probe bar: **61.304%** balanced accuracy, 128 classes.

The registered gate passes. Its best eligible cell is m=8192, k=64 at a deficit
of **0.10 points** against a 3.0-point tolerance, with mean active atoms exactly
64 at the registered ceiling.

**That pass is not evidence for H80.** A dictionary of the same shape with
random unit-norm atoms and no training at all reaches **61.230%** on the same
probe — `−0.02` points _better_ than the trained one. At k=64 the code is a
sparse random projection carrying the feature vector nearly intact, and the
linear probe is reading the projection, not the dictionary.

### The dissociation that makes this diagnosable

Reconstruction and probe accuracy come apart completely at k=64. The random
control's held-out R² is **negative** there — `−0.3137` at m=4096 and `−0.6597`
at m=8192, worse than predicting the mean — while its probe accuracy _ties the
trained dictionary_. A representation that reconstructs worse than a constant
supports the same linear decoding. Probe accuracy on top-k codes is therefore
not a measure of decomposition quality at large k, and the registered fidelity
operand is load-bearing only where the control separates.

This is the M77 failure mode in a new place: an operand that moves for reasons
unrelated to the quantity it is named after. It was caught only because a
control the plan did not require was measured anyway.

### Registration note N80.2 — the gate lacks a control clause

The registered gate selects the best-accuracy cell under a sparsity ceiling. It
does not require that cell to beat its own null. Both conditions were measured,
so the defect is recoverable here, but the gate as written would have certified
a random projection.

**The gate is not amended after the fact and its pass stands as registered.**
The following restriction is recorded instead, and it binds M81:

> A cell is admissible as evidence for H80 only if its probe accuracy exceeds
> the random-dictionary control on the identical split. Two cells qualify:
> m=8192/k=32 (deficit 0.51 pt, margin +3.77 pt) and m=8192/k=16 (deficit 2.99
> pt, margin +8.68 pt).

**M81 carries m=8192, k=32.** It is the only cell that is simultaneously inside
the fidelity tolerance with room to spare and clear of its null. m=8192/k=16 is
retained as the interpretability-favouring alternative: it sits at the tolerance
boundary (2.99 against 3.0, a margin too thin to survive a seed change) but has
the lowest atom entropy in the grid and the largest margin over the null.

### Sparsity trades against learnedness, monotonically

Reading the k columns at fixed m, every increase in k lowers the deficit and
lowers the margin over the null together. At m=8192 the deficit falls 2.99 →
0.51 → 0.10 while the margin falls +8.68 → +3.77 → −0.02. The apparent fidelity
win is bought with exactly the property H80 asserts. This is the substantive
finding of the milestone and it was invisible to the registered operand set.

### Monosemanticity is reported, not established — and the estimator is biased

Per N80.1, mean atom label entropy was added as a reported, non-gating operand
because none of the four registered operands measure the monosemanticity H80
asserts. The shuffled-label control discriminates in every cell, so the operand
is not vacuous, and entropy falls as m rises (5.52 → 4.34 bits at k=64), which
is the expected direction.

**The absolute values are not interpretable.** The evaluation split holds 8,192
rows; at k=16 that is 8,192 × 16 / 8,192 = **16 activations per atom on
average**, so a 128-way label distribution is estimated from about 16 draws and
the observable entropy is capped near log2(16) = 4 bits regardless of the truth.
This is why even the shuffled control reads 3.11 bits rather than the 7.0-bit
uniform bound. The operand is admissible **only as a within-grid comparison
against its own shuffled control at matched counts**, never as an absolute
purity figure. M82 must not quote these numbers as monosemanticity scores.

### Caveats recorded

- **Convergence is incomplete.** All nine cells were still descending at epoch
  40, losing 0.015–0.024 over the final ten epochs. Ordering across cells was
  stable over the last quarter of training, so the comparison holds, but every
  R² is a lower bound and a longer budget would move all of them.
- **Dead atoms are reported, never resampled.** Peak dead fraction is 4.8% at
  m=8192/k=16. Resampling is standard SAE practice and was deliberately omitted:
  dead fraction is a registered operand, and repairing an operand before
  measuring it is how the v12 probe objective survived four milestones.
- **The 64-atom ceiling conflicts with the M79 deployment context**, which
  registered ≤10 active atoms for a 30-second read. Both admissible cells (16
  and 32 atoms) exceed it. The two figures were registered for different things
  — k is the encoding width, the deployment budget is the explanation width —
  but M81 must report explanation length in atoms actually cited per decision
  and meet the 10-atom budget there, not inherit k.

### Determinism defect found and contained

Measured, not assumed, on the largest cell:

| torch threads | seconds/epoch | two identical fits agree? |
| ------------- | ------------- | ------------------------- |
| 8             | 8.0           | **No**                    |
| 1             | 25.1          | Yes                       |

Multi-threaded CPU reduction ordering makes the fit irreproducible. The 3.1×
speedup would have silently broken byte-identical replay on gated evidence. M80
runs single-threaded workers inside parallel processes, which recovers the
throughput without the defect, and the contract is registered in
`experiments/configs/v13/m80_sparse_dictionary.json` and regression-tested.

### Verdict

**H80 gate passed as registered.** The arm advances to M81 at m=8192, k=32,
under restriction N80.2. H80's fidelity clause is supported at that cell; its
monosemanticity clause remains **unestablished**, measured only by a biased
comparative operand, and M82 inherits the burden.

---

## M81 — Sparse head and the decisive I5 measurement

**Registered question.** Is a head reading the M80 sparse atoms substantially
more forward-simulable than a dense head _of comparable accuracy_?

**Registered gate (as amended by R4).** A conjunction over two task widths,
neither reportable alone. **I5-8**: eight classes frozen under seed 8101, chance
12.5%, original rule retained — ≥40% confirms, 25–40% partial, ≤25% refutes.
**I5-128**: full width, chance 0.781%, must exceed the re-measured kNN control
by more than the seed spread.

**Artifacts.** `logs/results/v13/m81_sparse_head/`;
`experiments/tier4/eval_v13_m81_sparse_head.py`; 15 focused tests. Dictionary
m=8192, k=32 per N80.2, refit per seed so that 11/23/37 vary the learned basis
and not only the head. 512 fit and 64 evaluation samples per class, final labels
sealed. Three seeds, 26.8 minutes.

### Result — the atom head wins at 8-way and has no admissible arm at 128-way

I5 is balanced accuracy of a probe predicting the model's own output from its
explanation, identity withheld. Every arm's explanation is reduced to the same
11-column form (N81.6), so no arm is handed a wider vector than another.

| width      | best atom arm within budget | its accuracy | cited atoms | **I5**     | kNN    | best dense control | shuffled null |
| ---------- | --------------------------- | ------------ | ----------- | ---------- | ------ | ------------------ | ------------- |
| **I5-8**   | `sparse_linear_budget_256`  | 84.83%       | **5.7**     | **40.22%** | 26.37% | 35.14% (MLP+IG/EG) | 12.03%        |
| **I5-128** | **none admissible**         | —            | —           | —          | 3.31%  | 9.42% (RBF)        | ~0.8%         |

At 8-way the result is real and clean: the sparse head beats kNN by 13.9 points,
beats the strongest dense explanation by 5.1 points, sits 28.2 points above its
own shuffled null, cites under 6 atoms against a 10-atom budget, and gives up
only 4.7 accuracy points to the MLP. This is the first v13 arm to beat a dense
control on the primary axis.

At full width **no atom arm is simultaneously accuracy-competitive and readable
within budget**, in any of the three seeds. The verdict is
**`task_width_artifact`**, the dominance claim is blocked, and only a frontier
claim survives.

### Why 128-way fails: the budget and accuracy cannot be met together

| atom arm at 128 classes     | accuracy | cited atoms | I5    |
| --------------------------- | -------- | ----------- | ----- |
| `metric_field_shrinkage_1`  | 66.40%   | 32.0        | 2.51% |
| `sparse_linear_l1_0.0`      | 60.84%   | 32.0        | 6.38% |
| `sparse_linear_l1_0.3`      | 56.67%   | 15.7        | 7.96% |
| `sparse_linear_budget_1024` | 43.90%   | **5.1**     | 4.36% |
| `sparse_linear_budget_512`  | 35.27%   | **3.4**     | 4.45% |
| `sparse_linear_budget_256`  | 21.94%   | **2.2**     | 4.28% |

kNN reaches 66.13% accuracy. Every arm that meets the 10-atom budget is 22 to 44
points below it; every arm that is accuracy-competitive cites 32 atoms. The
trade is monotone and there is no point on it that satisfies both registered
conditions. Separately, the dense **RBF control reaches I5 9.42%, higher than
any atom arm at this width** — so even setting the budget aside, sparsity does
not buy simulatability here.

### The first gate output was wrong, and how

The run initially returned **`confirmed`**. It was carried by
`decision_list`, which at 128 classes has **15.74% balanced accuracy** against
kNN's 66.13% and emits only **36 of 128 distinct classes**. Its majority
baseline was **2.77%** where every other arm's was 0.78% — the probe was reading
a collapsed prediction marginal, not an explanation. Its margin over that
baseline (+3.60) was in fact _below_ the arms it appeared to beat
(`sparse_linear_l1_0.3` +7.18, RBF +8.64).

Two conditions were missing from the selection rule, and **both are stated in
H81 itself** rather than being new constraints:

- **N81.7 — accuracy comparability.** H81 says "more simulable than a dense head
  _of comparable accuracy_". An arm 50 points below the best dense control
  cannot carry the verdict. Floor set at 5 points below the best dense control.
- **N81.8 — prediction-collapse screen.** An arm must emit at least half the
  label space. `degenerate_single_prediction` caught only the limiting case of
  exactly one class, which is far too weak.

The tolerance was chosen with the data in view, so the verdict is recomputed
across a range of tolerances and only a stable conclusion is claimed:

| comparability tolerance | I5-8 best arm              | I5-8   | I5-128 admissible arms  |
| ----------------------- | -------------------------- | ------ | ----------------------- |
| 2.5 points              | none                       | —      | **none**                |
| 5 points                | `sparse_linear_budget_256` | 40.22% | **none**                |
| 10 points               | `sparse_linear_budget_256` | 40.22% | **none**                |
| 15 points               | `sparse_linear_budget_256` | 40.22% | **none**                |
| unconstrained           | `sparse_linear_budget_256` | 40.22% | 4.52% (43.90% accuracy) |

The 128-way failure holds at every tolerance. Only by dropping accuracy
comparability entirely does an arm appear, and it still loses to RBF's 9.42%.
The 8-way result is stable from 5 points upward but vanishes at 2.5, so it does
depend on allowing roughly five points of accuracy headroom — recorded, not
hidden.

### The 8-way "confirms" is marginal and must be reported as such

| seed | I5-8   | clears the registered 40% bar? |
| ---- | ------ | ------------------------------ |
| 11   | 38.43% | no                             |
| 23   | 42.36% | yes                            |
| 37   | 39.87% | no                             |

Mean 40.22% against a 40.00% bar. **The margin is 0.22 points and the seed
spread is 3.93 points — eighteen times larger. One seed of three clears the
bar.** The gate records `eight_way_margin_exceeds_seed_spread: false`. The
verdict is reported as "confirms" because that is the registered rule applied to
the registered statistic, but a bar cleared by less than a fifth of its own
noise is not a robust confirmation, and no downstream claim may treat it as one.

### What R4 bought

Under the v12 rule this would have read as a clear win: 40.22% against GEODE
17.737%, RBF 22.772% and kNN 25.246%. Amendment R4 required the corpus-relative
width to be measured alongside, and that width fails outright. **The amendment
did exactly the job it was written for**, one milestone after the same defect
class invalidated L2 and N80.2.

### Defects found and fixed during construction

All four produced plausible-looking numbers, which is why each is pinned by a
regression test rather than merely corrected.

- **N81.3** — explanations were not standardised, so lbfgs never converged at
  `max_iter=2000` and I5 measured optimiser failure. Now standardised on
  probe-training rows only, identically for every arm.
- **N81.4** — the L1 penalty was added to the loss and differentiated by Adam,
  which never yields an exact zero: coefficients shrank uniformly, so accuracy
  collapsed to 14.89% while atoms cited per decision stayed at 30. Replaced with
  a proximal soft-threshold.
- **N81.5** — the hard atom budget ranked candidates by coefficient magnitude,
  which selects rare atoms precisely because rarity buys large weights. That
  head cited 0.6 atoms per decision at 13.7% accuracy. Now ranked by
  contribution mass.
- **Metric-field variance floor** — an absolute floor of 1e-6 gave precision 1e6
  to any atom a class never activated, so one unexpected atom outvoted the whole
  explanation and the head sat at exact chance. Replaced with a relative
  shrinkage prior; the head then reached 66.40%, the best accuracy of any arm.

The sparse linear head was validated against M80's independently implemented
probe on identical codes: **0.6465 against 0.6436**. Without that check the
early low numbers would have been read as a finding about sparse bases rather
than as a defect.

### Caveats recorded

- **N81.1 — the SHAP arm is expected gradients, not KernelSHAP.** `shap` is
  absent from the frozen replay `.venv` and installing it would break the sealed
  M73/M77 hashes; KernelSHAP is intractable at this width. Named as what it is
  throughout. IG and expected gradients land within 0.4 points of each other at
  both widths, so the substitution is not load-bearing.
- **N81.2 — this protocol cannot test nameability.** Identity is withheld, so
  M81 tests whether _sparsity_ aids simulatability, not whether atoms are
  nameable. The naming claim remains M82's burden in full.
- **The RBF control is a Nystroem map plus a linear head**, not v12's SVC, which
  is intractable at 128 classes. It cites 2048 components — it is the least
  readable arm in the study and its I5 lead at full width comes with no
  deployment story.
- **The decision list is weak everywhere** (78.91% at 8-way, 15.74% at 128-way)
  and its rule count saturates well below the cap. Reported as a negative
  result for that head, not tuned.

### Verdict

**H81 gate not passed. Conjunction verdict `task_width_artifact`.** Sparse atom
explanations are substantially more forward-simulable than dense ones **at 8-way
width and a 10-atom budget** — 40.22% against a 35.14% best dense control — and
that finding is marginal against the registered bar at one seed of three. At the
corpus's own 128-way width the claim fails on both required conditions.

The dominance claim is blocked. What survives is a **frontier claim**: on the
(accuracy, explanation length) plane the sparse head occupies points no dense
control reaches, at a cost in accuracy that grows steeply with task width. M82
inherits the nameability burden, and must not quote the 8-way number as a
general result.
