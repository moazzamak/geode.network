# BLOCKER ANALYSIS — prior work and cross-domain leverage (17 Aug 2026)

Registered second dispatch of the M164 instrument, aimed at the seven
buildout blockers (v24 §9 L1–L7). Anchor gate passed on stage 1
(6/6), zero rate-limits, one query empty (b12 — recorded, not
interpreted). Evidence: `logs/results/v23/m164b_blocker_solutions/`
config: `experiments/configs/v23/m164b_blocker_solutions.json`.

Standing rule: hits are **displacement-only**. "Solved-for-us" below
means _published constructions exist that we can measure against_ —
never that we are done; our contribution remains the matched-cost
measurement.

---

## L1 — the frozen-code ceiling (better codes)

**Intended approach:** the M176c ladder — deep-patch SPM → Fisher
vectors on deep patches → budget-capped trained encoder.

**Prior work found:**

- The deep-patch code family is mature: SPM-VLAD over CNN features
  (arXiv:1603.09046), deep dictionary learning (arXiv:2012.12509,
  arXiv:1912.10804).
- **New candidates:** learned _local descriptors_ then aggregate
  pooling — GeoDesc (arXiv:1807.06294) and DELG-style aggregation
  (arXiv:2007.13172) — frozen at deployment, MACs comparable to SPM
  bins. This is a M176c candidate we had not listed and should be
  tested at rank 2.
- Efficient ViTs (S2AFormer arXiv:2505.22195; structural
  reparameterization arXiv:2511.19718) are the trained-code end of the
  ladder; the M91 lesson (measure each family against its own fp32)
  applies.
- Scattering networks (anchor a3) remain the signal-processing route
  to fixed filters; the NTK-ridge literature (anchor a5) is the
  theory connecting trained nets to closed-form ridge.

**Verdict:** not fixed for us, but the family is mature — the open
question is which _cheap_ code wins at matched MACs, which is
measurable now. Add learned-local-descriptor pooling as M176c
candidate 2b (frozen GeoDesc/DELG features → SPM pooling → ridge).

**Cross-domain leverage:** signal processing (wavelets, scattering
filters); information theory (I(X;Y) bounds quantify the ceiling the
M176a probe approaches); physics coarse-graining (renormalization-style
multi-scale codes).

## L4 — the fit is quadratic in width

**Intended approach:** shipped chunked Gram + in-place LU; escapes
were registered from the first search (D&C KRR arXiv:1305.5029;
two-level preconditioning arXiv:1806.05826).

**Prior work found (the toolbox is essentially complete in print):**

- **Randomized NLA / sketching:** Sketch'n'Solve package
  (arXiv:2409.14309); sketched Krylov ridge-path (arXiv:2210.12212);
  localized sketching (arXiv:2003.09097); lower bounds
  (arXiv:2204.06653); Falkon sketching+preconditioning
  (arXiv:1611.03220); robust randomized preconditioning
  (arXiv:2304.12465); block-Nyström low-rank (arXiv:2506.17556).
- **Krylov solvers:** hybrid LSMR (arXiv:2409.09104); Arnoldi least
  squares (arXiv:2407.05945); randomized Krylov f(A)b
  (arXiv:2212.12758).
- **Quantum-inspired classical algorithms** (these run on ordinary
  hardware): dequantized low-rank arithmetic (arXiv:1910.06151);
  quantum-inspired linear regression (arXiv:2009.07268); QI PCA
  regression (arXiv:2010.08626).

**Verdict: solved-for-us in the useful sense** — the escape ladder
exists and is mature; M176b is a _benchmark_, not a research problem.
Measure: Falkon-style sketch+precondition vs the exact solver through
the sealed equivalence gate at 5k/15k/40k widths, then the first
infeasible width.

**Cross-domain leverage:** quantum algorithms as algorithm _design
inspiration_ (dequantization — the most practical cross-domain payoff
in this list); HPC linear algebra practice.

## L5 — numerical fragility

**Intended approach:** float64 promotion + equivalence gates (the
standing discipline after the 39% float32 weight-shift incident).

**Prior work:** mixed-precision solver strategies are a mature HPC
topic — GMRES on GPUs (arXiv:2109.01232), half-precision nested
Krylov (arXiv:2505.20719), floating-point autotuning
(arXiv:2606.08339).

**Verdict:** the physics/HPC communities solved the _technique_
(iterative refinement with low-precision preconditioners, autotuned
precisions); our contribution is applying the discipline gate to the
sealed system. M176b should include one half-precision arm.

## L2 — one corpus, one modality

**Intended approach:** cross-corpus transfer (M163) and new axes
(M175).

**Prior work:** frozen foundation features for domain generalization
(arXiv:2312.04265 and the b9 line).

**Verdict:** partial. The generic frozen-feature DG line exists, but
our specific question — does the SPM+sqrt+L2 recipe transfer with a
fresh whitener/dictionary per corpus — remains open and is exactly
M163.

## L3 — local saturation of the recipe

**Intended approach:** stop growing this axis; new axes instead.

**Prior work:** sparse-coding scalability (b11) is adjacent but not
displacing.

**Verdict:** open, and the sealed evidence (E2: the code lives in ~8
effective dimensions; M155–M158 negatives) says saturation may be
intrinsic to count codes on this corpus. The honest move is to treat
"more atoms/pools" as closed and spend compute on L1/L4 instead.

## L6 — the toolbox layer is unbuilt

**Intended approach:** v24 Phases A–C (registry, fingerprint, router,
ten gated capabilities).

**Prior work:** the task-embedding line (M164 anchors) and routing
literature; b12 returned zero hits (recorded; absence proves
nothing).

**Verdict:** this is a build, not a research question — the plan and
gates are registered; the prior art to cite is already in the
whitepaper §8. **Cross-domain leverage:** databases (transactional
registries, versioned catalogs) and information retrieval (inverted
indexes, LSH for fingerprint lookup).

## L7 — per-task cold start

**Intended approach:** encoder construction per axis + strongest-arm
fallback (I4).

**Prior work:** differentiable closed-form solvers as meta-learning
(arXiv:1805.08136 — surfaced in both searches); transductive
few-shot metric lines (b10).

**Verdict:** partial — the closed-form-solver meta-learning line is
directly adjacent and should be cited; the fallback design stays ours
to measure.

---

## Summary: what changes in the plan

1. **M176c** gains candidate 2b: frozen learned local descriptors
   (GeoDesc/DELG line) pooled with SPM → ridge.
2. **M176b** becomes a benchmark of published solvers (Falkon
   sketching+preconditioning, hybrid LSMR, one half-precision arm,
   quantum-inspired low-rank) through the sealed equivalence gate —
   not an invention task.
3. **L3** is treated as closed on the count-code axis; compute goes
   to L1/L4 probes.
4. The quantum-inspired line (arXiv:1910.06151, arXiv:2009.07268) is
   registered as the cross-domain fallback if the exact-solver wall
   arrives earlier than the sketched solvers allow.

---

## Fingerprint/routing reference ledger (M164c, 17 Aug 2026)

The user's suspicion was correct: the routing references were thin.
The M164c dispatch (5 anchors + 4 LeCun-scoped queries + 5 blockers;
anchor gate passed on stage 1) plus direct title pinning found:

- **The LeCun line exists and must be cited.** The arXiv-published AMI
  paper — Grathwohl, Wang, LeCun et al., "Introduction to Latent
  Variable Energy-Based Models: A Path Towards Autonomous Machine
  Intelligence" (arXiv:2306.02572) — proposes the **configurator**: a
  gating module that routes information and selects modules per task.
  That is a direct architectural antecedent of the GEODE router.
  I-JEPA (arXiv:2301.08243) adds task-conditioned predictors. The
  author-scoped topic queries (l2/l4) returned zero hits — recorded;
  absence in the index proves nothing, which is why the direct title
  pinning matters.
- **Routing literature (must-cite):** Routing Networks (Rosenbaum et
  al., arXiv:1711.01239); switch transformers (Fedus et al. 2022);
  DA-MoE dynamic expert allocation; hierarchical routing MoE;
  sparsely-gated MoE (Shazeer et al. 2017).
- **Task/dataset embeddings (must-cite):** unified task embeddings
  across models (arXiv:2402.14522); the dataset-similarity review,
  taxonomy and comparison (arXiv:2312.04078); the Task2Vec/Taskonomy
  line (M164 anchor a1).

**Verdict:** none of these displaces GEODE's router — the LeCun
configurator and the MoE lines are architecture proposals without
measured per-task routing results, and the embedding lines measure
similarity by statistics rather than by behavioral transfer. They are
design antecedents: they propose what v24 will measure. All are now
in the whitepaper claim ledger (§8.2).
