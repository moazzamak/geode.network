# GEODE System Blueprint — measured best-pieces registry (12 Aug 2026)

Purpose: record the _best measured option per pipeline stage per use case_, so the
system can be assembled from winners instead of re-deciding each time. Every selection
is backed by sealed evidence (path in the registry). Nothing here claims novelty —
this is the engineering consolidation of what we measured. Items marked **PENDING**
are in flight (M133/M134 frontier) and are not yet selected.

**Epistemic status (14 Aug 2026).** The selections below are the best measured
options FOR THE SEALED CONSTRUCTION on the sealed corpus. They are not bars
against new designs: per `LESSONS_ARCHIVE_v22.md`, no negative found here may be
cited to rule out a new approach outside its measured scope. This document is an
assembly guide for the sealed system, not a prohibition list for re-exploration.

---

## 1. Reference configuration (the assembled system)

The default "workable GEODE system" for a bounded, structured task set:

| Stage               | Selection                                                                                                          | Why (evidence)                                                                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Corpus              | Fixed shared subsample (digest-pinned), split by seed                                                              | Sealed corpus; byte-reproducible                                                                                                               |
| Encoder             | Whitened patch dictionary, **atoms = 6144** (width 24576), prefix-of-seeded-pool construction                      | Atoms axis saturates at ~6144 (M126); beyond is a flat/declining ridge                                                                         |
| Head                | **Closed-form ridge, λ = 1.0**, standardised + intercept                                                           | Exact solve = deterministic; depth arms lost (M115); objective axis flat (M136: λ ∈ [0.01, 10] ranges 0.005)                                   |
| Data                | Full available data                                                                                                | **Data is the lever** (steep Q(n), M116/M125; +50% rows -> +0.0161 at 6144 atoms, M140; full corpus 410k rows -> 0.2614, M141)                 |
| Specialists         | Per-domain 512-atom models **when the task set is multi-domain**                                                   | Super-additive per domain, ~5.6× fewer MACs (M119/M124)                                                                                        |
| Routing             | `ContractGatedRouter`: contract check → cheapest primitive → learned fallback; **reject gate** for out-of-contract | 100% out-of-contract rejection at zero learned cost (M130)                                                                                     |
| Programmatic layer  | Contract checks + math kernels as `ProgrammaticPrimitive`s                                                         | Zero-parameter, fingerprint-compatible (B1/B2)                                                                                                 |
| Sequence (optional) | `ProgrammaticMemory`, **window ≈ 4**                                                                               | Measured perplexity optimum on a constrained DSL (M131)                                                                                        |
| Honest framing      | Matched-cost frontier vs the DINOv2-small ladder at ~254.6M MACs                                                   | 0.2614 @ 254.6M (full corpus) beats dense r56 (0.2450 @ 367.5M); sparse wins at matched or lower cost; never absolute vs r70+ (M116/M126/M141) |

---

## 2. Stage-by-stage selection matrix

| Stage                       | Options tried (all measured)                                                                             | Best for use case                                                                                                                                                                                                                                                                                                          | Evidence         | Rejected (why)                                                                                                                                                                       |
| --------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Dictionary size**         | 1536 / 3072 / 6144 / 8192 / 12288 / 16384 atoms                                                          | Classification: **6144**                                                                                                                                                                                                                                                                                                   | M117, M126       | >6144: flat/declining ridge; saturation is structural (M128: eff-rank flat ~7.8)                                                                                                     |
| **Dictionary construction** | Seeded-prefix rule; appended re-draws                                                                    | Prefix of the sealed whitened-patch pool                                                                                                                                                                                                                                                                                   | M108/M117/M126   | Extended draws add nothing past saturation                                                                                                                                           |
| **Quantization**            | float32 codes; 108/216-bit Hamming (random/ITQ)                                                          | Quality: **float**. Cheap encode: binary (cost-only)                                                                                                                                                                                                                                                                       | M118, M122       | Binary bit loss is intrinsic, never closes (0.18 vs 0.22)                                                                                                                            |
| **Head**                    | closed-form ridge λ=1; depth stacks (L=1..3); penalty ladder λ∈{0.01..10}; smoothed targets; batch hinge | **Ridge, λ=1**                                                                                                                                                                                                                                                                                                             | M115, M136       | Depth does not lift the ceiling (B1 false); objective axis closed — λ flat to 0.005, smoothing ≡ penalty reparametrisation (verified exactly), hinge unconverged at registered cells |
| **Grouping**                | global model; per-domain specialists                                                                     | Multi-domain: **specialists** (512 atoms)                                                                                                                                                                                                                                                                                  | M119, M124       | Global wins only where domains are homogeneous                                                                                                                                       |
| **Routing**                 | flat certified top-k; contract-gated                                                                     | Robustness/out-of-contract: **contract-gated + reject gate**                                                                                                                                                                                                                                                               | M130             | Flat router computes garbage on out-of-contract                                                                                                                                      |
| **Prediction certificate**  | MSE-proxy (spectral); per-class margin model                                                             | (none — closed)                                                                                                                                                                                                                                                                                                            | M121, M123       | Neither predicts the argmax crossing; kept as diagnostics only                                                                                                                       |
| **Sequence memory**         | count memory windows 1/2/3/4/8; (reservoir, fixed-attention PENDING)                                     | Constrained next-token: **window ≈ 4**                                                                                                                                                                                                                                                                                     | M131             | w8: 114× footprint for a perplexity regression                                                                                                                                       |
| **Next-token family**       | additive count model; tiny transformers; reservoir + ridge; fixed-attention + ridge                      | **RESOLVED (M133+M134):** trained transformers win at EVERY matched footprint (2.56–2.80 ppl); count model second (3.32–5.78); reservoir (26.6–29.7) and fixed-attention (34.8–35.1) are an order of magnitude worse. Count kept only for zero-training/integer-only/KB-scale property; reservoir/fixed-attention rejected | M131, M133, M134 | count not competitive in quality (M133); fixed constructions ~10× worse than count (M134)                                                                                            |

---

## 3. Use-case recipes

**Recipe 1 — cost-matched classification (the headline claim).**
Whitened patch dictionary (6144 atoms) → closed-form ridge (λ=1) → full data → report
against DINOv2-small r42 at matched per-image MACs. Wins at matched cost (0.2246 vs
0.1972); absolute gap disclosed.

**Recipe 2 — multi-domain / per-group.** Per-domain 512-atom specialists + ridge each,
routed by task/domain fingerprint. ~5.6× fewer MACs than the global arm, super-additive
per domain (4/6), best domain 0.3245.

**Recipe 3 — robustness / out-of-contract.** `ContractGatedRouter` with a reject-gate
primitive (negated contract) in front of any learned model. Out-of-contract inputs are
rejected with zero learned forward passes (measured 100%).

**Recipe 4 — tiny constrained text (autocomplete/DSL).** `ProgrammaticMemory` at
window ≈ 4, add-one smoothing, per-token integer ops only, KB-scale footprint. **The
M133 frontier measured that a trained tiny transformer beats this at every matched
footprint (2.56–2.80 vs 3.32–5.78 ppl)** — so this recipe is only for deployments
where you cannot train a transformer at all (zero-training, integer-only,
programmatic, no GPU): its value is the property, not the perplexity. Reported
perplexity vs baselines; transformer comparison disclosed per the frontier.

**Recipe 5 — cheapest encode.** Binary 108/216-bit Hamming codes only when encode cost
dominates and ~4-pt accuracy loss is acceptable (cost-only route, permanent).

**Recipe 6 — hybrid learned + programmatic.** Any learned classifier + programmatic
primitives (checks, math) behind one fingerprint interface, dispatched by the
contract router; programmatic path costs zero learned forward passes.

---

## 4. Open selections (registered in v21, `RESEARCH_IMPLEMENTATION_PLAN_v21.md`)

- **M133/M134** — RESOLVED (sealed): the next-token family selection at each
  footprint is the trained tiny transformer; count kept for property-only
  deployments; reservoir/fixed-attention rejected.
- **M136** — SEALED (13 Aug): head-objective axis closed; λ = 1.0 ridge remains
  the head (λ flat to 0.005 over [0.01, 10]; smoothed targets provably ≡ penalty
  reparametrisation; hinge unconverged at the registered cells).
- **M137** — HTN-structured router over specialists (the orchestration layer): does
  plan-structured dispatch beat flat/contract routing on robustness and cost?
- **M138** — SEALED (13 Aug): seed direction CLOSED from both sides — score-level
  ensemble 0.22110 below the single 6144-pool (0.22487) at matched MACs; joint
  eff-rank 7.76 = single-pool ~7.8. The rank-8 ceiling is the whitened patch
  space's, not the pool draw's.
- **M139a** — SEALED: domain router on the codes 75.6% (style-adjacent
  confusions: painting→real 47%).
- **M139b** — SEALED: assembled specialist buy-back FAILS the A5 pattern (routed
  wins 3/6 vs dense r28; oracle ceiling 0.2050 below the global head 0.2246).
  Specialists keep their per-domain property + cost story, never pooled accuracy.
- **M140** — SEALED: data extension PASSED — +50% rows → +0.0161 (0.2407);
  crossover vs dense r42 widens to +0.0435.
- **M141** — SEALED: data escalation PASSED — the full corpus (410k rows) →
  0.2614 (+0.0207 over 207k); beats dense r56 at 31% fewer MACs. The data lever
  holds to the corpus's full extent; the corpus is now exhausted.

---

## 5. Negative results that shaped the selections (kept, not hidden)

- Depth (M115): closed — single-stage ridge is the head.
- Head objective axis (M136): closed — λ flat to 0.005 across [0.01, 10];
  smoothed targets are provably a penalty reparametrisation (verified exactly,
  match share 1.0); the batch hinge did not converge at the registered cells
  (kept as a one-sided instrument, not a refutation of margin objectives).
- Binary quality (M118/M122): closed — cost-only, permanent.
- Spectral/margin certificates (M121/M123): closed as predictors — the argmax
  crossing is not spectral-predictable; kept as explanatory diagnostics.
- Atoms growth past 6144 (M126): closed — flat ridge; the lever is data.
- Training-cost ≥10× claim (symmetric accounting): failed — never used in a headline.

---

## 6. How the pieces assemble (the library surface)

`Encoder` (whitener + dictionary + pooling, seed-pinned) → `codes` (float32 memmap) →
`RidgeAccumulator` head (exact solve) → `SpecialistRouter` (per-group) →
`ContractGatedRouter` (contracts + programmatic primitives + reject gate + decision
log) → optional `ProgrammaticMemory` (sequence). Every object carries a fingerprint
(`InputSpec`/`OutputSpec`) and a registered cost class, so dispatch is uniform and
out-of-contract is cheap by construction.
