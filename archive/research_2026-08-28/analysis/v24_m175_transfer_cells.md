# M175 transfer battery v1 (the M163 decision, made 17 Aug 2026)

User decision: **all four transfer directions are approved** — A, B, C, D
run as one parameterized battery. Each cell registers its task
definition and gates separately BEFORE it runs (the cells differ in
what their corpora support).

## Inventory correction (registered)

- The sealed "C4" anchors are **DomainNet-32** (cell naming in the M142
  sequence): SPM-1923 codes cached (61.65 GB train / 5.19 GB test),
  anchors 0.2605 (raw, full) / 0.2273623188405797 (p0.5 @ 138k) /
  0.2786 (p0.5 lambda0.1, full data).
- Actual text corpora on hand: wikitext-103 (80k train / 20k test
  token ids, cached) and a Wikipedia dump (400M/100M token ids).
  **No labels, and no frozen text encoder exists** — the text path is
  unbuilt (L2).

## Cells (order of execution)

| Cell   | Direction                                | Measurement                                                                                                                                                     | Status                                                                                                |
| ------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **B**  | vision→vision: DomainNet-32 → Flowers102 | ridge on Flowers102 labels with the frozen DomainNet SPM encoder vs the cached flowers backbone-features baseline                                               | SEALED 18 Aug: scoped negative (SPM 0.167 vs CLS baseline 0.990; both gates bit-exact)                |
| **A0** | FIRST text encoder build                 | additive next-token fit-and-report on wikitext-103 (the M131 machinery), anchors + held-out                                                                     | SEALED 18 Aug: gate fired (inverts beyond w=2); best w2 9.87-9.92; no arm selected; A/D pin their arm |
| **A**  | text→text transfer                       | the A0 encoder's codes on a held-out text slice (the C4-text corpus itself does not exist on disk; A is text→text between wikitext splits + the Wikipedia dump) | SEALED 18 Aug: HOLDS (out 11.09 vs in 10.66, gap 1.04x, OOV 6e-5; arm = uniform-w2)                   |
| **D**  | license-clean text                       | fit-and-report on the Wikipedia dump with the A0 encoder; licensing posture recorded                                                                            | SEALED 18 Aug: held-out 9.5142 (beats A's 11.09 transferred read; OOV 0.0); posture recorded          |
| **C**  | cross-modality routing                   | fingerprint/route-level: vision task routes to vision arms, text task routes to text arms, no modality confusion                                                | SEALED 18 Aug: guard PASS (every chain own-kind only, cross-contract query clean; router 16/16 tests) |

Per-cell task definitions, metrics, and gates are registered in
`RESEARCH_IMPLEMENTATION_PLAN_v24.md` §12 immediately before each cell
dispatches. Cell C is route-level only and carries no new data cost.

## Sequencing rationale

B first (the frozen vision encoder already exists; the target is
small), then A0 (smallest text corpus on disk), then A/D (reuse the
A0 encoder), then C (needs text arms). The RunPod M176c rental is a
separate vision task and is unaffected by this order.
