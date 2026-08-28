# Lessons Archive v22 — sealed findings, scope-annotated (14 Aug 2026)

**Purpose.** This document archives the programme's sealed measurements as
_informative evidence_, and changes their epistemic status: no archived finding
may be used to **rule out** a new approach unless the new approach falls inside
the finding's measured scope. Findings are priors, not verdicts, outside their
scope. This archive supersedes the reading in which v21's negatives "closed"
design-space questions generally.

**The three rules (registered 14 Aug 2026, binding on all future milestones):**

1. **Scope-bound citation.** Any document citing an archived negative against a
   new design must state the finding's measured scope (construction, corpus,
   schedule) and show the new design lies inside it. A citation that skips the
   scope statement is VOID, not persuasive.
2. **Re-test on axis change.** When a new design varies the axis a negative was
   measured on, the negative becomes a _prior_ (a predicted outcome to be
   registered) — never a pre-emptive rejection. The design is measured; the
   prior's confirmation or failure is recorded either way.
3. **The archive grows, it never re-decides.** New measurements extend this
   document. Old verdicts are not rewritten; their scope is.

---

## The archive

| #   | Sealed finding                                                           | Measured scope (what it IS evidence about)               | Does NOT license (what it cannot be cited for)                                                   |
| --- | ------------------------------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1   | Depth stacks do not lift the ceiling (M115)                              | closed-form depth (L=1..3) on the sealed triangle codes  | "depth never helps any additive code"                                                            |
| 2   | Code eff-rank ~7.8, thin margin tail (M128/M123)                         | the specific 6×6-patch → triangle → global-pool code     | "all additive/construction codes are rank-limited"                                               |
| 3   | Fresh pool draws don't help (M126 flat, M138 ensemble below single pool) | draws and score-ensembles of THIS construction           | "different constructions with different pooling/scales cannot help"                              |
| 4   | Trained linear head collapses on the codes (A2)                          | linear SGD heads on THIS code, same data                 | "trained heads never help on additive codes" (dense features show the opposite)                  |
| 5   | Head objective flat; hinge unconverged (M136)                            | THIS code + the registered λ grid and hinge schedule     | "margin objectives never matter for additive systems"                                            |
| 6   | Binary 108/216-bit codes lose ~3 pts (M118/M122)                         | Hamming compression of THIS code                         | "no compression of additive codes can preserve accuracy"                                         |
| 7   | VQ/discriminative dictionary learning doesn't transfer (M113/M108)       | VQ centroids and greedy selection with the triangle code | "dictionary learning never helps additive pipelines"                                             |
| 8   | Specialists fail to assemble into a pooled buy-back (M139b)              | 512-atom A5 specialists, routed/global, THIS corpus      | "per-domain/expert additive systems never buy back" (per-domain super-additivity stands)         |
| 9   | Same-data trained stems lose to the construction (M110)                  | small-scale from-scratch stems, same data                | "learned stems always lose at any scale/pretraining"                                             |
| 10  | Learned sequence components ≈ 10× fixed ones (M133/M134)                 | the deterministic DSL, tiny transformer regime           | "fixed constructions win sequence tasks" (the reverse reading)                                   |
| 11  | Data is the lever to the corpus's full extent (M140/M141)                | THIS construction on DomainNet                           | "the absolute level transfers to other corpora" (steepness may; M108 taught transfer isn't free) |

**Positives that carry forward without demotion** (their scope was always narrow):

- The sealed measurement protocol itself (anchors, gates, evidence hashing).
- Cost-matched non-domination through dense r56 at 31% fewer MACs (M141), for
  THIS system on THIS corpus.
- The engineering pieces (M130 contract router, M131 programmatic memory,
  M133 tiny transformers) as measured components.

---

## Re-exploration charter (the open design space)

The goal stated 14 Aug 2026: **make the additive approach work for generalized
learning.** The archived findings above are informative priors for this
exploration, nothing more. The registered next step is the construction
factorial (proposed as v22): spatial-pyramid pooling, power-normalisation,
multi-scale patches, cosine patch coding, dictionary-learning interactions —
each aimed at a measured defect (rank, spatial loss, correlation), every cell
prior art (M135 audit), every cell matched in MACs to the sealed frontier,
gate: beat 0.2614 @ ~254.6M MACs.

Unmeasured axes the archive explicitly leaves OPEN (any negative cited against
them is void under rule 1): pooling structure; patch scale; coding
nonlinearity; per-patch normalisation; joint trained-ends sandwich; other
corpora; other tasks (regression, retrieval, sequence beyond the DSL).

---

## Operational rules for future milestones

- Register hypotheses + the relevant archive priors BEFORE measurement, in the
  plan of record.
- A prior confirmed → record "prior confirmed, scope extended". A prior
  refuted → record "prior refuted, scope contracted"; the refutation IS the
  contribution.
- The blueprint remains the best-measured-options registry for the SEALED
  construction; it is not a barrier list for new designs.
