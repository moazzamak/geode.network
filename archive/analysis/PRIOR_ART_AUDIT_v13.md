# M87 — targeted prior-art audit of v13's unregistered findings

**Status at time of registration:** REGISTERED, NOT YET EXECUTED.
**Registered:** 30 July 2026, before any search was run.
**Predecessor:** v13 closed at Outcome C, commit `e67f88e`, M86 evidence
`74dcc3606e40034355a023a8719ff9d5547d7ca565d5abb73463f54756a1933b`.

---

## 1. Why this milestone exists

The v13 prior-art audit recorded at `analysis/CLAIM_LEDGER_v13.md` Section 4 is a
**pre-registered displacement audit of the original conjunction**. It was written
before v13 executed, and it asks one question: does any known work displace the
seven-stage composition the program set out to evaluate?

That is not the question this milestone asks. Three of v13's most interesting
results were **not predicted by the plan** and therefore were never searched
against the literature at all:

- M84 found outlier exposure **destructive**, where the registered prior-art row
  for Hendrycks et al. had assumed exposure works and that v13's contribution
  would be quantifying how much is needed.
- M85a's Simpson gap between pooled and within-domain AUROC was a diagnostic
  produced in passing.
- M85b's resolution/corpus decomposition was designed as a control, not as a
  finding.

A result that arrives unpredicted has had no opportunity to be checked against
prior work. Section 4 cannot be read as covering them, and the absence of a row
is not evidence of novelty. This milestone closes that gap.

**N87.1 — this audit may not add to v13's claims.** v13's evidence is sealed and
its ledger is closed. M87 can only ever _narrow_ or _withdraw_ a v13 claim, or
leave it standing. If a search finds that a v13 result was already known, the
result remains true and remains reported; what changes is the contribution
attached to it. No search outcome can promote a v13 finding, strengthen a
verdict, or reopen a refuted hypothesis.

**N87.2 — "not found" is not "new".** The only outcomes this audit can reach are
_displaced_, _narrowed_, _unresolved_. There is no _novel_ outcome and the word
is not used as a verdict anywhere in this document. Absence of a located source
is recorded as **unresolved**, with the searched surface stated so a reader can
judge how much weight it carries. Patent literature, non-English work, textbook
and folklore knowledge, closed industrial practice, and anything behind a paywall
this audit cannot reach all remain outside the searched surface permanently.

**N87.3 — the claim list is closed at registration.** The five claims in Section 3
are fixed by this commit. No claim may be added, removed, reworded or split after
the first search runs. Adding a claim after seeing results permits selecting the
one that happened to survive, which is the literature-search form of reporting
the seed that worked.

**N87.4 — the audit searches for disconfirmation.** Every registered query family
in Section 4 is phrased to locate work that **would displace** the claim, not work
that agrees with it. A search that returns only supporting or adjacent work has
not tested anything. Query families are fixed by this commit for the same reason
the claims are.

**N87.5 — searching stops when the registered families are exhausted**, not when
a satisfying answer appears. Every registered query runs and its raw result count
is recorded even where the hits are useless, so that a reader can see the queries
that found nothing.

**N87.6 — a primary source is required to displace.** Title and abstract are
sufficient to mark a source _live_ and to require follow-up, and are **not**
sufficient to displace or narrow a claim. Displacement requires the primary text
and a specific located passage, cited by section or figure. A source read only in
abstract is recorded as **unresolved, follow-up required**, never as clearing the
claim.

**N87.7 — the audit measures nothing.** This milestone runs no experiment, loads
no features, and recomputes no v13 figure. The numbers quoted in Section 3 are
transcribed from sealed evidence and are hash-cited. If a transcription is wrong
the audit is wrong, so each claim carries the evidence hash it was read from.

---

## 2. Vocabulary

| Verdict        | Meaning                                                                                                                            |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Displaced**  | A located primary source already establishes the claim. The v13 contribution on this point is withdrawn; the measurement stands.   |
| **Narrowed**   | A located primary source establishes part of the claim. The contribution is reduced to the stated remainder, in writing, here.     |
| **Unresolved** | The registered query families located no displacing source. The claim is not cleared; it is untested outside the searched surface. |
| **Instrument** | Not prior art. A source this program should be using as a tool or benchmark if the line is ever continued.                         |

---

## 3. The claims under audit, fixed at registration

Each claim is stated in the form that a displacing source would have to
anticipate. Numbers are transcribed from sealed v13 evidence.

### C1 — Outlier exposure degrades rejection for a union of per-class density models

> Training a union of per-class anisotropic ellipsoids on real out-group images
> does not merely fail to improve open-set rejection; it removes it. Rejection
> recall at matched known coverage is **0.11875** at the untrained zero rung and
> **≤ 0.00012** at every exposure-trained rung of the ladder, across all
> exposure-count and diversity cells. The mechanism registered by M84 is that
> acceptance is a union: a sample is rejected only if **all** 128 ellipsoids
> reject it, so training one ellipsoid to exclude a region expands others and
> ejection becomes whack-a-mole.

**Source:** `logs/results/v13/m84_exposure_ladder/evidence.json`, verdict
`ladder_flat`, `gate.baseline` 0.11874999999999998, `gate.best_mean`
0.00011574074074074075 at cell `n10_d10`, `gate.beats_baseline` false.
Four ladder cells are recorded infeasible — three in DomainNet's supply and one
in arithmetic — and the claim is made only over the cells that ran.
**Displaced if:** a primary source reports outlier-exposure-style training
degrading OOD rejection for class-conditional density, distance or one-class
models, or identifies the union-of-accepting-regions mechanism as the reason
exposure fails for such models.
**Narrowed if:** a source reports exposure failing for these models without the
mechanism, or reports the mechanism without measuring exposure.

### C2 — Domain-pooled OOD AUROC conceals within-domain performance

> Pooling novelty scores across visually distinct domains before computing AUROC
> produces **0.5851**, against a within-domain mean of **0.6580** and four of the
> five populated per-domain figures above the pooled value — 0.7092 clipart,
> 0.6759 painting, 0.6404 quickdraw, 0.8691 real, with only infograph at 0.5841
> below it. The pooled score separates domain more strongly than it separates
> novelty, so the pooled figure understates the geometry's within-domain
> behaviour and would be the number a conventional evaluation reported.

**Source:** `logs/results/v13/m85_open_set_auroc/evidence.json`,
`bf72f81de6f6bd7ed14f0f02101cfd13bd82a75bf1bf0791eada910f2910decb`,
`arms.geometry.per_domain` and `arms.geometry.within_domain_auroc`.

**Transcription correction made at registration (N87.7).** This claim was first
drafted as "below every within-domain figure". Reading the sealed evidence showed
infograph at 0.5841, below the pooled 0.5851, so the stronger form is false. The
weaker form above is what is audited. The correction is recorded rather than
silently applied, because the audited claim must be the one the evidence supports
and not the one that would have been more impressive.
**Displaced if:** a primary source identifies Simpson-type reversal or a
group-pooling artifact in OOD/novelty AUROC and recommends within-group
reporting.
**Narrowed if:** a source reports domain-stratified OOD evaluation as good
practice without identifying the reversal, or identifies the statistical effect
in a neighbouring evaluation setting.

### C3 — Cross-corpus dictionary transfer loss is resolution, not corpus

> Adding a **degraded-resolution control arm** — the native corpus resampled to
> the target corpus's resolution — separates input resolution from corpus
> identity in dictionary transfer. Here resolution costs **+0.0876** retention
> and corpus identity beyond resolution costs **−0.0097**, i.e. nothing. The
> implication is that cross-corpus transfer loss reported without a resolution
> control may be a resolution confound.

**Source:** `logs/results/v13/m85_transfer_eval/evidence.json`,
`46026f10a113c5dca015855b51bca4d2606d6af4c58b91c7a5bc63a475d94e51`, verdict
`loss_is_resolution_not_corpus`.
**Displaced if:** a primary source uses a resolution-matched or resolution-degraded
control to attribute cross-dataset representation-transfer loss to input
resolution rather than domain shift.
**Narrowed if:** a source establishes input resolution as a dominant confound in
frozen-backbone feature quality without applying it to dictionary or
sparse-code transfer.

### C4 — Exemplar-mediated concept naming channels are not independent

> On a backbone with **no joint image-text space** — DINOv2, as against CLIP —
> every route from a learned atom to a phrase passes through that atom's
> top-activating exemplars, because no direct atom-to-text comparison exists.
> Any two "independent" naming channels therefore read the same exemplars and
> differ only in what reads them. Their agreement cannot be evidence of naming
> correctness.

**Source:** Amendment R9, `analysis/CLAIM_LEDGER_v13.md`; M82 verdict
`names_unstable`.
**Displaced if:** a primary source states that exemplar-mediated concept-naming
channels are not mutually independent, or that inter-channel agreement is not
evidence of naming correctness.
**Narrowed if:** a source raises exemplar mediation as a confound without drawing
the independence conclusion.

### C5 — Simulatability metrics require a structure-matched shuffled null

> A forward-simulation or simulatability score is uninterpretable unless reported
> beside a null explanation sharing the real explanation's **structure, citation
> budget and data split**. Under such a null, v13's 128-way field separates from
> chance by only 1.4 to 8.6 points of I5.

**Source:** `logs/results/v13/m81_sparse_head/evidence.json`,
`61181c1c878cab273e66d11d5d45ad758b365766a7e83141e159c5af2cf1a6e3`;
`analysis/V13_FRONTIER.md`.
**Displaced if:** a primary source reports a human or automated simulatability
metric against a matched shuffled- or randomised-explanation null controlling
structure and budget.
**Narrowed if:** a source uses randomised-explanation controls for a different
interpretability metric, e.g. attribution sanity checks, without applying them to
simulatability.

---

## 4. Registered query families

Fixed at registration under N87.4. Executed against the arXiv API and the
Semantic Scholar Graph API by `experiments/tier4/audit_v13_prior_art.py`, which
records every query, its hit count, and the returned records verbatim.

| Claim | Family                                                                                                                                                                                                                 |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C1    | outlier exposure degrades / harms OOD detection; outlier exposure Mahalanobis; class-conditional Gaussian OOD exposure training; one-class per-class model outlier exposure failure; negative results outlier exposure |
| C1    | union of acceptance regions open set rejection; per-class ellipsoid novelty rejection; multiclass one-class classifier rejection failure                                                                               |
| C2    | Simpson's paradox AUROC; pooled versus stratified AUROC evaluation; OOD detection domain shift confound evaluation; group-conditional AUROC out-of-distribution; subpopulation AUROC reversal                          |
| C3    | image resolution confound frozen features transfer; input resolution domain shift representation quality; low resolution CIFAR upsampling ViT features; dictionary learning transfer across datasets resolution        |
| C4    | concept naming top activating exemplars confound; automated concept interpretation independence; network dissection exemplar critique; concept bottleneck naming validation without joint text space                   |
| C5    | simulatability forward simulation evaluation control; randomised explanation baseline interpretability; shuffled attribution sanity check; null model explanation evaluation                                           |

Also queried, once each, as **known-adjacent anchors** whose absence from the
results would indicate the search itself is broken: `outlier exposure`,
`generalized category discovery`, `sparse autoencoder interpretability`.

### N87.9 — instrument defects found and repaired before the audited run

Three defects were found in the search runner while bringing it up, all before
any adjudication. They are recorded because a search instrument that quietly
under-returns produces exactly the _unresolved_ verdicts a motivated auditor
wants, and a reader cannot otherwise tell an empty index from a broken query.

1. **arXiv exact-phrase syntax.** The first implementation sent
   `all:"<whole query>"`, which arXiv treats as a verbatim phrase. It returned
   **zero hits for every query including all three anchors**. Repaired to
   conjoin the query's terms.
2. **Conjunction still too strict.** ANDing every term requires a paper to
   contain all of them; `outlier exposure degrades out-of-distribution detection`
   returned zero because "degrades" is rare, not because no such work exists.
   Repaired to a **fixed two-stage rule applied uniformly to every query**: AND
   the terms; if that returns zero, re-issue the same terms ORed and sorted by
   relevance. The stage used is recorded per query. The rule is mechanical and
   applies to all queries, so it cannot be aimed at the ones whose emptiness
   would have been convenient.
3. **Rate-limiting recorded as emptiness.** The unauthenticated Semantic Scholar
   endpoint returns HTTP 429 under load. Without retry a throttled query is
   indistinguishable from a query that found nothing, which under N87.2 is the
   distinction the whole audit rests on. Retries with widening backoff were
   added, and any query that still fails is recorded in `failed_queries` and
   **is not counted as an empty result**.

**The repaired instrument re-runs every registered query, not only the ones that
returned nothing.** Applying a better instrument selectively to the queries whose
results one dislikes is the search equivalent of extending a training run until
the number improves. The registered query strings themselves are unchanged; only
how the index is asked has changed.

---

## 5. Registered stopping condition

The audit is complete when every family in Section 4 has run and every source
marked _live_ has either been read in primary text or recorded as _unresolved,
follow-up required_. The audit does **not** stop early on a displacement, because
the remaining claims are independent of it.

---

## 6. Findings

**Executed 30 July 2026.** Evidence:
`logs/results/v13/m87_prior_art_audit/evidence.json`. 37 registered queries plus
3 anchors, run against both indexes, 627 records returned, 1604 s.

### 6.1 The searched surface is materially incomplete, and one gap is demonstrable

This has to come before any finding, because it discounts every _unresolved_
verdict below.

**Semantic Scholar refused 26 of 37 queries** with HTTP 429 after retry. Per
claim: C1 7 of 8, C2 4 of 6, C3 **6 of 6**, C4 4 of 6, C5 4 of 6. arXiv answered
everything. So C3 was searched on **one index only**, and no claim was searched
on two indexes in full.

**A worked example that the search under-returns.** C5 concerns simulatability
metrics. Forward-simulation evaluation has a known literature — Hase and Bansal's
work on whether explanations help users predict model behaviour, and
Poursabzi-Sangdeh et al. on manipulating model interpretability, among others.
**None of it appeared in C5's 80 records.** The instrument therefore misses work
that certainly exists, on a query family aimed directly at it. That is not a
hypothetical limitation; it is a measured one, and it means _unresolved_ in this
document carries considerably less weight than it looks like it should.

The three anchors returned 100 records, so the instrument is live. Live and
sensitive are different properties, and only the first was tested.

### 6.2 Located sources, by claim

| Claim | Nearest located source                                                                                                                                                                                                                                   | Read     | Bearing                                                                                            |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------- |
| C1    | Liznerski et al., _Exposing Outlier Exposure_, TMLR 2022, arXiv:2205.11474                                                                                                                                                                               | abstract | Nearest work. Reports one-class methods are **robust** to outlier choice — the opposite direction. |
| C2    | Yang, Zhou & Liu, _Full-Spectrum Out-of-Distribution Detection_, arXiv:2204.05306                                                                                                                                                                        | abstract | **Anticipates the mechanism.** Covariate shift vs semantic shift conflation; SEM cancels style.    |
| C2    | Simpson's paradox in ML evaluation: offline recsys eval (arXiv:2104.08912), recommender fairness per-user vs aggregate, spatio-temporal model validation and data splits                                                                                 | titles   | The statistical phenomenon in evaluation is established in adjacent fields.                        |
| C3    | _Unsupervised Deep Feature Transfer for Low Resolution Image Classification_, arXiv:1908.10012                                                                                                                                                           | title    | Adjacent. Not a resolution-control attribution study.                                              |
| C4    | _Descriptive Collision in Sparse Autoencoder Auto-Interpretability_ (2026); _Steering grids for SAE features: when a top-context label names an activation regime rather than a causal axis_ (2026); _Contrastive Semantic Projection_, arXiv:2604.22477 | titles   | Auto-interp reliability is an **active** critique literature.                                      |
| C5    | _A Simple Saliency Method That Passes the Sanity Checks_, arXiv:1905.12152, and through it Adebayo et al., _Sanity Checks for Saliency Maps_                                                                                                             | title    | Randomisation controls for an interpretability metric — the registered narrowing condition, met.   |

### 6.3 C2 in detail, because it is the one that changes

Full-Spectrum OOD detection states that the OOD literature "clearly defines
semantic shift as a sign of OOD but does not have a consensus over covariate
shift", builds benchmarks that separate training ID, **covariate-shifted ID**,
near-OOD and far-OOD, and proposes a score whose explicit purpose is that "the
non-semantic part is cancelled out", leaving semantics.

C2 says v13's novelty score separates _domain_ — style, rendering, covariate —
more strongly than it separates _novelty_, and that pooling across domains
therefore reports a number that understates within-domain behaviour. The first
half of that is the problem Full-Spectrum OOD is built around. v13 measured it on
a corpus with six rendering styles and reported the gap; it did not discover the
confound.

---

## 7. Adjudication

| Claim                                                        | Verdict                                      | What remains                                                                                                                                                  |
| ------------------------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **C1** exposure destroys union-of-ellipsoid rejection        | **Unresolved**, weakly                       | No located source reports the degradation or the union mechanism. But 7 of 8 queries lost one index, and the nearest work must be cited and reconciled.       |
| **C2** domain-pooled AUROC conceals within-domain            | **Narrowed, heavily**                        | An instance, not an insight. The confound is Full-Spectrum OOD; the statistics are textbook. **No contribution claim survives.**                              |
| **C3** transfer loss is resolution, not corpus               | **Unresolved**, very weakly                  | Nothing located, but arXiv-only and the noisiest result set of the five. The weakest-supported _unresolved_ here.                                             |
| **C4** exemplar-mediated naming channels are not independent | **Narrowed**                                 | The structural argument was not located, but auto-interp reliability is an active 2025–26 critique area. Position within it; do not open it.                  |
| **C5** simulatability needs a structure-matched null         | **Narrowed** by its own registered criterion | Randomisation controls for interpretability are established (Adebayo et al.). The remainder is applying a budget- and split-matched null to _simulatability_. |

### 7.1 What this audit changes

**Nothing in v13's measurements.** Under N87.1 no result moved, no verdict
changed, and Outcome C stands exactly as sealed. What changed is the contribution
attached to two of them.

**The claim I would have led with does not survive.** Before this audit, C2 —
the pooled/within-domain AUROC gap — was on the shortlist of v13's most
interesting observations. It is an instance of a confound the OOD literature
named in 2022 and built benchmarks around. Reporting it as a discovery would have
been wrong, and the only reason it is not being reported that way is that the
audit ran before the write-up rather than after a reviewer found it.

**No claim was cleanly displaced, and no claim is cleanly new.** Three sit at
_unresolved_ and two at _narrowed_. Under N87.2 _unresolved_ is not a synonym for
novel, and given 6.1 it is a weaker verdict here than the word suggests.

### 7.2 Registered consequence for any write-up

- **C2 may not be presented as a contribution.** It may appear as an observation
  with Full-Spectrum OOD cited as the source of the mechanism.
- **C1 is the strongest remaining candidate**, and requires Liznerski et al. to
  be cited and the difference argued explicitly: they study one-class methods
  over a single normal class and find robustness to outlier choice; v13 studies a
  **union of 128 per-class regions**, where a sample survives only if all of them
  reject it. Whether that difference is the whole explanation is **not
  established**, and stating it as established would repeat the mistake C2 just
  demonstrated.
- **C3 must be re-searched on a second index before any claim rests on it.**
  Six of six queries lost Semantic Scholar.
- **C4 and C5 are positioning, not contribution.** Both belong in related work.
- **The C5 search failure must be disclosed** wherever the structure-matched null
  is described as unusual, since the audit demonstrably failed to retrieve known
  simulatability work.

### 7.3 What a competent adversary would still say

That this audit ran two public indexes with unauthenticated access, lost 70 % of
one of them, used relevance-ranked OR fallbacks that returned astrophysics for a
computer-vision query, and read **no primary text beyond two abstracts**. Every
_unresolved_ verdict above is a statement about that surface and not about the
literature. Nothing here licenses the word "first".

---

## 8. M88 — discharging M87's registered consequences

**Status:** REGISTERED at commit `bc44355`; EXECUTED 30 July 2026; findings in §8.2.
**Registered:** 30 July 2026, before any M88 query was run, on top of commit
`ce999f4`. N88.1–N88.6 below are as registered and are not edited.

M87 finished by writing three obligations against itself in Section 7.2. An
obligation a program writes and then does not discharge is worse than one it
never wrote, because the document now reads as though the work was done. M88
discharges exactly those three and nothing else.

**N88.1 — the scope is closed to M87's own obligations.** M88 may (a) read
Liznerski et al. in primary text for C1, (b) re-search C3 on a further index, and
(c) test whether the instrument can retrieve work that is known to exist. It may
not introduce a sixth claim, revisit C2's demotion, or reopen anything in v13.
C2's narrowing is **final**; a second search is not an appeal.

**N88.2 — recall probes, because live and sensitive are different properties.**
M87's anchors proved only that the indexes answered at all. They could not detect
under-return, which is how C5 returned none of the forward-simulation literature.
M88 registers a fixed list of **named papers that certainly exist**, each paired
with a _topic_ query that does not contain the paper's title. If a topic query
aimed at a paper cannot retrieve that paper, then "found nothing" from that
family is **not evidence of absence** and the family's verdict is downgraded to
_not searched_.

Querying a paper's title to see whether the index returns it would test nothing
except that the index has a title field. The probe must be the same kind of query
the audit actually uses.

**N88.3 — adding an index is asymmetric, and the asymmetry is registered.** A
third index can only ever **find** displacing work. It can never establish
absence, and a claim that survives three indexes is not thereby better supported
than one that survived two — it has merely failed to be refuted more times. No
verdict may be upgraded on the strength of having searched more places.

**N88.4 — re-running a family does not reset it.** M87's verdicts stand unless
this run displaces them. A re-run may move a claim _down_ — to _narrowed_,
_displaced_, or _not searched_ — and may never move one _up_. Specifically,
C3 cannot become better-established by being searched again; it can only stay
_unresolved_ or fall.

**N88.5 — reading Liznerski et al. in full may displace or narrow C1, and cannot
clear it.** Confirming that one paper does not anticipate a claim says nothing
about the papers not read. The purpose of reading it is to discharge the citation
obligation honestly and to test the union-mechanism hypothesis M87 was careful
not to assert.

**N88.6 — the probe list and the C3 query families are fixed at this commit**,
under the same reasoning as N87.3 and N87.4.

### 8.1 Registered recall probes

Each probe is a paper whose existence is not in doubt, paired with a topic query
that omits its title. The instrument must return the paper for the query.

| Probe | Paper that must be retrieved                                                                                  | Topic query (registered)                                              | Tests family |
| ----- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------ |
| P1    | Hendrycks et al., _Deep Anomaly Detection with Outlier Exposure_                                              | anomaly detection auxiliary dataset of outliers improves detection    | C1           |
| P2    | Liznerski et al., _Exposing Outlier Exposure_                                                                 | how many outlier images are needed for anomaly detection one-class    | C1           |
| P3    | Touvron et al., _Fixing the train-test resolution discrepancy_                                                | train test resolution discrepancy classification accuracy             | C3           |
| P4    | Adebayo et al., _Sanity Checks for Saliency Maps_                                                             | randomizing model weights leaves saliency maps unchanged              | C5           |
| P5    | Hase & Bansal, _Evaluating Explainable AI: Which Algorithmic Explanations Help Users Predict Model Behavior?_ | do explanations help users predict model behaviour forward simulation | C5           |
| P6    | Yang, Zhou & Liu, _Full-Spectrum Out-of-Distribution Detection_                                               | benchmark separating covariate shift from semantic shift detection    | C2           |
| P7    | Bau et al., _Network Dissection_                                                                              | quantifying interpretability of individual units visual concepts      | C4           |

P6 is a **positive control**: M87 already located it, so a probe run that fails
P6 has broken something rather than discovered anything.

### 8.2 Findings

**Run:** 30 July 2026, 1,489 s, configuration hash `a63e0a53b99eaf5dbc2268bb9e053d60151ae3914efe49b03017615ac26e454c`.
**Evidence:** `logs/results/v13/m88_prior_art_recheck/evidence.json`.

**N88.7 — not replayable.** Recorded in the run configuration at execution, not at
the registration commit, and it is a statement about the artifact rather than about
any claim. Unlike every other v13 artifact this file is a dated snapshot: re-running
it later returns different records, because the indexes change and the network is not
a sealed corpus. The date and the exact query strings are recorded so a reader can
repeat the search, not so the bytes can be reproduced. No downstream check may treat
it as byte-reproducible.

**The runner exited non-zero, and that is the correct outcome.** It is registered
to fail when the positive control fails. The failure is reported below rather than
repaired.

#### 8.2.1 The anchors said the instrument was healthy. The probes said it was not.

All three indexes were live and all three anchors returned work: 60 arXiv records,
60 OpenAlex records, 20 Semantic Scholar records, `instrument_live: true`. On
M87's evidence that is the whole health check, and it would have passed.

**The recall probes retrieved 4 of 7 papers that certainly exist.**

| Probe | Paper                                                            | Result            | Retrieved by    |
| ----- | ---------------------------------------------------------------- | ----------------- | --------------- |
| P1    | Deep Anomaly Detection with Outlier Exposure                     | found             | arXiv, OpenAlex |
| P2    | Exposing Outlier Exposure                                        | **not retrieved** | —               |
| P3    | Fixing the train-test resolution discrepancy                     | found             | arXiv, OpenAlex |
| P4    | Sanity Checks for Saliency Maps                                  | **not retrieved** | —               |
| P5    | Which Algorithmic Explanations Help Users Predict Model Behavior | found             | arXiv only      |
| P6    | Full-Spectrum Out-of-Distribution Detection                      | **not retrieved** | —               |
| P7    | Network Dissection                                               | found             | arXiv, OpenAlex |

Each miss is a miss across roughly forty records returned by two working indexes
for a topic query aimed squarely at the paper. This is the property N88.2 was
written to measure, and M87's anchor test could not have detected any of it.

#### 8.2.2 The positive control failed, and the registered interpretation of that failure was wrong

P6 was registered as a positive control on the reasoning that M87 had already
located _Full-Spectrum Out-of-Distribution Detection_, so a fair topic query must
find it, so a run that missed it "has broken something rather than discovered
anything". The runner still prints that sentence. It is left in place, and it is
wrong.

The code did not break. Against it: 25 tests pass, including one that asserts a
probe is **not** satisfied by an adjacent paper — _A Simple Saliency Method That
Passes the Sanity Checks_ must not count as _Sanity Checks for Saliency Maps_ —
and one that asserts no registered probe query leaks its own title. The same
matching code that missed P2, P4 and P6 matched P1, P3, P5 and P7. And M87 did
retrieve arXiv:2204.05306, using a _different_ query, through the same runner.

So the only thing that differs between finding the paper and missing it is the
phrasing of the query. **Retrieval is a lottery over query phrasing**, and the
assumption behind calling P6 a positive control — that a paper found once will be
found again by any fair query on its topic — is itself what this run refuted.

This distinction is recorded rather than smoothed over, because "the control
failed, so the control was a bad control" is exactly the move that makes a
pre-registration worthless. The registered reading is being overturned by evidence
that a specific alternative explanation is correct, not by preference: the code is
tested, the same path succeeded four times, and the paper is demonstrably in the
index. What is registered for the future is the stricter rule, not the excuse — a
control failure of this kind licenses **downgrading the search**, never upgrading
the code's reputation.

#### 8.2.3 The canonical prior art for C3 was invisible to C3's own search

C3's six registered queries returned **231 distinct records across three indexes**.
_Fixing the train-test resolution discrepancy_ — the canonical statement that
train/test resolution mismatch, not data, accounts for apparent accuracy loss, and
the single most obviously relevant paper to C3 — **appears in none of them.** It
appears in this evidence file exactly once, retrieved by probe P3, a query written
in advance for the express purpose of catching it.

This is the clearest thing M88 produced. A family can return two hundred records,
look thoroughly searched, and miss the one paper a reviewer would name first. The
only reason it was found here is that it had been named at registration.

#### 8.2.4 Semantic Scholar failed 25 of 30 queries, and its recall is unmeasured

Every one of the seven probe queries failed on Semantic Scholar with HTTP 429
after four attempts, so **the 4/7 recall figure describes arXiv and OpenAlex
only**. Semantic Scholar's recall was not measured at all, and the four claim
queries it did answer therefore rest on an instrument whose sensitivity is unknown.

Those four queries were not worthless — one of them returned the most on-point C3
hit in the entire run (§8.2.5). That is an argument for authenticated access, not
for treating the present setup as adequate.

#### 8.2.5 Located sources in the reopened families

| Claim | Records (distinct) | What the search located                                                                                                                                                         |
| ----- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C1    | 305 (221)          | Nothing displacing. Under N88.2 this means nothing at all — P2 failed. The displacing material for C1 came from _reading_, not searching (§8.2.6).                              |
| C3    | 242 (231)          | Arasteh et al., _Resolution scaling governs DINOv3 transfer performance in chest radiograph classification_, arXiv:2510.07191 — via Semantic Scholar. Plus FixRes, via P3 only. |
| C5    | 198 (165)          | Nothing displacing, and under N88.2 nothing that can be read as absence — P4 failed.                                                                                            |

**Arasteh et al.** benchmark DINOv3 against DINOv2 and supervised ImageNet
initialisation over seven chest-radiograph datasets, 816,183 images, at 224, 512
and 1024 px. Their finding is that the choice of backbone is not what governs
transfer — resolution is: DINOv3 "did not consistently outperform DINOv2 at
224×224" but "became the strongest initialization at 512×512". The title states
the general claim outright. This is the same direction as C3 and on the same
backbone family. It is not the same experiment — they fine-tune, on medical
images, comparing initialisations, and they run no degraded-resolution control
that separates resolution from corpus identity — but C3's _headline_, that
resolution rather than the corpus explains measured transfer loss, is published.

#### 8.2.6 C1, from primary text (N88.5 discharged)

Liznerski et al., _Exposing Outlier Exposure: What Can Be Learned From Few, One,
and Zero Outlier Images_, TMLR 2022, arXiv:2205.11474, read in full rather than by
abstract. M87 recorded it as pointing the _opposite_ way to C1, on the strength of
its abstract's claim that one-class methods are robust to outlier choice. The full
text contains two results the abstract does not:

- **§5.2, "There are settings that require more OE data."** On the leave-one-class-out
  benchmark, "where many classes are combined to form a multimodal normal class",
  more outlier exposure is needed: _"This indicates that more OE samples are
  necessary when the normal class is not concentrated."_ That is C1's proposed
  mechanism — an unconcentrated, multimodal normal region degrading what exposure
  can do — stated in the literature.
- **§5.4, Table 3.** The worst single OE sample drives detection **below chance**:
  CIFAR-10 43.3 (HSC) and 31.6 (BCE); ImageNet-10 39.2 (HSC) and 26.3 (BCE).
  "Outlier exposure can actively destroy rejection" therefore has precedent too.

What is not displaced is the specific structure. Their multimodal normal class is
9 or 29 classes under one learned representation scored by a single scalar; v13's
is a minimum over 128 independently fitted ellipsoids, every one of which must
reject a sample for it to be rejected. But M87 already registered that union
explanation as _a hypothesis, not established_, so nothing survives as a
contribution — only as a difference worth stating when the result is described.

### 8.3 Adjudication

Verdicts may only fall (N88.4). All three reopened claims fall.

| Claim | M87 verdict             | M88 verdict                    | Why                                                                                                                            |
| ----- | ----------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| C1    | Unresolved, weakly      | **Narrowed, and not searched** | Both components have precedent in Liznerski §5.2 and Table 3. Independently, P2 failed, so C1's search establishes no absence. |
| C3    | Unresolved, very weakly | **Narrowed**                   | Arasteh et al. publish the headline; FixRes is the standing prior and C3's own queries missed it entirely.                     |
| C5    | Narrowed (positioning)  | **Narrowed, and not searched** | Verdict unchanged in substance; P4 failed, so nothing about C5 rests on absence.                                               |

Two claims are unaffected by this run and are recorded so:

- **C2 stays narrowed heavily.** P6's failure does not undo it. C2 was narrowed
  because a paper _was found_, and a recall failure cannot un-find something
  (N88.3, in the direction it actually cuts). N88.1 forbids revisiting it in any
  case.
- **C4 stays narrowed.** Not reopened; P7 passed.

**The three obligations M87 wrote against itself are discharged.** Liznerski was
read in primary text and narrowed C1. C3 was re-searched on two further indexes
and was narrowed. The instrument's recall was measured, at 4/7, and the measurement
disqualifies more of M87's reasoning than it rescues.

### 8.4 Registered consequence for any write-up

Superseding §7.2 where it is stricter, and binding on
`FINAL_RESEARCH_PAPER.tex` and `MS_THESIS_REPORT.tex`:

1. **v13 makes no novelty claim of any kind.** Not "first", not "novel", not "to
   our knowledge". Of five claims audited, two are narrowed by papers that state
   the same thing, two are positioning, and one is both narrowed and unsearched.
2. **Absence of prior art is never asserted anywhere in the write-up.** Measured
   recall is 4/7 on the two indexes it could be measured on, unknown on the third,
   and C3's own search missed the paper a reviewer would name first. This corpus of
   evidence can support "here is related work" and cannot support "there is none".
3. **C1 cites Liznerski et al.**, states that both of its components have
   precedent, and presents the union-of-128 structure as a difference in setting
   rather than as an explanation that has been established.
4. **C3 cites Touvron et al. (FixRes) and Arasteh et al.** and is presented as a
   controlled decomposition of a known effect, not as a discovery.
5. **C2 remains an observation citing Full-Spectrum OOD**, exactly as §7.2 required.
6. **The search failures are disclosed**, including this one: the audit's second
   pass found that its first pass could not have supported the conclusions it drew.
