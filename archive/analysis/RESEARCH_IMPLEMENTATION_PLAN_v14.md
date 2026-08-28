# GEODE Research Implementation Plan v14

**Status:** registration draft, 1 August 2026. Sections 1--6 are registered
before any v14 figure is produced. Sections 7--9 are conditional and open only
on the gates named in them. **M89 executed 1 August 2026**; its figures are
sealed at `264bf2a145375a8c0958665f66bf1cea0359feefd9e9f3bd7d847a5dd6aa03a7`
and the §2 numbers below are that evidence, not recollection. **M90 executed 1
August 2026**, sealed at
`350167a2a24932585590060e4614a083cbfe112865c596904d950d44c46f7f99`; it refuted
H91, left H90 undetermined on an invalid instrument, and exposed a defect in an
inherited negative control that is registered as M90.1 in §6.3. **M90.1 executed
1 August 2026**, sealed at
`ce88fb2a737083765c544e1ba9cd22b4889ba5052c6f6dc1f6ecef941a026940`; the repaired
control passes on every arm, the arm operands reproduce M90 exactly, H90 is
refuted, and §7 is open. **M90.2 executed 1 August 2026**, sealed at
`0120ccc080154b627343e3c7f0c9c4e0c9f78d7b6048ef0cb461371f1a928185`; domain became
exactly linearly unreadable and stayed geometrically dominant, H94 is refuted,
H95 is untestable, and six defects and four failed registered judgements are
recorded in §6.4. **M91 executed 1 August 2026**, sealed at
`477837e16021f7b078806eda7183ee0b75ab19e66a19b833beff085111b3c728`; no larger
backbone demonstrated added capacity, H92 is **untestable**, and two confounds
discovered during execution -- idiosyncratic INT8 damage (N91.11) and a fixed
rank that penalises width (N91.12) -- are registered in §7.1 as limits on what
that verdict may be read to mean.

---

## 1. What v14 inherits, and one correction

v13 closed at **Outcome C**, sealed. H83 and H84 are refuted. M85a recorded the
geometry ranking unseen rows at AUROC 0.5851 against a free k-NN bar of 0.5749,
inside the 0.02 decisive margin and therefore a tie. M84 recorded rejection
recall at matched coverage falling from an untrained 0.11875 to 0.00012 under
the exposure ladder. None of that is reopened here. v14 may not revisit a v13
verdict, and no v14 number may be used to argue one.

**The correction.** At the v12 reframe the program permitted representation
learning, and Plan v12 §8 registered **M73**, a staged escalation: frozen ->
learned projection -> last-k/LoRA -> full fine-tune, each stage with a kill
switch. M73 Stage 1 executed and passed its registered gate, reaching 95.625%
known balanced accuracy against M72's 93.75% with a median threshold ratio of
1.282 and worst held-out 4x acceptance falling from 100% to 75%.

**V13-I3.1's lineage audit then voided it.** Defect A, sample adequacy: v12
M70--M73 fitted at 3.125 samples per fitted dimension against the standing floor
of 10, and M74 at 1.875. All void.

v13 rebuilt from the corpus upward to satisfy that floor -- 512 fit rows per
class at rank 51, giving 10.04 -- and in doing so ran entirely on frozen
features. **The escalation ladder was never re-climbed.** No v13 document
registers a decision to freeze the trunk; the freeze is inherited from a voided
milestone rather than chosen. v14 exists to close that gap.

Two things that have been blurred and are hereby separated:

- **N83.4 freezes Phase A geometry** inside the boundary experiment. That is
  necessary for the measurement and is retained.
- **The trunk being frozen** is a separate matter with no registered basis in
  v13. It is what v14 examines.

---

## 2. The motivating measurement

The v13 evidence says rejection fails. It does not say why. Three
re-derivations from sealed v13 inputs, on the M85 partition, locate the cause
in the representation rather than in the boundary. They are reproduced as
gating-free evidence by **M89** before anything is built on them.

**Acceptance is multiply owned.** At the 90% matched coverage M83, M84 and M85
all use, a known evaluation row is interior to a mean of **78.74** of the 128
class regions; the median is 114; **11.65%** of rows are inside all 128 and only
**10.94%** are inside exactly one. Rejection requires that _every_ class reject
a row. When the union covers the populated space, nothing can be rejected, and
M84's 2-samples-in-17,280 follows arithmetically.

**Class clouds are wider than the gaps between them.** Median distance from a
row to its own class centroid is **30.19**; median distance to the nearest
foreign centroid is **12.90**; ratio **2.34**. The closest pair of class
centroids sits at 4.46.

**The spread is domain, not semantics.** Splitting each class into per-domain
cells:

| quantity                                                 | value     |
| -------------------------------------------------------- | --------- |
| median gap between a class's **own** domain cells        | **32.30** |
| median gap to the **nearest foreign-class** cell         | **19.15** |
| spread/separation, one centroid per class                | 2.338     |
| spread/separation, one centroid per (class, domain) cell | 1.384     |

The same object rendered in two domains is further apart than two different
objects rendered in one. Per-domain spread about the class centroid runs from
27.14 (quickdraw) to 48.38 (real), and quickdraw is 39,808 of the 65,536 fit
rows, so every class centroid is dragged toward quickdraw, where the other five
domains do not live. Features are **not** normalised: median L2 norm 46.81.

**A corpus fact not previously recorded.** Domain 5 (sketch) is 158 of 73,728
known rows (0.21%) and **0 of 13,865** open-set rows, a consequence of the
`minimum_native_short_edge: 256` filter. The evaluation domain quota is
therefore `[9, 7, 6, 39, 3, 0]`. This is not a defect -- known and out-of-set
are matched, per N83.2 -- but it is a limitation that every v13 per-domain
figure carries and that no sealed document states.

**What this reframes.** v13's registered reading is that outlier exposure fails
to improve rejection for a union of per-class density models. That stands. But
the mechanism underneath it is that one unimodal region per class is a
misspecified model of a six-domain class, fitted in coordinates where domain
dominates label. v14 tests whether that is the binding constraint.

---

## 3. Registered question

> Is v13's rejection failure a property of the boundary model, of the
> coordinates it is fitted in, or of the backbone that produced them -- and how
> much of it is recoverable without training the trunk?

The order matters and is registered: **model misspecification first, coordinates
second, backbone capacity third, trunk training last.** Each stage must fail its
kill switch before the next opens. A cheaper explanation is not permitted to be
skipped because a more interesting one is available.

---

## 4. Hypotheses and kill switches

Registered before execution. Each names the operand that refutes it.

**H90 -- misspecification.** The acceptance overlap is a consequence of fitting
one unimodal region per class across six domains, not of the representation.
_Refuted unless_ domain-aware components reduce mean acceptance multiplicity
below 40 **and** improve rejection recall at matched coverage above 0.11875
**and** improve AUROC by more than 0.02 over 0.5851, all against the R5 null
below.

**H91 -- coordinates.** The overlap is an artefact of unnormalised, un-centred
coordinates in which domain is the dominant axis. _Refuted unless_ the cosine
or domain-centred arm clears the same three bars.

**H92 -- capacity.** The overlap is a capacity limit of dinov2-small.
_Refuted unless_ a larger backbone clears the same three bars on the identical
partition. **Opens only if H90 and H91 are both refuted.**

**H93 -- learned coordinates.** A learned domain-invariant projection separates
classes without collapsing the far field. **Opens only if H92 is refuted.**

No hypothesis may be rescued by substituting a corpus, relaxing coverage, or
selecting an arm after the fact.

---

## 5. Shared contract

Carried forward from v13 unchanged unless stated.

**Corpus.** `logs/results/v13/domainnet_large` and `logs/results/v13/openset`,
at the sealed hashes M85 verifies. R7 stands: no v12 or CIFAR-10 number is
compared to a v14 number. All v14 arms are measured on the M85 partition so
that they are directly comparable to the sealed v13 figures and to each other.

**Partition.** N83.7's domain-quota split: 512 fit rows per class, evaluation
quota `[9, 7, 6, 39, 3, 0]`, 64 evaluation rows per class, calibration and
report halves domain-stratified. Unchanged.

**Coverage.** 90% known coverage by split-conformal offsets (N83.3), unchanged,
so that recall is read off the same rule v13 used.

**Sample adequacy.** The standing floor of **10 fit samples per fitted tangent
dimension** applies to every component of every arm. Any arm that cannot meet it
is not run at that setting; the floor is never waived. Where an arm fits more
components than v13, rank falls to keep the floor -- this trade is real,
registered in advance, and controlled for by the null.

**R5 nulls.** Every comparative operand carries a null sharing structure,
budget and split:

| arm                  | null                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------- |
| domain-aware mixture | mixture over a **random** partition of identical component count and identical rank   |
| domain-centred       | centring by a **random** partition of identical group count and identical group sizes |
| cosine               | the raw arm on identical rows                                                         |
| larger backbone      | the same arm at dinov2-small on identical rows                                        |

An operand without its null is not evidence.

**Instrument validation (N85.4d).** Every arm validates at both ends before any
figure is read: a far-field positive control at 5x the data's median radius
requiring AUROC >= 0.99, and a known-split negative control requiring AUROC
within 0.02 of 0.5. An arm failing either reports `instrument_invalid` and
suppresses every figure below it.

**Determinism.** Gated evidence is CPU-only, single-threaded torch, byte-
identical replay, in the frozen `.venv`. The RX 9070 XT is permitted **only**
for upstream feature extraction under `.venv-rocm`, never for a gated
computation, and any extraction it produces is sealed as an artifact and
verified by hash before use. This preserves the property that makes every
negative in this program credible.

**Disclosure.** The domain-centred arm requires a domain label at evaluation
time. That is available in DomainNet and is not available in an open world. It
is recorded as an assumption of the arm, not as a capability.

---

## 6. Milestones

| id      | question                                                                                | execution                        |
| ------- | --------------------------------------------------------------------------------------- | -------------------------------- |
| **M89** | Reproduce the §2 diagnostics as sealed, replayable evidence                             | unconditional                    |
| **M90** | Do domain-aware components or coordinates recover rejection, without training anything? | after M89                        |
| **M91** | Is it backbone capacity?                                                                | only if H90 and H91 both refuted |
| **M92** | Reopened M73 Stage 1, at v13 scale, with domain invariance                              | only if H92 refuted              |
| **M93** | Stage 2, last-k blocks or LoRA                                                          | only if M92 clears its gate      |

### 6.1 M89 -- representation diagnostic

**Non-gating by construction.** M89 registers no hypothesis and no pass/fail.
It exists so that the numbers §2 quotes are replayable rather than recalled, and
so that M90's arms have a stated baseline. It trains nothing and re-fits
nothing beyond v13's closed-form Phase A on the sealed fit rows.

Reported: acceptance multiplicity distribution at matched coverage;
spread/separation at class and (class, domain) granularity; own-domain-cell gap
against nearest-foreign-class-cell gap; per-domain spread; feature norm
distribution; corpus composition by domain for both the known and open-set
splits; argmin-score and nearest-centroid class accuracy.

Because M89 asserts nothing, its only correctness requirement is that it
reproduce two sealed v13 quantities exactly from its own recomputed geometry:
M84's zero-rung rejection recall of **0.11875** and M85a's geometry AUROC of
**0.585085105895996**. A run that does not reproduce both is describing a
different object and reports `not_v13_geometry`.

### 6.2 M90 -- remedies that train nothing

Four arms plus nulls, all at 90% coverage on the M85 partition:

1. **baseline** -- v13 exactly, rank 51, one component per class.
2. **cosine** -- rows L2-normalised, otherwise identical.
3. **domain_centred** -- per-domain fit-set means subtracted, otherwise
   identical. Null: random-partition centring, matched group sizes.
4. **domain_mixture** -- one component per (class, domain) cell with at least
   the registered minimum support of 100 rows, **plus one class-level component
   fitted on all 512 of that class's fit rows**. M89 measured **216 cells**
   clearing the threshold, so the arm carries 216 + 128 = **344 components at a
   common rank of 10**, and that rank is registered and not retuned.

   The class-level component is what makes the partition well-formed, and the
   first design did not have it. Assigning below-threshold rows to their own
   per-class residual component was measured and rejected before M90 was
   written: at a threshold of 100 the smallest residual is **11 rows**, which
   the standing floor permits a rank of **1**, and at a threshold of 50 the
   smallest is 3 rows and the permitted rank is **0**. No threshold makes the
   residual-as-component design admissible, so every row is instead covered by
   its class-level component and the domain cells are a refinement on top of
   it. Components therefore overlap, which is consistent rather than awkward:
   acceptance was already a union.

   Coverage is matched **per class over the union of its components** -- one
   offset scaling all of a class's radii until 90% of that class's calibration
   rows are accepted by at least one of them. Matching per component instead
   would fix each component's own coverage and defeat the purpose.

   Null: identical component count, identical component sizes and identical
   rank, with the cells drawn as disjoint random subsets of the class's rows
   instead of by domain, plus the same class-level component. This is what
   isolates domain-awareness from both the rank change and the extra capacity.

   Registered observation, recorded now so it is not presented later as a
   finding: at a 512-row fit budget most classes clear the threshold in only
   **two** domains (per-class cell count min/median/max 1/2/2). This corpus
   does not support a rich six-way mixture at this budget, and a null result
   for this arm is therefore weak evidence about mixtures in general and strong
   evidence only about mixtures affordable here.

Operands, each against the sealed v13 value on identical rows: mean acceptance
multiplicity (baseline 78.74), rejection recall at matched coverage (0.11875),
AUROC (0.585085105895996), known argmin accuracy (0.5076). Decisive margin
0.02 on AUROC, as M85 registered.

**Registered in advance:** if every arm ties or loses, that is the result, and
it is a materially stronger statement of v13's finding than v13 could make --
that the failure survives the three cheapest corrections available. No arm is
selected post hoc, and a losing arm is not re-tuned.

**Outcome, 1 August 2026, sealed at**
`350167a2a24932585590060e4614a083cbfe112865c596904d950d44c46f7f99`. No arm
clears any of the three bars. H91 is **refuted**: L2 normalisation is inert
(78.35 / 0.10955 / 0.5864 against a baseline 78.74 / 0.11875 / 0.5851) and
per-domain centring is worse on every operand (80.22 / 0.11580 / 0.5837), while
its random-group null reproduces the baseline to five figures, which is the null
behaving as designed -- subtracting a random group mean is close to a global
translation, and an ellipsoidal region is translation-equivariant. H90 is
**undetermined**, not refuted: the mixture arm failed its own known-split
negative control at 0.47506 against a 0.48 floor, so its operands are suppressed
under N90.8, and an arm that fails its instrument is void rather than negative
(M83.1 precedent). **M91 therefore does not open.**

### 6.3 M90.1 -- repair of the known-split negative control (registered before execution)

The known-split negative control, written in M85 and inherited here through
N90.8, halves the report rows by position. Those rows are class-sorted, so the
halves are classes 0--64 and 64--127 with exactly one class in common. It is a
class-block split, not a random split of comparable rows, and what it actually
measures is whether a scorer scores the lower and upper halves of the class
index differently. The domain mixtures of the two halves do match
(`[284,252,185,1241,86,0]` against `[292,196,199,1255,106,0]`), which is what
M85's docstring was relying on, but class membership is disjoint. All six M90
arms sit between 0.4751 and 0.4860 -- a one-sided bias, not null noise.

Registered before M90.1 runs:

1. **N90.1.1.** The control becomes a random split of the report rows stratified
   jointly by class and by domain, drawn from a seed fixed in the config, so the
   two halves are exchangeable in both factors. The AUROC of the scorer at
   distinguishing them must fall within `known_split_tolerance` of 0.5.
2. **N90.1.2.** The repair is applied to **every** arm including the baseline
   and every null. Comparing an arm under a repaired control against an arm
   under the old one would be the defect wearing a different hat.
3. **N90.1.3.** M90's figures are not retouched. M90 stands as executed, with
   H90 undetermined; M90.1 is a separate milestone with its own evidence file
   and its own hash. The M90 arm operands are expected to be numerically
   identical, since only the control changes, and any arm whose operands move is
   a defect in the refit to be reported, not a finding.
4. **N90.1.4.** Only M90.1 can decide H90. If the mixture arm passes the
   repaired control and still fails all three bars, H90 is refuted and M91
   opens. If it fails the repaired control as well, the mixture design is
   unmeasurable at this budget and H90 stays undetermined, and it is recorded as
   undetermined rather than converted into a negative.
5. **N90.1.5.** M85's figures are **not** voided. Its arms passed this control,
   and passing a mis-aimed control does not invalidate a measurement. What is
   withdrawn is one interpretive sentence in the M85 docstring -- that the
   control "measures the scorer rather than the split" -- and the withdrawal is
   disclosed in the write-up. This is a narrowing, which N87.1 permits; it is
   not a revision of any sealed number.
6. **N90.1.6.** The old control is retained alongside the repaired one and both
   are reported, so the size of the class-block effect is on the record rather
   than deleted from it.

**Outcome, 1 August 2026, sealed at**
`ce88fb2a737083765c544e1ba9cd22b4889ba5052c6f6dc1f6ecef941a026940`. The repair
behaves as a negative control should. Under the stratified split all six arms
land between **0.49524 and 0.49765**, scattered around 0.5, against **0.47506 to
0.48602** under the class-block split, which was one-sided on every arm. The
split profile is recorded in the evidence: the class-block halves hold 65 and 64
classes with 1 in common, the stratified halves hold 128 and 128 with 128 in
common.

Every arm operand reproduces M90 with a largest absolute delta of **exactly
0.0**, so N90.1.3 is satisfied and the only thing that differs between the two
milestones is the control.

The mixture arm is now **measurable and it loses**: 75.87 multiplicity, 0.08715
recall and 0.5890 AUROC, failing all three bars, with its random-cell null ahead
of it on both operands (0.12882 and 0.5953). Under N90.1.4 that refutes **H90**.
With H91 already refuted at M90, **both hypotheses are refuted and M91 opens.**

---

### 6.4 M90.2 -- domain overlap, measured directly and attacked with prior art

**Why this milestone exists.** M90 and M90.1 answered "does rejection improve"
and never answered "did domain overlap improve". Every figure read was a
downstream operand: multiplicity, recall, AUROC, accuracy. M89's geometric
measure of domain dominance -- the median gap between a class's own domain cells
(32.295) against the median gap to the nearest foreign class cell (19.153) --
was never recomputed under any transform. So it is on the record that
domain centring did not help rejection, and it is **not** on the record whether
it reduced domain dominance at all. Reading the sealed evidence after the fact,
the indirect signs are that nothing moved: within-domain AUROC is 0.6580 at
baseline against 0.6566 under domain centring, and per-domain rejection recall
still spans 0.0 on quickdraw to 0.73 on real in every arm. Those are inferences,
not measurements, and this milestone replaces them with measurements.

**Prior art consulted before the arms were chosen.** Recorded with the standing
constraint from `PRIOR_ART_AUDIT_v13.md` §8.4 and the M88 finding that
unauthenticated public search cannot support a novelty claim: this list exists to
borrow from, not to establish priority, and its absence of a result means
nothing.

1. **LEACE**, Belrose et al., arXiv:2306.03819 -- least-squares concept erasure
   in closed form, provably preventing _all_ linear classifiers from recovering
   a concept while changing the embedding as little as possible. Fitted once on
   the fit set, it is a fixed affine map. **It needs no domain label at
   evaluation time**, which is exactly the objection N90.6 raised against the
   domain-centred arm, so it is a deployable form of the same idea.
2. **INLP**, Ravfogel et al., arXiv:2004.07667 -- iterative nullspace
   projection, the precursor LEACE supersedes with a closed form and a minimal
   edit guarantee. Cited as lineage; LEACE is the arm.
3. **All-but-the-Top**, Mu et al., arXiv:1702.01417 -- remove the common mean
   and the few top dominating directions. Label-free, and it matches M89's
   finding that a nuisance factor is the dominant axis of variation.
4. **RobustNet**, Choi et al., arXiv:2103.15597 -- instance selective whitening,
   which removes only the covariance components carrying domain style. Its
   lesson is that whitening everything also destroys content, so a whitening arm
   must be read against, not instead of, a selective one.
5. **Deep CORAL**, Sun and Saenko, arXiv:1607.01719 -- second-order statistic
   alignment. Its train-free form aligns per-domain covariances and therefore
   needs a domain label at test time, the same defect as our failed arm.
6. **Mahalanobis OOD**, Lee et al., arXiv:1807.03888 -- a class-conditional
   Gaussian score with a **tied** covariance across classes. Noted as a
   competing fitting strategy, deliberately **not** an arm here, because it
   changes the geometry rather than the coordinates and would confound this
   milestone's question.
7. **DomainBed**, Gulrajani and Lopez-Paz, arXiv:2007.01434 -- under fair model
   selection, plain ERM matches every domain generalisation algorithm tested.
   This is the cautionary prior, and it is registered here **before** the run:
   our domain-centring failure is what this literature predicts, not an anomaly,
   and a null result at M90.2 would be consistent with it.

**H94.** Domain is linearly encoded in the frozen features, and erasing it
linearly removes domain dominance. Measured on the geometry, not on rejection.

**H95.** Removing domain dominance improves rejection.

These are deliberately separate. **They are permitted to dissociate, and the
dissociation is the interesting outcome**: if H94 holds and H95 fails, domain
dominance is real, removable, and _not the cause_ of v13's rejection failure,
which is a stronger and more useful statement than either hypothesis alone. This
reading is registered now so it cannot be presented afterwards as a prediction.

**Arms**, all train-free, all fitted on fit rows only, all applied as a fixed map
to calibration, report and open-set rows:

| arm            | transform                                                      | what it tests                        |
| -------------- | -------------------------------------------------------------- | ------------------------------------ |
| `baseline`     | identity                                                       | v13 anchor                           |
| `leace_domain` | LEACE erasure of the 6-way domain concept                      | H94                                  |
| `leace_null`   | LEACE erasure of a random 6-way partition, matched group sizes | R5 null for LEACE                    |
| `abtt`         | remove global mean and top 5 principal directions              | H94, label-free                      |
| `abtt_null`    | remove 5 seeded random orthonormal directions                  | R5 null for ABTT                     |
| `whiten`       | ZCA whitening on the pooled fit covariance                     | RobustNet's caution, read as a bound |

The erased rank is **5** for every erasure arm, because a centred one-hot over
six domains has rank 5; ABTT's `k` is set to 5 to match, so the two arms remove
the same number of directions and differ only in _which_.

**Registered instruments, read for every arm:**

1. **Domain linear probe.** A multinomial logistic probe fitted on fit rows and
   scored on held-out report rows, reporting balanced accuracy against a chance
   of 1/6. This is the direct test of H94 and the direct check that the LEACE
   arm did what LEACE claims. Fitting and scoring on the same rows would measure
   memorisation, so the probe is held out.
2. **M89's `separation_report`**, recomputed per arm.
3. The M90 operands unchanged, so the two milestones remain comparable.

**N90.2.1 -- only scale-invariant quantities are compared across arms.**
Whitening and erasure change the overall scale of the space, so a raw gap of
32.295 in one arm and 4.1 in another is not a comparison, it is a unit change.
Across arms this milestone reads only the **domain dominance ratio** (own-class
sibling cell gap over nearest foreign-class cell gap), the spread-over-separation
ratios, the sign of `sibling_exceeds_foreign`, and the probe accuracy. Raw gaps
are recorded per arm and never compared between arms.

**N90.2.2 -- the H94 bar.** Domain dominance is removed only if the dominance
ratio falls below 1.0 (foreign-class cells become better separated than a class's
own domain cells) **and** the probe's balanced accuracy falls below
`probe_chance_bar`, with the arm beating its matched null on both.

**N90.2.3 -- H95 is judged on M90's three bars**, unchanged, so no bar moves
because a new arm arrived.

**N90.2.4 -- a failed erasure voids only H94 for that arm.** If the probe does
not fall to chance under `leace_domain`, LEACE was misapplied and the arm is a
defect to fix, not evidence about domain overlap.

**N90.2.4a -- N90.2.4's inference is withdrawn. The sentence above is left
exactly as registered and contradicted here rather than edited, on the M88
precedent.** It assumes that "probe at chance" and "LEACE correctly applied" are
the same event. They are not, in either direction. LEACE's guarantee is over the
**first moment**: it equalises the class-conditional means and provably zeroes the
cross-covariance between the transformed features and the one-hot concept. Whether
a probe then reads chance is a separate empirical question. The `erasure_certificate`
of N90.2.15 is the test of application, and it earned its place immediately by
catching a float32 defect (N90.2.16) that N90.2.4's rule would have misdiagnosed
as evidence about the data.

**N90.2.5 -- M91's gate is unchanged.** M90.1 already opened it. M90.2 runs
first because it is cheap and because it tells M91 what to measure, but a null
here does not close M91 and a success here does not cancel it.

**N90.2.5a -- the chance figure quoted in the instrument list above is wrong and
is corrected to 1/5.** The evaluation quota gives sketch zero rows, so five
domains are present in the report rows, not six. Chance balanced accuracy is
**0.2**. The registered probe bar of 0.25 is left where it is, which makes it a
narrower margin over chance than intended, so an arm landing between 0.2 and 0.25
would have to be reported as near chance rather than comfortably below it. No arm
did.

**N90.2.16 -- the certificate found a defect in this milestone's own code, which
is why it earns its place.** The corpus is float32. LEACE's closed form needs the
inverse square root of a 384-square covariance, and taken in float32 that left a
residual largest pairwise domain-mean gap of **1.76e-01** against an original
**3.52e+01** -- an exact guarantee degraded into a 200-fold reduction. The
identical computation in float64 leaves **2.05e-09**. Had the certificate not been
added, this milestone would have reported "LEACE only partly erases domain" as a
finding about the data when it was a finding about a dtype. All four transforms
are now fitted and applied in float64 and cast back to the corpus dtype, so the
geometry downstream still sees float32 exactly as the untransformed baseline does.
The certificate's own verdict is relative rather than absolute for the same
reason: a float64 map applied to float32 data and cast back cannot leave a
residual below the corpus rounding, so an absolute bar would measure the dtype.

**N90.2.17 -- the erased rank is capped at `group_count - 1` rather than read off
a singular value tolerance.** A centred one-hot over $g$ groups has rank exactly
$g-1$, so the budget is known analytically. This was found when the corrected run
removed five directions for `leace_domain` and **six** for `leace_null`, which
breaks R5's requirement that a null share the arm's budget. The cause is that the
null's cross-covariance is entirely noise, so a _relative_ tolerance has no signal
to be relative to. A tolerance is the wrong instrument for a quantity that is
known in closed form.

**N90.2.14a -- the second-moment explanation is withdrawn as speculation about an
artefact.** An earlier reading of this milestone attributed a post-erasure probe
accuracy of 0.7289 to covariance structure that LEACE does not touch. That figure
was the float32 defect of N90.2.16. Computed correctly the probe reads exactly
chance, so the linear probe finds nothing and no second-moment account is needed
or supported here. The general statement that LEACE guarantees only the first
moment remains true, and remains the reason RobustNet and Deep CORAL exist; what
is withdrawn is the claim that _this corpus_ demonstrated it. No nonlinear probe
was run, so nothing is claimed about what one would find.

**Outcome, 1 August 2026, sealed at
`0120ccc080154b627343e3c7f0c9c4e0c9f78d7b6048ef0cb461371f1a928185`, 14.0 min.**

**H94 refuted, H95 untestable, no dissociation -- and the shape of the refutation
is the finding.** The baseline replicates M90.1 exactly, so the object under
transform is the object M90 and M90.1 measured. The milestone was re-executed end
to end and hashed to `0120ccc0…` a second time, so the replay is byte-identical
and observed rather than asserted; the sklearn probe is deterministic under its
seed on this machine.

**Domain became linearly unreadable and geometrically dominant at the same time.**
`leace_domain` drives the held-out domain probe to **0.2000**, which is exactly
chance for the five domains present in the report rows, while its budget-matched
null -- five directions removed, exactly as many -- sits at **0.8875** against a
baseline of 0.8946. The erasure certificate confirms
the mechanism rather than inferring it: the largest pairwise domain-mean gap falls
from **35.2 to 1.06e-07** and the cross-covariance with the one-hot domain
likewise, at the predicted rank of 5. Every linear trace of domain is gone.

**And the geometry barely noticed.** The dominance ratio falls only from 1.450 to
**1.234** and stays above 1.0, so a class's own domain cells remain further apart
than its nearest foreign-class cell even when no linear classifier can tell the
domains apart at all. That dissociation -- linear unreadability without geometric
integration -- is the substantive result of this milestone. Domain structure in
these features is not confined to a five-dimensional linear subspace, so no
projection can remove it.

**Rejection did not move.** Recall and AUROC stay inside noise of the baseline
under every arm, including the one that achieved a perfect erasure. H95 is
recorded as **untestable rather than refuted**, because no arm satisfied its
antecedent of removing dominance; the observation that the removable part bought
nothing is recorded as an observation, not a verdict.

**`whiten` reproduces RobustNet's stated lesson.** Full ZCA whitening drops known
accuracy from 0.5076 to about **0.319** and inflates multiplicity from 78.74 to
93.37 while leaving the probe near 0.89. Whitening everything destroys content,
which is why RobustNet is selective.

**DomainBed's caution (N90.2.9) is upheld and is not a hedge.** No domain-aware
transform beat its own budget-matched null on any rejection operand.

**Four registered judgements failed and are recorded rather than repaired:**
N90.2.4's inference (N90.2.4a), N90.2.5's chance figure (N90.2.5a), N90.2.14's
second-moment explanation (N90.2.14a), and the probe's iteration budget
(N90.2.12). Two code defects were caught by instruments added during the
milestone: a float32 erasure (N90.2.16) and a null that removed a different number
of directions from the arm it controlled (N90.2.17). All are on the record because
a plan that plays back only its correct predictions is not evidence of anything.

**M91 is unaffected.** M90.1 opened it; this milestone neither closes nor cancels
it, exactly as N90.2.5 registered. What M90.2 contributes is a sharper question
for M91: the separation instrument and the erasure certificate carry forward, and
"domain dominance that survives complete linear erasure" is now a named property
a new backbone can be asked to improve.

---

## 7. M91 -- backbone capacity (conditional)

Opens only on H90 and H91 both being refuted. **This gate opened at M90.1 on 1
August 2026:** H91 was refuted at M90 and H90 at M90.1, once the mixture arm
became measurable under the repaired control. dinov2-base (768-d) and, if
affordable, dinov2-large (1024-d), extracted at batch size 1 under the V13-I2.1
finding that INT8 `DynamicQuantizeLinear` makes features a function of the batch.
Extraction on the RX 9070 XT under `.venv-rocm` is permitted; the resulting
arrays are sealed and hash-verified, and every gated computation on them runs
CPU-only. Rank is re-derived from the floor at the new dimension, not carried
over. R5 null: dinov2-small on identical rows and partition.

Registered expectation, recorded so it cannot be claimed afterwards as a
prediction: accuracy is expected to improve and rejection is not, because
neither the spread/separation ratio nor the domain dominance in §2 is obviously
a capacity limit. If rejection does improve, H92 survives and that is the
finding.

### 7.1 M91 registered content

Registered 1 August 2026, before any dinov2-base or dinov2-large feature was
extracted.

**Arms.** Three, one per backbone: `dinov2_small` (384-d), `dinov2_base` (768-d),
`dinov2_large` (1024-d). `dinov2_small` is simultaneously the reference and the
R5 null of §5's table.

**Instruments carried forward, so that the geometry is measured rather than
inferred:** M89's `separation_report` and `acceptance_multiplicity`, M90.1's
repaired stratified negative control and the far-field positive control, and
M90.2's `domain_probe` and `erasure_certificate`. M90.2 named a property --
domain dominance that survives complete linear erasure -- and a larger backbone
is being asked, among other things, whether it improves that property.

**N91.1 -- the comparison is single-factor by construction, and the construction
is checked rather than asserted.** All three graphs are the `onnx/model_int8.onnx`
export of the same producer, `onnx-community/dinov2-{size}-ONNX`, and all three
ship a `preprocessor_config.json` that is **byte-identical** at
`14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828`, the hash
already registered for dinov2-small in `experiments/configs/v13/domainnet_large.json`.
Producer, quantisation scheme, preprocessing, decode path, pooling policy
(`cls_token`), batch size and execution provider are therefore held fixed and
only the backbone changes. Weight hashes are registered here:
dinov2-small `e48d69984ee26d089e5c690cb39a727e38b95e2a525d7269f5ac366b02a6c22a`,
dinov2-base `f3006646af1ed07acceeefdd701e9be6e46e0b701ac0a1ab8e4ba7260e338302`,
dinov2-large `68910b3dff166c4853ac9ec87755249dfa5f59ab6e388b1d105ee017f4257d3d`.

**N91.2 -- "identical rows" is a check, not a claim.** v13's selection scan is a
deterministic prefix walk over hash-locked shards and depends only on the shard
set, the class list, `samples_per_class` and the 256-pixel short-edge filter --
never on the backbone. Re-extraction therefore re-derives the selection, and the
produced manifest must equal the sealed
`logs/results/v13/domainnet_large/selection_manifest.json` **entry for entry** on
`source_file`, `source_row`, `image_path`, `class_label`, `domain`,
`native_width` and `native_height`, and likewise against the sealed open-set
manifest. Any mismatch means the reader changed and the corpus is not comparable;
the run reports `rows_not_identical` and suppresses every figure below it. This
is the M89 N89.3 pattern applied to the corpus rather than to the geometry.

**N91.3 -- extraction is CPU-only, and this was re-measured rather than
inherited.** v13 registered that DirectML disagrees with the CPU provider by
11.7 percent relative at batch size one for dinov2-small, which is a divergence
in the quantised kernels themselves. The same measurement was repeated for
dinov2-base before this section was written and is **worse**: mean relative
divergence **0.2171**, maximum **0.2500**, over a shared probe set at batch size
one. The GPU cannot reproduce this backbone's features, so it cannot be used
here at any speed. The RX 9070 XT remains permitted for extraction in principle
under §5; it is excluded from M91 on a measurement, not a preference.

**N91.4 -- rank is re-derived and does not move, and the reason matters.** The
standing floor is ten fit samples per fitted tangent dimension. It binds on
**fit rows per class**, which is 512 for every arm, giving 51.2 and a rank of 51
at every ambient dimension; the standing cap of 53 is not reached. Rank is
therefore 51 for all three arms not by carry-over but because the floor does not
depend on $d$. Recorded explicitly so that "same rank" is not later mistaken for
an unexamined inheritance.

**N91.5 -- an arm that does not demonstrate added capacity is void on the
capacity question, not negative.** H92 is a hypothesis about capacity. If a
larger backbone does not beat dinov2-small on **known-class accuracy** on the
identical evaluation rows, then whatever it did to rejection is not evidence
about capacity, because the arm never demonstrated any. Such an arm reports
`capacity_not_demonstrated` and its rejection operands are recorded but excluded
from the H92 verdict. This is the M83.1 / N83.8 precedent: an arm failing its own
instrument is void, not negative.

**N91.6 -- INT8 damage is measured, because it is the obvious confound.** A
larger model can be quantised worse than a smaller one, and a null result would
then be about quantisation rather than capacity. For each backbone the run
records the mean relative divergence between the INT8 graph and the same
producer's fp32 `onnx/model.onnx` on a fixed probe set of images drawn from the
corpus. This is **reported, not gated**: it cannot rescue a failing arm, and it
exists so that N91.5's void verdict, if it is issued, has a candidate cause on
the record rather than a guess added afterwards.

**N91.7 -- the three bars are M90's, unchanged, and the gate is stated before the
numbers.** H92 is refuted unless a larger backbone clears mean acceptance
multiplicity below **40**, rejection recall above **0.11875**, and AUROC above
**0.585085105895996 + 0.02**, on the identical partition and coverage, having
first demonstrated capacity under N91.5. Accuracy improving is not a pass; it is
the precondition for the arm being readable at all.

**N91.8 -- the small arm must reproduce v13 exactly.** `dinov2_small` is
re-extracted by the same code path as the other two rather than read from the
sealed corpus, so that all three arms are produced identically. Its features must
hash to the sealed `features_sha256`, and its recomputed geometry must reproduce
M84's zero-rung recall `0.11875` and M85a's AUROC `0.585085105895996` exactly. A
run in which the reference arm does not reproduce is describing a different
object and reports nothing.

**N91.9 -- no figure in M91 may revisit a v13 verdict.** M91 measures whether a
larger backbone changes the rejection result. It cannot amend Outcome C, rescue
H83 or H84, or be read against any v12 or CIFAR-10 number (R7).

**N91.10 -- each arm carries the v14 diagnostic panel, and none of it is a bar.**
Every arm reports M89's separation instrument, M90.2's domain probe, and the
dominance ratio measured **before and after** a LEACE erasure fitted on that
arm's own fit rows. These exist to say _why_ an arm behaves as it does, not
whether it passes. No verdict in M91 may be decided on any of them. _This note
was registered in `experiments/configs/v14/m91_backbone_capacity.json` and
implemented in the evaluator before any arm was extracted, but was omitted from
this section's prose by oversight; it is written down here, still before any M91
verdict has been read, rather than added silently afterwards._

**N91.11 -- amendment issued after extraction, before any verdict: N91.1's
"single-factor" claim is false.** N91.1 registered the three arms as differing
only in capacity, on the grounds that graph provenance, preprocessing and pooling
are identical. The N91.6 control has now measured the quantisation damage that
N91.1 did not account for, and it is neither small nor constant:

| arm            | mean relative INT8-vs-fp32 divergence | max    |
| -------------- | ------------------------------------- | ------ |
| `dinov2-small` | 0.3150                                | 0.4062 |
| `dinov2-base`  | 1.1944                                | 1.2831 |
| `dinov2-large` | 0.4812                                | 0.5720 |

For scale, two unrelated vectors of similar norm diverge by about 1.41. The base
arm's INT8 CLS token therefore bears little resemblance to the representation the
producer's fp32 graph computes. N91.1 is left standing and contradicted here
rather than edited away (M88 precedent); the design is single-factor in
provenance but not in fidelity.

**Withdrawn sub-claim.** An earlier draft of this note, written when only the
small and base arms had been measured, asserted that the damage "grows with model
size" and was therefore correlated with the axis under test. The large arm
refutes that: at 0.4812 it is much closer to small than to base. The damage is
**idiosyncratic to each published graph**, and `dinov2-base` is simply an
anomalously bad one. The withdrawn sentence was a trend fitted to two points and
is recorded here as an error rather than deleted.

The consequence is registered now, before the evaluator has been run, because it
determines what may be concluded from each verdict. Quantisation damage can only
degrade an arm, never flatter it, so the confound is one-sided _per arm_ -- but,
being uncorrelated with size, it does not bias the capacity comparison
systematically. Therefore

- a verdict of **`survives`** is safe, and if anything understates the effect;
- a **`capacity_not_demonstrated`** or bar failure **on `dinov2-base`** is
  **confounded** and may not be attributed to capacity: this experiment cannot
  separate "capacity does not help" from "INT8 destroyed this particular graph";
- the same verdict **on `dinov2-large`** is far better supported, since its
  fidelity is comparable to the reference arm's. `dinov2-large` is consequently
  the arm on which H92 is most cleanly readable, and it should be read as such
  even though it was not singled out in advance.

The arms are **not** switched to the fp32 graphs to remove the confound. Doing so
would break N91.8, since an fp32 small arm cannot reproduce v13's sealed digest,
and would sever comparability with every sealed v13 and v14 measurement. M91 is
run as registered and reported with this limitation attached.

**N91.12 -- the fixed rank is a second confound, and it handicaps exactly the arms
under test.** N91.4 pinned the rank at 51 for every arm, because the ten-samples-
per-fitted-dimension floor binds on the 512 fit rows per class and is therefore
independent of ambient dimension. That was correctly derived, but its consequence
was not stated: 51 directions out of 384 and 51 out of 1024 are not the same
model. Measured on each arm's own fit rows, the fraction of within-class variance
surviving the rank-51 fit is

| arm            | ambient dimension | within-class variance retained at rank 51 |
| -------------- | ----------------- | ----------------------------------------- |
| `dinov2-small` | 384               | 0.8156                                    |
| `dinov2-base`  | 768               | 0.6977                                    |
| `dinov2-large` | 1024              | 0.6902                                    |

A wider backbone is thus necessarily fitted more lossily, and the capacity
operand itself -- known accuracy read off the fitted geometry -- is computed
through that tighter bottleneck. The handicap runs against H92 by construction.

This is **not repairable within the standing constraints**. Raising the rank for
the wider arms would breach the sample-adequacy floor, which is a standing v13
constraint and not M91's to waive; lowering every arm to a common retained-variance
fraction would change the reference arm and break N91.8. The figure is therefore
recorded per arm as a diagnostic, gated on nothing, and the following reading is
registered before the numbers are sealed:

- M91 cannot separate **"capacity does not help"** from **"capacity cannot be
  reached at 512 fit rows per class"**. Any `capacity_not_demonstrated` verdict
  must be reported as the second, weaker statement.
- The substantive claim M91 can support is about the _sample regime_, not about
  backbones: with 512 fit rows per class the permitted rank is 51 whatever the
  representation's width, so additional backbone width cannot be exploited by this
  geometry. That is a statement about the method's data requirements and belongs
  in the limitations, not a refutation of scale.

### 7.2 M91 outcome

Executed 1 August 2026. Evaluator sealed at
`477837e16021f7b078806eda7183ee0b75ab19e66a19b833beff085111b3c728`, builder at
`9dd9b80a8f0fdbb4dd815afcc70b5945f9d6f0286a17ad20c4a40c5f6e7077c7`, evidence hash
`a4aff6386c96dbed8816dd0b7728b2d1178946a943dd6953ea017716d6ce377e`, reproduced
byte-identically on a second run. 251 v13/v14 tests pass.

| arm                  | dim  | known accuracy | recall  | AUROC    | multiplicity | retained | INT8 divergence |
| -------------------- | ---- | -------------- | ------- | -------- | ------------ | -------- | --------------- |
| `dinov2_small` (ref) | 384  | 0.507568       | 0.11875 | 0.585085 | 78.74        | 0.8156   | 0.3150          |
| `dinov2_base`        | 768  | 0.122314       | 0.05087 | 0.5164   | 95.22        | 0.6977   | 1.1944          |
| `dinov2_large`       | 1024 | 0.410645       | 0.15625 | 0.5897   | 72.00        | 0.6902   | 0.4812          |

**H92 is untestable.** Neither larger backbone beat the reference on known-class
accuracy -- base by −0.385, large by −0.097 -- so under N91.5 no arm demonstrated
added capacity and H92's antecedent was never satisfied. The rejection operands
are recorded and excluded from the verdict.

Instrument state: the reference arm reproduced v13 exactly, hashing to the sealed
`features_sha256` and recovering M84's recall `0.11875` and M85a's AUROC
`0.585085105895996` (N91.8). All three arms produced manifests identical to v13's
selection entry-for-entry (N91.2). Every arm's negative controls were valid and
every domain probe converged, at 404, 262 and 193 iterations against the budget of 2000. So the null result is not an instrument failure; it is a real measurement of
these three graphs under this fitting regime.

What it does **not** license: `dinov2_base`'s collapse to 0.122 accuracy sits
alongside an INT8 divergence of 1.1944, near the ~1.41 of unrelated vectors, and is
best read as a broken published graph rather than evidence about capacity (N91.11).
`dinov2_large` is the cleanly readable arm, and it still lost 0.097 accuracy while
being fitted at 0.6902 retained variance against the reference's 0.8156 (N91.12).
Both effects run against H92, so the honest statement is the weaker one: **at 512
fit rows per class the permitted rank is 51 regardless of representation width, so
this geometry cannot exploit a wider backbone.** M91 does not show that scale fails
to help; it shows that scale is unreachable in this sample regime.

One registered judgement failed. N91.1 declared the comparison single-factor by
construction; it is single-factor in provenance but not in fidelity, and is
contradicted in place by N91.11 rather than edited away. An earlier draft of N91.11
itself asserted that quantisation damage grows with model size, a trend fitted to
two arms and refuted by the third; that sentence is recorded as withdrawn.

**The gate to §8 does not open, and an earlier sentence here said otherwise.**
That sentence -- "the gate to §8 is unaffected: M92 was never conditional on H92
surviving" -- is withdrawn. It misread the registration. §4 opens H93 **only if
H92 is refuted**, and §3 requires that each stage _fail its kill switch_ before
the next opens, precisely so that a cheaper explanation cannot be skipped in
favour of a more interesting one. H92 was not refuted; it was never testable,
because no arm satisfied its antecedent. The capacity explanation is therefore
still unexcluded, and M92 and M93 are **closed** under the registration as
written. Opening them would require a registered amendment that either makes H92
testable or states, with reasons, that the ordering in §3 is being set aside --
and the second of those is the exact move §3 exists to forbid.

Registered expectation also failed, and in the informative direction. §7
predicted that accuracy would improve and rejection would not. Accuracy did not
improve; it fell on both larger arms, by 0.385 and 0.097. Under N91.11 and N91.12
that is better explained by a broken checkpoint and a width-penalising rank than
by anything about capacity, which is the same reason H92 cannot be read.

**What is committed, and what is not.** The three re-extracted arms come to 778 MB:
731 MB of `arrays/*.npy` and 47 MB of `selection_manifest.json`. These are excluded
from version control and listed in `.gitignore`; the committed evidence is
`evidence.json`, `artifact_index.json` and the per-arm summaries. Nothing
verifiable is lost by this. Every array's SHA-256 is recorded inside the committed
evidence, the manifests were checked entry-for-entry against v13's sealed selection
with `differing_entries` 0 and that result is committed, and both are regenerable
from the sealed builder `9dd9b80a…` with its committed config. The small arm's
`features.npy` is in any case byte-identical to the already-tracked v13 corpus file,
which is a third independent confirmation of N91.8 alongside the hash check and the
recovered M84/M85a operands.

## 8. M92 -- reopened M73 Stage 1 (conditional)

The defect that voided M73 does not recur: the projection is **shared**, not
per-class, so it is fitted against all 65,536 fit rows rather than 512. Rank and
sample adequacy are re-derived and recorded.

Mandatory, carried forward verbatim from Plan v12 §8:

- collapse prevention, **plus an ablation demonstrating it is load-bearing**;
  absence of the ablation invalidates the stage;
- feature-space diagnostics before and after, using M89's exact instrument;
- inspectability restated in weakened form -- the decision rule is exactly
  inspectable, the feature semantics are not.

One amendment, on evidence. M73 Stage 1's zero-constraint ablation was
**operationally stronger** than the constrained arm: 96.375% accuracy and 77.5%
unknown recall against 95.625% and 59.5%. That result is void and may not be
cited as a finding, but it is sufficient reason to register M92's collapse
ablation as **read either way** rather than as a formality expected to confirm
the constraint.

## 9. M93 -- Stage 2 (conditional)

Last-k blocks or LoRA at reduced resolution, opening only if M92 clears its
gate. Determinism policy must be settled in writing before any GPU-produced
number is gated. Not registered further until M92 reports.

---

## 10. What v14 may not do

1. Reopen, revise or soften any v13 verdict. Outcome C stands as sealed.
2. Use a threshold-free result to overturn a threshold result (N85.4a).
3. Retouch a sealed evidence file, ever.
4. Waive the 10-samples-per-fitted-dimension floor for any arm.
5. Substitute a corpus to rescue a failing gate.
6. Select an arm after seeing the operands, or re-tune a losing arm.
7. Claim novelty. The §8.4 consequence registered in
   `PRIOR_ART_AUDIT_v13.md` binds every v14 write-up: no novelty claim of any
   kind, no assertion that prior art is absent, and the M88 search failures
   disclosed wherever a claim family is discussed.
8. Gate any figure computed on the GPU.
