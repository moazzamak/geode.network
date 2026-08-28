# Acceptance criteria — v13

**Status:** registered by M79, 29 July 2026. Supersedes
`analysis/ACCEPTANCE_CRITERIA_v12.md` for all v13 milestones. Prior versions
remain valid records of what they measured.

**Registered by:** `analysis/RESEARCH_IMPLEMENTATION_PLAN_v13.md` Section 6.
**Companion:** `analysis/CLAIM_LEDGER_v13.md`.

M79 is unconditional and blocks all architecture work. Nothing in M80–M86 may be
measured until this document is committed, because every one of those milestones
reports against thresholds that this document fixes in advance.

---

## 1. Decision

v12 closed at Outcome E: a learned projection made the analytic field
competitive on known-class accuracy, but open-space rejection did not
generalise. Two v13 forensic milestones then established that part of that
record was measuring something other than what it claimed.

- **M77** — the probe objective was gradient-dead (norms ≤ `6.78e-17`), so no
  v12 result attributable to it was ever validly exercised.
- **M78** — every v12 basis was fitted below the identifiability floor (1.88
  samples per fitted dimension at rank 32, against a floor of 10), so the
  transfer deficit was an estimation artifact.
- **N1** — unknown-class detection is near chance on realistic data for
  **every** scoring rule, including the freely composable controls.

The v13 frame is built on a single conclusion drawn from those three: **this
program has repeatedly gated on numbers whose operands it had not validated.**
The criteria below are therefore written so that each one names its control and
its corpus, and so that no threshold is inherited across a corpus boundary.

---

## 2. Deployment context — registered, because it was never specified

v12 specified no deployment setting, which is why its 1.0-point L1 gate could
never be justified: a tolerance is meaningless without a use for it. The v13
frame registers one. It is a **choice**, not a finding, and it is stated so that
later trade-off claims are checkable rather than rhetorical.

**Registered setting: assisted triage.** A domain expert reviews model decisions
that the model itself flags, and the explanation exists to make that review
faster or more accurate than the decision alone.

| Parameter                       | Registered value                                             |
| ------------------------------- | ------------------------------------------------------------ |
| Reviewer                        | Domain expert, not an ML practitioner                        |
| Explanation budget per decision | **≤ 30 seconds** to read                                     |
| Explanation complexity budget   | **≤ 10 active atoms** per decision                           |
| Review coverage                 | Flagged decisions only, not the full stream                  |
| Error cost                      | Asymmetric; a missed error costs more than a needless review |

**Consequences that follow, and are binding.**

- An explanation longer than 10 active atoms is **not** an explanation under
  this frame, however faithful it is. I2 exactness over a 384-dimensional dense
  score is exact and unreadable; it satisfies the letter of inspectability and
  none of its purpose. This is why I5 becomes primary.
- The L1 tolerance is now derivable rather than asserted. See Section 4.
- The utility expression in Section 8 becomes measurable, because
  `p(caught | explanation)` has a referent.

---

## 3. What changed from v12, and why

| #   | Change                                                         | Justification                                                                                                                                                                                                                         |
| --- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **I5 becomes primary and gating**                              | Forward simulation is the operand the program exists to move. v12 satisfied I2 at `1.14e-13` while I5 sat at 17.737%, below the kNN control at 25.246%. An exact explanation nobody can use is the failure mode the frame must price. |
| 2   | **I2 demoted to a non-gating structural check**                | Retained and binary. It is necessary and demonstrably far from sufficient.                                                                                                                                                            |
| 3   | **L1 becomes a reported axis with a 3.0-point tolerance**      | The 1.0-point figure was never derived from anything. The interpretable-by-design literature accepts 1–5 points. Now justified against Section 2 via Section 8.                                                                       |
| 4   | **L2 stays gating, but its threshold becomes corpus-relative** | **See Section 5. This is the substantive change and it is forced by N1.**                                                                                                                                                             |
| 5   | **New control: MLP + SHAP and Integrated Gradients**           | The program's stated goal is to beat a neural network on explainability, and it has never measured one. Without this control no comparative explainability claim is admissible.                                                       |
| 6   | **The frontier becomes the primary deliverable**               | `(accuracy, I5)` pairs with intervals for every head. A frontier is a result wherever the points fall, which removes the incentive to keep searching for a win.                                                                       |
| 7   | **Human-team utility must be quantified, not asserted**        | Section 8.                                                                                                                                                                                                                            |
| 8   | **Every open-set number carries its corpus and class count**   | Forced by N1. See Section 5.                                                                                                                                                                                                          |

---

## 4. Learnability (L-series)

| ID     | Name                | Threshold                                                                                              | Gating                                    | Measurement                                                                             |
| ------ | ------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------- | --------------------------------------------------------------------------------------- |
| **L1** | Accuracy parity     | Balanced accuracy within **3.0 points** of the strongest matched control                               | **Reported**, with the tolerance declared | Paired 95% bootstrap, seeds 11/23/37, against RBF, logistic, softmax, kNN               |
| **L2** | Open-set competence | **Corpus-relative — see Section 5**                                                                    | **Gating**                                | Unknown recall at matched known coverage, plus AUROC, on the same corpus as its control |
| **L3** | Sample efficiency   | Reported at 50, 200, 1000 per class; disqualified if minimum viable size scales with ambient dimension | Reported                                  | Component-family scaling                                                                |
| **L4** | Not-memorisation    | Parameter count and serialised size materially below the kNN control at equal or better accuracy       | Reported                                  | Against the corpus-matched kNN bar, **not** the 6.02 MB CIFAR figure                    |
| **L5** | Transfer            | Reproduces on at least one corpus beyond the primary                                                   | Reported                                  | Second-corpus replication at adequate sampling                                          |

**L1's 3.0 points, derived.** Under Section 2 a reviewer sees flagged decisions
only. A 3.0-point accuracy concession is admissible if the explanation recovers
more than 3.0 points of caught error within the 30-second budget — which is
exactly the quantity Section 8 requires be measured. If that measurement is not
made, **the trade is not claimed**, and L1 is reported as a deficit with no
compensating argument. The tolerance licenses a measurement, not an excuse.

---

## 5. L2 — restated, and why the 87.0% bar is withdrawn

**The v12 threshold was:** "unknown recall at matched known coverage at least
equal to the strongest support control (currently the v7 low-rank Gaussian at
**87.0%**)."

**That threshold is withdrawn as a cross-corpus constant.** It was established on
8-class CIFAR-10. N1 measured the achievable ceiling on 128-class DomainNet with
the instrument validated at both ends (far-field noise AUROC `1.0000`, held-out
knowns `0.5136`):

| Score on DomainNet   | AUROC  | Recall at 10% FA |
| -------------------- | ------ | ---------------- |
| Nearest class mean   | 0.5388 | 12.05%           |
| kNN nearest distance | 0.5824 | 19.86%           |
| Geometric, rank 16   | 0.5868 | **23.86%**       |

No method reaches 87% because **no method reaches 25%**. Carrying an absolute
recall bar from one corpus to another does not measure a detector; it measures
which corpus was chosen. Retaining it would have closed the program for a reason
unrelated to its hypothesis.

**Registered replacement.** L2 remains **gating**, with the threshold expressed
against controls measured on the identical corpus, features, and class count:

1. **Relative bar (gating).** Unknown recall at matched known coverage **at or
   above the strongest freely composable control** — kNN and nearest class mean —
   on the same corpus. Ties are failures; the bar is "at or above", and a method
   that merely matches free composition has not earned its complexity.
2. **Threshold-free bar (gating).** AUROC at or above the same controls.
   Reported alongside recall in every case, because recall at a threshold cannot
   distinguish a bad score from a bad threshold. **The program measured
   recall-at-threshold for years without recording the AUROC it was already
   computing.** That must not recur.
3. **Instrument validity (mandatory, non-negotiable).** Every open-set
   measurement carries a positive control (far-field synthetic points, AUROC must
   be near 1.0) and a negative control (held-out **known** samples, AUROC must be
   near 0.5). **A measurement without both is inadmissible**, whatever it reports.
4. **Absolute floor (reported, not gating).** State the absolute recall so a
   reader can see whether a relative win is deployable. **A relative win at 24%
   absolute recall is not a deployable detector, and must never be reported as
   one.**

**Corpus labelling.** Every open-set number in v13 is reported as
`(recall, AUROC, corpus, known-class count, samples per fitted dimension)`.
A bare percentage is not interpretable and is not admissible. N1 showed the v10–v12
figures of 0.902–0.972 collapse to 0.6374 when the same task width is run on a
different corpus; the difference was the corpus, not the detector.

---

## 6. Inspectability (I-series)

| ID     | Name                            | Threshold                                                                                                                                                                                                                                                           | Gating                            | Measurement                                                                                                                                               |
| ------ | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **I1** | Intrinsic parameter semantics   | Every parameter maps to a named geometric or statistical quantity                                                                                                                                                                                                   | **Gating**                        | Structural, binary                                                                                                                                        |
| **I2** | Exact score decomposition       | Residual zero to numerical tolerance                                                                                                                                                                                                                                | **Structural check, non-gating**  | Mean and max residual to machine epsilon                                                                                                                  |
| **I3** | Deletion faithfulness           | Top-k ablation exceeds random and bottom-k over a registered k sweep                                                                                                                                                                                                | Reported                          | k-sweep; ROAR only if retrained                                                                                                                           |
| **I4** | Minimum counterfactual distance | Closed form, and the displaced point empirically flips the decision                                                                                                                                                                                                 | Reported, **demoted from gating** | v12 established this is structurally unavailable for multiclass anisotropic quadratics; retaining it as a gate tests the model family, not the hypothesis |
| **I5** | **Forward simulation**          | **Two registered widths, conjunction rule, per plan Section 8 Amendment R4.** I5-8 (chance 12.50%): ≥40% confirms, 25–40% partial, ≤25% refutes. I5-128 (chance 0.781%): strictly above the kNN control by more than the seed spread. Neither may be reported alone | **Gating, primary**               | Registered simulatability protocol against chance, kNN, RBF, and MLP+SHAP/IG, no example leakage; explanations cite ≤10 atoms per decision                |

**Scope warning, carried forward unchanged.** If the representation is learned,
inspectability covers the **head over learned coordinates**, not the coordinates
themselves. The claim is: "the decision rule is exactly inspectable; the feature
semantics are not." M80–M82 exist to attack precisely that restriction; until
M82 passes its naming gate, the restriction stands.

---

## 7. Controls

No result is admissible without its control measured on the identical features.

| Control                        | Purpose                                                 | Registered at  |
| ------------------------------ | ------------------------------------------------------- | -------------- |
| Logistic / RBF / softmax       | Accuracy bar                                            | L1             |
| kNN                            | Free-composability bar for accuracy, open-set, and size | L1, L2, L4, I5 |
| Nearest class mean             | Second free-composability bar for open-set              | L2             |
| **MLP + SHAP**                 | The neural post-hoc explainability bar                  | I5             |
| **MLP + Integrated Gradients** | Second post-hoc bar                                     | I5             |
| Far-field synthetic points     | Positive control for any open-set measurement           | L2             |
| Held-out known samples         | Negative control for any open-set measurement           | L2             |
| Random subspace                | Positive control for basis identifiability              | M80–M83        |

---

## 8. Human-team utility — quantified or withheld

Where an accuracy-for-interpretability trade is claimed, it is expressed as:

$$U = \text{acc} + p(\text{caught} \mid \text{explanation}) \cdot \text{err} - \text{review\_cost}$$

with every term measured under the Section 2 setting. `p(caught | explanation)`
is **not currently measured** and has never been measured by this program. Until
it is, the frame permits reporting the frontier and forbids claiming the trade.
An unmeasured `p(caught | explanation)` is the assumption on which every
"interpretability is worth the accuracy" argument in this program has silently
rested.

---

## 9. Corpus policy

Registered in response to the I4 corpus defect and N1.

1. **Sample adequacy floor: 10 samples per fitted dimension.** Any cell below it
   is recorded void. Non-negotiable, carried from M78.
2. **Rank is capped at 53** on the current corpus (536 fitted samples per class).
   A higher rank requires a larger corpus first.
3. **Feature extraction at batch size 1.** The INT8 backbone's 49
   `DynamicQuantizeLinear` operators make batched features depend on batch
   membership.
4. **The current corpus is 61% quickdraw** and is stratified on class, not
   domain. Any milestone depending on semantic richness must restrict to a domain
   or rebuild with domain-stratified selection, and must say which.
5. **ImageNet is disqualified for any open-set or novelty measurement.** DINOv2's
   LVD-142M training set was built by retrieval seeded with ImageNet-22k and
   ImageNet-1k, and includes ImageNet-1k images. Held-out ImageNet classes are
   therefore not novel to this backbone, and a favourable result would be
   uninterpretable. If open-set is ever revisited, the instrument is a
   contamination-controlled benchmark such as NINCO or the OpenOOD
   iNaturalist/Places/Textures splits, registered in advance.
6. **ImageNet is permitted for nameability** (M82 and later), where no novelty
   claim is made and the WordNet synset hierarchy is a genuine asset. The switch
   must be registered, and cross-corpus comparisons against M77/M78 are void.
7. **Corpus substitution to rescue a failing gate is prohibited.** N1 established
   that this program's most favourable open-set numbers were corpus artifacts.
   A corpus may be changed for a stated methodological reason registered before
   the measurement, never after seeing a result.

---

## 10. Outcome taxonomy — defined, because it never was

Prior versions closed at lettered outcomes (v6.1 D, v7 C, v8 D, v9 D, v10 D,
v11 E, v12 E) but **no surviving document defines what those letters mean.** They
are therefore treated as opaque historical labels and are not relied upon. The
following applies to v13 only.

| Outcome | Meaning                                                                                          |
| ------- | ------------------------------------------------------------------------------------------------ |
| **A**   | Gating conjunction passes; the defended novelty position stands as registered                    |
| **B**   | Gating conjunction passes with a narrowed claim                                                  |
| **C**   | Frontier delivered: no dominance, but the `(accuracy, I5)` trade is characterised with intervals |
| **D**   | Decisive negative with an identified mechanism                                                   |
| **E**   | Decisive negative without an identified mechanism                                                |
| **F**   | Void — the measurement was invalid and nothing is concluded                                      |

**C is a success.** Registering it as such is the point of change 6: if only a
win counted, the rational move after each failure would be to keep changing the
setup until one appeared, which is how this program produced the v12 open-set
record that N1 later dissolved.

---

## 11. What does not change

1. **Final labels remain sealed** until M86.
2. **Byte-identical replay** from artifact indexes; the `.venv` replay
   environment (torch 2.13.0+cpu) is frozen and must not be modified.
3. **Determinism policy for any GPU backend must be resolved before that backend
   produces gated evidence.** The ROCm environment is qualified for exploration
   and throughput only.
4. **Every measurement operand requires a positive control.** M78 amendment R2
   voided a result whose stability operand was invalid, caught only because a
   control returned 40.45 degrees where near-zero was required.
5. **Rank and sample adequacy must vary orthogonally.** They moved together in
   every M78 cell, so no cell there separates capacity from estimation.
6. **A registered negative is a deliverable.** N1 is recorded, cited, and not
   buried.

---

## 12. Immediate consequence

M80 and M81 require **no new data and no trunk training** — they run against the
existing frozen features. M81 is the decisive milestone and carries the
registered three-way decision rule at 40% / 25% I5. The corpus question raised at
Section 9.6 is therefore **deferred to M82**, where naming makes it load-bearing,
rather than being settled by an expensive rebuild before the decisive measurement
has been taken.
