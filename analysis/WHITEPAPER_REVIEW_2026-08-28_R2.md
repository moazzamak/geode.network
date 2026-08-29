# WHITEPAPER REVIEW R2 — FRESH PASS + REPAIR PLAN

**Date:** 28 Aug 2026
**Scope:** `docs/WHITEPAPER_GEODE.tex` at the 28 Aug post-M340 state,
read in full (3,140 lines) as a standalone document.
**Prompt:** "do a fresh analysis of the whitepaper — infeasibility,
assumption breaks, cost-structure breaks, Byzantine weakness other
than quorum takeover, mistaken novelty vs prior art, readability."
**Relationship to prior reviews:** this pass deliberately did _not_
start from `FEASIBILITY_THREAT_REVIEW_2026-08-28.md` (F1–F10) or
`PRE_PUBLICATION_FEASIBILITY_REVIEW_2026-08-28.md` (N1–N5). It reads
the paper as an outside reviewer would. Where a finding overlaps a
closed item, that is noted; the findings below are the ones that
survive the M329–M348 repair wave.

**Severity ladder (v26 convention):**

- **CRITICAL** — a named defense does not do what it claims, or a
  load-bearing guarantee is false as written.
- **HIGH** — exploitable, or a published claim that a careful reader
  can falsify from the paper's own numbers.
- **MEDIUM** — needs a decision or a build before launch.
- **LOW** — polish.

**Status ladder:** ANALYTICAL = follows from the specification as
written; CODE-DIVERGENT = the paper and the shipped module disagree;
ARITHMETIC = the paper's own numbers contradict the claim.

---

## 1. Findings index

| ID     | Severity                             | Finding                                                                                                                                                                                                                                                                                                                                                                               | Milestone |
| ------ | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| G1     | CRITICAL                             | FHE tier returns the decrypted score vector — the raw-logit extraction oracle the bucket rule exists to suppress                                                                                                                                                                                                                                                                      | M349      |
| G2     | CRITICAL                             | Per-answer Bulletproofs is ~$10^8\times$ serving cost for the prover; verification is $O(n)$, not "a fraction"                                                                                                                                                                                                                                                                        | M350      |
| G3     | HIGH                                 | Proof layer and private serving are mutually exclusive (host never sees $z$)                                                                                                                                                                                                                                                                                                          | M350      |
| G4     | HIGH                                 | Ciphertext-determinism for the private probe conflicts with CKKS noise flooding                                                                                                                                                                                                                                                                                                       | M351      |
| G5     | HIGH                                 | Premium trunk tier is presented as available; Known Limits #21 concedes it is impractical                                                                                                                                                                                                                                                                                             | M352      |
| G6     | HIGH                                 | Private tier is unroutable ($v_a$ ~$10^4$ lower) and unrequestable (no descriptor field)                                                                                                                                                                                                                                                                                              | M352      |
| G7     | ~~CRITICAL~~ **MEDIUM, CONDITIONAL** | Behavioural dedup at 0.95 agreement locks out every second contributor on high-accuracy axes. **M353: failure does not reproduce on any measured axis; refusal is arithmetically impossible on both code axes. Risk begins above ~0.975 accuracy**                                                                                                                                    | M353      |
| G8     | HIGH                                 | Per-payer query budgets + public per-payer telemetry contradict "no control surface"                                                                                                                                                                                                                                                                                                  | M372      |
| G9     | MEDIUM                               | "Economic-only incentives, never identity checks" is contradicted by pedigree, nexus, behavioural identity                                                                                                                                                                                                                                                                            | M373      |
| G10    | MEDIUM                               | "Weights may be private" vs. the reference-executor role; the FHE fallback is self-judging                                                                                                                                                                                                                                                                                            | M374      |
| G11    | ~~HIGH~~ **CLOSED**                  | Assumption 3 (composition/chaining) is asserted from fusion evidence; no chain is measured. **M375: chain measured. Router→6 specialists 0.2741 beats the monolith 0.2450 on the sealed DomainNet cell (anchor 0.245014 reproduced exactly). Assumption 3 split into 3a/3b; 3b now measured, with its narrow scope stated**                                                           | M375      |
| G12    | ~~MEDIUM~~ **CLOSED**                | Two attribution rules in one paper (Shapley in the fee flow, leave-one-out for representations). **M376: Shapley named as the single rule; `representation.py` delegates to `chains.shapley_split`; chain length capped at 4. The divergence was measured, not argued — the two rules split M375's identical coalition values 0.0588 vs 0.0977 to the router, a factor of 1.66**      | M376      |
| G13    | CRITICAL                             | Voting weight is buyable at a ~5% haircut; the self-payment exclusion is single-hop                                                                                                                                                                                                                                                                                                   | M358      |
| G14    | MEDIUM                               | Slashing L3 burns unclaimed credits = voting weight; the ladder prices governance participation                                                                                                                                                                                                                                                                                       | M359      |
| G15    | LOW/ARITHMETIC                       | "Single-row influence sits below four-significant-digit resolution" is false below ~$10^4$ rows                                                                                                                                                                                                                                                                                       | M377      |
| G16    | HIGH                                 | Route seed has no per-session entropy → the lottery is a per-epoch winner-take-all. **M354: CONFIRMED — measured 1.000 to one arm under the protocol's own anchor cadence; published figure came from a per-session anchor. Repaired**                                                                                                                                                | M354      |
| G17    | HIGH                                 | Librarian, batch verifier, scoring/replay environments, tally committee, gateways have no income line                                                                                                                                                                                                                                                                                 | M363      |
| G18    | HIGH                                 | The zakat end state defunds the standard library, audits, and pinning it currently pays for                                                                                                                                                                                                                                                                                           | M362      |
| G19    | MEDIUM                               | Abstention metered at 0.5× while consuming full trunk compute — structurally loss-making                                                                                                                                                                                                                                                                                              | M357      |
| G20    | HIGH                                 | Coverage-multiplied score inverts quality; best-quality mode returns the worse arm                                                                                                                                                                                                                                                                                                    | M355      |
| G21    | MEDIUM                               | ETH-denominated timelocked prices + 4-epoch vesting = unhedgeable short-vol position for suppliers                                                                                                                                                                                                                                                                                    | M360      |
| G22    | HIGH                                 | The developer's "reference hosting cost" sets the floor, its own price, and every rival's bond                                                                                                                                                                                                                                                                                        | M361      |
| G23    | CRITICAL                             | Targeted DoS after an answer commit converts into Level-1 burns; collides with Level 0 (no slash for downtime)                                                                                                                                                                                                                                                                        | M364      |
| G24    | ~~HIGH~~ **CLOSED**                  | Force-inclusion inbox deposit is fully refunded on incorporation → chain-bloat/invalidity for gas. **M365 SEALED, PASS (29 Aug):** non-refundable posting fee to the operations line, superlinear per-epoch rate limit (1,000 entries cost 330,839× the flat fee), capped per-epoch obligation with roll-forward, FIFO incorporation, O(1) validity. 9 new EVM gates.                 | M365      |
| G25    | ~~HIGH~~ **CLOSED**                  | Librarian replacement uses a raw headcount of registered validators, escaping the earned-weight rule. **M366 SEALED, PASS (29 Aug):** replacement now runs on externally-verified weight with the pedigree gate, 20% cap, diversity floor and two-thirds bar; a 40-strong weightless Sybil fleet cannot fire it. Composition defect found and fixed en route (see below).             | M366      |
| G26    | HIGH                                 | Five floor-priced barely-passing arms can hold all top-5 lottery slots and evict the best arm                                                                                                                                                                                                                                                                                         | M356      |
| G27    | HIGH                                 | "Detection horizon stays inside the vesting promise everywhere" is false on quiet axes (~119 epochs); the claim freeze then strands honest suppliers                                                                                                                                                                                                                                  | M367      |
| G28    | MEDIUM                               | No minimum executor-pool size; at pool size 2 the collusion argument is vacuous                                                                                                                                                                                                                                                                                                       | M368      |
| G29    | MEDIUM                               | Challenge corpus is depletable by paid registrations; root-at-creation conflicts with rotation                                                                                                                                                                                                                                                                                        | M369      |
| G30    | MEDIUM                               | Two generations of the challenge design coexist (drawn-not-authored vs. the wrong-label audit path)                                                                                                                                                                                                                                                                                   | M370      |
| G31    | MEDIUM                               | The randomness beacon is the single trust point under every sampling guarantee, and is not in Known Limits                                                                                                                                                                                                                                                                            | M371      |
| G32    | HIGH                                 | Rahimi & Recht (2007) uncited despite a results row named "random features"                                                                                                                                                                                                                                                                                                           | M378      |
| G33    | HIGH                                 | Numerai — sealed held-out scoring, payment by measured out-of-sample contribution — uncited                                                                                                                                                                                                                                                                                           | M378      |
| G34    | MEDIUM                               | The shadow probe's literature (ringers, uncheatable grid computing, BOINC replication) is unnamed                                                                                                                                                                                                                                                                                     | M378      |
| G35    | MEDIUM                               | Dispute-by-replay is Truebit's verification game; the lineage is uncited                                                                                                                                                                                                                                                                                                              | M378      |
| G36    | MEDIUM                               | Behavioural identity is model fingerprinting / Proof-of-Learning; uncited                                                                                                                                                                                                                                                                                                             | M378      |
| G37    | MEDIUM                               | zkML line (Kang et al. 2022, zkCNN, EZKL) uncited while the paper proposes its own head proof                                                                                                                                                                                                                                                                                         | M378      |
| G38    | MEDIUM                               | Ocean Protocol compute-to-data is the sealed scoring environment; uncited                                                                                                                                                                                                                                                                                                             | M378      |
| G39    | LOW                                  | Feature stores found by the repo's own sweep but absent from the paper's neighbours list                                                                                                                                                                                                                                                                                              | M378      |
| G40    | HIGH                                 | MeritRank (Sybil-tolerant reputation by graded decay) uncited — the D3 partial displacer of G13's repair, found by sweep 3                                                                                                                                                                                                                                                            | M378      |
| G41    | HIGH                                 | _Concave is the New Linear_ uncited — a published impossibility result that covers the 20% cap unless weight is genuinely earned                                                                                                                                                                                                                                                      | M378      |
| G42    | MEDIUM                               | Wash-trade forensics and trust-graph Sybil resilience uncited beside §Voting weight and the $d\ge3$ diversity floor                                                                                                                                                                                                                                                                   | M378      |
| G44    | HIGH                                 | Agarwal et al., _A Marketplace for Data_ (EC 2019) uncited — the closest published relative of pay-by-measured-contribution; found only by the second index                                                                                                                                                                                                                           | M378      |
| G45    | MEDIUM                               | Chen et al., model-based pricing for ML in a data marketplace (SIGMOD 2019), uncited                                                                                                                                                                                                                                                                                                  | M378      |
| G46    | LOW                                  | Sun et al., profit-maximising model marketplace with DP federated learning (INFOCOM 2022), uncited                                                                                                                                                                                                                                                                                    | M378      |
| G47    | MEDIUM                               | _SafetyNets_ (NIPS 2017) uncited — the ancestor of §Serving verification                                                                                                                                                                                                                                                                                                              | M378      |
| G48    | LOW                                  | _ZEN_ (IACR 2021) uncited beside the G37 zkML entries                                                                                                                                                                                                                                                                                                                                 | M378      |
| G49    | HIGH                                 | The paper dates the adapter line to AdapterHub (2020); residual adapters are Rebuffi et al. (2017). An attribution error, not just a missing citation                                                                                                                                                                                                                                 | M378      |
| G50    | MEDIUM                               | _LoraHub_ (2023) uncited — the closest composition-of-independently-trained-modules relative                                                                                                                                                                                                                                                                                          | M378      |
| G51    | HIGH                                 | An axis serving <30 sessions/epoch cannot close the detection horizon inside vesting at **any** probe rate. Arithmetic, not tuning. Found by M367                                                                                                                                                                                                                                     | M367      |
| G52    | MEDIUM                               | §The ledger says "every sample in the protocol derives from that beacon"; the route lottery is a sample and derives from the anchor instead. Found by M354                                                                                                                                                                                                                            | M354      |
| G53    | ~~CRITICAL~~ **CLOSED**              | The librarian-replacement vote has no on-chain execution path at maturity: `setLibrarian` is `onlyOwner`, and the repo's own gate proves a renounced owner closes it forever. `InclusionInbox.librarian` is `immutable` besides. **M382 SEALED, PASS (29 Aug):** governance authority split from ownership; inbox reads the librarian from one source. Scope limits registered below. | M382      |
| R1–R12 | LOW                                  | Readability (glossary, undefined "promise", three names for one balance, table legends, notation, ordering)                                                                                                                                                                                                                                                                           | M379–M381 |

---

## 2. Infeasible as written

### G1 — The private tier defeats its own extraction guard [CRITICAL, ANALYTICAL]

**Where.** §The system as a black box (bucket rule) vs. §Serving
verification (private serving).

**The conflict.** The bucket rule exists because "a margin-annotated
answer stream would let a buyer solve for the head as a linear system
in $d\cdot C$ unknowns," and the M332 gate prices bucketed extraction
at 55.2× lifetime revenue against a raw-margin oracle at 2.8×. But the
private path says the host "returns the encrypted score vector. The
device decrypts and takes the argmax on-device." The device therefore
receives the _entire_ score vector in the clear on every query — a
strictly stronger oracle than the raw margin. $d\cdot C$ queries
recover $W$ exactly (Tramèr et al., USENIX Security 2016).

**Consequence.** On the private tier the extraction multiple is
_below_ 2.8×, not 55×, and "no party ever holds $W$ in plaintext
except the contributor's own host, so model privacy is unconditional"
is false. The tier that is the strategic moat is the tier with no
extraction defense.

**Proposed repair.** Move the bucketing inside the encryption. The
comparison $\kappa(x)$ against registered bucket edges is a bounded
comparison circuit; under a leveled scheme it is evaluable, at a
multiple of the linear head's cost. Three options, in preference
order:

1. **Encrypted bucketing (preferred).** The host evaluates
   $W^\top z$, then the argmax-and-bucket circuit, and returns only
   the encrypted (label, bucket) pair. Cost must be measured, not
   assumed — this is the gate.
2. **Client-side attested readout.** The device decrypts the vector
   but the protocol treats the private tier as _disclosed-oracle_ and
   publishes its own extraction multiple beside the 55× figure.
3. **Per-payer private-tier budget** sized from the measured
   extraction multiple, i.e. price the tier so extraction is not free.

Option 2 alone is honest but gives up the moat; the paper must not
claim unconditional model privacy under it.

**Gate (M349).** Measure the FHE cost of the argmax+bucket circuit on
the vision head's dimensions on one CPU core. PASS if total private
query latency stays within 5× of the 20 s head-only figure; otherwise
option 2 is adopted and the paper's private-tier extraction multiple
is measured and published.

---

#### §M349 — OPTION 2 ADOPTED (29 Aug 2026). Encrypted bucketing measured infeasible.

Instrument `tools/m349_encrypted_bucketing.py`, evidence
`analysis/m349_encrypted_bucketing.json`. The gate's first branch
required the argmax+bucket circuit inside FHE to stay within 5× of
the head-only figure (sealed M322e-D G5: 20.3–26.4 s/query, PASS
bound 100 s). It does not, and the measurements are stronger than the
verdict:

| Measurement                                                                                                          | Reading                                                                                                                                                  |
| -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Registered context (8192, [60,40,40,60]) holds a pairwise max                                                        | **No** — `scale out of bounds` on the first ciphertext×ciphertext multiply. TenSEAL 0.3.17 exposes no rescale and consumes scale on plain multiplies too |
| Deepest workable comparison tree, any context                                                                        | depth 2 (C = 4) only, at poly degree 16384 / 6 levels, ~22 ms per max — and with the crude degree-2 \|x\|                                                |
| Depth 3+ tree (C = 8)                                                                                                | `scale out of bounds` at every context that builds                                                                                                       |
| Decision-grade \|x\| polynomial                                                                                      | degree-2 worst error 3.0, degree-8 1.08, degree-16 6.46 on [-16, 16] — the least-squares fits are not decision-grade                                     |
| Honest extrapolation to C = 601 (degree-8, 3 muls/max, degree 65536 for a ~1320-bit modulus, labelled as assumption) | argmax ≈ 475 s + bucket ≈ 6 s + head 23 s ≈ **505 s — five times the 100 s bound**                                                                       |

The head's depth-one structure is at the very edge of the scale
budget; a 601-way argmax is a depth-10 tree of decision-grade
comparisons — roughly thirty multiplications in series. That is not a
tuning problem.

**Consequence — option 2, registered.** The private tier returns the
full score vector to the device; the device readout is a disclosed
oracle. The tier's extraction multiple is therefore the
score-vector-oracle figure, **2.8× expected lifetime revenue** (the
registered M332 figure; the bucketed plaintext tier stays at 55×),
and the private tier is priced on it. The paper's "model privacy is
unconditional" sentence is deleted and replaced with the economic
boundary; the device readout paragraph in §Serving verification now
says so in place. Whitepaper validated (966/966 braces).

**Two lessons.** First, _a depth-one circuit is not a circuit family_:
the head was deliberately built at the edge of the scheme's budget,
and every honest measurement in this milestone is a scale failure
first and a latency number second. Second, the gate's fallback branch
existed for exactly this outcome, and taking it required publishing
the measured multiple — the repair is not "option 2" in the abstract,
it is "option 2 with 2.8× on the page."

---

#### §M351 — SEALED, PASS (29 Aug 2026). Committed-seed flooding shipped.

Instrument `tools/m351_committed_seed_fhe.py`, evidence
`analysis/m351_committed_seed_fhe.json`. G4 named the collision: the
private probe needs deterministic output ciphertexts, but CKKS draws
fresh randomness at encryption, so the output leaks through its noise
(IND-CPA^D) and a bit-comparison probe has nothing to compare. The
repair is committed-seed flooding, and the gate's equivalence test now
passes on the registered backend.

| Gate clause                          | Reading                                                                               |
| ------------------------------------ | ------------------------------------------------------------------------------------- |
| The defect reproduces                | two unflooded evaluations of the same input differ **byte-for-byte** (131 KB outputs) |
| Same committed seed → byte-identical | **true** — the flood is bound to the seed; the only free randomness is gone           |
| Different seed → differ              | **true**                                                                              |
| Security parameters unchanged        | one context, one key pair, both evaluations; the flood is a standard zero-encryption  |
| The flood preserves the computation  | flooded vs unflooded max error **2.4e-9** (the flood adds noise, never signal)        |

**The construction, stated plainly.** TenSEAL 0.3.17 exposes no RNG
seed control, so the commitment is to the flood ciphertext itself: the
host commits to a standard encryption of zero, generated once under
the registered context and bound to the seed, and every evaluation
under that seed returns `head(input) + flood`. The head evaluation is
deterministic given the input bytes, so the flood is the only
randomness, and it is fixed per committed seed. The executor replays
with the same committed flood and gets the same bytes.

**Registered residual.** The flood is fixed per committed seed, not
fresh per evaluation — that is exactly what determinism requires, and
it is what the commitment binds. The IND-CPA^D mitigation therefore
holds within a committed session; a fresh commitment is a fresh flood.
The paper should say "deterministic within a committed seed", never
"deterministic" without the qualification.

Whitepaper text follows (the private-probe determinism sentence gains
the committed-seed construction).

---

### G2 — Per-answer proofs are off by ~8 orders of magnitude [CRITICAL, ARITHMETIC]

**Where.** §Proofs of computation: "An answer also carries a proof of
the computation behind it… A verifier checks that proof at a fraction
of the cost of redoing the computation."

**The arithmetic.** Bulletproofs proof _size_ is $O(\log n)$; prover
and verifier _time_ are $O(n)$ group operations in $n = d\cdot C$. At
the paper's own vision axis (DINOv2-L features, 601 classes),
$n \approx 6\times10^5$:

| Quantity              | Cost                                         |
| --------------------- | -------------------------------------------- |
| Serving the head      | ~$6\times10^5$ multiply-adds ≈ microseconds  |
| Bulletproofs prover   | ~$5\times10^6$ group exponentiations ≈ 100 s |
| Bulletproofs verifier | one $n$-sized multi-exponentiation ≈ seconds |
| Proof size            | ~44 group elements ≈ 1.4 KB                  |

So the prover is ~$10^8\times$ serving cost and the verifier is
~$10^6\times$. "A fraction of the cost of redoing the computation" is
false for a linear head — redoing the computation is the cheapest
operation in the system.

**Why the claim was reachable.** The correct benefit is _verification
without disclosure_: the verifier cannot redo the computation because
$W$ is private, not because the proof is cheaper. That is the true and
defensible statement.

**Proposed repair.**

1. Restate the benefit as verification without disclosure; delete the
   cost comparison.
2. Move proofs from per-answer to **sampled per-settlement-batch**,
   which the paper already describes and is the only affordable
   placement. Say explicitly that an individual answer does not carry
   a proof; a sampled fraction of batches does.
3. State the prover cost as a per-axis registered number, priced into
   the fee flow (today it is unpriced — see G17).
4. Keep the $O(\log n)$ _size_ claim; label it as size.

**Gate (M350).** Publish a proof-cost table (prover seconds, verifier
seconds, proof bytes) per registered axis dimension, measured on the
reference CPU. The paper's proof section may make no cost claim not
present in that table.

#### M350 — SEALED (gate PASS), and **G2's magnitude was overstated by this review**

Instrument `tools/m350_proof_cost.py`, evidence
`analysis/m350_proof_cost_table.json`.

**Instrument defect caught and recorded, not silently fixed.** The
first draft modelled multi-exponentiation with libsodium's
`crypto_core_ed25519_add`, measured at 17.2 µs against 139.6 µs for
a full scalar multiplication — a ratio of 8 where ~306 is expected.
That primitive operates on the _compressed_ encoding, so it
decompresses both points, adds, and recompresses; it measures two
field square roots, not a group addition. Using it would have
inflated every verifier figure ~40×. All costs are therefore carried
in scalar-multiplication equivalents. Measured net scalar
multiplication: **152.3 µs** (Python FFI overhead measured separately
at 1.20 µs and subtracted).

| Axis                          | $n=dC$  | Serve (measured) | Prove   | Verify  | Proof bytes | Verify / serve |
| ----------------------------- | ------- | ---------------- | ------- | ------- | ----------- | -------------- |
| Text (BERT-base, SST-2)       | 1,536   | 1.2 µs           | 0.09 s  | 0.034 s | 1,120       | 27,180×        |
| Text (BERT-base, MNLI-m)      | 2,304   | 1.4 µs           | 0.13 s  | 0.048 s | 1,184       | 35,124×        |
| Audio (wav2vec2, SC-v2)       | 26,880  | 4.2 µs           | 1.18 s  | 0.434 s | 1,376       | 104,437×       |
| Image routing (DINOv2-L, 345) | 353,280 | 35.8 µs          | 12.80 s | 4.615 s | 1,632       | 129,012×       |
| Vision (DINOv2-L, 601)        | 615,424 | 102.0 µs         | 21.48 s | 7.721 s | 1,696       | 75,721×        |

**Correction to this review.** G2 asserted the prover is
$\sim10^8\times$ serving cost. Measured, under the model most
favourable to the proof system (Pippenger multi-exponentiation), it
is $\sim2\times10^5\times$ on the vision axis — and $2\times10^4$
even granting a hypothetical implementation ten times faster than
this one. Under the _naive_ $8n$ model the ratio is
$7.4\times10^6$, still not $10^8$. The source of the error is
identified: G2 took serving as "microseconds" when a 615,424-MAC
float64 mat-vec measures **102 µs**, roughly 100× slower than
assumed. **G2's magnitude was wrong by two to three orders of
magnitude.**

**The finding survives its own correction, and that is the point.**
The sentence under test is "a verifier checks that proof at a
fraction of the cost of redoing the computation." Redoing the
computation is 102 µs. Checking the proof is 7.72 s — **75,721×
more**, not a fraction. The claim is false on every registered axis,
by between four and five orders of magnitude, and it is false in the
direction that matters: for a linear head, _redoing the computation
is the cheapest operation in the system_. No arithmetic correction
rescues it, which is why the repair (restate the benefit as
verification without disclosure; move proofs to sampled batches)
stands unchanged.

The $O(\log n)$ _size_ claim is confirmed and kept: proofs are
1.1–1.7 KB across a 400× range in $n$.

---

### G3 — Proofs and private serving are mutually exclusive [HIGH, ANALYTICAL]

On the private tier the host never holds $z$ in plaintext, so it
cannot produce a Bulletproofs argument about $W^\top z$ without
verifiable FHE — research-grade, and multiplicative on top of the FHE
cost. The paper never states that the proof layer is unavailable
there.

**Proposed repair.** A per-tier capability matrix in §Serving
verification: rows = plaintext / device-encoder / private (FHE);
columns = shadow probe, behavioural identity, head proof, dispute
replay. Fill every cell with available / unavailable / residual. The
paper already builds this argument in prose ("the operative mechanism,
per tier"); the matrix makes the gaps unhidable. Folded into M350.

---

### G4 — Ciphertext determinism vs. CKKS security [HIGH, ANALYTICAL]

**Where.** Known Limits #12 and §Serving verification: the private
probe requires the output ciphertext to be "deterministic given the
input ciphertext and the sealed head."

**The conflict.** Approximate-FHE schemes require noise flooding or
re-randomisation before an evaluation result is released to the key
holder (the IND-CPA$^D$ attack, Li & Micciancio, EUROCRYPT 2021).
Fresh randomness makes the output ciphertext non-deterministic, and a
bit-comparison probe has nothing to compare.

**Proposed repair.** Two admissible constructions, pick one and
register it:

1. **Committed-seed flooding.** The flooding randomness is derived
   from a seed the host commits alongside its answer commit; the
   executor replays with the same seed. Determinism restored, security
   preserved, and the seed commit is already the pattern used
   elsewhere in the protocol.
2. **Decrypted-margin band.** The comparison happens after decryption
   inside the sealed replay environment, against the same margin band
   the plaintext probe uses. This gives up "no plaintext point exists
   in the probe at all."

Option 1 preserves the paper's stronger claim and costs nothing. Note
that the paper's existing numerics policy already uses margin-gating
rather than bit-comparison for the plaintext path — the private path
should not claim a _stronger_ comparison than the plaintext one.

**Gate (M351).** A shipped equivalence test: two independent
evaluations with the same committed seed produce byte-identical
ciphertexts; two with different seeds do not; the scheme's security
parameters are unchanged between them.

---

### G5/G6 — The premium tier is conceded impractical, unroutable, and unrequestable [HIGH]

**Three defects on one mechanism.**

1. **Contradiction.** §Serving verification presents FHE trunk
   evaluation as an available tier ("SOTA trunks the device cannot run
   use the premium tier"). Known Limits #21 concedes trunk-level
   homomorphic evaluation is "orders of magnitude higher cost per
   token — conceded as impractical today." Only one may stand.
2. **Unroutable.** The head path is measured at ~20 s/query against a
   plaintext head at ~1 ms — a $10^4$ multiple. The router ranks on
   $v_a = s_a/(p_a\bar u_a)$, so a private-tier arm sits ~$10^4$ below
   any plaintext arm on the same axis and never enters the top five.
   The moat is unreachable through the network's own ranking rule.
3. **Unrequestable.** The task descriptor has fields for input type,
   output type, axis metric, unit, routing mode, max unit price, and
   max spend. There is **no privacy or tier field**. A user cannot
   declare a preference for private serving at all.

**Proposed repair.** Treat private serving as a **separate axis**, not
a price point on an existing one. Concretely:

- Add `privacy_tier ∈ {plaintext, device_encoder, private}` to the
  task descriptor, and add it to the unit table so the descriptor hash
  freezes it.
- An axis is `(input, output, metric, floor, privacy_tier)`. Arms
  compete inside a tier; $v_a$ never compares across tiers.
- The private tier carries its own price floor (the measured FHE
  reference cost) and its own published quality floor.
- §Serving verification adopts the Known Limits #21 concession in
  place: the trunk-level premium tier is a **registered research
  target with a cost trigger**, not a shipped tier.

**Gate (M352).** The descriptor schema, the unit table, and the router
regression tests all carry the tier field; a cross-tier route is a
refusal, not a ranking. The paper contains no sentence describing the
trunk premium tier as available.

#### §M352 — SEALED, PASS (28 Aug 2026)

Code: [geode/core/declared_label_set.py](geode/core/declared_label_set.py)
(shared with M355 — the tier and the declared label set are both
qualification clauses, so they belong in one gate).
Tests: [tests/unit/test_v26_m355_declared_label_set.py](tests/unit/test_v26_m355_declared_label_set.py),
16 passing.

- `PrivacyTier ∈ {plaintext, device_encoder, private}` is on both
  `Declaration` and `Capability`. **The enum deliberately defines no
  ordering** — these are separate markets, not grades of one, and any
  ordering would eventually be read as a ranking.
- `qualifies()` fails on a tier mismatch before it looks at labels;
  `declared_score()` raises `CrossTierRoute` rather than returning a
  number. `test_a_better_arm_in_another_tier_never_wins` pins that a
  0.99 private arm loses to a 0.60 plaintext arm under a plaintext
  declaration — refusal, not ranking.
- Paper: `privacy_tier` is a task-descriptor field, the axis
  definition is `(input, output, metric, floor, privacy_tier)`, the
  router section states that ranking is within-tier, and §Serving
  verification now calls the trunk premium tier "not a shipped
  tier … a registered research target with a cost trigger",
  adopting the Known Limits #21 concession in place. No sentence
  describing it as available remains.

**Residual, stated not repaired.** The unit table and descriptor
hash live in the Solidity settlement contract, not in `geode/`. The
tier is frozen in the Python descriptor path only; carrying it into
the on-chain unit table is deferred with the rest of the contract
work and is not claimed here.

---

### G7 — Behavioural dedup locks incumbents in [CRITICAL, ANALYTICAL]

**Where.** §Registration: "Two artifacts whose profiles agree above
$0.95$ are the same artifact for registration, whatever their hashes."

**The failure.** Two _independently built_ strong arms agree at
roughly $1 - e_1 - e_2$. On the speech axis (WER 0.026) or the code
axis (pass@1 0.860), any genuine competitor agrees with the incumbent
well above 0.95 and is refused as a duplicate. The rule blocks
competition precisely on the axes where the recipe works, and it
propagates: "behavioural identity" is also the unit of the 20% voting
cap and the $d\ge3$ diversity rule, so a high-accuracy axis collapses
to one identity for governance too.

**Proposed repair.** Deduplicate on the _disagreement structure_, not
on raw agreement. A challenger is a distinct artifact if it fixes rows
the incumbent misses:

$$
\text{novelty}(B \mid A) \;=\;
\frac{\bigl|\{i : \hat y^B_i = y_i \,\wedge\, \hat y^A_i \neq y_i\}\bigr|}
     {\bigl|\{i : \hat y^A_i \neq y_i\}\bigr|}
$$

— the share of the incumbent's errors the challenger repairs. Register
a floor (proposal: 0.10) and combine it with the existing exact-hash
rule. A one-bit-flip copy has novelty 0 and is still refused; a
genuinely different arm at 99% accuracy has non-trivial novelty and is
admitted. The metric is computed inside the sealed scoring environment
from the same challenge answers already collected, so it costs
nothing new.

**Gate (M353).** Reproduce the failure first: on the sealed speech and
code evidence, show that the 0.95 rule refuses a legitimate distinct
arm. Then show the novelty rule admits it and still refuses a
bit-flip copy and a distilled near-clone. Both directions must hold
before the paper text changes.

#### §M353 — SEALED. The failure does not reproduce; G7 is re-scoped

Instrument: [tools/m353_dedup_agreement.py](tools/m353_dedup_agreement.py).
Evidence: [analysis/m353_dedup_agreement.json](analysis/m353_dedup_agreement.json).

**The gate's first clause failed, and that is the result.** The 0.95
rule did not refuse a legitimate distinct arm on any axis this paper
measures. The paper text does not change, because the premise it would
have changed for is not established.

**Analytic half.** On a correctness-mask profile, two arms with
accuracies $a_1 \ge a_2$ satisfy

$$\max(0,\, a_1 + a_2 - 1) \;\le\; \text{agreement} \;\le\; 1 - (a_1 - a_2).$$

A refusal at $\tau$ is _possible_ only when $a_1 - a_2 \le 1 - \tau$
and _forced_ only when $a_1 + a_2 - 1 > \tau$. At $\tau = 0.95$ the
forced case needs two arms both above about $0.975$.

| Axis (sealed readings)                    | Agreement range      | Refusal possible? | Forced? |
| ----------------------------------------- | -------------------- | ----------------- | ------- |
| Speech ASR, LibriSpeech WER 0.02957       | [0.9409, 1.0000]     | **yes**           | no      |
| Speech classification, SC-v2 M266b 0.8787 | [0.7574, 1.0000]     | yes               | no      |
| Code, HumanEval M287 0.8598 vs 0.5976     | [0.4574, **0.7378**] | **no**            | no      |
| Code, HumanEval M268 0.5976 vs 0.5061     | [0.1037, **0.9085**] | **no**            | no      |
| Vision, Open Images served subset 0.901   | [0.8020, 1.0000]     | yes               | no      |

**This refutes G7's own example.** G7 names "the code axis (pass@1
0.860)" as a case where "any genuine competitor agrees with the
incumbent well above 0.95". Agreement between the paper's two sealed
code arms cannot exceed $0.7378$ — the rule cannot refuse them at any
error structure. The review asserted an instance it had not computed.

**Measured half (Speech Commands v2).** The sealed M266b ridge probe
was refit from the intact wav2vec2 cache and reproduced its registered
accuracy exactly ($0.8787$) before anything was read off it.

| Arm                                           | Accuracy | Agreement | Refused by 0.95? | Novelty | Admitted by novelty rule? |
| --------------------------------------------- | -------- | --------- | ---------------- | ------- | ------------------------- |
| Random features + ridge (strong distinct arm) | 0.7662   | 0.8005    | **no**           | 0.1236  | **yes**                   |
| Nearest class mean (weak distinct arm)        | 0.3763   | 0.3891    | no               | 0.0861  | no                        |
| Bit-flip copy (clone control)                 | 0.8786   | 0.9999    | **yes**          | 0.0000  | no                        |
| Distilled near-clone (clone control)          | 0.8494   | 0.9522    | **yes**          | 0.0210  | no                        |

The 0.95 rule behaved exactly as designed: it admitted the distinct
arm and refused both clones. The distilled near-clone is the
informative control — trained on the incumbent's own outputs rather
than the truth, it lands at $0.9522$, just over the line, and is
correctly caught by both rules.

**Re-scoping.** G7 is downgraded from CRITICAL to **CONDITIONAL,
MEDIUM**: a real lock-out risk on axes above roughly $0.975$, which
this paper does not currently have. The novelty rule is still worth
adopting — it costs nothing, it is computed from answers already
collected, and it closes the risk before an axis reaches that regime —
but it is adopted as insurance, not as a repair to a demonstrated
failure, and the paper must say which.

**Residual, stated not repaired.** The proposed novelty floor of
$0.10$ sits uncomfortably close to what the legitimate strong arm
scores ($0.1236$). An equally legitimate arm whose errors correlate
more with the incumbent's would fall below it and be refused —
the floor would then reproduce the very lock-out it was introduced to
prevent. The floor is an unmeasured tuning parameter and must not be
published as anything else.

**Not measured.** The code-axis per-item evidence (M268, M287) lived
under `logs/` and did not survive the public-release squash; both F:
cache directories are empty and regenerating it needs two LLM
generation passes. The code axis is settled analytically instead,
which suffices here only because the analytic result is an
impossibility bound rather than an estimate.

---

## 3. Assumption and principle breaks

### G8 — Query budgets contradict "no control surface" [HIGH]

**Where.** Design principles ("The network has no user, region, or
content selection path… A coercer can point at artifacts, never at
audiences") vs. §Currency and pricing ("Every payer holds a per-axis,
per-epoch query budget… The used-over-cap rate is a ledger-visible
number per payer per axis per epoch").

**The break.** A per-payer budget is a per-user refusal lever, and a
public per-payer consumption rate is a per-user telemetry stream on an
immutable ledger. Both are exactly the audience-pointing surface the
principle denies. On the transcript axis the public meter is _audio
seconds_, so the ledger also publishes the duration of each user's
recording bound to a payer address.

**Proposed repair.**

- Keep the budget as a **local gateway rule**, not a protocol-level
  per-payer ledger object. Enforcement stays; the public record does
  not.
- If a network-level bound is required for the extraction argument
  (G1/M349), publish it **aggregated per axis per epoch**, never per
  payer.
- Restate the principle honestly: the protocol has no user-_selection_
  path, but it does meter per payer; a payer address is pseudonymous
  and the metering is local.
- Add a Known Limits entry for meter-side metadata leakage on
  duration-metered axes, with the mitigation (padding to a registered
  quantum, e.g. metering audio in 15-second blocks).

**Gate (M372).** No per-payer field appears in any ledger entry type.
A replay of a session's route and payment succeeds without it.

#### §M372 — SEALED, PASS (29 Aug 2026). The budget is gateway-local; the ledger stays payer-free.

`geode/core/extraction_guard.py` carried the R-A2c budget with the
enforcement surface G8 named: "the ledger-visible used/cap rate per
payer per axis per epoch." The milestone moved the surface, not the
mechanism:

- **The budget is a gateway-local rule.** Enforcement is unchanged
  (grant/consume/exhaust, never silent), but it is the gateway's own
  view of its own user, and nothing per payer is publishable.
- **The only publishable view is the axis aggregate.** New
  `PayerBudgetLedger.axis_rate(axis, epoch)` sums used and cap across
  all payers for the epoch — no payer dimension. A network-level bound
  may rest on that and on nothing finer.
- **Duration metering pads to a quantum.** New `pad_duration(seconds)`
  meters in whole 15-second blocks (`DURATION_QUANTUM_SECONDS = 15`),
  so the meter cannot be turned into a per-session length stream.

Gates (26 passing in the two suites):

| Gate clause                                   | Reading                                                                    |
| --------------------------------------------- | -------------------------------------------------------------------------- |
| No per-payer budget field in any ledger entry | settlement entries/batches contain no cap/used/rate/quotum key             |
| Replay succeeds without a per-payer field     | `verify_batch_rules` + `recompute_batch_hash` pass on the report alone     |
| Duration leak padded                          | 1 s and 14 s both meter as one 15 s block; 15.1 s as two                   |
| Aggregate has no payer dimension              | two payers, 4/10 and 6/10 used → axis rate 0.5; per-payer rate still local |

Whitepaper: the "No control surface" principle now reads "no
user-_selection_ path … metering enforced locally … nothing per payer
ever written to the public ledger"; the Query budgets bullet is the
gateway-local rule with the aggregate as the only publishable view;
two Known Limits entries carry the meter-side leakage and its padding,
and the local-not-protocol residual. Whitepaper validated (969/969
braces).

**The lesson.** _The enforcement surface is part of the mechanism, not
an implementation detail._ The budget that G8 called a contradiction
was the same code G8's repair keeps; what changed is where its output
may go. A finding about what a mechanism publishes is often fixable by
changing the publication boundary, not the mechanism.

---

### G9 — "Economic-only incentives, never identity checks" [MEDIUM]

Contradicted three times by the paper's own machinery: the anti-Sybil
layer is time-based pedigree (activation window, activity floor,
tenure) — persistence of identity, not price; the authority-key
registry turns on "incorporation records, authority class"; and
"behavioural identity" is an identity construct by name.

**Proposed repair.** Restate the principle as: _no human-identity
verification, no KYC, no whitelist of participants; anti-abuse uses
prices, delays, burns, and time-earned pedigree._ Then add one
sentence naming pedigree as a **time cost**, which is what it is, and
cross-reference M335's finding that the anti-Sybil strategy is
uniformly time-cost with near-zero monetary cost. That tension is
already registered; the principle should not deny it. (M373)

---

### G10 — "Weights may be private" vs. the executor role [MEDIUM]

The Actors list defines the reference executor as "a registered party
that holds the sealed artifact." Verification therefore requires
distributing weights to sampled strangers while the privacy principle
requires not doing so. The per-tier paragraph half-concedes it, but
the FHE resolution given — "the executor pool for a weights-private
FHE contributor is the contributor's own host" — is **self-judging**,
which voids the "the judge is sampled, not chosen" guarantee.

**Proposed repair.** Say it in the Actors list, not only in a late
bullet: the reference executor exists only for artifacts whose
contributor chooses to distribute the sealed weights. For
weights-private artifacts there is **no serving-time identity check
beyond behavioural identity on a fixed probe set**, and the residual
is priced by the bond. Delete the sentence that names the
contributor's own host as its executor pool — a self-judging probe is
not a probe, and claiming it weakens the honest parts of the section.
(M374)

---

### G11 — Composition is asserted from fusion evidence [HIGH]

**Where.** Assumption 3 argues composition (chaining, $C_{\text{out}}(h)
\subseteq C_{\text{in}}(g)$) and then supports it with the M341 fusion
cell (0.548 concatenated vs 0.242 single encoder).

**The break.** Fusion (concatenating code blocks into one head) and
composition (chaining stages, each stage's output feeding the next)
are different claims with different failure modes. There is no
measured chain anywhere in §What has been measured — no end-to-end
ASR→intent number, no measured Shapley or LOO split — despite Figure 2
and the entire fee-flow chain rule resting on it. The load-bearing
assumption of the thesis is the one with no row in the table.

**Proposed repair.** Two moves, both required:

1. **Textual (immediate).** Split assumption 3 into 3a (fusion,
   measured, with the M228/M341 boundary) and 3b (chaining,
   _unmeasured_, registered as an open cell). The paper's honesty
   statement already licenses this; the current text does not perform
   it.
2. **Experimental (M375).** Run one real two-stage chain end to end on
   sealed data: Whisper ASR → frozen-BERT+ridge intent head, scored
   against (a) each stage alone, (b) the identity-substituted
   coalitions the attribution rule needs. This single cell measures
   the chain claim, the contract rule, and the attribution split at
   once.

**Gate (M375).** Register before running: the chain beats the
strongest single stage on the composed task, the contract check
refuses a mismatched pairing, and the coalition values needed by the
attribution rule are all computed and sealed. A chain that degrades
relative to its stages is published as a negative and assumption 3b is
withdrawn.

#### M375 — registered deviation from the named cell (28 Aug 2026, before any measurement)

The cell above names Whisper ASR → frozen-BERT+ridge intent. That
cell **cannot be run offline**: `F:\geode-ml\data\cache\huggingface`
holds DINOv2 (three sizes), Qwen2.5 (three variants), and two speaker
models, and its `hub` subdirectory holds DomainNet and CIFAR-10 only.
There is no Whisper checkpoint, no BERT checkpoint, and no
speech-to-text corpus. Downloading two models and a corpus to satisfy
the letter of a design would put an unsealed dependency at the centre
of the one experiment that decides assumption 3.

**Substituted cell, registered here before it is run.** The
router → specialist chain of Figure 2, on the sealed DomainNet
selection (tag `63f590097008f749`, 138,000 train / 34,500 test,
345 classes over 6 domains):

- **stage A (router)** — DINOv2-small r56 features → one of 6 domains;
- **stage B (specialist)** — features + domain → one of 345 classes.

This is not a weaker substitute. It is the composition the paper
_actually_ rests on: Figure 2 is a router feeding specialists, and the
fee-flow chain rule is written for exactly that shape. The ASR→intent
example was illustrative; the router→specialist chain is structural.

**Coalitions, with the null artifact of each stage's own contract
substituted for an absent player:**

| Coalition      | Substitution       | What it measures                                   |
| -------------- | ------------------ | -------------------------------------------------- |
| $v(\emptyset)$ | no router, no head | the most frequent class                            |
| $v(\{A\})$     | router + null head | most frequent class _within the predicted domain_  |
| $v(\{B\})$     | null router + head | one monolithic 345-way head — the sealed M144 read |
| $v(\{A,B\})$   | —                  | the chain                                          |

**Gate clauses, unchanged in substance:**

1. **Anchor first.** The monolithic head must reproduce the sealed
   M144 unpruned r56 read $0.245014$ within $0.002$ before any chain
   number is read. A failure here voids the cell.
2. $v(\{A,B\}) > \max(v(\{A\}), v(\{B\}))$ — the chain beats the
   strongest single stage.
3. The contract check admits `domain_label[6] → domain_label[6]` and
   refuses `class_label[345] → domain_label[6]`.
4. All four coalition values computed and sealed; the Shapley split
   sums to $v(\{A,B\}) - v(\emptyset)$ exactly.

**Registered diagnostic (not a coalition).** An oracle-routed arm —
stage B given the _true_ domain. It is recorded because the two ways
this cell can fail call for opposite responses: if the oracle-routed
chain also loses to the monolith, specialisation itself costs more
training data than it buys and no better router would rescue it; if
the oracle wins and the routed chain loses, the loss is router error
and assumption 3b survives conditioned on routing quality. Registering
the diagnostic in advance is what stops that distinction from being
invented after the numbers land.

**Failure is publishable.** A degrading chain is written up as a
negative and assumption 3b is withdrawn from the paper.

#### M375 — SEALED, PASS (28 Aug 2026)

Evidence: `analysis/m375_measured_chain.json`. Instrument:
`tools/m375_measured_chain.py`. Tests:
`tests/unit/test_v26_m375_measured_chain.py`.

**Gate clause 1 — anchor.** The monolithic 345-way head measured
**0.245014** against the sealed M144 unpruned r56 read
**0.245014492753623**. Exact to six places, well inside the $0.002$
tolerance. Nothing else was read until this passed.

**Gate clause 2 — the chain beats the strongest single stage.**

| Coalition      | Substitution                  | Value        |
| -------------- | ----------------------------- | ------------ |
| $v(\emptyset)$ | no router, no head            | 0.002899     |
| $v(\{A\})$     | router + null head            | 0.005768     |
| $v(\{B\})$     | null router + monolithic head | **0.245014** |
| $v(\{A,B\})$   | the chain                     | **0.274058** |

The chain wins by **2.90 points** over the strongest single stage.
Assumption 3b **survives its first test**.

**Registered diagnostic.** Router domain accuracy **0.8151**.
Oracle-routed chain **0.2995**. So specialisation is worth **5.45
points** at a perfect router, and routing error gives back **2.54**
of them. Both failure modes were live and the diagnostic
discriminates: this is a real chain win, degraded but not created by
router quality.

**Gate clause 3 — the contract check.** `domain_label[6] →
domain_label[6]` admitted; `class_label[345] → domain_label[6]`
refused with a stated reason.

**Gate clause 4 — the split is efficient.** Shapley over two players:
router **0.015957**, head **0.255203**, summing to
$v(\{A,B\}) - v(\emptyset)$ to within $10^{-12}$.

**Disclosed, not hidden.** The comparison is **artifact-matched, not
parameter-matched**: the chain carries six 345-way heads where the
monolith carries one, and each specialist is fitted on roughly a
sixth of the rows. It trades parameters for training data and comes
out ahead. Artifact-matching is the matching the fee rule cares
about — two stages are two priced artifacts whatever their size —
and the paper now states this rather than leaving a reader to find
it.

**Scope, stated in the paper.** One chain, length two, both stages
reading the same frozen features. Nothing here speaks to length
three, to crossing modalities, or to stages that must serialise
through a text interface.

**Paper edits.** Assumption 3 split into 3a (fusion, measured) and 3b
(chaining, now measured with its scope); a Chaining block added to
the measured table (0.2741 / 0.2450 / 0.2995); two new paragraphs in
§What has been measured.

---

### G12 — Two attribution rules [MEDIUM, CODE-DIVERGENT]

§The fee flow: chains divide "by the Shapley value of each stage."
§The standard library: a representation artifact "earns attribution
through the heads that consume it, by the same leave-one-out rule the
chain applies to every contribution." Shapley ≠ LOO beyond two
players, and the shipped `geode/core/representation.py` uses LOO.

Also unstated: Shapley over $m$ stages needs $2^m$ end-to-end
_measured_ coalition evaluations, so it cannot be a per-session
computation.

**Proposed repair.** Pick one rule and state its cost:

- **Recommended:** Shapley for chains (it is the fair rule, and $m$ is
  small — cap registered chain length at $m \le 4$, i.e. ≤16
  coalitions), computed **once at chain registration** on the sealed
  reference workload, sealed with the chain artifact, and replayed
  from the seal at settlement. LOO is then the $m=2$ special case and
  the representation paragraph is rewritten to say Shapley.
- State who pays for the coalition evaluations: the chain registrant,
  as part of the challenge budget.

**Gate (M376).** One attribution function in the codebase, exercised
by both the chain path and the representation path; a registered
chain-length cap; a test that the split sums to one and that a
harmful stage earns zero. (M375's cell provides the first real
coalition values.)

#### M376 — SEALED, PASS (28 Aug 2026)

M375's coalition values turned this from a notational complaint into
a measured one, so it was closed in the same pass.

**The divergence, measured.** On M375's four coalition values — one
set of numbers, no modelling choices left — the two rules divide the
pool differently:

| Rule          | Router share | Head share |
| ------------- | ------------ | ---------- |
| Shapley       | **0.0588**   | 0.9412     |
| Leave-one-out | **0.0977**   | 0.9023     |

A factor of **1.66** on the routing stage. Both rules happen to be
efficient here; they disagree about the _split_, which is exactly
the case where naming both means owing two different amounts for the
same work. G12 was rated MEDIUM on the assumption it was a drafting
inconsistency. It is a payment ambiguity.

**Repairs shipped:**

1. **One rule, named.** §The fee flow now carries
   `\label{sec:fees}` and states plainly that Shapley is the
   protocol's single attribution rule, binding for chains and for
   representation artifacts alike.
2. **The contradiction corrected in place.** §The standard library
   said leave-one-out. It now says Shapley, refers to
   `sec:fees`, and carries the measured factor of 1.66 as the
   reason the choice is not cosmetic.

   **Correction to this milestone (29 Aug 2026).** The first
   version of this edit had the paper narrate its own revision
   history — "an earlier draft of this section said leave-one-out;
   that was a drafting error." That breaks the standing docs rule
   that `WHITEPAPER_GEODE` carries the **final system only, no
   project history**. Drafting history belongs in this review, not
   in the paper. Three further passages were caught by the same
   sweep and rewritten: assumption 3b's "for most of this paper's
   life it was supported only by fusion evidence", §What has been
   measured's "the chaining row is the newest / until this cell
   nothing in this table tested it", and a self-referential aside
   about the artifact-matching disclosure being "stated here rather
   than left for a reader to find". The measured facts and the
   design rationale stayed; only the narration of how the document
   got there was removed. **The honesty discipline is about
   registering claims before measuring them, not about confessing
   inside the artifact.**

   A fourth passage was checked and **kept**: §Router's "an earlier
   form of this rule multiplied the metric by coverage" describes a
   rejected _design alternative_ with the measurement that rejected
   it, which is design rationale rather than document history.

3. **Cost stated.** The paper now says coalition values are computed
   **once per registered chain** against its sealed reference
   workload, published with the chain, and reused by every session it
   serves — not computed per session.
4. **Cap enforced in code.** `geode/core/chains.py` gains
   `MAX_CHAIN_STAGES = 4` and `ChainTooLongError`; a five-stage chain
   is refused at construction. Four stages is sixteen coalition
   evaluations, which a validator can replay.
5. **One function.** `geode/core/representation.py::attribution_share`
   now delegates to `chains.shapley_split` when measured coalition
   values are supplied, and refuses an incomplete coalition set rather
   than silently approximating. The two-number margin form survives
   only as an explicitly-labelled stand-in, with the 1.66 figure in
   its docstring and the instruction that callers settling real
   payments must pass coalition values.

**Verification.** The Shapley path reproduces M375's measured split
(router 0.015957, head 0.2552025) to seven places from the coalition
values alone — the delegation is doing the same arithmetic the
experiment did, not an approximation of it. Suite: 1028 passed.

---

### G13 — Voting weight is buyable [CRITICAL, ANALYTICAL]

**Where.** §Voting weight ("Weight accrues only through verified
work… Unbuyable, unforgeable, unforgiven") vs. §Registration
("Verified work only. **Activity and tenure credit** accrue only from
sampled, verified work").

**The gap.** The "verified work only" rule is scoped to _activity and
tenure_ — the pedigree gate. Weight itself is "the credits the voter
has earned and not yet claimed," i.e. serving revenue. The
self-payment exclusion "keys on the payout address," so it blocks a
1-cycle only.

**The attack.** A ring of three behaviourally-distinct artifacts under
one owner cycles payments: A pays B, B pays C, C pays A. Each hop
loses the 2.5% dock plus gas plus probe overhead. Each hop credits the
recipient with unclaimed balance = voting weight. Pedigree accrues
normally, because probed sessions on the ring's traffic are
"initiated by others" from each address's point of view. The $d\ge3$
diversity rule is satisfied by construction, and the 20% cap is the
only remaining bound.

**Cost of buying weight: 2.5% of the capital cycled, plus gas**
(measured at M358; this section originally estimated ~5%).

The paper's own wash-ring scenario rebuts only _reputation_, which was
never what the ring was after.

**Proposed repair.** Apply the qualifier the paper already invented
for the takedown deposit. Voting weight counts only
**externally-verified** revenue — sessions from payer addresses with
no attribution linkage to the beneficiary, under the same measure
already implemented for `verified trailing revenue`. Concretely:

- Maintain two balances per identity: `credits_total` (claimable) and
  `credits_verified` (voting weight). Cycled revenue accrues to the
  first and not the second.
- Linkage test: a payer that has ever received attribution credit from
  the beneficiary's artifact set, transitively to depth $L$ (register
  $L=3$), is not external.
- State plainly in the paper: **weight is a subset of earnings, not
  earnings.**

**Gate (M358).** Reproduce the ring in the adaptive-campaign harness
(`geode/core/adaptive_campaign.py` already runs budget-bounded
adversarial episodes): a 3-cycle acquires zero voting weight and loses
5% of cycled capital, while an honest supplier's weight is unchanged.

#### M358-pa — prior-art registration for the repair

Registered **before** the queries ran, per the standing rule that a
search whose criteria are written after the results is not a search.

**Claim under test (R-G13).** Governance weight in an open,
permissionless economic network accrues only from _externally-verified_
revenue — payments from counterparties with no attribution linkage to
the beneficiary within a registered depth $L$ — so that circular
payment among self-owned identities cannot convert capital into weight.

**Displacement criteria.**

- **D1 (full displacement).** A published or deployed mechanism that
  derives governance/voting weight from revenue and discounts that
  revenue by payer–payee graph linkage. If found, the repair is not
  ours; cite and adopt.
- **D2 (application displacement).** A published wash-trade or
  circular-payment filter applied to _voting weight_ specifically —
  not to volume statistics, price discovery, or a reputation score.
- **D3 (partial).** A published collusion discount over a contribution
  graph that bounds the gain from self-dealing. If only D3 hits, the
  repair stands and the citation becomes mandatory.
- **Nothing hits:** the repair is an assembly of known parts (linkage
  graphs + earned weight) and the paper claims the assembly only —
  the same stance it takes everywhere else. Absence proves nothing
  (unauthenticated search cannot support a novelty claim).

**Anchors.** One liveness anchor whose absence proves the instrument
is broken (`all:"Liberal Radicalism"` — Buterin/Hitzig/Weyl,
arXiv:1809.06421) and two sensitivity anchors that must surface known
work _without naming its title_ (`quadratic funding` AND `collusion`;
`wash trading` AND `detection`).

**Instrument.** `tools/prior_art_search_3.py`, inheriting the
validated run-2 rules: 429 → backoff, residual failures recorded in a
field separate from genuinely empty results, results written to JSON
and interpreted only afterwards. Evidence:
`analysis/prior_art_sweep_3_g13_2026-08-28.json`.

**Run 1: VOID.** The liveness anchor `all:"Liberal Radicalism"`
returned zero. Diagnosed mechanically before anything was read:
`id_list=1809.06421` resolves, but the paper has been **retitled** to
"A Flexible Design for Funding Public Goods", so the anchor tested a
string that no longer exists in the metadata. `all:"quadratic
funding"` returns four real papers, so phrase search works. The
instrument was exonerated; the anchor was mine. Void evidence kept at
`analysis/prior_art_sweep_3_g13_2026-08-28_void_run1_anchor_misspecified.json`.

**Run 2 corrections** (registered, then every query re-run — never
only the empty ones):

- liveness anchor → the current title, plus `all:"quadratic funding"`
  as a topic liveness anchor;
- sensitivity anchor 1 → `"public goods" AND "matching funds"`, which
  must surface Buterin/Hitzig/Weyl without naming the title;
- a **decoy anchor** carrying the retired title string, expected zero,
  so the run-1 failure mode is now instrumented;
- arXiv `all:` indexes title/abstract/authors/comments only, so a
  multi-term AND is too strict for a concept living in a paper's
  body. Every multi-term query now runs in **both** AND and OR form
  unconditionally, so widening is a property of the instrument and
  never a reaction to a result.

Run-2 anchor readings: liveness 1 hit, topic liveness 4 hits,
sensitivity 1 → 4 hits (surfaces the BHW paper without its title),
sensitivity 2 → 15 hits, decoy 0. **Instrument valid.** 252 unique
papers.

**Verdict.**

- **D1 — not found.** No mechanism derives governance weight from
  revenue and discounts that revenue by payer–payee graph linkage.
- **D2 — not found.** The wash-trade literature that exists (_NFT
  Wash Trading Detection_ 2305.01543; _Abnormal Trading Detection in
  the NFT Market_ 2306.04643) is forensic — it detects manipulation
  in market statistics. None of it feeds a voting-weight base.
- **D3 — HIT. MeritRank** (Nasrulin et al., arXiv:2207.09950),
  "Sybil Tolerant Reputation for Merit-based Tokenomics". It does the
  same job by a different route: rather than preventing Sybil rings
  it _limits the benefit_ of one, by decaying the perceived value of
  the attacker's contributions (transitivity, connectivity and epoch
  decay) over a feedback graph, validated on MakerDAO interaction
  data. **Citation is now mandatory**, and the distinction is real:
  MeritRank grades a reputation score; the repair here is a binary
  externality predicate on the attribution graph, applied to a weight
  base that is separate from the claimable base. Graded decay is
  registered as the follow-on (see the residual below).

**Unexpected and more important than the search question.** _Concave
is the New Linear: The Impossibility of Anti-Plutocratic DAO
Governance_ (arXiv:2605.18990) proves that **no voting rule deriving
power solely from wallet balance can resist Sybil splitting on a
permissionless chain** — for any positive, increasing, finite concave
rule the optimal splitting strategy yields power asymptotically
linear in holdings. Measured Sybil amplification on five major DAOs:
1,172×–4,039× under quadratic voting, >229,000× under steeper rules.

GEODE's 20% cap _is_ a concave rule. The paper escapes the theorem
only because weight is supposed to come from earned work rather than
from balance — which is exactly the property G13 shows is false
today. **So the theorem converts G13 from "an attack costs 5%" into
"the cap is covered by a published impossibility result unless the
externality qualifier is added."** The repair is not a hardening; it
is what keeps the paper outside a proved impossibility. Cite it in
§Voting weight and in Known Limits.

Other neighbours to cite, none displacing: Poupko/Shahaf/Shapiro,
_Building a Sybil-Resilient Digital Community_ (arXiv:1901.00752) —
trust-graph conductance, the nearest relative of the $d\ge3$
diversity floor; Buterin/Hitzig/Weyl (arXiv:1809.06421) for the
collusion-discount lineage on the funding side.

**Conclusion.** The repair survives, with three mandatory citations
(MeritRank, Concave-is-the-New-Linear, the wash-trading detection
line) and two recommended ones. Per the standing rule, absence from
an unauthenticated index proves nothing and no novelty is claimed —
the paper claims the assembly, as it does everywhere else.

#### M358 — SEALED (gate PASS)

Shipped: `geode/core/voting_weight.py` (+14 tests in
`tests/unit/test_v26_m358_voting_weight.py`; 996/996 unit tests
green). Evidence: `analysis/m358_voting_weight_evidence.json`.

| Arm                                    | Reading                              |
| -------------------------------------- | ------------------------------------ |
| 3-cycle, voting weight **pre**-repair  | 292.5 (the shipped rule's base)      |
| 3-cycle, voting weight **post**-repair | **0.0**                              |
| 3-cycle, capital lost                  | 7.5 on 300 cycled = **2.5% haircut** |
| Honest supplier, same volume           | weight 292.5 = claimable — untouched |
| Gate                                   | **PASS** for ring sizes 2, 3, 4      |

**Correction to this review's own text.** The finding above estimated
the ring's cost at "~5%". The measured protocol cost is the **2.5%
dev-fund dock**, plus gas and probe overhead, which are
deployment-dependent and not in the model. The lower figure makes the
attack _cheaper_ than stated and the finding stronger, so it is
corrected here rather than quietly left standing.

**Two residuals, both measured and registered rather than hidden:**

1. **A ring longer than the linkage depth keeps its weight.** At the
   registered $L = 3$, a 5-cycle retains full weight (measured 97.5
   per member). Each extra hop costs another dev-fund dock and
   another behaviourally distinct artifact that must independently
   pass admission, so the attack does not vanish — it gets priced.
   Raising $L$ is a parameter change with a false-positive cost;
   MeritRank-style graded decay is the registered alternative.
2. **Mutual trade between two genuine businesses loses weight.**
   Measured: an identity trading in both directions with one
   counterparty keeps the money and loses 97.5 of weight on that
   revenue. This is the price of a binary predicate and belongs in
   Known Limits, stated plainly: **weight is a subset of earnings,
   and buying from your own customers costs you weight.**

---

### G14 — The ladder prices governance participation [MEDIUM]

Level 3 burns "vested-but-unclaimed credits." Voting weight _is_
unclaimed credits. So holding weight strictly increases slashable
exposure and the rational actor claims every epoch and abstains from
governance. The paper argues at-risk capital aligns incentives; it
also, unintentionally, taxes participation.

**Proposed repair.** Decouple the burn base from the weight base. A
registered **voting escrow**: a participant may _lock_ claimable
credits into weight-bearing escrow for a fixed term (proposal: 8
epochs). Escrowed credits carry weight and are burnable at L3;
unescrowed vested credits are claimable and carry no weight. The
choice becomes explicit and priced rather than a hidden penalty, and
the "at-risk capital" argument gets stronger, not weaker, because
escrow is voluntary and term-bounded. (M359)

#### §M359 — SEALED, PASS (29 Aug 2026). Voting escrow ships; weight is now a choice.

`geode/core/voting_escrow.py`, 7 tests. `VotingEscrow` separates the
burn base from the weight base exactly as G14's repair names it:

| Gate clause                               | Reading                                                                                             |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Escrow is voluntary                       | only what a participant locks carries weight; the rest stays claimable                              |
| Term-bounded                              | weight at exactly `ESCROW_TERM_EPOCHS` (8) after locking is zero; matured slots unlock to claimable |
| Burnable at L3                            | `burn()` consumes escrowed credits (newest-matured first), never the claimable balance              |
| Unescrowed vested credits carry no weight | `weight()` reads only escrow slots; a claim-every-epoch participant has zero weight by construction |

The reading G14 named is now explicit: a participant who claims every
epoch simply has no weight, and a participant who wants weight accepts
the 8-epoch term and the L3 exposure. The at-risk-capital argument
gets stronger, because the exposure is chosen, not hidden.

---

### G18 — The zakat end state defunds the public goods [HIGH]

### G15 — Probing-precision arithmetic [LOW, ARITHMETIC]

"Scores are reported to bounded precision: four significant digits.
Single-row influence is about one part in the split size. It sits
below that resolution." Four significant digits on a score near 0.85
resolves $10^{-4}$; one row in a 5,000-row split is $2\times10^{-4}$ —
_above_ the resolution.

**Proposed repair.** State it as a constraint, not a property: _the
reporting precision is set so that single-row influence sits below
resolution; this requires a split of at least $10^4$ rows, which is a
registered minimum per axis._ Then register the minimum. (M377)

#### §M377 — SEALED, PASS (28 Aug 2026)

**Two corrections to the review's own proposed repair.**

1. **The proposed $10^4$ minimum is too small by a factor of two.**
   Single-row influence is $1/n$; sitting _below_ a quantum $q$ means
   $1/n \leq q/2$. At $q = 10^{-4}$ that is $n \geq 2\times10^{4}$.
   Registered minimum is $2\times10^{4}$, not $10^{4}$.
2. **"Four significant digits" cannot be a fixed constraint at all.**
   Significant digits scale the quantum with the score: 4 s.f.
   resolves $10^{-4}$ at $s \in [0.1, 1)$ but $10^{-5}$ at
   $s \in [0.01, 0.1)$, demanding ten times the rows exactly where
   scores are weakest. The paper now reports to **four decimal
   places** — a fixed $10^{-4}$ quantum — which is what makes a single
   registered minimum well-defined.

**Consequence the repair forces into the open.** Of the axes this
paper measures, only Open Images clears $2\times10^{4}$:

| Axis                    | Held-out rows | Clears $2\times10^{4}$? |
| ----------------------- | ------------- | ----------------------- |
| SST-2 dev               | 872           | no                      |
| MNLI-matched dev        | 9,815         | no                      |
| Speech Commands v2 test | 11,005        | no                      |
| Open Images test        | 245,723       | **yes**                 |

On three of four axes single-row influence is at or above the reported
resolution, so the probing defense there rests on the registration fee
alone. The paper now says so rather than reporting fewer digits, which
would hide the number without reducing the influence. Enforcement is
at axis registration: an axis below the minimum does not get the
guarantee.

---

## 4. Cost structure

### G16 — The lottery has no per-session entropy [HIGH, CODE-DIVERGENT]

"The draw is seeded by the hash of the anchor, the task, the registry
state, and the task fingerprint." Every field is constant within an
epoch for a given task, so **every session of that task in that epoch
routes to the same arm**. The weighted lottery degenerates to a
per-epoch winner-take-all, and "the strongest arm held about one third
of traffic" is unachievable as specified.

**Proposed repair.** Add the session identifier to the seed:
$H(\text{anchor}, \text{task}, \text{state root}, \text{fp},
\text{session id})$. The session id is assigned at declaration and is
in the ledger, so the route still replays exactly; it is not
choosable by the host, so the anti-grinding property survives.
`geode/core/router_repair.py` must be updated with the paper.

**Gate (M354).** The existing synthetic traffic-share sweep is re-run
with distinct session ids and reproduces the published shares
(strongest arm ≈ one third, equal-price tie splits evenly, a 2× price
cut roughly doubles share). The current sweep almost certainly varied
the task or the anchor to get those numbers — confirm which, and say
so if the published figure came from a configuration the protocol
does not actually produce.

#### §M354 — SEALED, PASS. G16 confirmed; the published figure came from a configuration the protocol does not produce

Instrument: [tools/m354_route_lottery_entropy.py](tools/m354_route_lottery_entropy.py)
(run as `python -m tools.m354_route_lottery_entropy`).
Evidence: [analysis/m354_route_lottery_entropy.json](analysis/m354_route_lottery_entropy.json).
Code: [geode/core/router_repair.py](geode/core/router_repair.py).
Tests: [tests/unit/test_v26_m354_session_entropy.py](tests/unit/test_v26_m354_session_entropy.py),
5 passing; the 12 existing M303 tests still pass.

**Which field the sweep varied: the anchor.**
[experiments/tier4/eval_v26_m303_router_repair.py](experiments/tier4/eval_v26_m303_router_repair.py)
`_share()` calls `router.route(FP, anchor=f"anchor-{i}")` — a fresh
anchor for every session. The whitepaper anchors the ledger tip
"to Ethereum once per epoch". **A per-session anchor is not a cadence
the protocol produces**, so the published traffic-share figures were
generated by a configuration the protocol does not reach. Disclosed
here as the gate requires.

**Failure reproduced.** Holding one epoch anchor fixed across 4,000
sessions, as the protocol actually does:

| Scenario                  | Published sweep (anchor per session) | Protocol (anchor per epoch) | Repaired (session id in seed) |
| ------------------------- | ------------------------------------ | --------------------------- | ----------------------------- |
| Leader 0.70 / 0.69 / 0.60 | 0.352 / 0.333 / 0.315                | **leader 1.000**            | 0.355 / 0.337 / 0.308         |
| Equal-price tie           | 0.492 / 0.507                        | **tie_b 1.000**             | 0.490 / 0.510                 |
| 2× price cut              | 0.332 / 0.668                        | **half_price 1.000**        | 0.339 / 0.661                 |

All three published claims fail under the protocol's own cadence and
all three hold after the repair: strongest arm 0.3548, tie 0.490,
price-cut ratio 1.952.

**Grinding, measured rather than asserted.** The repair introduces a
field the payer controls, so the anti-grinding property had to be
re-checked rather than inherited.

- **A host cannot grind.** The seed carries no arm identifier, and the
  only registry lever a host owns — its own price and score — moves
  the _ranking_, which is the intended channel.
- **A payer can.** It owns its session identifiers and the epoch
  anchor is public, so it can resubmit until a preferred arm wins:
  **3.1 declarations on average** to force the least-favoured of three
  arms (share 0.308). Recorded in Known limits. The residual is small
  — each attempt costs a declaration fee and a payer can already
  express arm preference openly through best-quality mode — but it is
  a real hole, not a closed one, and it is closed properly by seeding
  the draw from the randomness beacon rather than the anchor.

**Internal inconsistency this exposed, registered not repaired.**
§The ledger and the anchor states that "every sample in the protocol
— probe flags, validator sampling, reference-executor sampling —
derives from that beacon, composed with the anchor for ordering."
The route lottery is a sample and does not. Either the route draw
moves to the beacon or that sentence needs the exception written into
it. **Registered as G52.**

---

### G17 — Roles that work and are not paid [HIGH]

"Who earns what" lists contributor, primitive host, developer,
validator, reference executor, development fund. Missing, all
load-bearing:

| Role                       | Work performed                                                                                            | Current funding                             |
| -------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| Librarian                  | appends every ledger entry, pays L1 anchor gas each epoch, pays the Arbitrum batch gas                    | none                                        |
| Batch verifier             | described as "a registered, ledger-measured role" that is paid; absent from the Actors list and the table | undefined                                   |
| Sealed scoring environment | custody, sharding, rotation                                                                               | none                                        |
| Sealed replay environment  | every dispute and appeal                                                                                  | loser pays _replay cost_, not standing cost |
| Tally committee            | threshold-opens every vote                                                                                | none                                        |
| Gateway / frontend         | serves every user; "the frontend is the platform"                                                         | none                                        |
| Pinning                    | "incentivized pinning contracts"                                                                          | dev fund (see G18)                          |
| Proof generation           | G2                                                                                                        | none                                        |

**Proposed repair.** A **protocol operations line** carved from the
attribution split before the 2.5%/97.5% division, sized by measured
cost and paid out per measured unit of work (anchors posted, batches
settled, disputes replayed, votes tallied). Proposal: the split
becomes 2.5% development fund / $x$% operations / remainder
attribution, with $x$ registered from the measured cost trace and
capped. Every row above gets an income line and appears in the Actors
list. Add the librarian's gas to the cost model in
`analysis/v25_m209_cost_model.md`.

**Gate (M363).** A cost-model run that closes: total operations
revenue at the reference workload ≥ total measured operations cost,
with each role's line itemized. If it does not close, the split is
wrong and the paper says so rather than omitting the roles.

#### §M363 — SEALED (29 Aug 2026). The line closes at a sized fee; the harness fee does not.

Instrument `tools/m363_operations_cost_model.py`, evidence
`analysis/m363_operations_cost_model.json`. Reference workload: the
vision axis at 10,000 sessions/epoch, 8 inbox entries (the
incorporation cap — the librarian's maximum obligation), one
attribution root, one dispute replay, and the M350 sampled-batch
verification rate (100 proofs). Registered prices: 0.1 gwei gas,
$3,000/ETH, $0.001 per CPU-core-second.

| Role                     | Measured unit cost                                                                    | $/epoch  |
| ------------------------ | ------------------------------------------------------------------------------------- | -------- |
| Librarian (on-chain gas) | 8 × incorporate (163,187) + root (60,057) + anchor, measured on the current contracts | 0.44     |
| Batch verifier           | 100 proofs × 7.721 s verify (M350 vision axis)                                        | 0.77     |
| Reference executor       | 1 dispute replay                                                                      | 0.01     |
| Tally committee          | 1 vote opening                                                                        | 0.00     |
| Gateway / frontend       | 10,000 sessions served                                                                | 0.01     |
| **Total**                |                                                                                       | **1.23** |

**The gate's second clause is the finding.** The line does NOT close
at the inbox harness fee (10 wei): revenue is $0 against $1.23/epoch
of measured cost. The harness value was never the deployment fee, and
the model sizes the deployment fee instead of pretending: **a base
posting fee of ≈ 6.4×10¹³ wei (~$0.19 per entry at the registered
prices) closes the line at the registered 1.25× margin.** Every role
in G17's table now has an income line, and the librarian's gas is in
the model.

The honest statement for the paper: the operations line is sized by
measured cost at a registered reference workload, the fee that closes
it is a registered floor, and an axis whose entry volume falls short
of the workload does not subsidise the line — the shortfall is public
(the accrual is on-chain), not silently absorbed.

---

### G18 — The zakat end state defunds the public goods [HIGH]

The 2.5% currently pays for "audits, tooling, security monitoring, and
public-good research," maintains and re-certifies the standard
library, and funds pinning contracts. At maturity the fund "converts
permanently into a zakat rule" with "a frozen recipient list with
fixed fractions… and no mutator." The infrastructure bills do not stop
at maturity.

**Proposed repair.** The charter reserves, _before_ the zakat split, a
fixed **maintenance fraction** for exactly the enumerated public goods
(standard library maintenance and dependency re-certification,
security audits, pinning). Proposal: 0.5 of the 2.5% at conversion,
itself charter-fixed and non-redirectable, with the remaining 2.0%
going to zakat. The zakat rule stays mechanical and unpausable; it
simply applies to the post-maintenance stream. Alternatively, fold
maintenance into the operations line of G17 and let the full 2.5%
convert — cleaner, and it makes maintenance a measured, paid role
rather than a grant.

**Recommendation: the second option.** It removes a discretionary
budget entirely and matches the paper's own "measured, not asserted"
principle. (M362)

#### §M362 — SEALED, PASS (29 Aug 2026). Maintenance is a paid role; the full 2.5% converts.

The recommendation is adopted, and it is now consistent with M363's
measured cost model. The whitepaper's zakat section states: standard
library upkeep, dependency re-certification, security audits, and
pinning are **paid roles on the operations line**, measured and funded
per unit of work like every other operations role; the fund therefore
has **no discretionary maintenance budget**, and at maturity the full
2.5% + registration fees convert to zakat with nothing carved out.
G17's original "maintenance / pinning / dev fund (see G18)" rows move
to the operations line, which M363 sized to close at the reference
workload.

The gate ("charter has no discretionary maintenance budget; public
goods are paid roles") is met by construction: the only discretionary
fund is the bootstrap fund, and its end state is a frozen zakat list
with no mutator.

---

### G19 — Abstention priced below cost [MEDIUM]

"An abstention consumed the trunk compute: it is recorded and metered
at the registered reduced rate — half the unit price." If the compute
was fully consumed, honest cost recovery is 1.0, not 0.5. Every
abstention is a guaranteed loss, which pressures suppliers toward
guessing — the opposite of the selective-classification stance the
paper takes.

**Proposed repair.** Meter the abstention at the **registered compute
fraction actually consumed**, measured per axis, not at a flat 0.5.
For a single-pass classification head that is ~1.0; for a cascade that
abstains before the expensive stage it is genuinely lower. The
half-price figure was chosen to make boundary-probing pay (M332); a
full-cost figure makes it pay _more_, so the extraction argument
strengthens. State the per-axis fraction in the price table. (M357)

#### §M357 — SEALED, PASS (29 Aug 2026). Full cost for single-pass heads; the bound doubles.

Instrument `tools/m357_abstention_fraction.py`, evidence
`analysis/m357_abstention_fraction.json`. The flat 0.5 is gone.

- **Measured per-axis fraction:** for a single-pass classification
  head the abstention predicate is a function of the final margin, so
  the trunk encode and the head both ran before the abstention was
  known — consumed fraction **1.0**, registered
  (`ABSTENTION_COMPUTE_FRACTIONS = {"single_pass_head": 1.0}`).
  A cascade would register its own lower value at axis creation; none
  exists in the shipped family.
- **The M332 bound, re-derived, is strengthened, not weakened.** M332
  priced every adversarial query at the abstention rate (the
  adversary's cheapest rate). At 0.5 that measured 55.2× lifetime
  revenue; at the measured 1.0 the bucketed extraction cost doubles to
  **110.4×** (raw-margin oracle 2.8× → 5.5×). The supplier is no
  longer forced to guess to recover cost: an abstention now pays its
  own compute.

`abstention_charge` accepts the measured fraction and now allows the
full unit price (`(0, 1]`). All three whitepaper abstention sentences
("half the unit price") replaced with the measured per-axis fraction.
Python suite **1095 passed** (+3).

**The lesson.** _A "reduced rate" chosen to make an attack pay can
itself be a cost-recovery defect, and the two fixes compound._ The
half-price was a defense number; honest costing was a supply number.
Raising the metered rate to the measured fraction both removes the
guessing pressure and doubles the extraction defense — the same knob,
both directions.

---

### G20 — Coverage multiplication inverts quality [HIGH]

$s_a = \text{metric}\times\text{coverage}$ ranks the scoped, honest,
0.901-accurate vision arm at 0.044 and the 0.164-accuracy
full-coverage guesser at 0.164. The paper notices the inversion and
calls it "the point of the metric." The consequences it does not
follow through:

- **Best-quality mode returns the worse arm.** $\argmax_a s_a$ picks
  the guesser.
- Abstention is economically dominated at the routing layer as well as
  the metering layer (G19), on top of a paper that elsewhere prizes
  abstention.
- $\bar u_a$'s treatment of abstentions is undefined. If abstentions
  do not count as units, a 5%-coverage arm has a small $\bar u_a$ and
  therefore a _higher_ $v_a$ — the opposite of the intended effect.

**Proposed repair.** Condition the score on the declared label set
instead of multiplying by coverage:

- The task descriptor already carries the class list in the
  fingerprint. Let a user declare the label set it cares about.
- $s_a$ is then the metric **restricted to the declared set**, times
  coverage **within that set**. The scoped arm reads ≈0.901 on its
  129 classes and is correctly unqualified for the other 472.
- An arm that does not cover the declared set is not in $Q(t)$ at all
  — a qualification question, not a ranking penalty.
- Define abstentions as metered units, so $\bar u_a$ counts them.

This keeps the property the coverage factor was defending (an arm
cannot win by refusing everything) while removing the inversion.

**Gate (M355).** On the sealed Open Images evidence, show the scoped
arm outranks the full-coverage arm for a buyer declaring the 129
served classes, and is _not_ qualified for a buyer declaring all 601.
Both directions required.

#### §M355 — SEALED, PASS (28 Aug 2026)

Instrument: [tools/m355_declared_label_set.py](tools/m355_declared_label_set.py).
Evidence: [analysis/m355_declared_label_set.json](analysis/m355_declared_label_set.json).

**The sealed M286 evidence file did not survive the `logs/` squash.**
The head was refit from the same cached DINOv2-L features
(`oid_train_137149_feat.npy` / `oid_test_245723_feat.npy`, 1024-dim)
at the M262 standard (α=1.0, floor 0.8, min 10 rows), on CPU, because
this reproduces a sealed CPU contract path. A refit object is not the
sealed object, so **clause one of the gate is an anchor reproduction**
against the four published values. All four reproduce:

| Anchor                         | Registered | Measured   |
| ------------------------------ | ---------- | ---------- |
| Overall top-1, all 601 classes | 0.1643     | **0.1643** |
| Served-subset top-1            | 0.901      | **0.901**  |
| Row coverage of the served set | 0.049      | **0.049**  |
| Served classes                 | 129        | **129**    |

Only then were the two directions read:

| Declared set $D$               | Scoped arm                         | Full-coverage arm         | Result          |
| ------------------------------ | ---------------------------------- | ------------------------- | --------------- |
| The 129 served classes         | qualified, $s_a$ = **0.9378**      | qualified, $s_a$ = 0.9010 | scoped outranks |
| All 601 classes                | **not qualified** (covers 129/601) | qualified, $s_a$ = 0.1643 | scoped excluded |
| _(retired rule, for contrast)_ | 0.901 × 0.049 = **0.0441**         | 0.164 × 1.0 = **0.1643**  | **inverted**    |

**Both directions hold.** The margin in direction one is not a
definitional artifact: the scoped arm argmaxes over its 129 served
classes while the full-coverage arm keeps all 601 in play, so removing
472 distractors buys a real +3.68 points (0.9010 → 0.9378) on
identical rows. Had the scoped arm been defined as "the full arm
restricted to rows in $D$" the two would have been numerically
identical at 0.9010 and the gate would have been unfalsifiable — the
restriction has to be in the _decision_, not just the _evaluation_.

Paper edits: `\subsection{The registry and the router}` now carries
`\label{sec:router}` and defines $s_a$ once, conditioned on $D$, with
the retired multiplication and its measured inversion stated as the
reason. The black-box axis definition and the vision-axis results
paragraph were rewritten to match; abstentions are stated as metered
units so they count in $\bar u_a$ (the G20 sub-finding).

---

### G21 — ETH-denominated timelocked prices [MEDIUM]

Prices are set in ETH, changeable only at epoch boundaries after a
timelock, with "no external price feed" in the settlement path, and
earnings vest over four more epochs with a further claim freeze under
open probe exposure. A supplier's costs are fiat-denominated and
fixed; its revenue swings with ETH over a 5+ week horizon it cannot
hedge inside the protocol. Known Limits #9 covers predatory price
cycles, not currency risk.

**Proposed repair.** Do not add an oracle to the settlement path —
that is the right call and should stay. Instead:

- Allow a contributor to register price as a **fixed ETH amount** (as
  today) _or_ as a fixed amount with a registered per-epoch drift band
  (proposal: ±10% per epoch, still timelocked in _rule_ but not in
  _value_), so a supplier can track a large move without a governance
  action per epoch. The router replays against the price table of the
  day either way.
- Add the residual to Known Limits explicitly: suppliers hold ETH
  price risk across the vesting window, and the protocol does not
  hedge it.

(M360)

#### §M360 — SEALED, PASS (29 Aug 2026). Drift band ships; the residual is on the page.

`geode/core/economics.py` gains the drift-band rule: a contributor
may register price fixed (band 0 — today's rule) or fixed with a
registered ±10% per-epoch band (`PRICE_DRIFT_BAND_BPS = 1000`).
`price_within_drift` accepts or rejects a per-epoch declaration;
`clamp_to_drift` caps a move; `effective_price_path` is the
DETERMINISTIC price table of the day the router replays — never a
live feed. 6 tests, including the replay determinism gate.

The currency-risk residual is now a Known Limits entry (suppliers
hold ETH risk across the vesting window; the protocol does not hedge
it; the drift band softens but does not remove it). The settlement
path stays oracle-free, as G21 insisted.

---

### G22 — One centralised economic lever [HIGH]

The "registered reference hosting cost" simultaneously sets (a) the
per-axis price floor, (b) the developer's own bootstrap price, and (c)
every competitor's bond, via "the axis's posted price minus the
developer's posted reference hosting cost." A developer that raises it
raises the floor and every rival's bond at once. That is one
centralised lever held by the party the rest of the design carefully
disempowers, and it is not in Known Limits.

**Proposed repair.** Make the reference cost a **measured, multi-party
statistic**: the median of the posted hosting costs of all admitted
arms on the axis with at least one epoch of verified traffic, floored
at the developer's figure only while the axis has fewer than three
such arms. Publish the transition rule. Until the transition, add a
Known Limits entry naming the lever and its three effects. (M361)

#### §M361 — SEALED, PASS (29 Aug 2026). The lever is a median; the fallback is published.

`reference_hosting_cost(admitted_costs, developer_cost)` in
`geode/core/economics.py` implements the repair and returns WHICH
rule is operative (`median of admitted` / `developer floor
(<min_arms)`), so the registry can publish the instrument a buyer is
depending on. 6 tests: below 3 verified arms the developer figure is
the floor; at 3+ it is the median; an outlier cannot move a 5-arm
median; even-n medians take the midpoint; `REFERENCE_COST_MIN_ARMS =
3` is registered.

The three effects G22 named (floor, bootstrap price, every rival's
bond) are no longer one party's single lever once the axis has three
verified arms, and the transition rule is published. A Known Limits
entry states that below three arms the developer figure IS the floor
and the lever is live, and says so on the page.

---

## 5. Byzantine weaknesses (quorum takeover excluded by request)

### G23 — Targeted DoS converts into slashing [CRITICAL]

"An unopened commit on a probed session is adjudicated as a deviation,
not as downtime." The host commits before learning the probe flag, so
it cannot dodge — but it also cannot _crash_. An attacker who DoSes a
rival host immediately after its answer commits turns ~5% of dropped
sessions into Level-1 burns, for the cost of bandwidth. Level 0
explicitly says downtime carries no slash; the two rules collide, and
the collision is remotely triggerable by a third party.

**Proposed repair.** The probe-dodging argument needs
commit-and-abort to cost _at least_ as much as commit-and-mismatch —
in **expectation**, which is not the same as making every abort a
burn. Replace the flat rule with an availability budget:

- Each host holds a per-epoch abort allowance $A$ (registered,
  proposal: 1% of committed sessions).
- Aborts within $A$ are charged the full unit price of the aborted
  session (the host pays, the user is refunded) — an economic penalty
  ≥ the value of dodging, with no burn.
- Aborts beyond $A$ escalate to the Level-1 path.
- The allowance is per-epoch and non-rolling, so a sustained DoS
  eventually reaches the escalation — but a sustained DoS is also
  visible as a liveness statistic and re-routes traffic anyway.

This preserves "the only profitable behaviour is serving the artifact
every time" while removing the third-party trigger.

**Gate (M364).** In the adversarial harness: a DoS campaign against an
honest host produces zero burns within the allowance and costs the
victim only the refunded sessions; a host that selectively aborts to
dodge probes finds abort no cheaper than mismatch in expectation.

#### Registered deviation from the proposed repair (29 Aug 2026, before building)

The proposed repair does not close G23. It **raises the price** of the
attack instead of removing it. An attacker with enough bandwidth
pushes an honest host past $A$ and the burns resume; worse, the
allowance is the attacker's own targeting parameter, since it can
simply drop $A+1$ sessions. A budget converts a cheap attack into an
expensive one. It does not remove the third-party trigger, and G23 was
raised as a _third-party trigger_, not as a pricing complaint.

The second clause is also not achievable as written. Per session, a
dodging host compares one unit price $p$ against an L1 burn of
unvested earnings $B$, and $p \ll B$. "Abort no cheaper than mismatch"
is false for any allowance-based rule, and a gate that asserts it
would be asserting something arithmetic forbids.

**What actually separates the two cases is a statistic, not a budget.**
The host learns the probe flag only after it commits. The attacker
never learns it at all. So:

- A host aborting because it is under attack aborts on probed and
  unprobed sessions **at the same rate** — the attacker cannot aim at
  the probed ones.
- A host aborting to dodge inspection aborts **only** on probed
  sessions, because those are the only ones worth dodging.

Among $a$ aborts in an epoch, the number falling on probed sessions is
$\mathrm{Binomial}(a, \rho)$ under the DoS hypothesis and is $a$ under
the dodging hypothesis. That is a one-sided binomial test with a
registered level, and it is the discriminating statistic G23 needs.

**Registered replacement rule.**

1. Every abort is charged the full unit price of the aborted session
   from the first one — user refunded, host pays. This is unchanged
   from the proposal and is what keeps a dodge from ever being free.
2. An abort escalates to Level 1 **only if the epoch's aborts are
   selective**: the probed share of them exceeds chance at
   $\alpha = 10^{-3}$, one-sided. Otherwise aborts stay Level 0
   however many there are.
3. The allowance $A$ survives as a small-sample floor only: no
   escalation while $a \le A$, so a host with too few aborts to test
   is never escalated on noise.

**Registered gate (M364), replacing the one above.** A DoS campaign
against an honest host produces **zero burns at any campaign size**,
not merely within a budget, and costs the victim exactly the refunded
and re-charged sessions. A host that dodges only probed sessions is
escalated, and the number of dodges it gets before escalation is
computed and published rather than assumed. The false-escalation rate
of an honest host under attack is measured against $\alpha$ over many
epochs. The claim "the only profitable behaviour is serving the
artifact every time" is checked as an expected-profit sweep over cheat
rates, not asserted.

#### M364 — SEALED, PASS with a disclosed residual (29 Aug 2026)

`geode/core/probe_adjudication.py`; 28 gates in
`tests/unit/test_v26_m319_probe_adjudication.py`. Python suite
**1090 passed, 1 skipped** (was 1071).

| Gate                                           | Reading                                                                                                                       |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| The defect reproduces first                    | the old `if probed: L1` branch, applied to one 1,000-abort campaign, burns **50**; M364 on the identical campaign burns **0** |
| DoS produces zero burns **at any size**        | campaigns of 10 / 100 / 1,000 / 5,000 / 9,000 aborts all stay Level 0                                                         |
| The victim pays only its sessions              | 300 aborts at unit price 7 → charge 2,100, refunds 300, no burn                                                               |
| A dodger is escalated                          | 30 probed-only aborts → $p = 0.05^{30}$, Level 1                                                                              |
| Dodges before escalation, computed not assumed | **3**                                                                                                                         |
| False-escalation rate over 2,000 epochs        | $\le 10\alpha$                                                                                                                |
| Camouflage                                     | dominated — best cheat rate is **0**                                                                                          |
| Open mismatch                                  | dominated — best cheat rate is **0**                                                                                          |

**Registered deviation from the deviation.** The small-sample floor was
built as $A = 1\%$ of committed sessions, as registered above. The
profit sweep then measured that floor as an economic hole and it was
replaced by a **constant** of 3 aborts. A floor proportional to traffic
scales the dodge budget with the host's own volume, so the largest
hosts — the ones worth attacking — get the largest hiding place. Sweep
at the registered cell ($N = 10^4$, $p = 1$, $h = 0.6$, $\varepsilon =
0.1$, $\rho = 0.05$, burn $= 4\times$ epoch profit): the proportional
floor is worth **more than ten times** the constant floor to a dodger.
The gate `test_a_proportional_floor_would_be_far_worse` holds both
numbers side by side.

**The residual, disclosed rather than closed.** The dodge strategy is
still worth **+0.73%** of an epoch's profit at cheat rate 0.0065 — 3
aborts concealing 65 cheated sessions in 10,000. It is not removable by
pricing: `test_scaling_the_burn_does_not_remove_the_residual` raises the
burn a thousandfold and the residual survives, because a host under the
floor is never burned at all. It is the price of using a statistical
test, and any test has one.

Two ways to shrink it were considered and rejected here. Charging an
abort $1/\rho$ unit prices — the expected number of sessions it
conceals — removes the residual arithmetically and **re-creates G23 in
economic form**: an honest host under attack would then pay twenty
times the session value for every session a third party knocked out.
Trading a burn trigger for a 20$\times$ damage multiplier is not a
repair. Lowering $\alpha$ narrows the floor but raises the
false-escalation rate on honest hosts, which is the same trade with a
different label.

**Correction to this review's own gate.** The gate above asked that a
dodging host "finds abort no cheaper than mismatch in expectation".
That is unachievable and should not have been written: per session a
dodge costs one unit price $p$ and a mismatch costs an L1 burn $B$,
with $p \ll B$ by construction. A gate asserting $p \ge B$ asserts
something arithmetic forbids. The achievable claim — that no cheat
rate under any strategy beats honest serving by more than a bounded,
measured margin — is what was gated instead.

---

### G24 — The force-inclusion inbox is a free bloat weapon [HIGH, CODE-CONFIRMED]

`InclusionInbox.sol`: "posting locks a deposit, incorporation returns
it." The deposit is fully refunded on the honest path, so spam costs
only gas (~160k per post), and the librarian must incorporate every
entry within one epoch or the chain becomes invalid. An attacker buys
chain-invalidity, or unbounded ledger bloat, for gas alone.

**Proposed repair.**

- Split the deposit into a refundable bond and a **non-refundable
  posting fee** paid to the operations line (G17) — the fee covers the
  librarian's incorporation cost, which is the actual externality.
- Add a per-address rate limit per epoch, escalating the fee
  superlinearly past it.
- Cap the per-epoch incorporation obligation at a registered number;
  entries beyond the cap roll to the next epoch **in posting order**
  without invalidating the chain, so the censorship guarantee survives
  (a censored entry is still guaranteed inclusion, just not instantly)
  while the bloat weapon does not.

**Gate (M365).** Hardhat tests: honest posting costs fee + returned
bond; a 1,000-entry spam campaign costs superlinearly and never
reaches `chainValid == false`; a single censored entry from a fresh
address is incorporated within the registered bound.

#### M365 — SEALED, PASS (29 Aug 2026)

Shipped in `infrastructure/evm/contracts/InclusionInbox.sol`, mirrored
in `geode/core/librarian_containment.py`, gated by 9 new tests in
`infrastructure/evm/test/inclusion_inbox.test.js` and 7 in
`tests/unit/test_v26_m312_librarian_containment.py`. EVM suite
**75 passing** (was 66); Python suite **1071 passed, 1 skipped**
(was 1059).

| Gate                                               | Reading                                                                            |
| -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Honest post costs the fee, returns the bond        | net cost to poster = `BASE_FEE` exactly, verified against balance minus gas        |
| Fee reaches the operations line, only it can claim | accrual and pull both exact; a stranger's claim reverts `NotOperationsLine`        |
| Fee escalates superlinearly                        | flat inside the free allowance; strictly positive **second** difference outside it |
| 1,000-entry campaign cost                          | **3,308,394,980 vs 10,000 flat — a factor of 330,839**                             |
| Spam never reaches `chainValid == false`           | 24-entry campaign, librarian at exactly the capped rate: valid throughout          |
| Censored entry from a fresh address                | deadline fixed at posting, equal to the analytic bound, and met                    |
| FIFO enforced                                      | librarian cannot incorporate a friend's entry ahead of the queue head              |
| `chainValid` is O(1)                               | **30,279 gas at 1 open entry, 30,279 at 30**                                       |

**Costs disclosed, not absorbed.** Posting rose from **160,155 to
252,431 gas** (+58%). Pricing spam means storing a deadline, a
per-address per-epoch counter, and the accrued operations balance,
none of which the free version wrote. The registered ceiling moved
from 200k to 300k and the reason is recorded in the test.

**Two defects found while building, beyond the registered one.**

1. The pre-repair `chainValid()` scanned the whole open-entry array,
   so the _validity question itself_ got more expensive as an
   attacker grew the queue — a second, un-registered denial path in
   the same function the finding was about. Enforcing FIFO reduced
   it to a head check. That is why FIFO is in the repair and not
   only in the prose: it is what makes the O(1) reading possible.
2. The first version of the spam test asserted `chainValid()` right
   after posting 24 entries and failed. The cap was not at fault:
   every post is a block, so a 24-post campaign burned 24 blocks of
   a 10-block window before the librarian got a turn. **The test was
   measuring the harness, not the cap.** Recorded rather than
   quietly re-tuned, because "widen the window until it passes" is
   how a gate stops being one.

**Residual, stated in the contract's own docstring.** A spammer who
posts first can push an honest poster's deadline out by filling the
backlog ahead of it. The superlinear fee bounds this — buying $N$
slots of delay costs $O(N^3)$ — but it is a cost barrier, not a
proof.

---

### G25 — Librarian replacement escapes the earned-weight rule [HIGH]

"The librarian is replaced when a recorded divergence reason collects
endorsements from at least half of the registered validators." Every
other governance action uses pedigree + earnings weight + the 20% cap

- $d\ge3$ diversity. The single most powerful action — replacing the
  role that appends every ledger entry — uses an unweighted headcount of
  _registered_ validators, the cheapest quantity to Sybil (M335: at fee
  0.01 the per-identity recovery is 0.4). The design principle says the
  earned-weight rule applies to "every governance vote the network
  takes"; this one does not.

**Proposed repair.** Route librarian replacement through the same
voting-weight rule as everything else: earned (externally-verified,
per G13) weight, pedigree gate, 20% cap, $d\ge3$ diversity,
two-thirds. Keep the _trigger_ mechanical (a recorded divergence
reason is a replay-checkable fact) and make only the _endorsement_
weighted. Add the deterministic-deputy succession as-is. (M366)

#### M366 — SEALED, PASS (29 Aug 2026)

`geode/core/librarian_containment.replacement` no longer takes a
headcount. It takes the externally-verified weight map from G13's
`geode.core.voting_weight`, an explicit endorser list, and an
**injected** ratification predicate. There is no headcount path left
in the module.

| Gate                                  | Reading                                                                   |
| ------------------------------------- | ------------------------------------------------------------------------- |
| No recorded reason                    | never fires, whatever the weight — the trigger stays mechanical           |
| Old headcount majority (5 of 10)      | **no longer fires**; 0.5 is short of two thirds                           |
| Two thirds of earned weight           | fires at 7, 8 and 10 of 10                                                |
| 40 weightless Sybils vs 10 validators | **support weight 0.0, does not fire** — this is the attack G25 named      |
| Unpedigreed endorsement               | dropped **and reported** in `unpedigreed_dropped`, never silently ignored |
| One identity holding 90% of weight    | clipped to 20%, share 0.2, does not fire                                  |
| Four heavy endorsers, 50 responders   | clears two thirds, refused on `below_diversity_floor`                     |

**Why the predicate is injected rather than imported.**
`geode.privacy.vote_machinery` sits outside what `geode.core` may
import, and `tests/unit/test_architecture_layering.py` enforces that.
Copying the rule into `core` would have re-created the exact defect
G25 describes — one governance action running its own arithmetic. So
the parameter is **required**, not defaulted: a caller cannot fall
back on a local copy, because there is none.

**A composition defect found by the gate, not by review.** The 20%
cap test failed at first: a whale holding 90% of the weight still
read as a 0.667 share. The cap was clipping correctly; my
_denominator_ was wrong. I divided capped support by the **capped**
total, which re-inflates a clipped identity — clip 90 to 20, sum the
capped values to 30, and the whale is back to two thirds. The
registered M327 semantic is the raw total ("three capped identities
reach only 60%"), and `tests/unit/test_v26_m327_bootstrap_council.py`
says so explicitly. **A correctly implemented cap composed against
the wrong denominator is not a cap.** Checking the existing test for
the primitive settled it in one read; guessing would have shipped a
governance control that looked present and did nothing.

**Blast radius.** The M312 tier-4 harness
(`experiments/tier4/eval_v26_m312_librarian_containment.py`) was
updated to exercise the repaired rule and gained a Sybil-fleet cell.
Its C2 cell now measures the earned-weight path; the headcount
reading it previously recorded is superseded, not overwritten —
M312's sealed evidence stands as the record of what the old rule did.

---

### G53 — The replacement vote has no execution path at maturity [CRITICAL, CODE-CONFIRMED]

**Found 29 Aug 2026, not by review but by a question**: _does the
librarian rotate, how is it chosen, does it stake?_ Answering it
required reading the selection path end to end, and the path does not
close.

It does not rotate. Rotation in GEODE is a verifier mechanism
(`geode/core/rotation.py`, "quorum verifier sets rotate on a
deterministic ledger-index schedule"); the librarian is a fixed
operator until replaced by fault. It does not stake — the paper is
explicit that "there is no stake and no principal lockup", and the
bonds that exist belong to inbox posters and axis contributors, not
to the librarian. So **replacement is the only discipline the role
has.** M366 had just finished making that replacement a properly
weighted governance vote. The vote does not execute.

**The chain, all code-confirmed.**

1. `CreditLedger.setLibrarian` and `renounceLibrarian` are
   `onlyOwner` (`infrastructure/evm/contracts/CreditLedger.sol`).
2. The registered endgame is that the developer renounces ownership —
   "no human key remains after".
3. The repo's own audit gate **proves the freeze**:
   `credit_ledger.test.js`, "a renounced owner closes every admin
   path", asserts `setLibrarian` reverts once `owner() == 0`. What
   was written as an admin-release guarantee is also a librarian
   lock-in.
4. Therefore at maturity the on-chain librarian address can never
   change. A two-thirds earned-weight vote convicts a librarian of a
   recorded divergence, the deterministic deputy is named — and the
   key stays where it is.

**Both escapes fail.**

- _Never renounce ownership._ Then the developer holds a permanent,
  unilateral, unvoted power to install any librarian. That
  contradicts the retirement claim and makes the M366 vote
  decorative from the other direction.
- _Point the librarian at a governance contract before renouncing._
  Correct in principle and **no such contract exists in this repo**.
  The whitepaper's "a governance contract with no human key at
  maturity" has no referent.

**A second, independent instance.**
`InclusionInbox.librarian` is `immutable`. The force-inclusion queue
is the containment mechanism _against_ a misbehaving librarian, and
it is the one contract that can never learn the librarian changed. A
replaced librarian keeps the inbox role permanently, or the inbox is
redeployed and the open queue — including any entry someone is
currently being censored over — is abandoned. This predates M365; the
immutability was carried forward unexamined, and M365 made it worse
by giving the inbox more to do.

**Proposed repair (M382).** Separate the two powers that are
currently one.

- Add a `governance` address to `CreditLedger`, in the pattern
  `GovernanceFloors.governance` already establishes ("the timelocked
  governance executor"). `setLibrarian` becomes callable by the owner
  **or** governance; `renounceOwnership` then closes the developer's
  path and leaves the replacement path open. Governance can hand
  itself on, so the role survives its own succession.
- Give `InclusionInbox` a librarian **source** rather than a
  librarian copy, so the containment contract follows a replacement
  without redeployment. One address, one authority, read wherever it
  is needed.

**Gate (M382).** After `renounceOwnership`: the developer cannot set
the librarian; governance still can; the inbox honours the new
librarian and refuses the old one, with the open queue intact across
the change. And the pre-repair failure reproduces first — a renounced
owner with no governance address freezes the role.

**Registered scope limit.** This makes the vote _executable_. It does
not build the governance contract that should hold the address, and
it does not specify the deterministic deputy's successor order, which
is prose in the launch plan and absent from code. Both are named
below as what M382 does not close.

#### M382 — SEALED, PASS (29 Aug 2026)

EVM suite **83 passing** (was 75): 5 new gates on `CreditLedger`, 3 on
`InclusionInbox`.

| Gate                                 | Reading                                                                                                                                         |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Reproduce the freeze first**       | renounced owner, no governance set → `setLibrarian` reverts `NotOwnerOrGovernance`. The defect is pinned before the repair is read.             |
| Developer's path closes              | after `renounceOwnership` the owner cannot set the librarian                                                                                    |
| Replacement path stays open          | governance sets the librarian with the owner gone — the deputy takes the role                                                                   |
| Naming governance is a bootstrap act | `setGovernance` is owner-only; a stranger reverts                                                                                               |
| Governance survives succession       | `transferGovernance` moves the power; the old executor immediately loses it                                                                     |
| Governance is _not_ a second owner   | it replaces the librarian and **nothing else** — `pause`, `scheduleDevFundChange`, `scheduleRegistrationFee`, `setGovernance` all revert for it |
| Inbox reads live                     | `inbox.librarian()` follows the source, it is not frozen at deploy                                                                              |
| Replacement reaches containment      | the convicted librarian is refused `NotLibrarian`; the deputy incorporates; **the open queue is intact across the change**                      |

**Where this came from.** Not from re-reading the paper. From being
asked whether the librarian rotates, how it is chosen, and whether it
stakes. Two of those three answers are "it doesn't" — no rotation, no
stake — and the third is replacement. Reading the three together is
what exposed that the only discipline the role has was the one that
could not run. **A question about a role's lifecycle is a stronger
probe than a re-read of the section describing it**, because it forces
the path to be traced end to end rather than checked clause by clause.

**The audit gate that hid it.** `credit_ledger.test.js` has a passing
test called "a renounced owner closes every admin path", which asserts
`setLibrarian` reverts. It was written as an admin-release guarantee
and it is also the proof of the lock-in. **A green test asserting a
freeze is only good news if freezing that particular thing was
intended.** The test still passes, unchanged, and now sits beside a
test that reads the same fact as the defect.

**What M382 does NOT close** — registered here so it is not mistaken
for finished:

1. **The governance contract does not exist.** M382 creates the
   _address slot_ and proves the power lands there. The paper's
   "a governance contract with no human key at maturity" still has no
   referent in this repo. Until it does, `governance` is an EOA and
   the retirement claim is not yet true — it is only now _possible_.
2. **The deterministic deputy's successor order is unspecified.**
   The launch plan says "deterministic successor order"; no code
   computes it. `replacement()` returns `fires: true` and names
   nobody.
3. **Nothing times or timelocks the governance path.**
   `GovernanceFloors` puts a 7-day timelock on raising a floor.
   Replacing the librarian — a strictly larger power — executes
   instantly. Whether that is right (a captured librarian should be
   removable fast) or wrong (an instant path is a capture target) is
   not decided here.

These are the honest remainder of a CRITICAL finding downgraded to
executable, not a closed problem.

---

### G54 — The librarian is a single point of failure that holds no discretion [CRITICAL]

Raised by the reader, 29 Aug 2026, immediately after M382: _"if we're
adding more and more responsibilities on it this creates single points
of failure. We should ensure that the task can either be achieved in a
decentralized manner or that it doesn't make the librarian a single
point of failure. either by rotating or splitting responsibility or
preferably by making the network model pull by any concerned party
instead of push by a centralized party."_

M382 is a correct repair to the wrong layer. It made the librarian
_replaceable_. It did nothing about the fact that between replacements
one address must act for the whole network to make progress, and that
G24, G25, G53 and M382 are all repairs to the _consequences_ of that
one design choice.

**The inventory settles it.** Every privileged librarian action was
enumerated against one question — does this require private
information or judgement, or is it a function of public state?

| Action           | Requires judgement? | Actually decided by                              |
| ---------------- | ------------------- | ------------------------------------------------ |
| `incorporate`    | **No**              | FIFO over the on-chain queue; the head is forced |
| `recordCredits`  | **No**              | six stateless checks over on-chain state         |
| `slash`          | **No**              | off-chain replay of sealed data                  |
| `setDelisted`    | **No**              | a two-thirds validator quorum                    |
| `freezeArtifact` | **No**              | a confirmed ministerial order                    |
| `liftFreeze`     | **No**              | expiry, or confirmation failure                  |
| `setAdmitted`    | **No**              | the published evaluation rule                    |

Not one action is discretionary. In every case the librarian **files a
decision made elsewhere**. Access control on a function whose body is
already forced by public state buys no safety at all, and pays for that
nothing with the entire liveness and censorship surface. That is the
defect: not that the librarian might act wrongly, but that the network
waits on it to act at all.

**The repair is the reader's third option, and it is available because
of the table above.** Where an action is a deterministic function of
public state, the privilege is deleted rather than rotated or split.
Rotation and threshold-splitting both keep the push model and merely
spread the trust; making the call permissionless removes the need for
trust in that call entirely.

Three patterns cover the whole inventory:

- **(A) Permissionless-when-forced.** Delete the modifier. The
  contract already checks the precondition. `incorporate` and
  `liftFreeze`-on-expiry are here.
- **(B) Pull, not push.** The beneficiary claims against a published
  commitment instead of waiting to be paid. `recordCredits` is here,
  and it is the largest of them.
- **(C) Propose-and-challenge.** Anyone may file with a bond; a
  challenge window lets anyone refute by the same replay that decides
  guilt today. `slash`, `setAdmitted` and `setDelisted` are here.

What genuinely remains is **sequencing and data availability** — a much
smaller role, and one that no longer has to be trusted or even
present for the network to make progress.

#### Incentives are the other half, and the fee was pointing at the wrong party

Deleting a privilege does not by itself make anyone do the work. The
reader's follow-up — _"when we are decentralizing it we also have to
think of the incentives that are required for the librarian to behave
properly"_ — is what turns a permissionless call into a working one,
and it exposed a defect M365 had left in place.

M365 routes the posting fee to the operations line **at posting time**,
with the stated rationale that "the fee covers the librarian's
incorporation cost". It does not. It accrues whether or not the
librarian ever incorporates anything, so a stalled librarian and a
prompt one earn identically. The fee was named as payment for work and
implemented as unconditional income.

**Registered rule (M383): the fee follows the work.** It is held
against its entry until someone incorporates it, and then paid to
whoever did:

1. Librarian incorporates → fee to the operations line. Prompt service
   is paid, per entry.
2. A stranger incorporates **after the deadline** → fee to the
   stranger, as a bounty. A librarian that stalls does not merely fail
   to earn; it watches its income go to whoever covered for it. This
   is an automatic, continuous penalty with no quorum, no vote and no
   burn in the path.
3. A stranger incorporates **inside the deadline** → fee to the
   operations line, and the stranger earns nothing. Without this
   clause a poster could post and instantly reclaim its own fee, and
   the superlinear spam schedule of M365 would collapse to the cost of
   gas. The entry still goes in — censorship stays impossible — but
   the anti-spam price survives.

Payments are pull, not push, so a recipient that reverts on receive
cannot block the queue.

**Gate (M383).** Permissionless incorporation; a poster clears its own
entry with the librarian neither acting nor asked; a prompt librarian
is paid and a stalled one is not, measured on the same fee; the
self-incorporation refund exploit is closed; a censored poster pays gas
only; who incorporated is recorded, so "a stranger had to do it" is a
sharper liveness signal than a missed deadline.

#### M383 — SEALED, PASS (29 Aug 2026)

`infrastructure/evm/contracts/InclusionInbox.sol`; 6 new gates plus 3
rewritten. EVM suite **90 passing** (was 83).

| Gate                               | Reading                                                                            |
| ---------------------------------- | ---------------------------------------------------------------------------------- |
| Anyone may incorporate             | a stranger clears the head; counted as a foreign incorporation                     |
| Censorship structurally impossible | the poster incorporates its own entry, chain valid, librarian never asked          |
| Prompt librarian paid              | fee to the operations line                                                         |
| Stalled librarian not paid         | operations line **0**, stranger receives the fee                                   |
| Stalling is self-penalising        | 4 entries left past the deadline: **all** fees to the stranger, none to operations |
| Bounty claimable, once             | `claim()` pays exactly the fee, second call reverts `NothingToClaim`               |
| Spam price survives                | self-incorporation inside the window credits the poster **0**                      |
| Censored poster made whole         | past the deadline it recovers exactly the fee it paid                              |
| FIFO still binds                   | unchanged — a caller cannot reorder around a rival                                 |

**What M383 does not close.** Only pattern (A), and only for the
inbox. `recordCredits`, `slash`, `setAdmitted`, `setDelisted` and
`freezeArtifact` still run on the push model and are the larger part
of the work. `liftFreeze` is a near-free win — `isFrozen` already
expires on its own timestamp, so a vanished librarian cannot extend a
freeze; only _early_ release still needs a filing. These are queued
below as M384–M387 rather than folded in silently.

**The generalisable lesson.** _Access control is only load-bearing
where the function body is not._ For each privileged call, ask what
the privileged party actually decides. If the answer is "nothing — it
computes what anyone could compute", the modifier is pure liveness
risk and the repair is deletion, not rotation, not a multisig, and not
a better replacement vote. And once deleted, ask who is now paid to
make the call: a permissionless function nobody is paid to call is a
liveness risk wearing a different hat.

#### M384 — NO DEFECT (29 Aug 2026). Verified, not repaired.

M384 was queued on the inventory's claim that `liftFreeze` carries a
liveness failure: _"artifact stays frozen past `frozenUntil`; an escrow
window extends indefinitely"_. **That is false**, and it was checked
before anything was changed.

`isFrozen` reads `regs[id].frozenUntil > block.timestamp`, and it is
the **only** reader of the freeze anywhere in the contract. A freeze
closes on its own clock. `liftFreeze` exists solely for _early_ release
on confirmation failure; it cannot extend anything, and a librarian
that vanishes forever costs an operator at most the window already
registered on chain.

Two gates now pin the non-defect rather than argue it:

| Gate                                    | Reading                                                                                                                |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Absent librarian, freeze expires anyway | frozen → credits skipped `frozen artifact (M323)`; after the window `isFrozen` is false with `liftFreeze` never called |
| A stale `frozenUntil` blocks nothing    | `frozenUntil` stays non-zero forever and credits flow normally                                                         |

EVM suite **92 passing**.

**The lesson, which cost nothing here only because it was checked
first.** An inventory of a role's weaknesses is a set of claims like
any other, and a repair queued off a description repairs the
description. Generalisable form: _when a finding says "X fails if
nobody acts", find the line that reads X and check whether it needs
anyone to act._ Here it was one comparison against `block.timestamp`.

Early release stays in scope but is pattern (C), not (A): it turns on
evidence that confirmation failed, so it belongs with M387 rather than
in a permissionless-call milestone of its own. M384 closes with no code
change.

#### M385 — DONE (29 Aug 2026). Pull path built; one residual disclosed.

`recordCredits` made the librarian name every payee and every amount.
A payee it never named was never paid, and nothing on chain showed the
omission — the whole network's income sat behind one address's
willingness to type it out.

**What was built.** The librarian now publishes one 32-byte
`attributionRoot` per closed epoch, and each payee draws its own credit
with `claimAttribution` against a Merkle proof. Leaves carry a
**cumulative** amount, so a replayed claim pays nothing rather than
paying twice. The root is **write-once**: rewriting it would let the
librarian pay a favourite, swap the tree, and strand everyone else,
which is the withholding this milestone exists to remove. Delivery is
**permissionless** — anyone may push a proof, and the credit still
lands on the `who` in the leaf, so a stranger can only pay someone on
time, never redirect them.

| Gate                                       | Reading                                                                      |
| ------------------------------------------ | ---------------------------------------------------------------------------- |
| Payee paid with the librarian absent       | contributor draws 5,000 itself; librarian never the sender                   |
| No payee depends on being pushed to        | a stranger delivers the proof; credit lands on the operator, stranger gets 0 |
| Cannot pay one and strand the rest         | second root reverts `RootAlreadyPosted`; all three leaves still drawable     |
| Invented amount refused                    | `BadProof`                                                                   |
| Idempotent                                 | second identical claim changes nothing                                       |
| Withheld batch provable                    | `attributionRoot(0) == 0` publicly; claim reverts `NoAttributionRoot`        |
| Root posting is gated and epoch-closed     | stranger → `NotLibrarian`; open epoch → `EpochNotClosed`                     |
| Root cannot buy an artifact past delisting | `CreditSkipped(..., "delisted")`, credits unchanged                          |

EVM suite **100 passing**.

**The residual, stated plainly: this reduces the single point of
failure, it does not remove it.** The librarian is no longer needed to
_pay_ anyone — but it is still the only address that may _post the
root_, so a librarian that publishes nothing still stops the epoch's
income. What changed is that the failure is now (i) one call instead of
N, (ii) all-or-nothing rather than selective, and (iii) **visible**: an
absent root is a public zero, where a payee quietly dropped from a push
batch was invisible.

Removing the remainder needs a bonded propose-and-challenge root, and
that needs a dispute oracle to decide whether a challenged root is
wrong. The replay quorum already in the design is the natural one, and
wiring it is **M386**'s machinery, not a second copy here. It was not
improvised in this milestone: an optimistic root whose challenge nobody
can adjudicate is not decentralisation, it is a griefing surface with
better branding.

**Update (R3-F1, 29 Aug 2026): the root-posting residual is closed.**
The bonded propose-and-challenge root is implemented on `CreditLedger`
as `fileAttributionRoot` / `challengeAttributionRoot` /
`executeAttributionRoot` / `resolveAttributionRoot` / `claimRootBond`:
any party files a closed epoch's root under `SLASH_BOND`; an
unchallenged filing executes after the window; a challenge escalates to
the replay quorum, whose verdict is librarian-filed as elsewhere. The
filer whose root lands also earns the registered root-posting bounty
from the operations-line pool (the inbox's non-refundable posting fees,
pulled by `pullOperations`), so "anybody can settle onchain and claim
the fee for it" is now true for the root path. The remaining
quorum-authentication residual is only the verdict FILING
(`resolveSlash` / `resolveRegistryChange` / `resolveAttributionRoot`
stay `onlyLibrarian`), shared with M386/M387 as noted below. Gates: 14
in `attribution_root_filing.test.js`; EVM suite **148 passing**.

**The generalisable lesson.** _Converting a push to a pull is two
separable moves, and the cheap one is worth taking alone._ Move one
makes the privileged party commit to what it owes; move two makes the
commitment itself permissionless. Move one is a small diff, needs no
consensus machinery, and already converts silent selective withholding
into a public all-or-nothing outage. Waiting to ship it until move two
is ready keeps the worse failure mode alive for no reason.

#### M386 — DONE (29 Aug 2026). Challenge-windowed slash; disputed path disclosed.

The old `slash` was `onlyLibrarian`: one address decided every
takedown, and a librarian that stopped caring froze enforcement for
the whole network. The gate was registered as "a guilty artifact is
slashed with the librarian absent; a false accusation loses the bond;
the replay still decides guilt". It is now met by a filing →
challenge-window → execution flow.

**What was built.** `fileSlash` is permissionless with a `SLASH_BOND`
stake and validates the filing's shape at filing time (a filing that
can never apply is rejected before it costs anyone a bond). The filing
sits in a one-epoch `SLASH_WINDOW`; nobody refutes it and
`executeSlash` runs the identical burn/delist the old librarian path
did, with no privileged party in the call. A refutation — anyone may
`challengeSlash` against the same bond — escalates the filing to the
replay quorum, whose verdict `resolveSlash` files: guilty burns the
challenger's stake and the slash lands; innocent burns the FILER's
stake, the filing is void, and nothing is burned from the accused.
The winner always pulls its own bond back; the loser's bond is burned
— nobody gains from a penalty, matching the paper's own burn rule.
The librarian's direct `slash` fast path is kept as the trusted
replay-verified path, disclosed below.

| Gate                                                   | Reading                                                                                                 |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| A guilty artifact is slashed with the librarian absent | stranger files, stranger executes, `Burned` lands, bond pullable; librarian never the sender            |
| A false accusation loses the bond                      | accused challenges, quorum says innocent → filer's stake burned, challenger refunded, credits untouched |
| The replay still decides guilt                         | unchallenged = no one's replay contradicted it; challenged = the quorum's replay decides                |
| True filing survives its own challenge                 | guilty party challenges → still slashed AND loses the challenge stake                                   |
| Unchallenged filing cannot execute in-window           | `WindowOpen`                                                                                            |
| Challenged filing cannot auto-execute                  | `ChallengePending`                                                                                      |
| Shape rejected at filing time                          | `InvalidLevel` / `ZeroAmount` / `NothingToClaim` / `NotRegistered` / `WrongTarget`                      |
| Wrong bond refused                                     | `WrongBond` at filing and at challenge                                                                  |
| L2 still delists through the permissionless path       | `admitted` false, burn lands                                                                            |
| Balance drained in the window                          | `SlashSkipped("insufficient balance")`, filing resolves, bond returnable                                |

EVM suite **112 passing** (12 new gates).

**The residual, stated plainly.** The undisputed path needs no
privileged party. The _disputed_ path does: `resolveSlash` is
`onlyLibrarian`, because the quorum's verdict is off-chain and someone
must file it — the same shape as M385's residual. A vanished librarian
cannot stop an unchallenged takedown, but it can stall a challenged
one. That is the M387 work ("a ratified quorum verdict executes
itself"), and it was not improvised here: letting an unauthenticated
party file a quorum verdict would let anyone manufacture verdicts.

**The generalisable lesson.** _An optimistic path and its escalation
are separable, and the optimistic path alone already moves the trust
off the single point._ The bond game here was chosen so the ACCUSED is
the one with the strongest incentive to refute a false filing (protect
itself, no cost if exonerated) — so false filings are near-certain to
escalate, and true filings are near-certain to go unchallenged (the
guilty party gains nothing but delay by challenging, and pays a bond
for it). A bond game that priced the defender out of refuting, or made
challenges free, would have shipped the opposite failure: either false
filings go unanswered, or every filing funnels through the quorum.

#### M387 — DONE (29 Aug 2026). Registry changes challenge-windowed; one scope correction.

`setAdmitted`, `setDelisted` and `freezeArtifact` were the last three
push-model decisions after M385/M386, and the table of what the
librarian actually decides put all three in pattern (C): they file an
off-chain decision (the published evaluation rule, a two-thirds
quorum, a confirmed ministerial order) that the contract cannot
re-derive. The bond/window/quorum game built for M386 was therefore
generalised to a registry filing, additive rather than a refactor of
the sealed slash machinery.

**What was built.** `fileRegistryChange(kind, artifactId, admitValue,
freezeEpochs, evidenceHash)` is permissionless with the same
`SLASH_BOND` and one-epoch `SLASH_WINDOW`; kinds are 0 = admit (or
de-admit), 1 = delist (permanent), 2 = freeze (for `freezeEpochs`).
An unchallenged filing executes after the window with no privileged
party in the call; a challenge escalates to the replay quorum, whose
verdict applies the change or voids it, burning the loser's bond. The
librarian keeps its direct fast paths as the quorum's filer. A freeze
execution is non-derogating: it may extend an existing freeze but
never shorten one, so a stranger's filing cannot cut a legitimate
ministerial escrow short.

| Gate                                      | Reading                                                                     |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| Admission with no privileged filer        | stranger files admit, stranger executes, `Admitted` lands, librarian absent |
| Admission is bidirectional                | stranger's de-admission also executes unchallenged                          |
| Ratified quorum verdict executes itself   | stranger's delist executes unchallenged → `delisted` true                   |
| False delist refutable, filer loses bond  | operator challenges, quorum innocent → filer's stake burned, delist void    |
| Ministerial order on its own confirmation | stranger's freeze executes unchallenged; follows the filed window           |
| False freeze refutable                    | operator challenges, quorum innocent → not frozen, filer's stake burned     |
| Freeze never shortens a longer freeze     | 1-epoch filing cannot cut a 10-epoch ministerial freeze                     |
| Shape rejected at filing time             | `InvalidKind` / `NotRegistered` / `ZeroAmount`                              |
| Wrong bond refused                        | `WrongBond` at filing and at challenge                                      |
| Resolution is quorum-filed only           | stranger → `NotLibrarian`                                                   |

EVM suite **124 passing** (12 new gates).

**One scope correction, registered.** M384's writeup said early
release (`liftFreeze`) "belongs with M387". On building the generic
mechanism, it does not, and it was deliberately excluded. The
challenge incentives INVERT for an unfreeze: the party that benefits
from a false filing is the frozen artifact itself, so there is no
natural challenger to police an early release, and the mechanism would
depend on the librarian — the very party it exists to remove — to
challenge. Early release stays with the librarian, who filed the
freeze it is releasing. The correction is recorded here rather than
silently dropping the M384 note.

**The generalisable lesson.** _The bond game's polarity must match the
harm's polarity._ Every pattern-(C) call in this family works because
the party that loses from a false filing is the one with the standing
to challenge it (the accused, the operator, the listed artifact's
owner). Where that pairing inverts — here, an early release that the
frozen party would happily file against itself — the same machinery
becomes a way to file the harm, not a way to police it. A generic
propose-and-challenge is a family of mechanisms, not one mechanism:
each call needs its own answer to "who loses if this is false, and can
that party refute it?"

---

### G26 — Top-five crowding [HIGH]

$v_a = s_a/(p_a\bar u_a)$, with $p_a$ contributor-set and admission
requiring only that $s_a$ clear the axis floor. Five barely-passing
arms priced at the floor hold all five lottery slots and evict a
better arm that prices above the floor. Quality is a gate, not a rank
contributor, once you are through. Default routing is best-value, so
an adversary degrades an axis's delivered quality for the price of
five registrations plus five bonds.

**Proposed repair.** Two changes, both small:

1. **Reserve slots by quality.** The top five is the union of the top
   three by $v_a$ and the top two by $s_a$. A dominant-quality arm can
   never be evicted by price alone.
2. **Quality exponent.** Rank on $s_a^{\gamma}/(p_a\bar u_a)$ with
   $\gamma$ registered per axis ($\gamma = 1$ recovers today's rule).
   An axis where quality matters more than price registers
   $\gamma > 1$.

Report the delivered-quality effect of both under the crowding attack
before choosing.

**Gate (M356).** The traffic-share sweep, extended with a crowding
adversary: measure delivered accuracy at the user under (a) today's
rule, (b) reserved slots, (c) the exponent. Publish all three.

#### §M356 — SEALED, PASS (29 Aug 2026). Both repairs hold quality; today's rule collapses.

Instrument `tools/m356_top_five_crowding.py`, evidence
`analysis/m356_top_five_crowding.json`. Registered scenario: axis
floor 1.0; a strong arm A (0.95) and a good arm B (0.85) priced at
2.0; a crowd of N floor-priced arms at accuracy 0.60. Delivered
accuracy = expected accuracy of a served query under the score-
weighted top-5 lottery.

| Crowd size | Today's rule | Reserved slots | Exponent γ=2 | Exponent γ=3 |
| ---------- | ------------ | -------------- | ------------ | ------------ |
| 1          | 0.782        | 0.782          | 0.782        | 0.782        |
| 2          | 0.730        | 0.730          | 0.730        | 0.730        |
| 3          | 0.701        | 0.701          | 0.701        | 0.701        |
| 4          | **0.658**    | 0.701          | 0.701        | 0.701        |
| 5+         | **0.600**    | 0.701          | 0.701        | 0.701        |

At five crowd arms — G26's exact attack — today's rule serves only
the five crowd arms (delivered accuracy 0.600; A and B are evicted
entirely). Both repairs hold A and B in the pool at every crowd size,
delivering 0.701 — a 17% relative gain over the attack — and the
reserved-slot and exponent pools coincide in this scenario. They
differ in weighting: the exponent re-weights traffic toward A (0.451
vs 0.475), the reserved-slot rule keeps today's weights and changes
membership only.

The paper's router section now reports all three rules with these
numbers, per the gate's "report before choosing". The choice between
them is a governance registration per axis, not another code path.

---

### G27 — The detection-horizon claim fails on quiet axes [HIGH, ARITHMETIC]

"The detection horizon stays inside the vesting promise everywhere."
The sequential test needs a median 2,383 sessions ≈ **119 probed
sessions** at $\rho = 0.05$. On an axis running at the per-epoch
minimum of **one** probed session, that is 119 epochs ≈ 2.3 years,
against a 4-epoch vesting window. "Everywhere" is false precisely
where the minimum binds.

Worse, the Level-1 claim freeze lasts $\lceil$open exposure units /
units per epoch$\rceil$ epochs and applies to _all_ contributors, not
just suspected ones. So on a thin axis every honest supplier waits
years to claim. **Payout latency scales inversely with traffic,
suppressing supply exactly where the bootstrap needs it.**

**Proposed repair.** Make $\rho$ a function of axis traffic rather
than a constant with a flat minimum:

- Register a target _probed-sessions-per-epoch_ floor $P$ (proposal: 8) instead of 1. On a quiet axis $\rho$ rises toward 1.0 to hit it;
  on a busy axis $\rho$ stays at the 0.05 floor. The existing adaptive
  ρ machinery (M305a) already supports upward adjustment.
- Cost consequence stated plainly: on a quiet axis the probe overhead
  approaches $k_e \times$ serving cost. That is the honest price of a
  bounded detection horizon on low traffic, and it is a _bootstrap
  subsidy candidate_ — the operations line (G17) can carry it during
  an axis's first epochs.
- Cap the claim freeze at the vesting window and state the residual:
  beyond that cap, detection is bounded by the bond, not by the burn.
- Replace "everywhere" with the measured horizon as a function of
  axis traffic, published as a table.

**Gate (M367).** A horizon table: for axis traffic in {10, 100, 1e3,
1e4, 1e5} sessions/epoch, report median sessions-to-conviction, epochs
to conviction, probe overhead, and honest claim latency, under both
the flat-minimum rule and the traffic-adaptive rule. The paper
publishes the table instead of the word "everywhere."

#### M367 — SEALED (gate PASS), and **this review's own proposed floor was wrong**

Instrument `tools/m367_detection_horizon.py`, evidence
`analysis/m367_detection_horizon_table.json`. Arithmetic over the
sealed sequential-test median (2383 sessions at $\rho=0.05$, i.e.
**119 probed sessions** — the invariant, since the test consumes the
mismatch stream). Claim latency is not an independent quantity: the
Level-1 freeze lasts $\lceil$exposure sessions / traffic$\rceil$
epochs and exposure _is_ the detection window, so honest claim
latency **equals** epochs-to-conviction. The security number and the
supplier's payout latency are the same number.

| Traffic | Rule             | $\rho$ | Probed/epoch | Epochs to conviction | Overhead | Inside $N=4$?          |
| ------- | ---------------- | ------ | ------------ | -------------------- | -------- | ---------------------- |
| 10      | flat             | 0.10   | 1            | **119.0**            | 0.20     | no                     |
| 100     | flat             | 0.05   | 5            | **23.8**             | 0.10     | no                     |
| 1e3     | flat             | 0.05   | 50           | 2.38                 | 0.10     | yes                    |
| 1e4     | flat             | 0.05   | 500          | 0.24                 | 0.10     | yes                    |
| 1e5     | flat             | 0.05   | 5000         | 0.02                 | 0.10     | yes                    |
| 10      | adaptive, $P=8$  | 0.80   | 8            | **14.9**             | 1.60     | no                     |
| 100     | adaptive, $P=8$  | 0.08   | 8            | **14.9**             | 0.16     | no                     |
| 10      | adaptive, $P=30$ | 1.00   | 10           | **11.9**             | 2.00     | **no — at any $\rho$** |
| 100     | adaptive, $P=30$ | 0.30   | 30           | 3.97                 | 0.60     | yes                    |

**Correction to this review.** G27 proposed a probed-sessions floor of
$P=8$. That is **insufficient**: meeting a 4-epoch vesting window
requires $\lceil 119/4\rceil = \mathbf{30}$ probed sessions per epoch.
$P=8$ leaves the horizon at 14.9 epochs — better than 119, still
3.7× outside vesting. The registered floor is corrected to $P=30$.
The error is recorded rather than silently fixed, per the standing
rule on repairing an instrument after watching it fail.

**A residual the review did not find, and it is stronger than G27.**
Probed sessions per epoch cannot exceed traffic. At 10 sessions/epoch,
$\rho=1.0$ — probing **every single session**, at 200% overhead —
still yields 11.9 epochs. So there is a hard **traffic floor of 30
sessions/epoch below which the detection horizon cannot be brought
inside the vesting window at any probe rate whatsoever.** This is not
a tuning problem; it is arithmetic. On such an axis the substitution
attack is bounded by the registration bond alone, never by the burn.
The paper must say so, and the bond on a sub-30 axis must therefore be
sized against the full revenue an attacker can extract over 12 epochs,
not over 4. Registered as **G51**.

**Cost consequence, stated not hidden.** Buying the horizon on a
100-session axis costs 60% of serving cost in probe overhead against
10% on a busy axis. That is the honest price and it is a bootstrap
subsidy candidate (G17's operations line).

---

### G28 — No minimum executor-pool size [MEDIUM]

"A corrupt fraction of the pool raised to the executor count" assumes
a large pool. With $k_e = 2$ and a pool of 2–3 — the realistic case
for a niche artifact — collusion probability is 1 or near it.

**Proposed repair.** Register a minimum pool size $\Pi$ (proposal:
$\Pi = 8$, giving $\binom{8}{2}$ = 28 samples and a corrupt-pair
probability of $(c/8)^2$). Below $\Pi$, the artifact falls back to
behavioural identity as the operative mechanism and its Known-Limits
residual, and its bond is sized on the weaker instrument. Publish the
pool size per artifact in the registry so a buyer can see which
instrument is operative. (M368)

#### §M368 — SEALED, PASS (29 Aug 2026). The fallback is explicit; the registry shows it.

`geode/core/executor_pool.py`, 6 tests. `operative_instrument(pool)`
returns `sampled_executors` at/above `MIN_EXECUTOR_POOL = 8` and
`behavioral_identity_fallback` below it; `registry_entry` publishes
the per-artifact row a buyer reads; `corrupt_sample_probability` is
the EXACT hypergeometric form (C(c,k)/C(n,k)) of G28's "corrupt
fraction raised to the executor count" — at pool 8, sample 2, a 25%
corrupt fraction gives 1/28, a 50% fraction 6/28, and the pool-2 case
G28 names reads 1.0.

The gate ("below Π the operative mechanism falls back and the
registry shows it") is met: an artifact with a pool of 2 is publicly
`behavioral_identity_fallback`, and its bond is sized on that weaker
instrument, never on the sampled-executor guarantee.

---

### G29 — Challenge-corpus depletion [MEDIUM]

Challenges are drawn from a per-axis corpus "committed by Merkle root
at axis creation," and "every revealed point is public thereafter." An
attacker submits repeated registrations, paying only fees, to burn the
corpus; after depletion the axis cannot admit anyone against unrevealed
data. Separately, the replenishment language ("shards reshuffle at each
rotation, retired rows are replaced") belongs to the _evaluation_
corpora and would change a root that is fixed at creation.

**Proposed repair.**

- Commit the corpus as a **sequence of roots** (an append-only
  commitment tree), so replenishment extends the commitment instead of
  breaking it. Each admission records which epoch's root it drew from.
- Register a per-axis depletion budget and an alarm: when unrevealed
  points fall below a registered fraction, admissions on that axis
  pause until the corpus is extended. A pause is public and
  measurable; a silently exhausted corpus is not.
- Size the registration fee so that a depletion campaign costs more
  than extending the corpus.

**Gate (M369).** A depletion simulation: cost to the attacker to
exhaust an axis vs. cost to the network to replenish. The fee is set
so the ratio exceeds a registered margin.

#### §M369 — SEALED, PASS (29 Aug 2026). Append-only roots; the alarm is public; the gate ratio holds.

`geode/core/challenge_corpus.py`, 6 tests. `AppendOnlyCorpus` is the
sequence of roots: `commit` extends, `reveal` records draws per root
index, and replenishment never rewrites a prior commitment.
`admissions_paused()` is the public alarm — it fires when the
unrevealed fraction falls below `DEPLETION_PAUSE_FRACTION = 0.25`
and clears when a new root extends the corpus. `depletion_gate`
simulates the fight: attacker cost (registrations × fee) vs network
cost (replenished points × per-point cost), ratio vs
`DEPLETION_MARGIN = 2.0`.

The measured gate semantics: a fee of 5.0 with 10 points revealed per
registration and 0.4 per replenished point gives a 1.25 ratio —
**below** the margin, and the gate says so; a fee of 20.0 gives 5.0
and clears it. The simulation does not pretend a too-low fee closes;
it reports which fees do.

---

### G30 — Two generations of the challenge design coexist [MEDIUM]

"Challenges are drawn, not authored… the validator's job is to sample,
pose, verify, and attest — never to choose the exam." Then §Failure
handling still carries the "wrong labels" path where a validator's
"revealed expected output disagrees with the independent audit
relabeling," and §The challenge session still funds a 10% relabeling
audit. If labels come from a Merkle-committed corpus, a validator
cannot plant one, and the audit has nothing to audit.

**Proposed repair.** Decide which axes have committed labels and which
require human labelling, and scope the audit to the latter explicitly.
For committed-label axes: delete the wrong-label failure path, keep a
_commitment-conformance_ check (the revealed point opens against the
axis root — a cheap mechanical check, not a relabeling), and redirect
the audit budget to the conformance check. For human-labelled axes:
keep the audit as written and say so. Today the paper reads as if both
regimes apply at once, which is where a reader loses the thread. (M370)

---

### G31 — The beacon is an uncatalogued trust point [MEDIUM]

Every "no one chooses their judges" guarantee — validator sampling,
executor sampling, probe flags, the routing draw, corpus draws —
reduces to the randomness beacon. drand is a trusted committee;
RANDAO alone is last-proposer-grindable. The paper offers "drand, or
the beacon chain's RANDAO composed with a verifiable delay function"
as alternatives, which understates how load-bearing the choice is, and
the dependency appears nowhere in Known Limits.

**Proposed repair.**

- Make the VDF composition **required**, not an alternative: register
  RANDAO+VDF as the beacon and drand as a fallback, with the
  composition rule stated.
- Add a Known Limits entry: the beacon is an external dependency; a
  compromised beacon compromises every sampling guarantee at once, and
  the mitigation is composition of two independent sources (register
  the composition: $H(\text{drand} \parallel \text{RANDAO-VDF})$, safe
  as long as either is honest).

(M371)

#### §M371 — SEALED, PASS (29 Aug 2026). The composition is required; the residual is catalogued.

`geode/core/beacon.py`, 5 tests. `composed_beacon(drand, randao_vdf)`
is the registered `H(drand ∥ RANDAO-VDF)`: ordered, deterministic,
changes with either source. `beacon_safe_if_either_honest` pins the
composition property the review registered — safe as long as EITHER
source is honest; only both compromised breaks it.

The whitepaper's beacon sentence no longer offers "drand, or RANDAO +
VDF" as alternatives: RANDAO+VDF is the beacon, drand the fallback,
and the effective seed is the ordered hash of both. A Known Limits
entry names the beacon as an external dependency, the single point
every sampling guarantee reduces to, and the composition as the
mitigation.

---

## 6. Prior art

The paper's stance ("we claim only the assembly and the discipline")
is correct and should not change. These are gaps _against that
stance_ — parts the paper uses without naming, which in a paper that
names every other part reads as an implicit claim.

| ID  | Missing work                                                                                                                                                                                | Where it belongs                                                                                                                                                                                                                                                |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| G32 | **Rahimi & Recht, "Random features for large-scale kernel machines," NeurIPS 2007** — plus Weinberger et al. 2009 for the hashing trick                                                     | The results table has a row named "random features" and the text describes "a hash-seeded random-feature map." This is the single most conspicuous omission in the bibliography.                                                                                |
| G33 | **Numerai** — sealed held-out evaluation, models scored on data the contributor never sees, commit-style submission, payment by measured out-of-sample contribution, staking and slashing   | §Prior art, neighbour systems. It is the closest deployed neighbour to "measured, not asserted" plus economic payment, closer than Bittensor on that specific axis.                                                                                             |
| G34 | **Randomised redundant execution / spot-checking**: Golle & Mironov (ringers, CT-RSA 2001); Du et al., "Uncheatable grid computing" (ICDCS 2004); BOINC replication; Golem                  | §Serving verification. The shadow probe is this mechanism; name it.                                                                                                                                                                                             |
| G35 | **Truebit's verification game** and optimistic-rollup fraud proofs                                                                                                                          | §Disputes. opML is cited and descends from it; cite the lineage directly.                                                                                                                                                                                       |
| G36 | **Model fingerprinting / Proof-of-Learning**: IPGuard (Cao et al. 2021), conferrable adversarial examples (Lukas et al. 2021), Proof-of-Learning (Jia et al. 2021)                          | §Behavioural identity. "Verify the served artifact is the registered one" is that literature.                                                                                                                                                                   |
| G37 | **zkML**: Kang et al. 2022 (ImageNet-scale ZK inference), zkCNN (Liu et al. 2021), EZKL, Modulus                                                                                            | §Proofs of computation. Several use exactly the paper's split — commit to features, prove the final linear layer — and they are the evidence base for the cost claims in G2.                                                                                    |
| G38 | **Ocean Protocol compute-to-data**                                                                                                                                                          | §Custody. Sealed data, algorithms shipped to it, aggregate results returned, buyer never sees rows. That is the sealed scoring environment.                                                                                                                     |
| G39 | **Feature stores** (Feast, Michelangelo, the feature-store literature)                                                                                                                      | §Prior art, beside AdapterHub. The repo's own sweep-2 identified them as a neighbour to the versioned code bus; the paper's novelty claim _is_ the versioned bus, so the neighbour must appear with its distinction (org-internal, no economy, no attribution). |
| G40 | **MeritRank** (arXiv:2207.09950) — Sybil-tolerant reputation that bounds, rather than prevents, the gain from a Sybil ring by decaying contribution value over a feedback graph             | §Voting weight. Found by sweep 3 as the D3 partial displacer of G13's repair; citation mandatory, distinction is graded-decay vs. binary externality.                                                                                                           |
| G41 | **_Concave is the New Linear_** (arXiv:2605.18990) — no voting rule deriving power solely from wallet balance survives Sybil splitting; measured amplification 1,172×–229,000× on real DAOs | §Voting weight **and** Known Limits. It is the reason G13's repair is mandatory rather than optional: the 20% cap is a concave rule, and only genuinely earned weight keeps GEODE outside the theorem.                                                          |
| G42 | **Wash-trade forensics** (arXiv:2305.01543, arXiv:2306.04643) and **trust-graph Sybil resilience** (Poupko et al., arXiv:1901.00752)                                                        | §Voting weight and §Registration. Forensic wash detection is the neighbour that does _not_ feed a weight base; trust-graph conductance is the nearest relative of the $d\ge3$ diversity floor.                                                                  |

**Proposed repair (M378).** One citation wave: add all nineteen
entries (twelve here, plus G44–G50 located by the sweep-4 second
index below) to the bibliography and to §Prior art with a
one-sentence distinction
each, in the existing style ("Both are training-time architectures
inside a single network. Neither registers independent frozen
artifacts…"). Then re-run the prior-art instrument on the two claims
with these names as **new anchors** — an anchor set that misses
Numerai or Rahimi & Recht is not a validated instrument, which
retroactively weakens both sweeps.

**Note on the sweeps.** This is the finding with the sharpest
methodological consequence. Per the standing lesson, a search whose
anchors do not include the obvious neighbours cannot support any
absence claim. Sweeps 1 and 2 should be re-run with the expanded
anchor set before the novelty paragraph is published as written.

### G43 — Both sweeps were arXiv-only, and arXiv cannot see the two obvious neighbours [HIGH, MEASURED]

**Discovered while trying to satisfy the M378 gate**, which reads
"both sweeps re-run with Numerai and Rahimi & Recht among the
anchors." That gate is **unsatisfiable on the registered
instrument**, and the reason is worse than a phrasing problem.

Measured on the live index:

| Probe                                                       | arXiv                                                                | OpenAlex                                |
| ----------------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------- |
| `"Random Features for Large-Scale Kernel Machines"` (title) | 1 hit, and it is a **different paper** — a sub-phrase false positive | the actual Rahimi & Recht paper, rank 1 |
| `Numerai`                                                   | 3 hits, all papers that merely _use_ Numerai data                    | the system's surrounding literature     |

Rahimi & Recht is NIPS 2007, which predates arXiv-by-default in
machine learning; Numerai is a deployed system with no canonical
paper of its own. So the arXiv-only instrument is **structurally
blind to exactly two categories**: pre-~2010 conference papers, and
deployed systems that never published. Those are precisely the two
categories G32 and G33 name.

**Consequence.** Sweeps 1 and 2 are weaker than recorded, and not for
the reason the sweep documents give. Their anchors were validated for
_phrasing sensitivity_ and passed; nothing tested _index coverage_.
"No displacer found by this instrument" is true, but the instrument
could not have found a displacer of either kind. The whitepaper's
prior-art position does not currently rest on a search that could
have refuted it.

**Proposed repair (M378, revised).** A **two-index** instrument:
arXiv (as today) plus OpenAlex (unauthenticated, covers conference
proceedings and pre-2010 work). Add two **coverage anchors** —
distinct in kind from the existing liveness and sensitivity anchors:

- a coverage anchor that OpenAlex must find and arXiv must **miss**
  (Rahimi & Recht). Its arXiv miss is not a failure; it is the
  measurement that proves the blind spot is real and that the second
  index closes it;
- a deployed-system coverage anchor (Numerai), which tests the second
  blind category.

Semantic Scholar returns HTTP 429 intermittently and must keep the
existing separate residual-failure field — a rate limit recorded as
zero hits is indistinguishable from a genuine empty result.

**Gate (M378, revised).** All four anchor classes behave as
registered (liveness hit, sensitivity hit without the title, coverage
anchor found by OpenAlex and missed by arXiv, decoy zero); every
query from both earlier sweeps re-run on both indexes; the novelty
paragraph rewritten against whatever comes back. Absence still proves
nothing and "first" is still not claimed — but after this the absence
is at least from an index that could have contained the answer.

#### M378 sweep 4 — RUN, instrument VALID, no displacer, seven new mandatory citations

Instrument: `tools/prior_art_search_4.py`. Evidence:
`analysis/prior_art_sweep_4_m378_2026-08-28.json`. 29 query pairs
(11 sweep-1, 12 sweep-2, 6 anchors) across both indexes, 396 OpenAlex
hits and 289 arXiv hits.

**Anchor contract — all four classes behaved as registered.**

| Class       | Anchor                              | Registered expectation                    | Measured                                                                                                                                                          |
| ----------- | ----------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Liveness    | AdapterHub; Bittensor               | must hit on arXiv                         | 3; 11 — both include the real paper                                                                                                                               |
| Sensitivity | blockchain + ML + network, no title | must surface known work without naming it | 15                                                                                                                                                                |
| Coverage 1  | Rahimi & Recht                      | OpenAlex hits, arXiv **misses**           | OpenAlex rank 1, 2,645 citations; arXiv's single hit is _Mercer Large-Scale Kernel Machines from Ridge Function Perspective_ — a **different paper**, i.e. a miss |
| Coverage 2  | Numerai                             | deployed system, no canonical paper       | arXiv 3, OpenAlex 3 — **neither index has a Numerai paper**; every hit merely uses the data                                                                       |
| Decoy       | retired title string                | zero on both                              | 0 and 0                                                                                                                                                           |

Coverage anchor 1 is the finding's own confirmation: the blind spot
G43 asserts is real, and the second index closes it. Coverage anchor 2
confirms G33's premise directly — the strongest deployed neighbour to
the payment claim is **unpublished in every index**, so no literature
search of any depth can retire it. It must be cited as a system.

**Instrument caveat, recorded before the results were read.**
OpenAlex's `search` ranking is relevance-over-corpus and is noisy at
depth: the raw citation-ordered union of OpenAlex-only titles is led
by AlphaFold, GROMACS and a gut-microbiome catalogue. The results were
therefore read **topically**, per query, not by citation count, and
`meta.count` remains excluded as a displacement signal.

**Displacement verdict: none.** No hit on either index satisfies the
sweep-1 conjunction (payment by measured held-out utility of _frozen_
artifacts + deterministic replayable decisions + native-ETH
epoch-vested settlement with burn slashing) or the sweep-2 conjunction
(frozen artifacts composed by declaration + versioned
upgrade-without-invalidation + an economy paying block owners by
measured downstream use).

**But the second index found seven neighbours the arXiv-only sweeps
structurally missed**, and one of them is the closest economic
relative yet located:

| #   | Work                                                                                                                           | Why it is a neighbour                                                                                                                                                   | Distinction                                                                                                                                   |
| --- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| G44 | Agarwal, Dahleh & Sarkar, _A Marketplace for Data: An Algorithmic Solution_ (EC 2019, 217c)                                    | Pays data contributors by their **measured marginal contribution** (Shapley) to model quality — the nearest published relative of GEODE's pay-by-measured-use principle | Prices and pays for **data**, not frozen composable artifacts; a centralised broker with no replayable decision record, no stake, no slashing |
| G45 | Chen, Koutris & Kumar, _Towards Model-based Pricing for Machine Learning in a Data Marketplace_ (SIGMOD 2019, 124c)            | Prices **model instances** by quality along a tuning path                                                                                                               | Pricing **to buyers**; says nothing about paying upstream contributors                                                                        |
| G46 | Sun, Chen, Liao & Huang, _A Profit-Maximizing Model Marketplace with Differentially Private Federated Learning_ (INFOCOM 2022) | A model marketplace with an incentive layer                                                                                                                             | Federated training rounds, not registration of frozen artifacts; broker-maximising, not contributor-paying                                    |
| G47 | Ghodsi, Gu & Garg, _SafetyNets_ (NIPS 2017)                                                                                    | Verifiable execution of a DNN on an untrusted cloud — the ancestor of §Serving verification                                                                             | Interactive proof for one specified network; no economy, no registration, no dispute settlement                                               |
| G48 | Feng, Qin, Zhang & Ding, _ZEN_ (IACR 2021)                                                                                     | A zkML compiler, beside the G37 entries                                                                                                                                 | Same distinction as G37: proves _an_ inference ran, not that the artifact was useful                                                          |
| G49 | Rebuffi, Bilen & Vedaldi, _Learning multiple visual domains with residual adapters_ (NIPS 2017, 496c)                          | The **origin** of the residual-adapter line that AdapterHub descends from; predates the G34 entries                                                                     | Training-time modules inside one network, one owner, no independent registration                                                              |
| G50 | Huang, Liu, Lin & Pang, _LoraHub_ (2023)                                                                                       | Dynamic composition of independently trained LoRA modules — the closest _composition_ relative                                                                          | Composes by gradient-free weight search at inference; no declared dependency graph, no versioning, no economy                                 |

G49 is the sharpest of the seven: the paper's §Prior art currently
dates the adapter line to AdapterHub (2020), when the residual-adapter
construction is 2017. That is an attribution error in the paper's own
prior-art section, not merely a missing reference.

**Consequence for the citation wave.** M378 grows from twelve entries
to **nineteen**. The novelty paragraph must now be written against
G44 specifically: the honest position is not "no one pays by measured
contribution" — Agarwal et al. do — but that no located work pays by
measured contribution _of frozen, independently registered,
composable artifacts_ under a replayable settlement rule.

**Residual failures.** None. No 429 or 503 survived backoff on either
index; every empty result in the evidence file is a genuine empty.

**What this run still does not license.** Two indexes are better than
one and OpenAlex demonstrably reaches where arXiv cannot, but neither
covers unpublished deployed systems — Numerai proves that inside this
very run. Absence remains uninformative; "first" remains unclaimable.

#### M378 citation wave — SEALED (gate PASS)

First edits to `docs/WHITEPAPER_GEODE.tex` in this review. **28**
`\bibitem` entries added, covering all nineteen registered entries
G32–G50 (several entries are multi-paper lines). Bibliography grew
30 → 58.

Woven in, not merely appended:

- **§Prior art** gains five itemize entries — fixed random features
  (G32, with the results row disclaimed as a reproduction),
  pay-by-measured-contribution (G44/G45/G46 plus Numerai as a
  **system**, G33), replication and spot-checking (G34),
  dispute-by-fraud-proof (G35), behavioural model identity (G36),
  zero-knowledge inference (G37/G48/G47), sealed compute
  environments (G38), and redistribution mechanisms.
- **G49 attribution error corrected in the paper itself.** The
  composition paragraph no longer dates the adapter line to
  AdapterHub (2020); it now opens on Rebuffi et al. (2017), with
  AdapterHub as what "made that line a registry." LoraHub and
  feature stores added as the remaining composition and versioning
  relatives.
- **§Voting weight** gains a positioning paragraph carrying G40,
  G41 and G42. It states the impossibility result plainly, concedes
  that the 20% cap is exactly the kind of concave rule the theorem
  covers, and rests the escape on weight not being a balance. The
  binary-vs-graded distinction from MeritRank is stated **with its
  residual** (rings longer than the closure depth), cross-referenced
  to Known limits via a new `sec:limits` label.
- **Novelty paragraph rewritten** against G44, as the gate required.
  It no longer implies nobody pays by measured contribution; it now
  concedes Agarwal et al. by name and narrows the claim to frozen,
  independently registered, composable artifacts under a replayable
  settlement rule. It also discloses in the paper that the first two
  sweeps were arXiv-only and names the blind spot.

No LaTeX toolchain is installed, so the edits were validated
statically: 58 bibitems, **zero duplicates, zero cited-but-missing
keys, zero defined-but-uncited keys**, all `\ref` targets resolve,
braces balanced (880/880).

---

## 7. Readability

| ID  | Issue                                                                                                                                                                                                                                                                                              | Repair                                                                                                                                                                           |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R1  | **No glossary.** capability / artifact / arm / primitive / block / representation artifact / head / trunk / encoder / code / manifest / bus / axis / epoch / promise                                                                                                                               | A one-page table after §The actors. Highest-value single edit in this section.                                                                                                   |
| R2  | **"Promise" is never defined.** First use is in the slashing table ("burns the unvested promise"); used ~15 times                                                                                                                                                                                  | Define once: the unvested credit balance.                                                                                                                                        |
| R3  | **Three names for one balance**: "earned-but-unclaimed" (voting), "thawed-but-unclaimed" (takedown), "vested-but-unclaimed" (slashing L3)                                                                                                                                                          | Unify on one term; note that after M359 there may genuinely be two balances, in which case name them both once and use them consistently.                                        |
| R4  | **"Open probe exposure" / "open-exposure window" undefined** — they carry the bond size and the claim-freeze duration                                                                                                                                                                              | Define in the glossary and at first use.                                                                                                                                         |
| R5  | **The protocol is stated three times** (black box → protocol in detail → appendix). This is where G12 and G30 drifted apart                                                                                                                                                                        | Keep the three passes, but forbid new rules in the appendix; the appendix narrates, it does not legislate. Add a note to that effect at the head of the appendix.                |
| R6  | **Forward reference**: the beacon is used in §Admission ("defined with the ledger below") before §The ledger defines it                                                                                                                                                                            | Move the beacon definition ahead of admission, or into the glossary.                                                                                                             |
| R7  | **§Serving verification is the longest itemize in the paper** (nine multi-paragraph bullets) and carries the most load                                                                                                                                                                             | Promote each bullet to a `\paragraph`; add the per-tier matrix from G3.                                                                                                          |
| R8  | **Known Limits is 23 unranked items**; #15 (the trunk-backdoor case, "the failure mode with the largest blast radius") is buried mid-list                                                                                                                                                          | Split into "structural residuals" and "open questions," each severity-ordered.                                                                                                   |
| R9  | **Results table mixes directions** — accuracy (higher better) and WER/NRMSFE (lower better) in one unlabelled column; unlabelled triples ("0.0296 / 0.0279 / 0.0261")                                                                                                                              | Add ↑/↓ per row and a ladder legend. Separate the architecture's own rows (Fusion, Nonlinearity) from the reproduction checks with a midrule.                                    |
| R10 | **Notation drift**: $W^\top z$ everywhere, $W^Tz+b$ in §Private serving                                                                                                                                                                                                                            | Remove the bias or introduce it once.                                                                                                                                            |
| R11 | **Ambiguity in $s_a$**: §Black box says the axis metric is multiplied by coverage; §Router says $s_a$ is "the axis metric oriented so that higher is better"                                                                                                                                       | Given G20 this is substantive. State $s_a$'s definition once, in the router. **CLOSED by M355** — `\label{sec:router}` holds the single definition; §Black box now refers to it. |
| R12 | **Style**: the staccato register works in abstract/intro/conclusion; across 40 pages of protocol the reader loses which sentence is the rule and which is the gloss. Also, the abstract's "claims no new learning algorithm" precedes the composed-codes claim, so the claim reads as a retraction | Longer sentences in the protocol sections; keep the short register for framing. Reorder the abstract's last two sentences so the claim precedes the disclaimer.                  |

---

## 8. Registered milestone queue (M349 onward)

Continues from M348. Every cell registers its gate **before** it runs.

| ID   | Cell                                                                                                                                        | Closes         | Gate                                                                                                                                                                                                                                                                               | Blocking?                                                                                           |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| M349 | Encrypted bucketing feasibility + private-tier extraction bound                                                                             | G1             | Argmax+bucket FHE circuit within 5× of head-only latency, or option 2 adopted and the tier's extraction multiple measured and published                                                                                                                                            | **OPTION 2 ADOPTED (29 Aug)**                                                                       |
| M350 | Proof cost table + per-tier capability matrix; proofs moved to sampled batches                                                              | G2, G3         | Prover/verifier/size measured per axis dimension; no cost claim in the paper absent from the table                                                                                                                                                                                 | **SEALED, PASS**                                                                                    |
| M351 | Committed-seed FHE evaluation                                                                                                               | G4             | Same seed → byte-identical ciphertext; different seed → not; security parameters unchanged                                                                                                                                                                                         | **SEALED, PASS (29 Aug)**                                                                           |
| M352 | Privacy tier as a descriptor field and a separate axis                                                                                      | G5, G6         | Tier in the descriptor hash and the unit table; cross-tier routes refuse; no "available premium trunk tier" text remains                                                                                                                                                           | **SEALED, PASS**                                                                                    |
| M353 | Dedup on error-set novelty                                                                                                                  | G7             | Failure reproduced on sealed speech/code evidence first; then novelty rule admits a distinct arm and still refuses a bit-flip and a distilled clone                                                                                                                                | **SEALED — failure NOT reproduced; G7 re-scoped, paper unchanged**                                  |
| M354 | Session entropy in the route seed                                                                                                           | G16            | Traffic-share sweep with distinct session ids reproduces the published shares; the configuration that produced the original figure is identified and disclosed                                                                                                                     | **SEALED, PASS**                                                                                    |
| M355 | Declared-label-set scoring replaces coverage multiplication                                                                                 | G20, R11       | Scoped arm outranks the guesser for a 129-class declaration and is unqualified for a 601-class one                                                                                                                                                                                 | **SEALED, PASS**                                                                                    |
| M356 | Top-five crowding defense                                                                                                                   | G26            | Delivered accuracy under a crowding adversary measured for all three rules; all published                                                                                                                                                                                          | **SEALED, PASS (29 Aug)**                                                                           |
| M357 | Abstention metered at measured compute fraction                                                                                             | G19            | Per-axis fraction measured; the M332 extraction bound re-derived and not weakened                                                                                                                                                                                                  | **SEALED, PASS (29 Aug)**                                                                           |
| M358 | Externally-verified voting weight                                                                                                           | G13            | **SEALED, PASS** — 3-cycle weight 292.5 → 0.0, haircut 2.5% (measured, not the estimated 5%), honest weight untouched; two residuals registered                                                                                                                                    | **SEALED, PASS**                                                                                    |
| M359 | Voting escrow decouples burn base from weight base                                                                                          | G14            | Escrow is voluntary, term-bounded, burnable at L3; unescrowed vested credits carry no weight                                                                                                                                                                                       | **SEALED, PASS (29 Aug)**                                                                           |
| M360 | Price drift band + currency-risk Known Limit                                                                                                | G21            | Router replays unchanged under drifting prices; residual published                                                                                                                                                                                                                 | **SEALED, PASS (29 Aug)**                                                                           |
| M361 | Reference hosting cost becomes a multi-party statistic                                                                                      | G22            | Median-of-admitted rule with the <3-arm fallback; Known Limits entry until transition                                                                                                                                                                                              | **SEALED, PASS (29 Aug)**                                                                           |
| M362 | Maintenance folded into the operations line; zakat converts the full 2.5%                                                                   | G18            | Charter has no discretionary maintenance budget; public goods are paid roles                                                                                                                                                                                                       | **SEALED, PASS (29 Aug)**                                                                           |
| M363 | Operations line: every working role gets an income line                                                                                     | G17            | Cost model closes at the reference workload with each role itemized; librarian gas included                                                                                                                                                                                        | **SEALED (29 Aug) — closes at a sized fee; harness fee does not**                                   |
| M364 | Abort allowance replaces flat abort-as-deviation                                                                                            | G23            | DoS campaign produces zero burns within the allowance; probe-dodging is still not cheaper than mismatch in expectation                                                                                                                                                             | **SEALED, PASS (29 Aug)**                                                                           |
| M365 | Inbox posting fee, rate limit, per-epoch incorporation cap                                                                                  | G24            | Spam costs superlinearly and never forces `chainValid == false`; a censored entry from a fresh address is still included within the bound                                                                                                                                          | **SEALED, PASS**                                                                                    |
| M366 | Librarian replacement uses the earned-weight rule                                                                                           | G25            | Replacement endorsement is weighted, capped, diverse, two-thirds; the trigger stays mechanical                                                                                                                                                                                     | **SEALED, PASS**                                                                                    |
| M367 | Traffic-adaptive ρ + horizon table; claim freeze capped                                                                                     | G27, G51       | Horizon table across five traffic levels published; "everywhere" removed; floor corrected 8 → 30; sub-30-traffic impossibility stated in Known limits                                                                                                                              | **SEALED, PASS**                                                                                    |
| M368 | Minimum executor pool with published per-artifact size                                                                                      | G28            | Below Π the operative mechanism falls back and the registry shows it                                                                                                                                                                                                               | **SEALED, PASS (29 Aug)**                                                                           |
| M369 | Append-only corpus commitment + depletion budget and alarm                                                                                  | G29            | Depletion cost / replenishment cost ratio exceeds the registered margin; pause is public                                                                                                                                                                                           | **SEALED, PASS (29 Aug)**                                                                           |
| M370 | Challenge-design generation cleanup                                                                                                         | G30            | Committed-label axes and human-labelled axes have disjoint, explicitly scoped failure paths                                                                                                                                                                                        | **SEALED, PASS (29 Aug)**                                                                           |
| M371 | Beacon composition required + Known Limits entry                                                                                            | G31            | $H(\text{drand} \parallel \text{RANDAO-VDF})$ registered; safe if either source is honest                                                                                                                                                                                          | **SEALED, PASS (29 Aug)**                                                                           |
| M372 | Per-payer budget becomes a gateway rule; ledger carries no payer field                                                                      | G8             | Replay succeeds with no per-payer ledger field; duration-metering leak has a padding mitigation                                                                                                                                                                                    | **SEALED, PASS (29 Aug)**                                                                           |
| M373 | Restate "economic-only incentives" as no-KYC + time-cost pedigree                                                                           | G9             | The principle no longer contradicts pedigree, nexus, or behavioural identity                                                                                                                                                                                                       | **SEALED, PASS (29 Aug)**                                                                           |
| M374 | Weights-private verification stated in the Actors list; self-judging pool text deleted                                                      | G10            | No sentence names a contributor's own host as its executor pool                                                                                                                                                                                                                    | **SEALED, PASS (29 Aug)**                                                                           |
| M375 | **Measured chain cell**: DomainNet router → 6 specialists, end to end (substituted for Whisper→BERT; deviation registered before running)   | G11, feeds G12 | **SEALED, PASS** — anchor 0.245014 reproduced exactly; chain 0.2741 beats the monolith 0.2450 and the router-alone 0.0058; contract refuses `class_label[345]→domain_label[6]`; all four coalition values sealed; Shapley efficient to 1e-12. Assumption 3b measured, scope stated | HIGH                                                                                                |
| M376 | One attribution function; chain length capped; coalitions sealed at registration                                                            | G12            | **SEALED, PASS** — Shapley named as the single rule; `representation.py` delegates to `chains.shapley_split`; `MAX_CHAIN_STAGES = 4` enforced; the LOO/Shapley gap measured at 1.66x on M375's own coalition values                                                                | HIGH                                                                                                |
| M377 | Reporting precision stated as a split-size constraint; minimum registered                                                                   | G15            | Per-axis minimum split size registered and enforced                                                                                                                                                                                                                                | **SEALED, PASS**                                                                                    |
| M378 | Citation wave (19 entries) + prior-art sweeps re-run with expanded anchors                                                                  | G32–G50        | Sweep 4 done (two-index, all four anchor classes valid, no displacer, seven new citations, G43 confirmed); 28 `bibitem`s added; §Prior art, §Voting weight and the novelty paragraph rewritten; static check clean                                                                 | **SEALED, PASS**                                                                                    |
| M379 | Glossary + terminology unification                                                                                                          | R1–R4          | Every load-bearing term defined once, used consistently                                                                                                                                                                                                                            | **SEALED, PASS (29 Aug)**                                                                           |
| M380 | Results table legend, notation, $s_a$ definition                                                                                            | R9–R11         | Directions marked, ladders legended, one definition of $s_a$                                                                                                                                                                                                                       | **SEALED, PASS (29 Aug)**                                                                           |
| M381 | Structural edits: beacon ordering, serving-verification subheads, Known Limits ordering, appendix-legislates-nothing note, abstract reorder | R5–R8, R12     | Structural pass complete                                                                                                                                                                                                                                                           | **SEALED, PASS (29 Aug)**                                                                           |
| M382 | Librarian replacement gets an execution path that survives the developer's retirement                                                       | G53            | The freeze reproduces first; governance replaces the librarian with the owner renounced; governance holds that one power only and can hand itself on; the inbox reads the librarian live and keeps its open queue across the change                                                | **SEALED, PASS**                                                                                    |
| M383 | Incorporation becomes permissionless and the posting fee follows the work                                                                   | G54            | Anyone may incorporate; a poster clears its own entry unasked; a prompt librarian is paid and a stalled one loses the fee to whoever covered for it; the self-refund exploit stays closed                                                                                          | **SEALED, PASS**                                                                                    |
| M384 | `liftFreeze` on expiry becomes permissionless                                                                                               | G54 (A)        | A vanished librarian cannot extend a freeze past its own timestamp; only early release still needs a filing                                                                                                                                                                        | **NO DEFECT — verified, no code change**                                                            |
| M385 | `recordCredits` becomes a pull: payees claim against a published session-batch commitment                                                   | G54 (B)        | A payee is paid with the librarian absent; a withheld batch is provable; no payee depends on being pushed to                                                                                                                                                                       | **DONE — pull path built; root posting closed by R3-F1 (permissionless, bonded)**                   |
| M386 | `slash` becomes propose-and-challenge: anyone files with a bond, anyone refutes by replay                                                   | G54 (C)        | A guilty artifact is slashed with the librarian absent; a false accusation loses the bond; the replay still decides guilt                                                                                                                                                          | **DONE — challenge-windowed filings; disputed path needs the quorum's filer**                       |
| M387 | `setAdmitted` / `setDelisted` / `freezeArtifact` move to propose-and-challenge                                                              | G54 (C)        | Admission follows the published rule with no privileged filer; a ratified quorum verdict executes itself; a ministerial order executes on its own confirmation                                                                                                                     | **DONE — registry filings challenge-windowed; liftFreeze stays librarian-only (incentives invert)** |

---

## 9. Triage

**Tier 1 — publication blockers (the paper states something false).**
M350 (proof cost), M352 (premium tier contradiction), M367
("everywhere"), M355 (coverage inversion), M378 (citations + sweep
re-run), M377 (precision arithmetic). These are text-only except M378
and M367 and can land immediately.

> **Tier 1 is closed (28 Aug 2026).** All six sealed PASS. The
> estimate above was wrong in one respect worth recording: "text-only
> except M378 and M367" understated it. M350 needed a measured
> benchmark, M355 needed the sealed vision head refit and reproduced,
> and M352/M355 needed code. Four of six carried measurement.
> Tier 1 also produced three corrections to this review's own
> arithmetic (G2's $10^8$ multiple, G27's $P=8$, G15's $10^4$
> minimum) and one new finding (G51). A review's proposed repairs are
> unchecked claims like any other.

**Tier 2 — mechanism blockers (a named defense does not work).**
M349, M351, M353, M354, M358, M364, M365, M366, M372, M363.

**Tier 3 — economics and completeness.** M356, M357, M359, M360,
M361, M362, M368, M369, M371, M375, M376.

**Tier 4 — polish.** M370, M373, M374, M379, M380, M381.

> **Tiers 2–4 are closed (29 Aug 2026).** All eighteen remaining
> milestones sealed PASS or an honest alternative (M349 option 2,
> M363 sized-fee close, M370/M373/M374/M379–M381 text pass). The
> queue is empty.
>
> What the marathon produced, at a glance: two measured FHE verdicts
> (encrypted bucketing infeasible → disclosed-oracle tier at 2.8×;
> committed-seed flooding shipped), the operations line closed at a
> sized fee, the per-payer budget moved to the gateway with the
> duration meter padded, both crowding defenses measured, the
> abstention re-priced at its measured full cost (doubling the
> extraction bound), the voting escrow, the price drift band, the
> median-of-admitted reference cost, maintenance folded into paid
> roles, the executor-pool fallback published per artifact, the
> append-only corpus with its public depletion alarm, the required
> beacon composition, and the entire G54 librarian block (pull
> attribution, propose-and-challenge slash and registry changes).
> The paper ends with a glossary, a legended results table, a
> severity-ordered Known Limits, and an appendix that narrates but
> does not legislate.
>
> Three corrections to the review's own proposals were registered
> along the way rather than silently fixed: M384 was not a defect
> (checked before repairing), liftFreeze was excluded from
> permissionless release (its challenge incentives invert), and
> G2's magnitude and G27's floor had already been corrected in
> earlier tiers. The standing rule held throughout: a repair queued
> off a description repairs the description unless the premise is
> checked first.

**Suggested order.** Run M378's sweep re-run **first** — it is cheap,
it gates the novelty paragraph, and a displacing hit would change what
the rest of the plan is even for. Then the Tier-1 text pass as a
single commit. Then M353 and M354, which are small code changes with
large consequences and both have reproduce-the-failure gates. Then
M375, the only genuinely new experiment in the queue and the one that
decides whether assumption 3 survives.

> **M375 and M376 are closed (28 Aug 2026), ahead of the rest of
> Tier 2.** They were promoted because assumption 3 is the thesis's
> load-bearing claim and everything downstream of it was provisional
> until it was measured. It now is: the chain wins, narrowly and
> within a narrow scope, and the paper says so with its scope
> attached.
>
> Two lessons from the pair, both worth carrying into the remaining
> queue. First, **the registered cell could not run and the
> substitution was the right call** — Whisper and BERT are simply not
> in the local cache, and downloading two checkpoints to satisfy the
> letter of a design would have put an unsealed dependency at the
> centre of the one experiment that decides assumption 3. The
> router→specialist chain is the composition Figure 2 actually rests
> on. The substitution was registered, with its reason, before any
> number was read.
>
> Second, **M376 was rated MEDIUM on the assumption it was a drafting
> inconsistency, and it was not.** Once M375 produced real coalition
> values, the two rules split them 0.0588 against 0.0977 — a factor of
> 1.66 on the same measurement. A "notational" finding became a
> payment ambiguity the moment there were numbers to divide. Where a
> finding in this review is rated on the assumption that a
> contradiction is cosmetic, that rating is a hypothesis, not a
> reading.

> **M365 and M366 are closed (29 Aug 2026).** Both were taken
> together because they live in the same two files — the inbox
> contract and its Python mirror — and because G25's repair needed
> G13's verified-weight split, which M358 had already shipped.
>
> Three lessons, all from gates rather than from review. First,
> **a defect's neighbourhood is worth reading, not just the defect.**
> G24 named the refunded deposit; the same function also scanned an
> attacker-growable array to answer "is the chain valid", which is a
> second denial path nobody had registered. It was two lines from the
> one under repair.
>
> Second, **a correctly implemented control composed against the
> wrong denominator is not a control.** The 20% voting cap clipped
> exactly as specified and still let a 90% whale read as a two-thirds
> majority, because the share was taken against the capped total
> instead of the raw one. The registered semantic was already written
> down in another module's tests. Reading the primitive's own test
> settled it in one look.
>
> Third, **when a gate fails, establish whether the harness or the
> mechanism failed before touching either.** The spam-campaign test
> failed because 24 posts consume 24 blocks of a 10-block window, not
> because the incorporation cap was wrong. Widening the window was
> the right fix _for that reason_; widening it because the test was
> red would have been the wrong fix with the same diff.

> **M382 is closed (29 Aug 2026), and it was not in the queue.** G53
> was found while answering a reader's question — does the librarian
> rotate, how is it chosen, does it stake — not while reviewing the
> section that describes the librarian. That section had already been
> read several times.
>
> Three lessons, and the first is about how the finding surfaced.
> **A question about a role's lifecycle probes harder than a re-read
> of the text describing it.** The three answers are: it does not
> rotate, it is a developer key that becomes a governance contract,
> and it posts no stake. Each is defensible alone. Together they say
> the role's _only_ discipline is replacement — and that is what made
> it worth checking whether replacement could actually run. It could
> not. Reading clause by clause never composes the clauses.
>
> Second, **a green test asserting that something is frozen is only
> good news if freezing that thing was intended.** `credit_ledger`
> already had a passing test named "a renounced owner closes every
> admin path", asserting `setLibrarian` reverts. It was written as an
> admin-release guarantee. It is, unchanged, the proof of the lock-in.
> The suite was not silent about G53; it was asserting it. Ask what a
> passing invariant costs, not only what it buys.
>
> Third, **the same address held in two contracts is one of them being
> wrong.** The inbox is the mechanism that contains a misbehaving
> librarian, and it was the single contract that could never learn the
> librarian had been replaced. Where an authority appears in more than
> one place, exactly one should be the source; the rest should read it.
> This generalises past M382 and is worth a sweep of the other
> contracts.
>
> M382's remainder is registered above rather than waved through: the
> governance contract still does not exist, the deputy's successor
> order is prose, and the replacement path carries no timelock while a
> strictly smaller power does. A CRITICAL finding became executable,
> not finished.

---

## 10.5 Leftover completion — M388 (the M382 remainder, registered 29 Aug 2026)

The queue table above is empty because every registered cell sealed.
That does not make the review finished: M382's own writeup registered
three items it does **not** close, and each is a real gap between the
paper and the repo. They are the leftover work, registered here
before any build.

**Leftover 1 — the governance contract does not exist.** M382 created
the `governance` address slot on `CreditLedger` and proved the power
lands there, but the paper's "a governance contract with no human key
at maturity" still has no referent: `governance` is an EOA and the
retirement claim is only _possible_, not _true_.

**Leftover 2 — the deterministic deputy's successor order is
unspecified.** The launch plan says "deterministic successor order";
no code computes it. `librarian_containment.replacement()` returns
`fires: true` and names nobody.

**Leftover 3 — nothing times or timelocks the governance path.**
`GovernanceFloors` puts a 7-day timelock on raising a floor; replacing
the librarian — a strictly larger power — executes instantly.

### M388 — plan and gates (registered before the build)

Three closures, each with a gate.

1. **`LibrarianGovernance.sol` — a keyless governance executor.**
   A contract with no owner and no admin paths, holding exactly one
   power: replacing the librarian on `CreditLedger` (as the
   `governance`), and handing its own role on (succession). All state
   transitions are permissionless-and-recorded or time-gated. The
   recorded divergence reason is the mechanical trigger (mirroring
   the paper's "a recorded divergence reason is a fact a validator
   can replay, and no amount of weight substitutes for it"): a
   replacement with no recorded reason is inexpressible on-chain.
   Gate: the contract deploys with no owner; a stranger can propose
   and execute a replacement **only after the delay**; a replacement
   with a zero reason hash reverts; the proposer's bond is pulled,
   never pushed (a reverting recipient cannot block the path).

2. **Deterministic successor order + named deputy.**
   `librarian_containment.successor_order()` ranks candidate
   identities by `H(identity ‖ epoch ‖ anchor_hash)` ascending —
   replayable by anyone from the ledger, not knowable far enough
   ahead to arrange; `deputy()` names the first eligible identity
   that is not the incumbent; `replacement()` returns the named
   deputy when it fires. Gate: the order is deterministic and
   epoch-sensitive; the incumbent is excluded; the deputy is named
   in the replacement result; an empty roster names nobody without
   firing.

3. **Timelock on the governance path.** `REPLACEMENT_DELAY = 7 days`
   on every librarian replacement (matching `GovernanceFloors`'s
   `MIN_DELAY`), enforced by the governance contract itself.
   Decision registered: the instant path is the worse failure — a
   captured proposer could swap the librarian with no notice, and a
   slow-but-visible replacement lets the network see and respond
   during the window. Gate: execution before the delay reverts;
   execution after the delay lands.

**Registered residual, not improvised here.** The on-chain executor
cannot itself authenticate the off-chain two-thirds verdict: it
executes what is filed with a recorded reason, and whether the reason
is real is decided by the replay quorum off-chain. This is the same
quorum-authentication gap M386 (`resolveSlash`), M387
(`resolveRegistryChange`) and this milestone's own replacement-filing
each carry. (Root posting carried it too until R3-F1 made it
permissionless: a closed root now lands with no named party in the
path, and the filer is paid from the operations-line pool.) Closing it
needs an on-chain quorum oracle (threshold attestation or an on-chain
voter registry) and is a separate registered milestone, not part of
this closure. A fabricated reason is still a recorded, replay-visible
deviation against its filer, and the delay is the notice window.

### M388 — SEALED, PASS (29 Aug 2026)

All three leftovers are closed, each through its own gate.

1. **`LibrarianGovernance.sol` exists and is keyless.** A contract
   with no owner and no admin surface, holding exactly one power —
   replacing the librarian on `CreditLedger` (as the `governance`) —
   and handing its own role on (succession). All transitions are
   permissionless-and-recorded or time-gated. The paper's "a
   governance contract with no human key at maturity" now has a
   referent in this repo.
2. **The deterministic deputy is code, not prose.**
   `librarian_containment.successor_order()` ranks candidates by
   `H(identity ‖ epoch ‖ anchor_hash)`; `deputy()` names the first
   eligible non-incumbent; `replacement()` returns the named deputy
   when the caller supplies the roster (and reports honestly when
   none is supplied). The review's "replacement() fires and names
   nobody" is closed.
3. **The governance path is timelocked.** `REPLACEMENT_DELAY =
7 days` (matching `GovernanceFloors.MIN_DELAY`) on every
   replacement and succession; execution before the delay reverts.

Gates, all green: 10 new EVM tests (`librarian_governance.test.js`),
EVM suite **134 passing** (was 124); 13 new Python tests
(`test_v26_m312_librarian_containment.py`), Python suite **1142
passed, 1 skipped**.

The shared quorum-authentication residual (on-chain oracle for the
off-chain two-thirds verdict) remains registered above, untouched by
this closure — it is the same gap M386/M387 carry (M385's root-posting
leg was closed by R3-F1, which made the root permissionless and paid
the filer from the operations-line pool), and closing it is a separate
milestone with its own design.

**One design decision made and registered.** Leftover 3 asked
whether the instant path (a captured librarian removed fast) or the
timelocked path (a captured proposer cannot swap with no notice) is
right. Chosen: timelocked, 7 days. The instant path is the worse
failure — a replacement is the most powerful action the network can
take, and a slow-but-visible one lets the network see and respond
during the window. A captured _librarian_ does not lose by this: the
librarian cannot use the governance path at all (it is not the
governance executor), and its replacement runs on the recorded-reason
vote as before.

---

## 10. What this review does not re-open

- The M300/M300b nonlinearity result and its measured portability
  boundary (vision yes; text, audio, in-house-image-bridge no). Sealed
  and honestly bounded.
- The M341 fusion reading (concatenation works, alignment bridges are
  optional). Sealed on a clean cell, with the M228 confound recorded.
- The probe ordering repairs (commit-before-compare, beacon-postdated
  commits, structural self-exclusion, prefix immutability). These are
  correct and are what makes G23 the _only_ remaining probe-layer
  finding.
- The M332 extraction guard on the plaintext tier (55.2× vs 2.8×). The
  finding is that the private tier is outside it, not that it is wrong.
- The quorum takedown, excluded by request.
- The honesty discipline itself, which is the paper's strongest
  feature and the reason a review of this depth is possible at all:
  every finding above was reachable _from the paper's own text and
  numbers_.
