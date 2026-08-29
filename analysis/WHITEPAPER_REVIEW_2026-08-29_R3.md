# WHITEPAPER REVIEW R3 — FRESH PASS (29 Aug 2026)

**Scope:** `docs/WHITEPAPER_GEODE.tex` at the post-R2/post-writing-pass
state, fresh outside-reader pass. This pass focuses on what the R2
queue and the writing pass may have left behind: paper-code
divergences introduced by the final-MVP rewrite, appendix-vs-body
consistency after the beacon change (M371/M354), and remaining
wording overreaches.

Method: read the paper standalone, then verify each load-bearing
claim against the repo (code + sealed evidence). Where a finding is
a divergence, the fix is chosen by which side carries the intended
design — never silently.

## 1. Findings

| ID   | Severity | Finding | Fix |
| ---- | -------- | ------- | --- |
| F1   | HIGH     | §Settlement batching says "any party may publish the commitment under a bond, with a challenge decided by the same replay quorum that decides guilt elsewhere". Code: `CreditLedger.postAttributionRoot` is `onlyLibrarian` — the M385 residual, deliberately not closed (it needs the on-chain quorum oracle the review parked). The writing pass turned that registered residual into a false deployed claim. | Reword the paragraph to the deployed MVP: the librarian posts the root; a stopped librarian halts the epoch's income and the halt is a public zero that feeds the liveness statistics and the recorded-reason replacement discipline. Do not claim the bonded any-party path. |
| F2   | HIGH     | §Router says the lottery draw "is seeded by the randomness beacon"; `geode/core/router_repair.py` `draw_seed` seeds from the anchor. The paper (router, ledger sampling list, known-limits beacon dependency) uniformly requires beacon-seeded routing — it is the registered closure of the route-grinding residual. The writing pass made the paper claim the closure as deployed; the code has not implemented it. | Implement the beacon in `draw_seed`/`route` (beacon + anchor-for-ordering), update the M303/M354 tests and the M303 harness, re-run the share sweep to confirm the published traffic shares hold under the beacon seed. |
| F3   | MEDIUM    | Appendix "A probed session, honest path" and "Probe-dodging attempts" say the probe flag and the executor sample are drawn from the epoch anchor; the body (§Serving verification, M371) uses the randomness beacon. Internal inconsistency left by the beacon change. | Align the appendix to the beacon. |
| F4   | LOW      | "No cost claim appears anywhere in this paper that is not in this table" overreaches: probe overhead (10%/60%), the FHE head-path cost (~20 s/query), the five-hundred-second bound, and the operations line ($1.23/epoch) are cost claims outside the proof table. | Narrow to proof costs. |
| F5   | LOW      | The superlinear posting-fee multiple is measured at 330,839× (M365) but written "about three hundred thousand times" — 10% low. | Tighten to "about three hundred thirty thousand times". |
| F6   | LOW      | "See §Serving verification" is an unnumbered section reference; and "five hundred seconds against a hundred-second bound --- five times the head-only figure" reads ambiguously (the 5× describes the bound, not the 500 s). | Use the numbered reference; reword the bound sentence. |
| F7   | MEDIUM    | The Actors list says the librarian "at maturity is a governance contract with no human key"; the implemented keyless contract (M388 `LibrarianGovernance`) is the governance *executor* that replaces the librarian, not the librarian itself. The librarian is an operator key throughout the MVP. | Reword the bullet: the librarian is an operator key; a keyless governance executor names and replaces it. |
| F8   | HIGH      | §Serving verification (device readout) says the private tier's score-vector oracle needs "$d\cdot C$ queries recover the head exactly" and prices the tier at "the score-vector-oracle figure --- $2.8\times$". Both are the RAW-MARGIN figures (M332/M357: d·C queries, 2.8×). The score vector returns all C scores per query for a code the device computed itself, so ~d queries recover W — a factor of ~C below the raw margin. The private tier's model confidentiality is not an economic boundary at 2.8×; it is recoverable in roughly d queries. Inherited from R2's G1/M349, which reused the M332 raw-margin number for the stronger oracle. | Rewrite the paragraph: the score vector recovers W in about d queries (derived, linear algebra); the tier's claim is input privacy, not model confidentiality; drop the false 2.8× pricing basis. |
| F9   | MEDIUM    | Appendix "A wash ring tries and fails" says "Every round-trip pays the 2.5% dock twice, 2 × 0.025 = 0.05 of the looped amount" — 5%. The sealed M358 evidence measures the ring's haircut at 2.5% (`haircut: 0.025`; the gate's registered "loses 5%" was corrected DOWN during the measurement). The appendix uses the pre-correction estimate. | Correct the appendix to the measured 2.5% dev-fund dock, and state the post-repair zero-weight result (M358). |

## 2. Verified and closed (not findings)

- FHE head-only cost (~20 s, evidence 23.0 s) and the five-hundred-second
  private-query figure (evidence 504.7 s) — both in `m349_encrypted_bucketing.json`.
- Crowding delivered accuracy 0.600 / 0.701 (evidence 0.6 / 0.7009, `m356`).
- Route-grinding 3.1 declarations (`m354`).
- Security floors, chain-length cap 4, Shapley 1.66×, N=4 vesting,
  probe floor 30, fee multiple 2.5× — all match code.
- Beacon terminology / ledger / known-limits internal consistency (the
  body is uniform after the writing pass; only the appendix lags — F3).

## 3. Fix order

F2 first (code change, gates the paper's claim), then F1 (paper),
then F3 (paper appendix), then F4–F7 (wording). Re-run the Python
suite, the EVM suite and `tools/check_whitepaper_tex.py` after each
group; commit when green.

## 4. Sealed (29 Aug 2026)

- **F2 SEALED.** `geode/core/router_repair.py` `draw_seed`/`route`
  now require the beacon: the seed is
  `H(beacon, anchor, task, state_root, fp, session_id)` — the beacon
  output closes after declaration, so the draw is not grindable
  against the public anchor. The M303/M354 tests were updated and two
  beacon-dependence tests added; the M354 tool and the M303 harness
  were brought to the new signature (sealed evidence untouched). The
  route-grinding residual's closure is now code, not prose.
- **F1 SEALED.** The settlement paragraph no longer claims "any party
  may publish the commitment under a bond" (which was the M385
  residual, not implemented). It now states the deployed MVP: the
  librarian posts the write-once root; a stopped librarian halts the
  epoch's income and the absent root is a recorded divergence feeding
  the replacement discipline.
- **F3 SEALED.** The appendix's "A probed session, honest path" and
  "Probe-dodging attempts" now use the randomness beacon (ordered by
  the epoch anchor) for the probe flag and the executor sample,
  matching the body.
- **F4–F7 SEALED (wording).** Proof-cost claim narrowed; posting-fee
  multiple tightened to "about three hundred and thirty thousand
  times"; `§Serving verification` is now a numbered reference
  (`sec:serving`); the five-hundred-second bound sentence clarified;
  both "governance contract with no human key" statements reworded to
  name the keyless governance executor as the referent (the librarian
  is an operator key).
- **F8 SEALED.** §Serving verification (device readout) no longer
  prices the private tier at the raw-margin 2.8× figure. It now
  states the derived fact: the score-vector oracle returns all $C$
  scores per query for a code the device computed itself, so about
  $d$ queries recover the head (a factor of ~$C$ below the raw
  margin's $d\cdot C$); the private tier's claim is input privacy,
  not model confidentiality. The known-limits model-extraction item
  gains the same caveat. (This error survived R1/R2 and M349's
  option-2 consequence, which reused the M332 raw-margin number for
  the strictly stronger oracle.)
- **F9 SEALED.** The wash-ring appendix no longer claims "2 × 0.025 =
  0.05 of the looped amount" (the pre-correction ~5% estimate). It
  now states the measured 2.5% dev-fund dock and the post-repair
  zero-weight result (M358: `haircut: 0.025`).

Gates: Python **1144 passed / 1 skipped** (was 1142; +2 beacon
tests); EVM **134 passing** (unchanged); `check_whitepaper_tex.py`
**PASS** (1048/1048 braces, 17 labels, 14 refs, 58 bibitems).
