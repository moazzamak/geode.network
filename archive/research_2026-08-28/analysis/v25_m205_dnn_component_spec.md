# M205 — DNN-component admission contract

Registered 19 Aug 2026 (§4.13). A DNN component is a first-class but
OPTIONAL artifact. This is the registration contract the admission
validator (`geode/dnn_admission.py`) enforces; it implements §4.13
verbatim.

## Required submission fields

| Field                 | Meaning                                                     | Rule                                                                                                                                                                                                                                                                                                     |
| --------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `architecture_hash`   | hash of the architecture spec (layers, activations, shapes) | required, 40–128 hex chars                                                                                                                                                                                                                                                                               |
| `seed_hash`           | hash of the deterministic initializer (seed + init scheme)  | required (deterministic INIT contract)                                                                                                                                                                                                                                                                   |
| `data_digest`         | hash of the training-set manifest                           | required                                                                                                                                                                                                                                                                                                 |
| `software_hash`       | hash of the training software version                       | required                                                                                                                                                                                                                                                                                                 |
| `weights_hash`        | hash of the final weights file                              | required                                                                                                                                                                                                                                                                                                 |
| `training_log_digest` | hash of the training log                                    | required, audited not replayed                                                                                                                                                                                                                                                                           |
| `eval_report`         | `{split, n_test, accuracy}`                                 | `n_test >= MIN_TEST`; `0 <= accuracy <= 1`; split must be a declared held-out split; **accuracy must exceed the chance floor (1/classes + 1e-6)** — a chance-level artifact carries no signal (registered 19 Aug 2026, before the M206 re-run; the rule applies to ALL submissions, not only this probe) |

## Registered rules

1. **The registry verifies, never trains.** Admission checks hashes,
   evaluation plausibility, and the reproducibility contract — it does
   not run training.
2. **Deterministic initialization.** `seed_hash` present, else reject
   (the reproducibility contract of §4.13 item 3).
3. **Bit-exact inference replay.** The artifact's forward pass must be
   deterministic given `weights_hash` + `architecture_hash`; the
   validator records the replay hash that H6 will use on every paid
   session.
4. **Duplicate collapse.** A `data_digest` + `architecture_hash` pair
   already admitted ⇒ the new submission is a duplicate and earns no
   attribution (the M199 Sybil rule).
5. **Honest limits recorded with the artifact:** DNN training is not
   bit-reproducible; the training log is audited, not replayed; the
   component is less inspectable than a closed-form head.
6. **Evaluation is admission-floor, not ranking.** Ranking comes from
   the M151/M180 coalition machinery on the cached codes — admission
   only rejects invalid or implausible submissions.

## Gates

- All malformed submissions rejected with a reason.
- A complete, plausible submission admitted and its replay hash
  recorded.
- A duplicate submission rejected as duplicate.
