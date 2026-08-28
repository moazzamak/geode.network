# GEODE v8 Claim Ledger

**Program:** Adaptation Utility as the Registered Endpoint  
**Status:** final Outcome D; all v8 milestones dispositioned
**Registration date:** 27 July 2026  
**Parent evidence:** `analysis/V7_FINAL_CLAIM_LEDGER.md` and
`logs/results/v7/final_stagewise_replay/`

## Immutable parent outcomes

V8 does not revise either closed parent result:

1. **v6.1 Outcome D is final.** The weighted affine head reached 91.73%
   development balanced accuracy versus 96.77% for the same-space RBF control.
   The 5.03-point deficit exceeded the registered 2-point parity operand.
2. **v7 Outcome C is final.** Rejection, discovery, and transactional
   new-class insertion passed separately, but HDBSCAN and FINCH integrated 0/3
   confirmable classes. Reviewing every rejection integrated 3/3 only with no
   review reduction. M44 remained sealed.

V8 is not a predictive-parity rescue, a reopening of v7 confirmation, or an
attempt to reinterpret stage-wise metrics as end-to-end success.

## New claim under test

> At an equal fixed human-review budget, a discovery-to-review-to-adaptation
> loop selected for downstream adaptation utility improves post-integration
> balanced accuracy on known plus confirmed classes relative to the frozen
> stage-wise-qualified v7 composition, while limiting known-class regression
> and preserving every transactional safety gate.

This is a systems-composition claim. It remains valid if a non-SDF acceptance
or support mechanism wins the registered head ablation.

## Primary endpoint

For episode \(e\), let:

- \(K_e\) be the classes known before the episode;
- \(D_e\) be the classes confirmed and eligible for integration during the
  episode;
- \(B=50\) be the maximum number of human-labeled samples available in that
  episode;
- \(A_e^{\text{child}}\) be balanced accuracy on the frozen evaluation
  partition over \(K_e \cup D_e\) after the transaction;
- \(A_e^{\text{parent}}\) be balanced accuracy over the same labels and
  examples before adaptation, with rejected/unavailable discovered labels
  scored as incorrect.

The registered episode utility is

\[
U_e = A_e^{\text{child}} - A_e^{\text{parent}}.
\]

The cumulative endpoint after \(N\) episodes is

\[
U_{1:N} =
A_{K_N \cup D_{1:N}}^{\text{final}}
- A_{K_N \cup D_{1:N}}^{\text{frozen-parent}}.
\]

All methods receive the same episode order, representation, review budget,
confirmation oracle, evaluation examples, and update opportunity. Unused review
budget is not transferable across episodes.

## Safety co-primary and invariants

A utility result is ineligible unless all of the following hold:

- known-class balanced-accuracy regression is at most
  \(\epsilon=1.0\) percentage point per episode and cumulatively;
- no semantic class or mutation is published without linked confirmation;
- no false autonomous class creation occurs;
- every published transaction has zero graph-validation issues;
- replay is byte-identical;
- rollback restores the exact parent bundle and predictions;
- stale or uncertain routing falls back to exhaustive compatible-model
  evaluation;
- final-test and untouched confirmation labels remain sealed until the
  registered final gate.

AUROC, rejection coverage, cluster purity, persistence, routing top-1, and
latency are diagnostics. None can advance a branch without a passing utility
endpoint.

## Frozen baselines

| Baseline | Frozen disposition | V8 role |
|---|---|---|
| v7 Gaussian + HDBSCAN + confirmed rank-16 insertion | Stage-wise qualified; 0/3 integrated in M43 | Primary stage-wise-composition baseline |
| v7 Gaussian + FINCH + confirmed rank-16 insertion | 93.49% review reduction; 0/3 integrated | Distinct core-selection baseline |
| Review every rejection | 3/3 integrated; zero review reduction | Upper-support/lower-efficiency diagnostic |
| No adaptation | No child transaction | Utility-zero control |
| Full class-local refit | Failed v7 retention | Update-work control |
| Weighted affine/GEODE head | Failed v7 rejection retention | Honest head ablation, not default |
| Low-rank Gaussian/DDU-style head | Won v7 rejection | Density-head control |
| kNN-support hybrid | Useful v7 rejection signal; failed retention | Non-parametric support control |
| Exhaustive routing | Authoritative | Safety oracle and default |

## Registered hypotheses

### H1: threshold transfer

Recalibrating acceptance thresholds on a frozen anchor set after each class-order
change preserves remaining-unknown recall within 2.0 points of the
pre-integration episode value without violating the 1-point known-regression
ceiling.

### H2: review-set sufficient statistics

At equal \(B\), boundary-inclusive or utility-selected review sets produce
higher \(U_e\) than density-core selection because they better cover
within-class modes and decision-boundary variation.

### H3: learned utility selection

If H2 passes, a lightweight cross-episode selection scorer can predict
integration utility without consuming evaluation labels in the target episode
and can retain the Phase-2 utility gain.

### H4: localized residuals

An evidence-triggered local residual can improve \(U_e\) while preserving at
least 99.9% of predictions outside the registered affected region. This is an
adaptation mechanism, not a parity-rescue claim.

### H5: end-to-end qualification

The jointly qualified loop beats the frozen v7 stage-wise composition on
episode and cumulative utility at equal review budget, with the safety
co-primary intact.

## Claim restrictions

V8 may not claim:

- closed-set parity or rescue of v6.1 Outcome D;
- that purity, AUROC, coverage, or routing accuracy alone is success;
- that an unlabeled group is a semantic class;
- that an SDF is necessary for open-world recognition;
- autonomous class creation;
- authoritative sparse routing before the integrated E12 gate;
- locality from rollback alone;
- a learned selector advantage if target-episode evaluation labels influenced
  its selection;
- independent confirmation unless M50 opens and passes the untouched schedule.

## Outcome taxonomy

- **Outcome A — utility-qualified lifecycle:** M50 passes with a positive,
  non-dominated utility gain at equal budget and all safety gates.
- **Outcome B — utility gain under exhaustive routing:** M50 passes while
  routing remains exhaustive or shadow-only.
- **Outcome C — selection signal only:** M47 passes, but learned selection,
  localization, or E12 does not.
- **Outcome D — statistic-mismatch negative:** M47 finds no utility advantage
  for boundary-inclusive or utility-selected review.
- **Outcome E — locality blocked:** utility selection passes, but the optional
  residual branch fails the locality gate twice; the main loop continues
  without residuals.
- **Outcome F — transactional failure:** any publication, graph, replay,
  rollback, or confirmation invariant fails; publication is prohibited.

## Advancement ledger

| Milestone | Decision | Status |
|---|---|---|
| M45 registration and contracts | Freeze endpoint, episodes, interfaces, parents, and schemas | Passed: 3 episodes, 5 interfaces, 7 schemas, 6 fail-closed cases, byte-identical replay |
| M46 threshold/statistic diagnostics | Qualify threshold transfer and rank interface mismatches | Passed: global anchor quantile retained; 6 M47 features frozen |
| M47 utility-driven review | Test core, boundary-inclusive, and utility-selected sets at equal budget | Failed: +3.593 pp vs core, below +5.0 pp; safety conjunction failed |
| M48 learned selection | Train only if M47 utility signal passes | Blocked by M47 Outcome D |
| M49 localized residuals | Parallel two-attempt locality branch | Closed: locality passed at 100%, utility/safety failed; no residual retained |
| M50 / E12 lifecycle qualification | One frozen end-to-end run | Blocked by M47 Outcome D |

The terminal disposition and artifact-only reproduction command are frozen in
`analysis/V8_FINAL_CLAIM_LEDGER.md`.
