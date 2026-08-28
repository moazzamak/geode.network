# V7 Final Claim Ledger

## Disposition

V7 ends at **Outcome C: stage-wise lifecycle qualification**. M43 did not
produce a passing integrated winner, so M44 remained sealed and was not run.
No closed operational-loop, authoritative sparse-routing, general
existing-class expansion, or primary SDF claim is supported.

## Frozen branch ledger

| Branch | Disposition | Frozen reason |
|---|---|---|
| M38 prior-art displacement | Not triggered | No verified system covered all seven registered stages |
| M39 low-rank Gaussian rejection | Passed | 92.08% known coverage, 86.17% unknown recall, 98.67% review precision |
| M39 weighted-affine SDF | Failed | 64.67% review precision and a seed-level autonomy failure |
| M39 kNN/posterior controls | Failed retention | At least one autonomy or best-control precision operand failed |
| M39 EVM-style/RBF controls | Failed rejection | Unknown recall was 0.00% and 12.83% respectively |
| M40 HDBSCAN | Passed primary | 100% distinct-group recall and review precision; 9/9 cells |
| M40 FINCH | Passed control | 83.33% recall, 86.39% precision; 8/9 cells |
| M40 streaming micro-clusters | Failed discovery | No distinct withheld group was recovered |
| M41 existing-class expansion | Closed | Passed only seed 23; mean improvement 4.17 points |
| M41 confirmed new-class insertion | Passed stage gate | Mean target gain 43.33 points with exact rollback |
| M42 authoritative routing | Closed | Best top-1 93.53%, winner inclusion 99.57%, unknown recall 71.00% |
| M43 HDBSCAN/FINCH integrated loop | Failed | 0/3 confirmable classes integrated |
| M43 reject-everything control | Failed burden gate | Integrated 3/3 only with zero review reduction |
| M44 independent confirmation | Blocked | No passing M43 winner existed to freeze |

## Supported claims

1. A frozen low-rank Gaussian can produce useful review candidates on the
   registered leave-two-class-out DINOv2 proxy.
2. HDBSCAN and FINCH can turn those rejections into persistent, replay-stable
   review objects under the registered schedules.
3. Human-confirmed new-class insertion can be transactional and exactly
   reversible in isolation.
4. Empirical routing profiles can reduce exact-model evaluations in shadow
   mode, but do not meet authoritative routing safety.
5. The independently passing stages did not compose into a review-efficient
   integrated loop.

## Reproduction

The final verifier reads only frozen JSON artifacts and indexes:

```powershell
& '.\.venv\Scripts\python.exe' -m experiments.tier1.eval_v7_final_replay
```

It does not load training data or open sealed labels.
