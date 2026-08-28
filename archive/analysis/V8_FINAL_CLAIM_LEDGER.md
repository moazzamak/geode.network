# V8 Final Claim Ledger

**Program:** Adaptation Utility as the Registered Endpoint
**Final outcome:** D — statistic-mismatch negative
**Final-label status:** sealed

## Frozen conclusion

V8 does not qualify the jointly optimized discovery-to-adaptation lifecycle.
At 50 reviewed labels per episode, deterministic utility selection beat
density-core selection in all nine development cells by 3.593 balanced-accuracy
points on average, with paired 95% bootstrap interval [2.794, 4.382]. This was
below the preregistered 5.0-point minimum, and six cells exceeded the 2.0-point
remaining-unknown-recall band.

The result supports a narrower diagnostic conclusion: core sets are
geometrically unrepresentative, and improving support coverage increases
adaptation utility, but the measured gain is insufficient and does not preserve
the full open-world safety contract.

## Milestone disposition

| Milestone | Final disposition |
|---|---|
| M45 | Passed: endpoint, interfaces, replay, and six fail-closed cases qualified |
| M46 | Passed: global anchor-quantile transfer retained; six selector features frozen |
| M47 | Failed: +3.593 points versus core, below +5.0; safety conjunction failed |
| M48 | Blocked by M47 |
| M49 | Closed: 100% locality, at most +0.204 points utility, no residual retained |
| M50 / E12 | Blocked by M47 |

## Claim boundary

V8 may claim that:

- adaptation utility is a more discriminating endpoint than stage purity;
- boundary/coverage-aware review improved utility over density-core review in
  this frozen development harness;
- global anchor-quantile recalibration transferred safely in the M46 episodes;
- explicit parent-only residual fusion repaired A3 scope leakage.

V8 may not claim:

- end-to-end lifecycle qualification;
- a learned selector advantage;
- an untouched E12 confirmation;
- an authoritative sparse router;
- head-substitution superiority;
- closed-set parity or rescue of v6.1 Outcome D.

## Artifact-only verification

Run:

```powershell
& '.\.venv\Scripts\python.exe' -m experiments.tier1.eval_v8_final_replay
```

The command verifies the immutable M45, M46, M47, and M49 indexes and all
indexed files, reproduces six conclusion operands twice, and requires
byte-identical output without loading training data or opening final labels.
