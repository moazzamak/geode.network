# Archive — research phase (superseded 17 Aug 2026)

Everything here was moved out of the live tree when the repository
transitioned from research to development. Nothing was deleted: the files
below are preserved as-is, and their full history (including pre-move
paths) remains in git.

## Layout

```
analysis/               # historical plans v5–v22, claim ledgers, reports,
                        #   acceptance criteria, theses/paper builds
configs/                # legacy configs (flat files + v5–v15 dirs)
experiments/
  tier1/ tier2/ tier3/  # legacy experiment tiers (geometric system era)
  tier5/ tier6/ e2e/
  tier4_legacy/         # v5–v14 and unnumbered tier4 runners, prepare/
                        #   summarize/verify/audit helpers, rank_probes,
                        #   legacy ONNX backbones
  common_legacy/        # legacy common modules and tests (v5–v15, v20,
                        #   research-infrastructure tests)
infrastructure/e7/      # legacy distributed-execution rehearsal
logs/                   # pre-v16 results, E-phase registries, chat logs
tools/                  # legacy tooling (probes, benchmarks, summarisers)
verify_pipeline.py      # the legacy geometric-system verification pipeline
README_legacy.md        # the pre-cleanup root README (geometric-system era)
```

## Why each area was archived

- **analysis/**: superseded by the active plans v23–v25, the GEODE
  whitepaper, and the buildout-blocker literature search. The claim
  ledgers' conclusions are carried forward in the whitepaper §8.
- **experiments/tier1–3, 5–6, e2e + tier4_legacy**: the geometric
  ellipsoid system and its experiments; the promoted system lives in
  `experiments/tier4` (v15/v16/v23 runners) and no longer references
  these tiers.
- **common_legacy**: modules and tests that only the archived runners
  used. The kept `experiments/common/` modules (`data_cache`,
  `v5_artifacts`, `v5_protocol`, `experiment_manifest`, `litsearch_cache`,
  plus `test_v16_*`/`test_v23_*`) were verified to be the full import
  closure of the active line.
- **configs/**: configs for archived runners. Active configs are
  `experiments/configs/v16/` and `v23/`.
- **logs/**: evidence for superseded milestones. The whitepaper and the
  active plans cite only `logs/results/v16/` and `logs/results/v23/`,
  which stay live.
- **src/ is NOT archived**: the legacy geometric core and the
  programmatic primitives (`src/programmatic_primitive.py`,
  `src/programmatic_memory.py`, `src/contract_router.py`,
  `src/model_fingerprint.py`) are still imported by the sealed v16
  builders (M130/M131) and are part of the reproduction surface.

## If you need something

- Re-run the sealed evidence: `logs/results/v16/` and `v23/` remain in
  place.
- Old numbers and reasoning: find them here; git history retains the
  original paths via rename detection.
- Anything that should return to the live tree: `git mv` it back and
  register the change in the active plan's execution log.
