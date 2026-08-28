# v25 audit ladder — spec and API v0 (M177)

Registered 18 Aug 2026 in `RESEARCH_IMPLEMENTATION_PLAN_v25.md` §6.
Formalizes the existing sealed discipline (payload hashes,
`artifact_index.json`, `evidence.json`, anchor gates) into an audit API
with one method per ladder level. This document is the spec; the
implementation is `geode/audit.py` (v0: L0 + L1).

## The ladder (v25 §3)

- **L0 — deterministic replay.** Every decision replays bit-exact from
  its payload hash. API: `replay(runner, config_path, output_dir)`.
- **L1 — provenance chain.** data digest → code digest → weights
  digest → behavior digest, plus the fit that produced it. API:
  `provenance(artifact_dir)`.
- **L2 — component attribution.** which arms fired, which fingerprint
  matched, which marginal gain each arm contributed (the M151/M154
  ablation harness). v0 exposes `decision_fields(evidence)` as the
  input extractor; the full attribution method lands with M180.
- **L3 — behavior diffing.** every change is a new hashable artifact;
  before/after gates on a registered behavior suite.
- **L4 — capability mapping.** the fingerprint/task graph as a
  measured capability map (M178).

## API v0 contract

```python
class AuditAPI:
    def replay(self, runner, config_path, output_dir) -> ReplayReport
    def provenance(self, artifact_dir) -> ProvenanceReport
```

- `runner` is `callable(config_path, output_dir) -> evidence dict` (the
  milestone's own sealed runner function, run into a scratch directory).
- `ReplayReport`: `{evidence_hash, replayed_hash, bit_exact,
equal_excluding_timing, excluded_fields, diffs}`.
- `ProvenanceReport`: the chain object `{data, code, weights, behavior,
fit}` resolved from the artifact dir plus the registered gaps (which
  links are recorded as digests vs re-computed).

## Registered comparison rules

1. **Timing fields never enter a content hash** (the standing
   reproducibility-hash rule). The excluded set is
   `{"runtime_seconds"}` today; any future wall-clock field is added to
   the set BEFORE it is first compared, never after a mismatch.
2. `bit_exact` = content hashes equal. `equal_excluding_timing` = full
   evidence-dict equality with the excluded fields removed. A replay
   that is equal-excluding-timing but not content-hash-equal indicates
   a timing field escaped the exclusion set — reported, not silently
   passed.
3. Replays run into a scratch directory; sealed evidence is never
   overwritten.
4. The API is deterministic: no wall clocks in its own outputs.
5. **Both sides of a replay comparison are compared in their
   JSON-normalized shape** (in-memory dicts with int keys are a
   different object from their JSON round-trip; the stored evidence is
   what round-trips). Registered 18 Aug 2026 after the first replay
   exposed the shape mismatch on the A0 backoff histograms.

## H6 (audit completeness)

"100% of sampled paid-session decisions must replay bit-exact from
ledger hashes." For M177 the sample is the two registered milestones
(M175 A0, M175 C); the gate passes iff both `bit_exact` and
`equal_excluding_timing` hold for both.
