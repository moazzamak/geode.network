# M176e — Live upgrade / migration protocol (registered spec, v0)

Frozen 2026-08-17. Freeze-on-ship (I1) must not mean frozen-forever.
When a better encoder/arm/provider ships, this is the registered
migration path.

## Rules (frozen)

1. **Versioned artifacts only**: a new component ships with a new
   version and a migration report in the legacy `model_migration`
   format (schema_version, source/target representation hashes,
   component correspondence, edit survival, invalidated artifacts,
   rollback bundle hash). Reuse the existing
   `validate_migration_report` checks verbatim.
2. **Re-encode → re-fit → re-anchor, in that order**: existing tasks
   re-encode with the new component, refit their heads, and re-run the
   sealed anchors (ridge anchor 0.2273623188405797, tol 1e-6; the G1-G3
   gate set; the M170 traversability set). Nothing is declared until
   the anchors pass.
3. **Equivalence gate for solver changes**: any altered closed-form
   solver must reproduce the incumbent within 1e-9 on a registered
   reference problem before it may touch real fits (the M176b lesson:
   LSQR failed this gate and stayed out).
4. **Rollback path**: the rollback bundle (old weights + old registry
   hashes) is retained until the new version has served a full anchor
   pass; rollback is one hash comparison away.
5. **Who pays**: the party proposing the upgrade pays the re-encoding
   and re-anchoring cost (the v25 accounting rule's precedent).
6. **Router stays the incumbent**: a learned router may only replace
   the deterministic nearest-arm router behind a measured gate — routed
   accuracy with eps-advance must be ≥ the frozen router's on the same
   registry, on held-out rows (M143b's incumbent discipline).

## Cross-run fingerprint caveat (measured 17 Aug)

GPU training of the fingerprint encoder is non-deterministic ACROSS
processes (M172 measured: two identical trainings give cos 0.01-0.44).
Therefore every shipped encoder MUST persist its trained weights as a
frozen artifact (state_dict + payload hash + config hash) before any
cross-run use. The eval path is deterministic (G1); only the training
is not.
