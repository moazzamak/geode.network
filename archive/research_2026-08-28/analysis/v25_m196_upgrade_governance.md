# M196 — protocol upgrade & fork governance (registered design)

Registered 18 Aug 2026 in `RESEARCH_IMPLEMENTATION_PLAN_v25.md` §6.
Spec only. "Protocol" here = the registry rules, vesting parameters,
selection scores, and the value function — anything whose change is NOT
a per-component append but a rule change. Registered before any token
mints.

**Registered rules:**

- **Append-only is the default for content; upgrades are the only
  rule change.** New components, data, and measurements are appends
  (M185 ledger). A rule change is an upgrade, and an upgrade is a
  versioned protocol artifact with a payload hash.
- **Fork policy:** no contentious hard forks. An upgrade ships only
  with (a) the measured rationale citing sealed evidence, (b) a
  replay-verified before/after behavior suite (M177 L0/L3), (c) a
  registered deprecation window for the old rule. Anything less is a
  fork, and forks are the failure mode, not the mechanism.
- **The fixed points:** the 2.5% dev-fund route AND its end-state
  purpose — the zakat rule (one-fortieth of every fee to those who
  need it most; a general basic income in the best case), fixed
  24 Aug 2026 — the measurement-only thaw invariant (I1), and the
  append-only attribution record (I3) are NOT upgradeable by ordinary
  governance — changing any of them is a protocol-level event with
  the jurisdiction gate attached.
- **Rollback:** every upgrade ships with its rollback bundle (the
  M176e pattern: re-encode, re-fit, re-anchor, solver-equivalence
  gate, incumbent router gate).

**Not in this spec:** the on-chain enactment mechanics (M187's chain
decision) and the legal classification of upgrade events (M188).
