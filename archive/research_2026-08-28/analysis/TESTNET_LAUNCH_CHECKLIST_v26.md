# TESTNET LAUNCH CHECKLIST (v26, 27 Aug 2026)

Purpose: the single readiness ledger for a Sepolia/Arbitrum-Sepolia
launch. Every item carries its state and the artifact that decides
it. Nothing here is asserted; each item points at evidence or is
marked NOT BUILT. The developer is not launched from this document;
the user decides when to launch.

States: SHIPPED (built + gates recorded), PARTIAL (built, known
gaps), NOT BUILT (registered, no code), BLOCKED (registered,
awaiting a decision).

## 1. Protocol core (Python)

| Item                                                                                                                  | State     | Evidence / note                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --------------------------------------------------------------------------------------------------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Registry, router (M303 lottery), admission, challenge sessions, shadow probe                                          | SHIPPED   | geode/core/\*, v24/v25 milestones; H26-7 PASS                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Economic repairs (M313 verified-work accrual, self-payment exclusion, per-axis bonds)                                 | SHIPPED   | tests 18/18                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Takedown containment (M315: pool-scaled quorum, appeals, suspension-before-permanence, revenue-scaled deposits)       | SHIPPED   | tests 15/15, M315 PASS                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Behavioural identity dedup (M307), drawn challenges (M308), selective-abort (M319), floors (M314)                     | SHIPPED   | tests green                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Vote machinery (M328: diversity floor, secret-ballot Pedersen tally, weight snapshot)                                 | SHIPPED   | geode/privacy/vote_machinery.py, 14/14, 27 Aug                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Private serving head (M322e: FHE head, CKKS backend)                                                                  | SHIPPED   | QG1/QG1b/QG2 (quantization) and QG3a/QG3b (CKKS noise) PASS on both real heads; G5 measured ~23 s/query, ~1.7 MB                                                                                                                                                                                                                                                                                                                                                                       |
| Private serving gateway wiring (M322e-D: device encrypt / host evaluate / device decrypt; ciphertext-only transcript) | SHIPPED   | `geode/privacy/fhe_gateway.py`, 27 Aug: the transcript type has no plaintext field (structural); tier auditability wired through `serving_tiers`                                                                                                                                                                                                                                                                                                                                       |
| FHE-path probe (M330: ciphertext-commit + ciphertext-replay adjudication)                                             | SHIPPED   | `geode/privacy/fhe_probe.py`, 28 Aug, 9/9: the M319 table mapped to the ciphertext form; the executor transcript holds no plaintext field                                                                                                                                                                                                                                                                                                                                              |
| M322 premium trunk path (M322d FHE trunk)                                                                             | NOT BUILT | registered; blocked on the same HE toolchain choice, now resolved to TenSEAL/CKKS                                                                                                                                                                                                                                                                                                                                                                                                      |
| Replay oracle (M306 canonical pinned CPU/float64 oracle + cross-hardware audit)                                       | SHIPPED   | `geode/core/replay_oracle.py` 12/12 + runner, 27 Aug. MEASURED: G1 bit-exact reproduction of both sealed heads PASS (456 s); G2 cross-configuration digests FAIL — bit-exactness does not survive a thread change, so the margin-gated probe (R-A6d, M305) is the operative rule and the whitepaper states the measured result                                                                                                                                                         |
| Eval-custody repair (M309 rows never leave the sealed scoring environment)                                            | SHIPPED   | `geode/core/eval_custody.py`, 8/8, 27 Aug                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Chains as first-class artifacts (M316)                                                                                | SHIPPED   | `geode/core/chains.py`, 9/9, 27 Aug: Shapley split + first-class chain admission                                                                                                                                                                                                                                                                                                                                                                                                       |
| Proof-layer honesty (M318: Pedersen registry key or publish W)                                                        | PARTIAL   | R-A15a shipped (`geode/privacy/head_commitment.py`, 5/5); R-A15b alternative pending the user decision                                                                                                                                                                                                                                                                                                                                                                                 |
| Versioned feature bus (M320) + alignment module (M301)                                                                | SHIPPED   | `geode/core/feature_bus.py` + `geode/core/alignment.py` (incl. `cca_from_moments` streaming path), 12/12, 27 Aug. H26-4 MEASURED (§8.40): anchors reproduced, aligned CCA 0.0133 / concatenated 0.1943 both below the single encoder — the shared-space thesis unsupported at this scale; a native-resolution cell is the registered precondition for re-evaluation — RAN (M341, 28 Aug): concat 0.5479 vs ms-only 0.2421, CCA 0.5106 loses to concat — fusion works, bridges optional |
| Composite campaign harness (M321) + coverage-adjusted metric                                                          | SHIPPED   | `geode/core/composite_campaign.py` + `geode/core/coverage_adjusted.py`, 8/8, 27 Aug: 11 rows, every closure attributed; H26-9 now MEASURED (§8.39): the coverage-adjusted ranking inverts on the sealed M286 numbers; R-A7b stays a pending measurement                                                                                                                                                                                                                                |
| M297b grid extension {50,100,300,1000}                                                                                | SHIPPED   | 27 Aug, full run: LOOCV interior minimum at lambda*=300 (0.0028089) closes the M297 boundary flag; the sealed test at lambda* reads 0.22110 (below the anchor), so LOOCV lambda-selection is measured NOT to be a head repair on this axis                                                                                                                                                                                                                                             |

## 2. Governance / compliance machinery

| Item                                                                 | State   | Evidence / note                                                                                                                                                                                      |
| -------------------------------------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| Content report intake + ministerial freeze (M323 family)             | SHIPPED | module (`geode/core/content_orders.py`, 13/13), contract freeze machinery (CreditLedger, 5/5), and the settlement bridge (`geode/core/settlement_freeze.py`, 9/9, selectors verified against ethers) |
| Authority-key registry (M323b multi-channel pinning)                 | SHIPPED | `geode/core/authority_key_registry.py`, 7/7, 27 Aug                                                                                                                                                  |
| Control-escalation resistance audits (M324 schema/capability audits) | SHIPPED | `geode/core/control_audit.py`, 19/19, 27 Aug (G1-G4, M324a, M324b)                                                                                                                                   |
| Dev-fund governance (M325)                                           | SHIPPED | `geode/core/fund_pacing.py` + 7/7 tests, 27 Aug: silence releases; an affirmative negative majority is required to hold; zakat end state has no pause path                                           |
| Bootstrap council sunset (M327)                                      | SHIPPED | `geode/core/bootstrap_council.py`, 12/12, 27 Aug (multi-party, timelocked sunset, zero-stake admission, charter-fixed cap)                                                                           |     |

## 3. EVM contracts (infrastructure/evm)

| Item                                                                                                                          | State   | Evidence / note                                                                                                                                                             |
| ----------------------------------------------------------------------------------------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CreditLedger (vesting, claims, slash ladder, takedown filing)                                                                 | SHIPPED | hardhat tests green                                                                                                                                                         |
| ProofAnchor, LinearProofVerifier                                                                                              | SHIPPED | hardhat tests green                                                                                                                                                         |
| GovernanceFloors (floors mirror + charter-fixed cap/quorum/diversity)                                                         | SHIPPED | 6/6 hardhat tests, 27 Aug                                                                                                                                                   |
| Freeze/escrow path for M323 ministerial freezes (freeze on valid-format notice; validators cannot move funds during a freeze) | SHIPPED | CreditLedger freezeArtifact/liftFreeze/isFrozen (5/5) + the deterministic notice→calldata bridge (`settlement_freeze.py`, 9/9)                                              |
| Deployment scripts for Sepolia + Arbitrum Sepolia                                                                             | PARTIAL | `scripts/deploy_testnet.js` shipped 27 Aug; local dry-run PASS (proxy + floors + anchor, charter constants verified); live deployment awaits the user's keys/venue decision |

## 4. Serving / infrastructure

| Item                                                                              | State   | Evidence / note                                                                                                                               |
| --------------------------------------------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| API service (geode/api)                                                           | SHIPPED | existing                                                                                                                                      |
| CLI (`geode`), Dockerfile                                                         | SHIPPED | existing                                                                                                                                      |
| Deployment guide                                                                  | SHIPPED | docs/DEPLOYMENT.md                                                                                                                            |
| Private-serving integration into the serving flow (FHE head wired to a real axis) | SHIPPED | `geode/privacy/fhe_gateway.py` (ciphertext-only session on the CKKS backend) + `geode/core/serving_tiers.py` (tier audit ledger), 8/8, 27 Aug |

## 5. Evidence hygiene before launch

- [ ] Full unit suite green: **963/963 green 28 Aug** (with M306 oracle, M309 custody, M316 chains, M323 bridge + authority keys, M324 audits, M327 council, M301/M320 alignment + bus, M321 campaign, serving-tier/FHE wiring, M330 FHE probe, M332 extraction guard, M337 session TTL, M338 minor vectors, M343 representation registration, plus the earlier M328/M322e/M318/M325 ships). EVM suite 59/59.
- [ ] Tier4 evidence directory complete and VOID records preserved: in progress (M322e family).
- [x] Whitepaper cost figures replaced by measured numbers: FHE head ~23 s/query on one CPU and ~1.7 MB round-trip are measured and in the whitepaper; the whitepaper states the current design only (version tags and withdrawn-design history removed 27 Aug).
- [ ] M188 counsel items (Q9/Q10 and the launch-jurisdiction posture) resolved with counsel: NOT resolved (registered open).
- [ ] The authority-key registry jurisdiction set for the launch venues registered: NOT BUILT.

## 6. Blocking decisions the user owns

1. M318: Pedersen commitment as the registry key (recommended) vs
   publishing W and retiring the proof layer.
2. M322 premium trunk: confirm the FHE path is premium-only for
   launch (the registered posture) or defer private serving to
   post-launch.
3. Launch venue set (Sepolia + Arbitrum Sepolia per the whitepaper)
   and the authority-key nexus list for M323a.

## 6b. Launch gates (from `TESTNET_LAUNCH_PLAN_v26.md`, 27 Aug)

- [ ] N = 9 validators recruited, multi-party, distinct behavioural
      identities (the registered corruption budget is <= 1).
- [ ] Librarian key ceremony executed per the plan (air-gapped,
      2-of-3 Shamir custody, address registered on the target
      testnet, test transaction recorded).
- [ ] Privacy launch gates 1-8 audited and recorded (tier
      auditability, ciphertext-only FHE, no plaintext model,
      commitment-only ledger, gateway no-retention, economic-only
      incentives, the M324 upgrade gate, residuals restated).
- [ ] Disposal procedure rehearsed once on the testnet before the
      first real epoch.

## 7. Current build order (maintained by the agent)

Shipped this session (27 Aug): M306 oracle + audit (G1 PASS,
G2 measured negative → R-A6d operative), M309 custody, M316
chains, M323 settlement bridge, M323b authority keys, M324
audits, M325 pacing (earlier), M327 council, M301/M320 modules,
M321 campaign + coverage metric, serving-tier/FHE wiring,
testnet deploy script (dry-run PASS), M297b (LOOCV interior
minimum at lambda\*=300; test at star below the anchor), H26-9
(measured inversion on the sealed M286 numbers), H26-4 (measured
on the corrected ms-13244 cell: anchors reproduced, aligned CCA
0.0133 and concatenated 0.1943 both below the single encoder - the
shared-space thesis unsupported at this scale, per §8.40).
Remaining: R-A7b temperature/ECE (M302 pending half), M318 user
decision 1, live deployments (user decisions 2-3).
