# GEODE Threat, Game-Theoretic, and Security Analysis

**Date:** 25 Aug 2026
**Author:** analysis pass requested by the user ("absolutely sure we are
not leaving any gaps" before launch)
**Inputs read in full:** `analysis/WHITEPAPER_GEODE.tex` (23 pages, as
of commit `089b7346`), `analysis/GEODE_ECONOMIC_DESIGN_v1.md`,
`analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md` (changelog, parameter
defaults, open decisions, M209 cost model section).

This is a desk analysis, not a proof. Every gap below is stated with
the attack that opens it, why the current text does not close it, and
a proposed fix. Severity: CRITICAL = exploitable for profit against
the network as written; HIGH = degrades a core guarantee or prices
out a protection; MEDIUM = needs a decision before launch; LOW =
polish.

---

## 1. Attack model

| Actor              | Legitimate power                                        | Malicious capability                                                                |
| ------------------ | ------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| User               | pays per metered unit, declares task/caps               | wash-pay, probe the router, DDoS gateway, abuse disputes for free compute           |
| Contributor        | registers artifact, sets price, serves                  | substitute cheap model, bribe validators/executors, wash-ring, grind sampling seeds |
| Host (primitives)  | runs third-party primitives, takes host share           | serve altered primitive, overcharge execution, steal inputs                         |
| Reference executor | runs sealed artifact on probed sessions, earns flat fee | fabricate mismatches, rubber-stamp, harvest inputs, collude with host               |
| Validator          | poses scored challenges, audits peers                   | label-fabrication, flood the pool, cartel to pass/fail artifacts, leak eval data    |
| Librarian/Dev      | append ledger, execute registry updates, bootstrap arms | rewrite ledger, censor, self-deal (identity-based residual)                         |
| External           | publisher checkpoints, benchmarks                       | nothing on-chain; supply-side only                                                  |

Standing registered bounds (kept): no mechanism survives a lying
majority; identity is the wallet; economics-only incentives; the
wallet key can always be sold. These are stated in the paper and are
NOT re-flagged below.

---

## 2. What the design already closes (verified, not re-opened)

- **Wash-trade basics:** 2.5% dev dock per hop + self-payment
  exclusion keyed on the payout address + no volume→score path
  (scores are measured on held-out data, never usage). A wash ring
  with no other lever loses 5% per round-trip.
- **Slash incentives:** burns are destroyed, never awarded; replay
  gates every conviction; graded ladder L0–L3. No bounty-hunter
  economy, no dev-fund kickback.
- **Admission honesty:** commit-reveal sealing, held-out scoring,
  aggregate-only verdicts, bounded precision, sharded corpora with
  leak-identification, rotation.
- **Serving substitution:** live shadow probe on the session's own
  input, executor commit-before-compare, replay-settled disputes,
  flat executor fee, disputed-credit escrow (all registered 25 Aug).
- **Pricing:** unit derived from the descriptor, timelocked changes,
  session price lock, spend caps that are limits not pre-payments.
- **Vesting:** N=4 epochs, justified by the M293 detection-horizon
  sweep (p90 = 4 epochs; N clears it with 2× margin on the
  registered scenarios).
- **Takedown:** M294 eligibility (activation, tenure, recency),
  weighted verdict, no financial effect (delist only) so it cannot
  become a financial weapon.
- **Primitives:** Firecracker-class microVM, no ambient authority,
  host-side signing, determinism enforced at the sandbox.
- **Honest boundaries:** the paper already states the encoder proof
  gap, OOD failure, probing residuals, and the lying-majority limit.

---

## 3. Game-theoretic analysis by mechanism

### 3.1 Admission

**Judges are paid by the judged.** The contributor's challenge budget
pays the sampled validators. "No one chooses their judges" is true,
but _knowing_ them is enough: the sampled set is computable once the
admission commit `c` is fixed, so a contributor can bribe the nine
known judges before the session. If the per-challenge fee is
contributor-set, the bribe is even priced in. This is judge capture,
and it is the weakest link in the paper (gaps C3, C4 below).

**Pool flooding.** Validator registration is permissionless with a
fee whose value is unset. Nothing stops an attacker from registering
a flood of validators and taking majority of the _admission_ sample —
the M294 eligibility machinery (activation window, tenure ramp,
activity floor) was built for takedown voting but was NOT applied to
admission sampling. The same attack the user closed for takedown is
open at admission (C3).

**Sampling-seed grinding.** The sample is `H(e, a, c)` with `c` the
contributor's own commit. If the seal admits a free nonce, the
contributor grinds `c` across many nonces and picks the value whose
sample is friendliest. A fixed public seed is grindable (C4).

**Verdict economics.** With registry-set validator fees and audited
labels, a validator's honest expected value beats bribes only if
`bribe < P(caught)·burn`. That inequality is unquantified — an H-series
follow-on item, not a paper defect.

### 3.2 Serving and the probe

**Honest contributor vs substitute (as fixed).** Honest profit ≈
`0.975·units·price − cost`. A substitute saves `Δ` per query but
dies at the first probe it fails; with a per-epoch probe floor (H1)
the expected survival is inside the N=4 vesting window, so net cheat
EV = `savings·horizon − unvested burn` is negative for any material
`Δ`. The math holds _only_ with the floor: at ρ=0.05 relative and low
traffic, expected survival is 20 sessions — which on a quiet axis can
be months, far outside the vesting promise (H1).

**The executor has no profitable deviation** (verified): flat fee
either way, burns destroyed, false claim loses the replay and burns
the executor's own promise, and the serving answer is compared only
after the executor commits. The remaining deviation is _collusion_
with the serving host, which requires both roles corrupt in the same
session — probability `f²` for corrupt fraction `f` under
independent sampling, and replay exposes the executor afterward.
Acceptable if executor sampling is enforced (C2).

**The probe itself is currently predictable.** The gateway seeds
probe choice from the epoch anchor; the previous anchor is public
before the epoch begins, so a serving host can compute which sessions
will be probed in advance and be honest exactly on those. This defeats
the probe entirely and is the single most important finding (C1). The
fix is commitment ordering, not secrecy.

### 3.3 Settlement, vesting, self-payment

- Wash ring: −5% per round-trip plus probe dock on 5% of sessions.
  The only positive lever would be a usage-measured bonus — see H5.
- Sybil payers defeat address-keyed exclusion; the 2.5% tax is the
  real barrier. Correct as designed; note it only _prices_ washing,
  it does not prevent it.
- Claims are account-bound, pull-only, post-claim ETH is ordinary:
  bearer-key sale residual is already registered (Known limits §8).

### 3.4 Slashing

Burn-not-award is the right call and is implemented consistently.
Two sharpenings: (a) L1 burns only the _unvested_ remainder, so the
probe floor (H1) is what makes L1 bite; (b) a contributor with small
serving revenue but large reputation is only hurt by L2 delisting —
correct, since reputation is measured and cannot be washed.

### 3.5 Quorum takedown

The M294 design is sound for its purpose and its failure mode is
bounded (delist only, no burn). Remaining items: proposal deposit
value unset (M2); takedown cannot stop off-path serving (registered
honest boundary — keep).

### 3.6 Bootstrap and headroom

Dev self-restraint is enforced by the public registry only for the
_declared_ dev address; a dev that re-registers under a fresh
address is undetectable on-chain (identity residual, already
registered). The headroom rule's real enforcement is the market: a
bootstrap arm priced at reference cost loses the axis to any
strictly better contributor automatically. Sound.

### 3.7 Primitives and hosts

Author-host collusion is the existing wash vector, blocked by the
same stack. Hosts see plaintext by construction (they execute user
code) — registered as opt-in exposure. The residual attack is a host
that silently modifies a primitive: it earns the same fees but the
_author's_ royalty attribution stays — the author's recourse is the
determinism contract (the sealed hash means a modified run disagrees
with replay). Worth stating in the paper: a host is exposed to
replay-gated slash if it serves a modified primitive.

---

## 4. Security analysis

### 4.1 Ledger and anchoring

The fork rule "the chain with the latest Ethereum anchor wins" is
insufficient against a compromised librarian: the key can re-roll
the chain between anchors and re-anchor the fabricated history;
clients following the rule then bless the forgery. The anchor payload
(tip hash, record count) does not include the prefix, so divergence
detection depends on clients keeping old anchors — the rule should
say so (C5). Also: anchor cadence once per epoch leaves a 7-day
rewrite window (L1).

### 4.2 Proofs of computation

Bulletproofs over the closed-form head are cryptographically sound,
but the paper never names the verifier. A settlement batch anchors a
_hash of proofs_; an invalid proof whose hash is anchored is only
caught if someone verifies the proof itself. Without a paid, sampled
batch-verification step, the proofs are decorative and the "payment
tied to a provable computation" claim overstates (H2).

### 4.3 Privacy

Three plaintext points exist: the gateway (routes and duplicates),
the serving host, and — new — the reference executor, who must see
the probed input to run the artifact. The paper's privacy contract
("inputs never enter the public record, not retained beyond the
session") is true but incomplete: the executor is a second plaintext
handler and should be named as such, under the same no-retention /
no-training contract, with compromise bounded like host compromise
(H4). Disputes reproduce the one disputed input into the sealed
replay environment only — never the ledger (M6).

### 4.4 Keys and custody

Operator/payout split, pull claims, claimer pays gas, account-bound
credits: all correct. The librarian key during bootstrap is the
system's single largest technical trust point; the timelocked
`setRecorder` and the M189 quorum path are the registered migration,
and `renounceRecorder` must not be reachable before a governed
upgrade path exists (L3).

### 4.5 Sandbox

Firecracker-class isolation is the right posture. The code-execution
engine running user code on contributor machines is opt-in per arm
with ceilings — consistent. Escape is a standing class risk, bounded
by freshness and no-secrets; acceptable, published.

### 4.6 Numerics and replay determinism

"Answers must match exactly" and "the head replays bit-exactly"
require a numerics policy the paper does not state: float64
promotion for the solve, pinned kernels, fixed reduction order,
kernel versioning per platform. Float32 platform differences would
make honest contributors mismatch. The benign-nondeterminism band
(registered 25 Aug) must be _measured per artifact at admission_ and
published as an erasure-style certificate, and the substitution
margin (e.g., a quantized substitute's divergence) must be shown to
exceed the band — else the band is a substitution license (H6).
This is the M90.2 lesson applied at protocol level.

---

## 5. Gap register

### CRITICAL

**C1 — Probe choice is predictable; commitment ordering is the fix.**
Seeding probe choice from the epoch anchor lets the serving host
compute probed sessions in advance (the previous anchor is public
before the epoch) and be honest exactly there. This defeats the
probe with zero risk, and makes the paper's "cannot predict" claim
false as written. **Fix:** per session, the serving host commits
`H(answer)` before the gateway reveals that session's probe flag;
flagged sessions run the reference and compare against the
committed answer. Secrecy of the reference path becomes unnecessary,
which also retires the "fingerprint the reference path" residual.

**C2 — A contributor can be its own reference executor.**
"Anyone holding the sealed artifact can provide reference runs" —
including the artifact's own operator. A self-executing probe is a
rubber stamp. **Fix:** executors are sampled per session from the
provider pool by hash, with structural self-exclusion: the executor
of a probed session cannot be the operator or payout address of any
artifact that served it.

**C3 — Admission sampling has no flood/eligibility defense.**
The M294 machinery (activation A=2 epochs, tenure ramp T=4,
activity floor, recency W=2) exists only for takedown voting. A
validator flood can own admission verdicts. **Fix:** apply the same
eligibility gates to admission sampling, and make the per-challenge
validator fee a registry-set schedule (never contributor-set) so
judge capture is not priced in.

**C4 — Admission sampling seed is grindable.**
`H(e, a, c)` with contributor-chosen `c`: a free nonce in the seal
turns admission sampling into a lottery the submitter can re-roll.
**Fix:** sample from a beacon value that postdates the ledger-frozen
commit (the first anchored tip after the commit freezes; sample at
evaluation, not at submission). No free field may influence the
sample.

**C5 — Ledger rewrite between anchors is blessed by the fork rule.**
"Latest anchor wins" lets a compromised librarian re-roll and
re-anchor fabricated history; clients following the rule accept it.
**Fix:** prefix immutability — any chain whose already-anchored
prefix disagrees with any earlier anchor is invalid outright; the
fork rule applies only to unanchored suffixes. Divergence in an
anchored prefix is an automatic librarian-replacement incident.

### HIGH

**H1 — No absolute probe floor for low-traffic axes.**
ρ=0.05 relative ⇒ expected 20 sessions to detect. On a quiet axis
that can exceed the vesting window, so L1 burns little and the
"detection sits inside the vesting promise" claim fails exactly
where GEODE wants long-tail axes. **Fix:** per-epoch minimum probe
count per active axis (≥1) or per-axis ρ raised below a traffic
threshold; the claim then holds everywhere.

**H2 — Proof verification is unnamed and unpaid.**
No verifier role for settlement proofs; anchoring proof-hashes
verifies nothing. **Fix:** batch verification at settlement by a
paid, sampled verifier role (or on-chain batch verifier), invalid
proof = L1 replay-gated burn, verification failure is a ledger
event.

**H3 — Dispute accessibility and input binding.**
(a) A deposit equal to full replay cost prices small contributors
out of disputing a false mismatch claim — asymmetric wallets make
the executor's false-claim attack affordable against small players.
(b) The disputed input must be bound: both parties' session commits
should cover `H(input)` and the reveal must match, else a disputing
contributor substitutes a benign input and "wins" the replay.
**Fix:** deposit = registered reference-run price; loser pays the
full replay; session commits cover `H(input)`; a non-matching
reveal resolves against the party.

**H4 — The reference executor is an unstated plaintext point.**
State it: probing adds a second plaintext handler under the same
data contract (no retention, no training), bounded like host
compromise. The privacy section should name all three points
(gateway, host, executor).

**H5 — Coverage-novelty bonus is undefined and potentially
washable.**
The design doc §8 grants a coverage-novelty bonus to first
registration; the whitepaper's earn table omits it; no measurement
is specified. If measured from usage it is a washable subsidy
(the 2.5% tax may be cheaper than the bonus). **Fix:** define the
measurement (held-out utility or a registered trigger) or drop the
bonus; reconcile the two documents either way.

**H6 — Numerics policy and the match band.**
Bit-exact replay needs a stated numerics policy (float64 solve,
pinned kernels, fixed reduction order). The benign-nondeterminism
band must be a measured per-artifact certificate at admission, with
substitution margin shown to exceed it. **Fix:** register both;
ship the band measurement with every admission.

### MEDIUM

**M1 — Validator economics.** Registration fee value unset (M209
gate); audit work (0.1 re-label, two validators) has no stated
payment — unpaid audits get shirked and label fabrication goes
unchecked. Fix: audits paid from the challenge budget or demerited.

**M2 — Takedown deposit unset.** Too small ⇒ proposal spam tires
voters; too large ⇒ real reports deterred. Fix: deposit ≥ the
registered cost of one vote round.

**M3 — Answer liability posture undefined.** Honest-artifact wrong
answers (drift, OOD) have no stated policy. Publish: best-effort,
confidence is not a warranty, refunds only for measured contract
violations.

**M4 — Registry dedupe unspecified.** An identical artifact hash
re-registered by a copycat: same artifact, two registrations.
Specify: same hash = same artifact = one registration; a copycat
registers a different artifact and earns admission on its own.

**M5 — Gateway concentration.** One gateway sees all plaintext and
is a single DDoS/censorship point; ops-layer rate limiting is a
registered blocker. Keep as a launch gate; plan HA before public
launch.

**M6 — Dispute-driven input disclosure.** Bound by deposit (H3)
and by reproducing the disputed input only into the sealed replay
environment; the ledger still records no sample.

### LOW

**L1 — Anchor cadence.** Once per epoch leaves a 7-day rewrite
window; move to per-day or per-batch anchors as volume grows.

**L2 — Primitive attempt-metering.** Coarse unit; keep the
resource ceilings binding (they are).

**L3 — renounceRecorder irreversibility.** Freeze without upgrade
path; ensure the governance-contract path exists before renounce.

---

## 6. Pre-launch gates (quantities that need values)

1. Registration fee value (M209 cost model) — gates admission spam.
2. Registry-set validator fee schedule — closes the judge-capture
   pricing (C3).
3. Per-epoch minimum probe count per axis (H1).
4. Reference-run price register + dispute deposit rule (H3).
5. Takedown proposal deposit (M2).
6. Audit payment or demerit schedule (M1).
7. Numerics policy + per-artifact band certificate (H6).
8. Batch proof-verification role and payment (H2).
9. Anchor cadence for production (L1).
10. The ten parameter defaults already listed in the plan (pending
    user confirmation) — none of the above replaces that list; it
    extends it.

## 7. Residuals we accept and why

- **Lying majority** (validators, executors, or librarian):
  economics and visibility can only price it; registered.
- **Bearer-key sale:** inherent to permissionless design;
  registered.
- **Probe rate as a probabilistic instrument:** after C1's
  commitment-ordering fix, residual risk is executor-host
  collusion at probability ~f² — accepted, monitored via the
  coverage rate.
- **Encoder provability:** cost-triggered, not default; registered.
- **OOD detection:** measured as not working; the guard stays a
  report; registered.
- **Regulatory classification:** M188 gates any token and
  deployment decisions; unchanged by this analysis.

## 8. Bottom line

The design is unusually self-aware, and the 24–25 Aug amendments
(probe funding, identity-not-accuracy, live comparison, dispute
settlement) closed the obvious economic holes. Five CRITICAL items
remain, and they share a theme: **every one is a sampling or
ordering defect, not an incentive defect** — predictability (C1),
self-selection (C2), floodability (C3), grindability (C4), and
rewrite-acceptance (C5). All five have concrete, small fixes that
do not change the architecture. They should be closed before any
mainnet exposure; C1 and C2 alone would otherwise let the probe be
defeated silently, which is the design's central enforcement
instrument.

---

## 9. Disposition (25 Aug, user: fill the gaps)

All findings above were addressed in `WHITEPAPER_GEODE.tex` and
`GEODE_ECONOMIC_DESIGN_v1.md` (plan entry "THREAT-ANALYSIS
FINDINGS ADDRESSED IN THE WHITEPAPER"). C2 was STRENGTHENED on the
user's direction: key separation alone is worthless against cheap
anonymous accounts, so executors are sampled per probed session
from a pedigreed pool (validator-style activation, activity, and
tenure), k_e=2 by default, revealed only after the serving host's
answer is committed. H5 (coverage bonus) is the only item left
open by policy choice: the bonus ships nothing until its
measurement rule is registered.
