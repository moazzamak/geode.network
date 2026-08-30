# PRE-TESTNET LAUNCH PLAN — USAGE SURFACE PER PARTY (30 Aug 2026)

**Status:** planning document, not implementation. The questions this
answers: who touches the network, how each party actually reaches it
(front end, tool, toolkit, or program), what they read and write on
chain, and what each surface must be good at. Implementation comes
later, behind the README development notice: nothing here deploys
anything.

**Registered context this plan builds on.**

- M187 chain decision: Ethereum L1 for anchors; Arbitrum One for
  token/vesting/settlement/commit-reveal; OpenTimestamps as an
  optional seal. Premise: GEODE on-chain traffic is low-frequency;
  fees are a non-decision.
- M194 default proposal: Sepolia testnet first (chain id 11155111),
  then L1 after the legal review (M188). Funded keys stay
  user-owned. Nothing mints or anchors until the user acts.
- API packaging topology (24 Aug): users run NOTHING. The gateway is
  trustless-by-verification, so a hosted public API is the default
  surface for buyers; tiered self-host (same Docker image) covers the
  privacy tier and censor-resistance; supply-side images (serving,
  validation) run only on the supplier side.

## 1. The shape of the network

The network has two halves that meet in the settlement contract.

1. **The serving plane** (off-chain): hosts run frozen artifacts and
   answer queries; validators sample and score; the reference
   executor replays probed sessions. This plane is reached by
   network address, not by RPC.
2. **The settlement plane** (on-chain): the ledger, the inbox, the
   governance executor, the anchor. Every party reaches this plane
   through an Ethereum RPC — run locally or through a third-party
   hosted service — and reads and writes the chain state that is
   their part of the game.

Every party on the network is an Ethereum client first: they watch
the same chain and react to it. The usage surface question is: what
does each party need to see, what does it need to send, and how does
it prefer to be asked?

## 2. The Ethereum connection, common to all parties

One decision is shared: **how each party reaches the RPC.**

| Mode                                               | Who                                                                              | Qualities                                                                     |
| -------------------------------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Local node or local RPC (self-hosted)              | operators who need proof-grade reads, no third-party trust, or offline operation | privacy, no rate limit, full control; costs sync/ops                          |
| Third-party hosted RPC (Infura/Alchemy/DRPC class) | the default for everyone else                                                    | zero ops; rate limits; a third party sees the reads (reads are public anyway) |

The chain is public, so using a hosted RPC leaks nothing that is not
already public. The distinction that matters is **who is allowed to
WRITE**, and that is the party's key, never the RPC. Every party runs
a thin, deterministic client of the same events, so the "latest
state of the network" is the same for everyone:

- the latest closed epoch and the epoch's attribution root (has it
  landed? was it challenged?);
- incoming orders (settlement branches with leaves naming this
  address as a payee);
- the last route hash and the served-session records it commits;
- filings that name this party (a slash against it, a registry
  change, a governance proposal), and the challenge windows on them;
- the inbox queue head (is an entry overdue? must someone
  incorporate it?).

The client reads these by polling the relevant contract events and
view functions; write paths are triggered by the party's own state
machine, never by the chain.

## 3. Per-party usage surface

For each party: what the party does, the surface it gets, what it
reads and writes, and the qualities that surface must have. Ordered
by the user's examples first (contributors, validators, librarian),
then the rest.

### 3.1 User (buyer)

**What they do:** buy answers. Nothing else — no key rotation, no
queue watching, no settlement mechanics.

**Surface:** a hosted web front end (the default) and, for
programmatic buyers, an HTTP API over the gateway. No RPC, no wallet
in the happy path beyond signing a payment.

**Reads:** answers, prices, and the routed capability's provenance
(who served it, what artifact, what fingerprint). Writes: queries and
payments.

**Qualities:** answer latency and quality; simple, honest pricing
display; no jargon. The user must never see "epoch", "root", or
"challenge". The trust boundary is the gateway's verifiability: the
user does not verify, but a third party can, so the gateway cannot
profit by lying. This is the one surface where design taste is a
product feature.

### 3.2 Contributor

**What they do:** train or freeze a capability, register it, price
it, and collect measured earnings. They are suppliers, so they are
closer to the chain than users are.

**Surface:** a **toolkit** they integrate into their training
pipeline (Python, matching the paper's Python-first primitives), not
a web form. A contributor's artifact is a file plus metadata; the
toolkit signs a registration transaction, submits the fingerprint and
the price, and later submits the constraints/probes under the
commit-reveal discipline (M282). The same toolkit reads the party's
own chain state: registration status, admission, per-epoch measured
earnings, vesting, and the settlement branches that name them.

**Reads (via RPC, local or hosted):** their artifact's registry
status; the epoch's attribution root (did the settlement name them?);
their vested balance and claim schedule; any slash filing against
them and its window. **Writes:** registration + price + evidence
commitment, reveal, claims (pull), and challenge responses if named.

**Qualities:** the registration path must be one command from a
trained artifact; earnings must be readable as a simple statement,
not a log; the challenge window must be surfaced as a deadline, not
an event. Contributors are the supply side of the market, so
friction here is the market's cost of supply.

### 3.3 Validator

**What they do:** sample sessions, pose probes, verify, and attest
verdicts. They are a program, not a person.

**Surface:** a **headless program** (a Docker image or a package)
that runs on their own hardware. It watches the chain for sampling
assignments, does its work off-chain against the serving plane, and
writes its attestations and verdicts on chain. Its whole "UI" is a
log and a config file.

**Reads (RPC):** the epoch's sampling draw (beacon-seeded, so it is
the same for every validator), the sessions assigned to them, the
pending disputes and their attestation windows. **Writes:**
attestations, verdicts, dispute participation.

**Qualities:** determinism (same inputs, same verdict), low
false-positive rate so their stake is not wasted, and clear audit
trail. The program must be boring and correct, because it runs
unattended for epochs.

### 3.4 Librarian

**What they do:** the settlement operator's key acts: incorporate
inbox entries within their window, record the credits a session's
evidence supports, and post the fast-path attribution root.

**Surface:** a **daemon** (headless, runs as a service) connected to
an RPC — local or hosted, their choice — that watches the chain and
performs the acts in time. This is the user's example directly: the
librarian is "connected to an RPC which is running either locally or
through some other third party hosted service and fetching
information about the latest state of the network itself from the
Ethereum network, meaning any incoming orders, the last route hash,
and so on."

**Reads (RPC):** the inbox queue head and its deadline; incoming
order events (branches to incorporate); the last route hash and the
session records they must convert into credits; the epoch boundary
(to post the root on time); challenge windows on their own filings.
**Writes:** incorporation, credit recording, fast-path root posting.

**Qualities:** timeliness above all (the queue head has a deadline;
missing it is a recorded violation and invites a stranger to do the
job); crash-safe resumption (the daemon must not double-record or
double-incorporate after a restart); and a status page that shows
"am I on time" at a glance. The librarian is the one party whose
failure is a network-visible event, so the daemon must make its own
health legible.

### 3.5 Reference executor

**What they do:** re-run the frozen artifact on probed sessions and
compare. Like the validator, a program.

**Surface:** a headless program tied to the serving plane (it needs
the artifact). It reads the probe assignments from the chain, runs
off-chain, and writes its comparison result on chain. Same shape as
the validator's surface, fewer writes.

### 3.6 Host

**What they do:** serve frozen artifacts for a capability, answer
queries, meet probe obligations.

**Surface:** a serving image they run on their hardware, plus a
public endpoint the router and the gateway can reach. On-chain, they
read their assignment and probe obligations and write availability
and, for the private tier, commitments. The host's on-chain surface
is thin; its real surface is the serving endpoint and the
substitution-detection contract it must satisfy.

### 3.7 Developer

**What they do:** build on the network: register primitives, publish
the reference hosting cost, propose registry changes.

**Surface:** the same Python toolkit as contributors (primitives are
small deterministic programs with the same registration path), plus a
**sandbox** for primitives that execute code (the registered rule:
sandboxed execution is a guarded tool arm, never a primitive).

### 3.8 Outside inputs (not parties, but surfaces)

- **Publisher:** a model card and artifact manifest submitted with a
  fingerprint; no live surface beyond the toolkit.
- **Benchmark:** benchmark results submitted as the committed
  measurement that admission follows; a file + commit, no live
  surface.
- **Authority key:** a government order path that can freeze (never
  burn); reached through the multi-channel pinning procedure, not
  through an open surface.

## 4. The surface qualities table

| Party              | Surface                  | Reach               | Chain reads                                    | Chain writes                       | Must be good at                               |
| ------------------ | ------------------------ | ------------------- | ---------------------------------------------- | ---------------------------------- | --------------------------------------------- |
| User               | web front end + HTTP API | hosted gateway      | prices, answers, provenance                    | query, payment                     | latency, clarity, trustless-by-verification   |
| Contributor        | Python toolkit           | RPC (local/hosted)  | status, root, earnings, vesting, filings       | register, reveal, claim, challenge | one-command registration, readable statement  |
| Validator          | headless program         | RPC                 | sampling draws, assignments, disputes          | attest, verdict                    | determinism, low false positives, audit trail |
| Librarian          | daemon service           | RPC (local/hosted)  | queue head, orders, route hash, epoch, windows | incorporate, record, post root     | timeliness, crash safety, health legibility   |
| Reference executor | headless program         | serving plane + RPC | probe assignments                              | comparison result                  | reproducibility                               |
| Host               | serving image + endpoint | serving plane + RPC | assignments, probe obligations                 | availability, commitments          | uptime, probe compliance                      |
| Developer          | Python toolkit + sandbox | RPC                 | registry, floor                                | register primitive, propose        | sandbox safety, integration                   |

## 5. What this plan deliberately leaves open

- **The gateway's implementation** (hosted public API vs
  self-hosted image) is the registered packaging topology, but its
  build is a later milestone.
- **Whether the first testnet runs the full EVM settlement stack or a
  subset** (e.g. anchors + ledger only) is a go/no-go decision for
  the user, informed by M188/M190.
- **The exact RPC providers** for the hosted default are not chosen
  here; the client is provider-agnostic (standard Ethereum JSON-RPC).
- **Key custody:** funded keys remain user-owned; the testnet keys
  are the user's decision (M194).
- **The "latest state of the network" client** (the shared event
  client in section 2) is the first thing to build, because every
  party except the user uses it. It is implementation for later, but
  it is the dependency of everything else, so it is named here.

## 6. Order of work (for the implementation phase, not now)

1. The shared chain-state client (section 2) — every party depends on it.
2. The librarian daemon — the network cannot settle without it, and
   its timeliness is the sharpest on-chain property.
3. The contributor toolkit — the supply side; registration + earnings.
4. The validator/reference-executor programs.
5. The host serving image.
6. The user front end, last, on top of the gateway.

Rationale for that order: the network is useless to a buyer until it
has supply and settlement. The user surface is the last thing, and
the only place where taste, not correctness, is the requirement.
