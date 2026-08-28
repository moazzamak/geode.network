# M289 SPEC — Programmable Primitives: threat model, mitigations, and incentive changes (24 Aug 2026)

Registered BEFORE any build, per house rule. Scope: opening a
user-facing surface where anyone can author a Python program, have it
routed through the network, and earn arm-equivalent fees by measured
utility. Predecessor context: the internal programmatic-primitive
pattern already exists (M267 FFT/mel stage, §4.10 failover bottom
tier); the archived M129 literature review covers the field. This spec
registers the threats, the mitigations, the incentive changes, and the
build sequence. NOT DISPATCHED — user discussion gates it.

## 1. Threat taxonomy

Severity scale (ethskills audit scale, reused): Critical = direct fund
loss no preconditions; High = fund loss with conditions / permanent
DoS; Medium = degraded behavior, trust-model violation, incorrect
accounting; Low = best-practice; Info.

### A. Code-execution threats (the validator/host runs attacker code)

**[A-1] Resource exhaustion — Medium/High.** Infinite loops, memory
bombs (`"x"*10**12`, zip-bomb inputs), CPU burners, subprocess
spawning.
_Mitigation:_ OS-level sandbox only — subprocess with wall-clock
timeout, RSS cap, no child-process spawning, input AND output size
caps. Deterministic failure → the fallback chain takes the session.
_Gate:_ the sandbox harness (M290) is a release gate for public
submission; no program runs in the validation pipeline without it.

**[A-2] Environment escape / exfiltration — High.** Python
language-level sandboxes (AST allowlists) are bypassable via ctypes,
mmap, subprocess, pickle gadgets — an attacker who gets code execution
reads the sealed eval data, validator keys, or exfiltrates via
network.
_Mitigation:_ the barrier is OS/container-level, not language-level —
disposable VM/container per validation run, no network namespace,
read-only filesystem except a scratch workspace, no secrets mounted,
dropped privileges. Honest boundary registered: validators run
untrusted programs ONLY in secret-free disposable environments; this
extends the registered host threat model (trusted TEE vs untrusted
MPC) to code artifacts.

**[A-3] Non-determinism to dodge replay — Medium.** `time.time()`,
`random`, hash seeds, /dev/urandom, float variance across machines —
a program that passes admission then serves differently.
_Mitigation:_ the determinism contract (bit-exact for integer/logic
paths; registered tolerance for float paths — the M288 lesson);
admission replay + periodic validator re-replay; drift detection via
the existing deterministic health probes (§4.10: contract + payload
hash). A program that drifts from its sealed artifact is the Level-1
slash class (deviation from sealed behavior).

**[A-4] Supply chain — Medium.** A program pins a malicious
dependency or downloads code at validation time.
_Mitigation:_ dependency allowlist (stdlib + registered pure
packages), hash-pinned, no network access at validation/serving time.

### B. Evaluation gaming (the program cheats the metric)

**[B-1] Test-set memorization/hardcoding — Medium.** `if
input_hash in testset: return canned_answer` — trivially easy for
code vs. learned arms.
_Mitigation:_ (a) commit-reveal admission (M282); (b) the local
toolkit scores on a PUBLIC dev split only — the held-out admission
split is sealed and validator-run; (c) cross-suite generalization
checks (a second held-out suite the author never saw); (d) rotation of
held-out data on re-admission. HONEST BOUNDARY registered: for fully
public benchmarks (HumanEval-class) sealing is impossible — the
protection is rotation + generalization checks + slash-on-deviation,
so memorization is BOUNDED AND RISKY, not prevented. This boundary is
stated, never papered over.

**[B-2] Metric overfitting — Low.** A program designed for the
registered metric's quirks. Same class as learned arms; the metric is
registered per task; acceptable by design.

**[B-3] Side-channel leakage — Low.** The program's output encodes
info about the input (e.g., echoes the input hash) to detect eval
rows. _Mitigation:_ strict output-schema validation at admission and
at replay; schema violations reject.

**[B-4] Self-modifying behavior — High if unhandled.** The program
mutates itself or its data between admission replay and serving, so
it behaves during eval and misbehaves in production.
_Mitigation:_ the sealed artifact IS the code hash; the serving host
must serve the exact sealed hash (registry verifies the deployed
fingerprint — the existing fingerprint discipline); any drift is a
Level-1 slash. Periodic deterministic health probes catch drift
continuously, not just at admission.

### C. Routing / economics threats

**[C-1] Flooding cheap artifacts — Medium.** Programs are cheaper to
register than GPU arms, so mass registration to capture axes/spam the
registry is the natural attack.
_Mitigation:_ flat registration fee per artifact (already decided)
prices mass registration; measured-quality gates reject junk; the
novelty bonus is gated on passing the axis's quality bar, not mere
registration. OPEN: whether a per-axis registration cap is needed —
leave as a timelock-adjustable parameter, decide by simulation.

**[C-2] Availability gaming — Low.** Register, then go dark and
collect failover traffic. _Mitigation:_ thaw keys on actually served
sessions (§4.10) — a dark host earns nothing; deterministic health
probes price downtime automatically.

**[C-3] Novelty-bonus gaming — Medium.** A trivial garbage program
"unlocks" an axis and collects the coverage-novelty bonus.
_Mitigation:_ the bonus requires passing the axis's measured quality
bar; trivial programs pass nothing and earn nothing.

**[C-4] Curved grading via suite authorship — Medium.** For NEW task
axes, someone must author the eval suite (data + metric). A malicious
author designs a suite only their own program passes.
_Mitigation:_ suite proposal is its own contribution class with
validator quorum review, sealed suite hash, and the registered
data-contribution machinery (M149/M182); the suite itself is a sealed
artifact subject to the same ledger rules.

## 2. Incentive changes required (registered)

1. **Novelty bonus re-gated:** coverage-novelty pays only above the
   axis's measured quality bar — never for registration alone.
2. **Admission gates extended:** commit-reveal (M282) + determinism
   replay + sandboxed validator run + cross-suite generalization
   check become mandatory for program artifacts.
3. **Continuous drift surveillance:** the §4.10 health probes
   (contract + payload hash) run against the sealed code hash for
   programs exactly as for learned arms; drift = Level-1 slash.
4. **Suite authorship as a contribution class:** authors of eval
   suites earn via the data-contribution machinery; the suite is a
   sealed, quorum-reviewed artifact. This adds a SECOND contributor
   role (artifact authors vs. suite authors) with its own measured
   value — no central planner.
5. **Fee asymmetry monitoring:** if program registrations flood, the
   flat fee is timelock-raised; per-axis caps are a registered open
   parameter, not a default.
6. **Sandbox as a release gate:** public program submission does NOT
   open until M290 (sandbox harness) passes. Until then, only
   dev-authored programmatic primitives run (the current state).

## 3. Build sequence (not dispatched)

- **M289 (this spec):** registered. User discussion gates.
- **M290:** sandbox harness — disposable secret-free execution,
  resource ceilings, replay comparison; its own tests + a
  red-team admission attempt (an intentionally hostile program must
  fail every check).
- **M291:** Python toolkit — ProgramSpec template, task-schema
  registry, local dev-split scorer, artifact hashing.
- **M292:** end-to-end demo — one authored program admitted through
  the full gate chain and routed in a sealed session, fees + replay
  recorded.
- Public submission opens only after M290–M292 all pass.

## 4. Honest boundaries (stated, never papered over)

- Python-level sandboxing is unsound; the barrier is OS-level.
- Public benchmarks cannot be sealed; memorization is bounded and
  risky, not prevented.
- Validators execute untrusted code only in disposable environments;
  this is a real operational cost added to the validator role.
- The toolkit makes cheating EASIER than for learned arms (code vs.
  weights) — the compensations are replay, drift probes, rotation,
  and the slash ladder, not secrecy.

## 5. Standard primitive library (user, 24 Aug — free-with-network)

The network ships standard programmatic primitives from day one
(memory, mathematics, code-execution engine, canonical transforms):
dev-shipped, permissively licensed, hash-pinned, run on each
contributor's own machine, dev earns NOTHING from them; fees
attribute only to the contributor's arms in the chain. These are
trusted code, so the threat taxonomy above does NOT apply to the
stages themselves. THIRD-PARTY primitives are PAID (user
clarification, 24 Aug): usage fees split between the primitive's
PAYOUT ADDRESS (a field of the unified registration — not a role;
the contributor sets the rate, timelocked with a notice period so
users can migrate away before a change bites) and the HOST running
it — a royalty ledger path that the librarian attributes per
session. THE CARVE-OUT: the code-execution engine runs
arbitrary END-USER programs inside contributor machines — the A-1/A-2
threats (exhaustion, escape) apply to what runs INSIDE the engine.
Registered: the engine is (a) opt-in per arm, declared in the arm
spec; (b) sandboxed with the M290 machinery (disposable execution,
resource ceilings, no network/filesystem); (c) the contributor's
exposure is declared in the arm spec, never hidden.

## 6. Execution isolation architecture (user, 24 Aug — "proper

precautions for third-party code")

The marketplace is AUTHOR ≠ HOST: a contributor may run their own
primitive OR anyone else's registered primitive, so third-party code
execution is the DEFAULT path for every non-standard primitive. The
contributor machine holds other arms' weights, sealed eval data, and
the settlement wallet key — the hostile primitive wants all of it.
This is the serverless-provider threat model; we adopt that stack,
we do not invent one.

- **Tiered policy:** (a) standard library (dev-shipped, sealed hash)
  runs directly — trusted; (b) EVERY third-party primitive runs in a
  DISPOSABLE MICROVM (Firecracker-class on Linux servers / Hyper-V on
  Windows), NEVER a shared-kernel container (a kernel exploit escapes
  containers; the microVM's minimal device model is the barrier).
- **No ambient authority:** the sandbox receives no secrets — no
  wallet key, no network by default, no filesystem outside a fresh
  scratch workspace, no sibling-arm data.
- **Signing is host-side:** the settlement key lives in a host
  process; the sandbox computes, the host signs. Sandboxed code
  physically cannot reach the key.
- **Determinism = security:** no network / no randomness / no wall
  clock is enforced at the sandbox level (denied syscalls), not just
  declared in the spec — the same restrictions that make replay work
  are the ones that close the exfiltration channels.
- **Ceilings + freshness:** CPU/memory/disk caps, no subprocess
  spawning, hard timeouts, fresh image snapshot per run (no
  persistence between runs). A hung or crashing primitive fails
  deterministically into the fallback chain; repeated violations are
  recorded and escalate to slash review.
- **Hash-pinned runtime:** runtime image + dependency allowlist
  pinned by hash; nothing downloads at execution time.
- **Honest limit:** VM isolation is strong, not absolute — hardware
  side-channels matter only in multi-tenant clouds; on single-tenant
  contributor machines the bar is microVM + no secrets inside, the
  same bar every serverless provider ships.
- **Build impact:** M290 (sandbox harness) now includes the microVM
  execution path with a red-team escape attempt as a gate test;
  public third-party primitive submission does not open before M290
  passes.
