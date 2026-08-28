# M201 — Encrypted session + poisoning protocol spec

Registered 19 Aug 2026 (C3 of the hardening considerations). This spec
turns §C3 into a protocol with concrete message flows, evidence
formats, and the H9 gate. Reference implementation is queued; this
document is the registered contract the build must satisfy.

## 1. Session flow (encrypted end-to-end)

Participants: **U** (user), **S** (selected server), **R** (router/
registry), **V** (validators).

1. **Request.** U prepares `payload = task_id || inputs`, encrypts it
   to S's public key: `cipher = Enc(sk_U → pk_S, payload)` (hybrid
   X25519 + ML-KEM-1024 per §C2). U submits
   `(session_id, pk_S, cipher, fee_proof)` to R. R never sees inputs.
2. **Selection.** R runs the registered selection score (measured
   accuracy × availability × price, H8-gated) and commits the choice
   to the ledger: `route_event = (session_id, pk_S, score, hash(cipher),
timestamp)`.
3. **Serving.** S decrypts, runs the frozen encoder + head (or DNN
   component) LOCALLY — the model never ships — and produces scores.
4. **Response.** S encrypts the scores to U's key:
   `resp = Enc(sk_S → pk_U, scores)` and returns
   `(session_id, resp, output_commitment = H(scores || session_id))`.
   Colluding third parties learn nothing: ciphertexts are hybrid-PQ,
   and only U holds the response key.
5. **Settlement.** R records the served event
   `(session_id, pk_S, output_commitment, measured_ok, fee_split)`
   for attribution and payout (credit ledger, M207).

Local-encode option (P1 stage 0): U runs the frozen encoder locally
and submits features instead of raw inputs — then no party ever sees
raw user data, and S's job reduces to the head matmul.

## 2. Liveness protocol

- **Deterministic health probes:** V periodically sends a challenge
  with a known-reference payload hash; S must return the matching
  output commitment within the registered deadline. Probes are cheap,
  cover-free, and validator-measured ONLY (H8: self-reports are
  ignored).
- **Availability ledger:** `available(S) = probes_ok / probes_due`
  over the last registered window, written by V, not S.
- **Pricing of downtime:** thaw/payout keys on actually served
  sessions (§4.10) — downtime earns nothing automatically.
- **Failover:** R maintains an ordered failover chain per task region;
  on probe failure or deadline miss the next server is selected.

## 3. Poisoning detection and conviction

Threat: S returns random/bad outputs.

1. **Redundant sampling.** A registered fraction ρ of sessions is
   double-routed to two independent servers; mismatching output
   commitments flag the pair (evidence: both commitments + both
   responses opened under V's supervision).
2. **Held-out probes.** V submits tasks with known-reference outputs
   (from the sealed test sets); a mismatching commitment is evidence.
3. **Conviction.** A conviction record requires cryptographic
   evidence: `(session_id, output_commitment, opened_response,
reference)` whose mismatch is verifiable by anyone replaying the
   registered model hash. No validator discretion — the check is
   deterministic (H6 replay).
4. **Penalty.** Conviction ⇒ exclusion from selection for a registered
   period + slash of the server's stake scaled to attested damage
   (the vesting/credit ledger's slash path). Downgrade is automatic
   and evidence-bound; an un-convicted server is never punished.
5. **H9 gate (registered):** on the registered scenario set, the
   mechanism convicts a registered poisoner within a registered
   session budget and excludes no honest server. The gate runs in
   simulation BEFORE any deployment.

## 4. Copy protection on the wire

Only scores ever leave S (step 4). The encoder/head/DNN artifacts
stay on S; U receives model outputs, not model parameters. Honest
limits (§4.8, §C2) apply unchanged: enough queries can distill a
student, and nothing stops a rogue S from copying what it serves —
deterrence is economic and legal.

## 5. Registered parameters (defaults, all gate-swept)

- probe deadline: 30 s; probe window: 1 h; availability floor: 0.9
- redundant-sampling fraction ρ: 0.05
- conviction evidence threshold: 2 independent mismatches
- exclusion period: 7 days; slash scale: proportional to attested
  sessions affected
- encryption: AES-256-GCM payload, hybrid X25519+ML-KEM key
  agreement, Ed25519+ML-DSA signatures on route events
