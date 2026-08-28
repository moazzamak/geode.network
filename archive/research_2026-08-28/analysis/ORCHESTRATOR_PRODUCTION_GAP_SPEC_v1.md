# ORCHESTRATOR PRODUCTION-GAP SPEC v1 (M264 deliverable)

**21 August 2026. Spec only — no code shipped by this document.**
This is the deliverable of M264 ("orchestrator production-gap
spec", registered 21 Aug in
`analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`, NOT dispatched for
implementation). It specifies six production conveniences
identified in the gap analysis
(`analysis/PRODUCTION_READINESS_AND_ARMS_PLAN_v1.md` §2): response
caching, canary/rollout, streaming, signed-request auth, batching,
and the arm-record license field. Every component is specified as
deterministic rules, ledger record contracts, and acceptance
gates. Implementation cells (future, unnumbered) must carry these
gates and write evidence to
`logs/results/v25/m264_production_gap/evidence.json`.

---

## 1. Grounding — what this spec plugs into (v0.15.0 at spec

time; the M270 build lands in v0.16.0)

- `geode.core.orchestrator.Orchestrator.serve()` — the deterministic
  serve loop: route → ledger-record → return top-k. All six
  components modify or sit beside this loop; none replaces it.
- `geode.core.router.Router.chain()` — the failover chain with
  containment controls (freeze M248, OOD guard M251): a frozen or
  guarded query yields an EMPTY route, still ledger-recorded
  (M255). **The containment-first rule binds every component below.**
- `geode.core.ledger.AppendOnlyLedger` — append-only, hash-chained,
  timing fields excluded from content hashes (standing rule).
- `geode.core.arm.validate_arm_spec()` — the registration gate;
  C6 extends it.
- Arm records: `availability` (healthy flag), `price`,
  `held_out_accuracy` (per-task dict), `selection_accuracy`,
  `primitive`, `kind` (`dnn` | `sealed_head` | primitive form).

Standing rules that bind all six components:

1. **No wall clocks in any content hash.** Timing, nonces, and
   timestamps live outside hashes (they belong to signatures and
   metadata only).
2. **No RNG anywhere in the control plane.** Every split, bucket,
   and key derives from registered hashes over the payload.
3. **Containment first.** Nothing the cache, canary, or auth
   layers do may bypass or reorder the M255 containment controls.
4. **Identity never routes.** Requester identity is a quota key
   and a ledger field; it never enters selection scores or
   fingerprints.
5. **Every decision is a ledger record.** Cache hits, bucket
   assignments, and stream events are decisions and are recorded
   like routes.

---

## 2. C1 — hash-keyed response-cache tier (gap 7)

**Design.** The cache stores _routing decisions_ (which arm won and
with what handle), not arbitrary outputs. Lookup happens in
`serve()` only after the containment controls have passed:

1. fingerprint computed; freeze/OOD guards run exactly as today
   (an empty route stays empty — a cache hit must be impossible
   while containment says no);
2. cache key computed from the canonical query plus the _current
   registry state_;
3. hit → append a `cache_hit` ledger record and return the stored
   decision; miss → route as today, store the decision, and append
   a `cache_store` record.

**Cache key (deterministic).**

```
cache_key = payload_hash(json_canonical({
    "query":     payload_hash(query payload),
    "task_id":   task_id,
    "contract":  contract_kind,
    "registry":  router.content_hash(),   # any arm change invalidates
    "policy":    rollout_policy_version,  # C2
}))
```

Including the registry content hash means any arm registration or
update invalidates every entry — conservative, correct at this
scale, and auditable. A policy bump (C2) also invalidates.

**Ledger contracts.**

```
{"kind": "cache_hit", "key": "cache:<digest>:<query_id>",
 "query_id": ..., "source_route_index": ..., "source_route_hash": ...,
 "registry_hash": ..., "policy_version": ...}

{"kind": "cache_store", "key": "cache_store:<digest>:<query_id>",
 "query_id": ..., "registry_hash": ..., "policy_version": ...}
```

M270 build amendment: the ledger key is unique per EVENT — the bare
digest would collide on a repeated query against the append-only
ledger's unique-key rule, so `query_id` is appended to the key.

M270 build amendment (2): the digest is content-based — it covers
the query fingerprint, task, contract, registry content hash, and
policy version, and deliberately NOT the query_id, so the same
content re-served under a new event id still hits.

The stored entry shape:

```
{digest: {"decision": [routed records],
          "route_index": ..., "route_hash": ...}}
```

**Output-caching boundary (honest).** Caching arm _outputs_ is
admitted only for `primitive` arms and `sealed_head` arms
(deterministic by construction), and only behind a determinism
certificate; `dnn` arm outputs are never cached — the v13 precedent
measured GPU/CPU disagreement, and no claim of cross-hardware
bit-stability may be assumed. This spec caches decisions only.

**Acceptance criteria.** C1.1 a hit never precedes containment
checks (test: frozen registry → empty route → no `cache_hit`
record). C1.2 any registry change flips the key (test: register an
arm, old key misses). C1.3 hit/store records replay under
`ledger.verify()`.

---

## 3. C2 — canary/rollout policy (gap 6)

**Design.** Rollout is a registered _policy_ over a named arm
group: a stable arm and a canary arm, with a permille split
computed deterministically from the query fingerprint.

```
bucket = int.from_bytes(
    sha256(canonical_fingerprint_bytes || b":" ||
           arm_group_id.encode() || b":" ||
           str(policy_version).encode())[:4], "big") % 1_000_000
```

The policy is itself a ledger record (append-only versioning):

```
{"kind": "rollout_policy", "key": "policy:<version>",
 "policy_version": ..., "arm_group_id": ...,
 "stable_arm_id": ..., "canary_arm_id": ...,
 "canary_permille": ...}
```

**Serve integration.** After `chain()` returns candidates, if a
policy applies to the chosen arm's group, bucket the query; a
bucket inside `canary_permille` re-maps the choice to the canary
arm. The decision is ledger-recorded:

```
{"kind": "rollout", "key": "rollout:<query_id>",
 "query_id": ..., "bucket": ..., "policy_version": ...,
 "stable_arm_id": ..., "canary_arm_id": ...,
 "effective_arm_id": ...}
```

**Promotion is a measured act.** Advancing or closing a canary
requires a registered reading comparing canary vs stable on
held-out traffic buckets (M250-class behaviour-diff or best-arm
protocol), registered before the promotion; the promotion is a new
`rollout_policy` record — a permille change or retirement, never a
silent edit. No promotion by wall clock.

**Bound.** Policies never modify selection scores or fingerprints;
they re-map the final choice for a registered group only.
Multi-arm canaries partition the permille space deterministically
under the same rule.

**Acceptance criteria.** C2.1 same (fingerprint, group, version) →
same bucket across restarts. C2.2 every re-mapping is a `rollout`
record. C2.3 promotion requires a registered comparison record;
the spec of that record is fixed before the first canary runs.

---

## 4. C3 — streaming response contract (gap 10)

**Design.** One query → one ordered record sequence:
`stream_begin`, N × `stream_chunk`, `stream_end`. The ledger
chain already makes the sequence tamper-evident; per-chunk
payload hashes additionally support prefix verification and
resumption.

```
{"kind": "stream_begin", "key": "stream:<query_id>",
 "query_id": ..., "route_record_index": ..., "seed": ...,
 "policy_version": ...}

{"kind": "stream_chunk", "key": "stream:<query_id>:<seq>",
 "query_id": ..., "seq": ..., "payload_hash": ...}

{"kind": "stream_end", "key": "stream:<query_id>:end",
 "query_id": ..., "total_chunks": ..., "final_payload_hash": ...,
 "status": "complete" | "aborted"}
```

**Determinism honesty.** The chunking rule (max bytes/tokens per
chunk) is part of the registered contract and is deterministic.
The _content_ of a generative arm is not: generated text is not
bit-reproducible across hardware. The hash chain therefore
verifies **integrity of what was served**, not regeneration; model
identity comes from the route record's `replay_handle`. The
sampling seed is recorded in `stream_begin` so a replay attempt is
well-defined even when bit-exactness is not guaranteed.

**Aborts are visible.** A cancelled or failed stream still writes
`stream_end` with `status: "aborted"` — an incomplete stream is a
ledger event, never a silent drop.

**Acceptance criteria.** C3.1 chunk records verify end-to-end
under `ledger.verify()`. C3.2 every aborted stream has a terminal
record. C3.3 chunk boundaries are a pure function of the payload
and the registered chunk size.

---

## 5. C4 — signed-request auth (gap 11)

**Design.** Ed25519-class signatures over canonical request bytes:

```
signed_bytes = method || path || payload_hash || nonce ||
               not_before || not_after
```

Verification rejects invalid signatures, replayed nonces (a
registered nonce store), and expired windows. The verified
identity becomes the `requester` field on the route ledger record
and the key for quota enforcement — which happens **pre-route**
(reject cheap requests before expensive work).

**The hard rule: identity never routes.** The routing path does
not receive `requester` at all — structurally. Identity attaches
to ledger records and quota checks only. No per-user fingerprints,
no per-user selection scores, no per-user chain reordering. This
is a separation enforced by function signature, not a policy
promise.

**Ledger contract.**

```
{"kind": "auth", "key": "auth:<nonce>", "query_id": ...,
 "requester": "...", "outcome": "ok" | "bad_signature" |
 "replayed_nonce" | "expired"}
```

Route records gain `"requester": "..."` (metadata, hash-covered).

**Acceptance criteria.** C4.1 no routing function signature
receives `requester` (static check + test). C4.2 nonce replay is
rejected and recorded. C4.3 timestamps/nonces never enter content
hashes (standing rule).

---

## 6. C5 — batching note (gap 9)

**Policy (note, not a build).** The auditable unit stays one
query, one decision. Batching is admitted only for the
_arithmetic_ — one matmul over N input vectors — never for the
decision logic, and the containment guards run per item.

**Float honesty.** Matrix multiplication is non-associative:
batched vs single-item evaluation can differ in the last bits. The
registered rule: batched arithmetic must reproduce single-item
results bit-exactly (anchor-gate machinery at the standing
tolerance), or the batch reduction order is registered and the
difference bound recorded. Default: per-item arithmetic; batching
is an optimization admitted only with a bit-exactness
certificate.

**Record shape when batched.** Route records gain
`"batch_id": ..., "batch_position": ...` (hash-covered provenance);
one ledger record per item remains.

**Acceptance criteria.** C5.1 per-item records never merge under a
batch. C5.2 a batched run without a bit-exactness certificate is a
registered deviation, never silent.

---

## 7. C6 — arm-record license field (audit action 1)

**Design.** Every arm spec gains a required license object:

```
"license": {"code": "<SPDX or provenance>",
            "weights": "<SPDX or provenance>",
            "data": "<SPDX or provenance>"}
```

Empty string = not applicable (e.g., `data` for a head trained on
sealed private rows). `validate_arm_spec()` (the C6 gate) rejects
any spec without a license object with exactly those three keys.
Existing registered arms must be re-registered with the field
before the next admission wave (M261+).

**Uses.** (a) the `arm_register` ledger record carries the license
object — licenses at registration are tamper-evident; (b) registry
queries filter by license standing (e.g., all Tier-3-safe arms),
feeding the permissive-only stance and the future M269 merit
selection, where license standing is a registered metric; (c) the
audit (`analysis/LICENSING_AUDIT_v1.md`) reads it as its
action-1 deliverable.

**Acceptance criteria.** C6.1 invalid specs rejected at
registration. C6.2 `arm_register` records carry the object and
verify under the chain. C6.3 a license query returns the
Tier-3-safe subset deterministically.

---

## 8. Consolidated acceptance gates

- **G1 record replay** — every new `kind` verifies under
  `ledger.verify()`; timing excluded (standing rule).
- **G2 containment-first caching** — a cache hit is impossible
  before freeze/OOD checks; a frozen registry never serves cached
  decisions.
- **G3 canary determinism** — same (fingerprint, group, version) →
  same bucket across restarts; split reproducible from a recorded
  seed corpus.
- **G4 identity never routes** — static + test check that no
  routing function signature receives `requester`.
- **G5 license field required** — invalid spec rejected; audit
  action 1 satisfied at registration.
- **G6 stream replay** — chunk hashes verify end-to-end; aborted
  streams terminal-recorded.
- **G7 anchor hygiene** — cache keys, buckets, and chunk rules
  reproduce against registered vectors before any production
  number is read (anchor-first).

---

## 9. Honest limits

- **Spec only.** No implementation is claimed or shipped here;
  future implementation cells carry G1-G7 and write evidence to
  `logs/results/v25/m264_production_gap/evidence.json`.
- **Not specified here** (remaining gap-analysis rows, solvable but
  deliberately out of scope): #8 load balancing/autoscaling
  (deployment-topology decision), #12 timeouts and cancellation,
  #13 multi-tenancy beyond C4's `requester` field.
- **No output caching for `dnn` arms** without a determinism
  certificate (v13 GPU/CPU disagreement precedent).
- **Streaming verifies integrity, not regeneration** — generative
  content is not bit-reproducible across hardware; the seed and
  replay handle are recorded so the claim is well-defined.

---

## 10. Prior art — none of these components is original

Every mechanism specified here is established practice; GEODE's
contribution is their deterministic, ledger-audited composition,
not their invention:

- Canary/gradual traffic rollout: Google SRE practice (Beyer,
  Jones, Petoff & Murphy 2016, _Site Reliability Engineering_,
  O'Reilly); deterministic fingerprint-hash traffic bucketing is
  the standard A/B-assignment technique.
- Hash-chained, tamper-evident records: Haber & Stornetta 1991
  (J. Cryptology 3(2)) — the origin of chained timestamping.
- Ed25519 signatures: Bernstein, Duif, Lange, Schwabe & Yang 2012
  (J. Cryptogr. Eng. 2(2)).
- Semantic response caching for LLM systems: GPTCache ([Bang
  2023](https://arxiv.org/abs/2306.17543)).
- Per-chunk integrity by chaining chunk hashes: the same
  Haber-Stornetta primitive, applied per chunk.
- Arm-record licensing metadata: SPDX practice
  (https://spdx.org) — C6 field values follow SPDX identifiers
  where one exists.

Any claim this document makes is about composition and audit, not
about inventing caching, canarying, signing, or hashing.
