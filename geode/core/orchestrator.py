"""M210 — the orchestrator: register -> route -> attribute -> record,
over ANY heterogeneous arm set, with every decision hash-chained.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(19 Aug 2026, before the build). Thin and deterministic: it binds the
existing pieces (M171 router, M185 ledger, M205 admission) without
adding new selection logic. No RNG anywhere.
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

from geode.core.arm import validate_arm_spec
from geode.core.ledger import AppendOnlyLedger
from geode.core.router import Router
from geode.hashing import canonical_json, payload_hash


class Orchestrator:
    """Deterministic serve loop over an append-only arm registry.

    register(): validate the arm spec, add it to the router, and
    record the registration on the ledger.
    serve(): route a query fingerprint, record the decision on the
    ledger, return the top-k arm records (with replay handles).
    attribute(): orchestration-level LOO over arms with
    V(S) = best-arm held-out accuracy (registered coarse value; the
    fine-grained coalition attribution is M180/M181).
    """

    def __init__(self) -> None:
        self.router = Router()
        self.ledger = AppendOnlyLedger()
        self._sizes: dict[str, dict[str, Any]] = {}
        # M270 C2: registered rollout policies, latest per arm group
        self._rollout: dict[str, dict[str, Any]] = {}
        self._policy_versions: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------- lifecycle
    def register(self, arm_spec: dict[str, Any],
                 refusal_records: list | None = None) -> int:
        """Validate, augment (M247 wiring: measured-tag assembly is
        adds-only and happens BEFORE admission), add to the router,
        and record the registration on the ledger."""
        spec = dict(arm_spec)
        if refusal_records:
            from geode.core.refusal import augment_measured_tags
            spec = augment_measured_tags(spec, list(refusal_records))
        reasons = validate_arm_spec(spec)
        if reasons:
            raise ValueError(f"arm {spec.get('arm_id')!r} invalid: "
                             f"{reasons}")
        arm_id = str(spec["arm_id"])
        self.router.add_arm(spec)
        self._sizes[arm_id] = {"param_count": spec["param_count"],
                               "size_bytes": spec["size_bytes"],
                               "kind": spec["kind"]}
        acc = spec["held_out_accuracy"]
        overall = float(acc["all"] if isinstance(acc, dict) else acc)
        return self.ledger.append({
            "kind": "arm_register",
            "key": f"arm:{arm_id}",
            "arm_id": arm_id,
            "held_out_accuracy": overall,
            "measured_tags": list(spec.get("measured_tags") or []),
            "param_count": int(spec["param_count"]),
            "size_bytes": int(spec["size_bytes"]),
        })

    def serve(self, query_id: str, fingerprint: Sequence[float],
              k: int = 1, task_id: str | None = None,
              contract_kind: str | None = None,
              freeze: Any = None,
              as_of_index: int | None = None,
              ood_guard: Any = None,
              input_vec: Sequence[float] | None = None,
              cache: dict[str, Any] | None = None,
              requester: str | None = None,
              max_unit_price: float | None = None,
              best_quality: bool = False
              ) -> list[dict[str, Any]]:
        """Route one query through the registered failover chain
        (fingerprint-matched arms, then general arms by per-task
        selection score); the top-k decision is ledger-recorded.
        Returns the routed arm records in deterministic order.

        M255 (the gap-audit fix): the containment controls forward
        to the chain — a frozen registry (M248) or a guarded
        out-of-distribution input (M251) yields an EMPTY route,
        which is still ledger-recorded (an empty decision is a
        decision, and the audit needs it).

        M270 C1 (cache, containment-first): with a `cache` dict, an
        eligible route is looked up by hash-keyed digest before its
        route record is written; a hit appends a `cache_hit` record
        and returns the stored decision. An EMPTY chain never
        consults the cache. The digest covers the registry content
        hash and the current policy version, so any arm or policy
        change invalidates entries.

        M270 C2 (canary): a registered rollout policy for the chosen
        arm's group deterministically buckets the query fingerprint;
        a bucket inside the canary permille re-maps the choice. Every
        applied policy is a `rollout` ledger record (stable outcomes
        included).

        M270 C4: `requester` is attached to the route record only —
        no selection logic reads it (identity never routes)."""
        chain = self.router.chain(fingerprint, task_id=task_id,
                                  contract_kind=contract_kind,
                                  freeze=freeze,
                                  as_of_index=as_of_index,
                                  ood_guard=ood_guard,
                                  input_vec=input_vec,
                                  max_unit_price=max_unit_price,
                                  best_quality=best_quality)
        routed = chain[:max(k, 1)]
        policy_version = self._latest_policy_version()
        digest: str | None = None
        if cache is not None and routed:
            digest = self._cache_digest(fingerprint, task_id,
                                        contract_kind, policy_version)
            entry = cache.get(digest)
            if entry is not None:
                self.ledger.append({
                    "kind": "cache_hit",
                    "key": f"cache:{digest}:{query_id}",
                    "query_id": query_id,
                    "source_route_index": entry["route_index"],
                    "source_route_hash": entry["route_hash"],
                    "registry_hash": self.router.content_hash(),
                    "policy_version": policy_version,
                })
                return list(entry["decision"])
        routed, rollout = self._apply_rollout(routed, fingerprint)
        route_index = self.ledger.append({
            "kind": "route",
            "key": f"route:{query_id}",
            "query_id": query_id,
            "task_id": task_id,
            "contract_kind": contract_kind,
            "requester": requester,
            "policy_version": policy_version,
            "contained": bool(freeze is not None
                              and as_of_index is not None
                              and freeze.is_frozen(as_of_index)),
            "chosen": [{"arm_id": rec["arm_id"],
                        "route_cos": rec.get("route_cos"),
                        "replay_handle": self.replay_handle(rec["arm_id"])}
                       for rec in routed],
        })
        if rollout is not None:
            self.ledger.append({
                "kind": "rollout",
                "key": f"rollout:{query_id}",
                "query_id": query_id,
                **rollout,
            })
        if cache is not None and routed and digest is not None:
            route_rec = self.ledger.get(f"route:{query_id}")
            self.ledger.append({
                "kind": "cache_store",
                "key": f"cache_store:{digest}:{query_id}",
                "query_id": query_id,
                "registry_hash": self.router.content_hash(),
                "policy_version": policy_version,
            })
            cache[digest] = {
                "decision": [dict(rec) for rec in routed],
                "route_index": route_index,
                "route_hash": route_rec["hash"] if route_rec else None,
            }
        return routed

    # ----------------------------------------------------- M270 C1/C2 helpers
    def _latest_policy_version(self) -> str | None:
        return next(reversed(self._policy_versions), None)

    def _cache_digest(self, fingerprint: Sequence[float],
                      task_id: str | None,
                      contract_kind: str | None,
                      policy_version: str | None) -> str:
        """C1: the hash-keyed digest, CONTENT-based — the query_id is
        an event id, not content; the same content re-served under a
        new id must still hit. Covers the query content and the
        CURRENT registry + policy state, so any arm registration or
        policy change invalidates every stored entry."""
        return payload_hash(canonical_json({
            "fingerprint": [float(v) for v in fingerprint],
            "task_id": task_id,
            "contract_kind": contract_kind,
            "registry": self.router.content_hash(),
            "policy": policy_version,
        }))

    @staticmethod
    def rollout_bucket(fingerprint: Sequence[float], arm_group_id: str,
                       policy_version: str) -> int:
        """C2: deterministic permille bucket over [0, 1_000_000).
        Same (fingerprint, group, version) -> same bucket, across
        restarts. No RNG anywhere."""
        payload = (canonical_json([
            [float(v) for v in fingerprint], arm_group_id, policy_version,
        ]) + ":").encode("utf-8")
        return int.from_bytes(
            hashlib.sha256(payload).digest()[:4], "big") % 1_000_000

    def register_rollout_policy(self, policy_version: str,
                                arm_group_id: str,
                                stable_arm_id: str,
                                canary_arm_id: str,
                                canary_permille: int) -> int:
        """C2: register a canary policy as a ledger record (append-only
        versioning). Promotion is a new policy record, never an edit."""
        if not 0 <= canary_permille <= 1_000_000:
            raise ValueError("canary_permille must be in [0, 1000000]")
        if stable_arm_id == canary_arm_id:
            raise ValueError("stable and canary must differ")
        for arm_id in (stable_arm_id, canary_arm_id):
            if arm_id not in self.router._arms:
                raise ValueError(f"unknown arm {arm_id!r} in rollout "
                                 f"policy")
        policy = {"policy_version": policy_version,
                  "arm_group_id": arm_group_id,
                  "stable_arm_id": stable_arm_id,
                  "canary_arm_id": canary_arm_id,
                  "canary_permille": int(canary_permille)}
        index = self.ledger.append({
            "kind": "rollout_policy",
            "key": f"policy:{policy_version}",
            **policy,
        })
        self._rollout[arm_group_id] = policy
        self._policy_versions[policy_version] = policy
        return index

    def _apply_rollout(
            self, records: list[dict[str, Any]],
            fingerprint: Sequence[float]
            ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """C2 serve integration: if the chosen arm is the stable arm
        of a registered policy group, bucket the query; a bucket
        inside the canary permille re-maps the choice to the canary
        arm. Returns the (possibly re-mapped) records and the rollout
        note (None when no policy applied). The first policy whose
        group matches applies, in deterministic group order."""
        if not records:
            return records, None
        for group in sorted(self._rollout):
            policy = self._rollout[group]
            if records[0]["arm_id"] != policy["stable_arm_id"]:
                continue
            bucket = self.rollout_bucket(fingerprint, group,
                                         policy["policy_version"])
            effective = records[0]["arm_id"]
            if bucket < policy["canary_permille"]:
                canary = dict(self.router._arms[policy["canary_arm_id"]])
                canary["route_cos"] = records[0].get("route_cos")
                canary["ranked_by"] = "rollout_canary"
                canary["arm_id"] = policy["canary_arm_id"]
                records[0] = canary
                effective = policy["canary_arm_id"]
            note = {"bucket": bucket,
                    "policy_version": policy["policy_version"],
                    "arm_group_id": group,
                    "stable_arm_id": policy["stable_arm_id"],
                    "canary_arm_id": policy["canary_arm_id"],
                    "effective_arm_id": effective}
            return records, note
        return records, None

    # -------------------------------------------------------- M270 C3/C4
    def stream_begin(self, query_id: str, route_record_index: int,
                     seed: Any, policy_version: str | None = None) -> int:
        """C3: open a stream for a routed query. The seed is recorded
        so a replay attempt is well-defined (bit-exactness of the
        generated content is NOT claimed — integrity only)."""
        return self.ledger.append({
            "kind": "stream_begin",
            "key": f"stream:{query_id}",
            "query_id": query_id,
            "route_record_index": int(route_record_index),
            "seed": str(seed),
            "policy_version": policy_version,
        })

    def stream_chunk(self, query_id: str, seq: int,
                     payload_hash_value: str) -> int:
        """C3: one chunk of a stream — payload-hashed, chain-recorded."""
        return self.ledger.append({
            "kind": "stream_chunk",
            "key": f"stream:{query_id}:{int(seq)}",
            "query_id": query_id,
            "seq": int(seq),
            "payload_hash": str(payload_hash_value),
        })

    def stream_end(self, query_id: str, total_chunks: int,
                   final_payload_hash: str, status: str) -> int:
        """C3: close a stream. `status` is 'complete' or 'aborted' —
        an incomplete stream is a ledger event, never a silent drop."""
        if status not in ("complete", "aborted"):
            raise ValueError("stream status must be complete|aborted")
        return self.ledger.append({
            "kind": "stream_end",
            "key": f"stream:{query_id}:end",
            "query_id": query_id,
            "total_chunks": int(total_chunks),
            "final_payload_hash": str(final_payload_hash),
            "status": status,
        })

    def record_auth(self, query_id: str, requester: str, outcome: str,
                    nonce: str) -> int:
        """C4: the API layer records the verdict of signed-request
        verification. Outcome is one of ok|bad_signature|
        replayed_nonce|expired. Identity enters the ledger, never
        the routing path."""
        if outcome not in ("ok", "bad_signature", "replayed_nonce",
                           "expired"):
            raise ValueError(f"unknown auth outcome {outcome!r}")
        return self.ledger.append({
            "kind": "auth",
            "key": f"auth:{nonce}",
            "query_id": query_id,
            "requester": requester,
            "outcome": outcome,
            "nonce": nonce,
        })

    def replay_handle(self, arm_id: str) -> str | None:
        """The deterministic replay anchor of an arm: the M205 replay
        hash for DNN arms, the sealed evidence source for heads."""
        arm = self.router._arms.get(arm_id)
        if arm is None:
            return None
        if arm.get("kind") == "dnn":
            return arm.get("replay_hash")
        return arm.get("sealed_source")

    def admit_behavior_update(self, arm_id: str,
                              vector: Sequence[float],
                              attestations: frozenset[str],
                              ledger_index: int,
                              behavior_gate: Any,
                              bound: float | None = None,
                              ) -> dict[str, Any]:
        """M250 wiring (the M255-registered pending): behaviour-diff
        admission. The update is snapshotted through the gate — a
        below-quorum snapshot is quarantined, a drifted update is
        REJECTED (raises ValueError), an admitted update becomes
        the new baseline. Every decision is a ledger receipt (G5)."""
        decision = behavior_gate.admits_update(arm_id, vector,
                                               ledger_index,
                                               bound=bound)
        self.ledger.append({
            "kind": "behavior_update",
            "key": f"behavior:{arm_id}:{ledger_index}",
            "arm_id": str(arm_id),
            "ledger_index": int(ledger_index),
            "admitted": bool(decision["admitted"]),
            "reason": decision["reason"],
            "attestations": sorted(attestations),
        })
        if not decision["admitted"]:
            raise ValueError(
                f"behaviour update for {arm_id!r} rejected: "
                f"{decision['reason']} (M250)")
        return decision

    # ------------------------------------------------------------ attribution
    def attribute(self) -> dict[str, Any]:
        """Orchestration-level leave-one-out over the registered arms:
        V(S) = best-arm held-out accuracy (registered coarse value)."""
        arms = self.router.list_arms()

        def overall(arm_id: str) -> float:
            acc = self.router._arms[arm_id]["held_out_accuracy"]
            return float(acc["all"] if isinstance(acc, dict) else acc)

        acc = {a: overall(a) for a in arms}
        v_all = max(acc.values()) if acc else 0.0
        loo: dict[str, float] = {}
        for a in arms:
            rest = max((acc[b] for b in arms if b != a), default=0.0)
            loo[a] = v_all - rest
        order = sorted(arms, key=lambda a: (-acc[a], a))
        return {"arms": arms, "v_all": v_all,
                "loo_marginals": loo,
                "ranking": order,
                "reading": ("coarse by design: a non-top arm has LOO 0; "
                            "the top arm's LOO is its lead over the second "
                            "best")}

    # ---------------------------------------------------------------- checks
    def chain_verify(self) -> dict[str, Any]:
        return self.ledger.verify()

    def content_hash(self) -> str:
        return self.router.content_hash()

    def size_table(self) -> list[dict[str, Any]]:
        return [{"arm_id": a, **self._sizes[a]}
                for a in sorted(self._sizes)]
