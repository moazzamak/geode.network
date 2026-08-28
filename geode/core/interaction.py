"""GEODE interaction layer v0 (v25 M269, L1 plan-then-execute).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md``
(21 Aug 2026). Cells (a)-(f) with the registered gates:

- G1 plan validation rejects unknown arms/contracts before execution;
- G2 structural: the planner's output schema has NO fingerprint field
  (the fingerprint is computed by the registered fingerprint service,
  never the LLM);
- G3 cached plans replay from their payload hash;
- G4 merit ranking reproducible from a ledger snapshot (as-of index);
- G5 every selection decision is a ledger receipt carrying the metrics
  it was based on;
- G6 prompt-injection guard on the planner's inputs (structural:
  registered marker rejection — an honest surface guard, not a
  security claim).

BOUNDARY: L1 only — L2 bounded autonomy deferred; identity never
enters routing logic (the task-spec schema carries no identity field
and rejects one). The interface LLM is an injectable callable; the
production executor is the Qwen arm wrapped elsewhere (this module is
deterministic and GPU-free).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Any, Callable

from geode.hashing import payload_hash

# The registered plan schema: task_type, inputs, output_contract,
# constraints. No fingerprint field, no identity field (G2 / L1
# boundary).
TASK_SPEC_KEYS = {"task_type", "inputs", "output_contract",
                  "constraints"}
FORBIDDEN_KEYS = {"fingerprint", "identity", "route"}

# The registered injection-marker list (surface guard, honest scope).
INJECTION_MARKERS = (
    "ignore previous instructions",
    "disregard all prior",
    "system prompt override",
)

# Registered structural guard limits (the M263 family).
MAX_INTENT_CHARS = 2000
MAX_LLM_RESPONSE_CHARS = 8000


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the JSON object from an LLM response that may wrap it in
    markdown fences or prose (registered parser repair, 22 Aug: the
    raw-response parser read every real Qwen response as non-JSON)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(stripped[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


@dataclass
class TaskSpec:
    """Typed task spec. By construction it has no fingerprint field."""
    task_type: str
    inputs: dict[str, Any] = field(default_factory=dict)
    output_contract: str = "label"
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TaskSpec":
        unknown = set(raw) - TASK_SPEC_KEYS
        if unknown:
            raise ValueError(f"unknown task-spec keys: {sorted(unknown)}")
        for key in FORBIDDEN_KEYS:
            if key in raw:
                raise ValueError(f"forbidden key in task spec: {key!r}")
        def _mapping(value: Any, name: str) -> dict[str, Any]:
            try:
                return dict(value)
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be a mapping") from None
        return cls(
            task_type=str(raw["task_type"]),
            inputs=_mapping(raw.get("inputs", {}), "inputs"),
            output_contract=str(raw.get("output_contract", "label")),
            constraints=_mapping(raw.get("constraints", {}),
                                 "constraints"),
        )


@dataclass
class Plan:
    """A validated plan: the typed spec plus the registry-checked arm."""
    task_spec: TaskSpec
    arm: str
    payload_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {"task_spec": self.task_spec.to_dict(),
                "arm": self.arm, "payload_hash": self.payload_hash}


def guard_intent(intent: str) -> dict[str, Any]:
    """The registered G6 structural guard. Returns an admit record."""
    if not isinstance(intent, str) or not intent.strip():
        return {"admitted": False, "reason": "empty"}
    if len(intent) > MAX_INTENT_CHARS:
        return {"admitted": False, "reason": "too_long"}
    if not intent.isprintable():
        return {"admitted": False, "reason": "not_printable"}
    low = intent.lower()
    for marker in INJECTION_MARKERS:
        if marker in low:
            return {"admitted": False, "reason": "injection_marker",
                    "marker": marker}
    return {"admitted": True}


class PlanCache:
    """Hash-keyed plan store. Plans replay from their payload hash
    (G3); a read verifies the stored payload against its key."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def put(self, plan: Plan) -> str:
        key = payload_hash(json.dumps(plan.to_dict(), sort_keys=True,
                                      separators=(",", ":")))
        self._store[key] = plan.to_dict()
        return key

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the stored plan iff its payload still hashes to the
        key (tamper-evident replay)."""
        stored = self._store.get(key)
        if stored is None:
            return None
        actual = payload_hash(json.dumps(stored, sort_keys=True,
                                         separators=(",", ":")))
        return stored if actual == key else None


class PlanValidator:
    """G1: reject plans naming unknown arms or contracts."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def validate(self, raw_plan: dict[str, Any]) -> Plan | None:
        """Return a Plan or None; a None plan is an abstention (the
        caller records it)."""
        try:
            spec = TaskSpec.from_dict(raw_plan.get("task_spec", {}))
        except (ValueError, TypeError) as exc:
            self._last_reason = f"spec_invalid: {exc}"
            return None
        arm = raw_plan.get("arm")
        if not isinstance(arm, str) or not arm:
            self._last_reason = "arm_missing"
            return None
        try:
            entry = self._registry.get(arm)
        except (KeyError, TypeError):
            self._last_reason = f"arm_unknown: {arm}"
            return None
        # structural task-family check (registered G1 extension): an arm
        # may declare its task_types; a plan whose task_type is outside
        # the arm's family is rejected model-independently.
        if isinstance(entry, dict) and entry.get("task_types"):
            if spec.task_type not in entry["task_types"]:
                self._last_reason = (
                    f"task_arm_mismatch: {spec.task_type} not in "
                    f"{entry['task_types']} for {arm}")
                return None
        contract = spec.output_contract
        if contract not in ("label", "text", "generative", "exact"):
            self._last_reason = f"contract_unknown: {contract}"
            return None
        plan = Plan(task_spec=spec, arm=arm, payload_hash="")
        plan.payload_hash = payload_hash(json.dumps(
            {"task_spec": spec.to_dict(), "arm": arm}, sort_keys=True,
            separators=(",", ":")))
        return plan

    @property
    def last_reason(self) -> str:
        return getattr(self, "_last_reason", "")


class IntentPlanner:
    """L1 plan-then-execute: intent -> guarded LLM call -> parsed plan
    -> registry validation -> cached plan. The LLM is injectable; the
    registered prompt is the only instruction surface."""

    PROMPT = (
        "Convert the following user intent into a task plan. Reply with "
        "a JSON object with exactly two keys: 'task_spec' (an object "
        "with keys task_type, inputs, output_contract, constraints) and "
        "'arm' — an arm id chosen ONLY from this registered list: "
        "{arms}. 'output_contract' must be one of: label, text, "
        "generative, exact (arithmetic answers use 'exact'). Do not "
        "invent arms or contracts; do not "
        "include any fingerprint, identity, or route fields. Output "
        "JSON only. Your reply MUST contain both the 'task_spec' and "
        "'arm' keys. If NO listed arm fits the intent, reply with "
        "exactly {{\"unsupported\": true}}. "
        "Example: {{\"task_spec\": {{\"task_type\": \"sentiment\", "
        "\"inputs\": {{\"text\": \"the review\"}}, "
        "\"output_contract\": \"label\", \"constraints\": {{}}}}, "
        "\"arm\": \"arm_sentiment\"}}"
        "\nIntent: {intent}"
    )

    def __init__(self, llm: Callable[[str], str],
                 validator: PlanValidator, cache: PlanCache,
                 arms: list[str] | None = None,
                 examples: dict[str, str] | None = None) -> None:
        self._llm = llm
        self._validator = validator
        self._cache = cache
        self._arms = sorted(arms) if arms else []
        # M279: per-task-type few-shot exemplars (registered before
        # the run); each value is an "Intent -> plan JSON" line.
        self._examples = dict(examples or {})

    def _prompt(self, intent: str) -> str:
        arm_descriptions = {
            "arm_sentiment": "sentiment classification (sentiment tasks)",
            "arm_maths": "exact arithmetic (arithmetic tasks)",
            "arm_logic": "boolean evaluation (logic tasks)",
        }
        arm_list = ", ".join(
            f"{a} ({arm_descriptions.get(a, 'task arm')})"
            for a in self._arms)
        prompt = self.PROMPT.format(intent=intent, arms=arm_list)
        if self._examples:
            lines = ["Examples:"]
            for task_type, example in sorted(self._examples.items()):
                lines.append(f"- [{task_type}] {example}")
            prompt = prompt.replace("Output JSON only.",
                                   "Output JSON only.\n" +
                                   "\n".join(lines))
        return prompt

    def plan(self, intent: str) -> dict[str, Any]:
        """Return {admitted, plan, cache_key, abstention}."""
        guard = guard_intent(intent)
        if not guard["admitted"]:
            return {"admitted": False, "plan": None, "cache_key": None,
                    "abstention": {"reason": guard["reason"],
                                   "guard": guard}}
        response = self._llm(self._prompt(intent))
        raw = {"raw_response": response[:500]}
        if len(response) > MAX_LLM_RESPONSE_CHARS:
            return {"admitted": False, "plan": None, "cache_key": None,
                    "abstention": {"reason": "llm_response_too_long",
                                   **raw}}
        parsed = extract_json_object(response)
        if parsed is None:
            return {"admitted": False, "plan": None, "cache_key": None,
                    "abstention": {"reason": "llm_response_not_json",
                                   **raw}}
        if parsed.get("unsupported") is True:
            return {"admitted": False, "plan": None, "cache_key": None,
                    "abstention": {"reason": "unsupported_intent",
                                   **raw}}
        plan = self._validator.validate(parsed)
        if plan is None:
            return {"admitted": False, "plan": None, "cache_key": None,
                    "abstention": {"reason":
                                   self._validator.last_reason,
                                   **raw}}
        key = self._cache.put(plan)
        return {"admitted": True, "plan": plan, "cache_key": key,
                "abstention": None}


def merit_rank(ledger_snapshot: dict[str, Any],
               arm_ids: list[str],
               metric_kind: str = "selection_metric",
               cold_start_share: int = 1) -> list[tuple[str, float]]:
    """G4: reproducible merit ranking from a ledger snapshot (as-of
    index). Fixed criteria: system-collected metrics only (records of
    the registered kind); the cold-start exploration share gives
    metric-less arms one slot each; the ranking is recomputed fresh
    from the snapshot every call (no incumbency lock)."""
    scores: dict[str, float] = {}
    for rec in ledger_snapshot.get("records", []):
        content = rec.get("content", {})
        if content.get("kind") != metric_kind:
            continue
        arm = content.get("arm")
        metric = content.get("metric")
        if arm in arm_ids and isinstance(metric, (int, float)):
            scores[arm] = scores.get(arm, 0.0) + float(metric)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    seen = {arm for arm, _ in ranked}
    for arm in sorted(set(arm_ids) - seen):
        ranked.append((arm, float(-cold_start_share)))
    return ranked


def selection_receipt(ledger: Any, ranked: list[tuple[str, float]],
                      metrics_used: dict[str, float],
                      reason: str) -> int:
    """G5: append the selection decision as a ledger receipt carrying
    the exact metrics it was based on."""
    return ledger.append({
        "kind": "selection_decision",
        "ranked_arms": [a for a, _ in ranked],
        "metrics_used": metrics_used,
        "reason": reason,
    })
