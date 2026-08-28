"""Capability-aware orchestration layer for the GEODE model network.

The Orchestrator wraps a ModelNetwork and SemanticRouter to provide:

  1. **Capability registry** — enumerates what the network CAN and CANNOT do,
     distinguishing direct, semantic (IS-A), and composed capabilities.

  2. **Plan generation** — given a natural-language goal, finds the cheapest
     execution path without requiring a dedicated trained model for every
     combination.  Four resolution strategies are tried in order:

       DIRECT   — a registered model was trained on this exact class.
       SEMANTIC — a synonym/parent link covers the goal (e.g. tabby → cat).
       COMPOSED — multiple models ANDed/ORed at inference time with no new
                  training (e.g. "bird on car" from bird + car detectors).
       CASCADED — a downstream SDF-score model already covers the goal.

  3. **Gap detection** — surfaces capabilities that are missing, with typed
     suggestions for how to fill each gap.

  4. **Non-destructive extension** — ``extend()`` adds new capabilities
     without touching any existing node or model.

Quick start::

    orc = Orchestrator()
    orc.extend("animals", animal_model)
    orc.extend("vehicles", vehicle_model)

    print(orc.capabilities_report())
    # DIRECT   animals   → cat, dog, bird
    # DIRECT   vehicles  → car, truck, bus
    # SEMANTIC animals   → tabby (→ cat), kitty (→ cat), ...

    plan = orc.plan("tabby")           # SEMANTIC plan via cat
    result = orc.run(X_test, plan)     # {"label": array, "scores": array}

    plan2 = orc.plan("bird on car")    # COMPOSED AND plan
    result2 = orc.run(X_test, plan2)   # {"label": array, "scores": array}

    gaps = orc.gaps(["aircraft", "tabby", "dog"])
    # [CapabilityGap(goal="aircraft", type=UNKNOWN, ...)]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.model_network import FittedModel, ModelNetwork
from src.open_set import OpenSetBatchResult, RoutingStageCounters, SupportProfile
from src.semantic_router import SemanticRouter


# ---------------------------------------------------------------------------
# Capability types and gap types (string constants for readability)
# ---------------------------------------------------------------------------

DIRECT = "DIRECT"         # model trained on this exact class
SEMANTIC = "SEMANTIC"     # covered via IS-A / synonym
COMPOSED = "COMPOSED"     # multiple models combined at inference time
CASCADED = "CASCADED"     # downstream SDF-score model covers the goal
PRIMITIVE = "PRIMITIVE"   # deterministic code-defined transformation

GAP_UNKNOWN = "UNKNOWN"           # no model or semantic link at all
GAP_PARTIAL = "PARTIAL"           # weak semantic match only (score < 0.4)
GAP_NEEDS_COMBINATION = "NEEDS_COMBINATION"  # components exist; no plan yet


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Capability:
    """A single thing the network can currently do.

    Attributes
    ----------
    goal:
        The class or concept this capability addresses.
    capability_type:
        One of DIRECT, SEMANTIC, COMPOSED, CASCADED.
    node_names:
        Which network nodes are involved in achieving this.
    semantic_score:
        Confidence: 1.0 for direct/cascaded, 0.x for semantic/composed.
    via:
        Human-readable explanation, e.g. ``"tabby → cat (synonym)"`` or
        ``"bird AND car (composed)"``
    """

    goal: str
    capability_type: str
    node_names: list[str]
    semantic_score: float
    via: str


@dataclass
class CapabilityGap:
    """A goal the network currently cannot achieve.

    Attributes
    ----------
    goal:
        The requested but unachievable goal.
    gap_type:
        GAP_UNKNOWN, GAP_PARTIAL, or GAP_NEEDS_COMBINATION.
    best_match:
        The closest capability found (or empty string).
    best_score:
        Semantic score of the best match (0 if none).
    suggestion:
        Actionable advice for filling the gap.
    """

    goal: str
    gap_type: str
    best_match: str = ""
    best_score: float = 0.0
    suggestion: str = ""


@dataclass
class PlanStep:
    """One step in an execution plan.

    Attributes
    ----------
    node_name:
        Which registered network node to invoke.
    use_raw:
        If True, feed the original raw feature matrix X_raw.
        If False, feed the concatenated SDF scores from *upstream_nodes*.
    upstream_nodes:
        Nodes whose SDF scores are concatenated as input (only when
        ``use_raw=False``).
    target_classes:
        Subset of the node's output classes that are relevant to the goal.
        Empty means all classes matter.
    """

    node_name: str
    use_raw: bool = True
    upstream_nodes: list[str] = field(default_factory=list)
    target_classes: list[Any] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """A complete, executable specification for achieving a goal.

    Attributes
    ----------
    goal:
        The original goal string.
    capability_type:
        DIRECT | SEMANTIC | COMPOSED | CASCADED
    steps:
        Ordered list of :class:`PlanStep` objects.  Steps may run in parallel
        when they share the same ``use_raw=True`` — the executor handles this.
    combiner:
        How to merge multiple step outputs into a final answer:
        ``"classify"``       — argmin/calibrated predict on the single result node
        ``"and_threshold"``  — all target-class SDF scores must be < threshold
        ``"or_threshold"``   — any target-class SDF score must be < threshold
    combiner_threshold:
        SDF threshold for and/or combiners.  Points with SDF < threshold are
        considered *inside* the class volume (positive match).
    result_node:
        Name of the step whose output is the primary result (for DIRECT /
        SEMANTIC / CASCADED plans).
    reasoning:
        Human-readable explanation of how the plan was constructed.
    confidence:
        Overall plan confidence (1.0 = direct, lower for semantic/composed).
    """

    goal: str
    capability_type: str
    steps: list[PlanStep]
    combiner: str = "classify"
    combiner_threshold: float = 0.0
    result_node: str = ""
    reasoning: str = ""
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Goal parser  (rule-based; handles common compound patterns)
# ---------------------------------------------------------------------------

_AND_PATTERNS = [
    r"(.+?)\s+on\s+(.+)",        # "bird on car"
    r"(.+?)\s+with\s+(.+)",      # "cat with hat"
    r"(.+?)\s+and\s+(.+)",       # "cat and dog"
    r"(.+?)\s+\+\s+(.+)",        # "cat + dog"
]
_OR_PATTERNS = [
    r"(.+?)\s+or\s+(.+)",        # "cat or dog"
    r"(.+?)\s*/\s*(.+)",         # "cat/dog"
]


def _parse_compound_goal(goal: str) -> tuple[str, list[str]] | None:
    """Try to split a compound goal into (logic_op, [components]).

    Returns ``("AND", ["bird", "car"])`` for ``"bird on car"``, or
    ``None`` if the goal does not match a known compound pattern.
    """
    g = goal.strip().lower()
    for pat in _AND_PATTERNS:
        m = re.fullmatch(pat, g)
        if m:
            return "AND", [m.group(1).strip(), m.group(2).strip()]
    for pat in _OR_PATTERNS:
        m = re.fullmatch(pat, g)
        if m:
            return "OR", [m.group(1).strip(), m.group(2).strip()]
    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Capability-aware execution orchestrator for the GEODE model network.

    Parameters
    ----------
    router:
        :class:`~src.semantic_router.SemanticRouter` instance.  If omitted a
        ``"mock"`` router is used (no LLM, no semantic capabilities).
    sdf_threshold:
        Default SDF threshold for composed AND/OR plans.  A sample is
        considered inside a class volume when its SDF score < this value.
        The default ``0.0`` uses the natural SDF sign boundary.
    """

    def __init__(
        self,
        router: SemanticRouter | None = None,
        sdf_threshold: float = 0.0,
    ) -> None:
        self._network = ModelNetwork()
        self._router = router or SemanticRouter(backend="mock")
        self._sdf_threshold = sdf_threshold
        # node_name → FittedModel (mirror of _network._nodes for quick access)
        self._models: dict[str, FittedModel] = {}

    # ------------------------------------------------------------------
    # Building the network
    # ------------------------------------------------------------------

    def extend(
        self,
        name: str,
        model: FittedModel,
        upstream: list[str] | None = None,
    ) -> None:
        """Add a new ML model node non-destructively.

        Existing nodes are never modified.  The new node is registered with
        the semantic router so its classes become immediately routable.
        """
        self._network.add_node(name, model, upstream or [])
        self._models[name] = model
        self._router.register(model)

    def extend_primitive(
        self,
        name: str,
        primitive: "Primitive",
        upstream: list[str] | None = None,
    ) -> None:
        """Add a deterministic primitive node non-destructively.

        Primitives are registered in the semantic router under their
        ``category.name`` task name so they appear in capability reports
        and can be discovered by the orchestrator.  Existing nodes are
        never modified.

        Parameters
        ----------
        name:
            Unique node identifier.
        primitive:
            A :class:`~src.primitive.Primitive` instance.
        upstream:
            Upstream node names whose outputs are concatenated as input.
            ``None`` means the primitive receives raw features directly.
        """
        from src.primitive import Primitive  # local import avoids circular ref
        if not isinstance(primitive, Primitive):
            raise TypeError(f"Expected a Primitive, got {type(primitive).__name__}")
        self._network.add_node(name, primitive, upstream or [])
        self._models[name] = primitive  # type: ignore[assignment]
        # Register with the semantic router using the spec description as
        # the class label so it shows up in capability and routing queries.
        self._router.register(_PrimitiveModelAdapter(primitive))

    def swap(self, name: str, replacement: FittedModel) -> None:
        """Replace a node's model with a swappable equivalent.

        Uses :meth:`~src.model_network.ModelNetwork.swap_node` which
        verifies fingerprint compatibility before making any change.
        """
        self._network.swap_node(name, replacement)
        self._models[name] = replacement
        self._router.register(replacement)

    # ------------------------------------------------------------------
    # Capability inspection
    # ------------------------------------------------------------------

    def capabilities(self) -> list[Capability]:
        """Return all currently achievable capabilities.

        Includes DIRECT (exact class matches), SEMANTIC (IS-A links from the
        router cache), CASCADED (downstream SDF-score nodes), and PRIMITIVE
        (deterministic transformations).
        COMPOSED capabilities are not enumerated here — they are discovered
        lazily by :meth:`plan`.
        """
        from src.primitive import Primitive
        caps: list[Capability] = []
        seen_goals: set[str] = set()

        for node_name, node in self._network._nodes.items():
            fp = node.model.fingerprint

            # Primitives are listed separately
            if isinstance(node.model, Primitive):
                spec = node.model.spec
                goal_str = str(spec).lower()
                caps.append(Capability(
                    goal=goal_str,
                    capability_type=PRIMITIVE,
                    node_names=[node_name],
                    semantic_score=1.0,
                    via=f"{spec.category}.{spec.name} — node '{node_name}'",
                ))
                seen_goals.add(goal_str)
                continue

            # Cascade vs source ML model
            cap_type = CASCADED if node.upstream else DIRECT
            for cls in fp.output_spec.classes:
                goal_str = str(cls).lower()
                caps.append(Capability(
                    goal=goal_str,
                    capability_type=cap_type,
                    node_names=[node_name],
                    semantic_score=1.0,
                    via=f"direct — node '{node_name}'",
                ))
                seen_goals.add(goal_str)

        # Semantic capabilities: synonyms and parents from the router cache
        for task, class_map in self._router._cache.items():
            node_name = self._node_for_task(task)
            if node_name is None:
                continue
            for cls_str, desc in class_map.items():
                for synonym in desc.synonyms:
                    if synonym not in seen_goals:
                        caps.append(Capability(
                            goal=synonym,
                            capability_type=SEMANTIC,
                            node_names=[node_name],
                            semantic_score=0.9,
                            via=f"{synonym} → {cls_str} (synonym) via '{node_name}'",
                        ))
                        seen_goals.add(synonym)
                for parent in desc.parents:
                    if parent not in seen_goals:
                        caps.append(Capability(
                            goal=parent,
                            capability_type=SEMANTIC,
                            node_names=[node_name],
                            semantic_score=0.5,
                            via=f"{parent} ← {cls_str} (child class) via '{node_name}'",
                        ))
                        seen_goals.add(parent)

        return caps

    def gaps(self, goals: list[str]) -> list[CapabilityGap]:
        """Return a :class:`CapabilityGap` for each goal that cannot be achieved.

        Goals that are fully achievable (DIRECT, SEMANTIC ≥ 0.4, or COMPOSED)
        are omitted from the result.
        """
        result: list[CapabilityGap] = []
        for goal in goals:
            plan = self.plan(goal)
            if plan is not None and plan.confidence >= 0.4:
                continue   # achievable

            # Find best partial match
            hits = self._router.route(goal, self._network, top_k=1, threshold=0.0)
            best_node, best_score = hits[0] if hits else ("", 0.0)

            if best_score == 0.0:
                # Try compound parse — components may exist even if combined doesn't
                parsed = _parse_compound_goal(goal)
                if parsed:
                    logic, parts = parsed
                    missing = [p for p in parts if self.plan(p) is None]
                    if missing:
                        result.append(CapabilityGap(
                            goal=goal,
                            gap_type=GAP_NEEDS_COMBINATION,
                            suggestion=(
                                f"Components needed: {missing}.  "
                                f"Train models for these classes and call extend()."
                            ),
                        ))
                    else:
                        # All components exist but compound plan failed (shouldn't happen)
                        result.append(CapabilityGap(
                            goal=goal,
                            gap_type=GAP_NEEDS_COMBINATION,
                            suggestion=f"All components exist; try plan('{goal}') explicitly.",
                        ))
                else:
                    result.append(CapabilityGap(
                        goal=goal,
                        gap_type=GAP_UNKNOWN,
                        suggestion=(
                            f"No model or semantic link covers '{goal}'.  "
                            f"Train a model that includes this class and call extend()."
                        ),
                    ))
            else:
                result.append(CapabilityGap(
                    goal=goal,
                    gap_type=GAP_PARTIAL,
                    best_match=best_node,
                    best_score=best_score,
                    suggestion=(
                        f"Weak semantic match (score={best_score:.2f}) via node "
                        f"'{best_node}'.  Consider adding this class explicitly to "
                        f"a model, or register synonyms via the semantic router."
                    ),
                ))
        return result

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(self, goal: str) -> ExecutionPlan | None:
        """Find the cheapest execution plan to achieve *goal*.

        Resolution order: DIRECT → SEMANTIC → CASCADED → COMPOSED.
        Returns ``None`` if the goal is not achievable with current models.

        Parameters
        ----------
        goal:
            Class name, synonym, parent, or compound expression
            (e.g. ``"cat"``, ``"tabby"``, ``"mammal"``, ``"bird on car"``).
        """
        goal_lower = goal.strip().lower()

        # 1. Graph search: direct class match OR cascaded through any path of
        #    ML models and primitives.  Finds the shortest valid path.
        plan = self._plan_by_graph_search(goal_lower)
        if plan is not None:
            return plan

        # 2. SEMANTIC — IS-A / synonym match via the router cache
        plan = self._plan_semantic(goal_lower)
        if plan is not None:
            return plan

        # 3. COMPOSED — parse compound goal and combine existing plans
        plan = self._plan_composed(goal_lower)
        if plan is not None:
            return plan

        return None

    def _plan_by_graph_search(self, goal: str) -> ExecutionPlan | None:
        """Search every node for a direct, cascaded, or primitive match.

        For each candidate node whose output covers *goal*, the full execution
        path is reconstructed by backward-chaining through the DAG via
        :meth:`_find_path_to_node`.  This means primitives anywhere along the
        path (e.g. raw → [l2_normalize] → [cat_detector]) are automatically
        included in the returned steps.

        When multiple paths exist the shortest one (fewest steps) is preferred;
        DIRECT is preferred over CASCADED for equal length.
        """
        from src.primitive import Primitive

        candidates: list[tuple[ExecutionPlan, int]] = []

        for node_name, node in self._network._nodes.items():
            fp = node.model.fingerprint
            matched = False
            cap_type = DIRECT
            target_classes: list = []

            if isinstance(node.model, Primitive):
                spec = node.model.spec
                if goal in {spec.name.lower(), f"{spec.category}.{spec.name}".lower()}:
                    matched = True
                    cap_type = PRIMITIVE
            else:
                classes_lower = [str(c).lower() for c in fp.output_spec.classes]
                if goal in classes_lower:
                    matched = True
                    idx = classes_lower.index(goal)
                    target_classes = [fp.output_spec.classes[idx]]
                    cap_type = CASCADED if node.upstream else DIRECT

            if not matched:
                continue

            steps = self._find_path_to_node(node_name)
            if steps is None:
                continue  # cycle or unresolvable dependency

            if target_classes:
                steps[-1].target_classes = target_classes

            path_str = " → ".join(f"[{s.node_name}]" for s in steps)
            reasoning = (
                f"{cap_type}: path {path_str} covers '{goal}'."
                if len(steps) > 1
                else f"{cap_type}: node '{node_name}' covers '{goal}'."
            )
            candidates.append((
                ExecutionPlan(
                    goal=goal,
                    capability_type=cap_type,
                    steps=steps,
                    combiner="classify",
                    result_node=node_name,
                    reasoning=reasoning,
                    confidence=1.0,
                ),
                len(steps),
            ))

        if not candidates:
            return None

        # Prefer shortest path; break ties in favour of DIRECT over CASCADED.
        candidates.sort(key=lambda x: (x[1], 0 if x[0].capability_type == DIRECT else 1))
        return candidates[0][0]

    def _plan_semantic(self, goal: str) -> ExecutionPlan | None:
        hits = self._router.route(goal, self._network, top_k=1, threshold=0.3)
        if not hits:
            return None
        node_name, score = hits[0]
        node = self._network._nodes[node_name]
        task = node.model.fingerprint.task_name
        # Find which class in this node's cache matched
        matched_class = goal
        via_str = f"{goal} → (semantic) via node '{node_name}'"
        task_cache = self._router._cache.get(task, {})
        for cls_str, desc in task_cache.items():
            if desc.match_score(goal) >= score - 0.01:
                matched_class = cls_str
                via_str = desc.qualified_name(task)
                break

        return ExecutionPlan(
            goal=goal,
            capability_type=SEMANTIC,
            steps=[PlanStep(node_name=node_name, use_raw=True,
                            target_classes=[matched_class])],
            combiner="classify",
            result_node=node_name,
            reasoning=f"Semantic match (score={score:.2f}): '{goal}' resolved to '{via_str}'.",
            confidence=score,
        )

    def _plan_composed(self, goal: str) -> ExecutionPlan | None:
        parsed = _parse_compound_goal(goal)
        if parsed is None:
            return None

        logic, parts = parsed
        combiner = "and_threshold" if logic == "AND" else "or_threshold"

        component_plans = []
        for part in parts:
            sub = self.plan(part)
            if sub is None:
                return None   # can't compose if a component is missing
            component_plans.append(sub)

        # Collect unique steps across all component plans (deduplicate same node)
        seen_nodes: set[str] = set()
        steps: list[PlanStep] = []
        for cp in component_plans:
            for step in cp.steps:
                if step.node_name not in seen_nodes:
                    steps.append(step)
                    seen_nodes.add(step.node_name)

        composed_nodes = [cp.result_node for cp in component_plans]
        min_confidence = min(cp.confidence for cp in component_plans)
        part_descriptions = [f"'{p}' via '{cp.result_node}'" for p, cp in zip(parts, component_plans)]

        return ExecutionPlan(
            goal=goal,
            capability_type=COMPOSED,
            steps=steps,
            combiner=combiner,
            combiner_threshold=self._sdf_threshold,
            result_node="__composed__",
            reasoning=(
                f"Composed {logic}: {' ; '.join(part_descriptions)}. "
                f"No combined model needed — SDF scores thresholded at {self._sdf_threshold}."
            ),
            confidence=min_confidence * 0.9,  # slight penalty for composition
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, X_raw: np.ndarray, plan: ExecutionPlan) -> dict:
        """Execute a plan and return a result dict.

        Returns
        -------
        dict with keys:
            ``"label"``   — (N,) integer or string label array.
            ``"scores"``  — (N, n_classes) SDF score matrix (DIRECT/SEMANTIC/CASCADED)
                            or (N,) boolean match array (COMPOSED).
            ``"goal"``    — the original goal string.
            ``"confidence"`` — plan confidence score.
        """
        X_raw = np.asarray(X_raw, dtype=np.float64)
        sdf_cache: dict[str, np.ndarray] = {}

        # Execute all steps in plan order
        for step in plan.steps:
            node = self._network._nodes[step.node_name]
            X_in = X_raw if step.use_raw else np.concatenate(
                [sdf_cache[up] for up in step.upstream_nodes], axis=1
            )
            sdf_cache[step.node_name] = node.model.sdf_scores(X_in)

        # Apply combiner
        if plan.combiner == "classify":
            result_node = self._network._nodes[plan.result_node]
            scores = sdf_cache[plan.result_node]
            labels = result_node.model._predict_from_scores(scores)
            # If semantic plan: remap labels to the resolved class name
            if plan.capability_type == SEMANTIC and plan.steps[0].target_classes:
                target = str(plan.steps[0].target_classes[0]).lower()
                all_classes = [str(c).lower() for c in result_node.model.fingerprint.output_spec.classes]
                if target in all_classes:
                    col = all_classes.index(target)
                    scores = scores[:, col:col+1]
            return {
                "label": labels,
                "scores": scores,
                "goal": plan.goal,
                "confidence": plan.confidence,
            }

        elif plan.combiner in ("and_threshold", "or_threshold"):
            # Collect per-component binary masks
            component_masks = []
            for step in plan.steps:
                if step.node_name not in sdf_cache:
                    continue
                scores = sdf_cache[step.node_name]
                if step.target_classes:
                    node = self._network._nodes[step.node_name]
                    all_cls = [str(c).lower() for c in node.model.fingerprint.output_spec.classes]
                    cols = [all_cls.index(str(tc).lower()) for tc in step.target_classes
                            if str(tc).lower() in all_cls]
                    if cols:
                        scores = scores[:, cols]
                # inside = any target class SDF below threshold
                mask = np.any(scores < plan.combiner_threshold, axis=1)
                component_masks.append(mask)

            if not component_masks:
                match = np.zeros(len(X_raw), dtype=bool)
            elif plan.combiner == "and_threshold":
                match = np.ones(len(X_raw), dtype=bool)
                for m in component_masks:
                    match &= m
            else:
                match = np.zeros(len(X_raw), dtype=bool)
                for m in component_masks:
                    match |= m

            return {
                "label": match.astype(np.int32),
                "scores": match.astype(np.float64),
                "goal": plan.goal,
                "confidence": plan.confidence,
            }

        raise ValueError(f"Unknown combiner: {plan.combiner!r}")

    def run_open_set(
        self,
        X_raw: np.ndarray,
        plan: ExecutionPlan,
        support_profile: SupportProfile,
        *,
        feature_transform_fingerprint: str,
        calibrated_novelty_scores: np.ndarray | None = None,
    ) -> dict:
        """Execute a classify plan with explicit opt-in unknown rejection."""
        if plan.combiner != "classify":
            raise ValueError("Open-set execution currently requires a classify plan.")
        X_raw = np.asarray(X_raw, dtype=np.float64)
        sdf_cache: dict[str, np.ndarray] = {}
        compatible_candidate_pairs = 0
        exact_class_sdf_pairs = 0
        primitive_sdf_pairs = 0
        score_values_materialized = 0
        for step in plan.steps:
            node = self._network._nodes[step.node_name]
            X_in = X_raw if step.use_raw else np.concatenate(
                [sdf_cache[up] for up in step.upstream_nodes], axis=1
            )
            scores = node.model.sdf_scores(X_in)
            sdf_cache[step.node_name] = scores
            compatible_candidate_pairs += scores.size
            score_values_materialized += scores.size
            class_models = getattr(node.model, "class_models", {})
            exact_class_sdf_pairs += len(scores) * sum(
                bool(experts) for experts in class_models.values()
            )
            primitive_sdf_pairs += len(scores) * sum(
                len(expert.ellipsoids)
                for experts in class_models.values()
                for expert in experts
            )

        result_node = self._network._nodes[plan.result_node]
        model = result_node.model
        if not isinstance(model, FittedModel):
            raise TypeError("Open-set classify results require a FittedModel result node.")
        scores = sdf_cache[plan.result_node]
        open_set: OpenSetBatchResult = model._predict_open_set_from_scores(
            scores,
            support_profile,
            feature_transform_fingerprint=feature_transform_fingerprint,
            calibrated_novelty_scores=calibrated_novelty_scores,
            counters=RoutingStageCounters(
                sample_count=len(X_raw),
                nodes_executed=len(plan.steps),
                compatible_candidate_pairs=compatible_candidate_pairs,
                shortlisted_candidate_pairs=compatible_candidate_pairs,
                exact_class_sdf_pairs=exact_class_sdf_pairs,
                primitive_sdf_pairs=primitive_sdf_pairs,
                score_values_materialized=score_values_materialized,
            ),
        )
        return {
            "label": np.asarray(
                [prediction.label for prediction in open_set.predictions],
                dtype=object,
            ),
            "scores": scores,
            "open_set": open_set,
            "goal": plan.goal,
            "confidence": plan.confidence,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def capabilities_report(self) -> str:
        """Return a formatted summary of all current capabilities."""
        caps = self.capabilities()
        if not caps:
            return "Orchestrator: no capabilities registered."

        lines = [f"Orchestrator capabilities ({len(caps)} total):"]
        by_type: dict[str, list[Capability]] = {}
        for c in caps:
            by_type.setdefault(c.capability_type, []).append(c)

        order = [DIRECT, CASCADED, SEMANTIC, PRIMITIVE, COMPOSED]
        for ct in order:
            group = by_type.get(ct, [])
            if not group:
                continue
            lines.append(f"\n  [{ct}]")
            for cap in sorted(group, key=lambda c: c.goal):
                nodes = ", ".join(cap.node_names)
                score = f"  ({cap.semantic_score:.2f})" if cap.semantic_score < 1.0 else ""
                lines.append(f"    {cap.goal:<25s} via [{nodes}]{score}")
        return "\n".join(lines)

    def gaps_report(self, goals: list[str]) -> str:
        """Return a formatted summary of capability gaps for *goals*."""
        gaps = self.gaps(goals)
        if not gaps:
            return f"All {len(goals)} goals are achievable."
        lines = [f"Capability gaps ({len(gaps)}/{len(goals)} goals unachievable):"]
        for gap in gaps:
            score_str = f"  best_match={gap.best_match!r} ({gap.best_score:.2f})" if gap.best_score else ""
            lines.append(f"  [{gap.gap_type}] '{gap.goal}'{score_str}")
            lines.append(f"    → {gap.suggestion}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n = len(self._models)
        return (
            f"Orchestrator(nodes={n}, "
            f"router={self._router._backend!r}, "
            f"capabilities={len(self.capabilities())})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _node_for_task(self, task_name: str) -> str | None:
        """Return the first node whose model has the given task name."""
        for name, model in self._models.items():
            if model.fingerprint.task_name == task_name:
                return name
        return None

    def _find_path_to_node(
        self,
        node_name: str,
        _visited: frozenset = frozenset(),
    ) -> list[PlanStep] | None:
        """Backward-chain through the DAG to find all steps needed for *node_name*.

        Returns an ordered list of :class:`PlanStep` objects (topological order)
        that, when executed left-to-right, produce the output of *node_name*.
        Returns ``None`` if a dependency cycle is detected or a required node
        is missing from the network.

        The traversal is depth-first and deduplicates shared upstream nodes so
        the same node never appears twice in the returned list.
        """
        if node_name in _visited:
            return None  # cycle
        if node_name not in self._network._nodes:
            return None

        node = self._network._nodes[node_name]
        _visited = _visited | {node_name}

        if not node.upstream:
            return [PlanStep(node_name=node_name, use_raw=True)]

        steps: list[PlanStep] = []
        seen: set[str] = set()

        for up_name in node.upstream:
            sub = self._find_path_to_node(up_name, _visited)
            if sub is None:
                return None
            for s in sub:
                if s.node_name not in seen:
                    steps.append(s)
                    seen.add(s.node_name)

        steps.append(
            PlanStep(
                node_name=node_name,
                use_raw=False,
                upstream_nodes=list(node.upstream),
            )
        )
        return steps


# ---------------------------------------------------------------------------
# Internal adapter
# ---------------------------------------------------------------------------


class _PrimitiveModelAdapter:
    """Thin wrapper that gives a Primitive the FittedModel interface expected
    by SemanticRouter.register().

    SemanticRouter only needs ``fingerprint.task_name`` and
    ``fingerprint.output_spec.classes`` for registration queries.  The
    primitive spec's category+name becomes the task name and the name
    becomes the single output class so the LLM is queried for it like any
    other class label.
    """

    def __init__(self, primitive: "Primitive") -> None:  # type: ignore[name-defined]
        self.fingerprint = primitive.fingerprint
