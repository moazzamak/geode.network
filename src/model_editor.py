import hashlib
import json
import time
from collections.abc import Callable

import numpy as np

from src.nudge_engine import NudgeEngine
from src.sdf_engine import EllipsoidExpert, Expert


def _clone_ellipsoid(ellipsoid: EllipsoidExpert, polarity: int) -> EllipsoidExpert:
    return EllipsoidExpert(
        center=ellipsoid.center.copy(),
        radii=ellipsoid.radii.copy(),
        orientation=ellipsoid.orientation.copy(),
        polarity=polarity,
    )


class ModelEditor:
    """Apply localized, auditable edits to class-indexed GEODE models."""

    def __init__(
        self,
        models: dict[int, list[Expert]],
        *,
        alpha: float = 2.0,
        invalidate: Callable[[dict], None] | None = None,
    ):
        self.models = models
        self.alpha = alpha
        self.invalidate = invalidate or (lambda _models: None)
        self.audit_log: list[dict] = []

    def snapshot(self) -> bytes:
        payload = {
            "alpha": self.alpha,
            "classes": [
                {
                    "class_id": int(class_id),
                    "experts": [
                        {
                            "alpha": expert.alpha,
                            "ellipsoids": [
                                {
                                    "center": ellipsoid.center.tolist(),
                                    "radii": ellipsoid.radii.tolist(),
                                    "orientation": ellipsoid.orientation.tolist(),
                                    "polarity": ellipsoid.polarity,
                                }
                                for ellipsoid in expert.ellipsoids
                            ],
                        }
                        for expert in experts
                    ],
                }
                for class_id, experts in sorted(self.models.items())
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def snapshot_id(snapshot: bytes) -> str:
        return hashlib.sha256(snapshot).hexdigest()[:12]

    def _record(self, operation: str, class_id: int | None, **details) -> dict:
        experts = [expert for class_models in self.models.values() for expert in class_models]
        ellipsoids = [
            ellipsoid for expert in experts for ellipsoid in expert.ellipsoids
        ]
        record = {
            "operation": operation,
            "class_id": class_id,
            "structure": {
                "classes": len(self.models),
                "empty_classes": sum(not experts for experts in self.models.values()),
                "experts": len(experts),
                "additive_ellipsoids": sum(e.polarity > 0 for e in ellipsoids),
                "subtractive_ellipsoids": sum(e.polarity < 0 for e in ellipsoids),
            },
            **details,
        }
        self.audit_log.append(record)
        return record

    def _invalidate(self) -> None:
        self.invalidate(self.models)

    def apply_transaction(
        self,
        operation: Callable[[], None],
        validator: Callable[[dict[int, list[Expert]]], bool],
        *,
        operation_name: str,
        class_id: int | None = None,
    ) -> dict:
        """Apply an isolated edit and retain it only when validation passes."""
        started = time.perf_counter()
        before = self.snapshot()
        try:
            operation()
            self._invalidate()
            accepted = bool(validator(self.models))
        except Exception:
            self.rollback(before, record_audit=False)
            raise
        if not accepted:
            self.rollback(before, record_audit=False)
        return self._record(
            operation_name,
            class_id,
            accepted=accepted,
            before_snapshot_id=self.snapshot_id(before),
            after_snapshot_id=self.snapshot_id(self.snapshot()),
            edit_seconds=time.perf_counter() - started,
        )

    def insert_additive(
        self,
        class_id: int,
        ellipsoid: EllipsoidExpert,
        *,
        expert_index: int | None = None,
    ) -> dict:
        started = time.perf_counter()
        experts = self.models.setdefault(class_id, [])
        primitive = _clone_ellipsoid(ellipsoid, polarity=1)
        if expert_index is None:
            expert = Expert(alpha=self.alpha)
            expert.add_ellipsoid(primitive)
            experts.append(expert)
            expert_index = len(experts) - 1
        else:
            experts[expert_index].add_ellipsoid(primitive)
        self._invalidate()
        return self._record(
            "insert_additive",
            class_id,
            expert_index=expert_index,
            ellipsoid_index=len(experts[expert_index].ellipsoids) - 1,
            edit_seconds=time.perf_counter() - started,
        )

    def insert_additive_from_points(
        self,
        class_id: int,
        points: np.ndarray,
        fitter: Callable[[np.ndarray, int], EllipsoidExpert],
        *,
        seed: int = 42,
        expert_index: int | None = None,
    ) -> dict:
        started = time.perf_counter()
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or not len(points):
            raise ValueError("points must be a non-empty 2D array.")
        record = self.insert_additive(
            class_id, fitter(points, seed), expert_index=expert_index,
        )
        record["operation"] = "insert_additive_from_points"
        record["point_count"] = len(points)
        record["seed"] = seed
        record["edit_seconds"] = time.perf_counter() - started
        return record

    def insert_validated_subtractive(
        self,
        class_id: int,
        expert_index: int,
        ellipsoid: EllipsoidExpert,
        validator: Callable[[dict[int, list[Expert]]], bool],
    ) -> dict:
        started = time.perf_counter()
        before = self.snapshot()
        expert = self.models[class_id][expert_index]
        expert.add_ellipsoid(_clone_ellipsoid(ellipsoid, polarity=-1))
        self._invalidate()
        accepted = bool(validator(self.models))
        if not accepted:
            self.rollback(before, record_audit=False)
        return self._record(
            "insert_subtractive",
            class_id,
            expert_index=expert_index,
            accepted=accepted,
            edit_seconds=time.perf_counter() - started,
        )

    def delete_primitive(
        self,
        class_id: int,
        expert_index: int,
        ellipsoid_index: int,
    ) -> dict:
        started = time.perf_counter()
        experts = self.models[class_id]
        expert = experts[expert_index]
        del expert.ellipsoids[ellipsoid_index]
        expert._bs_cache = None
        if not expert.ellipsoids:
            del experts[expert_index]
        self._invalidate()
        return self._record(
            "delete_primitive",
            class_id,
            expert_index=expert_index,
            ellipsoid_index=ellipsoid_index,
            edit_seconds=time.perf_counter() - started,
        )

    def local_nudge(
        self,
        class_id: int,
        points: np.ndarray,
        *,
        learning_rate: float = 0.1,
        iterations: int = 1,
        validator: Callable[[dict[int, list[Expert]]], bool] | None = None,
    ) -> dict:
        started = time.perf_counter()
        before = self.snapshot()
        NudgeEngine(
            learning_rate=learning_rate, iterations=iterations,
        ).refine(self.models[class_id], np.asarray(points, dtype=np.float64))
        for expert in self.models[class_id]:
            expert._bs_cache = None
        self._invalidate()
        accepted = validator is None or bool(validator(self.models))
        if not accepted:
            self.rollback(before, record_audit=False)
        return self._record(
            "local_nudge",
            class_id,
            point_count=len(points),
            iterations=iterations,
            accepted=accepted,
            edit_seconds=time.perf_counter() - started,
        )

    def rollback(self, snapshot: bytes, *, record_audit: bool = True) -> dict | None:
        started = time.perf_counter()
        payload = json.loads(snapshot.decode("utf-8"))
        restored = {}
        for class_record in payload["classes"]:
            experts = []
            for expert_record in class_record["experts"]:
                expert = Expert(alpha=float(expert_record["alpha"]))
                for ellipsoid_record in expert_record["ellipsoids"]:
                    expert.add_ellipsoid(EllipsoidExpert(
                        center=np.asarray(ellipsoid_record["center"], dtype=np.float64),
                        radii=np.asarray(ellipsoid_record["radii"], dtype=np.float64),
                        orientation=np.asarray(
                            ellipsoid_record["orientation"], dtype=np.float64,
                        ),
                        polarity=int(ellipsoid_record["polarity"]),
                    ))
                experts.append(expert)
            restored[int(class_record["class_id"])] = experts
        self.models.clear()
        self.models.update(restored)
        self.alpha = float(payload["alpha"])
        self._invalidate()
        if record_audit:
            return self._record(
                "rollback",
                None,
                snapshot_id=self.snapshot_id(snapshot),
                edit_seconds=time.perf_counter() - started,
            )
        return None