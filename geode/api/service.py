"""M217 — the API/RPC service: the orchestrator end-to-end over HTTP.

Local-only by user decision (20 Aug 2026): an API to test the full
product loop — register arms, route queries, verify the chain, build
settlement batches — plus a minimal no-build-step frontend served
from ``/``.

Layering: the application layer on top of the product package. It may
import any ``geode`` subpackage; it imports nothing from
``experiments.*``.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from geode import __version__ as GEODE_VERSION
from geode.api.metrics import MetricsCollector
from geode.api.persistence import load_snapshot, save_snapshot
from geode.core.arm import arm_from_sealed_head, validate_arm_spec
from geode.core.orchestrator import Orchestrator
from geode.settlement.settlement import (
    address_of,
    build_credit_batches,
    verify_batch_rules,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
SNAPSHOT_ENV = "GEODE_SNAPSHOT_PATH"
DEFAULT_SNAPSHOT = Path("geode_snapshot.json")

# the sealed M210b ms per-domain table (registered in the plan) — the
# demo seed uses REAL sealed accuracies for the ms arm; competitors
# are synthetic and stated as such.
DEMO_MS_TABLE = {"d0": 0.25429, "d1": 0.07742, "d2": 0.14362,
                 "d3": 0.33638, "d4": 0.26780, "d5": 0.14136}


class ArmSpec(BaseModel):
    arm_id: str
    family: str = "generic"
    width: int = 0
    accuracy: float
    sealed_source: str = ""
    per_task: dict[str, float] | None = None
    price: float = 0.0
    # the unified registration form (24 Aug): the operator key may
    # differ from the payout address (cold-key hygiene)
    operator_key: str | None = None
    payout_address: str | None = None


class RouteRequest(BaseModel):
    query_id: str
    fingerprint: list[float] = []
    task_id: str | None = None
    k: int = 1
    contract_kind: str | None = None
    max_unit_price: float | None = None
    best_quality: bool = False


class SettlementRequest(BaseModel):
    price_per_query: int = 100
    registration_fee: int = 0
    payers: dict[str, str] = {}
    payout_overrides: dict[str, str] = {}


def create_app(snapshot_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="GEODE API", version=GEODE_VERSION)
    orch = Orchestrator()
    metrics = MetricsCollector()
    route_requests: list[dict[str, Any]] = []
    snapshot_path = Path(snapshot_path or os.environ.get(SNAPSHOT_ENV)
                         or DEFAULT_SNAPSHOT)
    if Path(snapshot_path).exists():
        load_snapshot(orch, snapshot_path)

    @app.get("/health")
    def health() -> dict[str, Any]:
        started = time.perf_counter()
        out = {"status": "ok",
               "arms": len(orch.router.list_arms()),
               "chain_ok": orch.chain_verify()["ok"]}
        metrics.record((time.perf_counter() - started) * 1000.0)
        return out

    @app.get("/metrics")
    def metrics_view() -> dict[str, Any]:
        return metrics.summary()

    @app.post("/arms")
    def register(spec: ArmSpec) -> dict[str, Any]:
        if not 0.0 <= spec.accuracy <= 1.0:
            raise HTTPException(status_code=422,
                                detail="accuracy must be in [0, 1]")
        arm = arm_from_sealed_head(
            spec.arm_id, spec.family, spec.width, spec.accuracy,
            spec.sealed_source or "api-registered (not sealed)",
            per_task=spec.per_task, price=spec.price,
            operator_key=spec.operator_key,
            payout_address=spec.payout_address)
        reasons = validate_arm_spec(arm)
        if reasons:
            raise HTTPException(status_code=422, detail=reasons)
        try:
            index = orch.register(arm)
        except ValueError as exc:
            # the ledger is append-only: a duplicate arm_id is a
            # conflict, not a server error
            raise HTTPException(status_code=409, detail=str(exc))
        return {"index": index, "arm_id": spec.arm_id}

    @app.post("/route")
    def route(req: RouteRequest) -> dict[str, Any]:
        started = time.perf_counter()
        route_requests.append(dict(req.model_dump()))
        routed = orch.serve(req.query_id, req.fingerprint, k=req.k,
                            task_id=req.task_id,
                            contract_kind=req.contract_kind,
                            max_unit_price=req.max_unit_price,
                            best_quality=req.best_quality)
        view = [{"arm_id": rec["arm_id"],
                 "route_cos": rec.get("route_cos"),
                 "family": rec.get("family"),
                 "selection_accuracy": rec.get("selection_accuracy"),
                 "price": rec.get("price")} for rec in routed]
        metrics.record((time.perf_counter() - started) * 1000.0)
        return {"query_id": req.query_id, "routed": view}

    @app.get("/ledger")
    def ledger() -> dict[str, Any]:
        return {**orch.ledger.to_dict(), "verify": orch.chain_verify()}

    @app.post("/settlement/batches")
    def settlement(req: SettlementRequest) -> dict[str, Any]:
        def payer_of(query_id: str) -> str:
            return req.payers.get(query_id, address_of(f"payer:{query_id}"))
        report = build_credit_batches(
            orch, req.price_per_query, payer_of,
            payout_of=lambda arm_id: req.payout_overrides.get(arm_id),
            registration_fee=req.registration_fee)
        violations = verify_batch_rules(report,
                                        pool=report["pool_expected"])
        return {**report, "violations": violations,
                "conforms": not violations}

    @app.post("/snapshot")
    def snapshot() -> dict[str, Any]:
        digest = save_snapshot(orch, route_requests, snapshot_path)
        return {"path": str(snapshot_path), "hash": digest}

    @app.post("/demo/seed")
    def demo_seed() -> dict[str, Any]:
        existing = set(orch.router.list_arms())
        registered: list[str] = []
        specs = [
            {"arm_id": "ms_ridge", "family": "ms", "width": 13244,
             "accuracy": 0.24214492753623187,
             "sealed_source":
                 "logs/results/v25/m210b_ms_per_domain/evidence.json",
             "per_task": dict(DEMO_MS_TABLE),
             "note": "sealed M210b table"},
            {"arm_id": "syn_fast", "family": "synthetic", "width": 32,
             "accuracy": 0.30, "sealed_source": "synthetic (stated)",
             "per_task": {f"d{i}": 0.30 for i in range(6)},
             "note": "synthetic"},
            {"arm_id": "syn_small", "family": "synthetic", "width": 8,
             "accuracy": 0.20, "sealed_source": "synthetic (stated)",
             "per_task": {f"d{i}": 0.20 for i in range(6)},
             "note": "synthetic"},
        ]
        for spec in specs:
            if spec["arm_id"] in existing:
                continue
            arm = arm_from_sealed_head(
                spec["arm_id"], spec["family"], spec["width"],
                spec["accuracy"], spec["sealed_source"],
                per_task=spec["per_task"])
            orch.register(arm)
            registered.append(spec["arm_id"])
        return {"registered": registered}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
