"""The ``geode`` command line — the §4.8 operating manual as commands.

Stdlib-only (argparse); the API package (uvicorn) is an optional
extra. Deterministic where the underlying layers are; no RNG, no
wall clocks in anything that is recorded.

    geode version
    geode route --fp 0.9,0.3,0.2,0.1 [--snapshot PATH] [--tags refusal]
    geode verify --evidence PATH
    geode freeze --attest v1,v2 --ttl 1000 [--reason ...]
    geode override --actor op --action kill_switch \\
        --justification "..." --counterfactual '{"would":"route to a"}'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from geode import __version__


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"geode {__version__}")
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    from geode.api.persistence import load_snapshot
    from geode.core.orchestrator import Orchestrator
    from geode.core.router import Router
    fp = [float(x) for x in args.fp.split(",")]
    orch = Orchestrator()
    if args.snapshot:
        load_snapshot(orch, Path(args.snapshot))
    elif not orch.router.list_arms():
        # demo registry so the command is self-contained
        orch.router.add_arm({
            "arm_id": "demo_general", "fingerprint": fp,
            "output_contract": {"kind": "class"},
            "held_out_accuracy": 0.5,
            "availability": {"healthy": True},
            "price": 1.0, "general": True, "primitive": False,
        })
    kwargs: dict[str, Any] = {}
    if args.tags:
        kwargs["required_tags"] = args.tags.split(",")
    recs = orch.router.route(fp, k=args.k, **kwargs)
    if not recs:
        print("[]  (empty route: abstained or constrained — escalate)")
        return 1
    for rec in recs:
        print(f"{rec['arm_id']}  cos={rec.get('route_cos'):.4f}  "
              f"ranked_by={rec.get('ranked_by')}  "
              f"provisional={rec.get('provisional')}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from geode.audit import AuditAPI, evidence_content_hash
    import json as _json
    api = AuditAPI()
    artifact_dir = Path(args.evidence)
    report = api.provenance(artifact_dir)
    evidence = _json.loads(
        (artifact_dir / "evidence.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "chain": report.chain,
        "gaps": report.gaps,
        "evidence_content_hash": evidence_content_hash(evidence),
    }, indent=2))
    return 0 if not report.gaps else 1


def _cmd_freeze(args: argparse.Namespace) -> int:
    from geode.core.freeze import FreezeRegistry
    registry = FreezeRegistry(k_of_n=len(args.attest.split(",")),
                              default_ttl=args.ttl)
    event = registry.freeze("cli", frozenset(args.attest.split(",")),
                            start_index=0, reason=args.reason or "",
                            ttl=args.ttl)
    print(f"freeze {event.event_id}: effective until ledger index "
          f"{event.expires_index} (a freeze cannot be permanent)")
    return 0


def _cmd_override(args: argparse.Namespace) -> int:
    from geode.core.override import OverrideLedger
    ledger = OverrideLedger()
    idx = ledger.record(args.actor, args.action, args.justification,
                        json.loads(args.counterfactual))
    print(f"override recorded at index {idx}; chain tip "
          f"{ledger.tip()[:16]}...")
    return 0


def _cmd_artifacts_verify(args: argparse.Namespace) -> int:
    from geode.core.artifacts import verify_artifact
    ok = verify_artifact(args.path, args.digest)
    if ok:
        print(f"digest matches: {args.path}")
        return 0
    print(f"DIGEST MISMATCH: {args.path}", file=sys.stderr)
    return 1


def _cmd_serve(args: argparse.Namespace) -> int:
    command = ("uvicorn geode.api.service:app --host "
               f"{args.host} --port {args.port}")
    if args.dry_run:
        print(command)
        return 0
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print("the api extra is not installed: "
              "pip install 'geode-ml[api]'", file=sys.stderr)
        return 2
    import uvicorn as u
    u.run("geode.api.service:app", host=args.host,
          port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geode", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version").set_defaults(func=_cmd_version)

    p_route = sub.add_parser("route", help="route a task fingerprint")
    p_route.add_argument("--fp", required=True,
                         help="comma-separated fingerprint")
    p_route.add_argument("--k", type=int, default=1)
    p_route.add_argument("--snapshot", default=None,
                         help="API snapshot to restore first")
    p_route.add_argument("--tags", default=None,
                         help="comma-separated required safety tags")
    p_route.set_defaults(func=_cmd_route)

    p_verify = sub.add_parser("verify", help="replay a sealed evidence file")
    p_verify.add_argument("--evidence", required=True)
    p_verify.set_defaults(func=_cmd_verify)

    p_freeze = sub.add_parser("freeze", help="issue a time-bounded freeze")
    p_freeze.add_argument("--attest", required=True,
                          help="comma-separated attesters (k-of-n)")
    p_freeze.add_argument("--ttl", type=int, default=1000)
    p_freeze.add_argument("--reason", default=None)
    p_freeze.set_defaults(func=_cmd_freeze)

    p_override = sub.add_parser("override", help="record a human override")
    p_override.add_argument("--actor", required=True)
    p_override.add_argument("--action", required=True)
    p_override.add_argument("--justification", required=True)
    p_override.add_argument("--counterfactual", required=True,
                            help="JSON of what the system would have done")
    p_override.set_defaults(func=_cmd_override)

    p_art = sub.add_parser("artifacts", help="content-addressed artifacts")
    art_sub = p_art.add_subparsers(dest="art_command", required=True)
    p_art_verify = art_sub.add_parser(
        "verify", help="verify a file against its sha256 digest")
    p_art_verify.add_argument("--path", required=True)
    p_art_verify.add_argument("--digest", required=True)
    p_art_verify.set_defaults(func=_cmd_artifacts_verify)

    p_serve = sub.add_parser("serve", help="serve the local API "
                             "(needs the api extra)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--dry-run", action="store_true",
                         help="print the command without binding")
    p_serve.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"geode: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
