# GEODE quickstart

Five minutes from zero to a routed, guarded, verifiable registry.

## 1. Install

```bash
pip install .            # the product package (numpy + torch)
pip install '.[api,dev]' # plus the HTTP API and the test tooling
```

The API is **local-only by design** (registered 20 Aug 2026): it
exists to exercise the full product loop, not to be exposed publicly.

## 2. The five-minute tour

```bash
python examples/hello_geode.py
```

walks register → route → guard → contain → override → verify and
prints every step. The same operations are available as commands:

```bash
geode version
geode route --fp 0.9,0.3,0.2,0.1
geode route --fp 0.9,0.3,0.2,0.1 --tags refusal     # safety-flagged task
geode verify --evidence logs/results/v25/<milestone>
geode freeze --attest v1,v2 --ttl 1000 --reason "drill"
geode override --actor op --action kill_switch \
    --justification "drill" --counterfactual '{"would_have": "route a"}'
```

A `--tags` route is a hard-constraint route: arms without MEASURED
coverage of every tag are excluded, and a below-floor result prints
an empty route — escalate, do not force a best guess.

## 3. Serve the API (optional)

```bash
pip install '.[api]'
uvicorn geode.api.service:app --host 127.0.0.1 --port 8000
```

Endpoints: register arms, route queries, verify the chain, build
settlement batches. Snapshots survive restarts via
`geode.api.persistence`.

## 4. Where the guarantees live

- **Safety semantics** (constraint tier, abstention, freeze, OOD
  guard, override ledger): the whitepaper §11, exercised by
  `tests/unit/test_v25_m248_m252_containment.py` and friends.
- **Economics** (staking, caps, incentives): whitepaper §12,
  `geode/attribution/{stake,payoff_cap,incentives}.py`.
- **The evidence discipline**: whitepaper §5 — every number replays
  through `geode.audit.AuditAPI`.
- **The full manual**: whitepaper §4.8.

## 5. Tests

```bash
pip install '.[dev]'
python -m pytest -q          # 433+ tests incl. architecture rules
```

## 6. Deploying

The full deployment walkthrough — conditions, step-by-step
procedure, failure table, and the safety rules — is in
`docs/DEPLOYMENT.md` (Simplified Technical English, no internal
references).
