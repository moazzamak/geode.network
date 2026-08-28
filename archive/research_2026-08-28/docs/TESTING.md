# GEODE Testing

The test suite is layered on the test pyramid and runs with
[pytest](https://docs.pytest.org/). The existing unittest-style classes
run natively; no rewrite was required.

## Layers

| Layer       | Directory            | Marker        | Scope                                                                              |
| ----------- | -------------------- | ------------- | ---------------------------------------------------------------------------------- |
| Unit        | `tests/unit/`        | `unit`        | Single-concern, in-process: one module, no cross-module flows                      |
| Integration | `tests/integration/` | `integration` | Multi-module in-process flows (orchestrator + ledger + settlement; repair overlay) |
| System      | `tests/system/`      | `system`      | Cross-process: the EVM Hardhat harness suite                                       |

Markers are applied directory-wide by `tests/<layer>/conftest.py`, so
milestone-named test files keep their names as the audit trail and
gain a layer automatically.

## Running

```powershell
# everything
python -m pytest

# one layer
python -m pytest -m unit
python -m pytest -m integration
python -m pytest -m system

# one module
python -m pytest tests/unit/test_v25_m185_ledger.py
```

Configuration lives in `pytest.ini` (`testpaths = tests`,
`pythonpath = .`, registered markers).

## Environment requirements

- Tests that read the corpus cache (the M182 repair overlay) need
  `GEODE_CACHE_DIR`. `tests/integration/conftest.py` defaults it to
  this machine's cache; override the environment variable to point
  elsewhere.
- The system layer needs Node + the installed Hardhat harness
  (`infrastructure/evm/node_modules`). It is skipped when `npx` is
  unavailable.

## The acceptance bar

A change is accepted when:

1. `python -m pytest` is green (all three layers);
2. the EVM harness suite (`npx hardhat test` in
   `infrastructure/evm`) is green;
3. sealed anchors reproduce — a refactor must never change a sealed
   value. The M212/M213/M214 runners re-produce their gates and are
   the regression net for the architecture itself.

## CI and coverage

`.github/workflows/ci.yml` runs on every push: pytest with
`--cov=geode --cov-fail-under=95` (measured 96%+, the gate trails the
measured coverage per the measured-then-raised policy), the
architecture rules, the Hardhat suite, and the Solidity coverage gate
(`npx hardhat coverage`). The coverage floor is raised as the suite
grows, never lowered.
