"""v26 overnight chain — run the first build wave sequentially.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` §8
(26 Aug 2026). One process, one milestone at a time, so the sealed CPU
closed-form path never competes with itself for memory or cores.

Registered execution rules (written before any dispatch):
- Order: M296 (repaired solver) -> M297 (LOOCV lambda) -> M298 (LDA +
  balanced ridge) -> M299 (hybrid per-block L2 norm).
- Each milestone writes its own evidence exactly as its standalone
  runner does; the chain appends one line per milestone to the chain
  log with the outcome (gates_ok / void / exception).
- The chain STOPS after a milestone only when a later milestone
  REGISTERED a hard dependency on it: M298 is skipped with a VOID
  note unless M297 sealed with gates_ok (M298's g3 already enforces
  this; the chain never runs M298 against an unsealed M297).
  M296->M297 and M297->M299 carry no file dependency (each runner's
  own instrument gates reproduce the anchors), so they run even when
  their predecessor VOIDs — the discipline for reading their evidence
  is unchanged (a VOID cell carries no readings).
- The chain log is a status log, not evidence; the sealed files are
  the per-milestone evidence.json artifacts.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAIN_LOG = (REPO_ROOT / "logs" / "results" / "v26"
             / "overnight_chain_log.jsonl")
M297_EVIDENCE = (REPO_ROOT / "logs" / "results" / "v26"
                 / "m297_loocv_lambda" / "evidence.json")


def _log(entry: dict[str, Any]) -> None:
    CHAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAIN_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()


def _run_milestone(name: str, module: str) -> dict[str, Any]:
    import importlib
    import time

    started = time.time()
    _log({"event": "start", "milestone": name,
          "unix_started": int(started)})
    try:
        runner = importlib.import_module(module)
        # call run_<name> directly; every v26 runner exposes it
        fn = getattr(runner, f"run_{name.lower().replace('-', '_')}")
        out = fn(runner.DEFAULT_CONFIG, runner.DEFAULT_OUTPUT)
        _log({"event": "done", "milestone": name,
              "gates_ok": bool(out.get("gates_ok")),
              "void": bool(out.get("void")),
              "elapsed_seconds": round(time.time() - started, 1)})
        return out
    except Exception:
        _log({"event": "exception", "milestone": name,
              "traceback": traceback.format_exc()})
        # an exception in one milestone does not stop the independent
        # ones; each runner's own gates decide its evidence
        return None


def main() -> None:
    _log({"event": "chain_start", "note": "v26 first build wave"})

    m296 = _run_milestone("m296", "experiments.tier4.eval_v26_m296_head_repair")
    m297 = _run_milestone("m297", "experiments.tier4.eval_v26_m297_loocv_lambda")

    if M297_EVIDENCE.exists():
        m297_sealed = json.loads(M297_EVIDENCE.read_text(
            encoding="utf-8")).get("gates_ok") is True
    else:
        m297_sealed = False
    if m297_sealed:
        _run_milestone("m298",
                       "experiments.tier4.eval_v26_m298_lda_balanced")
    else:
        _log({"event": "skip", "milestone": "m298",
              "reason": "M297 not sealed with gates_ok; M298's registered "
                        "g3 dependency cannot pass"})

    _run_milestone("m299",
                   "experiments.tier4.eval_v26_m299_hybrid_blocks")

    _log({"event": "chain_done",
          "m296_gates_ok": bool((m296 or {}).get("gates_ok")),
          "m297_gates_ok": bool((m297 or {}).get("gates_ok"))})


if __name__ == "__main__":
    main()
