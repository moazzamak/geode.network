"""M175 cell D — license-clean Wikipedia fit-and-report (the uniform-w2 arm
fitted ON the dump, with the licensing posture recorded).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` section 12
(18 Aug 2026) and ``analysis/v24_m175_transfer_cells.md``.

Slices (registered): fit = wikitext103_full train ids [100000:400000]
(300k); held-out = train ids [400000:500000] (100k) — disjoint from
every slice A/B/A0 measured. Model: uniform-w2 (selected and sealed in
cell A; no new selection).

Anchors (registered): g0 prefix pin (the 100k train is the full-stream
prefix); g1 machinery anchor — D's encoder on A's exact out-domain
slice reproduces A's sealed 11.0933 with delta 0.0.

License posture (recorded, not a legal audit): wikitext-103 derives
from Wikipedia articles (CC BY-SA 3.0 / GFDL); the cached token ids
carry a vocab fingerprint, so the lineage is traceable end-to-end.

Reading (registered): fit-and-report; beating A's transferred read
(11.0933) is a reported comparison point, not a gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    write_canonical_json,
)
from experiments.tier4.eval_v24_m175_cell_a0 import (
    _perplexity,
    _vocabulary,
)
from src.programmatic_memory import ProgrammaticMemory

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m175_cell_d.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v24" / "m175_cell_d"

G1_TOLERANCE = 0.0
WINDOW = 2
ALPHA = 1.0


def _load(config: dict[str, Any], smoke_tokens: int) -> dict[str, Any]:
    sub = np.load(REPO_ROOT / config["corpus"]["sub_cache_file"])
    full = np.load(REPO_ROOT / config["corpus"]["full_cache_file"],
                   mmap_mode="r")
    prefix_ok = bool(np.array_equal(sub["train_ids"],
                                    full["train_ids"][:len(sub["train_ids"])]))
    if not prefix_ok:
        raise SystemExit("M175 D VOID: g0 prefix pin failed")
    fit_n = int(config["slices"]["fit_rows"])
    test_n = int(config["slices"]["test_rows"])
    if smoke_tokens:
        fit_n = min(fit_n, smoke_tokens)
        test_n = min(test_n, max(1, smoke_tokens // 4))
    fit = [int(x) for x in full["train_ids"]
           [config["slices"]["fit_start"]:
            config["slices"]["fit_start"] + fit_n]]
    test = [int(x) for x in full["train_ids"]
            [config["slices"]["test_start"]:
             config["slices"]["test_start"] + test_n]]
    # A's exact out-domain slice for g1 (always full, never truncated)
    a_slice = [int(x) for x in full["train_ids"]
               [int(config["anchors"]["a_wiki_start"]):
                int(config["anchors"]["a_wiki_start"])
                + int(config["anchors"]["a_wiki_rows"])]]
    return {"fit": fit, "test": test, "a_slice": a_slice,
            "vocab_fp": str(sub["vocab_fingerprint"])}


def run_m175_cell_d(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible
    smoke_tokens = int(config.get("_smoke_tokens", 0))

    slices = _load(config, smoke_tokens)
    fit, test, a_slice = slices["fit"], slices["test"], slices["a_slice"]
    vocab = _vocabulary(fit)
    print(f"cell D: fit {len(fit)} / held-out {len(test)} tokens, "
          f"vocab {len(vocab)}", flush=True)

    # ---- g1: machinery anchor — reproduce A's sealed out-domain read -------
    a_evidence = json.loads(
        (REPO_ROOT / config["anchors"]["a_evidence"]).read_text(
            encoding="utf-8"))
    a_sealed = float(a_evidence["transfer"]["out_domain_wiki_ppl"])
    a_arm_vocab = _vocabulary(
        [int(x) for x in np.load(REPO_ROOT / config["corpus"]
                                 ["sub_cache_file"])["train_ids"][:60000]])
    g1_memory = ProgrammaticMemory(window=WINDOW)
    g1_memory.register(
        [int(x) for x in np.load(REPO_ROOT / config["corpus"]
                                 ["sub_cache_file"])["train_ids"][:60000]])
    ops_a: list[int] = []
    reproduced = _perplexity(g1_memory, a_slice, WINDOW, ALPHA,
                             a_arm_vocab, ops_a)
    g1_delta = abs(round(reproduced, 4) - a_sealed)
    g1_ok = g1_delta <= G1_TOLERANCE
    print(f"g1: reproduced A out-domain {reproduced:.4f} vs sealed "
          f"{a_sealed} delta {g1_delta:.4f} ok={g1_ok}", flush=True)
    if not g1_ok and not smoke:
        evidence = {
            "milestone": "M175", "cell": "D",
            "admissible_as_evidence": False, "void": True,
            "void_reason": "g1 machinery anchor failed",
            "g1_delta": g1_delta,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        return evidence

    # ---- fit-and-report on the dump ----------------------------------------
    memory = ProgrammaticMemory(window=WINDOW)
    memory.register(fit)
    ops_test: list[int] = []
    test_ppl = _perplexity(memory, test, WINDOW, ALPHA, vocab, ops_test)
    fit_set = set(fit)
    oov_rate = float(sum(1 for t in test if t not in fit_set) / len(test))
    beats_a_transfer = bool(test_ppl < a_sealed)
    print(f"dump-fitted held-out ppl {test_ppl:.4f} (A's transferred read "
          f"{a_sealed}); oov {oov_rate:.4f}", flush=True)

    evidence: dict[str, Any] = {
        "milestone": "M175",
        "cell": "D license-clean Wikipedia fit-and-report",
        "admissible_as_evidence": not smoke,
        "registered_in": config.get("registered_in"),
        "config_file": Path(config_path).name,
        "config": config,
        "corpus": {
            "sub_cache_file": config["corpus"]["sub_cache_file"],
            "full_cache_file": config["corpus"]["full_cache_file"],
            "vocab_fingerprint": slices["vocab_fp"],
            "fit_rows": len(fit),
            "held_out_rows": len(test),
            "fit_vocabulary_size": len(vocab),
            "uniform_baseline": len(vocab),
        },
        "model": {"variant": "uniform", "window": WINDOW, "alpha": ALPHA,
                  "selected_in": "cell A (valid-pinned, sealed)"},
        "anchors": {
            "g0_prefix_pin_ok": True,
            "g1_machinery_anchor": {
                "reproduced_a_out_domain_ppl": round(reproduced, 4),
                "a_sealed": a_sealed, "delta": g1_delta, "ok": g1_ok,
            },
        },
        "fit_and_report": {
            "held_out_ppl": round(test_ppl, 4),
            "footprint_bytes": memory.footprint_bytes(),
            "context_count": memory.context_count,
            "entry_count": memory.entry_count,
            "oov_rate": oov_rate,
            "ops_per_token_test_mean": round(float(np.mean(ops_test)), 2),
            "beats_a_transferred_read": beats_a_transfer,
            "note": "comparison point, not a gate (registered)",
        },
        "license_posture": config["license_posture"],
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"held_out_ppl": round(test_ppl, 4),
                      "beats_a_transfer": beats_a_transfer,
                      "g1_delta": g1_delta}, indent=1), flush=True)
    print(f"M175 cell D complete -> {output_dir / 'evidence.json'}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m175_cell_d(args.config, args.output)


if __name__ == "__main__":
    main()
