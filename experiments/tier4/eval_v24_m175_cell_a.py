"""M175 cell A — text->text transfer: the wikitext-fitted additive encoder
read on a Wikipedia dump slice (valid-pinned arm, machinery anchor,
in-domain vs out-domain).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` section 12
(18 Aug 2026) and ``analysis/v24_m175_transfer_cells.md``.

Slices (registered): fit = wikitext-100k train[:60000]; valid =
train[60000:80000]; in-domain test = the 100k test (20k); Wikipedia
slice = wikitext103_full train_ids[100000:200000] (100k, mmap) —
disjoint by construction (the 100k train is the verified prefix of the
full train stream).

Steps (registered before any test number):
1. selection on VALID: for variant x window over {uniform, backoff} x
   {1,2,4,8,16,32}, fit 60k, valid ppl; arm = argmin valid ppl; the
   whole selection table is reported.
2. g1 machinery anchor: the selected variant at its window reproduces
   the A0 SEALED number (A0's exact 80k fit, 20k test) with delta 0.0.
3. transfer: the frozen selected encoder (60k fit) scored on wikitext
   test (in-domain) and the Wikipedia slice (out-domain); slice OOV
   rate vs fit vocab reported.

Verdict (registered): transfer holds if out-domain ppl <= 2.0 x
in-domain ppl; else A closes as a scoped negative for text->text
transfer of this construction. Uniform baseline = fit-vocab size.
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
    _backoff_perplexity,
    _perplexity,
    _vocabulary,
)
from src.programmatic_memory import ProgrammaticMemory

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m175_cell_a.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v24" / "m175_cell_a"

G1_TOLERANCE = 0.0
TRANSFER_FACTOR = 2.0


def _load_slices(config: dict[str, Any], smoke_tokens: int
                 ) -> dict[str, Any]:
    sub = np.load(REPO_ROOT / config["corpus"]["sub_cache_file"])
    full = np.load(REPO_ROOT / config["corpus"]["full_cache_file"],
                   mmap_mode="r")
    fit_n = int(config["slices"]["fit_rows"])
    valid_n = int(config["slices"]["valid_rows"])
    wiki_start = int(config["slices"]["wiki_start"])
    wiki_rows = int(config["slices"]["wiki_rows"])
    if smoke_tokens:
        fit_n = min(fit_n, smoke_tokens)
        valid_n = min(valid_n, max(1, smoke_tokens // 4))
    # g0: the 100k train is exactly the full train stream's prefix.
    prefix_ok = bool(np.array_equal(sub["train_ids"],
                                    full["train_ids"][:len(sub["train_ids"])]))
    if not prefix_ok:
        raise SystemExit("M175 A VOID: g0 prefix pin failed (100k train is "
                         "not the full train stream prefix)")
    fit = [int(x) for x in sub["train_ids"][:fit_n]]
    valid = [int(x) for x in sub["train_ids"][fit_n:fit_n + valid_n]]
    test = [int(x) for x in sub["test_ids"]]
    if smoke_tokens:
        test = test[:max(1, smoke_tokens // 4)]
    wiki = [int(x) for x in full["train_ids"]
            [wiki_start:wiki_start + wiki_rows]]
    if smoke_tokens:
        wiki = wiki[:max(1, smoke_tokens // 2)]
    return {"fit": fit, "valid": valid, "test": test, "wiki": wiki,
            "sub_vocab_fp": str(sub["vocab_fingerprint"])}


def run_m175_cell_a(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible
    smoke_tokens = int(config.get("_smoke_tokens", 0))

    slices = _load_slices(config, smoke_tokens)
    fit, valid, test, wiki = (slices["fit"], slices["valid"],
                              slices["test"], slices["wiki"])
    vocab = _vocabulary(fit)
    alpha = float(config["model"]["alpha"])
    ladder = [int(w) for w in config["model"]["window_ladder"]]
    variants = ["uniform", "backoff"]
    print(f"slices: fit {len(fit)} / valid {len(valid)} / test {len(test)} / "
          f"wiki {len(wiki)}; vocab {len(vocab)}", flush=True)

    # ---- step 1: valid-based selection table --------------------------------
    selection: dict[str, dict[str, Any]] = {}
    for variant in variants:
        for window in ladder:
            memory = ProgrammaticMemory(window=window)
            memory.register(fit)
            ops: list[int] = []
            if variant == "uniform":
                ppl = _perplexity(memory, valid, window, alpha, vocab, ops)
            else:
                ppl, _hist = _backoff_perplexity(memory, valid, alpha,
                                                 vocab, ops)
            selection[f"{variant}-w{window}"] = {
                "variant": variant, "window": window,
                "valid_perplexity": round(ppl, 4),
                "footprint_bytes": memory.footprint_bytes(),
            }
            print(f"  {variant} w{window}: valid {ppl:.4f}", flush=True)
    arm_key = min(selection, key=lambda k: selection[k]["valid_perplexity"])
    arm = selection[arm_key]
    print(f"selected arm: {arm_key}", flush=True)

    # ---- step 2: g1 — reproduce the A0 sealed number for this arm -----------
    a0_evidence = json.loads(
        (REPO_ROOT / config["anchors"]["a0_evidence"]).read_text(
            encoding="utf-8"))
    a0_cells = (a0_evidence["per_window"] if arm["variant"] == "uniform"
                else a0_evidence["per_window_backoff"])
    a0_sealed = float(
        a0_cells[str(arm["window"])]["test_perplexity"])
    full_memory = ProgrammaticMemory(window=arm["window"])
    g1_train = [int(x) for x in np.load(
        REPO_ROOT / config["corpus"]["sub_cache_file"])["train_ids"]]
    full_memory.register(g1_train)
    # g1 always reproduces A0's EXACT computation: full 80k fit, full 80k
    # vocabulary, full 20k test — independent of any smoke truncation.
    g1_vocab = _vocabulary(g1_train)
    g1_test = [int(x) for x in np.load(
        REPO_ROOT / config["corpus"]["sub_cache_file"])["test_ids"]]
    ops_a0: list[int] = []
    if arm["variant"] == "uniform":
        reproduced = _perplexity(full_memory, g1_test, arm["window"], alpha,
                                 g1_vocab, ops_a0)
    else:
        reproduced, _h = _backoff_perplexity(full_memory, g1_test, alpha,
                                             g1_vocab, ops_a0)
    g1_delta = abs(round(reproduced, 4) - a0_sealed)
    g1_ok = g1_delta <= G1_TOLERANCE
    print(f"g1: reproduced {reproduced:.4f} vs A0 sealed {a0_sealed} "
          f"delta {g1_delta:.4f} ok={g1_ok}", flush=True)
    if not g1_ok and not smoke:
        evidence = {
            "milestone": "M175", "cell": "A",
            "admissible_as_evidence": False, "void": True,
            "void_reason": "g1 machinery anchor failed",
            "g1_delta": g1_delta,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        return evidence

    # ---- step 3: transfer with the frozen 60k-fit selected encoder ----------
    arm_memory = ProgrammaticMemory(window=arm["window"])
    arm_memory.register(fit)
    ops_in: list[int] = []
    ops_out: list[int] = []
    if arm["variant"] == "uniform":
        in_ppl = _perplexity(arm_memory, test, arm["window"], alpha, vocab,
                             ops_in)
        out_ppl = _perplexity(arm_memory, wiki, arm["window"], alpha, vocab,
                              ops_out)
    else:
        in_ppl, _hi = _backoff_perplexity(arm_memory, test, alpha, vocab,
                                          ops_in)
        out_ppl, _ho = _backoff_perplexity(arm_memory, wiki, alpha, vocab,
                                           ops_out)
    fit_set = set(fit)
    oov_rate = float(sum(1 for t in wiki if t not in fit_set) / len(wiki))
    transfer_holds = bool(out_ppl <= TRANSFER_FACTOR * in_ppl)
    print(f"transfer: in-domain {in_ppl:.4f} out-domain {out_ppl:.4f} "
          f"oov {oov_rate:.4f} holds={transfer_holds}", flush=True)

    evidence: dict[str, Any] = {
        "milestone": "M175",
        "cell": "A text->text transfer (wikitext -> Wikipedia slice)",
        "admissible_as_evidence": not smoke,
        "registered_in": config.get("registered_in"),
        "config_file": Path(config_path).name,
        "config": config,
        "slices": {
            "fit_rows": len(fit), "valid_rows": len(valid),
            "test_rows": len(test), "wiki_rows": len(wiki),
            "sub_vocab_fingerprint": slices["sub_vocab_fp"],
        },
        "selection_table": selection,
        "selected_arm": arm,
        "anchors": {
            "g0_prefix_pin_ok": True,
            "g1_machinery_anchor": {
                "reproduced_test_ppl": round(reproduced, 4),
                "a0_sealed": a0_sealed,
                "delta": g1_delta,
                "ok": g1_ok,
                "note": "A0's exact 80k fit on the 20k test",
            },
        },
        "transfer": {
            "in_domain_test_ppl": round(in_ppl, 4),
            "out_domain_wiki_ppl": round(out_ppl, 4),
            "gap_factor": round(out_ppl / in_ppl, 4),
            "oov_rate_wiki_vs_fit_vocab": oov_rate,
            "uniform_baseline": len(vocab),
            "threshold": round(TRANSFER_FACTOR * in_ppl, 4),
            "transfer_holds": transfer_holds,
            "reading": (config["verdict"]["consequence_pass"]
                        if transfer_holds
                        else config["verdict"]["consequence_fail"]),
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({
        "selected_arm": arm_key,
        "in_domain": round(in_ppl, 4),
        "out_domain": round(out_ppl, 4),
        "oov": oov_rate,
        "transfer_holds": transfer_holds,
    }, indent=1), flush=True)
    print(f"M175 cell A complete -> {output_dir / 'evidence.json'}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m175_cell_a(args.config, args.output)


if __name__ == "__main__":
    main()
