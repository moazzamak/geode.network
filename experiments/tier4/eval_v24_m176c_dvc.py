"""M176c-dvc — the rental device-verification cell.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` §12 (18 Aug
2026). The rental's numbers are gated on device parity, because the M108
instrument bit-pins the sealed SPM encoder to the local ROCm gfx1201 and
refuses every other GPU by design.

What this cell measures (register-before-run):
- G-dvc1 (inside `_load_corpus`): the subsample-index digest must equal
  the registered `expected_subsample_sha256` — proves the rental decodes
  the SAME rows.
- G-dvc2: re-encode the first `t1_encoder_check_rows` TRAIN rows at the
  6,144-atom dictionary with the M117-exact whitener (CPU numpy,
  device-independent) and the GPU code path on the rental, and compare
  against the sealed reference `v16/m142_c2/_check_pool6144.npy` (the
  same rows encoded by the sealed device during M142 C2, itself verified
  against the sealed f6144 memmap there).

Verdict: max-abs delta == 0.0 -> BIT-EXACT parity; <= 1e-6 -> NUMERICAL
parity (anchors may be reproduced with the sealed tolerances only, and
every rental evidence file must carry the dvc delta); > 1e-6 -> the
rental is VOID for the sealed pipeline. This runner deliberately does
NOT call `_verify_device` — that gate is the subject of this cell.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m107_dense import _verify_pixel_identity
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m113_learned import (
    _build_whitener_and_candidates,
    _random_dictionary,
)
from experiments.tier4.eval_v16_m115_lofi import _write_frozen_codes

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m176c_dvc.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v24" / "m176c_dvc")
REFERENCE_RELPATH = Path("v16") / "m142_c2" / "_check_pool6144.npy"


def run_m176c_dvc(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    torch.manual_seed(int(config["numerics"]["seed"]))
    configure_external_cache_environment()
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("loading corpus (subsample digest gate inside)", flush=True)
    corpus, train_index, test_index = _load_corpus(config)
    size = int(config["corpus"]["image_size"])
    for split, idx in (("train", train_index), ("test", test_index)):
        _verify_pixel_identity(split, idx, corpus[f"{split}_images"], size,
                               int(config["corpus"]["pixel_identity_rows"]))
    print("G-dvc1 corpus identity: PASS", flush=True)

    rep = config["sparse"]
    print("building sealed whitener + 6144-atom dictionary", flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    dict_6144 = _random_dictionary(candidates, len(candidates),
                                   int(rep["dictionary_seed"]), 6144)

    check_rows = int(config["anchors"]["t1_encoder_check_rows"])
    if smoke:
        check_rows = min(check_rows, int(config.get("_smoke_rows", 8)))

    output_dir.mkdir(parents=True, exist_ok=True)
    fresh_path = output_dir / "dvc_pool6144_encode.npy"
    print(f"encoding {check_rows} rows on the rental device", flush=True)
    fresh = _write_frozen_codes(
        corpus, dict_6144, whitener, 2, device, np.arange(check_rows),
        fresh_path, split="train",
        throttle_seconds=float(config["numerics"]["encode_throttle_seconds"]))

    ref_path = data_cache_root() / REFERENCE_RELPATH
    if not ref_path.exists():
        raise SystemExit(
            f"reference missing: {ref_path} — ship the sealed "
            "v16/m142_c2/_check_pool6144.npy to the rental cache first.")
    ref = np.load(ref_path, mmap_mode="r")
    if ref.shape[0] < check_rows:
        raise SystemExit(f"reference has {ref.shape[0]} rows < {check_rows}")
    delta = float(np.abs(
        np.asarray(ref[:check_rows], dtype=np.float64)
        - np.asarray(fresh, dtype=np.float64)).max())

    verdict = ("bit-exact" if delta == 0.0
               else "numerical" if delta <= 1e-6 else "void")
    evidence: dict[str, Any] = {
        "milestone": "M176c-dvc",
        "cell": "rental device verification (pool-6144 check rows)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "question": config["question"],
        "interpretation_registered_before_running":
            config["interpretation_registered_before_running"],
        "device": {
            "name": torch.cuda.get_device_properties(0).name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "check_rows": check_rows,
        "reference_relpath": REFERENCE_RELPATH.as_posix(),
        "reference_sha256": _sha256_file(ref_path),
        "max_abs_delta": delta,
        "verdict": verdict,
        "consequence": {
            "bit-exact": "full parity: the rental may run the sealed "
                         "runners behind a registered device override",
            "numerical": "anchors may be reproduced with the sealed "
                         "tolerances only; every rental evidence carries "
                         "this delta",
            "void": "the rental is void for the sealed pipeline; M176c "
                    "candidates need a device-independent design",
        }[verdict],
        "void": verdict == "void",
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"verdict {verdict}: max-abs delta {delta:.3e}", flush=True)
    print(f"M176c-dvc complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def _sha256_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m176c_dvc(args.config, args.output)


if __name__ == "__main__":
    main()
