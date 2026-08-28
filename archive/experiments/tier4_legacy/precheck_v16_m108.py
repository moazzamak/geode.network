"""PRE-SEAL instrument check for M108: does arm (a) at 128 atoms, encoded on
the GPU, reproduce M107's recorded 0.11171?

Registered in ``experiments/configs/v16/m108_dictionary.json`` under
``arm_a_reproduction._registered_before_measurement``. This is the check that
decides, before the sealed run commits to the GPU encode path, whether the GPU
float32 cdist stays within the registered +/- 0.002 accuracy tolerance of M107.
It is an engineering measurement of the instrument, NOT evidence.

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.precheck_v16_m108
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.common.data_cache import configure_external_cache_environment
from experiments.tier4.eval_v15_m103_atoms import (
    Whitener, _contrast_normalise, _extract_patches, _fit_zca,
)
from experiments.tier4.eval_v15_m104_experts import _load_domainnet
from experiments.tier4.eval_v15_m107_dense import (
    _class_subsample, _index_digest,
)
from experiments.tier4.eval_v16_m108_dictionary import (
    _random_order, _sparse_arm, _verify_device,
)

CONFIG = json.loads(
    (REPO_ROOT / "experiments/configs/v16/m108_dictionary.json")
    .read_text(encoding="utf-8")
)
M107 = json.loads(
    (REPO_ROOT / "logs/results/v15/m107_dense/evidence.json")
    .read_text(encoding="utf-8")
)

M107_ACC_128 = float(M107["arms"]["s_generalist_128"]
                     ["accuracy_by_penalty"]["1.0"])
TOLERANCE = CONFIG["arm_a_reproduction"]["tolerance_accuracy"]


def main() -> int:
    configure_external_cache_environment()
    device_report = _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.set_num_threads(CONFIG["numerics"]["torch_threads"])

    cc = CONFIG["corpus"]
    rep = CONFIG["representation"]
    size = cc["image_size"]
    patch, stride, pool_grid = (rep["patch"], rep["stride"], rep["pool_grid"])

    raw = _load_domainnet(size)
    train_index = _class_subsample(
        raw["train_labels"], cc["train_rows_per_class"], cc["subsample_seed"])
    test_index = _class_subsample(
        raw["test_labels"], cc["test_rows_per_class"], cc["subsample_seed"])
    corpus = {
        "train_images": raw["train_images"][train_index],
        "train_labels": raw["train_labels"][train_index],
        "train_domains": raw["train_domains"][train_index],
        "test_images": raw["test_images"][test_index],
        "test_labels": raw["test_labels"][test_index],
        "test_domains": raw["test_domains"][test_index],
    }
    digest = _index_digest({"train": train_index, "test": test_index})
    if digest != cc["expected_subsample_sha256"]:
        raise SystemExit(f"subsample digest {digest} != registered")

    rng = np.random.default_rng(rep["zca_fit_seed"])
    sample = corpus["train_images"][rng.choice(
        len(corpus["train_images"]), 20000, replace=False)]
    patches = _extract_patches(sample, patch, stride)
    grid = (size - patch) // stride + 1
    take = min(rep["zca_fit_patches"], len(patches))
    pool = _contrast_normalise(
        patches[rng.choice(len(patches), take, replace=False)],
        rep["contrast_epsilon"])
    mean, whiten = _fit_zca(pool, rep["zca_epsilon"])
    whitener = Whitener(patch, stride, rep["contrast_epsilon"], mean, whiten, grid)

    seed = CONFIG["sparse"]["dictionary_seed"]
    pool_size = CONFIG["sparse"]["candidate_pool_size"]
    srng = np.random.default_rng(seed)
    candidates = ((pool[srng.choice(len(pool), pool_size, replace=False)]
                   - mean) @ whiten).astype(np.float32)

    order = np.random.default_rng(cc["shuffle_seed"]).permutation(len(train_index))
    validation_rows = int(round(
        len(order) * CONFIG["head"]["selection_validation_fraction"]))
    penalties = [float(p) for p in CONFIG["head"]["regularisation_grid"]]
    classes = int(corpus["train_labels"].max()) + 1

    dictionary = candidates[_random_order(candidates, pool_size, seed)[:128]]
    print(f"arm (a) at 128 atoms on the GPU, {len(train_index)} train rows, "
          f"{validation_rows} validation rows", flush=True)
    payload = _sparse_arm(corpus, dictionary, whitener, pool_grid, order,
                          penalties, classes, validation_rows, device)
    chosen = max(payload["validation_accuracy_by_penalty"].items(),
                 key=lambda kv: (kv[1], -float(kv[0])))
    chosen_penalty = float(chosen[0])  # the dict keys are strings ("1.0")
    m107_penalty = float(M107["head"]["chosen_penalty"])
    accuracy = float(payload["accuracy_by_penalty"]["1.0"])
    delta = accuracy - M107_ACC_128

    print(f"  chosen penalty on arm (a) 128: {chosen_penalty} "
          f"(M107 chose {m107_penalty})", flush=True)
    print(f"  accuracy@1.0: {accuracy:.5f}  M107: {M107_ACC_128:.5f}  "
          f"delta: {delta:+.5f}  tolerance: +/-{TOLERANCE}", flush=True)
    ok = chosen_penalty == m107_penalty and abs(delta) <= TOLERANCE
    print(f"  PRE-SEAL CHECK: {'PASSED — GPU encode path is safe to seal.' if ok else 'FAILED — the sealed run must switch the encode to CPU.'}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
