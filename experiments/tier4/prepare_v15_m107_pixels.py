"""Pre-materialise M107's original-resolution pixels while M104 is running.

This does nothing the sealed M107 run would not do on its own. It exists so
that the JPEG decode -- which is I/O and single-core bound, and which the
sealed run would otherwise pay for on its critical path while sixteen cores sit
idle -- happens during M104's tail instead, and so that a disk or decode
failure surfaces now rather than fourteen hours into a run.

The output is content-addressed by the subsample digest, so it is reused only
if the subsample it was drawn for is the one the run asks for. Deleting it
costs nothing but time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("GEODE_CACHE_DIR", r"D:\geode-ml\data\cache")

from experiments.tier4.eval_v15_m104_experts import _load_domainnet  # noqa: E402
from experiments.tier4.eval_v15_m107_dense import (  # noqa: E402
    DEFAULT_CONFIG,
    _class_subsample,
    _index_digest,
    _materialise_original,
    _verify_pixel_identity,
)


def main() -> None:
    config = json.loads(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))
    corpus = config["corpus"]
    size = corpus["image_size"]
    raw = _load_domainnet(size)

    index = {
        "train": _class_subsample(raw["train_labels"],
                                  corpus["train_rows_per_class"],
                                  corpus["subsample_seed"]),
        "test": _class_subsample(raw["test_labels"],
                                 corpus["test_rows_per_class"],
                                 corpus["subsample_seed"]),
    }
    digest = _index_digest(index)
    expected = corpus.get("expected_subsample_sha256")
    if expected and expected != digest:
        raise SystemExit(
            f"subsample digest {digest} does not match the pinned {expected}"
        )
    print(f"subsample {digest[:16]}  train {len(index['train'])}  "
          f"test {len(index['test'])}", flush=True)

    resolutions = sorted({
        int(arm["resolution"]) for arm in config["dense"]["arms"]
        if arm["pixels"] == "original"
    })
    print(f"resolutions {resolutions}", flush=True)

    for split in ("train", "test"):
        rows = index[split]
        checked = corpus["pixel_identity_rows"]
        _verify_pixel_identity(split, rows,
                               raw[f"{split}_images"][rows[:checked]], size,
                               checked)
        print(f"  {split}: pixel identity verified bitwise", flush=True)
        paths = _materialise_original(split, rows, resolutions, digest[:16])
        for resolution in resolutions:
            path = paths[resolution]
            print(f"  {split} {resolution}: {path.stat().st_size / 1e9:.2f} GB",
                  flush=True)


if __name__ == "__main__":
    main()
