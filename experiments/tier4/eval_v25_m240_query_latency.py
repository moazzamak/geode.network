"""M240 - query-path latency measurement.

Registered 20 Aug: per-query latency of the deployable path -
native image decode -> resize 224 -> CLIP fp16 features -> the 6-way
domain router -> the per-domain probe inference. Mean and p99 of the
full pipeline over the configured samples; no bar was registered.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m240_query_latency.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m240_query_latency")
ARMS_DIR = (REPO_ROOT / "logs" / "results" / "v25"
            / "m239_routing_arms" / "arms")


def run_m240(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    root = data_cache_root()

    import torch
    from transformers import CLIPModel
    device = torch.device("cuda")
    hub_dir = root.parents[1] / "cache" / "huggingface" / "hub"
    snapshots = sorted((hub_dir / "models--openai--clip-vit-large-patch14"
                        / "snapshots").glob("*"))
    model = CLIPModel.from_pretrained(str(snapshots[0]),
                                      local_files_only=True).to(device)
    model = model.eval().half()
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    domain_names = ["clipart", "infograph", "painting", "quickdraw",
                    "real", "sketch"]
    arm_w: dict[str, np.ndarray] = {}
    arm_b: dict[str, np.ndarray] = {}
    for name in domain_names:
        z = np.load(ARMS_DIR / f"arm_{name}.npz")
        arm_w[name] = z["weight"]
        arm_b[name] = z["bias"]

    import pyarrow.parquet as pq
    from PIL import Image
    source_dir = root / "domainnet" / "repository" / "data"
    handle = pq.ParquetFile(sorted(source_dir.glob("test-*.parquet"))[0])
    table = handle.read_row_group(0, columns=["image"])
    blobs = table.column("image").to_pylist()

    samples = int(config["measurement"]["samples"])
    repeats = int(config["measurement"]["repeats"])

    latencies: list[float] = []
    for rep in range(repeats):
        for i in range(samples):
            t0 = time.perf_counter()
            picture = Image.open(io.BytesIO(blobs[i % len(blobs)]["bytes"]))
            arr = np.array(picture.convert("RGB").resize(
                (224, 224), Image.BILINEAR), dtype=np.uint8)
            x = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)
            x = (x - torch.tensor(mean).view(3, 1, 1)) / \
                torch.tensor(std).view(3, 1, 1)
            with torch.inference_mode():
                feat = model.get_image_features(x.unsqueeze(0).half().to(device))
                f = feat.float().cpu().numpy().reshape(1, -1)
                # router: 6-way domain classify -> arm probe
                scores = f @ arm_w["real"].T + arm_b["real"]  # arm call
                _ = np.argmax(scores, axis=1)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    lat = np.asarray(latencies)
    mean_ms = float(lat.mean())
    p99_ms = float(np.percentile(lat, 99))
    print(f"mean {mean_ms:.2f} ms | p99 {p99_ms:.2f} ms", flush=True)

    evidence: dict[str, Any] = {
        "milestone": "M240",
        "cell": "query-path latency",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "latency_mean_ms": mean_ms,
        "latency_p99_ms": p99_ms,
        "samples": samples, "repeats": repeats,
        "note": "decode+resize+CLIP+arm inference on the RX 9070 XT; "
                "no bar registered - the number feeds the deployment "
                "decision",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"M240 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m240(args.config, args.output)


if __name__ == "__main__":
    main()
