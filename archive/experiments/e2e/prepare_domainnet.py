"""Download and verify the pinned Hugging Face DomainNet snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.common.data_cache import configure_external_cache_environment
from src.runtime.domainnet_manifest import DomainNetFile, DomainNetManifest


REPOSITORY = "wltjr1007/DomainNet"
REVISION = "ee20570ae7a29c51571e55a9a17983f7625295d6"
FILES = (
    DomainNetFile(
        "data/test-00000-of-00001.parquet",
        "7eb34c7c9c020f265db6c4b2405c873f4bda0259cd06b43aa31df45a17a55409",
        5_597_563_518,
    ),
    DomainNetFile(
        "data/train-00000-of-00003.parquet",
        "37dfda4256254a53d58352ba6f3ea8a1ae24d13f3d39eb27a143b859f73b3e5a",
        758_577_202,
    ),
    DomainNetFile(
        "data/train-00001-of-00003.parquet",
        "c5d86606a2fa7b1418895717803cfa7c3e7adad37454ee7b1ad44f1ec0eb3e15",
        7_205_696_451,
    ),
    DomainNetFile(
        "data/train-00002-of-00003.parquet",
        "bb3dd680e02ac1cf539fb6d32959b096a06535892aa8c494e31378762f02613c",
        4_959_599_036,
    ),
)


def prepare_domainnet(cache_root: Path | None = None) -> dict:
    configured_root = configure_external_cache_environment()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError("huggingface_hub is required for DomainNet") from error
    root = cache_root.resolve() if cache_root is not None else configured_root
    dataset_root = root / "domainnet"
    repository_root = dataset_root / "repository"
    repository_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPOSITORY,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=[item.path for item in FILES],
        local_dir=repository_root,
    )
    manifest = DomainNetManifest(
        files=FILES,
        class_count=345,
        version="huggingface-parquet-v1",
        source_repository=REPOSITORY,
        source_revision=REVISION,
        split_samples=(("train", 409_832), ("test", 176_743)),
    )
    manifest_path = dataset_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report = manifest.verify(repository_root)
    return {
        "cache_root": str(root),
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        **report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = prepare_domainnet(arguments.cache_root)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8", newline="\n")
    print(serialized, end="")


if __name__ == "__main__":
    main()