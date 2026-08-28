"""M261b acquisition — Open Images V7 (600 boxable classes):
metadata CSVs + the class-balanced capped train images and the full
test-split positive rows, downloaded from CVDF's anonymous S3 bucket.

Registered 22 Aug 2026 (plan v25, the M261 switch). License:
annotations CC BY 4.0, images listed CC BY 2.0 (commercial use with
attribution; Google's per-image caveat recorded). The attribution
obligation is recorded in the manifest; per-image License/Author
fields are joined from the official image-info CSV.

Rows: one per (image, positive human-verified label) within the 600
boxable classes. Train rows capped at 200,000, class-balanced,
seeded; test rows uncapped. Image integrity: HTTP fetch + JPEG
decodability + retry (the CVDF rescaling changes the bytes, so the
original-image MD5 does not apply — recorded).

Usage:
    python eval_v25_m261b_acquire.py --smoke     # 10 train + 10 test images
    python eval_v25_m261b_acquire.py             # the full acquisition
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

from experiments.common.data_cache import data_cache_root
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OID_ROOT = Path("F:/geode-ml/data/cache/oid")

CSVS = {
    "train_labels": ("https://storage.googleapis.com/openimages/v7/"
                     "oidv7-train-annotations-human-imagelabels.csv"),
    "val_labels": ("https://storage.googleapis.com/openimages/v7/"
                   "oidv7-val-annotations-human-imagelabels.csv"),
    "test_labels": ("https://storage.googleapis.com/openimages/v7/"
                    "oidv7-test-annotations-human-imagelabels.csv"),
    "boxable": ("https://storage.googleapis.com/openimages/v5/"
                "class-descriptions-boxable.csv"),
    "hierarchy_600": ("https://storage.googleapis.com/openimages/"
                      "2018_04/bbox_labels_600_hierarchy.json"),
    "image_info": ("https://storage.googleapis.com/openimages/2018_04/"
                   "image_ids_and_rotation.csv"),
}

S3_BASE = "https://open-images-dataset.s3.amazonaws.com"

REGISTERED = {
    "n_classes": 601,
    "class_list_note": ("the released class-descriptions-boxable.csv "
                        "carries 601 unique MIDs while the docs say "
                        "600 — the released file is the operational "
                        "list, used verbatim, discrepancy recorded"),
    "train_row_cap": 200_000,
    "seed": 20260822,
    "workers": 8,
    "retries": 2,
    "splits": {"train", "test"},
}


def _download(url: str, dest: Path, timeout: int = 300) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent":
                                               "geode-m261b/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    tmp.write_bytes(data)
    tmp.replace(dest)


def fetch_csv(name: str) -> Path:
    dest = OID_ROOT / "meta" / Path(CSVS[name]).name
    if dest.exists():
        return dest
    print(f"  downloading {name} ...", flush=True)
    _download(CSVS[name], dest, timeout=3600)
    return dest


def load_boxable_classes() -> tuple[dict[str, str], set[str]]:
    """Class set = the released class-descriptions-boxable.csv verbatim
    (601 MIDs; the docs' '600' vs the released 601 is recorded in
    REGISTERED['class_list_note'])."""
    path = fetch_csv("boxable")
    names: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for mid, name in csv.reader(fh):
            names[mid] = name
    assert len(names) == REGISTERED["n_classes"], (
        f"boxable class count {len(names)} != "
        f"{REGISTERED['n_classes']}")
    return names, set(names)


def rows_in_classes(csv_path: Path, classes: set[str],
                    split: str) -> list[tuple[str, str]]:
    """Positive (Confidence=1) rows restricted to the class set."""
    out: list[tuple[str, str]] = []
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for image_id, _src, label, conf in csv.reader(fh):
            if conf == "1" and label in classes:
                out.append((image_id, label))
    return out


def fetch_image(split: str, image_id: str, dest: Path) -> bool:
    """Download one CVDF image and verify JPEG decodability."""
    for attempt in range(REGISTERED["retries"]):
        try:
            _download(f"{S3_BASE}/{split}/{image_id}.jpg", dest,
                      timeout=120)
            from PIL import Image
            with Image.open(dest) as im:
                im.verify()
            return True
        except Exception:
            if dest.exists():
                dest.unlink(missing_ok=True)
            time.sleep(1.0 + attempt)
    return False


def download_images(rows: list[tuple[str, str]], split: str,
                    images_dir: Path) -> tuple[int, int]:
    """Download the unique images for a split with a worker pool."""
    unique = sorted({iid for iid, _ in rows})
    images_dir.mkdir(parents=True, exist_ok=True)
    missing = [iid for iid in unique
               if not (images_dir / f"{iid}.jpg").exists()]
    done = 0
    failed: list[str] = []
    lock = threading.Lock()
    if not missing:
        return len(unique), 0
    with ThreadPoolExecutor(max_workers=REGISTERED["workers"]) as pool:
        futures = {pool.submit(fetch_image, split, iid,
                               images_dir / f"{iid}.jpg"): iid
                   for iid in missing}
        for fut in as_completed(futures):
            iid = futures[fut]
            ok = fut.result()
            with lock:
                done += 1
                if not ok:
                    failed.append(iid)
                if done % 5000 == 0:
                    print(f"  {split}: {done}/{len(missing)} images, "
                          f"{len(failed)} failed", flush=True)
    return len(unique) - len(failed), len(failed)


def build_manifest(rows: list[tuple[str, str]], split: str,
                   class_names: dict[str, str],
                   license_by_id: dict[str, tuple[str, str]] | None,
                   images_dir: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for iid, label in rows:
        img = images_dir / f"{iid}.jpg"
        if not img.exists():
            continue
        rec: dict[str, Any] = {"split": split, "image_id": iid,
                               "label_mid": label,
                               "label_name": class_names.get(label,
                                                            label),
                               "image_path": str(img)}
        if license_by_id and iid in license_by_id:
            lic, author = license_by_id[iid]
            rec["license"] = lic
            rec["author"] = author
        manifest.append(rec)
    return manifest


def run_acquire(smoke: bool = False) -> dict[str, Any]:
    started = time.time()
    OID_ROOT.mkdir(parents=True, exist_ok=True)
    (OID_ROOT / "meta").mkdir(exist_ok=True)

    class_names, class_set = load_boxable_classes()
    assert len(class_names) == REGISTERED["n_classes"], (
        f"boxable class count {len(class_names)} != "
        f"{REGISTERED['n_classes']}")
    classes = class_set

    print("  reading label CSVs ...", flush=True)
    train_rows = rows_in_classes(fetch_csv("train_labels"), classes,
                                 "train")
    test_rows = rows_in_classes(fetch_csv("test_labels"), classes,
                                "test")
    val_rows = rows_in_classes(fetch_csv("val_labels"), classes, "val")

    # class-balanced, seeded train cap
    rng = random.Random(REGISTERED["seed"])
    rng.shuffle(train_rows)
    per_class: dict[str, int] = {}
    capped: list[tuple[str, str]] = []
    cap_per_class = REGISTERED["train_row_cap"] // REGISTERED[
        "n_classes"]
    for iid, label in train_rows:
        if per_class.get(label, 0) >= cap_per_class:
            continue
        per_class[label] = per_class.get(label, 0) + 1
        capped.append((iid, label))
        if len(capped) >= REGISTERED["train_row_cap"]:
            break

    if smoke:
        capped = capped[:10]
        test_rows = test_rows[:10]

    # image license/author join (attribution obligation)
    license_by_id: dict[str, tuple[str, str]] | None = None
    try:
        info_path = fetch_csv("image_info")
        license_by_id = {}
        with open(info_path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                license_by_id[row["ImageID"]] = (
                    row.get("License", ""),
                    row.get("Author", ""))
        print(f"  image-info CSV joined: {len(license_by_id)} images",
              flush=True)
    except Exception as exc:  # the join is a bonus, not a gate
        print(f"  image-info join failed (recorded): {exc}", flush=True)

    print(f"  downloading train images "
          f"({len(capped)} rows, {len({i for i, _ in capped})} unique) "
          f"...", flush=True)
    train_ok, train_fail = download_images(capped, "train",
                                           OID_ROOT / "images" / "train")
    print(f"  downloading test images "
          f"({len(test_rows)} rows, {len({i for i, _ in test_rows})} "
          f"unique) ...", flush=True)
    test_ok, test_fail = download_images(test_rows, "test",
                                         OID_ROOT / "images" / "test")

    train_manifest = build_manifest(capped, "train", class_names,
                                    license_by_id,
                                    OID_ROOT / "images" / "train")
    test_manifest = build_manifest(test_rows, "test", class_names,
                                   license_by_id,
                                   OID_ROOT / "images" / "test")

    registered_json = {k: (sorted(v) if isinstance(v, set) else v)
                       for k, v in REGISTERED.items()}
    evidence: dict[str, Any] = {
        "milestone": "M261b",
        "cell": "acquisition — Open Images V7 601-class boxable subset",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "registered": registered_json,
        "counts": {
            "train_rows_total_in_600": len(train_rows),
            "train_rows_capped": len(capped),
            "test_rows": len(test_rows),
            "val_rows_total_in_600": len(val_rows),
            "train_unique_images_downloaded": train_ok,
            "train_download_failures": train_fail,
            "test_unique_images_downloaded": test_ok,
            "test_download_failures": test_fail,
            "train_manifest_rows": len(train_manifest),
            "test_manifest_rows": len(test_manifest),
        },
        "license": {
            "annotations": "CC BY 4.0 (Google LLC)",
            "images": "listed CC BY 2.0 — commercial use with "
                      "attribution; Google's per-image caveat recorded",
            "attribution_obligation": ("the product must ship per-image "
                                       "attribution; License/Author "
                                       "joined from the official "
                                       "image-info CSV"),
        },
        "manifest_paths": {
            "train": str(OID_ROOT / "manifests" / "train_manifest.json"),
            "test": str(OID_ROOT / "manifests" / "test_manifest.json"),
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    (OID_ROOT / "manifests").mkdir(exist_ok=True)
    write_canonical_json(OID_ROOT / "manifests" / "train_manifest.json",
                         {"rows": train_manifest,
                          "payload_hash": payload_hash(train_manifest)})
    write_canonical_json(OID_ROOT / "manifests" / "test_manifest.json",
                         {"rows": test_manifest,
                          "payload_hash": payload_hash(test_manifest)})
    evidence["configuration_hash"] = payload_hash(registered_json)
    out_dir = (REPO_ROOT / "logs" / "results" / "v25" / "m261b_oid_vision")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(out_dir / "evidence_acquisition.json", evidence)
    build_artifact_index(out_dir)
    print(json.dumps({"counts": evidence["counts"]}, indent=1),
          flush=True)
    print(f"M261b acquisition complete -> "
          f"{out_dir / 'evidence_acquisition.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_acquire(smoke=args.smoke)


if __name__ == "__main__":
    main()
