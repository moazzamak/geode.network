"""M266a — Whisper ASR arm: frozen whisper-small.en, greedy decode,
WER on LibriSpeech test-clean (CC-BY-4.0).

Registered and dispatched 21 Aug 2026 (plan v25, M266 + amendment
13), local-first, F: cache conventions. Anchor-first: reproduce the
official model card's whisper-small.en LibriSpeech test-clean WER
(3.053) with the card's own protocol before our reading is the
result.

Honesty notes, registered before the run:
- the model is a PUBLISHER checkpoint (MIT per the project LICENSE;
  the HF card displays apache-2.0 — both recorded); frozen, never
  trained here;
- the anchor is reproduced, not beaten; any difference between the
  anchor and our reading is reported, never hidden (word-level WER
  implementations may differ in tokenisation at the margin);
- per-sample duration and RMS energy are recorded (the registered
  input guard); all test-clean samples are expected in-distribution;
- evidence at logs/results/v25/m266_audio_arm/evidence.json.
"""
from __future__ import annotations

import argparse
import hashlib
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
                  / "m266a_whisper_asr.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m266_audio_arm")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Word-level edit distance (the classic WER metric, O(|a||b|))."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1,
                          prev[j - 1] + (x != y))
        prev = curr
    return prev[-1]


def word_error_rate(reference: str, prediction: str) -> float:
    ref = reference.split()
    hyp = prediction.split()
    if not ref:
        return 1.0 if hyp else 0.0
    return _levenshtein(ref, hyp) / len(ref)


def run_m266a(config_path: Path, output_dir: Path, smoke: bool = False
              ) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    cache_root = data_cache_root() / config["feature_cache_relpath"]
    cache_root.mkdir(parents=True, exist_ok=True)

    import torch
    import soundfile as sf
    from datasets import Audio, load_dataset as _hf_load
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(
        config["encoder"]["checkpoint"])
    model = WhisperForConditionalGeneration.from_pretrained(
        config["encoder"]["checkpoint"])
    model.to(device).eval()

    ds = _hf_load(config["dataset"]["hf_id"], config["dataset"]["config"],
                  split=config["dataset"]["split"])
    # torchcodec is incompatible with the ROCm torch build; decode the
    # cached FLAC paths ourselves with soundfile (registered env note)
    ds = ds.cast_column("audio", Audio(decode=False))
    max_samples = config["smoke"]["max_samples"] if smoke else None
    if max_samples is not None:
        ds = ds.select(range(max_samples))

    normalize = processor.tokenizer.normalize
    wer_sum = 0.0
    n_words = 0
    per_sample: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    throttle = 0.01  # the registered display-GPU TDR mitigation
    for i, row in enumerate(ds):
        audio = row["audio"]
        # torchcodec is incompatible with the ROCm torch build; the
        # parquet embeds the encoded audio as bytes (the recorded
        # "path" is the original name, not a local file) — decode
        # with soundfile from BytesIO (registered env note)
        array, sample_rate = sf.read(io.BytesIO(audio["bytes"]),
                                     dtype="float32", always_2d=False)
        if array.ndim > 1:
            array = array.mean(axis=1)  # downmix to mono (none expected)
        duration = float(len(array)) / float(sample_rate)
        energy = float(np.sqrt((array ** 2).mean()))
        lo, hi = config["guard"]["duration_range_seconds"]
        ok_duration = lo <= duration <= hi
        ok_energy = energy >= config["guard"]["energy_floor"]
        if not (ok_duration and ok_energy):
            flagged.append({"index": i, "duration": duration,
                            "energy": energy})
            continue
        with torch.no_grad():
            feats = processor(array, sampling_rate=sample_rate,
                              return_tensors="pt").input_features.to(device)
            ids = model.generate(feats, num_beams=1)[0]
        pred = normalize(processor.decode(ids, skip_special_tokens=True))
        ref = normalize(row["text"])
        wer = word_error_rate(ref, pred)
        wer_sum += _levenshtein(ref.split(), pred.split())
        n_words += max(len(ref.split()), 1)
        per_sample.append({"index": i, "wer": wer, "duration": duration,
                           "energy": energy, "reference": ref,
                           "prediction": pred})
        if throttle:
            time.sleep(throttle)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(ds)} running WER "
                  f"{wer_sum / n_words:.4f}", flush=True)

    overall_wer = float(wer_sum / n_words) if n_words else float("nan")
    evidence: dict[str, Any] = {
        "milestone": "M266a",
        "cell": "Whisper ASR arm (frozen whisper-small.en, greedy decode)",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "anchor": config["anchor"],
        "license_recorded": config["encoder"]["license_recorded"],
        "evaluation": {
            "split": "librispeech_asr/clean/test",
            "n_samples": len(per_sample),
            "overall_wer": overall_wer,
            "anchor_wer": config["anchor"]["published_wer_test_clean"],
            "reading": ("our held-out greedy WER, reported against the "
                        "published anchor — reproduced, never beaten"),
        },
        "guard": {
            "flagged_out_of_range": flagged,
            "n_flagged": len(flagged),
        },
        "per_sample": per_sample,
        "scope_note": ("publisher checkpoint frozen; one held-out read; "
                       "the anchor is the official card protocol"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"n_samples": len(per_sample),
                      "overall_wer": overall_wer,
                      "anchor_wer": config["anchor"][
                          "published_wer_test_clean"],
                      "n_flagged": len(flagged)}, indent=1), flush=True)
    print(f"M266a complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output
    if args.smoke and output == DEFAULT_OUTPUT:
        output = DEFAULT_OUTPUT.parent / (DEFAULT_OUTPUT.name + "_smoke")
    run_m266a(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
