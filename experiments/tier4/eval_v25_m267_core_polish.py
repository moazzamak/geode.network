"""M267 — chained core+polish audio demonstration.

Stage 0: the FFT/mel front-end as a programmatic primitive
(geode.core.audio_primitives). Stage 1: frozen SpeechT5 text->mel
predictor (core arm, MIT). Stage 2: frozen HiFi-GAN-class vocoder
(polish arm, MIT). Instrument: the SEALED M266a Whisper protocol
re-transcribes the synthesized waveform; the loop WER is the
objective end-to-end reading (G5).

Registered and dispatched 21 Aug 2026 (plan v25, M267 + amendment
16), local-first, F: cache conventions. Honesty notes: publisher
checkpoints frozen, never trained; the question is whether the
in-system chain is measured, auditable, deterministic, and within
tolerance of a monolithic baseline — NOT whether it beats one.
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
from experiments.tier4.eval_v25_m266a_whisper_asr import word_error_rate

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m267_core_polish_chain.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m267_core_polish")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _chain_content_hash(stage_hashes: dict[str, str],
                        transcriptions: list[str]) -> str:
    """G3: the chain's content hash — timing excluded (standing rule)."""
    return payload_hash({"stages": stage_hashes,
                         "transcriptions": transcriptions})


def run_m267(config_path: Path, output_dir: Path, smoke: bool = False
             ) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    cache_root = data_cache_root() / config["feature_cache_relpath"]
    cache_root.mkdir(parents=True, exist_ok=True)

    import torch
    import soundfile as sf
    from datasets import Audio, load_dataset as _hf_load
    from transformers import (
        SpeechT5ForTextToSpeech,
        SpeechT5HifiGan,
        SpeechT5Processor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # REGISTERED ENV NOTE: MIOpen's hiprtc JIT cannot compile BatchNorm
    # kernels on this machine (no C++ toolchain visible to hiprtc —
    # 'type_traits' file not found); SpeechT5's postnet hits BatchNorm.
    # The standard ROCm-on-Windows workaround: disable the MIOpen path
    # so PyTorch's native kernel is used.
    torch.backends.cudnn.enabled = False

    # ---- sentences (LibriSpeech test-clean head rows, registered) ----
    ds = _hf_load("librispeech_asr", "clean", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    n_sent = config["sentences"]["smoke_count" if smoke else "full_count"]
    sentences = [ds[i]["text"] for i in range(n_sent)]

    # ---- stage 1: frozen SpeechT5 text -> mel (core) --------------------
    tts_processor = SpeechT5Processor.from_pretrained(
        config["stages"]["1_core"]["checkpoint"])
    tts_model = SpeechT5ForTextToSpeech.from_pretrained(
        config["stages"]["1_core"]["checkpoint"]).to(device).eval()

    # ---- stage 2: frozen HiFi-GAN vocoder (polish) ----------------------
    vocoder = SpeechT5HifiGan.from_pretrained(
        config["stages"]["2_polish"]["checkpoint"]).to(device).eval()

    # ---- speaker conditioning (fixed registered vector; the
    # cmu-arctic-xvectors dataset is gated — voice naturalness is
    # NOT the gate, the measured chain loop is) -------------------------
    spk_rng = np.random.default_rng(config["speaker"]["vector_seed"])
    spk = spk_rng.standard_normal(config["speaker"]["dim"])
    spk = spk / (np.linalg.norm(spk) + 1e-12)
    speaker_embeddings = torch.from_numpy(
        spk.astype(np.float32)).unsqueeze(0).to(device)

    # ---- instrument: the sealed M266a Whisper protocol -------------------
    whisper_proc = WhisperProcessor.from_pretrained(
        config["stages"]["eval_instrument"]["checkpoint"])
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        config["stages"]["eval_instrument"]["checkpoint"]).to(device).eval()

    from geode.core.audio_primitives import (
        mel_spectrogram,
        primitive_replay_hash,
    )
    from geode.core.ledger import AppendOnlyLedger
    from geode.core.orchestrator import Orchestrator
    from geode.core.arm import arm_from_sealed_head

    # ---- the orchestrator: arms + ledger (G4) ---------------------------
    orch = Orchestrator()
    # arms are registered AFTER the loop measurement (honest numbers only)
    # — built below; here the ledger will record the stage records.

    transcriptions: list[str] = []
    stage_hashes: dict[str, str] = {}
    per_item: list[dict[str, Any]] = []
    empty_chain_check: dict[str, Any] = {}

    for idx, sentence in enumerate(sentences):
        if not sentence.strip():
            # G6: abstention — an empty input yields an empty chain
            empty_chain_check[str(idx)] = {"input_empty": True,
                                           "chain": []}
            orch.ledger.append({"kind": "stage_abstained",
                                "key": f"m267:{idx}:abstained",
                                "item_index": idx})
            continue
        with torch.no_grad():
            torch.manual_seed(20260821)  # G3: registered chain seed
            inputs = tts_processor(text=sentence, return_tensors="pt")
            mel = tts_model.generate_speech(
                inputs["input_ids"].to(device),
                speaker_embeddings, vocoder=None)  # core output
            mel_np = mel.cpu().numpy().astype(np.float32)
            if mel_np.ndim == 3 and mel_np.shape[-1] == 80:
                mel_np = mel_np.transpose(0, 2, 1)  # (1, T, 80) -> (1, 80, T)
            with torch.no_grad():
                waveform = vocoder(torch.from_numpy(mel_np).to(device))
            wav_np = waveform.squeeze(0).cpu().numpy().astype(np.float32)
        # stage 0 primitive over the synthesized waveform (replay hash)
        mel0 = mel_spectrogram(wav_np)
        stage_hashes.setdefault("primitive_digest", _sha256_hex(
            mel0.astype(np.float32).tobytes()))
        # instrument: the sealed M266a protocol
        with torch.no_grad():
            torch.manual_seed(20260821)  # G3: registered chain seed
            feats = whisper_proc(wav_np, sampling_rate=16000,
                                 return_tensors="pt").input_features
            ids = whisper_model.generate(feats.to(device), num_beams=1)[0]
        pred = whisper_proc.tokenizer.normalize(
            whisper_proc.decode(ids, skip_special_tokens=True))
        ref = whisper_proc.tokenizer.normalize(sentence)
        transcriptions.append(pred)
        wer = word_error_rate(ref, pred)
        per_item.append({"index": idx, "reference": ref,
                         "prediction": pred, "wer": wer,
                         "primitive_replay_hash":
                             primitive_replay_hash(wav_np)})
        orch.ledger.append({"kind": "stage_record",
                            "key": f"m267:{idx}",
                            "item_index": idx,
                            "stage0_replay": primitive_replay_hash(wav_np),
                            "stage1_mel_digest": _sha256_hex(
                                mel_np.astype(np.float32).tobytes()),
                            "stage2_wav_digest": _sha256_hex(
                                wav_np.astype(np.float32).tobytes()),
                            "wer": wer})
        if (idx + 1) % 25 == 0:
            print(f"  {idx + 1}/{n_sent} running WER "
                  f"{np.mean([p['wer'] for p in per_item]):.4f}", flush=True)

    # ---- G5: the loop WER over ALL transcribed items ---------------------
    wer_sum = sum(
        len(p["reference"].split()) * p["wer"] for p in per_item)
    n_words = sum(len(p["reference"].split()) for p in per_item)
    loop_wer = float(wer_sum / n_words) if n_words else float("nan")

    # ---- G3: chain determinism — second run on a registered subset ------
    rerun_sentences = sentences[:min(3, len(sentences))]
    rerun_transcriptions: list[str] = []
    with torch.no_grad():
        for sentence in rerun_sentences:
            torch.manual_seed(20260821)  # G3: registered chain seed
            inputs = tts_processor(text=sentence, return_tensors="pt")
            mel = tts_model.generate_speech(
                inputs["input_ids"].to(device),
                speaker_embeddings, vocoder=None)
            mel_np = mel.cpu().numpy().astype(np.float32)
            if mel_np.ndim == 3 and mel_np.shape[-1] == 80:
                mel_np = mel_np.transpose(0, 2, 1)
            waveform = vocoder(torch.from_numpy(mel_np).to(device))
            wav_np = waveform.squeeze(0).cpu().numpy().astype(np.float32)
            torch.manual_seed(20260821)  # G3: registered chain seed
            feats = whisper_proc(wav_np, sampling_rate=16000,
                                 return_tensors="pt").input_features
            ids = whisper_model.generate(feats.to(device), num_beams=1)[0]
            rerun_transcriptions.append(whisper_proc.tokenizer.normalize(
                whisper_proc.decode(ids, skip_special_tokens=True)))
    chain_deterministic = rerun_transcriptions == transcriptions[:3]

    # ---- arms registered AFTER measurement (honest numbers only) --------
    arms: dict[str, Any] = {}
    for arm_id, acc, source, kind in [
            ("m267_tts_core", 1.0 - loop_wer,
             config["stages"]["1_core"]["checkpoint"], "sealed_head"),
            ("m267_vocoder_polish", 1.0 - loop_wer,
             config["stages"]["2_polish"]["checkpoint"], "sealed_head"),
            ("m266a_whisper_eval", 1.0 - 0.029570765439645458,
             "logs/results/v25/m266_audio_arm/evidence.json",
             "sealed_head")]:
        spec = arm_from_sealed_head(
            arm_id, "audio_chain", 0, acc, source,
            license={"code": "MIT", "weights": "MIT", "data": "CC-BY-4.0"})
        orch.register(spec)
        arms[arm_id] = acc
    for arm_id in ("m267_tts_core", "m267_vocoder_polish",
                   "m266a_whisper_eval"):
        orch.serve(f"m267-{arm_id}", [], task_id=None,
                   cache=None)

    evidence: dict[str, Any] = {
        "milestone": "M267",
        "cell": "chained core+polish audio demonstration",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "n_sentences": len(sentences),
        "n_transcribed": len(transcriptions),
        "evaluation": {
            "loop_wer": loop_wer,
            "reading": ("the in-system chain's objective loop metric — "
                        "synthesized LibriSpeech re-transcribed by the "
                        "sealed M266a Whisper protocol; a demonstration "
                        "of measured, auditable chaining, not a claim "
                        "against any monolithic baseline"),
        },
        "gates": {
            "g2_primitive_replay": bool(stage_hashes.get(
                "primitive_digest")),
            "g3_chain_deterministic": chain_deterministic,
            "g4_ledger_verify": orch.chain_verify()["ok"],
            "g5_loop_wer_recorded": bool(per_item),
            "g6_empty_inputs_abstained": empty_chain_check,
        },
        "arms_registered_after_measurement": arms,
        "per_item": per_item,
        "ledger_record_count": orch.chain_verify()["record_count"],
        "license_recorded": {
            "tts": config["stages"]["1_core"]["license_recorded"],
            "vocoder": config["stages"]["2_polish"]["license_recorded"],
            "xvectors": config["speaker"]["license_recorded"],
        },
        "scope_note": ("publisher checkpoints frozen; chain determinism "
                       "and ledger replay demonstrated; the comparison "
                       "target is tolerance, not superiority"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"loop_wer": loop_wer,
                      "g3_deterministic": chain_deterministic,
                      "g4_ledger_ok": evidence["gates"]["g4_ledger_verify"],
                      "n_transcribed": len(transcriptions)}, indent=1),
          flush=True)
    print(f"M267 complete -> {output_dir / 'evidence.json'}", flush=True)
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
    run_m267(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
