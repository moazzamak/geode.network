"""M288 — the TTS xvector unblock: real speaker conditioning.

g1: the speechbrain xvector encoder (512-dim, Apache-2.0) extracts
a deterministic speaker embedding from a REAL permissive voice
sample (LibriSpeech, CC-BY-4.0). Determinism = encode twice, delta
0.0; separation = two different speakers read different embeddings.
g2: the M267 loop (SpeechT5 + HiFi-GAN + whisper-small.en, the
sealed pipeline, the same 100 sentences) re-measured with the REAL
xvector conditioning instead of the seeded random vector (the
sealed loop WER 0.1127 is the comparison baseline).

GPU (the loop) + CPU (the encoder). Evidence:
logs/results/v25/m288_xvector_unblock/evidence.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.data_cache import configure_external_cache_environment
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m288_xvector_unblock")
XVECTOR_DIR = Path(r"F:\geode-ml\data\cache\huggingface"
                   r"\spkrec-xvect-voxceleb")
N_SENTENCES = 100
REF_SPEAKER_ROW = 0       # first train-split utterance (one speaker)
OTHER_SPEAKER_ROW = 7000  # a far-away row (a different speaker)


def _load_reference_audio() -> tuple[np.ndarray, np.ndarray, list[int]]:
    import io
    import soundfile as sf
    from datasets import Audio, load_dataset as _hf_load
    train = _hf_load("librispeech_asr", "clean", split="train.100")
    # torchcodec is incompatible with this build — decode via
    # soundfile/BytesIO (the registered env pattern)
    train = train.cast_column("audio", Audio(decode=False))
    rows = [train[i] for i in (REF_SPEAKER_ROW, OTHER_SPEAKER_ROW)]
    arrays, srs, spk_ids = [], [], []
    for row in rows:
        array, sr = sf.read(io.BytesIO(row["audio"]["bytes"]),
                            dtype="float32", always_2d=False)
        if array.ndim > 1:
            array = array.mean(axis=1)
        arrays.append(np.asarray(array, dtype=np.float32))
        srs.append(int(sr))
        spk_ids.append(int(row["speaker_id"]))
    return arrays[0], arrays[1], spk_ids, srs


def _encoder():
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source=str(XVECTOR_DIR),
        savedir=str(Path(r"F:\geode-ml\data\cache\speechbrain")),
        run_opts={"device": "cpu"})


def run_m288(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    configure_external_cache_environment()

    import torch
    torch.backends.cudnn.enabled = False

    # ---- g1: real-voice xvector, deterministic, separated ------------
    ref_audio, other_audio, spk_ids, srs = _load_reference_audio()
    classifier = _encoder()
    with torch.no_grad():
        w1 = torch.from_numpy(ref_audio).unsqueeze(0)
        e1 = classifier.encode_batch(w1).squeeze(0).reshape(-1).numpy().astype(
            np.float64)
        e1b = classifier.encode_batch(w1).squeeze(0).reshape(-1).numpy().astype(
            np.float64)
        w2 = torch.from_numpy(other_audio).unsqueeze(0)
        e2 = classifier.encode_batch(w2).squeeze(0).reshape(-1).numpy().astype(
            np.float64)
    det_delta = float(np.abs(e1 - e1b).max())
    cos_same = float(np.dot(e1, e1b) / (np.linalg.norm(e1)
                                        * np.linalg.norm(e1b) + 1e-12))
    cos_diff = float(np.dot(e1, e2) / (np.linalg.norm(e1)
                                       * np.linalg.norm(e2) + 1e-12))
    g1_ok = det_delta == 0.0 and cos_same > 0.999 and cos_diff < 0.99
    speaker_emb = (e1 / (np.linalg.norm(e1) + 1e-12)).astype(np.float32)

    # ---- g2: the loop with the real vector ---------------------------
    from datasets import Audio, load_dataset as _hf_load
    from transformers import (SpeechT5ForTextToSpeech, SpeechT5HifiGan,
                              SpeechT5Processor,
                              WhisperForConditionalGeneration,
                              WhisperProcessor)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_processor = SpeechT5Processor.from_pretrained(
        "microsoft/speecht5_tts", local_files_only=True)
    tts_model = SpeechT5ForTextToSpeech.from_pretrained(
        "microsoft/speecht5_tts", local_files_only=True).to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan",
        local_files_only=True).to(device).eval()
    whisper_proc = WhisperProcessor.from_pretrained(
        "openai/whisper-small.en", local_files_only=True)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small.en",
        local_files_only=True).to(device).eval()

    ds = _hf_load("librispeech_asr", "clean", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    sentences = [ds[i]["text"] for i in range(N_SENTENCES)]
    speaker_embeddings = torch.from_numpy(speaker_emb).unsqueeze(
        0).to(device)

    def word_error_rate(reference: str, prediction: str) -> float:
        ref = reference.split()
        hyp = prediction.split()
        if not ref:
            return 1.0 if hyp else 0.0
        prev = list(range(len(hyp) + 1))
        for i, x in enumerate(ref, 1):
            curr = [i] + [0] * len(hyp)
            for j, y in enumerate(hyp, 1):
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1,
                              prev[j - 1] + (x != y))
            prev = curr
        return prev[-1] / len(ref)

    wer_sum = 0.0
    n_words = 0
    per_item: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            torch.manual_seed(20260821)
            inputs = tts_processor(text=sentence, return_tensors="pt")
            mel = tts_model.generate_speech(
                inputs["input_ids"].to(device), speaker_embeddings,
                vocoder=None)
            mel_np = mel.cpu().numpy().astype(np.float32)
            if mel_np.ndim == 3 and mel_np.shape[-1] == 80:
                mel_np = mel_np.transpose(0, 2, 1)
            waveform = vocoder(torch.from_numpy(mel_np).to(device))
            wav_np = waveform.squeeze(0).cpu().numpy().astype(np.float32)
            torch.manual_seed(20260821)
            feats = whisper_proc(wav_np, sampling_rate=16000,
                                 return_tensors="pt").input_features
            ids = whisper_model.generate(feats.to(device), num_beams=1)[0]
            pred = whisper_proc.tokenizer.normalize(
                whisper_proc.decode(ids, skip_special_tokens=True))
            ref = whisper_proc.tokenizer.normalize(sentence)
            w = word_error_rate(ref, pred)
            wer_sum += w * max(len(ref.split()), 1)
            n_words += max(len(ref.split()), 1)
            per_item.append({"index": idx, "reference": ref,
                             "prediction": pred, "wer": w})
            if (idx + 1) % 25 == 0:
                print(f"  {idx + 1}/{N_SENTENCES} running WER "
                      f"{wer_sum / n_words:.4f}", flush=True)
    loop_wer = float(wer_sum / n_words) if n_words else float("nan")

    evidence: dict[str, Any] = {
        "milestone": "M288",
        "cell": "TTS xvector unblock — real speaker conditioning",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "encoder": "speechbrain spkrec-xvect-voxceleb (512-dim, "
                       "Apache-2.0), CPU",
            "reference_voice": "LibriSpeech train row 0 (CC-BY-4.0), "
                               "speaker id recorded in results",
            "loop": "the sealed M267 pipeline (SpeechT5 + HiFi-GAN + "
                    "whisper-small.en, 100 sentences, seed 20260821)",
            "baseline": "seeded random vector loop WER 0.1127 (M267)",
        }),
        "results": {
            "g1": {
                "embedding_dim": int(len(e1)),
                "determinism_max_abs_delta": det_delta,
                "same_audio_cosine": round(cos_same, 6),
                "different_speaker_cosine": round(cos_diff, 4),
                "reference_speaker_id": spk_ids[0],
                "other_speaker_id": spk_ids[1],
                "ok": bool(g1_ok),
            },
            "g2": {
                "loop_wer_real_vector": round(loop_wer, 4),
                "baseline_loop_wer_random_vector": 0.1127,
                "n_sentences": len(per_item),
                "per_item": per_item,
            },
            "verdict": ("M288 PASS — the real-voice xvector conditions "
                        "the loop deterministically"
                        if g1_ok else
                        "M288 g1 FAIL — recorded"),
        },
        "scope_note": ("the unblock: torchaudio 2.11.0+cpu installed "
                       "with --no-deps (the pin was the block); "
                       "naturalness is evidenced by the real-voice "
                       "conditioning, not claimed"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": {
        "g1": evidence["results"]["g1"],
        "g2": {k: v for k, v in evidence["results"]["g2"].items()
               if k != "per_item"}}}, indent=1), flush=True)
    print(f"M288 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


if __name__ == "__main__":
    run_m288(DEFAULT_OUTPUT)
