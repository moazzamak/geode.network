"""M267 stage 0 — the FFT/mel-spectrogram front-end as a PROGRAMMATIC
PRIMITIVE.

A programmatic primitive is deterministic, parameter-free (frozen
constants), bit-exact replayable, and license-free: it needs a
content hash and a determinism test, never a measured safety tag.

Prior art, cited not invented: the STFT and the mel filterbank
(Davis & Mermelstein 1980, IEEE TASSP 28(4)); the phase problem the
front-end introduces is the classical one the trained polisher
stages address (Griffin & Lim 1984, IEEE TASSP 32(2) is the
deterministic baseline). This module implements the deterministic
transform only.

Determinism contract (G2 of the M267 registration): for the same
(f32 waveform, sample rate), `mel_spectrogram` returns bit-identical
arrays across runs and machines with the same numpy float semantics,
and its payload hash replays exactly.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from geode.hashing import payload_hash

# Frozen constants (the primitive's only parameters)
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
SAMPLE_RATE = 16_000
F_MIN = 0.0
F_MAX = 8000.0
EPSILON = 1e-10  # log floor; never x + eps under sqrt/log (standing rule)


def _mel_filterbank(n_fft: int, n_mels: int, sample_rate: int,
                    f_min: float, f_max: float) -> np.ndarray:
    """Deterministic mel filterbank (Davis & Mermelstein 1980)."""
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(n_mels):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        for k in range(left, min(center, fb.shape[1])):
            fb[m, k] = (k - left) / max(center - left, 1)
        for k in range(center, min(right, fb.shape[1])):
            fb[m, k] = (right - k) / max(right - center, 1)
    return fb


_FILTERBANK = _mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE, F_MIN, F_MAX)


def mel_spectrogram(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE
                    ) -> np.ndarray:
    """Log-mel spectrogram of a mono float32 waveform (16 kHz).

    Deterministic: hann window -> rfft -> |.|^2 -> mel filterbank ->
    log. No learned weights, no RNG. Returns (n_mels, n_frames)
    float32."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"the primitive is registered at "
                         f"{SAMPLE_RATE} Hz (got {sample_rate})")
    wave = np.asarray(waveform, dtype=np.float32)
    if wave.ndim != 1:
        raise ValueError("waveform must be mono 1-D")
    window = np.hanning(N_FFT).astype(np.float32)
    n_frames = 1 + max(0, (len(wave) - N_FFT)) // HOP_LENGTH
    specs = np.zeros((n_frames, N_FFT // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        start = i * HOP_LENGTH
        frame = wave[start:start + N_FFT] * window
        spec = np.fft.rfft(frame.astype(np.float64), n=N_FFT)
        specs[i] = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    mel = specs @ _FILTERBANK.T.astype(np.float32)
    return np.log(np.maximum(mel, EPSILON)).astype(np.float32)


def primitive_payload(waveform: np.ndarray,
                      sample_rate: int = SAMPLE_RATE) -> dict[str, Any]:
    """The payload whose hash is the primitive's replay anchor
    (timing excluded — the standing rule)."""
    return {"sample_rate": int(sample_rate),
            "waveform_hash": hashlib.sha256(
                np.asarray(waveform, dtype=np.float32)
                .tobytes()).hexdigest(),
            "constants": {"n_fft": N_FFT, "hop": HOP_LENGTH,
                          "n_mels": N_MELS, "sr": SAMPLE_RATE,
                          "f_min": F_MIN, "f_max": F_MAX}}


def primitive_replay_hash(waveform: np.ndarray,
                          sample_rate: int = SAMPLE_RATE) -> str:
    """G2: the payload hash the stage-0 output replays from."""
    return payload_hash(primitive_payload(waveform, sample_rate))
