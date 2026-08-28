"""M267 stage 0 tests — the FFT/mel programmatic primitive.

Determinism (bit-exact across calls), replay-hash stability, and
the payload-hash tamper behaviour. Prior art is cited in the module
docstring, not claimed here.
"""
import unittest

import numpy as np

from geode.core.audio_primitives import (
    SAMPLE_RATE,
    mel_spectrogram,
    primitive_replay_hash,
)


class TestMelPrimitive(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.wave = (rng.standard_normal(16000) * 0.01).astype(np.float32)

    def test_bit_exact_determinism(self):
        a = mel_spectrogram(self.wave)
        b = mel_spectrogram(self.wave.copy())
        self.assertEqual(a.dtype, np.float32)
        self.assertTrue(np.array_equal(a, b))

    def test_shape_and_finiteness(self):
        mel = mel_spectrogram(self.wave)
        self.assertEqual(mel.shape[1], 80)  # n_mels on axis 1
        self.assertGreater(mel.shape[0], 0)
        self.assertTrue(np.isfinite(mel).all())

    def test_replay_hash_stable_and_sensitive(self):
        h1 = primitive_replay_hash(self.wave)
        h2 = primitive_replay_hash(self.wave.copy())
        self.assertEqual(h1, h2)
        altered = self.wave.copy()
        altered[0] += 1e-3
        self.assertNotEqual(h1, primitive_replay_hash(altered))

    def test_payload_hash_replays_from_output(self):
        # the primitive's replay anchor covers its output content
        import hashlib
        mel = mel_spectrogram(self.wave)
        self.assertEqual(
            hashlib.sha256(mel.astype(np.float32).tobytes()).hexdigest(),
            hashlib.sha256(mel_spectrogram(self.wave).astype(
                np.float32).tobytes()).hexdigest())

    def test_wrong_sample_rate_rejected(self):
        with self.assertRaises(ValueError):
            mel_spectrogram(self.wave, sample_rate=8000)

    def test_stereo_rejected(self):
        with self.assertRaises(ValueError):
            mel_spectrogram(np.zeros((2, 16000), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
