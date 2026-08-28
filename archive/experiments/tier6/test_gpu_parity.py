import unittest

import numpy as np

from experiments.tier4.eval_complex_classification import (
    compute_score_scales,
    invalidate_gpu_engine_cache,
)
from src.gpu_engine import GPUInferenceEngine, batch_sdf_and_score, select_device
from src.inference_engine import InferenceEngine
from src.sdf_engine import EllipsoidExpert, Expert


def _expert(*ellipsoids: EllipsoidExpert, alpha: float = 2.0) -> Expert:
    expert = Expert(alpha=alpha)
    for ellipsoid in ellipsoids:
        expert.add_ellipsoid(ellipsoid)
    return expert


class GPUParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.device = select_device()
        except Exception as error:
            raise unittest.SkipTest(f"OpenCL GPU unavailable: {error}") from error

    def tearDown(self):
        invalidate_gpu_engine_cache()

    def test_inference_matches_cpu_and_handles_empty_class(self):
        models = [
            [_expert(EllipsoidExpert([0.0, 0.0], [1.0, 0.5]))],
            [_expert(
                EllipsoidExpert([2.0, 0.0], [0.8, 1.2]),
                EllipsoidExpert([2.5, 0.4], [0.5, 0.6]),
            )],
            [_expert(
                EllipsoidExpert([-2.0, 0.0], [1.3, 1.0]),
                EllipsoidExpert([-2.0, 0.0], [0.3, 0.2], polarity=-1),
            )],
            [],
        ]
        points = np.random.default_rng(9).normal(size=(128, 2)) * 3.0
        cpu = np.column_stack([
            InferenceEngine(model, alpha=2.0).get_fused_sdf(points)
            for model in models
        ])
        gpu = GPUInferenceEngine(models, alpha=2.0, device=self.device).class_sdfs(points)
        np.testing.assert_allclose(gpu[:, :3], cpu[:, :3], atol=1e-5)
        self.assertTrue(np.isinf(gpu[:, 3]).all())
        np.testing.assert_array_equal(gpu.argmin(axis=1), cpu.argmin(axis=1))

    def test_inference_matches_cpu_when_global_pruning_is_active(self):
        model = [
            _expert(EllipsoidExpert([center, 0.0], [1.0, 1.0]))
            for center in [0.0, 100.0, 200.0, 300.0]
        ]
        points = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32)

        cpu = InferenceEngine(model, alpha=2.0).get_fused_sdf(points)
        gpu = GPUInferenceEngine(
            [model], alpha=2.0, device=self.device,
        ).class_sdfs(points)[:, 0]

        np.testing.assert_allclose(gpu, cpu, atol=1e-5)

    def test_ransac_candidate_counts_match_multi_ellipsoid_cpu(self):
        expert = _expert(
            EllipsoidExpert([-0.8, 0.0], [0.7, 0.5]),
            EllipsoidExpert([0.8, 0.0], [0.7, 0.5]),
        )
        candidates = [
            EllipsoidExpert([0.0, 0.7], [0.6, 0.4]),
            EllipsoidExpert([0.0, 1.5], [0.5, 0.3]),
        ]
        points = np.random.default_rng(12).normal(size=(400, 2)).astype(np.float32)
        existing_sdf = expert.compute_sdf(points)
        gpu_positive, gpu_negative = batch_sdf_and_score(
            candidates,
            points,
            N_pool=300,
            ex_sdf_np=existing_sdf,
            alpha=2.0,
            threshold=0.1,
            task_regression=False,
            existing_count=2,
        )

        cpu_counts = []
        for candidate in candidates:
            expert.add_ellipsoid(candidate)
            captured = expert.compute_sdf(points) < 0.1
            cpu_counts.append((int(captured[:300].sum()), int(captured[300:].sum())))
            expert.ellipsoids.pop()

        self.assertEqual(
            cpu_counts,
            list(zip(gpu_positive.tolist(), gpu_negative.tolist())),
        )

    def test_class_scales_and_cache_invalidation_match_cpu(self):
        models = {
            0: [_expert(EllipsoidExpert([0.0, 0.0], [0.8, 0.7]))],
            1: [_expert(EllipsoidExpert([2.0, 0.0], [0.8, 0.9]))],
        }
        labels = np.array([0] * 100 + [1] * 100)
        points = np.vstack([
            np.random.default_rng(1).normal([0.0, 0.0], 0.4, size=(100, 2)),
            np.random.default_rng(2).normal([2.0, 0.0], 0.4, size=(100, 2)),
        ])
        compute_score_scales(models, points, 2.0, class_labels=labels, use_gpu=True)
        models[0][0].ellipsoids[0].center += np.array([0.4, 0.0])
        invalidate_gpu_engine_cache(models)

        cpu = compute_score_scales(
            models, points, 2.0, class_labels=labels, use_gpu=False,
        )
        gpu = compute_score_scales(
            models, points, 2.0, class_labels=labels, use_gpu=True,
        )
        for class_id in models:
            self.assertAlmostEqual(cpu[class_id], gpu[class_id], places=5)


if __name__ == "__main__":
    unittest.main()