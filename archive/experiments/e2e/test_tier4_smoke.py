import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.e2e.run_tier4_smoke import (
    InjectedStageFailure,
    main,
    run_resumable_tier4_smoke,
)
from src.sdf_engine import EllipsoidExpert, Expert
from src.runtime import LocalArtifactStore, LocalExecutor


class ResumableTier4SmokeTests(unittest.TestCase):
    @staticmethod
    def _dataset() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(31)
        class_zero = rng.normal(loc=(-2.0, 0.0, 0.0, 0.0), scale=0.35, size=(40, 4))
        class_one = rng.normal(loc=(2.0, 0.0, 0.0, 0.0), scale=0.35, size=(40, 4))
        return (
            np.vstack([class_zero, class_one]),
            np.concatenate([np.zeros(40, dtype=np.int32), np.ones(40, dtype=np.int32)]),
        )

    @staticmethod
    def _fitter(calls: list[int]):
        def fit(points: np.ndarray, exclusions: np.ndarray, seed: int) -> list[Expert]:
            del exclusions
            calls.append(seed)
            expert = Expert(alpha=2.0)
            expert.add_ellipsoid(EllipsoidExpert(
                center=np.mean(points, axis=0),
                radii=np.maximum(np.std(points, axis=0), 0.1),
            ))
            return [expert]

        return fit

    def test_interrupted_class_fit_resumes_to_identical_assembly(self):
        features, labels = self._dataset()
        load_calls = []
        fit_calls = []

        def load_features() -> tuple[np.ndarray, np.ndarray]:
            load_calls.append(1)
            return features, labels

        with tempfile.TemporaryDirectory() as resumed_directory:
            with self.assertRaisesRegex(InjectedStageFailure, "class-0"):
                run_resumable_tier4_smoke(
                    runtime_root=resumed_directory,
                    attempt_id="attempt-1",
                    dataset_fingerprint="synthetic-tier4-v1",
                    feature_loader=load_features,
                    seed=7,
                    pca_components=3,
                    class_fitter=self._fitter(fit_calls),
                    fail_after="class-0",
                )
            resumed = run_resumable_tier4_smoke(
                runtime_root=resumed_directory,
                attempt_id="attempt-1",
                dataset_fingerprint="synthetic-tier4-v1",
                feature_loader=load_features,
                seed=7,
                pca_components=3,
                class_fitter=self._fitter(fit_calls),
            )
            transform_path = (
                Path(resumed_directory)
                / "runs"
                / resumed["run_id"]
                / "attempts"
                / "attempt-1"
                / "transform"
            )
            self.assertTrue((transform_path / "final_test_indices.npy").exists())
            self.assertFalse((transform_path / "final_test_features.npy").exists())
            self.assertFalse((transform_path / "final_test_labels.npy").exists())

        uninterrupted_fit_calls = []
        with tempfile.TemporaryDirectory() as uninterrupted_directory:
            uninterrupted = run_resumable_tier4_smoke(
                runtime_root=uninterrupted_directory,
                attempt_id="attempt-1",
                dataset_fingerprint="synthetic-tier4-v1",
                feature_loader=lambda: (features, labels),
                seed=7,
                pca_components=3,
                class_fitter=self._fitter(uninterrupted_fit_calls),
            )

        self.assertEqual(len(load_calls), 1)
        self.assertEqual(fit_calls, [7, 8])
        self.assertEqual(uninterrupted_fit_calls, [7, 8])
        self.assertEqual(resumed["summary"], uninterrupted["summary"])
        self.assertEqual(
            resumed["assembly_output_hashes"],
            uninterrupted["assembly_output_hashes"],
        )
        self.assertEqual(resumed["metric_count"], 1)
        self.assertTrue(resumed["runtime_status"]["complete"])

    def test_status_cli_does_not_extract_features(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.npz"
            dataset_path.write_bytes(b"fingerprint-only")
            output = io.StringIO()
            with patch.object(sys, "argv", [
                "run_tier4_smoke",
                "--runtime-root", directory,
                "--dataset-path", str(dataset_path),
                "--status",
            ]), patch("sys.stdout", output):
                main()

        status = json.loads(output.getvalue())
        self.assertFalse(status["complete"])
        self.assertEqual(
            [stage["status"] for stage in status["stages"]],
            ["pending", "pending", "pending"],
        )

    def test_partial_feature_write_resumes_to_identical_assembly(self):
        features, labels = self._dataset()
        load_calls = []

        def load_features() -> tuple[np.ndarray, np.ndarray]:
            load_calls.append(1)
            return features, labels

        kwargs = {
            "attempt_id": "attempt-1",
            "dataset_fingerprint": "synthetic-tier4-feature-failure-v1",
            "feature_loader": load_features,
            "seed": 7,
            "pca_components": 3,
            "class_fitter": self._fitter([]),
        }
        with tempfile.TemporaryDirectory() as resumed_directory:
            with self.assertRaisesRegex(InjectedStageFailure, "during features"):
                run_resumable_tier4_smoke(
                    runtime_root=resumed_directory,
                    fail_during="features",
                    **kwargs,
                )
            run_id = next(
                (Path(resumed_directory) / "runs").iterdir()
            ).name
            interrupted = LocalExecutor(
                LocalArtifactStore(resumed_directory)
            ).status(run_id, "attempt-1", ("features",))
            self.assertEqual(
                interrupted.stages[0].status.value,
                "partial",
            )
            resumed = run_resumable_tier4_smoke(
                runtime_root=resumed_directory,
                **kwargs,
            )

        with tempfile.TemporaryDirectory() as uninterrupted_directory:
            uninterrupted = run_resumable_tier4_smoke(
                runtime_root=uninterrupted_directory,
                **kwargs,
            )

        self.assertEqual(len(load_calls), 3)
        self.assertEqual(
            resumed["assembly_output_hashes"],
            uninterrupted["assembly_output_hashes"],
        )

    def test_partial_class_fit_resumes_without_reloading_features(self):
        features, labels = self._dataset()
        load_calls = []
        fit_calls = []

        def load_features() -> tuple[np.ndarray, np.ndarray]:
            load_calls.append(1)
            return features, labels

        kwargs = {
            "attempt_id": "attempt-1",
            "dataset_fingerprint": "synthetic-tier4-class-failure-v1",
            "feature_loader": load_features,
            "seed": 7,
            "pca_components": 3,
            "class_fitter": self._fitter(fit_calls),
        }
        with tempfile.TemporaryDirectory() as resumed_directory:
            with self.assertRaisesRegex(InjectedStageFailure, "during class-0"):
                run_resumable_tier4_smoke(
                    runtime_root=resumed_directory,
                    fail_during="class-0",
                    **kwargs,
                )
            run_id = next(
                (Path(resumed_directory) / "runs").iterdir()
            ).name
            interrupted = LocalExecutor(
                LocalArtifactStore(resumed_directory)
            ).status(run_id, "attempt-1", ("features", "transform", "class-0"))
            self.assertEqual(
                [stage.status.value for stage in interrupted.stages],
                ["committed", "committed", "partial"],
            )
            resumed = run_resumable_tier4_smoke(
                runtime_root=resumed_directory,
                **kwargs,
            )

        uninterrupted_fit_calls = []
        with tempfile.TemporaryDirectory() as uninterrupted_directory:
            uninterrupted = run_resumable_tier4_smoke(
                runtime_root=uninterrupted_directory,
                **{
                    **kwargs,
                    "class_fitter": self._fitter(uninterrupted_fit_calls),
                },
            )

        self.assertEqual(len(load_calls), 2)
        self.assertEqual(fit_calls, [7, 7, 8])
        self.assertEqual(uninterrupted_fit_calls, [7, 8])
        self.assertEqual(
            resumed["assembly_output_hashes"],
            uninterrupted["assembly_output_hashes"],
        )


if __name__ == "__main__":
    unittest.main()