from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


@dataclass
class ScoreReadout:
    mode: str
    classes: np.ndarray
    temperature: float = 1.0
    diagonal_slopes: np.ndarray | None = None
    diagonal_intercepts: np.ndarray | None = None
    classifier: LogisticRegression | None = None
    classifier_mean: np.ndarray | None = None
    classifier_scale: np.ndarray | None = None
    converged: bool = True
    fit_iterations: int = 0
    iteration_limit: int | None = None
    fit_warnings: tuple[str, ...] = ()

    def _classifier_probabilities(self, inputs: np.ndarray) -> np.ndarray:
        standardized = (
            (inputs - self.classifier_mean) / self.classifier_scale
            if self.classifier_mean is not None else inputs
        )
        probabilities = self.classifier.predict_proba(standardized)
        columns = [
            int(np.flatnonzero(self.classifier.classes_ == class_id)[0])
            for class_id in self.classes
        ]
        return probabilities[:, columns]

    def predict_proba(
        self,
        scores: np.ndarray,
        features: np.ndarray | None = None,
    ) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if self.mode in {"raw", "temperature"}:
            return _softmax(-scores / self.temperature)
        if self.mode == "diagonal":
            logits = scores * self.diagonal_slopes + self.diagonal_intercepts
            return _softmax(logits)
        if self.mode == "multinomial":
            return self._classifier_probabilities(scores)
        if self.mode == "feature_logistic":
            if features is None:
                raise ValueError("feature_logistic requires transformed features.")
            return self._classifier_probabilities(features)
        raise ValueError(f"Unknown readout mode: {self.mode}")

    def predict(
        self,
        scores: np.ndarray,
        features: np.ndarray | None = None,
    ) -> np.ndarray:
        probabilities = self.predict_proba(scores, features)
        return self.classes[probabilities.argmax(axis=1)]


def _standardize_calibration_inputs(
    inputs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(inputs, axis=0)
    scale = np.std(inputs, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return (inputs - mean) / scale, mean, scale


def _fit_logistic(
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    max_iter: int,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray, int, tuple[str, ...]]:
    standardized, mean, scale = _standardize_calibration_inputs(inputs)
    classifier = LogisticRegression(
        C=1.0, max_iter=max_iter, solver="lbfgs", random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(standardized, labels)
    fit_iterations = int(np.max(classifier.n_iter_))
    messages = tuple(
        str(item.message) for item in caught
        if issubclass(item.category, ConvergenceWarning)
    )
    if fit_iterations >= max_iter and not messages:
        messages = (f"lbfgs reached iteration limit {max_iter}",)
    return classifier, mean, scale, fit_iterations, messages

def fit_score_readout(
    mode: str,
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    classes: np.ndarray,
    calibration_features: np.ndarray | None = None,
    seed: int = 42,
    logistic_max_iter: int = 1000,
) -> ScoreReadout:
    scores = np.asarray(calibration_scores, dtype=np.float64)
    labels = np.asarray(calibration_labels)
    classes = np.asarray(classes)
    if scores.ndim != 2 or len(scores) != len(labels):
        raise ValueError("calibration_scores must have shape (samples, inputs).")
    if mode in {"raw", "temperature", "diagonal"} and scores.shape[1] != len(classes):
        raise ValueError(
            f"{mode} calibration_scores must have shape (samples, classes)."
        )

    if mode == "raw":
        return ScoreReadout(mode=mode, classes=classes)

    if mode == "temperature":
        class_lookup = {label: index for index, label in enumerate(classes.tolist())}
        columns = np.array([class_lookup[label] for label in labels.tolist()])

        def objective(log_temperature: float) -> float:
            probabilities = _softmax(-scores / np.exp(log_temperature))
            selected = probabilities[np.arange(len(labels)), columns]
            return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))

        result = minimize_scalar(objective, bounds=(-6.0, 6.0), method="bounded")
        return ScoreReadout(
            mode=mode,
            classes=classes,
            temperature=float(np.exp(result.x)),
            converged=bool(result.success),
            fit_iterations=int(result.nfev),
        )

    if mode == "diagonal":
        slopes = np.empty(len(classes), dtype=np.float64)
        intercepts = np.empty(len(classes), dtype=np.float64)
        fit_iterations = 0
        fit_warnings = []
        for column, class_id in enumerate(classes):
            binary_labels = (labels == class_id).astype(np.int32)
            classifier, mean, scale, iterations, messages = _fit_logistic(
                scores[:, column:column + 1],
                binary_labels,
                seed=seed,
                max_iter=logistic_max_iter,
            )
            slopes[column] = classifier.coef_[0, 0] / scale[0]
            intercepts[column] = (
                classifier.intercept_[0] - classifier.coef_[0, 0] * mean[0] / scale[0]
            )
            fit_iterations = max(fit_iterations, iterations)
            fit_warnings.extend(f"class {class_id}: {message}" for message in messages)
        return ScoreReadout(
            mode=mode,
            classes=classes,
            diagonal_slopes=slopes,
            diagonal_intercepts=intercepts,
            converged=not fit_warnings,
            fit_iterations=fit_iterations,
            iteration_limit=logistic_max_iter,
            fit_warnings=tuple(fit_warnings),
        )

    if mode == "multinomial":
        classifier, mean, scale, iterations, messages = _fit_logistic(
            scores, labels, seed=seed, max_iter=logistic_max_iter,
        )
        return ScoreReadout(
            mode=mode,
            classes=classes,
            classifier=classifier,
            classifier_mean=mean,
            classifier_scale=scale,
            converged=not messages,
            fit_iterations=iterations,
            iteration_limit=logistic_max_iter,
            fit_warnings=messages,
        )

    if mode == "feature_logistic":
        if calibration_features is None:
            raise ValueError("feature_logistic requires calibration_features.")
        classifier, mean, scale, iterations, messages = _fit_logistic(
            calibration_features,
            labels,
            seed=seed,
            max_iter=logistic_max_iter,
        )
        return ScoreReadout(
            mode=mode,
            classes=classes,
            classifier=classifier,
            classifier_mean=mean,
            classifier_scale=scale,
            converged=not messages,
            fit_iterations=iterations,
            iteration_limit=logistic_max_iter,
            fit_warnings=messages,
        )

    raise ValueError(f"Unknown readout mode: {mode}")


def fit_all_readouts(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    classes: np.ndarray,
    calibration_features: np.ndarray,
    seed: int = 42,
    logistic_max_iter: int = 1000,
) -> dict[str, ScoreReadout]:
    return {
        mode: fit_score_readout(
            mode=mode,
            calibration_scores=calibration_scores,
            calibration_labels=calibration_labels,
            classes=classes,
            calibration_features=calibration_features,
            seed=seed,
            logistic_max_iter=logistic_max_iter,
        )
        for mode in (
            "raw", "temperature", "diagonal", "multinomial", "feature_logistic",
        )
    }