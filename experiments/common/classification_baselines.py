from dataclasses import dataclass
import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


@dataclass
class ClassAlignedEstimator:
    estimator: object
    classes_: np.ndarray

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.estimator.predict_proba(X)
        columns = [
            int(np.flatnonzero(self.estimator.classes_ == class_id)[0])
            for class_id in self.classes_
        ]
        return probabilities[:, columns]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class NearestCentroidClassifier:
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == class_id].mean(axis=0) for class_id in self.classes_])
        distances = np.sum((X[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2)
        self.temperature_ = max(float(np.median(np.min(distances, axis=1))), 1e-6)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distances = np.sum((X[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2)
        return _softmax(-distances / self.temperature_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


@dataclass
class WeightedKNNClassifier:
    """DINO-style weighted kNN probe over L2-normalized frozen features."""

    n_neighbors: int = 20
    temperature: float = 0.07
    query_batch_size: int = 1024

    def __post_init__(self) -> None:
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive.")
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive.")
        if self.query_batch_size < 1:
            raise ValueError("query_batch_size must be positive.")
        self.classes_: np.ndarray | None = None
        self._features: np.ndarray | None = None
        self._label_indices: np.ndarray | None = None

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("features must be a two-dimensional array.")
        if not np.all(np.isfinite(values)):
            raise ValueError("features must contain only finite values.")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0.0)

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> "WeightedKNNClassifier":
        labels = np.asarray(y_train)
        if labels.ndim != 1 or len(labels) != len(X_train):
            raise ValueError("y_train must be one-dimensional and match X_train.")
        if len(labels) < self.n_neighbors:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} exceeds training size {len(labels)}."
            )
        self.classes_, self._label_indices = np.unique(labels, return_inverse=True)
        self._features = self._normalize(X_train)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if (
            self.classes_ is None
            or self._features is None
            or self._label_indices is None
        ):
            raise ValueError("WeightedKNNClassifier must be fitted before prediction.")
        queries = self._normalize(features)
        if queries.shape[1] != self._features.shape[1]:
            raise ValueError(
                f"Feature dimension {queries.shape[1]} does not match fitted "
                f"dimension {self._features.shape[1]}."
            )

        probabilities = np.zeros(
            (len(queries), len(self.classes_)), dtype=np.float64
        )
        for start in range(0, len(queries), self.query_batch_size):
            stop = min(start + self.query_batch_size, len(queries))
            similarities = queries[start:stop] @ self._features.T
            neighbor_indices = np.argsort(
                -similarities, axis=1, kind="stable"
            )[:, : self.n_neighbors]
            neighbor_similarities = np.take_along_axis(
                similarities, neighbor_indices, axis=1
            )
            shifted = neighbor_similarities - neighbor_similarities.max(
                axis=1, keepdims=True
            )
            weights = np.exp(shifted / self.temperature)
            neighbor_labels = self._label_indices[neighbor_indices]
            rows = np.repeat(np.arange(start, stop), self.n_neighbors)
            np.add.at(
                probabilities,
                (rows, neighbor_labels.reshape(-1)),
                weights.reshape(-1),
            )
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("WeightedKNNClassifier must be fitted before prediction.")
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]


class ShrinkageGaussianClassifier:
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.log_priors_ = np.log(counts / counts.sum())
        self.means_ = []
        self.precisions_ = []
        self.log_determinants_ = []
        for class_id in self.classes_:
            class_points = X[y == class_id]
            covariance = LedoitWolf().fit(class_points).covariance_
            sign, log_determinant = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise ValueError("Shrinkage covariance must be positive definite.")
            self.means_.append(class_points.mean(axis=0))
            self.precisions_.append(np.linalg.inv(covariance))
            self.log_determinants_.append(log_determinant)
        self.means_ = np.asarray(self.means_)
        self.precisions_ = np.asarray(self.precisions_)
        self.log_determinants_ = np.asarray(self.log_determinants_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        differences = X[:, None, :] - self.means_[None, :, :]
        mahalanobis = np.einsum(
            "ncd,cde,nce->nc", differences, self.precisions_, differences,
        )
        logits = self.log_priors_ - 0.5 * (mahalanobis + self.log_determinants_)
        return _softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class ClassConditionalGMMClassifier:
    def __init__(self, components_by_class: dict[int, int], seed: int = 42):
        self.components_by_class = components_by_class
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.log_priors_ = np.log(counts / counts.sum())
        self.models_ = []
        for class_position, class_id in enumerate(self.classes_):
            class_points = X[y == class_id]
            components = max(1, min(
                int(self.components_by_class.get(int(class_id), 1)),
                len(class_points),
            ))
            model = GaussianMixture(
                n_components=components,
                covariance_type="full",
                reg_covar=1e-6,
                random_state=self.seed + class_position,
            )
            model.fit(class_points)
            self.models_.append(model)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.column_stack([
            model.score_samples(X) + log_prior
            for model, log_prior in zip(self.models_, self.log_priors_)
        ])
        return _softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def fit_classification_baselines(
    X: np.ndarray,
    y: np.ndarray,
    components_by_class: dict[int, int],
    seed: int = 42,
    rbf_sample_limit: int = 10_000,
    include_names: set[str] | None = None,
) -> dict[str, object]:
    classes = np.unique(y)
    estimators = {
        "logistic_regression": LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", random_state=seed,
        ),
        "nearest_centroid": NearestCentroidClassifier(),
        "shrinkage_gaussian": ShrinkageGaussianClassifier(),
        "matched_gmm": ClassConditionalGMMClassifier(components_by_class, seed),
        "knn": KNeighborsClassifier(n_neighbors=min(5, len(X))),
        "linear_svm": CalibratedClassifierCV(
            LinearSVC(C=1.0, random_state=seed), method="sigmoid", cv=3,
        ),
        "histogram_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, random_state=seed,
        ),
    }
    if len(X) <= rbf_sample_limit:
        estimators["rbf_svm"] = CalibratedClassifierCV(
            SVC(C=1.0, kernel="rbf", random_state=seed),
            method="sigmoid",
            cv=3,
            ensemble=False,
        )
    if include_names is not None:
        unknown = include_names - estimators.keys()
        if unknown:
            raise ValueError(f"Unknown baseline names: {sorted(unknown)}")
        estimators = {
            name: estimator for name, estimator in estimators.items()
            if name in include_names
        }

    fitted = {}
    for name, estimator in estimators.items():
        fit_started = time.perf_counter()
        estimator.fit(X, y)
        fit_seconds = time.perf_counter() - fit_started
        if isinstance(
            estimator,
            (NearestCentroidClassifier, ShrinkageGaussianClassifier, ClassConditionalGMMClassifier),
        ):
            fitted[name] = estimator
        else:
            fitted[name] = ClassAlignedEstimator(estimator, classes)
        fitted[name].fit_seconds_ = float(fit_seconds)
    return fitted