import numpy as np
from collections.abc import Callable
from typing import List
try:
    from src.sdf_engine import EllipsoidExpert, SoftminFusion, Expert
    from src.nudge_engine import _nudge_ellipsoid_from_points
except ModuleNotFoundError:
    from sdf_engine import EllipsoidExpert, SoftminFusion, Expert
    from nudge_engine import _nudge_ellipsoid_from_points


def _ransac_budget(
    inlier_frac: float,
    min_seed: int,
    confidence: float = 0.99,
    cap: int = 10_000,
) -> int:
    """RANSAC trial count for *confidence* given inlier fraction and minimal seed size.

    Uses the classic formula: N = ⌈log(1−confidence) / log(1−inlier_frac^min_seed)⌉
    Clamped to [50, cap] to guard against degenerate fractions.

    :param inlier_frac: Estimated fraction of pool points that are inliers.
    :param min_seed: Family-specific sample size for one hypothesis.
    :param confidence: Desired probability that at least one all-inlier sample is drawn.
    :param cap: Upper bound on returned trial count.
    :return: Number of RANSAC trials.
    """
    p_all_in = inlier_frac ** min_seed
    if p_all_in <= 0.0 or p_all_in >= 1.0:
        return cap if p_all_in <= 0.0 else 50
    log_fail = np.log1p(-p_all_in)  # log(1 - p_all_in), numerically stable for small p
    if log_fail == 0.0:
        return cap  # p_all_in below float64 precision — return cap
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = np.log(1.0 - confidence) / log_fail
    if not np.isfinite(ratio) or ratio >= cap:
        return cap  # denominator near zero → formula gives ∞ → use cap
    n = int(np.ceil(ratio))
    return max(50, min(n, cap))

class GreedyConstructor:
    """
    Implements the RANSAC-inspired greedy assembly process for GEODE.
    Each *expert* is a group of EllipsoidExpert primitives grown incrementally:
    - Inner loop: RANSAC finds the ellipsoid that most extends the expert's
      capture region.  When growth stalls the expert is locked.
    - Outer loop: locked experts are removed from the unexplained pool and
      the process repeats until consensus can no longer be met.
    """
    def __init__(
        self,
        consensus_threshold: float = 0.1,
        capture_threshold: float = 0.0,
        task_type: str = "classification",
        alpha: float = 1.0,
        max_iterations: int | None = None,
        min_growth_fraction: float = 0.01,
        use_gpu: bool = False,
        score_beta: float = 1.0,
        knn_seeding: bool = True,
        seed: int = 42,
        candidate_fitter: Callable[[np.ndarray, int], EllipsoidExpert] | None = None,
        candidate_seed_size: int | None = None,
        primitive_family: str | None = None,
        gpu_candidate_fitting: bool = False,
        metric_family: str | None = None,
        metric_rank: int = 4,
        metric_eigenvalue_floor: float = 1e-6,
    ):
        """
        :param consensus_threshold: Minimum fraction of the unexplained pool that
                                     an expert must capture to be locked in.
        :param capture_threshold: SDF threshold for a point to be "captured".
                                   Regression uses |SDF| < threshold; classification
                                   uses SDF < threshold.
        :param task_type: "classification" or "regression".
        :param alpha: Softmin concentration parameter.
        :param max_iterations: RANSAC iterations per primitive search. If None,
                               derive a bounded budget from the family-specific
                               seed size and estimated inlier fraction.
        :param min_growth_fraction: Minimum fraction of the pool that must be
                                     newly captured by each added ellipsoid.
                                     Growth below this signals stagnation.
        :param score_beta: β for F_β discriminative scoring when exclude_points are
                            supplied.  score = (1+β²)·p / ((1+β²)·p + β²·n + 1).
                            β=1 (default) balances precision and coverage equally;
                            β<1 penalises false captures more (higher precision);
                            β>1 rewards coverage more (higher recall).
        :param knn_seeding: When True (default) and d > 6, use kNN-anchor seeding
                             for ~half the RANSAC iterations: pick a random anchor,
                             find its k nearest neighbours in the seed pool, and use
                             those as the minimal seed.  This gives much higher inlier
                             purity per trial than purely random seeds, compensating
                             for the exponential collapse of the all-inlier probability
                             in high dimensions.
                    :param seed: Seed for all random sampling performed by this constructor.
        """
        if task_type not in {"classification", "regression"}:
            raise ValueError(
                f"Unsupported task_type '{task_type}'. Expected 'classification' or 'regression'."
            )
        self.consensus_threshold = consensus_threshold
        self.capture_threshold = capture_threshold
        self.task_type = task_type
        self.alpha = alpha
        self.max_iterations: int | None = max_iterations
        self.min_growth_fraction = min_growth_fraction
        self.use_gpu = use_gpu
        self.score_beta = float(score_beta)
        self.knn_seeding = knn_seeding
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.candidate_fitter = candidate_fitter
        self.candidate_seed_size = candidate_seed_size
        if primitive_family is None:
            primitive_family = (
                "sphere"
                if candidate_fitter is None and metric_family is None
                else "ellipsoid"
            )
        if primitive_family not in {"ellipsoid", "diagonal_ellipsoid", "sphere"}:
            raise ValueError("primitive_family must be ellipsoid, diagonal_ellipsoid, or sphere")
        if metric_family is not None:
            from src.metric_parameterization import LOCAL_FAMILIES

            if metric_family not in LOCAL_FAMILIES:
                raise ValueError(
                    f"metric_family must be one of {LOCAL_FAMILIES}, got {metric_family!r}."
                )
            if candidate_fitter is not None:
                raise ValueError("metric_family and candidate_fitter are mutually exclusive.")
        if isinstance(metric_rank, bool) or not isinstance(metric_rank, int) or metric_rank < 0:
            raise ValueError("metric_rank must be a nonnegative integer.")
        if (
            not np.isfinite(metric_eigenvalue_floor)
            or metric_eigenvalue_floor <= 0.0
        ):
            raise ValueError("metric_eigenvalue_floor must be finite and positive.")
        self.primitive_family = primitive_family
        self.gpu_candidate_fitting = bool(gpu_candidate_fitting)
        self.metric_family = metric_family
        self.metric_rank = metric_rank
        self.metric_eigenvalue_floor = float(metric_eigenvalue_floor)
        self.model: List[Expert] = []

    @property
    def _primitive_label(self) -> str:
        return self.primitive_family.replace("_", " ")

    @property
    def _primitive_plural(self) -> str:
        return "spheres" if self.primitive_family == "sphere" else f"{self._primitive_label}s"

    def _uses_gpu_candidate_path(self) -> bool:
        return self.metric_family is None and self.use_gpu and (
            self.candidate_fitter is None or self.gpu_candidate_fitting
        )

    def _minimal_seed_size(self, dimension: int) -> int:
        if self.candidate_seed_size is not None:
            if self.candidate_seed_size <= dimension:
                raise ValueError("candidate_seed_size must exceed the dimension.")
            return self.candidate_seed_size
        if self.primitive_family == "sphere":
            # One center vector and one radius, plus one point of
            # overdetermination for a stable direct spherical fit.
            return dimension + 2
        if self.candidate_fitter is not None:
            return 2 * dimension + 1
        if self.metric_family is not None:
            return 2 * dimension + 1
        return (dimension * (dimension + 3)) // 2

    def _generate_candidate(self, k_points: np.ndarray) -> EllipsoidExpert:
        """
        Model Generation: Fit a d-dimensional ellipsoid from k points.
        This serves as the 'Model Generation' step within the RANSAC loop.
        """
        points = np.asarray(k_points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError("k_points must be a 2D array of shape (k, d).")

        k, d = points.shape
        if self.candidate_fitter is not None:
            candidate = self.candidate_fitter(points, self.seed)
            self._project_primitive_family(candidate)
            return candidate
        if self.primitive_family == "sphere":
            if k < d + 2:
                raise ValueError(
                    f"At least {d + 2} points are required to fit a {d}D sphere, "
                    f"got {k}."
                )
            center = np.mean(points, axis=0)
            radius = float(
                np.sqrt(np.sum(np.var(points, axis=0, ddof=1)))
            )
            if not np.isfinite(radius) or radius <= 1e-12:
                raise ValueError("Degenerate sphere fit: radius is not positive.")
            return EllipsoidExpert(
                center,
                np.full(d, radius, dtype=np.float64),
                np.eye(d, dtype=np.float64),
            )
        if self.metric_family is not None:
            from src.metric_parameterization import fit_precision_metric

            fit = fit_precision_metric(
                points,
                self.metric_family,
                rank=self.metric_rank,
                eigenvalue_floor=self.metric_eigenvalue_floor,
            )
            candidate = EllipsoidExpert.from_precision_metric(
                fit.center,
                fit.metric,
                radius_scale=float(np.sqrt(d)),
            )
            self._project_primitive_family(candidate)
            return candidate
        required_k = (d * (d + 3)) // 2
        if k < required_k:
            raise ValueError(
                f"At least {required_k} points are required to fit a {d}D ellipsoid, got {k}."
            )

        # Fit the general quadratic form:
        # x^T Q x + q^T x + r = 0
        # using SVD on the design matrix.
        quad_features = []
        for i in range(d):
            for j in range(i, d):
                if i == j:
                    quad_features.append(points[:, i] * points[:, j])
                else:
                    quad_features.append(2.0 * points[:, i] * points[:, j])
        quadratic_block = np.stack(quad_features, axis=1)
        linear_block = points
        constant_block = np.ones((k, 1), dtype=np.float64)
        design_matrix = np.hstack([quadratic_block, linear_block, constant_block])

        _, _, vh = np.linalg.svd(design_matrix, full_matrices=True)
        coeffs = vh[-1]

        quad_param_count = d * (d + 1) // 2
        quad_params = coeffs[:quad_param_count]
        linear_params = coeffs[quad_param_count:quad_param_count + d]
        constant_param = coeffs[-1]

        q_matrix = np.zeros((d, d), dtype=np.float64)
        idx = 0
        for i in range(d):
            for j in range(i, d):
                q_matrix[i, j] = quad_params[idx]
                q_matrix[j, i] = quad_params[idx]
                idx += 1

        center = -0.5 * np.linalg.solve(q_matrix, linear_params)
        scale = center @ q_matrix @ center - constant_param
        if np.isclose(scale, 0.0):
            raise ValueError("Degenerate ellipsoid fit: scale is near zero.")

        shape_matrix = q_matrix / scale
        eigenvalues, eigenvectors = np.linalg.eigh(shape_matrix)
        if np.any(eigenvalues <= 1e-12):
            # The quadric is not an ellipsoid (hyperboloid or paraboloid).
            # This is common in high dimensions where random seeds rarely lie on
            # an ellipsoidal surface.  Fall back to a covariance-based ellipsoid:
            # orient along principal axes of the seed-point cloud and scale radii
            # by sqrt(d) so the seed points sit approximately on the surface
            # (mean Mahalanobis distance ≈ 1 → mean SDF ≈ 0).
            cov_center = np.mean(points, axis=0)
            cov = np.cov(points, rowvar=False)
            cov_eigenvalues, cov_eigenvectors = np.linalg.eigh(cov)
            if np.any(cov_eigenvalues <= 1e-12):
                raise ValueError("Degenerate covariance matrix; cannot fit ellipsoid.")
            # Sort descending by eigenvalue (largest axis first)
            order = np.argsort(cov_eigenvalues)[::-1]
            cov_eigenvalues = cov_eigenvalues[order]
            cov_eigenvectors = cov_eigenvectors[:, order]
            radii = np.sqrt(cov_eigenvalues) * np.sqrt(d)
            orientation = cov_eigenvectors
            return EllipsoidExpert(cov_center, radii, orientation)

        radii = 1.0 / np.sqrt(eigenvalues)
        orientation = eigenvectors

        return EllipsoidExpert(center, radii, orientation)

    def _project_primitive_family(self, candidate: EllipsoidExpert) -> None:
        if self.primitive_family == "sphere":
            radius = float(np.sqrt(np.mean(np.square(candidate.radii))))
            candidate.radii[:] = radius
            candidate.orientation[:] = np.eye(len(candidate.radii))
        elif self.primitive_family == "diagonal_ellipsoid":
            candidate.orientation[:] = np.eye(len(candidate.radii))

    def _disc_score(self, pos_count: int, neg_count: int) -> float:
        """F_β discriminative score: balances coverage (recall) and purity (precision).

        F_β = (1 + β²) · p / ((1 + β²) · p + β² · n + 1)

        - β=1: equally weights precision and recall (default).
        - β<1: emphasises precision — penalises false captures heavily.
        - β>1: emphasises recall — rewards covering more class points.

        The +1 denominator smoothing avoids division by zero and prevents
        zero-precision candidates from scoring above zero.
        """
        b2 = self.score_beta ** 2
        return (1.0 + b2) * pos_count / ((1.0 + b2) * pos_count + b2 * neg_count + 1.0)

    @staticmethod
    def _knn_seed(
        seed_pool: np.ndarray, k: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Sample a kNN-anchored seed: pick a random anchor, return its k nearest neighbours.

        For high-dimensional data the all-inlier probability ``wᵏ`` of a fully
        random seed collapses to near zero (``0.5⁵⁴ ≈ 6·10⁻¹⁷`` for d=9).
        Anchoring on a random point and selecting its nearest neighbours gives a
        locally dense sample that is much more likely to lie on the same cluster,
        yielding vastly higher inlier purity per trial.
        """
        rng = rng if rng is not None else np.random.default_rng()
        n = len(seed_pool)
        if n < k:
            raise ValueError("Seed pool must contain at least k points.")
        if n == k:
            return seed_pool.copy()
        anchor_idx = int(rng.integers(n))
        anchor = seed_pool[anchor_idx]
        dists = np.sum((seed_pool - anchor) ** 2, axis=1)
        nn_idx = np.argpartition(dists, k - 1)[:k]
        return seed_pool[nn_idx]

    def _batch_generate_candidates_gpu(
        self, seed_pool: np.ndarray, n_iter: int
    ) -> list:
        """Generate *n_iter* ellipsoid candidates via vectorised covariance fitting.

        Used exclusively in the GPU training path.  Replaces the sequential
        :meth:`_generate_candidate` loop with a single NumPy einsum + batched
        :func:`~numpy.linalg.eigh` call, which is ~40× faster than repeated SVD
        fits for CIFAR-10 dimensionalities (d=9, k=54, n_iter=270).

        Each candidate is the covariance ellipsoid of a random seed subset —
        the same fitting strategy used by :meth:`_generate_candidate` as its
        fallback when the quadratic-form SVD yields non-positive eigenvalues
        (which is common in high dimensions).  Plausibility pre-filtering is
        omitted: the GPU scoring step assigns zero captures to empty candidates,
        excluding them from selection via the discriminative score threshold.

        Sampling uses ``np.argpartition`` on a uniform random matrix to draw
        *k_size* indices without replacement for each trial in a single
        vectorised call — 10–20× faster than a Python list-comprehension of
        ``np.random.choice(..., replace=False)`` for large *n_iter*.  Covariance
        matrices are accumulated in float32 to halve memory traffic (318 MB →
        159 MB for d=19, n_iter=10000).

        :param seed_pool: ``(n, d)`` array to sample seeds from.
        :param n_iter: Number of candidate designs to attempt.
        :return: List of valid :class:`~src.sdf_engine.EllipsoidExpert` objects.
        """
        n, d   = seed_pool.shape
        k_size = self._minimal_seed_size(d)
        # argpartition requires kth < axis_size, so we need n > k_size (strictly).
        # When n == k_size there is only one subset (all points) and no randomness;
        # the covariance fitting will always yield the same degenerate result.
        if n < k_size:
            return []
        if n == k_size:
            try:
                candidate = self._generate_candidate(seed_pool)
                return [candidate] if self._is_plausible(candidate, seed_pool) else []
            except (np.linalg.LinAlgError, ValueError):
                return []

        # For small d (≤6), covariance fitting is too noisy with k_size points
        # (e.g. d=3 → k_size=9 surface points give inaccurate covariance ellipsoids,
        # causing near-zero captures and RANSAC failure).  SVD is fast at these
        # sizes (50 × SVD(9,10) ≈ 2 ms) and produces accurate ellipsoids.
        if d <= 6 and self.primitive_family == "ellipsoid":
            candidates: list = []
            for _ in range(n_iter):
                idx = self.rng.choice(n, k_size, replace=False)
                try:
                    cand = self._generate_candidate(seed_pool[idx])
                    if self._is_plausible(cand, seed_pool):
                        candidates.append(cand)
                except (np.linalg.LinAlgError, ValueError):
                    continue
            return candidates

        # Large d: vectorised covariance fitting.
        # --- Without-replacement sampling via argpartition on random floats ---
        # Faster than a list-comp of np.random.choice for large n_iter.
        rands   = self.rng.random((n_iter, n), dtype=np.float32)
        idx_arr = np.argpartition(rands, k_size, axis=1)[:, :k_size]  # (n_iter, k_size)

        seeds    = seed_pool[idx_arr]                              # (n_iter, k_size, d)
        if self.primitive_family in {"diagonal_ellipsoid", "sphere"}:
            from src.gpu_engine import fit_axis_aligned_candidates_gpu

            centers, radii = fit_axis_aligned_candidates_gpu(
                seeds, self.primitive_family,
            )
            valid = np.all(np.isfinite(radii) & (radii > 1e-6), axis=1)
            identity = np.eye(d)
            candidates = [
                EllipsoidExpert(centers[index], radii[index], identity)
                for index in np.flatnonzero(valid)
            ]
            return [
                candidate for candidate in candidates
                if self._is_plausible(candidate, seed_pool)
            ]

        centers  = seeds.mean(axis=1)                             # (n_iter, d)
        centered = (seeds - centers[:, np.newaxis, :]).astype(np.float32)  # float32 saves 50% memory

        # Batch covariance: one einsum replaces n_iter np.cov calls.
        # float32 accumulation halves peak memory (318 MB → 159 MB for d=19, n=10000).
        covs = np.einsum("bki,bkj->bij", centered, centered) / (k_size - 1)  # (n_iter, d, d)

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covs)     # (n_iter, d), (n_iter, d, d)
        except np.linalg.LinAlgError:
            return []

        valid = np.all(eigenvalues > 1e-12, axis=1)               # (n_iter,)
        scale = float(np.sqrt(d))

        candidates: list = []
        for ci in np.where(valid)[0]:
            radii = np.sqrt(eigenvalues[ci]) * scale
            candidate = EllipsoidExpert(
                centers[ci].astype(np.float32).copy(),
                radii.astype(np.float32),
                eigenvectors[ci].copy(),
            )
            if self._is_plausible(candidate, seed_pool):
                candidates.append(candidate)
        return candidates

    def build_model(self, points: np.ndarray, exclude_points: np.ndarray | None = None) -> List[Expert]:
        """
        Build the model by iteratively assembling Expert groups.

        Outer loop: while enough unexplained points remain, grow one Expert then
        lock it if it meets consensus.
        Inner loop: RANSAC grows the current Expert by adding the ellipsoid that
        most increases the expert's total captured count.  Stops when growth
        falls below *min_growth_fraction* of the current pool size.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError("points must be a 2D array of shape (N, d).")

        unexplained_pool  = points.copy()
        initial_pool_size = len(points)          # used to estimate inlier fraction
        d = unexplained_pool.shape[1]
        minimal_seed_size = self._minimal_seed_size(d)

        # Subsample exclude_points for efficiency in discriminative scoring.
        # 300 points gives a reliable precision estimate without dominating runtime.
        if exclude_points is not None and len(exclude_points) > 0:
            exc = np.asarray(exclude_points, dtype=np.float64)
            n_exc = min(len(exc), 300)
            exc_idx = self.rng.choice(len(exc), n_exc, replace=False)
            exclude_pool: np.ndarray | None = exc[exc_idx]
        else:
            exclude_pool = None

        def get_captured_mask(sdf_values: np.ndarray) -> np.ndarray:
            if self.task_type == "regression":
                return np.abs(sdf_values) < self.capture_threshold
            return sdf_values < self.capture_threshold

        while len(unexplained_pool) >= minimal_seed_size:
            expert = Expert(alpha=self.alpha)
            prev_captured_count = 0
            # Baseline score in the *same metric* as candidate scoring.
            # In discriminative mode candidate scores are F_β ∈ [0, 1), so the
            # baseline must be the expert's own previous F_β score — comparing
            # against a raw capture count would make growth impossible once
            # a single point is captured.
            prev_score = 0.0

            # Dynamic RANSAC budget: inlier fraction estimated from pool occupancy.
            # As the unexplained pool shrinks, the remaining points form a smaller
            # fraction of the original set, yielding a tighter budget estimate.
            if self.max_iterations is not None:
                max_iters = self.max_iterations
            else:
                inlier_frac = max(0.01, len(unexplained_pool) / initial_pool_size)
                max_iters = _ransac_budget(inlier_frac, minimal_seed_size)

            # For the GPU covariance-fitting path (d > 6), the classical RANSAC
            # budget formula over-estimates because it assumes minimal-subset
            # sampling where a single outlier ruins the fit.  Covariance ellipsoids
            # are robust to a few outliers: quality plateaus at 500 trials for d≈19
            # and at 1000 for 6 < d ≤ 10.  Capping avoids building 300 MB arrays
            # that dominate runtime without improving best-candidate quality.
            if self._uses_gpu_candidate_path() and d > 6:
                cov_cap = 500 if d > 10 else 1000
                max_iters = min(max_iters, cov_cap)

            # ---- Inner loop: grow this expert ----
            # eval_pts is constant for the life of this expert (pool and
            # exclude_pool don't change until the expert is locked).
            # Pre-computing it here lets batch_sdf_and_score reuse the
            # sticky-pts device buffer across all inner loop iterations.
            if self._uses_gpu_candidate_path():
                from src.gpu_engine import batch_sdf_and_score as _batch_sdf_score
                eval_pts   = (np.vstack([unexplained_pool, exclude_pool]).astype(np.float32)
                              if exclude_pool is not None
                              else np.ascontiguousarray(unexplained_pool, dtype=np.float32))
                _N_pool    = len(unexplained_pool)
                _alpha     = self.alpha
                _threshold = self.capture_threshold
                _is_reg    = (self.task_type == "regression")

            while True:
                # Seed RANSAC from points not yet captured by this expert
                if expert.ellipsoids:
                    already = get_captured_mask(expert.compute_sdf(unexplained_pool))
                    seed_pool = unexplained_pool[~already]
                else:
                    seed_pool = unexplained_pool

                if len(seed_pool) < minimal_seed_size:
                    break

                best_candidate      = None
                best_captured_count = prev_captured_count
                best_disc_score     = prev_score  # updated by whichever path accepts a candidate

                if self._uses_gpu_candidate_path():
                    # GPU batched RANSAC: vectorised covariance-based candidate
                    # generation + pipelined GPU scoring via batch_sdf_and_score.
                    # The (N×K) SDF matrix never crosses PCIe; only (K,) int32
                    # counts are downloaded (500× smaller transfer).
                    all_cands = self._batch_generate_candidates_gpu(
                        seed_pool, max_iters
                    )

                    if all_cands:
                        # Expert SDF for combination (cheap: 0-3 ellipsoids, CPU)
                        ex_sdf_np = (expert.compute_sdf(eval_pts)
                                     if expert.ellipsoids else None)

                        pos_counts, neg_counts = _batch_sdf_score(
                            all_cands, eval_pts, _N_pool, ex_sdf_np,
                            _alpha, _threshold, _is_reg,
                            existing_count=len(expert.ellipsoids),
                        )  # both (K,) int32

                        if exclude_pool is not None:
                            b2 = self.score_beta ** 2
                            disc_scores = (
                                (1.0 + b2) * pos_counts.astype(float)
                                / ((1.0 + b2) * pos_counts + b2 * neg_counts + 1.0)
                            )
                        else:
                            disc_scores = pos_counts.astype(float)

                        best_k = int(disc_scores.argmax())
                        if disc_scores[best_k] > prev_score:
                            best_candidate      = all_cands[best_k]
                            best_captured_count = int(pos_counts[best_k])
                            best_disc_score     = float(disc_scores[best_k])

                else:
                    # CPU sequential RANSAC
                    best_disc_score = prev_score
                    use_knn = self.knn_seeding and d > 6
                    for trial_i in range(max_iters):
                        if use_knn and trial_i % 2 == 0 and len(seed_pool) >= minimal_seed_size:
                            # kNN-anchored seed: high inlier purity in high-d
                            seed_pts = self._knn_seed(
                                seed_pool, minimal_seed_size, self.rng,
                            )
                        else:
                            idx = self.rng.choice(
                                len(seed_pool), minimal_seed_size, replace=False,
                            )
                            seed_pts = seed_pool[idx]
                        try:
                            candidate = self._generate_candidate(seed_pts)
                        except (np.linalg.LinAlgError, ValueError):
                            continue
                        if not self._is_plausible(candidate, unexplained_pool):
                            continue

                        expert.ellipsoids.append(candidate)
                        pos_count = int(np.sum(get_captured_mask(
                            expert.compute_sdf(unexplained_pool)
                        )))
                        if exclude_pool is not None:
                            neg_count = int(np.sum(get_captured_mask(
                                expert.compute_sdf(exclude_pool)
                            )))
                            disc_score = self._disc_score(pos_count, neg_count)
                        else:
                            disc_score = float(pos_count)
                        expert.ellipsoids.pop()

                        if disc_score > best_disc_score:
                            best_disc_score     = disc_score
                            best_captured_count = pos_count
                            best_candidate      = candidate

                if best_candidate is None:
                    break  # RANSAC found nothing useful

                growth = best_captured_count - prev_captured_count
                min_growth = max(1, int(self.min_growth_fraction * len(unexplained_pool)))
                if growth < min_growth:
                    break  # Stagnated

                expert.add_ellipsoid(best_candidate)
                prev_captured_count = best_captured_count
                prev_score = best_disc_score
                print(
                    f"  Expert {self._primitive_label} #{len(expert.ellipsoids)} added; "
                    f"captures {best_captured_count}/{len(unexplained_pool)} pool pts."
                )

            # ---- Lock or stop ----
            if not expert.ellipsoids:
                break

            captured_mask = get_captured_mask(expert.compute_sdf(unexplained_pool))
            captured_count = int(np.sum(captured_mask))

            if (captured_count / len(unexplained_pool)) >= self.consensus_threshold:
                self.model.append(expert)
                unexplained_pool = unexplained_pool[~captured_mask]
                print(
                    f"Locked Expert #{len(self.model)}: "
                    f"{len(expert.ellipsoids)} {self._primitive_plural}, "
                    f"{captured_count} pts captured, "
                    f"{len(unexplained_pool)} remain."
                )
            else:
                break

        return self.model

    # ------------------------------------------------------------------
    # Subtractive ellipsoid fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _is_plausible(candidate: EllipsoidExpert, pool: np.ndarray) -> bool:
        """Geometry sanity-check reused by both build_model and fit_subtractive_ellipsoids."""
        pool_min = np.min(pool, axis=0)
        pool_max = np.max(pool, axis=0)
        pool_span = np.maximum(pool_max - pool_min, 1e-6)
        max_span = np.max(pool_span)
        center_margin = 0.25 * pool_span
        if np.any(candidate.center < pool_min - center_margin) or \
           np.any(candidate.center > pool_max + center_margin):
            return False
        if np.any(candidate.radii > 2.5 * max_span):
            return False
        return True

    def fit_subtractive_ellipsoids(
        self,
        expert: Expert,
        exclusion_points: np.ndarray,
        max_sub_ellipsoids: int = 5,
        min_coverage_fraction: float = 0.05,
        nudge_iterations: int = 10,
        nudge_learning_rate: float = 0.02,
        max_rounds: int = 3,
        acceptance_positive_points: np.ndarray | None = None,
        acceptance_negative_points: np.ndarray | None = None,
        audit_trail: list[dict] | None = None,
        mdl_penalty_weight: float = 0.0,
        min_penalized_gain: float = 0.0,
    ) -> None:
        """
        Iteratively find and refine subtractive (polarity=-1) ellipsoids for an expert.

        Each round:
        1. RANSAC: fit ellipsoids to current false-capture points (exclusion points
           that are still captured by the expert after previous rounds).
        2. Nudge: refine newly-added subtractive ellipsoids by nudging their center
           toward the centroid and radii toward the std of the false-capture points
           they cover (same force-directed logic as NudgeEngine).

        Repeats up to *max_rounds* times or until false captures stop decreasing.

        :param expert: The Expert to augment with subtractive ellipsoids.
        :param exclusion_points: Points that should NOT be captured by this expert.
        :param max_sub_ellipsoids: Hard cap on total subtractive ellipsoids added.
        :param min_coverage_fraction: RANSAC candidate is accepted only if it covers
                                       at least this fraction of the current false-capture pool.
        :param nudge_iterations: Force-directed refinement steps per round.
        :param nudge_learning_rate: Step size for center/radii nudge.
        :param max_rounds: Maximum RANSAC + nudge rounds.
        """
        pts = np.asarray(exclusion_points, dtype=np.float64)
        if pts.ndim != 2 or len(pts) == 0:
            return

        d = pts.shape[1]
        min_seed = self._minimal_seed_size(d)

        explicit_acceptance = (
            acceptance_positive_points is not None
            and acceptance_negative_points is not None
        )
        acceptance_positive = (
            np.asarray(acceptance_positive_points, dtype=np.float64)
            if explicit_acceptance else None
        )
        acceptance_negative = (
            np.asarray(acceptance_negative_points, dtype=np.float64)
            if explicit_acceptance else None
        )

        # Reserve 20% of exclusion points for fallback hold-out validation against subtractive
        # overfitting.  Only split when enough points remain after the split for RANSAC
        # to have diverse seeds (training portion must have ≥ 2·min_seed false captures).
        n_val = int(0.2 * len(pts))
        if not explicit_acceptance and n_val >= 1 and (len(pts) - n_val) >= 2 * min_seed:
            perm    = self.rng.permutation(len(pts))
            val_pts = pts[perm[:n_val]]
            pts     = pts[perm[n_val:]]
        else:
            val_pts = None  # too few points — skip hold-out guard

        # --- Initial overlap measurement ---
        # Compute false-capture fraction once before the round loop.
        # This serves two purposes:
        #   1. Early exit if the pool is too small for RANSAC to have diverse seeds.
        #   2. Estimate the RANSAC budget via the convergence formula.
        initial_sdf = expert.compute_sdf(pts)
        initial_false = int(np.sum(initial_sdf < self.capture_threshold))

        if initial_false < 2 * min_seed:
            return  # too few false captures for RANSAC to have diverse seeds

        # Dynamic RANSAC budget: replace ad-hoc effort tiers with the RANSAC convergence
        # formula.  The inlier fraction is estimated from the fraction of (training)
        # exclusion points that are currently falsely captured — higher overlap means
        # denser clusters and higher all-inlier probability per trial.
        overlap_frac = initial_false / len(pts)
        if self.max_iterations is not None:
            sub_iters = self.max_iterations
        else:
            sub_iters = _ransac_budget(max(0.01, overlap_frac), min_seed)
        print(f"  Overlap {overlap_frac:.1%} -> {sub_iters} RANSAC iters/round.")

        total_added = 0
        prev_false_count = initial_false + 1  # sentinel: first round always runs

        # Track false captures on the hold-out slice so the acceptance guard below
        # can compare before/after each candidate addition.
        val_false_count: int | None = (
            int(np.sum(expert.compute_sdf(val_pts) < self.capture_threshold))
            if val_pts is not None else None
        )

        for round_i in range(max_rounds):
            if total_added >= max_sub_ellipsoids:
                break

            # --- Evaluate current false captures ---
            sdf = expert.compute_sdf(pts)
            false_mask = sdf < self.capture_threshold
            pool = pts[false_mask]
            false_count = len(pool)

            if false_count == 0 or false_count >= prev_false_count:
                break  # converged or no improvement from last round
            if false_count < 2 * min_seed:
                break  # pool has shrunk below useful RANSAC size mid-run
            prev_false_count = false_count

            # --- RANSAC: find ellipsoids that cover false-capture pool ---
            added_this_round: List[EllipsoidExpert] = []
            remaining = pool.copy()

            while (total_added + len(added_this_round) < max_sub_ellipsoids
                   and len(remaining) >= min_seed):
                best_candidate = None
                best_count     = 0

                if self._uses_gpu_candidate_path():
                    all_cands = self._batch_generate_candidates_gpu(
                        remaining, sub_iters
                    )

                    if all_cands:
                        from src.gpu_engine import batch_sdf as _batch_sdf
                        sdf_matrix = _batch_sdf(all_cands, remaining)  # (len(remaining), K)
                        counts     = (np.abs(sdf_matrix) < self.capture_threshold).sum(axis=0)
                        best_k     = int(counts.argmax())
                        best_count = int(counts[best_k])
                        best_candidate = all_cands[best_k]
                else:
                    for _ in range(sub_iters):
                        idx = self.rng.choice(
                            len(remaining), min_seed, replace=False,
                        )
                        try:
                            candidate = self._generate_candidate(remaining[idx])
                        except (np.linalg.LinAlgError, ValueError):
                            continue
                        if not self._is_plausible(candidate, remaining):
                            continue
                        sdf_c = candidate.compute_sdf(remaining)
                        count = int(np.sum(np.abs(sdf_c) < self.capture_threshold))
                        if count > best_count:
                            best_count = count
                            best_candidate = candidate

                min_req = max(1, int(min_coverage_fraction * false_count))
                if best_candidate is None or best_count < min_req:
                    break

                # Inflate by capture_threshold so the false-capture points end up
                # *inside* the subtractive ellipsoid by at least the threshold.
                # Without this, RANSAC places the surface through the cluster and
                # max(f_add, -f_sub) stays near 0 — the CSG carving has no effect.
                # The nudge pass corrects any over-expansion toward the actual spread.
                best_candidate.radii += self.capture_threshold
                best_candidate.polarity = -1

                # Hold-out validation guard: only accept the candidate if it reduces
                # false captures on the held-out slice.  Rejection stops the round
                # because later candidates (fitting to an already-shrinking remaining
                # pool) are unlikely to improve on the hold-out either.
                expert.add_ellipsoid(best_candidate)
                if explicit_acceptance:
                    decision = self._carve_acceptance_decision(
                        expert=expert,
                        candidate=best_candidate,
                        positive_points=acceptance_positive,
                        negative_points=acceptance_negative,
                        mdl_penalty_weight=mdl_penalty_weight,
                        min_penalized_gain=min_penalized_gain,
                    )
                    if audit_trail is not None:
                        audit_trail.append(decision)
                    if not decision["accepted"]:
                        expert.ellipsoids.pop()
                        expert._bs_cache = None
                        break
                elif val_pts is not None and val_false_count is not None and val_false_count > 0:
                    new_val_false = int(np.sum(expert.compute_sdf(val_pts) < self.capture_threshold))
                    if new_val_false >= val_false_count:
                        expert.ellipsoids.pop()
                        break  # no hold-out improvement — stop adding in this round
                    val_false_count = new_val_false

                added_this_round.append(best_candidate)

                sdf_best = best_candidate.compute_sdf(remaining)
                remaining = remaining[sdf_best >= self.capture_threshold]

            if not added_this_round:
                break

            total_added += len(added_this_round)

            # --- Nudge: refine newly-added subtractive ellipsoids ---
            if nudge_iterations > 0:
                covered = np.zeros(len(pool), dtype=bool)
                for e in added_this_round:
                    covered |= (np.abs(e.compute_sdf(pool)) < self.capture_threshold)
                nudge_pts = pool[covered]

                if len(nudge_pts) >= 2:
                    for _ in range(nudge_iterations):
                        if len(added_this_round) == 1:
                            _nudge_ellipsoid_from_points(
                                added_this_round[0], nudge_pts, nudge_learning_rate
                            )
                        else:
                            inner_sdf = np.array([e.compute_sdf(nudge_pts)
                                                  for e in added_this_round])
                            nearest = np.argmin(inner_sdf, axis=0)
                            for ei, e in enumerate(added_this_round):
                                sub = nudge_pts[nearest == ei]
                                if len(sub) == 0:
                                    continue
                                _nudge_ellipsoid_from_points(e, sub, nudge_learning_rate)

            for candidate in added_this_round:
                self._project_primitive_family(candidate)

            final_false = int(np.sum(
                np.abs(expert.compute_sdf(pts)) < self.capture_threshold
            ))
            print(
                f"  Sub-round {round_i + 1}: +{len(added_this_round)} "
                f"subtractive {self._primitive_plural}, "
                f"false captures {false_count} -> {final_false}."
            )

        if total_added:
            print(f"  Total subtractive {self._primitive_plural}: {total_added}.")

    def _carve_acceptance_decision(
        self,
        expert: Expert,
        candidate: EllipsoidExpert,
        positive_points: np.ndarray,
        negative_points: np.ndarray,
        mdl_penalty_weight: float,
        min_penalized_gain: float,
    ) -> dict:
        """Score an already-added carve on held-out positive/negative points."""
        expert.ellipsoids.pop()
        expert._bs_cache = None
        before_positive = expert.compute_sdf(positive_points) < self.capture_threshold
        before_negative = expert.compute_sdf(negative_points) < self.capture_threshold
        expert.add_ellipsoid(candidate)
        after_positive = expert.compute_sdf(positive_points) < self.capture_threshold
        after_negative = expert.compute_sdf(negative_points) < self.capture_threshold

        before_balanced = 0.5 * (
            float(np.mean(before_positive)) + float(np.mean(~before_negative))
        )
        after_balanced = 0.5 * (
            float(np.mean(after_positive)) + float(np.mean(~after_negative))
        )
        parameter_count = (
            candidate.center.size
            + candidate.radii.size
            + candidate.orientation.size
        )
        acceptance_count = len(positive_points) + len(negative_points)
        penalty = mdl_penalty_weight * parameter_count / max(acceptance_count, 1)
        gain = after_balanced - before_balanced
        penalized_gain = gain - penalty
        return {
            "accepted": bool(penalized_gain > min_penalized_gain),
            "balanced_accuracy_before": before_balanced,
            "balanced_accuracy_after": after_balanced,
            "validation_gain": gain,
            "mdl_penalty": penalty,
            "penalized_gain": penalized_gain,
            "recovered_false_positives": int(np.sum(before_negative & ~after_negative)),
            "damaged_true_positives": int(np.sum(before_positive & ~after_positive)),
            "parameter_count": int(parameter_count),
        }


def test_greedy_constructor():
    """
    Basic tests for the greedy constructor.
    """
    # Direct test: fit a rotated 2D ellipsoid from minimal sample size k=d*(d+3)/2=5
    fit_constructor = GreedyConstructor(
        max_iterations=10,
        primitive_family="ellipsoid",
    )
    true_center = np.array([0.3, -0.7])
    true_radii = np.array([2.0, 1.0])
    theta_rot = np.deg2rad(30.0)
    true_orientation = np.array([
        [np.cos(theta_rot), -np.sin(theta_rot)],
        [np.sin(theta_rot),  np.cos(theta_rot)],
    ])

    theta = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
    local_samples = np.stack([true_radii[0] * np.cos(theta), true_radii[1] * np.sin(theta)], axis=1)
    surface_points = true_center + local_samples @ true_orientation.T

    fitted = fit_constructor._generate_candidate(surface_points)
    fitted_sdf = fitted.compute_sdf(surface_points)
    assert np.all(np.abs(fitted_sdf) < 1e-5), "Fitted ellipsoid should pass through the seed points."

    true_shape = true_orientation @ np.diag(1.0 / (true_radii ** 2)) @ true_orientation.T
    fitted_shape = fitted.orientation @ np.diag(1.0 / (fitted.radii ** 2)) @ fitted.orientation.T
    assert np.allclose(fitted.center, true_center, atol=1e-5), "Fitted center is incorrect."
    assert np.allclose(fitted_shape, true_shape, atol=1e-5), "Fitted shape matrix is incorrect."

    # Generate a cluster of points (a circle) and some noise
    center = np.array([1.0, 1.0])
    radii = np.array([1.0, 1.0])
    
    # Circle points
    theta = np.linspace(0, 2*np.pi, 50)
    circle_points = np.array([
        [np.cos(t), np.sin(t)] for t in theta
    ]) + center
    
    # Noise
    noise = np.random.normal(loc=center, scale=0.5, size=(20, 2))
    
    all_points = np.vstack([circle_points, noise])
    
    constructor = GreedyConstructor(
        consensus_threshold=0.2, 
        capture_threshold=0.2,
        task_type="regression",
        alpha=1.0,
        max_iterations=100
    )
    
    model = constructor.build_model(all_points)
    
    print(f"Built model with {len(model)} experts.")
    assert len(model) > 0, "Model should have at least one expert"
    
    # Check if the first expert is somewhat near the circle center
    assert np.allclose(model[0].center, center, atol=0.5)

if __name__ == "__main__":
    test_greedy_constructor()