import numpy as np
from typing import List

class EllipsoidExpert:
    """
    Represents a d-dimensional ellipsoid using a signed normalized radial field.
    """
    def __init__(self, center: np.ndarray, radii: np.ndarray,
                 orientation: np.ndarray | None = None, polarity: int = 1):
        """
        :param center: Center of the ellipsoid, shape (d,).
        :param radii: Semi-axes, shape (d,).
        :param orientation: Orientation matrix R, shape (d, d).
                            Defaults to the identity matrix (axis-aligned ellipsoid).
        :param polarity: +1 for additive (default); -1 for subtractive (acts as a CSG hole).
        """
        self.center = np.array(center, dtype=np.float64)
        self.radii = np.array(radii, dtype=np.float64)
        d = len(self.center)
        self.orientation = (
            np.eye(d, dtype=np.float64) if orientation is None
            else np.array(orientation, dtype=np.float64)
        )
        self.polarity = int(polarity)

    @classmethod
    def from_precision_metric(
        cls,
        center: np.ndarray,
        metric,
        *,
        radius_scale: float = 1.0,
        polarity: int = 1,
    ) -> "EllipsoidExpert":
        """Convert a positive-definite precision metric to ellipsoid axes."""
        from src.metric_parameterization import PrecisionMetric

        if not isinstance(metric, PrecisionMetric):
            raise TypeError("metric must be a PrecisionMetric.")
        if not np.isfinite(radius_scale) or radius_scale <= 0.0:
            raise ValueError("radius_scale must be finite and positive.")
        precision = metric.dense_precision()
        eigenvalues, eigenvectors = np.linalg.eigh(precision)
        if np.any(eigenvalues <= 0.0):
            raise ValueError("metric precision must be positive definite.")
        radii = float(radius_scale) / np.sqrt(eigenvalues)
        return cls(center, radii, eigenvectors, polarity=polarity)
    
    def bounding_sphere(self) -> tuple:
        """Return (center, radius) of the smallest sphere enclosing this ellipsoid.

        For an oriented ellipsoid the bounding sphere has radius = max(radii),
        because the longest semi-axis gives the greatest distance from the center
        and rotation does not change Euclidean distances.
        This provides the SDF lower bound: SDF(x) >= ||x - c|| / radius - 1.
        """
        return self.center, float(np.max(self.radii))

    def compute_sdf(self, points: np.ndarray) -> np.ndarray:
        """
        Compute the SDF for a set of points.
        
        :param points: A 2D numpy array of shape (N, d) representing N points in d-dimensional space.
        :return: A 1D numpy array of shape (N,) containing the SDF values.
        """
        # points: (N, d), self.center: (d,)
        # Subtract center from each point
        shifted_points = points - self.center

        # Convert world-space coordinates to ellipsoid-local coordinates
        # using the current orientation matrix.
        local_points = shifted_points @ self.orientation
        
        # Compute the normalized distance: sum((x_i - c_i)^2 / a_i^2)
        # (shifted_points**2 / self.radii**2).sum(axis=1)
        # We use the formula f(x) = sqrt(sum((x_i - c_i)^2 / a_i^2)) - 1
        
        squared_distances = np.sum((local_points**2) / (self.radii**2), axis=1)
        sdf_values = np.sqrt(squared_distances) - 1
        
        return sdf_values

    def compute_metric_sdf(self, points: np.ndarray) -> np.ndarray:
        """First-order Euclidean-distance correction of the normalized SDF.

        ``compute_sdf`` returns a *normalized* (Mahalanobis-style) distance:
        exact for spheres, but for anisotropic ellipsoids ``‖∇f‖ ≠ 1`` so the
        same numeric value corresponds to different Euclidean gaps along
        different axes.  The standard first-order metric correction is::

            f_metric(x) = f(x) / ‖∇f(x)‖

        With q = Rᵀ(x−c) and D = f+1, the gradient is R·(q/(a²·D)) whose norm
        is ‖q/a²‖ / D (R is orthonormal), giving::

            f_metric = f · D / ‖q / a²‖

        This is exact for spheres and a local approximation elsewhere. It can
        improve scale consistency, but it is not a conservative Euclidean
        distance bound and is not guaranteed to be a safe sphere-tracing step.

        :param points: (N, d) array.
        :return: (N,) array of approximately-Euclidean signed distances.
        """
        points = np.asarray(points, dtype=np.float64)
        local = (points - self.center) @ self.orientation      # (N, d)
        q_scaled = local / (self.radii ** 2)                    # q / a²
        norm_q = np.sqrt(np.sum((local ** 2) / (self.radii ** 2), axis=1))  # (N,)
        f = norm_q - 1.0
        grad_norm = np.linalg.norm(q_scaled, axis=1) / np.maximum(norm_q, 1e-12)
        return f / np.maximum(grad_norm, 1e-12)

class SoftminFusion:
    """
    Combines multiple overlapping experts into a single continuous manifold using
    **normalized** Softmin fusion:

        f_fused = -(1/α) · ln( (1/M) · Σ exp(-α·f_i) )

    The 1/M normalization makes the fusion mixture-consistent: without it,
    log-sum-exp lies strictly below the hard min, so M coincident members would
    yield f - ln(M)/α — groups with many members would capture an artificial
    halo that grows with member count.  With normalization, M coincident members
    fuse to exactly f, and the fused value approaches hard min as α → ∞ from a
    bounded offset ≤ ln(M)/α.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initialize the fusion engine.
        
        :param alpha: The concentration parameter for the Softmin operation.
                       Larger alpha makes the fusion sharper (closer to hard min).
        """
        self.alpha = alpha

    def fuse(self, experts, points: np.ndarray) -> np.ndarray:
        """Fuse the SDFs of multiple experts/ellipsoids at given points.

        Dispatches to one of two specialised implementations:

        **EllipsoidExpert list** — ``_fuse_ellipsoids``

            Stacks all centers / radii / orientation matrices and computes the
            full (N, M, d) SDF in a single batched ``einsum``, replacing the
            M-iteration Python loop.  No approximation.

        **Expert list** — ``_fuse_experts_pruned``

            Two-pass bounding-sphere pruning:

            1. Compute lower bounds (O(N·M·d), no orientation matmul) and
               pick the globally closest Expert.
            2. Compute actual SDF for that Expert as a per-point reference.
               Any Expert whose lower bound exceeds ``reference_SDF + cutoff``
               is skipped (contribution < exp(-cutoff·α) relative to dominant).
               With cutoff = 10/α, the error is < M·exp(-10)/α ≈ 7e-4.
            3. For active Experts, compute SDF only for the locally active
               subset of points.

        :param experts: List of ``EllipsoidExpert`` *or* ``Expert`` objects.
        :param points: Array of shape (N, d).
        :return: Array of shape (N,) with fused SDF values.
        """
        if not experts:
            return np.full(len(points), np.inf)
        if len(experts) == 1:
            return experts[0].compute_sdf(points)

        points = np.asarray(points, dtype=np.float64)

        # EllipsoidExpert: direct loop — fastest for M ≤ 50 (BLAS calls amortise
        # Python overhead, and a batched einsum creates a large (N,M,d)
        # intermediate that overflows L3 cache for M > 20).
        if isinstance(experts[0], EllipsoidExpert):
            M = len(experts)
            sdf_m = np.array([e.compute_sdf(points) for e in experts])  # (M, N)
            if self.alpha == 0.0:
                return sdf_m.mean(axis=0)
            logits = -self.alpha * sdf_m
            maximum = logits.max(axis=0)
            log_mean_exp = maximum + np.log(
                np.exp(logits - maximum).sum(axis=0) / M
            )
            return -log_mean_exp / self.alpha

        # Expert objects: apply bounding-sphere global pruning.
        return self._fuse_experts_pruned(experts, points)

    def _fuse_experts_pruned(self, experts, points: np.ndarray) -> np.ndarray:
        """Bounding-sphere global pruning for a list of Expert objects.

        For each Expert, ``SDF(x) >= ||x - c_bs|| / r_bs - 1`` where (c_bs, r_bs)
        is the bounding sphere of the Expert's additive ellipsoids.

        An Expert is **globally pruned** when its bounding-sphere lower bound
        for *every* query point exceeds ``ref_sdf_max + cutoff``:

        - The reference SDF is the actual SDF of the globally closest Expert
          (evaluated for all N points).
        - Any globally pruned Expert has SDF > ref_sdf_max + cutoff everywhere,
          so its contribution exp(-α·SDF) < exp(-α·cutoff) relative to the
          dominant term, for all N points simultaneously.
        - With ``cutoff = 10/α``: max error < M·exp(-10)/α ≈ 7e-4.

        Globally pruned Experts are skipped entirely — no fancy indexing, no
        partial array copies.  The active Experts are evaluated for all N points.
        This is most effective when Experts are spatially separated (multi-cluster
        classes) and most effective for large M or well-separated query sets.
        """
        M = len(experts)

        if self.alpha == 0.0:
            return np.array([
                expert.compute_sdf(points) for expert in experts
            ]).mean(axis=0)

        if M < 4:
            sdf_m = np.array([e.compute_sdf(points) for e in experts])  # (M, N)
            logits = -self.alpha * sdf_m
            maximum = logits.max(axis=0)
            return -(maximum + np.log(
                np.exp(logits - maximum).sum(axis=0) / M
            )) / self.alpha

        # --- Bounding-sphere SDF lower bounds (O(N·M·d), no orientation matmul) ---
        bs  = [e.bounding_sphere() for e in experts]
        bsc = np.array([c for c, _ in bs])           # (M, d)
        bsr = np.maximum([r for _, r in bs], 1e-8)   # (M,)

        # ||x_i - c_j||  via identity  ||a-b||² = ||a||² + ||b||² - 2a·b
        psq   = (points ** 2).sum(axis=1, keepdims=True)                     # (N, 1)
        csq   = (bsc    ** 2).sum(axis=1, keepdims=True).T                   # (1, M)
        dists = np.sqrt(np.maximum(psq + csq - 2.0 * points @ bsc.T, 0.0))  # (N, M)
        sdf_lb = dists / bsr[np.newaxis, :] - 1.0                            # (N, M)

        # Global minimum lower bound per Expert across all query points.
        lb_min = sdf_lb.min(axis=0)  # (M,): could this Expert ever matter?

        # --- Pass 1: actual SDF for the globally closest Expert ---
        g       = int(lb_min.argmin())
        ref_sdf = experts[g].compute_sdf(points)   # (N,) — exact, no approximation
        ref_max = float(ref_sdf.max())

        # --- Global prune: skip Experts whose lower bound everywhere exceeds ---
        # ref_sdf_max + cutoff.  They cannot improve the Softmin for any point.
        sdf_cutoff    = 10.0 / self.alpha
        globally_active = lb_min <= ref_max + sdf_cutoff  # (M,)
        if not globally_active[g]:
            globally_active[g] = True  # reference Expert always kept

        # --- Softmin over active Experts, full N points, no fancy indexing ---
        active_sdfs = [ref_sdf]
        for j, expert in enumerate(experts):
            if j == g or not globally_active[j]:
                continue
            active_sdfs.append(expert.compute_sdf(points))

        sdf_matrix = np.array(active_sdfs)                              # (M_a, N)
        logits = -self.alpha * sdf_matrix
        maximum = logits.max(axis=0)
        return -(maximum + np.log(
            np.exp(logits - maximum).sum(axis=0) / M
        )) / self.alpha

class Expert:
    """
    A composite expert consisting of a group of EllipsoidExpert primitives.
    The group grows incrementally during construction: each new ellipsoid extends
    the region captured by the expert. The expert's SDF is the Softmin fusion of
    all its constituent ellipsoids, so the combined surface is smooth and continuous.
    """

    def __init__(self, alpha: float = 1.0):
        self.ellipsoids: List[EllipsoidExpert] = []
        self.alpha = alpha
        self._fusion = SoftminFusion(alpha=alpha)
        self._bs_cache: tuple | None = None  # cached (center, radius) bounding sphere

    def add_ellipsoid(self, ellipsoid: EllipsoidExpert) -> None:
        self.ellipsoids.append(ellipsoid)
        self._bs_cache = None  # invalidate on structural change

    def bounding_sphere(self) -> tuple:
        """Return (center, radius) of a bounding sphere enclosing all additive ellipsoids.

        The sphere is centred at the centroid of the additive ellipsoid centres.
        Its radius extends to the furthest surface point of any additive ellipsoid::

            radius = max_k ( ||c_k - centroid|| + r_max_k )

        Subtractive ellipsoids only make the SDF *larger* inside the additive volume
        (CSG difference), so they cannot create regions where SDF < 0 outside the
        additive bounding sphere — the additive sphere is the correct pruning bound.

        The result is cached and invalidated whenever an ellipsoid is added.
        """
        if self._bs_cache is not None:
            return self._bs_cache
        pos = [e for e in self.ellipsoids if e.polarity > 0]
        if not pos:
            # No additive ellipsoids: SDF is +inf everywhere — return a zero-radius sphere.
            fallback = (self.ellipsoids[0].center.copy() if self.ellipsoids
                        else np.zeros(1)), 0.0
            self._bs_cache = fallback
            return self._bs_cache
        centers  = np.array([e.center for e in pos])                    # (K, d)
        max_rads = np.array([float(np.max(e.radii)) for e in pos])     # (K,)
        centroid = centers.mean(axis=0)                                 # (d,)
        reach    = np.linalg.norm(centers - centroid, axis=1) + max_rads  # (K,)
        self._bs_cache = (centroid, float(reach.max()))
        return self._bs_cache

    # ------------------------------------------------------------------
    # Compatibility shims: expose aggregate geometry so downstream code
    # that reads .center / .radii gets meaningful values for multi-member
    # experts instead of silently returning only the first ellipsoid.
    # ------------------------------------------------------------------
    @property
    def center(self) -> np.ndarray:
        """Centroid of all additive ellipsoid centers (or first center as fallback)."""
        if not self.ellipsoids:
            return None
        pos = [e for e in self.ellipsoids if e.polarity > 0]
        if pos:
            return np.mean([e.center for e in pos], axis=0)
        return self.ellipsoids[0].center.copy()

    @property
    def radii(self) -> np.ndarray:
        """Bounding semi-axes: per-axis max over all additive ellipsoids.

        Returns the element-wise maximum of each ellipsoid's (axis-aligned)
        radii, which conservatively bounds the union in each axis direction.
        For single-ellipsoid experts this is identical to the single radii array.
        """
        if not self.ellipsoids:
            return None
        pos = [e for e in self.ellipsoids if e.polarity > 0]
        group = pos if pos else self.ellipsoids
        return np.max([e.radii for e in group], axis=0)

    def compute_sdf(self, points: np.ndarray) -> np.ndarray:
        """Fused SDF over all ellipsoids in this expert.

        For a pure-additive expert this is Softmin of the positive ellipsoids.
        When subtractive ellipsoids are present, applies a CSG difference:
            f_exp = max(f_add, -f_sub)
        where f_add = Softmin of positive ellipsoids and
              f_sub = Softmin of negative ellipsoids.
        Points inside a subtractive ellipsoid receive a positive SDF (not captured).
        """
        if not self.ellipsoids:
            return np.full(len(np.asarray(points)), np.inf)

        pos = [e for e in self.ellipsoids if e.polarity > 0]
        neg = [e for e in self.ellipsoids if e.polarity < 0]

        if not pos:
            return np.full(len(np.asarray(points)), np.inf)

        f_add = pos[0].compute_sdf(points) if len(pos) == 1 \
            else self._fusion.fuse(pos, points)

        if not neg:
            return f_add

        f_sub = neg[0].compute_sdf(points) if len(neg) == 1 \
            else self._fusion.fuse(neg, points)

        # CSG difference (hard max — exact geometry, correct capture logic)
        return np.maximum(f_add, -f_sub)

    def compute_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of this expert's fused SDF at point x (shape (d,)).

        For pure-additive experts: weighted average of per-ellipsoid orientation-aware
        gradients (Softmin weights).

        For experts with subtractive ellipsoids, follows the active branch of
        the hard CSG difference ``max(f_add, -f_sub)``. At the nondifferentiable
        tie, returns the average of the two branch gradients.
        """
        if not self.ellipsoids:
            return np.zeros_like(x)

        pos = [e for e in self.ellipsoids if e.polarity > 0]
        neg = [e for e in self.ellipsoids if e.polarity < 0]

        def _ellipsoid_gradient(e: EllipsoidExpert) -> np.ndarray:
            D = float(e.compute_sdf(x.reshape(1, -1))[0]) + 1.0
            if D < 1e-6:
                return np.zeros_like(x)
            q = (x - e.center) @ e.orientation
            return e.orientation @ (q / (e.radii ** 2 * D))

        def _group_gradient(group):
            """Softmin-weighted gradient over a group of ellipsoids."""
            if not group:
                return np.zeros_like(x)
            sdf_vals = np.array([e.compute_sdf(x.reshape(1, -1))[0] for e in group])
            weights = np.exp(-self.alpha * sdf_vals)
            grads = np.array([_ellipsoid_gradient(e) for e in group])
            return np.sum(grads * weights.reshape(-1, 1), axis=0) / np.sum(weights)

        grad_add = _group_gradient(pos)

        if not neg:
            return grad_add

        grad_sub = _group_gradient(neg)

        f_add = float(self._fusion.fuse(pos, x.reshape(1, -1))[0]) if len(pos) > 1 \
            else float(pos[0].compute_sdf(x.reshape(1, -1))[0])
        f_sub = float(self._fusion.fuse(neg, x.reshape(1, -1))[0]) if len(neg) > 1 \
            else float(neg[0].compute_sdf(x.reshape(1, -1))[0])

        if f_add > -f_sub:
            return grad_add
        if f_add < -f_sub:
            return -grad_sub
        return 0.5 * (grad_add - grad_sub)


def test_sdf_engine():
    """
    Basic tests for the SDF engine components.
    """
    # Test EllipsoidExpert SDF
    e1 = EllipsoidExpert(center=np.array([0.0, 0.0]), radii=np.array([1.0, 1.0]))
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 2.0]])
    sdf = e1.compute_sdf(points)
    
    # At center, SDF should be -1
    assert np.isclose(sdf[0], -1.0), f"Expected -1.0, got {sdf[0]}"
    # At edge (radius 1), SDF should be 0
    assert np.isclose(sdf[1], 0.0), f"Expected 0.0, got {sdf[1]}"
    # Outside, SDF should be > 0
    assert sdf[2] > 0, f"Expected > 0, got {sdf[2]}"

    # Test orientation-aware SDF (90-degree rotation in 2D)
    e_rot = EllipsoidExpert(center=np.array([0.0, 0.0]), radii=np.array([2.0, 1.0]))
    e_rot.orientation = np.array([[0.0, -1.0], [1.0, 0.0]])  # +90 deg rotation
    rot_points = np.array([[0.0, 2.0], [2.0, 0.0]])
    rot_sdf = e_rot.compute_sdf(rot_points)
    assert np.isclose(rot_sdf[0], 0.0), f"Expected 0.0, got {rot_sdf[0]}"
    assert rot_sdf[1] > 0, f"Expected > 0, got {rot_sdf[1]}"

    # Test SoftminFusion
    fusion = SoftminFusion(alpha=2.0)
    e2 = EllipsoidExpert(center=np.array([1.5, 0.0]), radii=np.array([1.0, 1.0]))
    fused = fusion.fuse([e1, e2], points)
    
    # Check that fused values are calculated (all values should be finite)
    assert np.all(np.isfinite(fused)), "Fused SDF values contain non-finite numbers"
    
    print("All SDF engine tests passed!")

if __name__ == "__main__":
    test_sdf_engine()
