"""GPU-accelerated batch inference for GEODE using PyOpenCL.

Targets AMD GPUs on Windows via AMD's OpenCL Accelerated Parallel Processing
(APP) driver.  The RX 9070 XT (gfx1201, 32 CUs) is preferred automatically
over the integrated GPU (gfx1036, 1 CU) by selecting the device with the most
compute units.

Three-stage GPU pipeline
------------------------
All three stages run on-device; only ``(N, d)`` input points and ``(N, C)``
class SDFs cross the PCIe bus.

  Stage 1 — **Ellipsoid SDF**  (N × K_pos workitems):
      For each point ``n`` and each positive ellipsoid ``k``::

          diff   = points[n] - centers[k]
          q      = diff @ R_k                     (d×d orientation matmul)
          sdf[n, k] = sqrt( sum( (q / radii_k)² ) ) - 1

  Stage 2 — **Expert softmin + CSG**  (N × E workitems):
      For each expert ``e``, collect its slice of ``sdf_pos`` and apply
      log-sum-exp::

          f_add[n, e] = -(1/α) log Σ_{k ∈ e} exp(-α · sdf_pos[n, k])

      For experts with subtractive (CSG) ellipsoids also compute ``f_sub``
      and apply the CSG hard-max::

          expert_sdf[n, e] = max(f_add[n, e], -f_sub[n, e])

  Stage 3 — **Class softmin**  (N × C workitems):
      ::

          class_sdf[n, c] = -(1/α) log Σ_{e ∈ c} exp(-α · expert_sdf[n, e])

Usage::

    from src.gpu_engine import GPUInferenceEngine

    # class_models[c] = list[Expert] for class c  (same as InferenceEngine)
    gpu = GPUInferenceEngine(class_models, alpha=2.0)
    print(gpu.device_name)            # e.g. "gfx1201"

    sdf_matrix = gpu.class_sdfs(X)   # (N, C) — float64
    labels      = gpu.predict(X)      # (N,)   — int indices
"""

from __future__ import annotations

import numpy as np
import pyopencl as cl

from src.sdf_engine import Expert

# ---------------------------------------------------------------------------
# RDNA4 (gfx1201) tuning constants
# ---------------------------------------------------------------------------
# Local work size for the inner (K/E/C) dimension of all 2-D kernel dispatches.
# 64 = 2 × wave32, fits within the 256-thread max-WG limit, and keeps full
# wavefronts even when K/E/C are moderate.  Global sizes are rounded up to the
# next multiple so every wavefront is fully packed; out-of-bounds work-items
# exit after the existing bounds check.
_WG: int = 64

# Compiler flags: enable fused-multiply-add, discard -0 semantics, and flush
# sub-normal floats to zero.  These are safe for the SDF range used here
# (values in approximately [−1, 10]) and give measurable speed-ups on RDNA4.
_BUILD_OPTIONS: str = "-cl-mad-enable -cl-no-signed-zeros -cl-denorms-are-zero"

# Number of pipeline slots for async candidate batching.  Slot i's data
# (upload) overlaps with slot i-1's kernel execution on AMD's DMA engine.
_N_STAGES: int = 4


def _round_up(n: int, multiple: int) -> int:
    """Return the smallest multiple of *multiple* that is >= n."""
    return ((n + multiple - 1) // multiple) * multiple

# ---------------------------------------------------------------------------
# OpenCL kernel source — compiled once per GPUInferenceEngine instance.
# All arithmetic uses float32 (4× faster than float64 on consumer RDNA GPUs).
# For the SDF range typical in this codebase (−1 to ~5) and α = 2,
# float32 rounding error is < 1e-5, well below classification thresholds.
# ---------------------------------------------------------------------------
_KERNEL_SRC = r"""
/* -----------------------------------------------------------------------
   Stage 1: batch ellipsoid SDF
   Global work size: (N, K_pos)
   ----------------------------------------------------------------------- */
__kernel void batch_ellipsoid_sdf(
    __global const float* restrict points,   /* (N, d) row-major */
    __global const float* restrict centers,  /* (K, d)           */
    __global const float* restrict radii,    /* (K, d)           */
    __global const float* restrict Rs,       /* (K, d, d)        */
    __global float*       restrict sdf_out,  /* (N, K) row-major */
    const int N,
    const int K,
    const int d
) {
    const int n = get_global_id(0);
    const int k = get_global_id(1);
    if (n >= N || k >= K) return;

    const __global float* pt = points  + n * d;
    const __global float* c  = centers + k * d;
    const __global float* r  = radii   + k * d;
    const __global float* R  = Rs      + k * d * d;  /* R_k: (d, d) */

    /* q_i = sum_j (pt[j] - c[j]) * R[j * d + i]   (row-major matmul)
       mad() hints the compiler to emit fused-multiply-add on RDNA4. */
    float sq = 0.0f;
    for (int i = 0; i < d; i++) {
        float q_i = 0.0f;
        for (int j = 0; j < d; j++) {
            q_i = mad(pt[j] - c[j], R[j * d + i], q_i);
        }
        float t = q_i / r[i];
        sq = mad(t, t, sq);
    }
    sdf_out[n * K + k] = sqrt(sq) - 1.0f;
}


/* Gaussian NLL implied by covariance-fitter radii r_i = sqrt(d * lambda_i). */
__kernel void batch_gaussian_nll(
    __global const float* restrict points,
    __global const float* restrict centers,
    __global const float* restrict radii,
    __global const float* restrict Rs,
    __global float*       restrict nll_out,
    const int N,
    const int K,
    const int d,
    __global const float* restrict covariance_temperatures
) {
    const int n = get_global_id(0);
    const int k = get_global_id(1);
    if (n >= N || k >= K) return;

    const __global float* pt = points + n * d;
    const __global float* c = centers + k * d;
    const __global float* r = radii + k * d;
    const __global float* R = Rs + k * d * d;
    const float covariance_temperature = covariance_temperatures[k];
    float normalized_sq = 0.0f;
    float log_determinant = 0.0f;
    for (int i = 0; i < d; i++) {
        float local_value = 0.0f;
        for (int j = 0; j < d; j++) {
            local_value = mad(pt[j] - c[j], R[j * d + i], local_value);
        }
        const float scaled = local_value / r[i];
        normalized_sq = mad(scaled, scaled, normalized_sq);
        log_determinant += log((r[i] * r[i]) / (float)d);
    }
    const float log_two_pi = 1.8378770664093453f;
    nll_out[n * K + k] = 0.5f * (
        (float)d * normalized_sq / covariance_temperature
        + log_determinant
        + (float)d * log(covariance_temperature)
        + (float)d * log_two_pi
    );
}


/* -----------------------------------------------------------------------
   Training-time axis-aligned covariance fitting
   One work-item fits one candidate seed batch.

   mode 1: diagonal covariance ellipsoid
   mode 2: spherical covariance (trace-matched equal radii)
   ----------------------------------------------------------------------- */
__kernel void fit_axis_aligned_candidates(
    __global const float* restrict seeds,   /* (K, S, d) */
    __global float*       restrict centers, /* (K, d)    */
    __global float*       restrict radii,   /* (K, d)    */
    const int K,
    const int S,
    const int d,
    const int mode
) {
    const int k = get_global_id(0);
    if (k >= K) return;

    float trace = 0.0f;
    for (int j = 0; j < d; j++) {
        float mean = 0.0f;
        for (int s = 0; s < S; s++) {
            mean += seeds[(k * S + s) * d + j];
        }
        mean /= (float)S;
        centers[k * d + j] = mean;

        float sum_sq = 0.0f;
        for (int s = 0; s < S; s++) {
            const float delta = seeds[(k * S + s) * d + j] - mean;
            sum_sq = mad(delta, delta, sum_sq);
        }
        const float variance = sum_sq / (float)(S - 1);
        trace += variance;
        radii[k * d + j] = sqrt(variance * (float)d);
    }

    if (mode == 2) {
        const float radius = sqrt(trace);
        for (int j = 0; j < d; j++) {
            radii[k * d + j] = radius;
        }
    }
}


/* -----------------------------------------------------------------------
   Stage 2: per-expert softmin + optional CSG hard-max
   Global work size: (N, E)

   pos_starts[e] / pos_counts[e] — slice into sdf_pos for expert e's
   additive ellipsoids (contiguous after pre-sorting).
   neg_counts[e] == 0  →  pure-additive expert (CSG branch skipped).
   ----------------------------------------------------------------------- */
__kernel void expert_softmin_csg(
    __global const float* restrict sdf_pos,    /* (N, K_pos)  */
    __global const float* restrict sdf_neg,    /* (N, K_neg) — may be unused */
    __global float*       restrict expert_sdf, /* (N, E)      */
    __global const int*   restrict pos_starts, /* (E,)        */
    __global const int*   restrict pos_counts, /* (E,)        */
    __global const int*   restrict neg_starts, /* (E,)        */
    __global const int*   restrict neg_counts, /* (E,)        */
    const int N,
    const int K_pos,
    const int K_neg,
    const int E,
    const float alpha
) {
    const int n = get_global_id(0);
    const int e = get_global_id(1);
    if (n >= N || e >= E) return;

    /* -- additive softmin -- */
    const int s_pos   = pos_starts[e];
    const int cnt_pos = pos_counts[e];
    if (cnt_pos <= 0) {
        expert_sdf[n * E + e] = INFINITY;
        return;
    }
    float max_add = -INFINITY;
    for (int i = 0; i < cnt_pos; i++) {
        max_add = fmax(max_add, -alpha * sdf_pos[n * K_pos + s_pos + i]);
    }
    float sumexp_add  = 0.0f;
    for (int i = 0; i < cnt_pos; i++) {
        sumexp_add += exp(-alpha * sdf_pos[n * K_pos + s_pos + i] - max_add);
    }
    const float f_add = -1.0f / alpha * (max_add + log(sumexp_add / (float)cnt_pos));

    /* -- subtractive CSG (skipped when cnt_neg == 0) -- */
    const int cnt_neg = neg_counts[e];
    if (cnt_neg == 0) {
        expert_sdf[n * E + e] = f_add;
        return;
    }

    const int s_neg  = neg_starts[e];
    float max_sub = -INFINITY;
    for (int i = 0; i < cnt_neg; i++) {
        max_sub = fmax(max_sub, -alpha * sdf_neg[n * K_neg + s_neg + i]);
    }
    float sumexp_sub = 0.0f;
    for (int i = 0; i < cnt_neg; i++) {
        sumexp_sub += exp(-alpha * sdf_neg[n * K_neg + s_neg + i] - max_sub);
    }
    const float f_sub = -1.0f / alpha * (max_sub + log(sumexp_sub / (float)cnt_neg));

    expert_sdf[n * E + e] = fmax(f_add, -f_sub);
}


/* -----------------------------------------------------------------------
   Stage 3: per-class softmin over experts
   Global work size: (N, C)
   ----------------------------------------------------------------------- */
__kernel void class_softmin(
    __global const float* restrict expert_sdf, /* (N, E)  */
    __global float*       restrict class_sdf,  /* (N, C)  */
    __global const int*   restrict exp_starts, /* (C,)    */
    __global const int*   restrict exp_counts, /* (C,)    */
    const int N,
    const int E,
    const int C,
    const float alpha
) {
    const int n = get_global_id(0);
    const int c = get_global_id(1);
    if (n >= N || c >= C) return;

    const int s   = exp_starts[c];
    const int cnt = exp_counts[c];
    if (cnt <= 0) {
        class_sdf[n * C + c] = INFINITY;
        return;
    }
    float maximum = -INFINITY;
    for (int i = 0; i < cnt; i++) {
        maximum = fmax(maximum, -alpha * expert_sdf[n * E + s + i]);
    }
    float sumexp  = 0.0f;
    for (int i = 0; i < cnt; i++) {
        sumexp += exp(-alpha * expert_sdf[n * E + s + i] - maximum);
    }
    class_sdf[n * C + c] = -1.0f / alpha * (maximum + log(sumexp / (float)cnt));
}


/* -----------------------------------------------------------------------
   Stage 4: RANSAC combine + score
   Computes per-candidate capture counts entirely on-device, avoiding the
   expensive (N×K) GPU→CPU download.  Only (K,) int32 counts cross PCIe.

   For each candidate k, computes:
     combined[n] = has_expert
         ? -(1/α) log(0.5·(e^{-α·ex_sdf[n]} + e^{-α·cand_sdf[n,k]}))
         : cand_sdf[n,k]
     pos_counts[k] += 1  if capture condition holds and n < N_pool
     neg_counts[k] += 1  if capture condition holds and n >= N_pool

   Global work size: (round_up(K, SCORE_WG), SCORE_WG)
   Local work size:  (1, SCORE_WG)
   Each work-group (1 × SCORE_WG) handles exactly one candidate k;
   the SCORE_WG threads share the N_eval iterations via tree reduction.
   ----------------------------------------------------------------------- */
#define SCORE_WG 64
__kernel void ransac_combine_score(
    __global const float* restrict cand_sdf,    /* (N_eval, K) N-major   */
    __global const float* restrict ex_sdf,      /* (N_eval,) expert SDF  */
    __global int*          restrict pos_counts, /* (K,)                  */
    __global int*          restrict neg_counts, /* (K,)                  */
    const int N_eval,
    const int K,
    const int N_pool,
    const float alpha,
    const float threshold,
    const int has_expert,
    const int existing_count,
    const int task_regression  /* 1 = fabs(sdf) < thr, 0 = sdf < thr  */
) {
    const int k = get_global_id(0);
    const int t = get_local_id(1);
    if (k >= K) return;

    __local int lpc[SCORE_WG], lnc[SCORE_WG];
    const float inv_alpha = 1.0f / alpha;
    int pc = 0, nc = 0;

    for (int n = t; n < N_eval; n += SCORE_WG) {
        float csdf = cand_sdf[n * K + k];
        float sdf;
        if (has_expert) {
            float ex = ex_sdf[n];
            float old_logit = -alpha * ex;
            float new_logit = -alpha * csdf;
            float maximum = fmax(old_logit, new_logit);
            float weighted_sum = (float)existing_count * exp(old_logit - maximum)
                               + exp(new_logit - maximum);
            sdf = -inv_alpha * (maximum
                + log(weighted_sum / (float)(existing_count + 1)));
        } else {
            sdf = csdf;
        }
        float val = task_regression ? fabs(sdf) : sdf;
        if (val < threshold) {
            if (n < N_pool) pc++; else nc++;
        }
    }

    lpc[t] = pc;
    lnc[t] = nc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = SCORE_WG >> 1; s > 0; s >>= 1) {
        if (t < s) { lpc[t] += lpc[t + s]; lnc[t] += lnc[t + s]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) { pos_counts[k] = lpc[0]; neg_counts[k] = lnc[0]; }
}
"""


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device(prefer_name: str | None = None) -> cl.Device:
    """Return the best OpenCL GPU device.

    Preference order:

    1. Device whose name contains ``prefer_name`` (if given).
    2. Device with the most compute units (discrete GPU wins over iGPU).

    The AMD iGPU (gfx1036, 1 CU) reports very large "global_mem_size"
    because it uses shared system RAM, so CU count is used instead of
    memory to distinguish discrete from integrated.

    :raises RuntimeError: if no GPU device is found.
    """
    seen: set[str] = set()
    candidates: list[cl.Device] = []

    for platform in cl.get_platforms():
        try:
            devs = platform.get_devices(device_type=cl.device_type.GPU)
        except cl.Error:
            continue
        for dev in devs:
            key = f"{dev.name}-{dev.max_compute_units}"
            if key not in seen:
                seen.add(key)
                candidates.append(dev)

    if not candidates:
        raise RuntimeError(
            "No OpenCL GPU device found.  Make sure AMD GPU drivers are installed."
        )

    if prefer_name:
        for dev in candidates:
            if prefer_name in dev.name:
                return dev

    # Pick device with the most compute units (discrete > iGPU)
    return max(candidates, key=lambda d: d.max_compute_units)


# ---------------------------------------------------------------------------
# Training-time batch SDF (module-level singleton)
# ---------------------------------------------------------------------------

class _OCLTrainingContext:
    """Module-level singleton: compiled SDF kernels reused across all RANSAC calls.

    Provides two execution paths:

    **Legacy path** (:meth:`batch_sdf`): returns a ``(N, K)`` SDF matrix.
    Used when the caller needs raw SDF values.

    **Scoring path** (:meth:`batch_sdf_and_score`): evaluates K candidates,
    combines with an existing expert SDF, and counts captures entirely on-device.
    Only ``(K,)`` int32 count arrays cross PCIe — a 500× reduction vs the
    legacy path's full SDF matrix download.  Internally the K candidates are
    split into :data:`_N_STAGES` slots whose upload/kernel/download operations
    are enqueued simultaneously into an out-of-order queue; AMD's DMA engine
    then overlaps slot-i upload with slot-(i-1) kernel execution.
    """

    _instance: "_OCLTrainingContext | None" = None

    @classmethod
    def get(cls) -> "_OCLTrainingContext":
        if cls._instance is None:
            dev   = select_device()
            ctx   = cl.Context([dev])
            prog  = cl.Program(ctx, _KERNEL_SRC).build(options=_BUILD_OPTIONS)
            obj   = object.__new__(cls)
            obj._ctx = ctx

            # In-order queue for the legacy batch_sdf path.
            obj._queue = cl.CommandQueue(ctx)

            # Out-of-order queue for the async scoring path.
            # Falls back to in-order if the driver does not expose OOO mode
            # (rare on AMD Windows but possible on some ROCm builds).
            try:
                ooo = cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE
                obj._queue_ooo = cl.CommandQueue(ctx, properties=ooo)
            except cl.Error:
                obj._queue_ooo = obj._queue  # graceful degradation

            obj._k_sdf           = cl.Kernel(prog, "batch_ellipsoid_sdf")
            obj._k_combine_score = cl.Kernel(prog, "ransac_combine_score")
            obj._k_fit_axis      = cl.Kernel(prog, "fit_axis_aligned_candidates")

            # Legacy path buffers (grown on demand by _ensure_buffers).
            obj._alloc_N = obj._alloc_K = obj._alloc_d = 0
            obj._buf_pts = obj._buf_c = obj._buf_r = obj._buf_R = obj._buf_out = None

            # Scoring path buffers (grown on demand by _ensure_scoring_buffers).
            obj._score_N = obj._score_Ks = obj._score_d = 0
            obj._buf_pts_s  = None   # (N, d) float32 — shared across all slots
            obj._buf_ex_sdf = None   # (N,)   float32 — current expert SDF
            obj._slot_c   = []       # _N_STAGES × (K_slot, d)     float32
            obj._slot_r   = []       # _N_STAGES × (K_slot, d)     float32
            obj._slot_R   = []       # _N_STAGES × (K_slot, d, d)  float32
            obj._slot_sdf = []       # _N_STAGES × (N, K_slot)     float32
            obj._slot_pc  = []       # _N_STAGES × (K_slot,)       int32
            obj._slot_nc  = []       # _N_STAGES × (K_slot,)       int32
            obj._pts_s_key = None    # (data_ptr, shape) for sticky-pts tracking

            cls._instance = obj
        return cls._instance

    # ------------------------------------------------------------------
    # Legacy path — unchanged public API
    # ------------------------------------------------------------------

    def fit_axis_aligned_candidates(
        self,
        seed_batches: np.ndarray,
        primitive_family: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit diagonal ellipsoids or trace-matched spheres on OpenCL."""
        modes = {"diagonal_ellipsoid": 1, "sphere": 2}
        if primitive_family not in modes:
            raise ValueError("GPU axis-aligned fitting requires diagonal_ellipsoid or sphere")
        seeds = np.ascontiguousarray(seed_batches, dtype=np.float32)
        if seeds.ndim != 3 or seeds.shape[1] < 2:
            raise ValueError("seed_batches must have shape (K, S, d) with S >= 2")
        candidate_count, seed_count, dimension = seeds.shape
        centers = np.empty((candidate_count, dimension), dtype=np.float32)
        radii = np.empty_like(centers)
        flags = cl.mem_flags
        seeds_buffer = cl.Buffer(
            self._ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=seeds,
        )
        centers_buffer = cl.Buffer(self._ctx, flags.WRITE_ONLY, centers.nbytes)
        radii_buffer = cl.Buffer(self._ctx, flags.WRITE_ONLY, radii.nbytes)
        self._k_fit_axis(
            self._queue,
            (_round_up(candidate_count, _WG),),
            (_WG,),
            seeds_buffer,
            centers_buffer,
            radii_buffer,
            np.int32(candidate_count),
            np.int32(seed_count),
            np.int32(dimension),
            np.int32(modes[primitive_family]),
        )
        cl.enqueue_copy(self._queue, centers, centers_buffer)
        cl.enqueue_copy(self._queue, radii, radii_buffer)
        self._queue.finish()
        return centers.astype(np.float64), radii.astype(np.float64)

    def _ensure_buffers(self, N: int, K: int, d: int) -> None:
        """Grow pre-allocated device buffers if any dimension has increased.

        Allocation is rare (typically at most once per tier run) so we can
        afford the synchronous reallocation cost here.
        """
        if N <= self._alloc_N and K <= self._alloc_K and d <= self._alloc_d:
            return  # existing buffers are large enough
        new_N = max(N, self._alloc_N)
        new_K = max(K, self._alloc_K)
        new_d = max(d, self._alloc_d)
        mf = cl.mem_flags
        self._buf_pts = cl.Buffer(self._ctx, mf.READ_WRITE, size=new_N * new_d * 4)
        self._buf_c   = cl.Buffer(self._ctx, mf.READ_WRITE, size=new_K * new_d * 4)
        self._buf_r   = cl.Buffer(self._ctx, mf.READ_WRITE, size=new_K * new_d * 4)
        self._buf_R   = cl.Buffer(self._ctx, mf.READ_WRITE, size=new_K * new_d * new_d * 4)
        self._buf_out = cl.Buffer(self._ctx, mf.READ_WRITE, size=new_N * new_K * 4)
        self._alloc_N, self._alloc_K, self._alloc_d = new_N, new_K, new_d

    def batch_sdf(self, ellipsoids: list, points: np.ndarray) -> np.ndarray:
        """Compute (N, K) SDF matrix — raw ellipsoid SDFs, no Softmin.

        :param ellipsoids: List of :class:`~src.sdf_engine.EllipsoidExpert`.
        :param points: (N, d) float array.
        :return: (N, K) float64 array.
        """
        K = len(ellipsoids)
        if K == 0:
            return np.empty((len(points), 0), dtype=np.float64)

        pts_f32 = np.ascontiguousarray(points, dtype=np.float32)
        N, d    = pts_f32.shape

        centers = np.array([e.center      for e in ellipsoids], dtype=np.float32)
        radii   = np.array([e.radii       for e in ellipsoids], dtype=np.float32)
        Rs      = np.array([e.orientation for e in ellipsoids], dtype=np.float32)

        # Reuse pre-allocated device buffers (avoids per-call allocation overhead).
        self._ensure_buffers(N, K, d)

        cl.enqueue_copy(self._queue, self._buf_pts, pts_f32)
        cl.enqueue_copy(self._queue, self._buf_c,   centers)
        cl.enqueue_copy(self._queue, self._buf_r,   radii)
        cl.enqueue_copy(self._queue, self._buf_R,   Rs)

        global_K = _round_up(K, _WG)
        self._k_sdf(
            self._queue, (N, global_K), (1, _WG),
            self._buf_pts, self._buf_c, self._buf_r, self._buf_R, self._buf_out,
            np.int32(N), np.int32(K), np.int32(d),
        )

        out = np.empty(N * K, dtype=np.float32)
        cl.enqueue_copy(self._queue, out, self._buf_out)
        self._queue.finish()
        return out.reshape(N, K).astype(np.float64)

    # ------------------------------------------------------------------
    # Scoring path — async pipelined, no SDF matrix download
    # ------------------------------------------------------------------

    def _ensure_scoring_buffers(self, N: int, K: int, d: int) -> None:
        """Allocate/grow per-slot buffers for :meth:`batch_sdf_and_score`.

        Uses a separate buffer set from the legacy path so both paths can
        coexist without interference.
        """
        K_slot = (K + _N_STAGES - 1) // _N_STAGES
        if N <= self._score_N and K_slot <= self._score_Ks and d <= self._score_d:
            return
        new_N  = max(N,      self._score_N)
        new_Ks = max(K_slot, self._score_Ks)
        new_d  = max(d,      self._score_d)
        mf = cl.mem_flags

        def rw(n: int) -> cl.Buffer:
            return cl.Buffer(self._ctx, mf.READ_WRITE, size=max(n, 4))

        self._buf_pts_s  = rw(new_N  * new_d  * 4)
        self._buf_ex_sdf = rw(new_N  * 4)
        self._slot_c   = [rw(new_Ks * new_d          * 4) for _ in range(_N_STAGES)]
        self._slot_r   = [rw(new_Ks * new_d          * 4) for _ in range(_N_STAGES)]
        self._slot_R   = [rw(new_Ks * new_d * new_d  * 4) for _ in range(_N_STAGES)]
        self._slot_sdf = [rw(new_N  * new_Ks          * 4) for _ in range(_N_STAGES)]
        self._slot_pc  = [rw(new_Ks * 4)               for _ in range(_N_STAGES)]
        self._slot_nc  = [rw(new_Ks * 4)               for _ in range(_N_STAGES)]

        self._score_N  = new_N
        self._score_Ks = new_Ks
        self._score_d  = new_d
        self._pts_s_key = None  # invalidate sticky-pts cache

    def batch_sdf_and_score(
        self,
        ellipsoids: list,
        pts_f32: np.ndarray,
        N_pool: int,
        ex_sdf_np: "np.ndarray | None",
        alpha: float,
        threshold: float,
        task_regression: bool,
        existing_count: int = 0,
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Evaluate K candidates and count captures entirely on-device.

        Splits the K candidates into :data:`_N_STAGES` equal slots and enqueues
        all upload + SDF kernel + scoring kernel + download operations into an
        out-of-order command queue.  AMD's DMA engine then overlaps slot-i
        upload with slot-(i-1) kernel execution automatically.

        :param ellipsoids: K :class:`~src.sdf_engine.EllipsoidExpert` candidates.
        :param pts_f32: ``(N_eval, d)`` float32 eval points (pre-converted).
        :param N_pool: First ``N_pool`` rows are positive pool points.
        :param ex_sdf_np: ``(N_eval,)`` float64/32 current expert SDF, or
            ``None`` if no expert ellipsoids exist yet.
        :param alpha: Softmin concentration parameter.
        :param threshold: Capture distance threshold.
        :param task_regression: If ``True`` use ``|sdf| < threshold``.
        :return: ``(pos_counts, neg_counts)`` — two ``(K,)`` int32 arrays.
        """
        K = len(ellipsoids)
        if K == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

        N, d = pts_f32.shape
        self._ensure_scoring_buffers(N, K, d)
        q = self._queue_ooo

        # --- Upload eval points (only when the array changes) ---
        pts_key = (pts_f32.ctypes.data, pts_f32.shape)
        if pts_key != self._pts_s_key:
            ev_pts = cl.enqueue_copy(q, self._buf_pts_s, pts_f32, is_blocking=False)
            self._pts_s_key = pts_key
        else:
            ev_pts = None  # already on device

        # --- Upload expert SDF ---
        has_expert = ex_sdf_np is not None
        if has_expert and existing_count < 1:
            raise ValueError("existing_count must be positive when ex_sdf_np is provided.")
        if has_expert:
            ex_f32 = np.ascontiguousarray(ex_sdf_np, dtype=np.float32)
            ev_ex = cl.enqueue_copy(q, self._buf_ex_sdf, ex_f32, is_blocking=False)
        else:
            ev_ex = None

        # Events all kernels must wait for (pts + expert SDF uploads)
        base_wait = [e for e in (ev_pts, ev_ex) if e is not None]

        # --- Pipeline: enqueue all slots before waiting for any ---
        K_slot = (K + _N_STAGES - 1) // _N_STAGES
        pc_arrs: list[np.ndarray] = []
        nc_arrs: list[np.ndarray] = []
        dl_events: list = []

        for si in range(_N_STAGES):
            lo, hi = si * K_slot, min((si + 1) * K_slot, K)
            chunk   = ellipsoids[lo:hi]
            Kc      = len(chunk)
            if Kc == 0:
                break

            # Host-side array extraction (runs while earlier slots' kernels execute)
            c_np = np.array([e.center      for e in chunk], dtype=np.float32)
            r_np = np.array([e.radii       for e in chunk], dtype=np.float32)
            R_np = np.array([e.orientation for e in chunk], dtype=np.float32)

            # Non-blocking upload — DMA runs concurrently with prior kernels
            ev_c = cl.enqueue_copy(q, self._slot_c[si], c_np, is_blocking=False)
            ev_r = cl.enqueue_copy(q, self._slot_r[si], r_np, is_blocking=False)
            ev_R = cl.enqueue_copy(q, self._slot_R[si], R_np, is_blocking=False)

            # SDF kernel waits for: eval pts + candidate uploads
            gK = _round_up(Kc, _WG)
            ev_sdf = self._k_sdf(
                q, (N, gK), (1, _WG),
                self._buf_pts_s,
                self._slot_c[si], self._slot_r[si], self._slot_R[si],
                self._slot_sdf[si],
                np.int32(N), np.int32(Kc), np.int32(d),
                wait_for=base_wait + [ev_c, ev_r, ev_R],
            )

            # Scoring kernel waits for SDF kernel
            gK_sc = _round_up(Kc, _WG)
            ev_score = self._k_combine_score(
                q, (gK_sc, _WG), (1, _WG),
                self._slot_sdf[si],
                self._buf_ex_sdf,
                self._slot_pc[si], self._slot_nc[si],
                np.int32(N), np.int32(Kc), np.int32(N_pool),
                np.float32(alpha), np.float32(threshold),
                np.int32(has_expert), np.int32(existing_count), np.int32(task_regression),
                wait_for=[ev_sdf],
            )

            # Non-blocking download of (K_slot,) int32 counts
            pc_arr = np.empty(Kc, dtype=np.int32)
            nc_arr = np.empty(Kc, dtype=np.int32)
            pc_arrs.append(pc_arr)
            nc_arrs.append(nc_arr)
            ev_dl_pc = cl.enqueue_copy(q, pc_arr, self._slot_pc[si],
                                        is_blocking=False, wait_for=[ev_score])
            ev_dl_nc = cl.enqueue_copy(q, nc_arr, self._slot_nc[si],
                                        is_blocking=False, wait_for=[ev_score])
            dl_events.extend([ev_dl_pc, ev_dl_nc])

        # Wait for all slot downloads to complete
        cl.wait_for_events(dl_events)

        pos_counts = np.concatenate(pc_arrs)  # (K,) int32
        neg_counts = np.concatenate(nc_arrs)  # (K,) int32
        return pos_counts, neg_counts


def batch_sdf(ellipsoids: list, points: np.ndarray) -> np.ndarray:
    """Evaluate K :class:`~src.sdf_engine.EllipsoidExpert` SDFs at N points on GPU.

    Returns a ``(N, K)`` float64 array.  Uses a module-level cached OpenCL
    context so repeated calls (e.g. the RANSAC inner loop) incur no
    re-initialisation cost — only per-call PCIe transfer overhead.

    :param ellipsoids: List of :class:`~src.sdf_engine.EllipsoidExpert` objects.
    :param points: ``(N, d)`` array (any float dtype; converted to float32).
    :return: ``(N, K)`` float64 SDF matrix.
    """
    return _OCLTrainingContext.get().batch_sdf(ellipsoids, points)


def fit_axis_aligned_candidates_gpu(
    seed_batches: np.ndarray,
    primitive_family: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit sphere or diagonal-ellipsoid candidate batches on OpenCL."""
    return _OCLTrainingContext.get().fit_axis_aligned_candidates(
        seed_batches, primitive_family,
    )


def batch_sdf_and_score(
    ellipsoids: list,
    pts_f32: np.ndarray,
    N_pool: int,
    ex_sdf_np: "np.ndarray | None",
    alpha: float,
    threshold: float,
    task_regression: bool,
    existing_count: int = 0,
) -> "tuple[np.ndarray, np.ndarray]":
    """Pipelined GPU scoring for RANSAC candidate selection.

    Equivalent to ``batch_sdf`` + softmin combination + thresholding + sum,
    but performed entirely on-device.  Uses :data:`_N_STAGES` async slots so
    AMD's DMA engine overlaps candidate uploads with kernel execution.

    :param ellipsoids: K candidate :class:`~src.sdf_engine.EllipsoidExpert`.
    :param pts_f32: ``(N_eval, d)`` float32 eval points.
    :param N_pool: First ``N_pool`` rows are positive pool points.
    :param ex_sdf_np: ``(N_eval,)`` current expert SDF, or ``None``.
    :param alpha: Softmin concentration.
    :param threshold: Capture threshold.
    :param task_regression: Use ``|sdf| < threshold`` if ``True``.
    :return: ``(pos_counts, neg_counts)`` — ``(K,)`` int32 arrays.
    """
    return _OCLTrainingContext.get().batch_sdf_and_score(
        ellipsoids, pts_f32, N_pool, ex_sdf_np, alpha, threshold,
        task_regression, existing_count,
    )


# ---------------------------------------------------------------------------
# GPUInferenceEngine
# ---------------------------------------------------------------------------

class GPUInferenceEngine:
    """Batch GPU inference: all classes scored in a single three-kernel pass.

    **Construction** uploads all ellipsoid parameters (centers, radii,
    orientation matrices) to device memory once.  The uploaded buffers are
    reused across every :meth:`class_sdfs` call.

    **Inference** transfers ``(N, d)`` float32 points to the device, runs
    the three kernels, and transfers only ``(N, C)`` float32 results back
    to host — O(N·(d + C)) bytes across PCIe regardless of how many
    ellipsoids the model contains.

    :param class_models: ``class_models[c]`` is a :class:`list` of
        :class:`~src.sdf_engine.Expert` objects for class ``c``.
        Identical structure to what :class:`~src.inference_engine.InferenceEngine`
        uses per class.
    :param alpha: Softmin concentration parameter (must match training).
    :param device: Optional :class:`pyopencl.Device`.  ``None`` auto-selects
        the GPU with the most compute units.
    """

    def __init__(
        self,
        class_models: list[list[Expert]],
        alpha: float = 2.0,
        device: cl.Device | None = None,
    ) -> None:
        self.alpha = float(alpha)
        if self.alpha <= 0.0:
            raise ValueError("GPUInferenceEngine requires alpha > 0.")
        self.n_classes = len(class_models)

        self._dev   = device or select_device()
        self._ctx   = cl.Context([self._dev])
        self._queue = cl.CommandQueue(self._ctx)
        prog        = cl.Program(self._ctx, _KERNEL_SRC).build(options=_BUILD_OPTIONS)

        # Cache kernel objects once — reusing avoids per-call retrieval overhead
        self._k_sdf     = cl.Kernel(prog, "batch_ellipsoid_sdf")
        self._k_nll     = cl.Kernel(prog, "batch_gaussian_nll")
        self._k_exp     = cl.Kernel(prog, "expert_softmin_csg")
        self._k_cls     = cl.Kernel(prog, "class_softmin")

        self._build(class_models)
        self._cached_N: int = -1   # tracks current intermediate buffer size

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build(self, class_models: list[list[Expert]]) -> None:
        """Flatten all ellipsoid parameters and upload static buffers."""
        pos_centers, pos_radii, pos_Rs, pos_class_indices = [], [], [], []
        neg_centers, neg_radii, neg_Rs = [], [], []

        expert_pos_starts: list[int] = []
        expert_pos_counts: list[int] = []
        expert_neg_starts: list[int] = []
        expert_neg_counts: list[int] = []

        class_exp_starts: list[int] = []
        class_exp_counts: list[int] = []

        k_pos = k_neg = 0

        for class_index, experts in enumerate(class_models):
            class_exp_starts.append(len(expert_pos_starts))
            for expert in experts:
                pos = [e for e in expert.ellipsoids if e.polarity > 0]
                neg = [e for e in expert.ellipsoids if e.polarity < 0]

                expert_pos_starts.append(k_pos)
                expert_pos_counts.append(len(pos))
                expert_neg_starts.append(k_neg)
                expert_neg_counts.append(len(neg))

                for e in pos:
                    pos_centers.append(e.center)
                    pos_radii.append(e.radii)
                    pos_Rs.append(e.orientation)
                    pos_class_indices.append(class_index)
                    k_pos += 1

                for e in neg:
                    neg_centers.append(e.center)
                    neg_radii.append(e.radii)
                    neg_Rs.append(e.orientation)
                    k_neg += 1

            class_exp_counts.append(len(experts))

        self._K_pos     = k_pos
        self._K_neg     = k_neg
        self._n_experts = len(expert_pos_starts)
        self._d         = len(pos_centers[0]) if pos_centers else 1

        # Store CPU-side index arrays for the softmin kernels
        self._pos_starts_np = np.array(expert_pos_starts, dtype=np.int32)
        self._pos_counts_np = np.array(expert_pos_counts, dtype=np.int32)
        self._neg_starts_np = np.array(expert_neg_starts, dtype=np.int32)
        self._neg_counts_np = np.array(expert_neg_counts, dtype=np.int32)
        self._exp_starts_np = np.array(class_exp_starts,  dtype=np.int32)
        self._exp_counts_np = np.array(class_exp_counts,  dtype=np.int32)
        self._pos_class_indices_np = np.array(pos_class_indices, dtype=np.int32)
        self._probabilistic_invalid_radii = bool(
            k_pos > 0 and (
                np.any(~np.isfinite(pos_radii))
                or np.any(np.asarray(pos_radii) <= 0.0)
            )
        )

        mf = cl.mem_flags

        def ro(arr, dtype=np.float32) -> cl.Buffer:
            a = np.asarray(arr, dtype=dtype)
            return cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)

        # Static ellipsoid parameter buffers (never change after build)
        self._buf_pos_centers = ro(pos_centers)
        self._buf_pos_radii   = ro(pos_radii)
        self._buf_pos_Rs      = ro(pos_Rs)

        if k_neg > 0:
            self._buf_neg_centers = ro(neg_centers)
            self._buf_neg_radii   = ro(neg_radii)
            self._buf_neg_Rs      = ro(neg_Rs)
        else:
            # Placeholder 1-element buffers so kernel args are always valid
            placeholder = np.zeros(1, dtype=np.float32)
            self._buf_neg_centers = ro(placeholder)
            self._buf_neg_radii   = ro(placeholder)
            self._buf_neg_Rs      = ro(placeholder)

        # Static index buffers for stages 2 and 3
        self._buf_pos_starts = ro(self._pos_starts_np, np.int32)
        self._buf_pos_counts = ro(self._pos_counts_np, np.int32)
        self._buf_neg_starts = ro(self._neg_starts_np, np.int32)
        self._buf_neg_counts = ro(self._neg_counts_np, np.int32)
        self._buf_exp_starts = ro(self._exp_starts_np, np.int32)
        self._buf_exp_counts = ro(self._exp_counts_np, np.int32)

        # Intermediate buffers — size depends on N; allocated on first call
        self._buf_sdf_pos    : cl.Buffer | None = None
        self._buf_sdf_neg    : cl.Buffer | None = None
        self._buf_expert_sdf : cl.Buffer | None = None
        self._buf_class_sdf  : cl.Buffer | None = None

    def _ensure_intermediate_buffers(self, N: int) -> None:
        """(Re-)allocate device-side intermediate buffers if N changed."""
        if N == self._cached_N:
            return
        mf = cl.mem_flags
        E, C = self._n_experts, self.n_classes

        def rw(n_elems: int) -> cl.Buffer:
            return cl.Buffer(self._ctx, mf.READ_WRITE, size=n_elems * 4)  # float32

        # Include points buffer here so class_sdfs() never allocates per call.
        self._buf_pts        = rw(N * self._d)
        self._buf_sdf_pos    = rw(N * self._K_pos)
        self._buf_sdf_neg    = rw(max(N * self._K_neg, 1))
        self._buf_expert_sdf = rw(N * E)
        self._buf_class_sdf  = rw(N * C)
        self._cached_N       = N

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def class_sdfs(self, points: np.ndarray) -> np.ndarray:
        """Compute fused SDF for every class at every query point.

        :param points: ``(N, d)`` numpy array (any float dtype; converted to
            float32 internally).
        :return: ``(N, C)`` float64 array.  ``result[n, c]`` is the Softmin
            fused SDF of class ``c`` at point ``n``.  Argmin over axis 1
            gives the predicted class label.
        """
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != self._d:
            raise ValueError(f"Expected points with shape (N, {self._d}), got {points.shape}.")
        if len(points) == 0:
            return np.empty((0, self.n_classes), dtype=np.float64)
        pts_f32 = np.ascontiguousarray(points, dtype=np.float32)
        N = pts_f32.shape[0]
        self._ensure_intermediate_buffers(N)

        queue = self._queue
        E     = self._n_experts
        C     = self.n_classes
        K_pos = self._K_pos
        K_neg = self._K_neg
        d     = self._d
        a     = np.float32(self.alpha)

        # Upload points into pre-allocated device buffer (no per-call allocation)
        cl.enqueue_copy(queue, self._buf_pts, pts_f32)

        # ------------------------------------------------------------------
        # Stage 1: batch_ellipsoid_sdf  →  buf_sdf_pos  (N × K_pos)
        # Global size padded to _WG multiples so every wavefront is full.
        # ------------------------------------------------------------------
        gK_pos = _round_up(K_pos, _WG)
        self._k_sdf(
            queue, (N, gK_pos), (1, _WG),
            self._buf_pts,
            self._buf_pos_centers, self._buf_pos_radii, self._buf_pos_Rs,
            self._buf_sdf_pos,
            np.int32(N), np.int32(K_pos), np.int32(d),
        )

        if K_neg > 0:
            gK_neg = _round_up(K_neg, _WG)
            self._k_sdf(
                queue, (N, gK_neg), (1, _WG),
                self._buf_pts,
                self._buf_neg_centers, self._buf_neg_radii, self._buf_neg_Rs,
                self._buf_sdf_neg,
                np.int32(N), np.int32(K_neg), np.int32(d),
            )

        # ------------------------------------------------------------------
        # Stage 2: expert_softmin_csg  →  buf_expert_sdf  (N × E)
        # ------------------------------------------------------------------
        gE = _round_up(E, _WG)
        self._k_exp(
            queue, (N, gE), (1, _WG),
            self._buf_sdf_pos,
            self._buf_sdf_neg,
            self._buf_expert_sdf,
            self._buf_pos_starts, self._buf_pos_counts,
            self._buf_neg_starts, self._buf_neg_counts,
            np.int32(N), np.int32(K_pos), np.int32(K_neg), np.int32(E), a,
        )

        # ------------------------------------------------------------------
        # Stage 3: class_softmin  →  buf_class_sdf  (N × C)
        # ------------------------------------------------------------------
        gC = _round_up(C, _WG)
        self._k_cls(
            queue, (N, gC), (1, _WG),
            self._buf_expert_sdf,
            self._buf_class_sdf,
            self._buf_exp_starts, self._buf_exp_counts,
            np.int32(N), np.int32(E), np.int32(C), a,
        )

        # Transfer result to host (blocking copy — no separate finish needed)
        result_f32 = np.empty(N * C, dtype=np.float32)
        cl.enqueue_copy(queue, result_f32, self._buf_class_sdf)
        queue.finish()

        return result_f32.reshape(N, C).astype(np.float64)

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Return predicted class index (argmin SDF) for each query point.

        :param points: ``(N, d)`` numpy array.
        :return: ``(N,)`` int array of class indices.
        """
        return self.class_sdfs(points).argmin(axis=1)

    def class_nlls(
        self,
        points: np.ndarray,
        covariance_temperature: float | np.ndarray = 1.0,
    ) -> np.ndarray:
        """Return hierarchical uniform Gaussian-mixture NLLs per class."""
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != self._d:
            raise ValueError(f"Expected points with shape (N, {self._d}), got {points.shape}.")
        if self._K_neg > 0:
            raise ValueError("probabilistic inference does not support subtractive primitives")
        if np.any(self._exp_counts_np == 0) or np.any(self._pos_counts_np == 0):
            raise ValueError("every class and expert must define a probability model")
        if self._probabilistic_invalid_radii:
            raise ValueError("primitive radii must be finite and positive")
        temperatures = np.asarray(covariance_temperature, dtype=np.float32)
        if temperatures.ndim == 0:
            temperatures = np.full(self.n_classes, temperatures.item(), dtype=np.float32)
        if temperatures.shape != (self.n_classes,):
            raise ValueError("covariance_temperature must be scalar or class-width")
        if np.any(~np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
            raise ValueError("covariance_temperature must be finite and positive")
        if len(points) == 0:
            return np.empty((0, self.n_classes), dtype=np.float64)
        pts_f32 = np.ascontiguousarray(points, dtype=np.float32)
        sample_count = len(points)
        self._ensure_intermediate_buffers(sample_count)
        queue = self._queue
        cl.enqueue_copy(queue, self._buf_pts, pts_f32)
        primitive_temperatures = np.ascontiguousarray(
            temperatures[self._pos_class_indices_np], dtype=np.float32,
        )
        temperature_buffer = cl.Buffer(
            self._ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=primitive_temperatures,
        )

        self._k_nll(
            queue,
            (sample_count, _round_up(self._K_pos, _WG)),
            (1, _WG),
            self._buf_pts,
            self._buf_pos_centers,
            self._buf_pos_radii,
            self._buf_pos_Rs,
            self._buf_sdf_pos,
            np.int32(sample_count),
            np.int32(self._K_pos),
            np.int32(self._d),
            temperature_buffer,
        )
        self._k_exp(
            queue,
            (sample_count, _round_up(self._n_experts, _WG)),
            (1, _WG),
            self._buf_sdf_pos,
            self._buf_sdf_neg,
            self._buf_expert_sdf,
            self._buf_pos_starts,
            self._buf_pos_counts,
            self._buf_neg_starts,
            self._buf_neg_counts,
            np.int32(sample_count),
            np.int32(self._K_pos),
            np.int32(0),
            np.int32(self._n_experts),
            np.float32(1.0),
        )
        self._k_cls(
            queue,
            (sample_count, _round_up(self.n_classes, _WG)),
            (1, _WG),
            self._buf_expert_sdf,
            self._buf_class_sdf,
            self._buf_exp_starts,
            self._buf_exp_counts,
            np.int32(sample_count),
            np.int32(self._n_experts),
            np.int32(self.n_classes),
            np.float32(1.0),
        )
        result = np.empty(sample_count * self.n_classes, dtype=np.float32)
        cl.enqueue_copy(queue, result, self._buf_class_sdf)
        queue.finish()
        return result.reshape(sample_count, self.n_classes).astype(np.float64)

    @property
    def device_name(self) -> str:
        """OpenCL device name (e.g. ``'gfx1201'`` for the RX 9070 XT)."""
        return self._dev.name
