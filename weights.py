import cupy as cp
import numpy as np
import time
from lime_common import (
    WeightBuffer, PerturbationBuffer, DistanceMetric,
    adaptive_kernel_width, BLOCK_SIZE
)

_WEIGHT_KERNELS_CODE = r"""
// Euclidean distance
extern "C" __global__ void euclidean_distance_kernel(
    const float *d_original,
    const float *d_samples,
    float       *d_distances,
    int          n_samples,
    int          n_features)
{
    extern __shared__ float shmem[];
    int s   = blockIdx.x;
    int tid = threadIdx.x;
    if (s >= n_samples) return;

    const float *row = d_samples + s * n_features;
    float partial = 0.0f;
    for (int f = tid; f < n_features; f += blockDim.x) {
        float diff = d_original[f] - row[f];
        partial += diff * diff;
    }
    shmem[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) d_distances[s] = sqrtf(shmem[0]);
}

// Cosine distance
// Uses 3 sections of shared memory: [dot | norm_a | norm_b]
extern "C" __global__ void cosine_distance_kernel(
    const float *d_original,
    const float *d_samples,
    float       *d_distances,
    int          n_samples,
    int          n_features)
{
    extern __shared__ float shmem[];
    float *sh_dot = shmem;
    float *sh_na  = shmem + blockDim.x;
    float *sh_nb  = shmem + 2 * blockDim.x;

    int s   = blockIdx.x;
    int tid = threadIdx.x;
    if (s >= n_samples) return;

    const float *row = d_samples + s * n_features;
    float dot = 0.f, na = 0.f, nb = 0.f;
    for (int f = tid; f < n_features; f += blockDim.x) {
        dot += d_original[f] * row[f];
        na  += d_original[f] * d_original[f];
        nb  += row[f] * row[f];
    }
    sh_dot[tid] = dot; sh_na[tid] = na; sh_nb[tid] = nb;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_dot[tid] += sh_dot[tid + stride];
            sh_na [tid] += sh_na [tid + stride];
            sh_nb [tid] += sh_nb [tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float denom = sqrtf(sh_na[0]) * sqrtf(sh_nb[0]);
        float cos_sim = (denom > 1e-8f) ? sh_dot[0] / denom : 0.0f;
        cos_sim = fminf(1.0f, fmaxf(-1.0f, cos_sim));   // numerical clamp
        d_distances[s] = 1.0f - cos_sim;
    }
}

// Exponential kernel weights
extern "C" __global__ void exponential_kernel_weight(
    const float *d_distances,
    float       *d_weights,
    int          n_samples,
    float        sigma)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_samples) return;
    float d  = d_distances[s];
    d_weights[s] = expf(-(d * d) / (2.0f * sigma * sigma));
}
"""
_WEIGHT_KERNELS = cp.RawModule(code=_WEIGHT_KERNELS_CODE)


_euclid_kernel  = _WEIGHT_KERNELS.get_function("euclidean_distance_kernel")
# Pre-compile all three kernels so benchmark timings exclude JIT overhead.
def _warmup():
    try:
        _d = cp.zeros(4, dtype=cp.float32)
        euclidean_distance_kernel = _WEIGHT_KERNELS.get_function("euclidean_distance_kernel")
        cp.cuda.Device().synchronize()
    except Exception:
        pass

_warmup()
_cosine_kernel  = _WEIGHT_KERNELS.get_function("cosine_distance_kernel")
_kernel_weight  = _WEIGHT_KERNELS.get_function("exponential_kernel_weight")


def weights_compute(
    wb:         WeightBuffer,
    buf:        PerturbationBuffer,
    h_original: np.ndarray,
    metric:     str   = DistanceMetric.EUCLIDEAN,
    sigma:      float = None,
) -> None:
    if sigma is None:
        sigma = adaptive_kernel_width(buf.n_features, metric)
    n_samples  = buf.n_samples
    n_features = buf.n_features

    d_original = cp.asarray(h_original, dtype=cp.float32)

    # Round threads to next power-of-2
    threads = 1
    while threads < n_features and threads < BLOCK_SIZE:
        threads <<= 1
    threads = min(threads, BLOCK_SIZE)

    if metric == DistanceMetric.COSINE:
        shmem = 3 * threads * 4   # 3 arrays × sizeof(float)
        _cosine_kernel(
            (n_samples,), (threads,),
            (d_original, buf.d_samples, wb.d_distances,
             np.int32(n_samples), np.int32(n_features)),
            shared_mem=shmem
        )
    else:
        shmem = threads * 4
        _euclid_kernel(
            (n_samples,), (threads,),
            (d_original, buf.d_samples, wb.d_distances,
             np.int32(n_samples), np.int32(n_features)),
            shared_mem=shmem
        )

    # Kernel weights — trivially parallel
    wblocks = (n_samples + BLOCK_SIZE - 1) // BLOCK_SIZE
    _kernel_weight(
        (wblocks,), (BLOCK_SIZE,),
        (wb.d_distances, wb.d_weights, np.int32(n_samples), np.float32(sigma))
    )
    cp.cuda.Device().synchronize()


def weights_cpu_reference(
    h_original: np.ndarray,
    h_samples:  np.ndarray,
    metric:     str   = DistanceMetric.EUCLIDEAN,
    sigma:      float = None,
) -> tuple:
    if sigma is None:
        sigma = adaptive_kernel_width(h_original.shape[0], metric)
    if metric == DistanceMetric.COSINE:
        dot  = h_samples @ h_original
        na   = np.linalg.norm(h_original)
        nb   = np.linalg.norm(h_samples, axis=1)
        denom = na * nb
        cos_sim = np.where(denom > 1e-8, dot / denom, 0.0)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        dists = (1.0 - cos_sim).astype(np.float32)
    else:
        diff  = h_samples - h_original[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=1).astype(np.float32)

    weights = np.exp(-(dists ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    return dists, weights

def validate_weights(
    wb:         WeightBuffer,
    buf:        PerturbationBuffer,
    h_original: np.ndarray,
    metric:     str   = DistanceMetric.EUCLIDEAN,
    sigma:      float = None,
) -> dict:
    if sigma is None:
        sigma = adaptive_kernel_width(buf.n_features, metric)
    h_samples   = cp.asnumpy(buf.d_samples)
    h_dist_gpu  = cp.asnumpy(wb.d_distances)
    h_wt_gpu    = cp.asnumpy(wb.d_weights)

    h_dist_cpu, h_wt_cpu = weights_cpu_reference(h_original, h_samples, metric, sigma)

    dist_err = float(np.abs(h_dist_gpu - h_dist_cpu).max())
    wt_err   = float(np.abs(h_wt_gpu   - h_wt_cpu  ).max())

    return {
        "dist_max_err":   dist_err,
        "dist_status":    "PASS ✓" if dist_err < 1e-4 else f"FAIL ✗ ({dist_err:.2e})",
        "weight_max_err": wt_err,
        "weight_status":  "PASS ✓" if wt_err   < 1e-4 else f"FAIL ✗ ({wt_err:.2e})",
    }

def benchmark_weights(wb, buf, h_original, metric=DistanceMetric.EUCLIDEAN, n_repeats=5):
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    for _ in range(n_repeats):
        weights_compute(wb, buf, h_original, metric)
    end.record(); end.synchronize()
    gpu_ms = cp.cuda.get_elapsed_time(start, end) / n_repeats

    h_samples = cp.asnumpy(buf.d_samples)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        weights_cpu_reference(h_original, h_samples, metric)
    cpu_ms = (time.perf_counter() - t0) / n_repeats * 1000

    return {"gpu_ms": gpu_ms, "cpu_ms": cpu_ms, "speedup": cpu_ms / gpu_ms}

