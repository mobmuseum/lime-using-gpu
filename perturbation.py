import cupy as cp
import numpy as np
import time
from lime_common import PerturbationBuffer, KEEP_PROB, BLOCK_SIZE, adaptive_kernel_width

_PERTURBATION_KERNELS = cp.RawModule(code=r"""
// Kernel 1: generate binary masks using Xorshift32 (one state per thread)
extern "C" __global__ void generate_mask_kernel(
    int   *d_masks,
    int    n_samples,
    int    n_features,
    float  keep_prob,
    unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_samples * n_features;
    if (idx >= total) return;

    // Simple thread-local Xorshift32 PRNG
    unsigned int state = (unsigned int)(seed ^ (idx * 0x27d4eb2d));
    if (state == 0) state = 1;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;

    // Convert to float in [0, 1)
    float r = (float)(state & 0x00FFFFFF) / (float)0x01000000;
    d_masks[idx] = (r < keep_prob) ? 1 : 0;
}

// Kernel 2: apply mask — sample[s,f] = original[f] * mask[s,f]
extern "C" __global__ void apply_mask_kernel(
    const float *d_original,
    const int   *d_masks,
    float       *d_samples,
    int          n_samples,
    int          n_features)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;   // feature index
    int s = blockIdx.y;                               // sample index
    if (s >= n_samples || f >= n_features) return;

    int idx = s * n_features + f;
    d_samples[idx] = d_original[f] * (float)d_masks[idx];
}
""")

_generate_mask = _PERTURBATION_KERNELS.get_function("generate_mask_kernel")
_apply_mask    = _PERTURBATION_KERNELS.get_function("apply_mask_kernel")


def perturbation_generate(
    h_original:  np.ndarray,
    n_samples:   int,
    n_features:  int,
    keep_prob:   float = KEEP_PROB,
    seed:        int   = 12345,
) -> PerturbationBuffer:

    buf = PerturbationBuffer(n_samples=n_samples, n_features=n_features)

    d_original = cp.asarray(h_original, dtype=cp.float32)
    buf.d_masks   = cp.empty((n_samples, n_features), dtype=cp.int32)
    buf.d_samples = cp.empty((n_samples, n_features), dtype=cp.float32)

    total   = n_samples * n_features
    threads = BLOCK_SIZE
    blocks  = (total + threads - 1) // threads

    # Kernel 1 — masks
    _generate_mask(
        (blocks,), (threads,),
        (buf.d_masks, np.int32(n_samples), np.int32(n_features),
         np.float32(keep_prob), np.uint64(seed))
    )

    # Kernel 2 — apply (2-D grid: x=feature-blocks, y=samples)
    bx = (n_features + threads - 1) // threads
    _apply_mask(
        (bx, n_samples), (threads,),
        (d_original, buf.d_masks, buf.d_samples,
         np.int32(n_samples), np.int32(n_features))
    )

    cp.cuda.Device().synchronize()
    return buf



def perturbation_cpu_reference(
    h_original: np.ndarray,
    n_samples:  int,
    n_features: int,
    keep_prob:  float = KEEP_PROB,
    seed:       int   = 99999, # different seed → independent RNG
) -> tuple:
    rng     = np.random.default_rng(seed)
    masks   = (rng.random((n_samples, n_features)) < keep_prob).astype(np.int32)
    samples = h_original[np.newaxis, :] * masks.astype(np.float32)
    return samples, masks


def validate_perturbation(buf: PerturbationBuffer, h_original: np.ndarray) -> dict:
    h_masks   = cp.asnumpy(buf.d_masks)
    h_samples = cp.asnumpy(buf.d_samples)

    keep_rate = h_masks.mean()

    # Every sample value must equal original * mask exactly (up to float32 precision)
    expected  = h_original[np.newaxis, :] * h_masks.astype(np.float32)
    max_err   = np.abs(h_samples - expected).max()
    mask_ok   = max_err < 1e-5

    return {
        "keep_rate":  keep_rate,
        "mask_apply": "PASS ✓" if mask_ok else f"FAIL ✗ (max_err={max_err:.2e})",
    }



def benchmark_perturbation(h_original, n_samples, n_features, n_repeats=5):
    # GPU
    start_gpu = cp.cuda.Event(); end_gpu = cp.cuda.Event()
    start_gpu.record()
    for _ in range(n_repeats):
        perturbation_generate(h_original, n_samples, n_features)
    end_gpu.record(); end_gpu.synchronize()
    gpu_ms = cp.cuda.get_elapsed_time(start_gpu, end_gpu) / n_repeats
    
    # CPU
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        perturbation_cpu_reference(h_original, n_samples, n_features)
    cpu_ms = (time.perf_counter() - t0) / n_repeats * 1000

    return {"gpu_ms": gpu_ms, "cpu_ms": cpu_ms, "speedup": cpu_ms / gpu_ms}

