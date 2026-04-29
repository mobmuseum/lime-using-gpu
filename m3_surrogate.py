import time
import numpy as np
import cupy as cp
from lime_common import BLOCK_SIZE

_GD_KERNELS_CODE = r"""

extern "C" __global__ void compute_residuals(
    const float *X,      // [N, F] perturbed samples
    const float *beta,   // [F]    current weight vector
    const float *y,      // [N]    target predictions
          float *resid,  // [N]    output residuals
    int N, int F)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float dot = 0.0f;
    for (int f = 0; f < F; f++)
        dot += X[i * F + f] * beta[f];

    resid[i] = dot - y[i];   // positive → over-predicted, negative → under-predicted
}


extern "C" __global__ void weighted_gradient(
    const float *X,      // [N, F]
    const float *resid,  // [N]
    const float *w,      // [N]    kernel importance weights
          float *grad,   // [F]    output gradient (one value per feature)
    int N, int F)
{
    // Each block is responsible for exactly one feature dimension.
    int f   = blockIdx.x;
    int tid = threadIdx.x;
    if (f >= F) return;

    extern __shared__ float shmem[];   // blockDim.x floats

    float acc = 0.0f;
    // Stride over samples — each thread accumulates a partial sum
    for (int i = tid; i < N; i += blockDim.x)
        acc += w[i] * resid[i] * X[i * F + f];

    shmem[tid] = acc;
    __syncthreads();

    // Standard power-of-2 tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shmem[tid] += shmem[tid + stride];
        __syncthreads();
    }

    // Thread 0 writes the final gradient for this feature
    if (tid == 0)
        grad[f] = shmem[0] / (float)N;   // normalize by N to keep LR scale-independent
}


extern "C" __global__ void update_params(
          float *beta,   // [F]  in-place update
    const float *grad,   // [F]
    float lr,
    int F)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;
    beta[f] -= lr * grad[f];
}


extern "C" __global__ void weighted_mse_loss(
    const float *resid,  // [N]
    const float *w,      // [N]
          float *loss,   // [1]  scalar output
    int N)
{
    extern __shared__ float shmem[];
    int tid = threadIdx.x;

    float acc = 0.0f;
    for (int i = tid; i < N; i += blockDim.x)
        acc += w[i] * resid[i] * resid[i];

    shmem[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shmem[tid] += shmem[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        loss[0] = shmem[0] / (float)N;
}

"""

# Compile once at module load
_GD_MODULE     = cp.RawModule(code=_GD_KERNELS_CODE)
_k_residuals   = _GD_MODULE.get_function("compute_residuals")
_k_gradient    = _GD_MODULE.get_function("weighted_gradient")
_k_update      = _GD_MODULE.get_function("update_params")
_k_mse         = _GD_MODULE.get_function("weighted_mse_loss")

def solve_closed_form(
    d_X: cp.ndarray,   # (N, F)  perturbed samples — already on GPU
    d_y: cp.ndarray,   # (N,)    black-box predictions — already on GPU
    d_w: cp.ndarray,   # (N,)    kernel weights — already on GPU
    ridge: float = 1e-4,  # L2 regularisation to prevent singular matrix
) -> cp.ndarray:

    N, F = d_X.shape

    d_sqrt_w = cp.sqrt(d_w).reshape(-1, 1)   # (N, 1) broadcast
    d_Xw = d_X * d_sqrt_w                    # (N, F) — weighted design matrix
    d_yw = d_y * d_sqrt_w.reshape(-1)         # (N,)  — weighted targets

    d_A = d_Xw.T @ d_Xw                                  # cuBLAS gemm
    d_A += ridge * cp.eye(F, dtype=cp.float32)            # Tikhonov regularisation
    d_b = d_Xw.T @ d_yw                                   # cuBLAS gemv

    d_beta = cp.linalg.solve(d_A, d_b)   # (F,)  stays on GPU

    cp.cuda.Device().synchronize()
    return d_beta.astype(cp.float32)

def solve_gradient_descent(
    d_X:       cp.ndarray,   # (N, F)
    d_y:       cp.ndarray,   # (N,)
    d_w:       cp.ndarray,   # (N,)
    lr:        float = 1e-2,
    n_iters:   int   = 1000,
    verbose:   bool  = False,
    log_every: int   = 100,
) -> tuple:

    N, F = d_X.shape

    d_col_norm = cp.linalg.norm(d_X, axis=0) + 1e-8   # (F,) — column norms
    d_X_sc     = d_X / d_col_norm[cp.newaxis, :]       # (N, F) scaled copy

    # Ensure contiguous float32 for the raw CUDA kernels
    d_X_f = cp.ascontiguousarray(d_X_sc, dtype=cp.float32)
    d_y_f = cp.ascontiguousarray(d_y,    dtype=cp.float32)
    d_w_f = cp.ascontiguousarray(d_w,    dtype=cp.float32)

    d_beta  = cp.zeros(F, dtype=cp.float32)   # start at zero in scaled space
    d_resid = cp.empty(N, dtype=cp.float32)
    d_grad  = cp.zeros(F, dtype=cp.float32)
    d_loss  = cp.zeros(1, dtype=cp.float32)

    blocks_N  = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    blocks_F  = F                                # one block per feature
    threads   = BLOCK_SIZE
    shmem_gd  = threads * 4

    loss_history = []

    for it in range(n_iters):
        _k_residuals(
            (blocks_N,), (threads,),
            (d_X_f, d_beta, d_y_f, d_resid, np.int32(N), np.int32(F))
        )

        _k_gradient(
            (blocks_F,), (threads,),
            (d_X_f, d_resid, d_w_f, d_grad, np.int32(N), np.int32(F)),
            shared_mem=shmem_gd
        )

        blocks_upd = (F + threads - 1) // threads
        _k_update(
            (blocks_upd,), (threads,),
            (d_beta, d_grad, np.float32(lr), np.int32(F))
        )

        if verbose and (it + 1) % log_every == 0:
            _k_mse(
                (1,), (threads,),
                (d_resid, d_w_f, d_loss, np.int32(N)),
                shared_mem=shmem_gd
            )
            cp.cuda.Device().synchronize()
            loss_val = float(d_loss[0])
            loss_history.append(loss_val)
            print(f"  iter {it+1:5d}/{n_iters}  |  Weighted MSE: {loss_val:.6f}")
        elif (it + 1) % log_every == 0:
            _k_mse(
                (1,), (threads,),
                (d_resid, d_w_f, d_loss, np.int32(N)),
                shared_mem=shmem_gd
            )
            cp.cuda.Device().synchronize()
            loss_history.append(float(d_loss[0]))

    cp.cuda.Device().synchronize()

    d_beta = d_beta / d_col_norm

    return d_beta.astype(cp.float32), loss_history



def train_surrogate_gpu(
    d_X:     cp.ndarray,         # (N, F) already on GPU
    d_y:     cp.ndarray,         # (N,)   already on GPU
    d_w:     cp.ndarray,         # (N,)   already on GPU
    method:  str   = "closed",   # "closed" or "gd"
    ridge:   float = 1e-4,       # ridge regularisation (closed-form only)
    lr:      float = 1e-2,       # learning rate (GD only)
    n_iters: int   = 1000,       # number of GD iterations
    verbose: bool  = False,
) -> tuple:

    d_w = d_w / (d_w.max() + 1e-8)

    if method == "closed":
        d_beta = solve_closed_form(d_X, d_y, d_w, ridge=ridge)

        d_resid = d_X @ d_beta - d_y
        d_loss  = cp.zeros(1, dtype=cp.float32)
        N       = d_X.shape[0]
        _k_mse(
            (1,), (BLOCK_SIZE,),
            (d_resid.astype(cp.float32), d_w.astype(cp.float32),
             d_loss, np.int32(N)),
            shared_mem=BLOCK_SIZE * 4
        )
        cp.cuda.Device().synchronize()
        loss_history = [float(d_loss[0])]

    elif method == "gd":
        d_beta, loss_history = solve_gradient_descent(
            d_X, d_y, d_w,
            lr=lr, n_iters=n_iters, verbose=verbose
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'closed' or 'gd'.")

    return d_beta, loss_history


def train_surrogate_cpu(
    X:     np.ndarray,   # (N, F)
    y:     np.ndarray,   # (N,)
    w:     np.ndarray,   # (N,)
    ridge: float = 1e-4,
) -> np.ndarray:

    w = w / (w.max() + 1e-8)
    sqrt_w = np.sqrt(w).reshape(-1, 1)
    Xw = X * sqrt_w
    yw = y * sqrt_w.reshape(-1)
    A  = Xw.T @ Xw + ridge * np.eye(X.shape[1], dtype=np.float32)
    b  = Xw.T @ yw
    return np.linalg.solve(A, b).astype(np.float32)


def benchmark_surrogate_gpu(
    d_X:     cp.ndarray,
    d_y:     cp.ndarray,
    d_w:     cp.ndarray,
    method:  str   = "closed",
    lr:      float = 1e-2,
    n_iters: int   = 1000,
    n_repeats: int = 5,
) -> dict:

    start_ev = cp.cuda.Event()
    end_ev   = cp.cuda.Event()

    # One warmup run to avoid measuring JIT on the first timed call
    train_surrogate_gpu(d_X, d_y, d_w, method=method, lr=lr, n_iters=n_iters)

    start_ev.record()
    for _ in range(n_repeats):
        train_surrogate_gpu(d_X, d_y, d_w, method=method, lr=lr, n_iters=n_iters)
    end_ev.record()
    end_ev.synchronize()

    gpu_ms = cp.cuda.get_elapsed_time(start_ev, end_ev) / n_repeats
    return {"gpu_ms": gpu_ms}


def benchmark_surrogate_cpu(
    X:        np.ndarray,
    y:        np.ndarray,
    w:        np.ndarray,
    n_repeats: int = 5,
) -> dict:
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        train_surrogate_cpu(X, y, w)
        times.append((time.perf_counter() - t0) * 1000)
    return {"cpu_ms": float(np.mean(times)), "std_ms": float(np.std(times))}
