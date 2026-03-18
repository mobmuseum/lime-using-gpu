import cupy as cp
import numpy as np
import time
from lime_common import LogisticModel, WeightBuffer, PerturbationBuffer

# Warm up with a tiny dummy call so benchmark timings reflect 
# steady-state GPU performance, not compile time.
def _warmup():
    _x = cp.ones((4, 4), dtype=cp.float32)
    _w = cp.ones(4, dtype=cp.float32)
    _ = _x @ _w
    cp.cuda.Device().synchronize()

_warmup()

def model_create(h_weights: np.ndarray, bias: float = 0.0) -> LogisticModel:
    m = LogisticModel(n_features=len(h_weights), bias=bias)
    m.d_weights = cp.asarray(h_weights, dtype=cp.float32)
    return m

# Run logistic regression inference on all perturbed samples.
def inference_run(buf: PerturbationBuffer, model: LogisticModel) -> WeightBuffer:
    logits = buf.d_samples @ model.d_weights + model.bias
    preds  = (1.0 / (1.0 + cp.exp(-logits))).astype(cp.float32)

    wb = WeightBuffer(n_samples=buf.n_samples)
    wb.d_predictions = preds
    wb.d_distances   = cp.empty(buf.n_samples, dtype=cp.float32)
    wb.d_weights     = cp.empty(buf.n_samples, dtype=cp.float32)

    cp.cuda.Device().synchronize()
    return wb



def inference_cpu_reference(
    h_samples: np.ndarray,
    h_weights: np.ndarray,
    bias:      float,
) -> np.ndarray:
    logits = h_samples @ h_weights + bias
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)



def validate_inference(
    wb:        WeightBuffer,
    buf:       PerturbationBuffer,
    h_weights: np.ndarray,
    bias:      float,
) -> dict:
    h_samples   = cp.asnumpy(buf.d_samples)
    h_preds_gpu = cp.asnumpy(wb.d_predictions)
    h_preds_cpu = inference_cpu_reference(h_samples, h_weights, bias)
    max_err = float(np.abs(h_preds_gpu - h_preds_cpu).max())
    return {
        "max_abs_error": max_err,
        "status": "PASS" if max_err < 1e-4 else f"FAIL (err={max_err:.2e})",
    }


def benchmark_inference(buf: PerturbationBuffer, model: LogisticModel,
                        n_repeats: int = 10) -> dict:
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    for _ in range(n_repeats):
        inference_run(buf, model)
    end.record(); end.synchronize()
    gpu_ms = cp.cuda.get_elapsed_time(start, end) / n_repeats

    # CPU
    h_samples = cp.asnumpy(buf.d_samples)
    h_weights = cp.asnumpy(model.d_weights)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        inference_cpu_reference(h_samples, h_weights, model.bias)
    cpu_ms = (time.perf_counter() - t0) / n_repeats * 1000

    return {"gpu_ms": gpu_ms, "cpu_ms": cpu_ms, "speedup": cpu_ms / gpu_ms}
