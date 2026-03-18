import argparse
import sys
import time

import cupy as cp
import numpy as np

try:
    cp.cuda.Device(0).use()
except cp.cuda.runtime.CUDARuntimeError as exc:
    sys.exit(f"[ERROR] No CUDA GPU detected: {exc}")

from lime_common   import DistanceMetric, adaptive_kernel_width
from perturbation  import (perturbation_generate, validate_perturbation,
                           benchmark_perturbation)
from inference     import (model_create, inference_run, validate_inference,
                           benchmark_inference)
from weights       import (weights_compute, validate_weights,
                           benchmark_weights)


def banner(title: str) -> None:
    line = "=" * 58
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def row(label: str, value, width: int = 42) -> None:
    print(f"  {label:<{width}} {value}")


def warmup_gpu(n_features: int = 64) -> None:
    rng    = np.random.default_rng(0)
    orig   = rng.random(n_features).astype(np.float32)
    wts    = (rng.random(n_features) - 0.5).astype(np.float32)
    pbuf   = perturbation_generate(orig, 64, n_features, seed=1)
    model  = model_create(wts, 0.0)
    wb     = inference_run(pbuf, model)
    weights_compute(wb, pbuf, orig)
    cp.cuda.Device().synchronize()

def run_pipeline(n_samples: int, n_features: int, metric: str, seed: int = 42):

    sigma = adaptive_kernel_width(n_features, metric)

    # GPU info
    dev  = cp.cuda.Device(0)
    prop = cp.cuda.runtime.getDeviceProperties(dev.id)
    banner("GPU Information")
    row("Device name:",  prop["name"].decode())
    row("Total memory:", f"{prop['totalGlobalMem'] / 1e9:.1f} GB")
    row("SM count:",     prop["multiProcessorCount"])
    print()
    print(f"  Samples   : {n_samples}")
    print(f"  Features  : {n_features}")
    print(f"  Metric    : {metric}")
    sigma_desc = "cosine: fixed 0.25" if metric == "cosine" else f"euclidean: 0.75 * sqrt({n_features})"
    print(f"  sigma     : {sigma:.4f}  ({sigma_desc})")

    # Warmup
    print("\n  [Warming up GPU kernels...]")
    warmup_gpu(n_features)
    print("  [Done]\n")

    # Synthetic inputs
    rng        = np.random.default_rng(seed)
    h_original = rng.random(n_features).astype(np.float32)
    h_weights  = (rng.random(n_features) - 0.5).astype(np.float32)
    bias       = 0.1

    banner("MODULE 1 — Perturbation Generation")

    bench_p = benchmark_perturbation(h_original, n_samples, n_features)
    row("GPU perturbation time:", f"{bench_p['gpu_ms']:.3f} ms")
    row("CPU perturbation time:", f"{bench_p['cpu_ms']:.3f} ms")
    row("Speedup (perturbation):", f"{bench_p['speedup']:.2f}x")

    pbuf = perturbation_generate(h_original, n_samples, n_features)
    v_p  = validate_perturbation(pbuf, h_original)
    row("Mask keep rate:",   f"{v_p['keep_rate']:.4f}  (expected ~0.5000)")
    row("Mask application:", v_p["mask_apply"])

    banner("MODULE 2 — Logistic Regression Inference")
    row("Method:", "cuBLAS matmul (d_samples @ d_weights + bias)")

    model   = model_create(h_weights, bias)
    bench_i = benchmark_inference(pbuf, model)
    row("GPU inference time:", f"{bench_i['gpu_ms']:.3f} ms")
    row("CPU inference time:", f"{bench_i['cpu_ms']:.3f} ms")
    row("Speedup (inference):", f"{bench_i['speedup']:.2f}x")

    wb    = inference_run(pbuf, model)
    v_inf = validate_inference(wb, pbuf, h_weights, bias)
    row("Max abs error (predictions):", f"{v_inf['max_abs_error']:.2e}")
    row("Inference validation:", v_inf["status"])

    banner("MODULE 3 — Distance & Kernel Weights")
    sigma_desc = "cosine: fixed 0.25" if metric == "cosine" else f"euclidean: 0.75 * sqrt({n_features})"
    row("Sigma (kernel bandwidth):", f"{sigma:.4f}  ({sigma_desc})")

    bench_w = benchmark_weights(wb, pbuf, h_original, metric)
    row("GPU weights time:", f"{bench_w['gpu_ms']:.3f} ms")
    row("CPU weights time:", f"{bench_w['cpu_ms']:.3f} ms")
    row("Speedup (weights):", f"{bench_w['speedup']:.2f}x")

    weights_compute(wb, pbuf, h_original, metric, sigma)
    v_wt = validate_weights(wb, pbuf, h_original, metric, sigma)
    row("Max abs error (distances):", f"{v_wt['dist_max_err']:.2e}")
    row("Distance validation:", v_wt["dist_status"])
    row("Max abs error (weights):", f"{v_wt['weight_max_err']:.2e}")
    row("Weight validation:", v_wt["weight_status"])

    banner("PIPELINE SUMMARY")

    gpu_total = bench_p["gpu_ms"] + bench_i["gpu_ms"] + bench_w["gpu_ms"]
    cpu_total = bench_p["cpu_ms"] + bench_i["cpu_ms"] + bench_w["cpu_ms"]

    row("GPU total (pert + inf + wt):", f"{gpu_total:.3f} ms")
    row("CPU total:",                   f"{cpu_total:.3f} ms")
    row("Overall GPU speedup:",         f"{cpu_total / gpu_total:.2f}x")

    # Sample output table
    h_preds = cp.asnumpy(wb.d_predictions)
    h_dists = cp.asnumpy(wb.d_distances)
    h_wts   = cp.asnumpy(wb.d_weights)

    print(f"\n  {'idx':>4}  {'prediction':>10}  {'distance':>10}  {'weight':>10}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}")
    for i in range(min(8, n_samples)):
        print(f"  {i:>4}  {h_preds[i]:>10.6f}  {h_dists[i]:>10.6f}  {h_wts[i]:>10.6f}")
    print()

    return {"gpu_total_ms": gpu_total, "cpu_total_ms": cpu_total,
            "speedup": cpu_total / gpu_total}


def run_benchmark_sweep():
    banner("BENCHMARK SWEEP")

    print("  [Warming up GPU kernels...]")
    warmup_gpu(256)
    print("  [Done]\n")

    print(f"  {'samples':>8}  {'features':>8}  {'gpu_ms':>8}  {'cpu_ms':>8}  {'speedup':>8}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    rng = np.random.default_rng(0)
    for n_s in [512, 1024, 2048, 4096]:
        for n_f in [32, 64, 128, 256]:
            sigma  = adaptive_kernel_width(n_f, "euclidean")
            h_orig = rng.random(n_f).astype(np.float32)
            h_w    = (rng.random(n_f) - 0.5).astype(np.float32)
            model  = model_create(h_w, 0.0)

            pbuf = perturbation_generate(h_orig, n_s, n_f)
            wb   = inference_run(pbuf, model)
            weights_compute(wb, pbuf, h_orig, sigma=sigma)

            bp = benchmark_perturbation(h_orig, n_s, n_f, n_repeats=5)
            bi = benchmark_inference(pbuf, model,          n_repeats=5)
            bw = benchmark_weights(wb, pbuf, h_orig,       n_repeats=5)

            gpu = bp["gpu_ms"] + bi["gpu_ms"] + bw["gpu_ms"]
            cpu = bp["cpu_ms"] + bi["cpu_ms"] + bw["cpu_ms"]
            print(f"  {n_s:>8}  {n_f:>8}  {gpu:>8.2f}  {cpu:>8.2f}  {cpu/gpu:>7.2f}x")


def main():
    parser = argparse.ArgumentParser(description="LIME GPU Milestone 1")
    parser.add_argument("--samples",  type=int, default=2048)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--metric",   type=str, default="euclidean",
                        choices=["euclidean", "cosine"])
    parser.add_argument("--sweep",    action="store_true")
    args = parser.parse_args()

    if args.sweep:
        run_benchmark_sweep()
    else:
        run_pipeline(args.samples, args.features, args.metric)


if __name__ == "__main__":
    main()
