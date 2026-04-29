import argparse
import sys
import time
from pathlib import Path

try:
    import cupy as cp
    cp.cuda.Device(0).use()
except Exception as exc:
    sys.exit(f"[ERROR] No CUDA GPU detected: {exc}")

import numpy as np


from lime_common  import adaptive_kernel_width, DistanceMetric
from perturbation import perturbation_generate, validate_perturbation
from inference    import model_create, inference_run, validate_inference
from weights      import weights_compute, validate_weights, benchmark_weights

from m3_surrogate  import (
    train_surrogate_gpu,
    train_surrogate_cpu,
    benchmark_surrogate_gpu,
    benchmark_surrogate_cpu,
)
from m3_explanation import (
    extract_coefficients_gpu,
    top_k_features_gpu,
    validate_coefficients,
    validate_ablation_gpu,
    validate_consistency_gpu,
    plot_coefficients,
    plot_convergence,
)

from perturbation import benchmark_perturbation
from inference    import benchmark_inference


def banner(title: str) -> None:
    line = "=" * 62
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def row(label: str, value, width: int = 46) -> None:
    print(f"  {label:<{width}} {value}")


def section(title: str) -> None:
    print(f"\n  ── {title} ──")


def run_m3_pipeline(
    n_samples:       int   = 2048,
    n_features:      int   = 64,
    metric:          str   = DistanceMetric.EUCLIDEAN,
    top_k:           int   = 8,
    surrogate_method: str  = "closed",   # "closed" or "gd"
    n_iters:         int   = 1000,       # only used by GD
    lr:              float = 1e-2,       # learning rate for GD
    run_consistency: bool  = True,
    seed:            int   = 42,
) -> None:

    sigma   = adaptive_kernel_width(n_features, metric)
    out_dir = Path("output_m3")
    out_dir.mkdir(exist_ok=True)

    banner("MILESTONE 3 — Fully GPU-Native LIME")
    row("Samples:",          n_samples)
    row("Features:",         n_features)
    row("Distance metric:",  metric)
    row("Kernel bandwidth:", f"{sigma:.4f}")
    row("Surrogate method:", surrogate_method.upper() +
        (" (normal equations, cuSolver)" if surrogate_method == "closed"
         else " (iterative CUDA kernels)"))
    if surrogate_method == "gd":
        row("GD iterations:", n_iters)
        row("GD learning rate:", lr)
    row("Top-K features:", top_k)

    rng        = np.random.default_rng(seed)
    h_original = rng.random(n_features).astype(np.float32)
    h_weights  = (rng.random(n_features) - 0.5).astype(np.float32)
    bias       = 0.1

    banner("STAGE 1 — Perturbation Generation  [GPU, M1 kernels]")

    pbuf = perturbation_generate(h_original, n_samples, n_features, seed=seed)
    v_p  = validate_perturbation(pbuf, h_original)
    row("Perturbed samples:", n_samples)
    row("Mask keep rate:",    f"{v_p['keep_rate']:.4f}  (expected ~0.5000)")
    row("Mask validation:",   v_p["mask_apply"])

    banner("STAGE 2 — Black-box Inference  [GPU, cuBLAS]")

    bb_model = model_create(h_weights, bias)
    wb       = inference_run(pbuf, bb_model)
    v_inf    = validate_inference(wb, pbuf, h_weights, bias)
    row("Inference method:",     "cuBLAS  d_samples @ d_weights + bias  → sigmoid")
    row("Max abs error (preds):", f"{v_inf['max_abs_error']:.2e}")
    row("Inference validation:", v_inf["status"])

    banner("STAGE 3 — Distance & Kernel Weights  [GPU, M1 kernels]")

    weights_compute(wb, pbuf, h_original, metric, sigma)
    v_wt = validate_weights(wb, pbuf, h_original, metric, sigma)
    row("Distance metric:",      metric)
    row("Max dist error:",       f"{v_wt['dist_max_err']:.2e}")
    row("Distance validation:",  v_wt["dist_status"])
    row("Max weight error:",     f"{v_wt['weight_max_err']:.2e}")
    row("Weight validation:",    v_wt["weight_status"])

    d_X = pbuf.d_samples      # (N, F) float32 on GPU
    d_y = wb.d_predictions    # (N,)   float32 on GPU
    d_w = wb.d_weights        # (N,)   float32 on GPU

    print(f"\n  Training data lives entirely on GPU:")
    print(f"    d_X  shape={d_X.shape}  dtype={d_X.dtype}")
    print(f"    d_y  shape={d_y.shape}  dtype={d_y.dtype}")
    print(f"    d_w  shape={d_w.shape}  dtype={d_w.dtype}")

    banner("STAGE 4 — GPU-native Surrogate Training  [M3 custom CUDA]")

    if surrogate_method == "closed":
        print(f"\n  Method: Closed-form normal equations")
        print(f"    β* = (X^T W X + λI)⁻¹  X^T W y")
        print(f"    cuBLAS for matmul, cuSolver for linear solve (LU)")
        print(f"    Ridge regularisation λ = 1e-4")
    else:
        print(f"\n  Method: Gradient Descent ({n_iters} iterations, lr={lr})")
        print(f"    Per-iter: residuals → weighted gradient → parameter update")
        print(f"    All 3 steps use custom CUDA kernels (no framework)")

    d_beta, loss_history = train_surrogate_gpu(
        d_X, d_y, d_w,
        method=surrogate_method,
        n_iters=n_iters,
        lr=lr,
        verbose=(surrogate_method == "gd"),
    )

    row("\n  Surrogate trained.",  f"d_beta shape={d_beta.shape}  on GPU")
    row("  Final weighted MSE:", f"{loss_history[-1]:.6f}")

    # CPU reference
    h_X      = cp.asnumpy(d_X)
    h_y      = cp.asnumpy(d_y)
    h_w      = cp.asnumpy(d_w)
    beta_cpu = train_surrogate_cpu(h_X, h_y, h_w)
    v_coef   = validate_coefficients(d_beta, beta_cpu)
    row("  Coefficient max abs error:", f"{v_coef['max_abs_err']:.2e}")
    row("  Coefficient validation:",    v_coef["status"])

    banner("STAGE 5 — Explanation Extraction  [GPU → minimal host transfer]")

    print(f"\n  Extracting top-{top_k} features by |β| on GPU...")

    top_idx, top_vals = top_k_features_gpu(d_beta, top_k)
    h_beta = extract_coefficients_gpu(d_beta)   # full vector, for plotting

    print(f"\n  {'Rank':<6}  {'Feature':<12}  {'Coefficient':>14}  {'Direction'}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*14}  {'─'*12}")
    for rank, (feat_idx, coef_val) in enumerate(zip(top_idx, top_vals), start=1):
        direction = "↑ positive" if coef_val > 0 else "↓ negative"
        print(f"  {rank:<6}  feature {feat_idx:<4}  {coef_val:>+14.6f}  {direction}")

    banner("STAGE 6 — Validation")

    section("Feature Ablation (GPU)")
    v_abl = validate_ablation_gpu(d_X, d_beta, h_original, top_k=top_k)
    row("Baseline prediction:",      f"{v_abl['baseline_pred']:.6f}")
    row("Ablated prediction:",       f"{v_abl['ablated_pred']:.6f}")
    row("Delta |baseline - abl|:",   f"{v_abl['delta']:.6f}")
    row("Top-K features ablated:",   str(v_abl["top_k_features"]))
    row("Ablation status:",          v_abl["status"])

    if run_consistency:
        section("Consistency Check (3 seeds)")
        print("  Re-running full M3 GPU pipeline with seeds [42, 99, 7]...\n")
        v_con = validate_consistency_gpu(
            h_original, n_features, n_samples,
            h_weights=h_weights, bias=bias,
            top_k=top_k, seeds=[42, 99, 7],
            method=surrogate_method,
        )
        for s, feat_set in zip(v_con["seeds"], v_con["top_k_per_seed"]):
            print(f"    seed={s}: top-{top_k} features = {feat_set}")
        print()
        row("Stable features across seeds:", str(v_con["stable_features"]))
        row("Stability ratio:",             f"{v_con['stability_ratio']:.2f}")
        row("Consistency status:",          v_con["status"])
    else:
        print("\n  Consistency check skipped (--no-consistency).")

    banner("3-WAY PERFORMANCE COMPARISON")

    print(f"\n  Configuration: N={n_samples}  F={n_features}  metric={metric}")
    print(f"  Repeats per benchmark: 5 (GPU uses CUDA events)\n")

    bp = benchmark_perturbation(h_original, n_samples, n_features, n_repeats=5)
    bi = benchmark_inference(pbuf, bb_model, n_repeats=5)
    bw = benchmark_weights(wb, pbuf, h_original, metric, n_repeats=5)
    m1_gpu_ms = bp["gpu_ms"] + bi["gpu_ms"] + bw["gpu_ms"]
    m1_cpu_ms = bp["cpu_ms"] + bi["cpu_ms"] + bw["cpu_ms"]

    # Surrogate benchmarks
    bg_gpu = benchmark_surrogate_gpu(d_X, d_y, d_w,
                                     method=surrogate_method,
                                     lr=lr,
                                     n_iters=n_iters, n_repeats=5)
    bg_cpu = benchmark_surrogate_cpu(h_X, h_y, h_w, n_repeats=5)

    # CPU-only
    cpu_pipeline_total = m1_cpu_ms + bg_cpu["cpu_ms"]

    # GPU pipeline + CPU surrogate
    m2_total = m1_gpu_ms + bg_cpu["cpu_ms"]  

    # GPU pipeline + GPU surrogate
    m3_total = m1_gpu_ms + bg_gpu["gpu_ms"]

    print(f"  {'Stage':<32}  {'CPU-only':>9}  {'M2-style':>9}  {'M3-native':>9}")
    print(f"  {'─'*32}  {'─'*9}  {'─'*9}  {'─'*9}")
    print(f"  {'Perturbation (ms)':<32}  {bp['cpu_ms']:>9.2f}  {bp['gpu_ms']:>9.2f}  {bp['gpu_ms']:>9.2f}")
    print(f"  {'Inference (ms)':<32}  {bi['cpu_ms']:>9.2f}  {bi['gpu_ms']:>9.2f}  {bi['gpu_ms']:>9.2f}")
    print(f"  {'Kernel weights (ms)':<32}  {bw['cpu_ms']:>9.2f}  {bw['gpu_ms']:>9.2f}  {bw['gpu_ms']:>9.2f}")
    print(f"  {'Surrogate training (ms)':<32}  {bg_cpu['cpu_ms']:>9.2f}  {bg_cpu['cpu_ms']:>9.2f}  {bg_gpu['gpu_ms']:>9.2f}")
    print(f"  {'─'*32}  {'─'*9}  {'─'*9}  {'─'*9}")
    print(f"  {'TOTAL (ms)':<32}  {cpu_pipeline_total:>9.2f}  {m2_total:>9.2f}  {m3_total:>9.2f}")
    print()
    row("M3 speedup over CPU-only:", f"{cpu_pipeline_total / m3_total:.2f}x")
    row("M3 speedup over M2-style:", f"{m2_total / m3_total:.2f}x")

    banner("VISUALISATION")

    print("\n  Rendering coefficient bar chart...")
    plot_coefficients(
        h_beta,
        title=f"LIME Coefficients — GPU WLS Surrogate  ({surrogate_method.upper()}, top {top_k})",
        top_k=top_k,
        save_path=str(out_dir / "m3_lime_coefficients.png"),
    )

    if surrogate_method == "gd" and len(loss_history) > 1:
        print("\n  Rendering GD convergence curve...")
        plot_convergence(
            loss_history,
            title="Gradient Descent Convergence — Weighted MSE (GPU Kernels)",
            save_path=str(out_dir / "m3_gd_convergence.png"),
        )

    banner("DONE")
    print(f"  Outputs saved to {out_dir}/\n")

def run_benchmark_sweep(method: str = "closed") -> None:
    banner(f"BENCHMARK SWEEP — GPU-native M3  (surrogate method: {method.upper()})")

    rng = np.random.default_rng(0)

    print(f"\n  {'N':>6}  {'F':>6}  {'CPU total':>10}  {'M3 total':>10}  {'Speedup':>8}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}")

    for n_s in [512, 1024, 2048, 4096]:
        for n_f in [32, 64, 128, 256]:
            sigma  = adaptive_kernel_width(n_f)
            h_orig = rng.random(n_f).astype(np.float32)
            h_wts  = (rng.random(n_f) - 0.5).astype(np.float32)
            model  = model_create(h_wts, 0.0)

            pbuf = perturbation_generate(h_orig, n_s, n_f)
            wb   = inference_run(pbuf, model)
            weights_compute(wb, pbuf, h_orig, sigma=sigma)

            d_X = pbuf.d_samples
            d_y = wb.d_predictions
            d_w = wb.d_weights
            h_X = cp.asnumpy(d_X)
            h_y = cp.asnumpy(d_y)
            h_w = cp.asnumpy(d_w)

            bp = benchmark_perturbation(h_orig, n_s, n_f, n_repeats=3)
            bi = benchmark_inference(pbuf, model,         n_repeats=3)
            bw = benchmark_weights(wb, pbuf, h_orig,      n_repeats=3)

            bg_gpu = benchmark_surrogate_gpu(d_X, d_y, d_w, method=method, n_repeats=3)
            bg_cpu = benchmark_surrogate_cpu(h_X, h_y, h_w, n_repeats=3)

            cpu_total = bp["cpu_ms"] + bi["cpu_ms"] + bw["cpu_ms"] + bg_cpu["cpu_ms"]
            m3_total  = bp["gpu_ms"] + bi["gpu_ms"] + bw["gpu_ms"] + bg_gpu["gpu_ms"]
            speedup   = cpu_total / m3_total

            print(f"  {n_s:>6}  {n_f:>6}  {cpu_total:>10.2f}  {m3_total:>10.2f}  {speedup:>7.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="LIME GPU — Milestone 3: Fully GPU-native pipeline"
    )
    parser.add_argument("--samples",        type=int,  default=2048,
                        help="Number of perturbation samples (default: 2048)")
    parser.add_argument("--features",       type=int,  default=64,
                        help="Number of features (default: 64)")
    parser.add_argument("--metric",         type=str,  default="euclidean",
                        choices=["euclidean", "cosine"],
                        help="Distance metric for kernel weights (default: euclidean)")
    parser.add_argument("--top-k",          type=int,  default=8,
                        help="Number of top features to display/ablate (default: 8)")
    parser.add_argument("--method",         type=str,  default="closed",
                        choices=["closed", "gd"],
                        help="Surrogate solver: 'closed' (normal equations) or 'gd' (gradient descent)")
    parser.add_argument("--iters",          type=int,   default=1000,
                        help="Gradient descent iterations (only with --method gd, default: 1000)")
    parser.add_argument("--lr",             type=float, default=1e-2,
                        help="GD learning rate (only with --method gd, default: 0.01 — use 0.1 for faster convergence)")
    parser.add_argument("--no-consistency", action="store_true",
                        help="Skip the consistency check (faster run)")
    parser.add_argument("--sweep",          action="store_true",
                        help="Run benchmark sweep over (N, F) combinations")
    args = parser.parse_args()

    if args.sweep:
        run_benchmark_sweep(method=args.method)
    else:
        run_m3_pipeline(
            n_samples        = args.samples,
            n_features       = args.features,
            metric           = args.metric,
            top_k            = args.top_k,
            surrogate_method = args.method,
            n_iters          = args.iters,
            lr               = args.lr,
            run_consistency  = not args.no_consistency,
        )


if __name__ == "__main__":
    main()
