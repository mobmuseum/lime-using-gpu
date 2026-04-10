import argparse
import sys

try:
    import cupy as cp
    cp.cuda.Device(0).use()
except Exception as exc:
    sys.exit(f"[ERROR] No CUDA GPU detected: {exc}")

import numpy as np
import torch
from pathlib import Path

# M1 modules
from lime_common  import adaptive_kernel_width, DistanceMetric
from perturbation import perturbation_generate, validate_perturbation
from inference    import model_create, inference_run, validate_inference
from weights      import weights_compute, validate_weights

# M2 modules
from surrogate    import train_surrogate, benchmark_surrogate
from explanation  import (
    extract_attributions,
    validate_ablation,
    validate_consistency,
    plot_attributions,
    plot_loss_curve,
)

def banner(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def row(label: str, value, width: int = 44) -> None:
    print(f"  {label:<{width}} {value}")


# Main M2 pipeline
def run_m2_pipeline(
    n_samples:      int   = 2048,
    n_features:     int   = 64,
    metric:         str   = DistanceMetric.EUCLIDEAN,
    hidden_size:    int   = 64,
    n_epochs:       int   = 500,
    top_k:          int   = 8,
    run_consistency: bool = True,
    seed:           int   = 42,
) -> None:

    sigma = adaptive_kernel_width(n_features, metric)

    # Output directory
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # Synthetic inputs
    rng        = np.random.default_rng(seed)
    h_original = rng.random(n_features).astype(np.float32)
    h_weights  = (rng.random(n_features) - 0.5).astype(np.float32)
    bias       = 0.1

    row("Samples:",      n_samples)
    row("Features:",     n_features)
    row("Distance metric:", metric)
    row("Surrogate hidden size:", hidden_size)
    row("Training epochs:", n_epochs)
    row("Top-K features:", top_k)

    # STEP 1: Run M1 GPU pipeline to build the surrogate training dataset
    banner("STEP 1: Generate Perturbed Dataset (GPU)")

    pbuf = perturbation_generate(h_original, n_samples, n_features, seed=seed)
    row("Perturbed samples generated:", f"{n_samples}")

    v_p = validate_perturbation(pbuf, h_original)
    row("Mask keep rate:",   f"{v_p['keep_rate']:.4f}  (expected ~0.5000)")
    row("Mask application:", v_p["mask_apply"])

    bb_model = model_create(h_weights, bias)
    wb       = inference_run(pbuf, bb_model)
    v_inf    = validate_inference(wb, pbuf, h_weights, bias)
    row("Black-box inference max error:", f"{v_inf['max_abs_error']:.2e}")
    row("Inference validation:",          v_inf["status"])

    weights_compute(wb, pbuf, h_original, metric, sigma)
    v_wt = validate_weights(wb, pbuf, h_original, metric, sigma)
    row("Distance max error:", f"{v_wt['dist_max_err']:.2e}")
    row("Weight validation:",  v_wt["weight_status"])

    # Pull everything from GPU → CPU as NumPy arrays
    X = cp.asnumpy(pbuf.d_samples)       # (N, F)  perturbed feature vectors
    y = cp.asnumpy(wb.d_predictions)     # (N,)    black-box predictions
    w = cp.asnumpy(wb.d_weights)         # (N,)    kernel importance weights

    print(f"\n  Training dataset:  X={X.shape}, y={y.shape}, w={w.shape}")
    print(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  w range: [{w.min():.4f}, {w.max():.4f}]")

    # Save dataset to disk as CSV
    # Main dataset: one row per perturbed sample.
    # Columns: feature_0 … feature_F-1 | y_pred | kernel_weight
    import pandas as pd

    feat_cols   = [f"feature_{i}" for i in range(n_features)]
    df_dataset  = pd.DataFrame(X, columns=feat_cols)
    df_dataset["y_pred"]        = y
    df_dataset["kernel_weight"] = w

    dataset_path  = out_dir / f"lime_dataset_n{n_samples}_f{n_features}_{metric}.csv"
    df_dataset.to_csv(dataset_path, index=False)

    # Original instance: single-row CSV for reference
    original_path = out_dir / f"lime_original_f{n_features}.csv"
    pd.DataFrame([h_original], columns=feat_cols).to_csv(original_path, index=False)

    print(f"\n  Dataset saved  → {dataset_path}")
    print(f"    Rows: {len(df_dataset)}  |  "
          f"Cols: {n_features} features + y_pred + kernel_weight")
    print(f"  Original saved → {original_path}")

    # STEP 2: Train Shallow NN Surrogate (PyTorch)
    banner("STEP 2: Surrogate NN Training (PyTorch)")

    print(f"\n  Architecture: Linear({n_features}, {hidden_size}) → ReLU "
          f"→ Linear({hidden_size}, 1) → Sigmoid")
    print(f"  Loss: Weighted Binary Cross-Entropy\n")

    surrogate, loss_history = train_surrogate(
        X, y, w,
        hidden_size=hidden_size,
        n_epochs=n_epochs,
        verbose=True,
    )

    bench = benchmark_surrogate(X, y, w, hidden_size=hidden_size, n_epochs=n_epochs, n_repeats=3)
    print()
    row("Training time (mean):", f"{bench['mean_ms']:.1f} ms")
    row("Training time (std):",  f"{bench['std_ms']:.1f} ms")

    # STEP 3: Extract Feature Attributions
    banner("STEP 3: Feature Attribution (Input Gradients)")

    attributions = extract_attributions(surrogate, h_original)

    print(f"\n  Attribution computed via: ∂(output) / ∂(input) at x_original")
    print(f"\n  {'Rank':<6}  {'Feature':<12}  {'Attribution':>14}  {'Direction'}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*14}  {'─'*12}")
    ranked = np.argsort(np.abs(attributions))[::-1]
    for rank, feat_idx in enumerate(ranked[:top_k], start=1):
        direction = "↑ positive" if attributions[feat_idx] > 0 else "↓ negative"
        print(f"  {rank:<6}  feature {feat_idx:<4}  {attributions[feat_idx]:>+14.6f}  {direction}")

    # STEP 4: Validate Explanations
    banner("STEP 4: Validation")

    # 4a. Ablation
    print("\n  [4a] Feature Ablation Test")
    v_abl = validate_ablation(surrogate, h_original, attributions, top_k=top_k)
    row("Baseline prediction:",     f"{v_abl['baseline_pred']:.6f}")
    row("Ablated prediction:",      f"{v_abl['ablated_pred']:.6f}")
    row("Delta (|baseline - abl|):", f"{v_abl['delta']:.6f}")
    row("Top-K features ablated:",  str(v_abl['top_k_features']))
    row("Ablation status:",         v_abl["status"])

    # 4b. Consistency
    if run_consistency:
        print("\n  [4b] Consistency Check (3 different perturbation seeds)")
        print("       Re-running M1 + training per seed — please wait...\n")
        v_con = validate_consistency(
            h_original, n_features, n_samples,
            h_weights   = h_weights,
            bias        = bias,
            hidden_size = hidden_size,
            n_epochs    = n_epochs,
            top_k       = top_k,
            seeds       = [42, 99, 7],
        )
        for s, feat_set in zip(v_con["seeds"], v_con["top_k_per_seed"]):
            print(f"    seed={s}: top-{top_k} features = {feat_set}")
        print()
        row("Stable features across seeds:", str(v_con["stable_features"]))
        row("Stability ratio:",              f"{v_con['stability_ratio']:.2f}")
        row("Consistency status:",           v_con["status"])
    else:
        print("\n  [4b] Consistency Check skipped (--no-consistency)")

    # STEP 5: Visualise
    banner("STEP 5: Visualisation")
    print("\n  Rendering explanation bar chart...")
    plot_attributions(
        attributions,
        title=f"LIME Feature Attributions — Shallow NN Surrogate  (top {top_k})",
        top_k=top_k,
        save_path=str(out_dir / "lime_explanation.png"),
    )
    print("\n  Rendering training loss curve...")
    plot_loss_curve(
        loss_history,
        save_path=str(out_dir / "lime_loss_curve.png"),
    )


def main():
    parser = argparse.ArgumentParser(description="LIME GPU — Milestone 2: Surrogate NN")
    parser.add_argument("--samples",        type=int,   default=2048)
    parser.add_argument("--features",       type=int,   default=64)
    parser.add_argument("--metric",         type=str,   default="euclidean",
                        choices=["euclidean", "cosine"])
    parser.add_argument("--hidden",         type=int,   default=64,
                        help="Hidden layer size for surrogate NN")
    parser.add_argument("--epochs",         type=int,   default=500,
                        help="Training epochs for surrogate NN")
    parser.add_argument("--top-k",          type=int,   default=8,
                        help="Number of top features to show/ablate")
    parser.add_argument("--no-consistency", action="store_true",
                        help="Skip the consistency validation (faster)")
    args = parser.parse_args()

    run_m2_pipeline(
        n_samples       = args.samples,
        n_features      = args.features,
        metric          = args.metric,
        hidden_size     = args.hidden,
        n_epochs        = args.epochs,
        top_k           = args.top_k,
        run_consistency = not args.no_consistency,
    )


if __name__ == "__main__":
    main()
