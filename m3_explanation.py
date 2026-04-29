import numpy as np
import cupy as cp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from lime_common import BLOCK_SIZE

def extract_coefficients_gpu(d_beta: cp.ndarray) -> np.ndarray:
    return d_beta.get().astype(np.float32)


def top_k_features_gpu(d_beta: cp.ndarray, top_k: int) -> tuple:
    d_order = cp.argsort(cp.abs(d_beta))[::-1]   # descending
    d_top_k_idx = d_order[:top_k]
    d_top_k_val = d_beta[d_top_k_idx]

    # Only now do we transfer two small arrays to host
    return d_top_k_idx.get(), d_top_k_val.get()

def validate_coefficients(
    d_beta_gpu: cp.ndarray,   # GPU result from M3 solver
    beta_cpu:   np.ndarray,   # CPU numpy reference result
    tol:        float = 1e-3, # tolerance; WLS allows slightly larger error than M1 ops
) -> dict:
    
    h_beta = d_beta_gpu.get().astype(np.float32)
    max_err = float(np.abs(h_beta - beta_cpu).max())
    rel_err = float(np.abs(h_beta - beta_cpu).max() / (np.abs(beta_cpu).max() + 1e-8))

    return {
        "max_abs_err": max_err,
        "max_rel_err": rel_err,
        "status": "PASS ✓" if max_err < tol else f"FAIL ✗ (max_err={max_err:.2e})",
    }


def validate_ablation_gpu(
    d_X:        cp.ndarray,    # (N, F) test samples (use the training set)
    d_beta:     cp.ndarray,    # (F,)   GPU coefficients
    x_original: np.ndarray,    # (F,)   the original instance
    top_k:      int = 8,
) -> dict:
    
    d_x_orig = cp.asarray(x_original, dtype=cp.float32)

    # Baseline prediction on GPU
    d_baseline = float((d_x_orig @ d_beta).get())

    # Find top-k indices by |beta| on GPU
    d_order   = cp.argsort(cp.abs(d_beta))[::-1]
    d_top_idx = d_order[:top_k]

    # Zero out top-k features (still on GPU)
    d_x_abl = d_x_orig.copy()
    d_x_abl[d_top_idx] = 0.0

    d_ablated = float((d_x_abl @ d_beta).get())
    delta     = abs(d_baseline - d_ablated)

    return {
        "baseline_pred":  round(d_baseline, 6),
        "ablated_pred":   round(d_ablated,  6),
        "delta":          round(delta,       6),
        "top_k_features": sorted(d_top_idx.get().tolist()),
        "status": "PASS ✓" if delta > 0.01
                  else "WARN ✗ (small delta — features may not be impactful)",
    }


def validate_consistency_gpu(
    x_original: np.ndarray,
    n_features:  int,
    n_samples:   int,
    h_weights:   np.ndarray,
    bias:        float   = 0.1,
    top_k:       int     = 8,
    seeds:       list    = None,
    method:      str     = "closed",
) -> dict:
  
    from perturbation import perturbation_generate
    from inference    import model_create, inference_run
    from weights      import weights_compute
    from m3_surrogate import train_surrogate_gpu

    if seeds is None:
        seeds = [42, 99, 7]

    bb_model   = model_create(h_weights, bias=bias)
    sigma      = 0.25 * (n_features ** 0.5)
    top_k_sets = []

    for seed in seeds:
        pbuf = perturbation_generate(x_original, n_samples, n_features, seed=seed)
        wb   = inference_run(pbuf, bb_model)
        weights_compute(wb, pbuf, x_original)

        d_beta, _ = train_surrogate_gpu(
            pbuf.d_samples, wb.d_predictions, wb.d_weights,
            method=method
        )

        # Top-k by |beta| — GPU sort, tiny host transfer
        top_idx, _ = top_k_features_gpu(d_beta, top_k)
        top_k_sets.append(set(top_idx.tolist()))

    intersection = top_k_sets[0].intersection(*top_k_sets[1:])
    stability    = len(intersection) / top_k

    return {
        "seeds":           seeds,
        "top_k_per_seed":  [sorted(s) for s in top_k_sets],
        "stable_features": sorted(intersection),
        "stability_ratio": round(stability, 2),
        "status": "PASS ✓" if stability >= 0.6
                  else "WARN ✗ (low stability across seeds)",
    }

def plot_coefficients(
    coefficients: np.ndarray,
    title:        str = "LIME Feature Attributions — GPU WLS Surrogate",
    top_k:        int = None,
    save_path:    str = None,
) -> None:
    
    if top_k is not None and top_k < len(coefficients):
        top_idx = np.argsort(np.abs(coefficients))[-top_k:]
        order   = top_idx[np.argsort(coefficients[top_idx])]
        vals    = coefficients[order]
        labels  = [f"feature {i}" for i in order]
    else:
        order  = np.argsort(coefficients)
        vals   = coefficients[order]
        labels = [f"feature {i}" for i in order]

    BG_DARK  = "#0f0f1a"
    BG_PANEL = "#1a1a2e"
    POS_CLR  = "#e05c5c"
    NEG_CLR  = "#4d9de0"
    GRID_CLR = "#2a2a40"
    TEXT_CLR = "#e0e0f0"

    colors = [POS_CLR if v > 0 else NEG_CLR for v in vals]

    fig_h = max(5, len(labels) * 0.38)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_PANEL)

    bars = ax.barh(labels, vals, color=colors,
                   edgecolor=BG_PANEL, linewidth=0.6, height=0.65)

    for bar, val in zip(bars, vals):
        offset = 0.002 if val >= 0 else -0.002
        ha     = "left" if val >= 0 else "right"
        ax.text(val + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center", ha=ha,
                fontsize=8.5, color=TEXT_CLR, fontweight="bold")

    max_val = np.max(np.abs(vals)) if len(vals) > 0 else 1.0
    ax.set_xlim(-max_val * 1.3, max_val * 1.3)
    ax.axvline(0, color="#aaaacc", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.xaxis.grid(True, color=GRID_CLR, linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlabel("Coefficient  β  (linear surrogate weight)", fontsize=11,
                  color=TEXT_CLR, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_CLR, pad=14)

    for spine in ax.spines.values():
        spine.set_color("#333355")
    ax.tick_params(colors=TEXT_CLR, labelsize=9)

    pos_patch = mpatches.Patch(color=POS_CLR, label="Increases prediction")
    neg_patch = mpatches.Patch(color=NEG_CLR, label="Decreases prediction")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right",
              framealpha=0.25, labelcolor=TEXT_CLR, facecolor=BG_PANEL,
              edgecolor="#333355", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
        print(f"  Saved coefficient plot → {save_path}")
    plt.show()


def plot_convergence(
    loss_history: list,
    title:        str = "GPU Gradient Descent — Weighted MSE Convergence",
    save_path:    str = None,
) -> None:
    BG_DARK  = "#0f0f1a"
    BG_PANEL = "#1a1a2e"
    TEXT_CLR = "#e0e0f0"
    LINE_CLR = "#a8e6cf"
    GRID_CLR = "#2a2a40"

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_PANEL)

    steps = np.arange(1, len(loss_history) + 1)
    ax.plot(steps, loss_history, color=LINE_CLR, linewidth=1.8,
            label="Weighted MSE")
    ax.fill_between(steps, loss_history, alpha=0.12, color=LINE_CLR)

    ax.set_xlabel("Iteration (×log_every)", fontsize=11, color=TEXT_CLR, labelpad=8)
    ax.set_ylabel("Weighted MSE", fontsize=11, color=TEXT_CLR, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_CLR, pad=14)

    ax.xaxis.grid(True, color=GRID_CLR, linewidth=0.6, linestyle="--")
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#333355")
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    ax.legend(labelcolor=TEXT_CLR, facecolor=BG_PANEL,
              edgecolor="#333355", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
        print(f"  Saved convergence plot → {save_path}")
    plt.show()
