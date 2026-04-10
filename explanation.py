import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from surrogate import SurrogateNN


# Attribution Extraction
def extract_attributions(
    model:      SurrogateNN, # Trained SurrogateNN
    x_original: np.ndarray, # Original input instance
    device:     str = "cpu", # 'cpu' or 'cuda'
) -> np.ndarray:
    device = torch.device(device)
    model.eval()

    x_t = (
        torch.tensor(x_original, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )
    x_t.requires_grad_(True)

    output = model(x_t)
    output.backward()

    attributions = x_t.grad.detach().cpu().numpy().squeeze()
    return attributions 
            # attributions: np.ndarray, shape (n_features)
            # Positive  → feature pushes surrogate prediction upward.
            # Negative  → feature pushes surrogate prediction downward.


# Feature Ablation
def validate_ablation(
    model:      SurrogateNN, # Trained SurrogateNN
    x_original: np.ndarray, # Original input instance
    attributions: np.ndarray, # Gradient-based attributions
    top_k:      int = 5, # Number of top features to ablate
    device:     str = "cpu", # 'cpu' or 'cuda'
) -> dict:
    device_t = torch.device(device)
    model.eval()

    with torch.no_grad():
        x_t      = torch.tensor(x_original, dtype=torch.float32).unsqueeze(0).to(device_t)
        baseline = model(x_t).item()

        top_k_idx  = np.argsort(np.abs(attributions))[-top_k:]
        x_ablated  = x_original.copy()
        x_ablated[top_k_idx] = 0.0

        x_abl_t  = torch.tensor(x_ablated, dtype=torch.float32).unsqueeze(0).to(device_t)
        ablated  = model(x_abl_t).item()

    delta = abs(baseline - ablated) 
    return {
        "baseline_pred":  round(baseline, 6),
        "ablated_pred":   round(ablated, 6),
        "delta":          round(delta, 6),
        "top_k_features": sorted(top_k_idx.tolist()),
        "status":         "PASS ✓" if delta > 0.01 else "WARN ✗ (small delta — features may not be impactful)",
    } # Returns dict with baseline_pred, ablated_pred, delta, top_k_features, status


# Consistency Check
def validate_consistency(
    x_original:  np.ndarray, # Original input instance
    n_features:  int, # Number of features
    n_samples:   int, # Number of perturbed samples
    h_weights:   np.ndarray, # Black-box model weights
    bias:        float = 0.1, # Black-box model bias
    hidden_size: int   = 64, # Surrogate hidden layer size
    n_epochs:    int   = 500, # Number of training epochs
    top_k:       int   = 5, # Number of top features
    seeds:       list  = [42, 99, 7], # List of perturbation seeds
    device:      str   = "cpu", # 'cpu' or 'cuda'
) -> dict:
    import cupy as cp
    from perturbation import perturbation_generate
    from inference    import model_create, inference_run
    from weights      import weights_compute
    from surrogate    import train_surrogate

    # Use the SAME black-box model as the main pipeline (passed in via h_weights)
    bb_model  = model_create(h_weights, bias=bias)

    top_k_sets = []

    for seed in seeds:
        pbuf = perturbation_generate(x_original, n_samples, n_features, seed=seed)
        wb   = inference_run(pbuf, bb_model)
        weights_compute(wb, pbuf, x_original)

        X = cp.asnumpy(pbuf.d_samples)
        y = cp.asnumpy(wb.d_predictions)
        w = cp.asnumpy(wb.d_weights)

        surr, _ = train_surrogate(
            X, y, w,
            hidden_size=hidden_size,
            n_epochs=n_epochs,
            device=device,
            verbose=False,
        )
        attrs   = extract_attributions(surr, x_original, device=device)
        top_set = set(np.argsort(np.abs(attrs))[-top_k:].tolist())
        top_k_sets.append(top_set)

    intersection = top_k_sets[0].intersection(*top_k_sets[1:])
    stability    = len(intersection) / top_k

    return {
        "seeds":          seeds,
        "top_k_per_seed": [sorted(s) for s in top_k_sets],
        "stable_features": sorted(intersection),
        "stability_ratio": round(stability, 2),
        "status":          "PASS ✓" if stability >= 0.6 else "WARN ✗ (low stability across seeds)",
    } # Returns dict with top_k_per_seed, stable_features, stability_ratio, status



# Visualisation
def plot_attributions(
    attributions: np.ndarray, # Gradient-based attributions
    title:        str  = "LIME Feature Attributions - Shallow NN Surrogate", # Chart title
    top_k:        int  = None, # Number of top features to display
    save_path:    str  = None, # Path to save the figure
) -> None:
    # Select features to show
    if top_k is not None and top_k < len(attributions):
        top_idx  = np.argsort(np.abs(attributions))[-top_k:]
        # Sort selected features by attribution value for a cleaner chart
        order    = top_idx[np.argsort(attributions[top_idx])]
        vals     = attributions[order]
        labels   = [f"feature {i}" for i in order]
    else:
        order  = np.argsort(attributions)
        vals   = attributions[order]
        labels = [f"feature {i}" for i in order]

    # Colour palette
    BG_DARK  = "#0f0f1a"
    BG_PANEL = "#1a1a2e"
    POS_CLR  = "#e05c5c"   # red — pushes prediction up
    NEG_CLR  = "#4d9de0"   # blue — pushes prediction down
    GRID_CLR = "#2a2a40"
    TEXT_CLR = "#e0e0f0"

    colors = [POS_CLR if v > 0 else NEG_CLR for v in vals]

    fig_h = max(5, len(labels) * 0.38)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_PANEL)

    bars = ax.barh(
        labels, vals, color=colors,
        edgecolor=BG_PANEL, linewidth=0.6, height=0.65,
    )

    # Value labels on bars
    for bar, val in zip(bars, vals):
        offset = 0.002 if val >= 0 else -0.002
        ha     = "left"  if val >= 0 else "right"
        ax.text(
            val + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", va="center", ha=ha,
            fontsize=8.5, color=TEXT_CLR, fontweight="bold",
        )

    # Zero line
    ax.axvline(0, color="#aaaacc", linewidth=1.0, linestyle="--", alpha=0.6)

    # Grid
    ax.xaxis.grid(True, color=GRID_CLR, linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)

    # Labels and title
    ax.set_xlabel("Attribution  (∂ output / ∂ input)", fontsize=11, color=TEXT_CLR, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_CLR, pad=14)

    # Axis styling
    for spine in ax.spines.values():
        spine.set_color("#333355")
    ax.tick_params(colors=TEXT_CLR, labelsize=9)

    # Legend
    pos_patch = mpatches.Patch(color=POS_CLR, label="Increases prediction")
    neg_patch = mpatches.Patch(color=NEG_CLR, label="Decreases prediction")
    ax.legend(
        handles=[pos_patch, neg_patch],
        loc="lower right", framealpha=0.25,
        labelcolor=TEXT_CLR, facecolor=BG_PANEL, edgecolor="#333355",
        fontsize=9,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
        print(f"  Saved explanation plot → {save_path}")

    plt.show()


def plot_loss_curve(
    loss_history: list, # List of per-epoch loss values from train_surrogate()
    save_path:    str = None, # Path to save the figure
) -> None:
    BG_DARK  = "#0f0f1a"
    BG_PANEL = "#1a1a2e"
    TEXT_CLR = "#e0e0f0"
    LINE_CLR = "#7ec8e3"
    GRID_CLR = "#2a2a40"

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_PANEL)

    epochs = np.arange(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, color=LINE_CLR, linewidth=1.8, label="Weighted BCE Loss")
    ax.fill_between(epochs, loss_history, alpha=0.12, color=LINE_CLR)

    ax.set_xlabel("Epoch", fontsize=11, color=TEXT_CLR, labelpad=8)
    ax.set_ylabel("Weighted BCE Loss", fontsize=11, color=TEXT_CLR, labelpad=8)
    ax.set_title("Surrogate NN — Training Loss Curve", fontsize=13, fontweight="bold", color=TEXT_CLR, pad=14)

    ax.xaxis.grid(True, color=GRID_CLR, linewidth=0.6, linestyle="--")
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#333355")
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    ax.legend(labelcolor=TEXT_CLR, facecolor=BG_PANEL, edgecolor="#333355", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
        print(f"  Saved loss curve → {save_path}")

    plt.show()
