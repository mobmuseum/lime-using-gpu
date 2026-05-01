# LIME GPU Implementation

This repository contains a progressive from-scratch implementation of the LIME (Local Interpretable Model-agnostic Explanations) algorithm, focusing on accelerating core components with custom GPU kernels.

As outlined in the objective document ([lime-using-gpu.md](lime-using-gpu.md)), the project is broken down into milestones. We have currently implemented **Milestone 1 (M1)**, **Milestone 2 (M2)**, and **Milestone 3 (M3)**.

## Milestones Implemented

### Milestone 1: GPU-Accelerated Core 
Implemented the computational core of LIME using explicit GPU programming (CuPy/CUDA) without relying on heavy ML frameworks:
- **Perturbation Generation:** Generates binary masks to hide/keep features.
- **Base Model Inference:** Calculates black-box model predictions for perturbed samples.
- **Weight Computation:** Calculates distances (Euclidean/Cosine) and kernel weights.

### Milestone 2: Surrogate Model & Explanation Validation
Completed the explanatory pipeline by training a surrogate model using high-level frameworks (PyTorch):
- **Surrogate Training (`surrogate.py`):** Trains a shallow Neural Network (NN) to mimic the black-box model locally using weighted Binary Cross-Entropy.
- **Feature Attribution (`explanation.py`):** Extracts feature importance using **input gradients** (`∂output / ∂input`) at the original data point. 
- **Validation:** Provides rigorous diagnostic mechanisms:
  - **Feature Ablation:** Zeroes out the most important features to mathematically verify their impact on the prediction.
  - **Consistency Check:** Re-runs the perturbation and training pipeline across multiple random seeds to measure stability of the extracted explanations.
- **Visualization:** Generates dynamic horizontal bar charts for attributions and loss curves to assess model convergence.

### Milestone 3: Fully GPU-Native Pipeline
Replaced the framework-based surrogate model training with custom GPU implementations, completing the end-to-end GPU-native LIME pipeline:
- **GPU-based Surrogate Training (`m3_surrogate.py`):** Implemented weighted linear regression on the GPU. Supports two solvers:
  - Closed-form normal equations (cuSolver)
  - Iterative Gradient Descent (custom CUDA kernels)
- **Explanation Extraction (`m3_explanation.py`):** Extracts feature importance directly on the GPU by accessing the regression coefficients. Validated through feature ablation and consistency checks.
- **Performance Benchmarks:** Added rigorous 3-way benchmarking (CPU-only vs. M2-style vs. M3-native) to analyze raw execution time across all components.

## Project Structure

- `m3_main.py`: The entry point for the **Milestone 3** fully GPU-native pipeline.
- `m3_surrogate.py`, `m3_explanation.py`: M3 custom CUDA and CuPy implementation for surrogate training and explanation extraction.
- `m2_main.py`: The entry point for the **Milestone 2** pipeline. It bridges the M1 GPU dataset generation with M2's PyTorch surrogate training.
- `surrogate.py`: Contains the architecture (`SurrogateNN`) and the `train_surrogate` loop using PyTorch.
- `explanation.py`: Holds logic for `extract_attributions`, `validate_ablation`, `validate_consistency`, and visualisations (`plot_attributions`, `plot_loss_curve`).
- `main.py`: The entry point for the **Milestone 1** CPU vs GPU benchmarking pipeline.
- `perturbation.py`, `inference.py`, `weights.py`: M1 custom CUDA implementations for dataset generation and calculation.

## Prerequisites & Setup

To run this pipeline efficiently, you need an **NVIDIA GPU** with CUDA support.

1. Ensure you have Python 3.8+ installed.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

*Note: For Windows users, if you encounter unicode encoding errors during CuPy kernel compilation, try setting the environment variable `PYTHONUTF8=1` before running.*

## Running the Pipeline

### Milestone 3 (Fully GPU-Native LIME)

To execute the fully native pipeline (data generation → GPU surrogate training → validation → visualisation):

```bash
# Standard run (default: uses closed-form cuSolver)
python m3_main.py

# Custom configurations
python m3_main.py --samples 4096 --features 128 --metric cosine --top-k 10

# Run with custom CUDA Gradient Descent solver (Recommended if cuSolver fails)
python m3_main.py --method gd --iters 2000 --lr 0.05

# Run performance benchmark sweep across various dimensions
python m3_main.py --sweep

# Skip consistency validation (for faster execution)
python m3_main.py --no-consistency
```
*Outputs (plots, convergence curve) will be generated in the `output/` directory.*

### Milestone 2 (PyTorch Surrogate)

```bash
# Standard run
python m2_main.py

# Custom configurations
python m2_main.py --samples 4096 --features 128 --metric cosine --hidden 128 --epochs 800

# Skip consistency validation (for faster execution)
python m2_main.py --no-consistency
```
*Outputs will be generated in the `output/` directory.*

### Milestone 1 Benchmarking

To test the raw performance of the custom CuPy/CUDA kernels vs CPU implementations:

```bash
# Standard run
python main.py

# Benchmark sweep across multiple configurations
python main.py --sweep
```

## Understanding the Output (Milestone 3)

When running `m3_main.py`, the console will detail:
1. **M1 Generation:** Creation of perturbed samples, predictions, and kernel weights using custom CUDA kernels.
2. **Surrogate Training:** Details the parameter optimization (Weighted MSE loss decay if using Gradient Descent).
3. **Explanation Extraction:** A ranked list showing the most positively and negatively influential features according to surrogate regression coefficients.
4. **Validation Status:** Validation metrics for Ablation and Consistency Checks.
5. **3-Way Benchmark:** Speedup comparisons analyzing the performance benefit of the fully native approach.
6. **Visualisations:** `m3_lime_coefficients.png` and `m3_gd_convergence.png` rendered and saved to `output/`.
