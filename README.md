# LIME GPU Implementation

This repository contains a progressive from-scratch implementation of the LIME (Local Interpretable Model-agnostic Explanations) algorithm, focusing on accelerating core components with custom GPU kernels.

As outlined in the objective document ([lime-using-gpu.md](lime-using-gpu.md)), the project is broken down into milestones. We have currently implemented **Milestone 1 (M1)** and **Milestone 2 (M2)**.

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

## Project Structure

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

## Running the Pipeline

To execute the full LIME extraction process (data generation → surrogate training → validation → visualisation):

```bash
# Standard run (2048 samples, 64 features)
python m2_main.py

# Custom configurations
python m2_main.py --samples 4096 --features 128 --metric cosine --hidden 128 --epochs 800

# Skip consistency validation (for faster execution)
python m2_main.py --no-consistency
```
*Outputs (datasets as CSVs, and plots as PNGs) will be generated in the `output/` directory.*

### Milestone 1 Benchmarking

To test the raw performance of the custom CuPy/CUDA kernels vs CPU implementations:

```bash
# Standard run
python main.py

# Benchmark sweep across multiple configurations
python main.py --sweep
```

## Understanding the Output

When running `m2_main.py`, the console will detail:
1. **M1 Generation:** Creation and validation of perturbed samples, predictions, and weights in milliseconds.
2. **Surrogate Training:** The average weighted BCE loss declining over the specified training epochs.
3. **Feature Attribution:** A ranked list showing the raw gradient of the most positively and negatively influential features.
4. **Validation Status:** Clear pass/fail/warn scores for both Feature Ablation testing and the Multi-Seed Consistency checks.
5. **Visualisations:** `lime_explanation.png` and `lime_loss_curve.png` rendered and saved to `output/`.
