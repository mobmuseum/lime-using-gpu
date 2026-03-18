# Running LIME GPU (Milestone 1)

This repository contains a full Custom GPU implementation of the LIME (Local Interpretable Model-agnostic Explanations) algorithm's core components: Perturbation Generation, Logistic Regression Inference, and Distance & Kernel Weight Computation. 

The pipeline is written in Python and uses **CuPy** to compile and execute raw CUDA kernels directly on the GPU.

## Prerequisites

To run this pipeline on any PC, you must meet the following requirements:
1. **NVIDIA GPU** with CUDA support.
2. **Python 3.8+** installed.
3. Appropriate CUDA drivers installed on your system.

## Setup Instructions

### 1. Install Dependencies
The single biggest requirement is **CuPy**. Depending on your operating system, installation slightly varies.

#### For Windows Users:
If you do not have the complete (sizeable) NVIDIA CUDA Toolkit installed on your machine, you can still run this by installing the pre-built NVIDIA runtime wheels alongside CuPy. Open your terminal and run:

```bash
# Install PyTorch's/NVIDIA's standalone CUDA 12 runtime packages
pip install nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12

# Install CuPy for CUDA 12
pip install cupy-cuda12x

# Install standard numeric libraries
pip install numpy
```
*(Note: If you already have the full CUDA Toolkit 12.x installed, you only need `pip install cupy-cuda12x numpy`)*

#### For Linux Users:
```bash
pip install cupy-cuda12x numpy
```

### 2. Running the Pipeline

Once the environment is set up, you can execute the main script. The script automatically orchestrates the entire CPU vs. GPU validation and performance benchmarking.

**Standard Run:**
This will execute the pipeline using the default parameters (2048 samples, 64 features, euclidean distance metric):
```bash
python main.py
```

**Custom Run:**
You can specify the number of samples, the number of features, and the distance metric (`euclidean` or `cosine`):
```bash
python main.py --samples 4096 --features 128 --metric cosine
```

**Benchmark Sweep Run:**
To run a comprehensive benchmark sweep measuring performance and speedups across grid combinations of `samples` (512 to 4096) and `features` (32 to 256), use the `--sweep` flag:
```bash
python main.py --sweep
```

## Understanding the Output

When you run `main.py`, the CLI will output:
1. **GPU Information:** Detects and prints your specific NVIDIA GPU's memory and Streaming Multiprocessor (SM) count.
2. **Module validations:** For Perturbation, Inference, and Weights, it executes both the CPU version and the custom CUDA kernel version. It validates the output mathematically (verifying Max Absolute Error `PASS`) to ensure the custom kernels are analytically correct.
3. **Pipeline Summary:** Outputs the latency speedup (e.g., `2.93x`) that the GPU kernels achieved over the traditional CPU approaches.
