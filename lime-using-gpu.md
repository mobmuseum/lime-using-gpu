## Implementing LIME from Scratch with GPU Acceleration (CUDA / OpenCL)


## 1. Project Objective

The objective of this project is to implement the **LIME (Local Interpretable Model-agnostic Explanations)** algorithm from scratch, with a progressive focus on **GPU acceleration** of its core components.

Students will incrementally build a LIME pipeline by:

1. Accelerating computationally intensive components using low-level GPU programming
2. Validating correctness using a high-level machine learning framework
3. Replacing high-level abstractions with custom GPU implementations

The project emphasizes:

* Understanding the internal mechanics of LIME
* Identifying parallelizable workloads
* Gaining experience with low-level GPU programming
* Evaluating trade-offs between abstraction, performance, and correctness

---

## 2. Reference LIME Pipeline

For a given input instance, LIME performs the following steps:

1. Generate a large set of perturbed samples around the input
2. Run the original model’s inference on the perturbed samples
3. Compute similarity-based weights between the original input and perturbed samples
4. Train a local surrogate model using weighted samples
5. Use surrogate model parameters as feature attributions

Each milestone in this project corresponds to specific stages of this pipeline.

---

## 3. Milestones

---

## Milestone 1 (M1): GPU-based Perturbation, Inference, and Weight Computation

### Objective

Implement the computational core of LIME using **explicit GPU programming**, without relying on any machine learning frameworks.

This milestone focuses on identifying and accelerating the most parallel components of LIME.

---

### Requirements

#### Perturbation Module (GPU)

* Generate a large number of perturbed samples per input instance
* Support basic perturbation strategies such as:

  * Feature masking
  * Noise injection
* Implement perturbation generation using CUDA or OpenCL kernels

#### Base Model Inference (GPU)

* Implement a simple binary classification model (e.g., logistic regression or shallow neural network)
* Forward-pass inference only
* Execute inference on all perturbed samples in parallel on the GPU

#### Weight Computation (GPU)

* Compute similarity-based weights between the original input and each perturbed sample
* Use a distance metric such as Euclidean or cosine distance
* Implement kernel-based weighting functions directly on the GPU

---

### Deliverables (M1)

* Source code for GPU kernels handling perturbation, inference, and weight computation
* CPU vs GPU output comparison to validate correctness
* Performance measurements for GPU-accelerated components
* Technical documentation describing kernel design and data flow

---

## Milestone 2 (M2): Surrogate Model Training Using High-Level Frameworks

### Objective

Complete the full LIME pipeline by training and evaluating the surrogate model using a **high-level machine learning framework**, with emphasis on correctness and interpretability.

This milestone serves as a validation reference before low-level surrogate training is implemented.

---

### Requirements

#### Data Preparation

* Use perturbed samples, model predictions, and weights generated in Milestone 1
* Construct a training dataset for the surrogate model

#### Surrogate Model Training

* Train one local surrogate model per explained instance
* Use a simple and interpretable model, preferably:

  * Weighted linear regression
  * Or a shallow neural network
* Use a framework such as PyTorch for model training and evaluation

#### Explanation Generation

* Extract feature attributions from the trained surrogate model
* Visualize and interpret explanation outputs
* Validate explanations using feature ablation or consistency checks

---

### Deliverables (M2)

* Framework-based surrogate training code
* Generated explanations and visualizations
* Validation results demonstrating explanation stability
* Comparison between CPU-based and GPU-accelerated LIME components

---

## Milestone 3 (M3): GPU-based Surrogate Model Training

### Objective

Replace the framework-based surrogate training from Milestone 2 with a **custom GPU implementation**, resulting in a fully GPU-native LIME pipeline.

This milestone emphasizes low-level system design and numerical computation.

---

### Requirements

#### GPU-based Surrogate Training

* Implement weighted linear regression on the GPU using:

  * A closed-form solution, or
  * An iterative optimization method such as gradient descent
* Execute training in a batched and parallel manner

#### GPU-based Explanation Extraction

* Extract surrogate model coefficients directly on the GPU
* Minimize data transfers between device and host

#### End-to-End GPU LIME

* Execute the full LIME pipeline using:

  * GPU-based perturbation
  * GPU-based inference
  * GPU-based weight computation
  * GPU-based surrogate training

---

### Deliverables (M3)

* CUDA or OpenCL implementation of surrogate model training
* End-to-end GPU-native LIME demonstration
* Performance comparison between:

  * CPU-only implementation
  * Framework-assisted implementation
  * Fully GPU-native implementation
* Final technical report discussing design decisions, numerical stability, and performance trade-offs

---

## 4. Technical Constraints

* Deep learning frameworks:

  * Not permitted in Milestones 1 and 3
  * Permitted only in Milestone 2
* NumPy may be used for CPU-side numerical utilities
* GPU programming must be done explicitly using CUDA or OpenCL
* Code must be modular, well-documented, and reproducible

---

## 5. Evaluation Criteria

Evaluation will emphasize:

* Correct implementation of LIME mechanics
* Proper identification and use of GPU-parallel workloads
* Correctness and stability of explanations
* Clear system design and code structure
* Quality of documentation and analysis

Performance improvements are considered beneficial but are secondary to correctness and conceptual understanding.

---

## 6. Summary

This project provides hands-on experience in explainable machine learning and GPU computing by guiding students from high-level algorithm validation to low-level GPU implementation. The milestone-based structure ensures progressive development while maintaining a clear focus on system design, parallel computation, and interpretability.