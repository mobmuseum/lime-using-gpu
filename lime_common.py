from dataclasses import dataclass, field
import numpy as np

BLOCK_SIZE = 256   # CUDA threads per block
KEEP_PROB  = 0.5   # Probability a feature is kept in each perturbation

# Returns an appropriate kernel bandwidth for the given metric:
def adaptive_kernel_width(n_features: int, metric: str = "euclidean") -> float:
    if metric == "cosine":
        return 0.25
    return 0.75 * (n_features ** 0.5)

# Holds perturbed samples and their binary masks on the GPU.
@dataclass
class PerturbationBuffer:
    n_samples:  int
    n_features: int
    # GPU arrays (cupy) set after generation
    d_samples: object = field(default=None, repr=False)  # [n_samples × n_features] float32
    d_masks:   object = field(default=None, repr=False)  # [n_samples × n_features] int32

# Simple binary logistic regression model stored on GPU.
@dataclass
class LogisticModel:
    n_features: int
    d_weights:  object = field(default=None, repr=False)  # [n_features] float32
    bias:       float  = 0.0

# Stores inference predictions, distances, and kernel weights on GPU.
@dataclass
class WeightBuffer:
    n_samples:    int
    d_predictions: object = field(default=None, repr=False)  # [n_samples] float32
    d_distances:   object = field(default=None, repr=False)  # [n_samples] float32
    d_weights:     object = field(default=None, repr=False)  # [n_samples] float32


class DistanceMetric:
    EUCLIDEAN = "euclidean"
    COSINE    = "cosine"
