"""
Bagian pengelolaan Optimalisasi yaitu:

Menyesuaikan Weight & Bias secara berulang-ulang untuk meminimalkan nilai Loss berdasarkan Gradient.
"""

from abc import ABC, abstractmethod

from src.utils.custom_types import (
  FloatVector,
  FloatPositive,
)

# ===========================
# === BLUEPRINT OPTIMIZER ===
# ===========================

class Optimizer(ABC):
  # UPDATE — Proses penyesuaian Weight & Bias berdasarkan Gradient
  @abstractmethod
  def __call__(
    self,
    weights: FloatVector,
    bias: float,
    gradient_weights: FloatVector,
    gradient_bias: float,
  ) -> tuple[FloatVector, float]:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method __call__().")

# =========================
# === METODE² OPTIMIZER ===
# =========================

# Stochastic Gradient Descent — Memperbarui Weight & Bias dgn bergerak berlawanan arah terhadap Gradient Loss agar nilai Loss cenderung menurun.

class SGD(Optimizer):
  # CONSTRUCTOR — Initialization
  def __init__(self, learning_rate: FloatPositive) -> None:
    # Hyper Parameter untuk menentukan seberapa bsr perubahan Weight & Bias dlm tiap Update
    self.learning_rate: FloatPositive = learning_rate

  def __call__(
    self,
    weights: FloatVector,
    bias: float,
    gradient_weights: FloatVector,
    gradient_bias: float,
  ) -> tuple[FloatVector, float]:
    """ Update Weights: w_new = w - η × ∂L/∂w """
    updated_weights = [
      weight - (self.learning_rate * gradient)
      for weight, gradient in zip(weights, gradient_weights)
    ]

    """ Update Bias: b_new = b - η × ∂L/∂b """
    updated_bias = bias - (self.learning_rate * gradient_bias)

    return updated_weights, updated_bias