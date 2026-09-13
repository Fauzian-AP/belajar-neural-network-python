"""
Bagian pengelolaan Optimasi yaitu:

Menyesuaikan Weight & Bias secara berulang-ulang untuk meminimalkan nilai Loss berdasarkan Gradient.
"""

from abc import ABC, abstractmethod
from pydantic import validate_call

from src.utils.custom_types import (
  ListFloat,
  PositiveFloat
)

# ===========================
# === BLUEPRINT OPTIMIZER ===
# ===========================

class Optimizer(ABC):
  # UPDATE — Proses penyesuaian Weight & Bias setelah menjalani sebuah pelatihan
  @abstractmethod
  def update(
    self,
    weights: ListFloat,
    bias: float,
    gradient_weights: ListFloat,
    gradient_bias: float,
  ) -> tuple[ListFloat, float]:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method update().")


# =========================
# === METODE² OPTIMIZER ===
# =========================

# Stochastic Gradient Descent — Memperbarui Weight & Bias dgn bergerak berlawanan arah terhadap Gradient Loss agar nilai Loss cenderung menurun.

class SGD(Optimizer):
  # CONSTRUCTOR — Initialization
  @validate_call
  def __init__(self, learning_rate: PositiveFloat) -> None:
    # Menentukan seberapa besar perubahan Weight & Bias dalam setiap update.
    self.learning_rate: PositiveFloat = learning_rate

  def update(
    self,
    weights: ListFloat,
    bias: float,
    gradient_weights: ListFloat,
    gradient_bias: float,
  ) -> tuple[ListFloat, float]:
    """ w_baru = w - η × ∂L/∂w """
    updated_weights = [
      weight - (self.learning_rate * gradient)
      
      for weight, gradient in zip(weights, gradient_weights)
    ]

    """ b_baru = b - η × ∂L/∂b """
    updated_bias = bias - (self.learning_rate * gradient_bias)

    return updated_weights, updated_bias