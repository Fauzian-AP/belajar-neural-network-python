"""
Bagian pengelolaan 'Optimizer', yaitu:

Menyesuaikan Weight & Bias secara berulang-ulang untuk meminimalkan nilai Loss berdasarkan Gradient.
"""

from abc import ABC, abstractmethod

from src.utils.custom_types import (
  FloatArray,
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
    W: FloatArray,
    b: FloatArray,
    gradient_W: FloatArray,
    gradient_b: FloatArray,
  ) -> tuple[FloatArray, FloatArray]:
    # Lempar Error
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
    W: FloatArray,
    b: FloatArray,
    gradient_W: FloatArray,
    gradient_b: FloatArray,
  ) -> tuple[FloatArray, FloatArray]:
    """
    Update Weight: w_new = w - η × ∂L/∂w
    """
    updated_W = W - (self.learning_rate * gradient_W)

    """
    Update Bias: b_new = b - η × ∂L/∂b
    """
    updated_b = b - (self.learning_rate * gradient_b)

    return (updated_W, updated_b)