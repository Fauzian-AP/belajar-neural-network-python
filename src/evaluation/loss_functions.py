"""
Bagian Pengelolaan Loss Function, yaitu:

Angka yg digunakan untuk mengukur seberapa Error (salah) yg dihasilkan pd sebuah Prediksi Model.
"""

import numpy as np

from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

from src.utils.custom_types import (
  FloatArray,
)

P = ParamSpec("P")
R = TypeVar("R")

# ======================
# === BLUEPRINT LOSS ===
# ======================

class Loss(ABC):
  # VALIDATE ARGUMENS — Decorator yg memvalidasi hubungan antara Argument
  @staticmethod
  def validate_arguments(func: Callable[P, R]) -> Callable[P, R]:
    # Membungkus Function asli dgn proses tambahan
    @wraps(func)
    def wrapper(
      self: Any,
      targets: FloatArray,
      predictions: FloatArray
    ) -> R:
      # Validasi Shape Argument
      if targets.shape != predictions.shape:
        # Lempar Error
        raise ValueError(
          f"Shape targets ({targets.shape}) dgn"
          f"Shape predictions ({predictions.shape}) tdk cocok."
        )

      # Jalankan Function asli setelah validasi berhasil
      return func(self, targets, predictions)

    # Kembalikan Function yg sdh dibungkus validasi
    return wrapper

  # DUNDER — Menghitung Loss setelah Initialization
  @abstractmethod
  def __call__(
    self, 
    targets: FloatArray, 
    predictions: FloatArray,
  ) -> float:
    # Lempar Error
    raise NotImplementedError("Sub Class hrs mengimplementasikan method __call__().")

  # GRADIENT — Menghitung Loss pd Gradient
  @abstractmethod
  def gradient(
    self, 
    targets: FloatArray, 
    predictions: FloatArray,
  ) -> FloatArray:
    # Lempar Error
    raise NotImplementedError("Sub Class hrs mengimplementasikan method gradient().")

# ====================
# === METODE² LOSS ===
# ====================

# Mean Squared Error — Menghitung rata² kuadrat selisih antara Target dgn Prediction sehingga Error yg bsr diberi Penalti yg lbh bsr

class MSE(Loss):
  @Loss.validate_arguments
  def __call__(self, targets: FloatArray, predictions: FloatArray) -> float:  
    """ MSE = (1/n) × Σ(y - ŷ)² """

    errors = targets - predictions

    return float(np.mean(errors ** 2))

  @Loss.validate_arguments
  def gradient(self, targets: FloatArray, predictions: FloatArray) -> FloatArray:
    """
    ∂MSE/∂ŷ = (2/n) × (ŷ - y)
    """
    return 2.0 * (predictions - targets) / targets.size


# Mean Absolute Error — Menghitung rata² jarak absolut antara Target dgn Prediction sehingga lbh tahan terhadap Outlier

class MAE(Loss):
  @Loss.validate_arguments
  def __call__(self, targets: FloatArray, predictions: FloatArray) -> float:  
    """ MAE = (1/n) × Σ|y - ŷ| """

    errors = targets - predictions

    return float(np.abs(errors).mean())

  @Loss.validate_arguments
  def gradient(self, targets: FloatArray, predictions: FloatArray) -> FloatArray:
    """
    ∂MAE/∂ŷ = sign(ŷ - y) / n
    """
    return np.sign(predictions - targets) / targets.size
