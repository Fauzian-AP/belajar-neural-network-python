"""
Bagian Pengelolaan Loss Function, yaitu:

Angka yg digunakan untuk mengukur seberapa Error (salah) yg dihasilkan pd sebuah Prediksi Model.
"""

from abc import ABC, abstractmethod
from functools import wraps
from typing_extensions import Any, Callable

from src.utils.custom_types import (
  FloatVector,
  FloatSequence,
)

# ======================
# === BLUEPRINT LOSS ===
# ======================

class Loss(ABC):
  # VALIDATE ARGUMENS — Decorator yg memvalidasi hubungan antara Argument
  @staticmethod
  def validate_arguments(func: Callable[..., Any]) -> Callable[..., Any]:
    # Melanjutkan informasi function yg dibungkus
    @wraps(func)
    def wrapper(
      self,
      targets: FloatSequence,
      predictions: FloatSequence,
    ) -> Any:
      # Validasi panjang input
      if len(targets) != len(predictions):
        raise ValueError(f"Panjang targets ({len(targets)}) dgn predictions ({len(predictions)}) tdk cocok.")

      return func(self, targets, predictions)

    return wrapper

  # DUNDER — Menjalankan Method setelah Initialization yaitu menghitung Loss
  @abstractmethod
  def __call__(
    self,
    targets: FloatSequence,
    predictions: FloatSequence,
  ) -> float:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method __call__().")

  # GRADIENT — Menghitung Loss pd Gradient
  @abstractmethod
  def gradient(
    self,
    targets: FloatSequence,
    predictions: FloatSequence,
  ) -> FloatVector:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method gradient().")

# ====================
# === METODE² LOSS ===
# ====================

# Mean Squared Error — Menghitung rata² kuadrat selisih antara Target dgn Prediction sehingga Error yg bsr diberi Penalti yg lbh bsr

class MSE(Loss):
  @Loss.validate_arguments
  def __call__(
    self,
    targets: FloatSequence,
    predictions: FloatSequence,
  ) -> float:  
    """ MSE = (1/n) × Σ(y - ŷ)² """
    total = sum(
      (target - prediction) ** 2
      for target, prediction in zip(targets, predictions)
    )

    return total / len(targets)

  @Loss.validate_arguments
  def gradient(
    self,
    targets: FloatSequence,
    predictions: FloatSequence,
  ) -> FloatVector:
    """ ∂MSE/∂ŷ = (2/n) × (ŷ - y) """
    return [
      2.0 * (prediction - target) / len(targets)

      for target, prediction in zip(targets, predictions)
    ]


# Mean Absolute Error — Menghitung rata² jarak absolut antara Target dgn Prediction sehingga lbh tahan terhadap Outlier

class MAE(Loss):
  @Loss.validate_arguments
  def __call__(
    self,
    targets: FloatSequence,
    predictions: FloatSequence,
  ) -> float:  
    """ MAE = (1/n) × Σ|y - ŷ| """
    total_error = sum(
      abs(target - prediction)

      for target, prediction in zip(targets, predictions)
    )
  
    return total_error / len(targets)

  @Loss.validate_arguments
  def gradient(
    self,
    targets: FloatSequence,
    predictions: FloatSequence,
  ) -> FloatVector:
    """ ∂MAE/∂ŷ = sign(ŷ - y) """
    return [
      (
        1.0
        if prediction > target
        else -1.0
        if prediction < target
        else 0.0
      )

      for target, prediction in zip(targets, predictions)
    ]
