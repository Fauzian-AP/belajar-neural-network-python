"""
Bagian Pengelolaan Loss Function, yaitu:

Angka yg digunakan untuk mengukur seberapa salah atau error yg dihasilkan pd sebuah Prediksi Model.
"""

from abc import ABC, abstractmethod
from pydantic import validate_call, model_validator
from typing_extensions import Any

from src.utils.custom_types import ListFloat, SequenceFloat

# ======================
# === BLUEPRINT LOSS ===
# ======================

class Loss(ABC):
  # VALIDATE INPUT — Decorator yg memvalidasi Input
  @staticmethod
  def validate_inputs(func):
    @validate_call
    @model_validator(mode='after')
    def wrapper(self_validator: Any) -> Any:
      targets = getattr(self_validator, 'targets', None)
      predictions = getattr(self_validator, 'predictions', None)

      # Validasi input
      if targets is not None or predictions is not None:
        # Validasi panjang input
        if len(targets) != len(predictions):
          raise ValueError(f"Panjang targets ({len(targets)}) dgn predictions ({len(predictions)}) tdk cocok.")

      return self_validator

    return validate_call(func, config={"arbitrary_types_allowed": True})

  # DUNDER — Menjalankan Method setelah Initialization yaitu menjalankan aktivasi
  @abstractmethod
  def __call__(
    self,
    targets: SequenceFloat,
    predictions: SequenceFloat,
  ) -> float:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method __call__().")

  # GRADIENT — Menghitung Loss pd Gradient
  @abstractmethod
  def gradient(
    self,
    targets: SequenceFloat,
    predictions: SequenceFloat,
  ) -> ListFloat:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method gradient().")

# ====================
# === METODE² LOSS ===
# ====================

# Mean Squared Error — 

class MSE(Loss):
  @Loss.validate_inputs
  def __call__(
    self,
    targets: SequenceFloat,
    predictions: SequenceFloat
  ) -> float:  
    """ MSE = (1/n) × Σ(y - ŷ)² """
    total = sum(
      (target - prediction) ** 2
      for target, prediction in zip(targets, predictions)
    )
  
    return total / len(targets)

  @Loss.validate_inputs
  def gradient(
    self,
    targets: SequenceFloat,
    predictions: SequenceFloat
  ) -> ListFloat:
    """ ∂MSE/∂ŷ = (2/n) × (ŷ - y) """
    return [
      2 * (prediction - target) / len(targets)

      for target, prediction in zip(targets, predictions)
    ]


# Mean Absolute Error Menghitung nilai rata² jarak absolut error

class MAE(Loss):
  @Loss.validate_inputs
  def __call__(
    self,
    targets: SequenceFloat,
    predictions: SequenceFloat
  ) -> float:  
    """ MAE = (1/n) × Σ|y - ŷ| """
    total_error = sum(
      abs(target - prediction)

      for target, prediction in zip(targets, predictions)
    )
  
    return total_error / len(targets)

  @Loss.validate_inputs
  def gradient(
    self,
    targets: SequenceFloat,
    predictions: SequenceFloat,
  ) -> ListFloat:
    return [
      (
        1.0
        if prediction > target
        else -1.0
        if predictions < target
        else 0.0
      )
      for target, prediction in zip(targets, predictions)
    ]
