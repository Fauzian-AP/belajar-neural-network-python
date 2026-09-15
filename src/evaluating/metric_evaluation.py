"""
Bagian Pengelolaan Metric, yaitu:

Ukuran yg gunakan untuk mengetahui seberapa bagus Performa sebuah Model.
"""

import math
from abc import ABC, abstractmethod
from pydantic import validate_call, model_validator
from typing_extensions import Any

from src.utils.custom_types import ListFloat, SequenceFloat, Metrics

# ========================
# === BLUEPRINT METRIC ===
# ========================

class Metric(ABC):
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

# ======================
# === METODE² METRIC ===
# ======================

#  Mean Squared Error — 

class MSE(Metrics):
  @Metrics.validate_inputs
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


# Mean Absolute Error Menghitung nilai rata² jarak absolut error

class MAE(Metrics):
  @Metrics.validate_inputs
  def calculate(targets: SequenceFloat, predictions: SequenceFloat) -> float:
    """ MAE = (1/n) × Σ|y - ŷ| """
    total_error = sum(
      abs(target - prediction)
      for target, prediction in zip(targets, predictions)
    )
  
    return total_error / len(targets)


# Root Mean Squared Error — Hasil MSE di-akar sehingga satuannya kembali sama dgn Target

class RMSE(Metrics):
  @Metrics.validate_inputs
  def calculate(targets: SequenceFloat, predictions: SequenceFloat) -> float:
    """ RMSE = √MSE """
    mse = MSE.calculate(targets, predictions)

    return math.sqrt(mse)


# PERHITUNGAN METRICS

@validate_call
def calculate_metrics(targets: SequenceFloat, predictions: SequenceFloat) -> Metrics:
  return {
    "MSE": MSE.calculate(targets, predictions),
    "MAE": MAE.calculate(targets, predictions),
    "RMSE": RMSE.calculate(targets, predictions),
  }