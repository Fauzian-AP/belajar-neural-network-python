"""
Bagian Pengelolaan Metric, yaitu:

Ukuran yg digunakan untuk mengetahui seberapa bagus Performa sebuah Model berdasarkan hasil Prediksi terhadap Target.
"""

import math

from abc import ABC, abstractmethod
from functools import wraps
from typing_extensions import Any, Callable

from src.utils.custom_types import (
  FloatSequence,
  Metrics,
)

# ========================
# === BLUEPRINT METRIC ===
# ========================

class Metric(ABC):
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

  # DUNDER — Menjalankan Method setelah Initialization yaitu menghitung Metric
  @abstractmethod
  def __call__(
    self,
    targets: FloatSequence,
    predictions: FloatSequence,
  ) -> float:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method __call__().")

# ======================
# === METODE² METRIC ===
# ======================

#  Mean Squared Error — Menghitung rata² kuadrat selisih antara Target dgn Prediction sehingga Error yg bsr diberi Penalti yg lbh bsr

class MSE(Metric):
  @Metric.validate_arguments
  def __call__(
    self,
    targets: FloatSequence,
    predictions: FloatSequence
  ) -> float:  
    """ MSE = (1/n) × Σ(y - ŷ)² """
    total = sum(
      (target - prediction) ** 2
      for target, prediction in zip(targets, predictions)
    )
  
    return total / len(targets)


# Mean Absolute Error — Mengukur rata² jarak absolut Error Prediksi Model sehingga mdh diinterpretasikan dlm satuan Target

class MAE(Metric):
  @Metric.validate_arguments
  def __call__(
    self,
    targets: FloatSequence,
    predictions: FloatSequence
  ) -> float:
    """ MAE = (1/n) × Σ|y - ŷ| """
    total_error = sum(
      abs(target - prediction)
      for target, prediction in zip(targets, predictions)
    )
  
    return total_error / len(targets)


# Root Mean Squared Error — Hasil MSE di-akar sehingga satuannya kembali sama dgn Target

class RMSE(Metric):
  @Metric.validate_arguments
  def __call__(
    self,
    targets: FloatSequence,
    predictions: FloatSequence,
  ) -> float:
    """ RMSE = √MSE """
    rmse = math.sqrt(
      MSE()(targets, predictions)
    )

    return rmse


# CALCULATE METRICS

def calculate_metrics(
  targets: FloatSequence,
  predictions: FloatSequence
) -> Metrics:
  return {
    metric.__name__:
      metric()(targets, predictions)

    # Ambil otomatis tiap Sub Class dari Metric
    for metric in Metric.__subclasses__()
  }
  