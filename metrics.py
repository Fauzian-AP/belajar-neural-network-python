# Bagian Metrics, yaitu ukuran yg gunakan untuk mengetahui seberapa bagus performa Model

import math
from pydantic import validate_call

from custom_types import SequenceFloat, Metrics
from loss import MSE

# METODE: MAE – Mean Absolute Error 

class MAE:
  # VALIDATE INPUT
  @staticmethod
  @validate_call
  def _validate_inputs(targets: SequenceFloat, predictions: SequenceFloat) -> None:
    if not targets or not predictions:
      raise ValueError("targets & predictions tdk boleh kosong.")
  
    if len(targets) != len(predictions):
      raise ValueError(f"Panjang targets ({len(targets)}) dgn predictions ({len(predictions)}) tdk cocok.")      

  # CALCULATE — Menghitung nilai rata² jarak absolut error
  @staticmethod
  @validate_call
  def calculate(targets: SequenceFloat, predictions: SequenceFloat) -> float:
    # Validasi
    MAE._validate_inputs(targets, predictions)
  
    """ MAE = (1/n) × Σ|y - ŷ| """
    total_error = sum(
      abs(target - prediction)
      for target, prediction in zip(targets, predictions)
    )
  
    return total_error / len(targets)


# METODE: RMSE — Root Mean Squared Error

class RMSE:
  # CALCULATE — Hasil MSE di-akar sehingga satuannya kembali sama dgn Target
  @staticmethod
  @validate_call
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