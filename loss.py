# Bagian Pengelolaan Loss Function, yaitu Angka yg digunakan untuk mengukur seberapa salah Prediksi Model

import math
from pydantic import validate_call

from custom_types import ListFloat, SequenceFloat

# Validasi Input
def _validate_inputs(targets: SequenceFloat, predictions: SequenceFloat) -> None:
  if not targets or not predictions:
    raise ValueError("targets dan predictions tdk boleh kosong.")

  if len(targets) != len(predictions):
    raise ValueError(f"Panjang targets ({len(targets)}) dan predictions ({len(predictions)}) tdk cocok.")

# MSE (Mean Squared Error) — Menghitung seberapa besar Error

@validate_call
def MSE(targets: SequenceFloat, predictions: SequenceFloat) -> float:
  # Validasi
  _validate_inputs(targets, predictions)

  # Rumus: MSE = (1/Total) × Σ(Target - Prediksi)²
  total = sum(
    (target - prediction) ** 2
    for target, prediction in zip(targets, predictions)
  )

  return total / len(targets)

# MSE Gradient — Menetukan Update Weight & Bias

@validate_call
def MSE_gradient(targets: SequenceFloat, predictions: SequenceFloat) -> ListFloat:
  # Validasi
  _validate_inputs(targets, predictions)

  # Rumus: Gradient = (2/Total) × (Prediksi - Target)
  return [
    (2 * (prediction - target)) / len(targets)
    for target, prediction in zip(targets, predictions)
  ]

# MAE (Mean Absolute Error) — Menghitung rata² jarak absolut error

@validate_call
def MAE(targets: SequenceFloat, predictions: SequenceFloat) -> float:
  # Validasi
  _validate_inputs(targets, predictions)

  # Rumus: MAE = (1/Total) × Σ|Target - Prediksi|
  total_error = sum(
    abs(target - prediction)
    for target, prediction in zip(targets, predictions)
  )

  return total_error / len(targets)

# RMSE (Root Mean Squared Error) — Hasil MSE di-akar sehingga satuannya kembali sama dgn Target

@validate_call
def RMSE(targets: SequenceFloat, predictions: SequenceFloat) -> float:
  # Rumus: RMSE = √MSE
  return math.sqrt(MSE(targets, predictions))