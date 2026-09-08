# Bagian Pengelolaan Loss Function, yaitu Angka yg digunakan untuk mengukur seberapa salah Prediksi Model

from pydantic import validate_call

from custom_types import ListFloat, SequenceFloat

# Metode: MSE — Mean Squared Error

class MSE:
  # VALIDATE INPUT
  @staticmethod
  @validate_call
  def _validate_inputs(targets: SequenceFloat, predictions: SequenceFloat) -> None:
    if not targets or not predictions:
      raise ValueError("targets & predictions tdk boleh kosong.")
  
    if len(targets) != len(predictions):
      raise ValueError(f"Panjang targets ({len(targets)}) dgn predictions ({len(predictions)}) tdk cocok.")

  # CALCULATE  — Menghitung nilai Loss dari MSE
  @staticmethod
  @validate_call
  def calculate(targets: SequenceFloat, predictions: SequenceFloat) -> float:
    # Validasi
    MSE._validate_inputs(targets, predictions)
  
    """ MSE = (1/n) × Σ(y - ŷ)² """
    total = sum(
      (target - prediction) ** 2
      for target, prediction in zip(targets, predictions)
    )
  
    return total / len(targets)

  # GRADIENT — Menghitung Gradient Loss sehingga dpt ditentukan Update Weight & Bias nya
  @staticmethod
  @validate_call
  def gradient(targets: SequenceFloat, predictions: SequenceFloat) -> ListFloat:
    # Validasi
    MSE._validate_inputs(targets, predictions)
  
    """ ∂MSE/∂ŷ = (2/n) × (ŷ - y) """
    return [
      (2 * (prediction - target)) / len(targets)
      for target, prediction in zip(targets, predictions)
    ]