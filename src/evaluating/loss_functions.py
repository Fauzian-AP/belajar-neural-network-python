# Bagian Pengelolaan Loss Function, yaitu Angka yg digunakan untuk mengukur seberapa salah Prediksi Model

from abc import ABC, abstractmethod

from src.utils.custom_types import ListFloat, SequenceFloat


# BLUEPRINT LOSS

class Loss(ABC):
  # VALIDATE INPUT
  def _validate_inputs(
    self,
    targets: SequenceFloat,
    predictions: SequenceFloat
  ) -> None:
    # Validasi input
    if not targets or not predictions:
      raise ValueError("targets & predictions tdk boleh kosong.")

    # Validasi panjang input
    if len(targets) != len(predictions):
      raise ValueError(f"Panjang targets ({len(targets)}) dgn predictions ({len(predictions)}) tdk cocok.")

  # DUNDER — Menjalankan Method utama yaitu menghitung aktivasi
  @abstractmethod
  def __call__(
    self,
    targets: SequenceFloat,
    predictions: SequenceFloat,
  ) -> float:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method loss.")

  # GRADIENT — Menghitung Loss pd Gradient
  @abstractmethod
  def gradient(
    self,
    targets: SequenceFloat,
    predictions: SequenceFloat,
  ) -> ListFloat:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method loss gradient.")


# Mean Squared Error

class MSE(Loss):
  # DUNDER  — Menghitung nilai Loss dari MSE
  def __call__(
    self,
    targets: SequenceFloat,
    predictions: SequenceFloat
  ) -> float:
    # Validasi
    self._validate_inputs(targets, predictions)
  
    """ MSE = (1/n) × Σ(y - ŷ)² """
    total = sum(
      (target - prediction) ** 2
      for target, prediction in zip(targets, predictions)
    )
  
    return total / len(targets)

  # GRADIENT — Menghitung Gradient Loss sehingga dpt ditentukan Update Weight & Bias nya
  def gradient(
    self,
    targets: SequenceFloat,
    predictions: SequenceFloat
  ) -> ListFloat:
    # Validasi
    self._validate_inputs(targets, predictions)
  
    """ ∂MSE/∂ŷ = (2/n) × (ŷ - y) """
    return [
      (2 * (prediction - target)) / len(targets)
      for target, prediction in zip(targets, predictions)
    ]


# Mean Absolute Error 

class MAE(Loss):
  # DUNDER — Menghitung nilai rata² jarak absolut error
  def __call__(
    self,
    targets: SequenceFloat,
    predictions: SequenceFloat
  ) -> float:
    # Validasi
    self._validate_inputs(targets, predictions)
  
    """ MAE = (1/n) × Σ|y - ŷ| """
    total_error = sum(
      abs(target - prediction)
      for target, prediction in zip(targets, predictions)
    )
  
    return total_error / len(targets)