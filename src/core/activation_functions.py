# Bagian Pengelolaan Activation Function yaitu:
# Fungsi Aktivasi pd Neural Network yg menentukan Output suatu Neuron berdasarkan nilai Pre Activation

from abc import ABC, abstractmethod
from pydantic import validate_call

from src.utils.custom_types import Numeric, AlphaRange

# BLUEPRINT ACTIVATION

class Activation(ABC):
  # DUNDER — Menjalankan Method utama yaitu menghitung aktivasi
  @abstractmethod
  def __call__(self, pre_activation: Numeric) -> float:
    raise NotImplementedError("Method aktivasi hrs dibuat.")

  # GRADIENT — Menghitung Gradient dari aktivasi
  @abstractmethod
  def gradient(self, pre_activation: Numeric) -> float:
    raise NotImplementedError("Method aktivasi gradient hrs dibuat.")


# RECTIFIED LINEAR UNIT ACTIVATION

class ReLU(Activation):
  # DUNDER — Menghitung aktivasi ReLU
  def __call__(self, pre_activation: Numeric) -> float:
    """
    f(x) = { x jika x > 0
           { 0 jika x ⩽ 0
    """
    return max(0.0, pre_activation)

  # GRADIENT — Menghitung aktivasi gradient ReLU
  def gradient(self, pre_activation: Numeric) -> float:
    """
    f'(x) = { 1 jika x > 0
            { 0 jika x ⩽ 0
    """
    return (
      1.0 if pre_activation > 0.0 else 0.0
    )


# LEAKY RECTIFIED LINEAR UNIT ACTIVATION

class LeakyReLU(Activation):
  # CONSTRUCTOR — Initialization
  @validate_call
  def __init__(self, alpha: AlphaRange = 0.01) -> None:
    # Nilai Alpha menentukan seberapa bsr bagian negatif yg tetap dilewatkan oleh aktivasi
    self._alpha = alpha

  # ALPHA — Getter untuk mendapatkan nilai dari Alpha
  @property
  def alpha(self) -> float:
    return self._alpha

  # DUNDER — Menghitung aktivasi Leaky ReLU
  def __call__(self, pre_activation: Numeric) -> float:
    """
    f(x) = { x  jika x > 0
           { αx jika x ⩽ 0
    """
    return (
      pre_activation if pre_activation > 0.0 else self._alpha * pre_activation
    )

  # GRADIENT — Menghitung aktivasi gradient Leaky ReLU
  def gradient(self, pre_activation: Numeric) -> float:
    """
    f'(x) = { 1 jika x > 0
            { α jika x ⩽ 0
    """
    return (
      1.0 if pre_activation > 0.0 else self._alpha
    )