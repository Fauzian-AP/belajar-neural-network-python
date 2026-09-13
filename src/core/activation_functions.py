"""
Bagian Pengelolaan 'Activation Function' yaitu:

Fungsi Aktivasi pd NN yg digunakan menentukan Output suatu Neuron berdasarkan nilai Pre-Activation.
"""

from abc import ABC, abstractmethod
from pydantic import validate_call

from src.utils.custom_types import Numeric, AlphaRange

# ============================
# === BLUEPRINT ACTIVATION ===
# ============================

class Activation(ABC):
  # DUNDER — Menjalankan Method setelah Initialization yaitu menghitung Activation
  @abstractmethod
  def __call__(self, pre_activation: Numeric) -> float:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method __call__().")

  # GRADIENT — Menghitung Gradient dari Activation
  @abstractmethod
  def gradient(self, pre_activation: Numeric) -> float:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method gradient().")

# ==========================
# === METODE² ACTIVATION ===
# ==========================

# Linear — Meneruskan nilai apa adanya tanpa non-linearitas

class Linear(Activation):
  def __call__(self, pre_activation: Numeric) -> float:
    """ f(z) = z """
    return float(pre_activation)

  def gradient(self, _: Numeric) -> float:
    """ f'(z) = 1 """
    return 1.0


# Rectified Linear Unit — Meneruskan nilai positif dan mengubah nilai negatif menjadi 0

class ReLU(Activation):
  def __call__(self, pre_activation: Numeric) -> float:
    """ f(z) = max(0, z) """
    return max(0.0, pre_activation)

  def gradient(self, pre_activation: Numeric) -> float:
    """
    f'(z) = { 1, jika z > 0
            { 0, jika z ⩽ 0
    """
    return 1.0 if pre_activation > 0.0 else 0.0


# Leaky Rectified Linear Unit — Meneruskan nilai positif dan sebagian kecil nilai negatif

class LeakyReLU(Activation):
  # CONSTRUCTOR — Initialization
  @validate_call
  def __init__(self, alpha: AlphaRange = 0.01) -> None:
    # Nilai Alpha menentukan seberapa bsr nilai negatif yg tetap dpt dilewati
    self._alpha: AlphaRange = alpha

  # ALPHA — Getter untuk mendapatkan nilai Alpha
  @property
  def alpha(self) -> float:
    return self._alpha

  def __call__(self, pre_activation: Numeric) -> float:
    """
    f(z) = { z,  jika z > 0
           { αz, jika z ⩽ 0
    """
    return (
      pre_activation if pre_activation > 0.0 else self._alpha * pre_activation
    )

  def gradient(self, pre_activation: Numeric) -> float:
    """
    f'(z) = { 1 jika z > 0
            { α jika z ⩽ 0
    """
    return 1.0 if pre_activation > 0.0 else self._alpha
