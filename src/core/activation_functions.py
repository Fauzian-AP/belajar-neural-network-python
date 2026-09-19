"""
Bagian Pengelolaan 'Activation Function', yaitu:

Metode dari Fungsi Aktivasi yg digunakan untuk menentukan Output suatu Neuron berdasarkan nilai Pre-Activation.
"""

import numpy as np

from abc import ABC, abstractmethod

from src.utils.custom_types import(
  FloatArray,
  AlphaRange,
)

# ============================
# === BLUEPRINT ACTIVATION ===
# ============================

class Activation(ABC):
  # DUNDER — Menghitung Aktivasi Fungsi setelah Initialization
  @abstractmethod
  def __call__(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    # Lempar Error
    raise NotImplementedError("Sub Class hrs mengimplementasikan method __call__().")

  # GRADIENT — Menghitung Gradient dari Activation
  @abstractmethod
  def gradient(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    # Lempar Error
    raise NotImplementedError("Sub Class hrs mengimplementasikan method gradient().")

# ==========================
# === METODE² ACTIVATION ===
# ==========================

# Linear — Meneruskan nilai apa adanya tanpa non-linearitas.

class Linear(Activation):
  def __call__(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    """
    f(z) = z
    """
    return np.asarray(pre_activation, dtype=np.float64)

  def gradient(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    """
    f'(z) = 1
    """
    return np.ones_like(pre_activation, dtype=np.float64)


# Rectified Linear Unit — Meneruskan nilai positif dan mengubah nilai negatif menjadi 0.

class ReLU(Activation):
  def __call__(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    """
    f(z) = max(0, z)
    """
    return np.maximum(0.0, pre_activation)

  def gradient(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    """
    f'(z) = { 1, jika z > 0
            { 0, jika z ⩽ 0
    """
    return np.where(pre_activation > 0.0, 1.0, 0.0)


# Leaky Rectified Linear Unit — Nilai positif diteruskan dan sebagian kecil nilai negatif tetap diteruskan.

class LeakyReLU(Activation):
  # CONSTRUCTOR — Initialization
  def __init__(self, alpha: AlphaRange = 0.01) -> None:
    # Nilai Alpha menentukan seberapa bsr nilai negatif yg tetap dpt dilewati
    self.__alpha: AlphaRange = alpha

  # ALPHA — Getter untuk mendapatkan nilai Alpha
  @property
  def alpha(self) -> float:
    return self.__alpha

  def __call__(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    """
    f(z) = { z,  jika z > 0
           { αz, jika z ⩽ 0
    """
    return np.where(
      pre_activation > 0.0,
      pre_activation,
      self.__alpha * pre_activation,
    )

  def gradient(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    """
    f'(z) = { 1, jika z > 0
            { α, jika z ⩽ 0
    """
    return np.where(pre_activation > 0.0, 1.0, self.__alpha)