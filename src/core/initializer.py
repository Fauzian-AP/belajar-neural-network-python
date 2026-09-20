"""
Bagian Pengelolaan 'Initialization', yaitu:

Menentukan Metode dan menghasilkan nilai awal Weight yg akan digunakan oleh Neuron.
"""

import numpy as np

from abc import ABC, abstractmethod

from src.utils.custom_types import (
  FloatArray,
  IntPositive,
)

# =============================
# === BLUEPRINT INITIALIZER ===
# =============================

class Initializer(ABC):
  # DUNDER — Generate Weights setelah Initialization
  @abstractmethod
  def __call__(
    self,
    shape: tuple[IntPositive, ...],   # Pola: (Jumlah_input, jumlah_neuron)
  ) -> FloatArray:
    # Lempar Error
    raise NotImplementedError("Sub Class hrs mengimplementasi method __call__().")

# ===========================
# === METODE² INITIALIZER ===
# ===========================

# He/Kaiming Normal — Menentukan nilai Weight awal menggunakan Distribusi Gaussian.

class HeNormal(Initializer):
  def __call__(
    self,
    shape: tuple[IntPositive, ...],
  ) -> FloatArray:
    """
    σ = √(2 / fan_in)
    """

    # Validasi Argument
    if len(shape) < 2:
      raise ValueError("Shape Weight minimal hrs memiliki 2 dimensi.")

    # Ambil jumlah Input yg msk ke tiap Neuron 
    fan_in = shape[0]

    # Menghitung Standard Deviation
    stddev = np.sqrt(2.0 / fan_in)

    # Gunakan Generator Random
    rng = np.random.default_rng()

    # Generate Weight berupa skala Gaussian dgn Mean 0.0 dan Standar Deviasi
    return rng.normal(loc=0.0, scale=stddev, size=shape)
