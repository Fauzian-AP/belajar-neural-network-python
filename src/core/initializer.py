"""
Bagian Pengelolaan Initialization, yaitu:

Menentukan Metode dan menghasilkan nilai awal Weight yg akan digunakan oleh Neuron.
"""

import math
import random
from abc import ABC, abstractmethod

from src.utils.custom_types import FloatVector, IntPositive

# =============================
# === BLUEPRINT INITIALIZER ===
# =============================

class Initializer(ABC):
  # DUNDER — Menjalankan Method setelah Initialization yaitu Generate Weight menggunakan Initializer
  @abstractmethod
  def __call__(self, input_size: IntPositive) -> FloatVector:
    raise NotImplementedError("Sub Class hrs mengimplementasi method __call__().")

# ===========================
# === METODE² INITIALIZER ===
# ===========================

# He/Kaiming Normal — Menentukan nilai Weight awal menggunakan Distribusi Gaussian berbentuk Lonceng

class HeNormal(Initializer):
  def __call__(self, input_size: IntPositive) -> FloatVector:
    """ σ = √(2 / nᵢₙ) """

    # Menghitung Standar Deviasi
    stddev = math.sqrt(2.0 / input_size)

    return [
      # Generate Float Random antara skala Gaussian dgn Mean 0.0 dan Standar Deviasi
      random.gauss(0.0, stddev)

      # Berdasarkan jumlah Input
      for _ in range(input_size)
    ]