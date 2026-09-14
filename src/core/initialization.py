"""
Bagian Pengelolaan Initialization 
"""

import math
import random

from abc import ABC, abstractmethod
from pydantic import validate_call

from src.utils.custom_types import (
  ListFloat,
  PositiveInt,
)

# ====================================
# === BLUEPRINT WEIGHT INITIALIZER ===
# ====================================

class Initializer(ABC):
  # DUNDER — Menjalankan Method setelah Initialization yaitu Weight menggunakan Initializer
  @abstractmethod
  def __call__(self, input_size: PositiveInt) -> ListFloat:
    raise NotImplementedError("Sub Class hrs mengimplementasi method __call__().")


# ===========================
# === METODE² INITIALIZER ===
# ===========================

# He / Kaiming — Menentukan nilai awal Weight yg sesuai skala dgn jumlah Input

class HeNormal(Initializer):
  @validate_call
  def __call__(self, input_size: PositiveInt) -> ListFloat:
    """ σ = √(2 / nᵢₙ) """

    # Menghitung Standar Deviasi
    stddev = math.sqrt(2.0 / input_size)

    return [  
      # Generate Float Random antara skala Gaussian dgn Mean 0.0 dan Standar Deviasi
      random.gauss(0.0, stddev)

      # Berdasarkan jumlah Input
      for _ in range(input_size)
    ]