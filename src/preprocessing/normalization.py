"""
Bagian Pengelolaan Normalisasi Data, yaitu:

Proses mengubah skala Data agar lebih sesuai untuk digunakan oleh Model Neural Network.
"""

from abc import ABC, abstractmethod

from src.utils.custom_types import (
  FloatSequence,
)

# ============================
# === BLUEPRINT NORMALIZER ===
# ============================

class Normalizer(ABC):
  # CONSTRUCTOR — Initialization
  def __init__(self) -> None:
    # Menentukan apakah Normalizer sdh di-Fit
    self._is_fitted: bool = False

  # IS FITTED — Getter untuk mengetahui apakah sdh melakukan Fit
  @property
  def is_fitted(self) -> bool:
    return self._is_fitted

  # VALIDATE FITTED — Memastikan Normalizer telah di-Fit
  def _validate_fitted(self) -> None:
    if not self._is_fitted:
      raise ValueError("Normalizer hrs menjalankan fit() dahulu.")

  # FIT — Mempelajari parameter berdasarkan Data
  @abstractmethod
  def fit(self, values: FloatSequence) -> None:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method fit().")

  # NORMALIZE — Membuat sebuah nilai menjadi skala Normalisasi
  @abstractmethod
  def normalize(self, value: float) -> float:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method normalize().")

  # DENORMALIZE — Mengembalikan sebuah nilai ke skala Original
  @abstractmethod
  def denormalize(self, value: float) -> float:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method denormalize().")

# ==========================
# === METODE² NORMALIZER ===
# ==========================

# Min Max — Menghasilkan nilai dgn rentan antara 0 sampai 1 selama berada dlm nilai minimum & maximum

class MinMaxNormalizer(Normalizer):
  # CONSTRUCTOR — Initialization
  def __init__(self) -> None:
    # Jalankan Constructor Normalizer
    super().__init__()

    # Nilai Minimum Data
    self.minimum: float | None = None

    # Nilai Maximum Data
    self.maximum: float | None = None

  # CHECK MIN MAX — Memastikan Property Minimum & Maximum ada
  def _validate_min_max(self) -> None:
    if (
      self.minimum is None or
      self.maximum is None
    ):
      raise ValueError("Nilai Min & Max tdk tersedia.")

  # FIT — Menetukan nilai Minimum & Maximum dari kumpulan data
  def fit(self, values: FloatSequence) -> None:
    # Cari nilai Min & Max dari kumpulan data
    min_value = min(values)
    max_value = max(values)

    # Validasi Min & Max tdk boleh sama
    if min_value == max_value:
      raise ValueError("Minimum dgn Maximum tdk boleh sama.")

    # Simpan nilai Min & Max
    self.minimum = min_value
    self.maximum = max_value

    # Set bahwa Normalizer sdh di-Fit
    self._is_fitted = True

  def normalize(self, value: float) -> float:
    # Validasi Cek Fit
    self._validate_fitted()

    # Validasi Cek Min & Max
    self._validate_min_max()

    """ x' = (x - x_min) / (x_max - x_min) """
    return (value - self.minimum) / (self.maximum - self.minimum)

  def denormalize(self, value: float) -> float:
    # Validasi Cek Fit
    self._validate_fitted()

    # Validasi Cek Min & Max
    self._validate_min_max()

    """ x = x' × (x_max - x_min) + x_min """
    return value * (self.maximum - self.minimum) + self.minimum
