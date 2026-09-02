# Normalisasi Data merupakan proses mengatur dan menstrukturkan data agar lebih rapi

# Teknik Min Max Normalization — Menghasilkan nilai dgn rentan antara 0 sampai 1 selama berada dlm nilai minimum & maximum

from pydantic import validate_call

from custom_types import SequenceFloat

class Normalizer:
  # CONSTRUCTOR — Initialization
  def __init__(self) -> None:
    self.minimum: float | None = None
    self.maximum: float | None = None

  # FIT — Menetukan batas minimum & maximum dari kumpulan data
  @validate_call
  def fit(self, values: SequenceFloat) -> None:
    # Validasi
    if not values:
      raise ValueError("values tdk boleh kosong.")

    minimum = min(values)   # Cari nilai terkecil
    maximum = max(values)   # Cari nilai terbesar

    # Validasi
    if minimum == maximum:
      raise ValueError("minimum dgn maximum tdk boleh sama.")

    # Simpan nilai² nya
    self.minimum = minimum
    self.maximum = maximum

  # VALIDASI FIT — Memastikan Normalizer telah di Fit
  def _validate_fitted(self) -> None:
    # Pastikan input tdk kosong
    if self.minimum is None or self.maximum is None:
      raise ValueError("Normalisasi hrs menjalankan Fit dulu.")

  # NORMALIZE — Membuat sebuah nilai menjadi skala Normalisasi
  def normalize(self, value: float) -> float:
    # Validasi
    self._validate_fitted()

    # Rumus: x' = (x - x_min) / (x_max - x_min)
    return (value - self.minimum) / (self.maximum - self.minimum)

  # DENORMALIZE — Mengembalikan sebuah nilai ke skala Original
  def denormalize(self, value: float) -> float:
    # Validasi
    self._validate_fitted()

    # Rumus: x = x' × (x_max - x_min) + x_min
    return value * (self.maximum - self.minimum) + self.minimum