# Normalisasi Data merupakan proses mengatur dan menstrukturkan data agar lebih rapi

# Teknik Min Max Normalization — Menghasilkan nilai dgn rentan antara 0 sampai 1 selama berada dlm nilai minimum & maximum

from pydantic import validate_call

from custom_types import (
  Dataset,
  DictDataset,
  ListFloat,
  SequenceFloat,
)

class Normalizer:
  # CONSTRUCTOR — Initialization
  def __init__(self) -> None:
    self.minimum: float | None = None
    self.maximum: float | None = None

  # FIT — Menetukan nilai batas dari kumpulan data
  @validate_call
  def fit(self, values: SequenceFloat) -> None:
    # Validasi input
    if not values:
      raise ValueError("values tdk boleh kosong.")

    minimum = min(values)   # Cari nilai terkecil
    maximum = max(values)   # Cari nilai terbesar

    # Validasi input
    if minimum == maximum:
      raise ValueError("minimum dgn maximum tdk boleh sama.")

    # Simpan nilai²
    self.minimum = minimum
    self.maximum = maximum

  # VALIDASI FIT — Memastikan Normalizer telah di Fit
  def _validate_fitted(self) -> None:
    # Pastikan input tdk kosong
    if self.minimum is None or self.maximum is None:
      raise ValueError("Normalisasi hrs menjalankan Fit dulu.")

  # NORMALIZE — Menormalisasikan sebuah data
  def normalize(self, value: float) -> float:
    # Validasi
    self._validate_fitted()

    # Rumus: x' = (x - x_min) / (x_max - x_min)
    return (value - self.minimum) / (self.maximum - self.minimum)

  # DENORMALIZE — Mengembalikan sebuah data yg telah di Normalisasi
  def denormalize(self, value: float) -> float:
    # Validasi
    self._validate_fitted()

    # Rumus: x = x' × (x_max - x_min) + x_min
    return value * (self.maximum - self.minimum) + self.minimum

# LIST HELPER

# Normalisasi Kumpulan Data
def normalize_list(values: SequenceFloat, normalizer: Normalizer) -> ListFloat:
  return [
    # Jalankan Method normalize
    normalizer.normalize(value)

    # Berdasarkan panjang values
    for value in values
  ]

# Denormalisasi Kumpulan Data
def denormalize_list(values: SequenceFloat, normalizer: Normalizer) -> ListFloat:
  return [
    # Jalankan Method denormalize
    normalizer.denormalize(value)

    # Berdasarkan panjang values
    for value in values
  ]

# DATASET HELPER

# Normalisasi sebuah Dataset
def normalize_dataset(
  dataset: Dataset,
  input_normalizer: Normalizer,
  target_normalizer: Normalizer,
) -> Dataset:
  # Siapkan Wadah List
  normalized_dataset: Dataset = []

  # Jalankan tiap nilai pd Dataset
  for inputs, targets in dataset:
    # Normalisasi Input² Data
    normalized_inputs = normalize_list(inputs, input_normalizer)

    # Normalisasi Target² Data
    normalized_targets = normalize_list(targets, target_normalizer)

    # Simpan Input & Target Data ke Wadah yg sdh dibuat
    normalized_dataset.append((normalized_inputs, normalized_targets))

  return normalized_dataset

# HELPER MULTI DATASET

# Normalisasi byk Dataset sekaligus
def normalize_datasets(
  datasets: DictDataset,
  input_normalizer: Normalizer,
  target_normalizer: Normalizer,
) -> DictDataset:
  # Siapkan Wadah Dictionary
  normalized_datasets: DictDataset = {}

  # Jalankan tiap Item dari Dataset
  for name, dataset in datasets.items():
    # Normalisasikan Dataset
    normalized_datasets[name] = (
      normalize_dataset(dataset, input_normalizer, target_normalizer)
    )

  return normalized_datasets