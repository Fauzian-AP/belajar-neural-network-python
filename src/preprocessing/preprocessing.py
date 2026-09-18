""" Bagian Preprocessing Data sebelum Digunakan oleh Model """

from .normalization import (
  Normalizer,
  MinMaxNormalizer
)

from src.utils.custom_types import (
  FloatVector,
  FloatSequence,
  Dataset,
  DatasetType,
)

class Preprocessor:
  # CONSTRUCTOR — Initialization
  def __init__(
    self,
    input_scaler: Normalizer | None = None,
    target_scaler: Normalizer | None = None,
  ) -> None:
    # Simpan Normalizer untuk Input
    self.input_scaler: Normalizer = input_scaler or MinMaxNormalizer()

    # Simpan Normalizer untuk Target
    self.target_scaler: Normalizer = target_scaler or MinMaxNormalizer()

  # NORMALIZE INPUTS — Normalisasi Kumpulan Nilai Input
  def normalize_inputs(self, values: FloatSequence) -> FloatVector:
    return [
      # Normalisasikan tiap nilai Input Scaler
      self.input_scaler.normalize(value)
  
      # Jalankan berdasarkan panjang values
      for value in values
    ]

  # NORMALIZE TARGET — Normalisasi Kumpulan Nilai Target
  def normalize_targets(self, values: FloatSequence) -> FloatVector:
    return [
      # Normalisasikan tiap nilai Target Scaler
      self.target_scaler.normalize(value)
  
      # Jalankan berdasarkan panjang values
      for value in values
    ]

  # NORMALIZE DATASET — Normalisasi sebuah Dataset
  def normalize_dataset(self, dataset: Dataset) -> Dataset:
    # Siapkan Wadah Dataset
    normalized_dataset: Dataset = []
  
    # Proses tiap Data
    for inputs, targets in dataset:
      # Normalisasi Input
      normalized_inputs = self.normalize_inputs(inputs)
  
      # Normalisasi Target
      normalized_targets = self.normalize_targets(targets)
  
      # Simpan Data yg sdh di Normalisasi
      normalized_dataset.append(
        (normalized_inputs, normalized_targets)
      )
  
    return normalized_dataset
  
  # NORMALIZE DATASETS — Normalisasi byk Dataset
  def normalize_datasets(self, datasets: DatasetType) -> DatasetType:
    # Siapkan Wadah Dataset
    normalized_datasets: DatasetType = {}
  
    # Proses tiap Dataset
    for name, dataset in datasets.items():
      # Normalisasikan Dataset
      normalized_datasets[name] = self.normalize_dataset(dataset)
  
    return normalized_datasets

  # DENORMALIZE INPUTS — Nengembalikan Kumpulan Nilai Input ke Skala Asli
  def denormalize_inputs(self, values: FloatSequence) -> FloatVector:
    return [
      # Denormalisasikan tiap nilai Input Scaler
      self.input_scaler.denormalize(value)
  
      # Jalankan berdasarkan panjang values
      for value in values
    ]
  
  # DENORMALIZE TARGETS — Nengembalikan Kumpulan Nilai Target ke Skala Asli
  def denormalize_targets(self, values: FloatSequence) -> FloatVector:
    return [
      # Denormalisasikan tiap nilai Target Scaler
      self.target_scaler.denormalize(value)
  
      # Jalankan berdasarkan panjang values
      for value in values
    ]
  
  # DENORMALIZE DATASET — Nengembalikan sebuah Dataset ke Skala Asli
  def denormalize_dataset(self, dataset: Dataset) -> Dataset:
    # Siapkan Wadah Dataset
    denormalized_dataset: Dataset = []
  
    # Proses tiap Data
    for inputs, targets in dataset:
      # Denormalisasi Input
      denormalized_inputs = self.denormalize_inputs(inputs)
  
      # Denormalisasi Target
      denormalized_targets = self.denormalize_targets(targets)
  
      # Simpan Data yg sdh di kembalikan
      denormalized_dataset.append(
        (denormalized_inputs, denormalized_targets)
      )
  
    return denormalized_dataset
  
  # DENORMALIZE DATASETS — Mengembalikan byk Dataset ke Skala Asli
  def denormalize_datasets(self, datasets: DatasetType) -> DatasetType:
    # Siapkan Wadah Dataset
    denormalized_datasets: DatasetType = {}
  
    # Proses tiap Dataset
    for name, dataset in datasets.items():
      # Denormalisasikan Dataset
      denormalized_datasets[name] = self.denormalize_dataset(dataset)
  
    return denormalized_datasets    

  # FIT DATASET — Mencari nilai minimum & maximum dari training data
  def fit_dataset(self, dataset: Dataset) -> None:
    # Wadah seluruh nilai Input
    input_values: FloatVector = []

    # Wadah seluruh nilai Target
    target_values: FloatVector = []

    # Ambil Data dari Dataset
    for inputs, targets in dataset:
      # Gabungkan seluruh data Input
      input_values.extend(inputs)

      # Gabungkan seluruh data Target
      target_values.extend(targets)

    # Fit Gabungan Input ke Normalizer
    self.input_scaler.fit(input_values)

    # Fit Gabungan Target ke Normalizer
    self.target_scaler.fit(target_values)

  # PREPROCESS — Menjalankan seluruh Proses Preprocessor
  def preprocess(self, datasets: DatasetType) -> DatasetType:
    # Fit Normalizer
    self.fit_dataset(datasets["training"])

    # Normalize Datasets
    return self.normalize_datasets(datasets)