"""
Bagian Pengelolaan Batch, yaitu:

Kumpulan data² yg diproses secara bersamaan.
"""

import math
import random

from src.utils.custom_types import (
  IntPositive,
  FloatPositive,
  Dataset,
  ListDataset,
  DatasetType,
)

class Batch:
  # CONSTRUCTOR — Initialization
  def __init__(self, seed: int | None = None) -> None:
    # Simpan Random Generator untuk Reproducibility
    self.random = random.Random(seed)

  # SPLIT DATASET — Membagi Dataset menjadi Sesi² seperti Training, Validation, dan Testing
  def split_dataset(
    self,
    dataset: Dataset,
    training_ratio: FloatPositive,
    validation_ratio: FloatPositive,
    testing_ratio: FloatPositive,
  ) -> DatasetType:
    # Hitung total Rasio
    total_ratio = training_ratio + validation_ratio + testing_ratio

    # Validasi Total Ratio adalah 1
    if not math.isclose(total_ratio, 1.0):
      raise ValueError("Total Rasio hrs sama dgn 1.")

    # Salin Dataset agar yg asli tdk berubah
    shuffle_dataset = list(dataset)

    # Acak data agar mengurangi ketergantungan Model pd pola urutan
    self.random.shuffle(shuffle_dataset)

    # Hitung total data
    dataset_size = len(shuffle_dataset)
    training_size = int(dataset_size * training_ratio)
    validation_size = int(dataset_size * validation_ratio)

    # Tentukan Index
    training_end = training_size
    validation_end = training_end + validation_size

    # Pisahkan Dataset
    training_data = shuffle_dataset[:training_end]
    validation_data = shuffle_dataset[training_end:validation_end]
    testing_data = shuffle_dataset[validation_end:]

    # Kembalikan Dataset yg telah dipisah
    return {
      "training": training_data,
      "validation": validation_data,
      "testing": testing_data,
    }

  # CREATE BATCH — Membagi Dataset menjadi beberapa Mini-Batch
  def create_batches(self, dataset: Dataset, batch_size: IntPositive) -> ListDataset:
    # Salin Dataset agar yg asli tdk berubah
    shuffle_dataset = list(dataset)

    # Acak data agar mengurangi ketergantungan Model pd pola urutan
    self.random.shuffle(shuffle_dataset)

    # Siapkan Wadah Batch
    batches: ListDataset = []

    # Proses pembagian Dataset berdasarkan pembagian ukuran Batch
    for i in range(0, len(shuffle_dataset), batch_size):
      # Slicing Index
      batch = shuffle_dataset[i : i + batch_size]

      # Simpan Mini-Batch
      batches.append(batch)

    return batches