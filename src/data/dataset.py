"""Bagian Pengelolaan Data² yang akan digunakan dalam melatih Model."""

import numpy as np

from src.validation import runtime_function

from src.utils.custom_types import (
  NumSequence,
  FloatVector,
  DataSample,
  Dataset,
)


# ==============
# === HELPER ===
# ==============

# CREATE FLOAT VECTOR — Membuat NumPy Vector dgn dtype float64.

@runtime_function(strict=True)
def create_float_vector(values: NumSequence) -> FloatVector:
  # Konversi Values menjadi Numpy Array float64
  return np.asarray(values, dtype=np.float64)


# CREATE SAMPLE DATA —  Membuat sebuah DataSample.

@runtime_function(strict=True)
def create_data_sample(inputs: NumSequence, targets: NumSequence) -> DataSample:
  # Buat menggunakan Base Model Pydantic
  return DataSample(
    inputs=create_float_vector(inputs),   # Input Features
    targets=create_float_vector(targets),   # Expected Output
  )


# ===============
# === Dataset ===
# ===============

dataset = Dataset(
  # TRAINING — Model belajar dari data ini.
  training=[
    create_data_sample([1, 2, 3], [10, 20]),
    create_data_sample([2, 4, 6], [20, 40]),
    create_data_sample([3, 6, 9], [30, 60]),
    create_data_sample([4, 8, 12], [40, 80]),
    create_data_sample([5, 10, 15], [50, 100]),
    create_data_sample([6, 12, 18], [60, 120]),
    create_data_sample([7, 14, 21], [70, 140]),
    create_data_sample([8, 16, 24], [80, 160]),
    create_data_sample([9, 18, 27], [90, 180]),
    create_data_sample([10, 20, 30], [100, 200]),
  ],

  # VALIDATION — Mengecek perkembangan Model selama eksperimen.
  validation=[
    create_data_sample([11, 22, 33], [110, 220]),
    create_data_sample([12, 24, 36], [120, 240]),
    create_data_sample([13, 26, 39], [130, 260]),
  ],

  # TESTING — Evaluasi final pada data yang tidak digunakan untuk mengambil keputusan selama training.
  testing=[
    create_data_sample([14, 28, 42], [140, 280]),
    create_data_sample([15, 30, 45], [150, 300]),
    create_data_sample([16, 32, 48], [160, 320]),
  ],

  # GENERALIZATION — Menguji kemampuan Model pada input baru yg masih mengikuti pola data.
  generalization=[
    create_data_sample([5, 10, 15], [50, 100]),
    create_data_sample([5.5, 11, 16.5], [55, 110]),
    create_data_sample([7.5, 15, 22.5], [75, 150]),
    create_data_sample([9.5, 19, 28.5], [95, 190]),
  ],
)