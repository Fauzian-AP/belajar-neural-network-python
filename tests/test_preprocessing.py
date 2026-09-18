from src.preprocessing.preprocessing import Preprocessor
from src.preprocessing.normalization import (
  Normalizer,
  MinMaxNormalizer,
)

# DATASET

training_data = [
  ([10.0, 20.0, 30.0], [40.0, 50.0]),
  ([20.0, 30.0, 40.0], [50.0, 60.0]),
]

validation_data = [
  ([30.0, 40.0, 50.0], [60.0, 70.0]),
]

testing_data = [
  ([40.0, 50.0, 60.0], [70.0, 80.0]),
]

datasets = {
  "training": training_data,
  "validation": validation_data,
  "testing": testing_data,
}


# CONSTRUCTOR

preprocessor = Preprocessor()

assert isinstance(
  preprocessor.input_scaler,
  MinMaxNormalizer,
)

assert isinstance(
  preprocessor.target_scaler,
  MinMaxNormalizer,
)

print("✓ Constructor berhasil\n")


# FIT DATASET

preprocessor.fit_dataset(training_data)

assert preprocessor.input_scaler.minimum == 10.0
assert preprocessor.input_scaler.maximum == 40.0

assert preprocessor.target_scaler.minimum == 40.0
assert preprocessor.target_scaler.maximum == 60.0

print("✓ Fit Dataset berhasil\n")


# NORMALIZE INPUTS

normalized_inputs = (
  preprocessor.normalize_inputs(
    [10.0, 25.0, 40.0],
  )
)

assert normalized_inputs == [0.0, 0.5, 1.0]

print("✓ Normalize Inputs berhasil\n")


# NORMALIZE TARGETS

normalized_targets = (
  preprocessor.normalize_targets(
    [40.0, 50.0, 60.0],
  )
)

assert normalized_targets == [0.0, 0.5, 1.0]

print("✓ Normalize Targets berhasil\n")


# NORMALIZE DATASET

normalized_dataset = (
  preprocessor.normalize_dataset(
    training_data,
  )
)

assert normalized_dataset == [
  ([0.0, 0.3333333333333333, 0.6666666666666666], [0.0, 0.5]),
  ([0.3333333333333333, 0.6666666666666666, 1.0], [0.5, 1.0]),
]

print("✓ Normalize Dataset berhasil\n")


# NORMALIZE DATASETS

normalized_datasets = (
  preprocessor.normalize_datasets(datasets)
)

assert set(normalized_datasets.keys()) == {"training", "validation", "testing"}

print("✓ Normalize Datasets berhasil\n")


# DENORMALIZE INPUTS

denormalized_inputs = (
  preprocessor.denormalize_inputs(
    [0.0, 0.5, 1.0],
  )
)

assert denormalized_inputs == [10.0, 25.0, 40.0]

print("✓ Denormalize Inputs berhasil\n")


# DENORMALIZE TARGETS

denormalized_targets = (
  preprocessor.denormalize_targets(
    [0.0, 0.5, 1.0],
  )
)

assert denormalized_targets == [40.0, 50.0, 60.0]

print("✓ Denormalize Targets berhasil\n")


# DENORMALIZE DATASET

denormalized_dataset = (
  preprocessor.denormalize_dataset(
    normalized_dataset,
  )
)

assert denormalized_dataset == training_data

print("✓ Denormalize Dataset berhasil\n")


# DENORMALIZE DATASETS

denormalized_datasets = (
  preprocessor.denormalize_datasets(
    normalized_datasets,
  )
)

assert set(denormalized_datasets.keys()) == {"training", "validation", "testing"}

print("✓ Denormalize Datasets berhasil\n")


# PREPROCESS

preprocessor_2 = Preprocessor()

processed_datasets = (
  preprocessor_2.preprocess(datasets)
)

assert preprocessor_2.input_scaler.minimum == 10.0
assert preprocessor_2.input_scaler.maximum == 40.0
assert preprocessor_2.target_scaler.minimum == 40.0
assert preprocessor_2.target_scaler.maximum == 60.0

assert set(processed_datasets.keys()) == {"training", "validation", "testing"}

print("✓ Preprocess berhasil\n")


# FIT HANYA TRAINING DATA

# Validation memiliki nilai 50, sedangkan Training maksimum hanya 40.
# Jika Fit menggunakan seluruh Dataset, maximum akan menjadi 50.
# Karena Fit hanya menggunakan Training, maximum harus tetap 40.

assert preprocessor_2.input_scaler.maximum == 40.0

print("✓ Fit hanya menggunakan Training berhasil\n")


# NOT FITTED

not_fitted = Preprocessor()

try:
  not_fitted.normalize_inputs([10.0, 20.0],)
except Exception as error:
  print(f"Pesan Error: {error}")
finally:
  print()


# CUSTOM SCALER

custom_input_scaler = MinMaxNormalizer()
custom_target_scaler = MinMaxNormalizer()

custom_preprocessor = Preprocessor(
  input_scaler=custom_input_scaler,
  target_scaler=custom_target_scaler,
)

assert custom_preprocessor.input_scaler is custom_input_scaler
assert custom_preprocessor.target_scaler is custom_target_scaler

print("✓ Custom Scaler berhasil\n")


# RESULT

print("Semua Test Preprocessor Berhasil.")