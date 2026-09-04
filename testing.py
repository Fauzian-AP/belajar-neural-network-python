# ============================================================
# END-TO-END TEST
# ============================================================

from batch import Batch
from preprocessing import Preprocessor
from trainer import Trainer
from model import Model
from metrics import calculate_metrics


# ============================================================
# DATASET
# ============================================================

dataset = [
  ([1.0, 2.0, 3.0], [1.0, 2.0]),
  ([2.0, 3.0, 4.0], [2.0, 3.0]),
  ([3.0, 4.0, 5.0], [3.0, 4.0]),
  ([4.0, 5.0, 6.0], [4.0, 5.0]),
  ([5.0, 6.0, 7.0], [5.0, 6.0]),
  ([6.0, 7.0, 8.0], [6.0, 7.0]),
  ([7.0, 8.0, 9.0], [7.0, 8.0]),
  ([8.0, 9.0, 10.0], [8.0, 9.0]),
  ([9.0, 10.0, 11.0], [9.0, 10.0]),
  ([10.0, 11.0, 12.0], [10.0, 11.0]),
]


# ============================================================
# TEST BATCH SPLIT
# ============================================================

batch = Batch(seed=42)

datasets = batch.split_dataset(
  dataset=dataset,
  training_ratio=0.6,
  validating_ratio=0.2,
  testing_ratio=0.2,
)

training_data = datasets["training"]
validating_data = datasets["validating"]
testing_data = datasets["testing"]

assert len(training_data) == 6
assert len(validating_data) == 2
assert len(testing_data) == 2

assert (
  len(training_data)
  + len(validating_data)
  + len(testing_data)
  == len(dataset)
)

print("✅ E2E BATCH SPLIT — PASS")


# ============================================================
# TEST PREPROCESSING
# ============================================================

preprocessor = Preprocessor()

normalized_datasets = preprocessor.preprocess(
  datasets
)

normalized_training = normalized_datasets[
  "training"
]

normalized_validating = normalized_datasets[
  "validating"
]

normalized_testing = normalized_datasets[
  "testing"
]

# Pastikan jumlah data tetap sama
assert len(normalized_training) == len(training_data)
assert len(normalized_validating) == len(validating_data)
assert len(normalized_testing) == len(testing_data)

# Pastikan Normalizer berasal dari Training Data
assert preprocessor.input_scaler.minimum == min(
  value
  for inputs, _ in training_data
  for value in inputs
)

assert preprocessor.input_scaler.maximum == max(
  value
  for inputs, _ in training_data
  for value in inputs
)

assert preprocessor.target_scaler.minimum == min(
  value
  for _, targets in training_data
  for value in targets
)

assert preprocessor.target_scaler.maximum == max(
  value
  for _, targets in training_data
  for value in targets
)

print("✅ E2E PREPROCESSING — PASS")


# ============================================================
# TEST MODEL + TRAINER
# ============================================================

model = Model(
  learning_rate=0.01,
)

trainer = Trainer(
  model=model,
  batch=batch,
)

result = trainer.fit(
  training_data=normalized_training,
  validating_data=normalized_validating,
  epochs=100,
  batch_size=2,
  patience=10,
)

assert result["best_epoch"] >= 1

assert result["last_epoch"] >= 1

assert result["best_epoch"] <= result["last_epoch"]

assert result["best_validating_mse"] >= 0.0

print("✅ E2E TRAINING — PASS")


# ============================================================
# TEST NORMALIZED TESTING DATA
# ============================================================

normalized_test_metrics = trainer.evaluating(
  normalized_testing
)

assert normalized_test_metrics["MSE"] >= 0.0
assert normalized_test_metrics["MAE"] >= 0.0
assert normalized_test_metrics["RMSE"] >= 0.0

print("✅ E2E NORMALIZED TESTING — PASS")


# ============================================================
# DENORMALIZE PREDICTION + TARGET
# ============================================================

total_mse = 0.0
total_mae = 0.0
total_rmse = 0.0

for inputs, targets in normalized_testing:

  # Prediction dalam skala normalized
  predictions = model.forward(inputs)

  # Kembalikan Target ke skala original
  original_targets = (
    preprocessor.denormalize_targets(
      targets
    )
  )

  # Kembalikan Prediction ke skala original
  original_predictions = (
    preprocessor.denormalize_targets(
      predictions
    )
  )

  # Hitung Metrics pada original scale
  metrics = calculate_metrics(
    original_targets,
    original_predictions,
  )

  total_mse += metrics["MSE"]
  total_mae += metrics["MAE"]
  total_rmse += metrics["RMSE"]


testing_size = len(normalized_testing)

original_test_metrics = {
  "MSE": total_mse / testing_size,
  "MAE": total_mae / testing_size,
  "RMSE": total_rmse / testing_size,
}

assert original_test_metrics["MSE"] >= 0.0
assert original_test_metrics["MAE"] >= 0.0
assert original_test_metrics["RMSE"] >= 0.0

print("✅ E2E ORIGINAL SCALE TESTING — PASS")


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=== END-TO-END RESULT ===")

print(
  f"Best Epoch          : "
  f"{result['best_epoch']}"
)

print(
  f"Last Epoch          : "
  f"{result['last_epoch']}"
)

print(
  f"Best Validating MSE : "
  f"{result['best_validating_mse']:.6f}"
)

print(
  f"Testing MSE         : "
  f"{original_test_metrics['MSE']:.6f}"
)

print(
  f"Testing MAE         : "
  f"{original_test_metrics['MAE']:.6f}"
)

print(
  f"Testing RMSE        : "
  f"{original_test_metrics['RMSE']:.6f}"
)

print()
print("🎉 END-TO-END TEST BERHASIL.")