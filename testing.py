import random

from model import Model
from trainer import Trainer

from normalizer import (
  Normalizer,
  normalize_list,
  normalize_datasets,
)

from metrics import calculate_metrics

random.seed(42)

# =========================
# DATA TRAINING
# =========================

training_data = [
  ([1, 2, 3], [12, 17]),
  ([2, 4, 6], [18, 45]),
  ([3, 6, 9], [34, 55]),
  ([4, 8, 12], [37, 85]),
  ([5, 10, 15], [58, 92]),
  ([6, 12, 18], [55, 130]),
  ([7, 14, 21], [76, 132]),
  ([8, 16, 24], [75, 170]),
]

# =========================
# DATA VALIDATION
# =========================

validation_data = [
  ([1.5, 3, 4.5], [15, 30]),
  ([2.5, 5, 7.5], [25, 50]),
  ([3.5, 7, 10.5], [35, 70]),
  ([4.5, 9, 13.5], [45, 90]),
]

# =========================
# NORMALIZER
# =========================

input_normalizer = Normalizer()
target_normalizer = Normalizer()

input_values = []
target_values = []

for inputs, targets in training_data:
  input_values.extend(inputs)
  target_values.extend(targets)

# FIT NORMALIZER

input_normalizer.fit(input_values)
target_normalizer.fit(target_values)

print("=== NORMALIZER ===")
print("Input Min  :", input_normalizer.minimum)
print("Input Max  :", input_normalizer.maximum)
print("Target Min :", target_normalizer.minimum)
print("Target Max :", target_normalizer.maximum)
print()

# =========================
# NORMALIZE DATASET
# =========================

datasets = {
  "training_data": training_data,
  "validation_data": validation_data,
}

normalize_datasets = normalize_datasets(
  datasets,
  input_normalizer,
  target_normalizer,
)

normalized_training_data = normalize_datasets["training_data"]
normalized_validation_data = normalize_datasets["validation_data"]

# =========================
# MODEL
# =========================

model = Model(learning_rate=0.0005)
trainer = Trainer(model)

# =========================
# TRAINING
# =========================

print("=== TRAINING ===")

result = trainer.fit(
  normalized_training_data,
  normalized_validation_data,
  epochs=5000,
  batch_size=3,
  patience=10
)

# =========================
# TRAINING RESULT
# =========================

print()
print("=== TRAINING RESULT ===")
print(f"Best Epoch          : {result["best_epoch"]}")
print(f"Best Validation MSE : {result["best_validation_mse"]}")
print(f"Stopped Epoch       : {result["last_epoch"]}")
print()

# =========================
# NORMALIZED METRICS
# =========================

training_metrics = trainer.evaluate(normalized_training_data)
validation_metrics = trainer.evaluate(normalized_validation_data)

print("=== NORMALIZED METRICS ===")

print()

print("Training")

print(f"MSE  : {training_metrics["mse"]}")
print(f"MAE  : {training_metrics["mae"]}")
print(f"RMSE : {training_metrics["rmse"]}")

print()

print("Validation")
print(f"MSE  : {validation_metrics["mse"]}")
print(f"MAE  : {validation_metrics["mae"]}")
print(f"RMSE : {validation_metrics["rmse"]}")

print()

# =========================
# DENORMALIZATION TEST
# =========================

print("=== DENORMALIZATION TEST ===")

for inputs, targets in validation_data:
  # Normalize input
  normalized_inputs = normalize_list(inputs, input_normalizer)

  # Prediction dalam skala normalized
  prediction_normalized = model.forward(normalized_inputs)

  # Kembalikan prediction ke skala asli
  prediction = [
    target_normalizer.denormalize(value)
    for value in prediction_normalized
  ]

  # Target normalized
  target_normalized = normalize_list(targets, target_normalizer)

  print()

  print(f"Input                 : {inputs}")
  print(f"Target Asli           : {targets}")
  print(f"Target Normalized     : {target_normalized}")
  print(f"Prediction Normalized : {prediction_normalized}")
  print(f"Prediction Asli       : {prediction}")

print()

# =========================
# ORIGINAL SCALE METRICS
# =========================

print("=== ORIGINAL SCALE METRICS ===")

total_mse = total_mae = total_rmse = 0

for inputs, targets in validation_data:
  # NORMALIZE INPUT
  normalized_inputs = normalize_list(inputs, input_normalizer)

  # MODEL PREDICTION
  prediction_normalized = model.forward(normalized_inputs)

  # DENORMALIZE PREDICTION
  predictions = [
    target_normalizer.denormalize(value)
    for value in prediction_normalized
  ]

  # CALCULATE METRICS
  metrics = calculate_metrics(targets, predictions)

  total_mse += metrics["mse"]
  total_mae += metrics["mae"]
  total_rmse += metrics["rmse"]

# =========================
# AVERAGE METRICS
# =========================

dataset_size = len(validation_data)

original_mse = total_mse / dataset_size
original_mae = total_mae / dataset_size
original_rmse = total_rmse / dataset_size

# =========================
# DISPLAY
# =========================

print()

print("Validation Original Scale")
print(f"MSE  : {original_mse}")
print(f"MAE  : {original_mae}")
print(f"RMSE : {original_rmse}")