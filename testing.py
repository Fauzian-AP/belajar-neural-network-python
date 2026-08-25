import random

from model import Model
from trainer import Trainer

from normalizer import (
  Normalizer,
  normalize_list
)

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
# VALIDATION
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

input_normalizer.fit(input_values)
target_normalizer.fit(target_values)


# =========================
# NORMALIZE DATASET
# =========================

def normalize_dataset(
  dataset,
  input_normalizer,
  target_normalizer
):
  normalized_dataset = []

  for inputs, targets in dataset:
    normalized_inputs = normalize_list(
      inputs,
      input_normalizer
    )

    normalized_targets = normalize_list(
      targets,
      target_normalizer
    )

    normalized_dataset.append(
      (normalized_inputs, normalized_targets)
    )

  return normalized_dataset


normalized_training_data = normalize_dataset(
  training_data,
  input_normalizer,
  target_normalizer
)

normalized_validation_data = normalize_dataset(
  validation_data,
  input_normalizer,
  target_normalizer
)


# =========================
# MODEL
# =========================

model = Model(
  learning_rate=0.0005
)

trainer = Trainer(model)


# =========================
# TRAINING
# =========================

print("=== OVERFITTING NOISE TEST ===")

trainer.fit(
  normalized_training_data,
  normalized_validation_data,
  epochs=5000,
  batch_size=3
)


# =========================
# FINAL METRICS
# =========================

training_metrics = trainer.evaluate(
  normalized_training_data
)

validation_metrics = trainer.evaluate(
  normalized_validation_data
)


print()
print("=== FINAL METRICS ===")

print("Training MSE  :", training_metrics["mse"])
print("Training MAE  :", training_metrics["mae"])
print("Training RMSE :", training_metrics["rmse"])

print()

print("Validation MSE  :", validation_metrics["mse"])
print("Validation MAE  :", validation_metrics["mae"])
print("Validation RMSE :", validation_metrics["rmse"])