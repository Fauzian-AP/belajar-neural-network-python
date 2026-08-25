import random

from model import Model
from trainer import Trainer
from plot import plot_history

from dataset import (
  training_data,
  validation_data,
  test_data
)

from normalizer import (
  Normalizer,
  normalize_datasets,
)

# REPRODUCIBILITY
random.seed(42)

# NORMALIZER

input_normalizer = Normalizer()
target_normalizer = Normalizer()

# KELOMPOKAN NILAI TRAINING

input_values = []
target_values = []

for inputs, targets in training_data:
  input_values.extend(inputs)
  target_values.extend(targets)

# FIT NORMALIZER

input_normalizer.fit(input_values)
target_normalizer.fit(target_values)

# CEK NILAI MIN & MAX

print("=== Normalizer ===")
print("Input Min   :", input_normalizer.minimum)
print("Input Max   :", input_normalizer.maximum)
print("Target Min  :", target_normalizer.minimum)
print("Target Max  :", target_normalizer.maximum)
print()

# NORMALISASI DATASET

datasets = {
  "training": training_data,
  "validation": validation_data,
  "test": test_data
}

normalized_datasets = normalize_datasets(
  datasets,
  input_normalizer,
  target_normalizer
)

normalized_training_data = normalized_datasets["training"]
normalized_validation_data = normalized_datasets["validation"]
normalized_test_data = normalized_datasets["test"]

# MODEL

model = Model(learning_rate=0.0005)

trainer = Trainer(model)

# TRAINING

print("=== TRAINING ===")

result = trainer.fit(
  normalized_training_data,
  normalized_validation_data,
  epochs=5000,
  batch_size=3,
  patience=10
)

history = result["history"]

print("=== HISTORY ===")
print("Jumlah Epoch :", len(history["epoch"]))
print()

print("=== RESULT ===")
print(f"Best Epoch          : {result['best_epoch']}")
print(f"Best Validation MSE : {result['best_validation_mse']}")
print(f"Stopped Epoch       : {result['last_epoch']}")
print()

# EVALUATION

print("=== Evaluation ===")

testing_metrics = trainer.evaluate(normalized_test_data)

print(f"Testing MSE  : {testing_metrics['mse']}")
print(f"Testing MAE  : {testing_metrics['mae']}")
print(f"Testing RMSE : {testing_metrics['rmse']}")

# TAMPILKAN GRAFIK

plot_history(**history)  # Gunakan Syntax Unpacking Dictionary