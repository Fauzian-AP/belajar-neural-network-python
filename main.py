from model import Model
from trainer import Trainer

from dataset import (
  training_data,
  validation_data,
  test_data
)

from normalizer import (
  Normalizer,
  normalize_list,
  denormalize_list
)

# Normalizer

input_normalizer = Normalizer()
target_normalizer = Normalizer()

# Ambil Nilai Training

input_values = []
target_values = []


for inputs, targets in training_data:
  input_values.extend(inputs)
  target_values.extend(targets)

# Fit Normalizer

input_normalizer.fit(input_values)
target_normalizer.fit(target_values)

# Cek Min & Max

print("=== Normalizer ===")
print("Input Min   :", input_normalizer.minimum)
print("Input Max   :", input_normalizer.maximum)
print("Target Min  :", target_normalizer.minimum)
print("Target Max  :", target_normalizer.maximum)
print()

# Normalisasi Dataset

def normalize_dataset(dataset, input_normalizer, target_normalizer):
  normalized_dataset = []

  for inputs, targets in dataset:
    normalized_inputs = normalize_list(inputs, input_normalizer)
    normalized_targets = normalize_list(targets, target_normalizer)

    normalized_dataset.append((normalized_inputs, normalized_targets))

  return normalized_dataset

# Normalisasi Dataset

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

normalized_test_data = normalize_dataset(
  test_data,
  input_normalizer,
  target_normalizer
)

# Model

model = Model(learning_rate=0.0005)

trainer = Trainer(model)

# Training

print("=== Training ===")

trainer.fit(
  normalized_training_data,
  normalized_validation_data,
  epochs=1000,
  batch_size=3
)

# Final Evaluation

print("=== Final Evaluation ===")

testing_loss = trainer.evaluate(
  normalized_test_data
)

print(f"Testing Loss : {testing_loss}")

# Prediction

inputs = [5, 10, 15]

normalized_inputs = normalize_list(
  inputs,
  input_normalizer
)

normalized_prediction = model.forward(
  normalized_inputs
)

prediction = denormalize_list(
  normalized_prediction,
  target_normalizer
)

# Hasil

print()
print("=== Test Data ===")
print("Input      :", inputs)
print("Prediction :", prediction)
print("Target     :", [50, 100])