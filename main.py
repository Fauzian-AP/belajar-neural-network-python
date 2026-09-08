# Main Pipeline Neural Network

import copy
import math
import random

from preprocessing import Preprocessor
from batch import Batch
from model import Model
from trainer import Trainer
from metrics import calculate_metrics
from loss import MSE
from plot import plot_training, plot_evaluating
from custom_types import ScaleType, EvaluatingType
from dataset import (
  training_data,
  validation_data,
  test_data,
  generalization_data,
)

# REPRODUCIBILITY

SEED = 42

random.seed(SEED)

# DATASET

print("=== DATASET ===")
print(f"Training       : {len(training_data)}")
print(f"Validation     : {len(validation_data)}")
print(f"Testing        : {len(test_data)}")
print(f"Generalization : {len(generalization_data)}")
print()

# PREPROCESSOR

print("=== PREPROCESSING ===")

preprocessor = Preprocessor()

# Kumpulkan Dataset yg digunakan untuk Normalisasi
datasets = {
  "training": training_data,
  "validation": validation_data,
  "test": test_data,
  "generalization": generalization_data,
}

# PREPROCESSING

normalized_datasets = preprocessor.preprocess(datasets)

# NORMALIZER INFORMATION

print(f"Input Min   : {preprocessor.input_scaler.minimum}")
print(f"Input Max   : {preprocessor.input_scaler.maximum}")
print(f"Target Min  : {preprocessor.target_scaler.minimum}")
print(f"Target Max  : {preprocessor.target_scaler.maximum}")
print()

# MODEL

learning_rate = 0.113

model = Model(learning_rate)

# LOG WEIGHT & BIAS

print("=== INITIAL MODEL ===")

for layer_index, layer in enumerate(model.layers, start=1):
  print(f"Layer {layer_index}")

  for neuron_index, neuron in enumerate(layer.neurons, start=1):
    print(f"Neuron {neuron_index} | Weights: {neuron.weights} | Bias: {neuron.bias}")

  print()

# BATCH

batch = Batch(seed=SEED)


print("=== DIAGNOSTIC 1 BATCH ===")

# Simpan seluruh State Model
initial_model = copy.deepcopy(model.layers)

# Buat Batch khusus Diagnostic
diagnostic_batch = Batch(seed=SEED).create_batches(normalized_datasets["training"], 5)[0]

# Reset Gradient
model.reset_gradient()

# Forward + Backward
for inputs, targets in diagnostic_batch:
  predictions = model.forward(inputs)

  gradient_outputs = MSE.gradient(targets, predictions)

  model.backward(gradient_outputs)

# Ambil Gradient Neuron 1 Layer 1
gradient_weights = model.layers[0].neurons[0].gradient_weights.copy()
gradient_bias = model.layers[0].neurons[0].gradient_bias

print(f"Initial Weights : {initial_model[0].neurons[0].weights}")
print(f"Gradient Weights: {gradient_weights}")
print(f"Gradient Bias   : {gradient_bias}")

# Average Gradient
model.average_gradient(len(diagnostic_batch))

# Update Parameter
model.step()

print(f"Updated Weights : {model.layers[0].neurons[0].weights}")
print()

# Kembalikan SELURUH Model
model.layers = copy.deepcopy(initial_model)

# Pastikan Gradient bersih
model.reset_gradient()


print("=== DIAGNOSTIC 2 — GRADIENT PER LAYER ===")

initial_model = copy.deepcopy(model.layers)

diagnostic_batch = Batch(seed=SEED).create_batches(normalized_datasets["training"], 5)[0]

model.reset_gradient()

# Backward
for inputs, targets in diagnostic_batch:
  predictions = model.forward(inputs)

  gradient_outputs = MSE.gradient(targets, predictions)

  model.backward(gradient_outputs)

# Average
model.average_gradient(len(diagnostic_batch))

# Analisis Tiap Layer
for layer_index, layer in enumerate(model.layers, start=1):
  layer_max_gradient = layer_max_update = 0.0

  for neuron_index, neuron in enumerate(layer.neurons, start=1):
    for weight_index, gradient in enumerate(neuron.gradient_weights, start=1):
      update = -learning_rate * gradient

      layer_max_gradient = max(layer_max_gradient, abs(gradient))
      layer_max_update = max(layer_max_update, abs(update))

    update_bias = -learning_rate * neuron.gradient_bias

    layer_max_gradient = max(layer_max_gradient, abs(neuron.gradient_bias))
    layer_max_update = max(layer_max_update, abs(update_bias))

  print(f"Layer {layer_index}")
  print(f"  Maximum Gradient : {layer_max_gradient:.6f}")
  print(f"  Maximum Update   : {layer_max_update:.6f}")

print()

# Restore
model.layers = copy.deepcopy(initial_model)

model.reset_gradient()


print("=== DIAGNOSTIC 3 — EFFECT OF FIRST UPDATE ===")

# Simpan State Model
initial_model = copy.deepcopy(model.layers)

# Buat Batch Diagnostic
diagnostic_batch = Batch(seed=SEED).create_batches(normalized_datasets["training"], 5)[0]

# Fungsi Hitung MSE
def diagnostic_mse(dataset):
  total_mse = 0.0

  for inputs, targets in dataset:
    predictions = model.forward(inputs)

    metrics = calculate_metrics(targets, predictions)

    total_mse += metrics["MSE"]

  return total_mse / len(dataset)

# Sebelum Update
before_training_mse = diagnostic_mse(diagnostic_batch)
before_validation_mse = diagnostic_mse(normalized_datasets["validation"])

print(f"Training MSE BEFORE   : {before_training_mse}")
print(f"Validation MSE BEFORE : {before_validation_mse}")

model.reset_gradient()

# Proses Forward & Backward
for inputs, targets in diagnostic_batch:
  predictions = model.forward(inputs)

  gradient_outputs = MSE.gradient(targets, predictions)

  model.backward(gradient_outputs)

# Average Gradient
model.average_gradient(len(diagnostic_batch))

# Update
model.step()

# Setelah Update
after_training_mse = diagnostic_mse(diagnostic_batch)
after_validation_mse = diagnostic_mse(normalized_datasets["validation"])

print(f"Training MSE AFTER    : {after_training_mse}")
print(f"Validation MSE AFTER  : {after_validation_mse}")
print(f"Training MSE CHANGE   : {after_training_mse - before_training_mse}")
print(f"Validation MSE CHANGE : {after_validation_mse - before_validation_mse}")
print()

# Restore Model
model.layers = copy.deepcopy(initial_model)

model.reset_gradient()


print("=== DIAGNOSTIC 4 — WEIGHT vs GRADIENT ===")

initial_model = copy.deepcopy(model.layers)

diagnostic_batch = Batch(seed=SEED).create_batches(normalized_datasets["training"], 5)[0]

model.reset_gradient()

# Backward
for inputs, targets in diagnostic_batch:
  predictions = model.forward(inputs)

  gradient_outputs = MSE.gradient(targets, predictions)

  model.backward(gradient_outputs)

# Average Gradient
model.average_gradient(len(diagnostic_batch))

# Analisis Tiap Layer
for layer_index, layer in enumerate(model.layers, start=1):
  max_weight = max_gradient = 0.0

  for neuron in layer.neurons:
    for weight in neuron.weights:
      max_weight = max(max_weight, abs(weight))

    for gradient in neuron.gradient_weights:
      max_gradient = max(max_gradient, abs(gradient))

    max_gradient = max(max_gradient, abs(neuron.gradient_bias))

  print(f"Layer {layer_index}")
  print(f"  Maximum |Weight|   : {max_weight:.6f}")
  print(f"  Maximum |Gradient| : {max_gradient:.6f}")

  if max_weight != 0:
    print(f"  Gradient / Weight  : {max_gradient / max_weight:.6f}")

print()

# Restore Model
model.layers = copy.deepcopy(initial_model)

model.reset_gradient()


print("=== DIAGNOSTIC 5 — GRADIENT FLOW ===")

initial_model = copy.deepcopy(model.layers)

diagnostic_batch = Batch(seed=SEED).create_batches(normalized_datasets["training"], 5)[0]

model.reset_gradient()

# Proses setiap data
for inputs, targets in diagnostic_batch:
  # FORWARD
  predictions = model.forward(inputs)

  # Gradient dari Loss terhadap Output Model
  gradient = MSE.gradient(targets, predictions)

  # Gradient Output
  print(f"Gradient Output | Max: {max(abs(g) for g in gradient):.6f}")

  # Backward Manual
  for layer_index in range(len(model.layers) - 1, -1, -1):
    layer = model.layers[layer_index]

    gradient = layer.backward(gradient)

    print(f"Gradient after L{layer_index + 1} | Max: {max(abs(g) for g in gradient):.6f}")

  print()

# Restore
model.layers = copy.deepcopy(initial_model)

model.reset_gradient()


print("=== DIAGNOSTIC 6 — SEED SENSITIVITY ===")

# Menguji sensitivitas Model terhadap Initial Weight
# menggunakan beberapa Seed berbeda.

SEEDS = [1, 2, 3, 4, 5, 42]

for seed in SEEDS:
  # Initial Model
  random.seed(seed)

  diagnostic_model = Model(learning_rate=learning_rate)
  diagnostic_batcher = Batch(seed=seed)

  # MSE Sebelum

  diagnostic_trainer = Trainer(
    model=diagnostic_model,
    batch=diagnostic_batcher,
  )

  mse_before = diagnostic_trainer.evaluating(normalized_datasets["training"])["MSE"]

  # First Batch

  diagnostic_model.reset_gradient()

  batches = diagnostic_batcher.create_batches(
    normalized_datasets["training"],
    batch_size=5,
  )

  first_batch = batches[0]

  # Forward & Backward
  for inputs, targets in first_batch:
    predictions = diagnostic_model.forward(inputs)

    gradient_outputs = MSE.gradient(targets, predictions)

    diagnostic_model.backward(gradient_outputs)

  # Max Gradient

  max_gradient = 0.0

  for layer in diagnostic_model.layers:
    for neuron in layer.neurons:
      max_gradient = max(
        max_gradient,
        max(
          abs(gradient)
          for gradient in neuron.gradient_weights
        ),
        abs(neuron.gradient_bias),
      )

  # Average Gradient
  diagnostic_model.average_gradient(len(first_batch))

  # Max Update

  max_update = 0.0

  for layer in diagnostic_model.layers:
    for neuron in layer.neurons:
      max_update = max(
        max_update,
        max(
          abs(learning_rate * gradient)
          for gradient in neuron.gradient_weights
        ),
        abs(learning_rate * neuron.gradient_bias),
      )

  # Update
  diagnostic_model.step()
  diagnostic_model.reset_gradient()

  # MSE Setelah
  mse_after = diagnostic_trainer.evaluating(normalized_datasets["training"])["MSE"]

  mse_change = mse_after - mse_before

  # Hasil
  print(
    f"Seed {seed:<2} → "
    f"Max Gradient: {max_gradient:.6f} → "
    f"Max Update: {max_update:.6f} → "
    f"First-step MSE Change: {mse_change:.6f}"
  )

print()


print("=== DIAGNOSTIC 7 — INITIALIZATION SENSITIVITY ===")

# Seed hanya digunakan untuk membuat Initial Weight berbeda.
SEEDS = [1, 2, 3, 4, 5, 42]

# Buat satu Batch tetap.
# Batch ini akan digunakan oleh SEMUA Seed.
fixed_batch = Batch(seed=SEED).create_batches(normalized_datasets["training"], batch_size=5)[0]

for seed in SEEDS:
  random.seed(seed)

  # Initial Model
  diagnostic_model = Model(learning_rate=learning_rate)

  # MSE Sebelum

  diagnostic_trainer = Trainer(
    model=diagnostic_model,
    batch=Batch(seed=SEED),
  )

  mse_before = diagnostic_trainer.evaluating(normalized_datasets["training"])["MSE"]

  diagnostic_model.reset_gradient()

  # Forward & Backward
  for inputs, targets in fixed_batch:
    predictions = diagnostic_model.forward(inputs)

    gradient_outputs = MSE.gradient(targets, predictions)

    diagnostic_model.backward(gradient_outputs)

  # Max Gradient

  max_gradient = 0.0

  for layer in diagnostic_model.layers:
    for neuron in layer.neurons:
      max_gradient = max(
        max_gradient,
        max(
          abs(gradient)
          for gradient in neuron.gradient_weights
        ),
        abs(neuron.gradient_bias),
      )

  # Average Gradient
  diagnostic_model.average_gradient(len(fixed_batch))

  # Max Update

  max_update = 0.0

  for layer in diagnostic_model.layers:
    for neuron in layer.neurons:
      max_update = max(
        max_update,
        max(
          abs(learning_rate * gradient)
          for gradient in neuron.gradient_weights
        ),
        abs(learning_rate * neuron.gradient_bias),
      )

  # Update
  diagnostic_model.step()

  diagnostic_model.reset_gradient()

  # MSE Setelah
  mse_after = diagnostic_trainer.evaluating(normalized_datasets["training"])["MSE"]

  mse_change = mse_after - mse_before

  # Hasil
  print(
    f"Seed {seed:<2} → "
    f"Max Gradient: {max_gradient:.6f} → "
    f"Max Update: {max_update:.6f} → "
    f"First-step MSE Change: {mse_change:.6f}"
  )

print()


# =================================================================
# === DIAGNOSTIC 8 — GRADIENT FLOW PER SEED ======================
# =================================================================

print("=== DIAGNOSTIC 8 — GRADIENT FLOW PER SEED ===")

SEEDS = [1, 2, 3, 4, 5, 42]

# Gunakan Batch yang sama untuk semua Seed
fixed_batch = Batch(
  seed=SEED,
).create_batches(
  normalized_datasets["training"],
  batch_size=5,
)[0]

for seed in SEEDS:

  # ---------------------------------------------------------------
  # INITIALIZE MODEL
  # ---------------------------------------------------------------

  random.seed(seed)

  diagnostic_model = Model(
    learning_rate=learning_rate,
  )

  # ---------------------------------------------------------------
  # FORWARD
  # ---------------------------------------------------------------

  # Gunakan satu data yang sama untuk semua Seed.
  inputs, targets = fixed_batch[0]

  predictions = diagnostic_model.forward(
    inputs,
  )

  # ---------------------------------------------------------------
  # OUTPUT GRADIENT
  # ---------------------------------------------------------------

  gradient = MSE.gradient(
    targets,
    predictions,
  )

  print(
    f"Seed {seed:<2} | "
    f"Output: {max(abs(g) for g in gradient):.6f}"
  )

  # ---------------------------------------------------------------
  # BACKWARD FLOW
  # ---------------------------------------------------------------

  for layer_index in range(
    len(diagnostic_model.layers) - 1,
    -1,
    -1,
  ):

    layer = diagnostic_model.layers[layer_index]

    gradient = layer.backward(
      gradient,
    )

    print(
      f"          | "
      f"After L{layer_index + 1}: "
      f"{max(abs(g) for g in gradient):.6f}"
    )

  print()

print()


# TRAINER

trainer = Trainer(model=model, batch=batch)

# TRAINING

print("=== FIT TRAINING ===")

fit_result = trainer.fit(
  training_data=normalized_datasets["training"],
  validating_data=normalized_datasets["validation"],
  epochs=360,
  batch_size=5,
  patience=5,
)

# Ambil History Fit
history = fit_result["history"]

# Tetapkan index terbaik
best_index = fit_result["best_epoch"] - 1

# Ambil salah satu data Training
training_metrics = {
  metric_name: values[best_index]
  for metric_name, values in history["training_metrics"].items()
}

# Ambil salah satu data Validating
validating_metrics = {
  metric_name: values[best_index]
  for metric_name, values in history["validating_metrics"].items()
}

# FIT RESULT

print("=== FIT RESULT ===")
print(f"Best Epoch          : {fit_result['best_epoch']}")
print(f"Best Validating MSE : {fit_result['best_validating_mse']}")
print(f"Last Epoch          : {fit_result['last_epoch']}")
print()

# TESTING

print("=== TESTING ===")

testing_metrics = trainer.evaluating(normalized_datasets["test"])

print(f"Testing MSE  : {testing_metrics['MSE']}")
print(f"Testing MAE  : {testing_metrics['MAE']}")
print(f"Testing RMSE : {testing_metrics['RMSE']}")
print()

# GENERALIZATION

print("=== GENERALIZATION ===")

generalization_metrics = trainer.evaluating(normalized_datasets["generalization"])

print(f"Generalization MSE  : {generalization_metrics['MSE']}")
print(f"Generalization MAE  : {generalization_metrics['MAE']}")
print(f"Generalization RMSE : {generalization_metrics['RMSE']}")
print()

# KUMPULKAN SEMUA HASIL EVALUATING DATASET
evaluating_metrics: EvaluatingType = {
  "training": training_metrics,
  "validating": validating_metrics,
  "testing": testing_metrics,
  "generalization": generalization_metrics,
}

# GENERATE PLOT

plot_training(history, ScaleType.NORMALIZE)

plot_evaluating(evaluating_metrics, ScaleType.NORMALIZE)