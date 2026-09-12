# Main Pipeline Neural Network

import random
from typing_extensions import Final

from neural_network.core import Model
from neural_network.data import Preprocessor
from neural_network.training import Batch, Trainer
from neural_network.visualization import plot_training, plot_evaluating
from neural_network.utils.custom_types import ScaleType, EvaluatingType, ActivationType
from neural_network.data.dataset import (
  training_data,
  validation_data,
  test_data,
  generalization_data,
)

# CONFIGURATION

SEED: Final = 42

LEARNING_RATE: Final = 0.113

EPOCHS: Final = 360

BATCH_SIZE: Final = 5

PATIENCE: Final = 5

ARCHITECTURE: Final = (3, 4, 4, 2)

# REPRODUCIBILITY

random.seed(SEED)

# DATASET INFO

print("=== DATASET INFO ===")
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

model = Model(
  learning_rate=LEARNING_RATE,
  architecture=ARCHITECTURE,
  activation=ActivationType.LEAKY_RELU,
)

# BATCH

batch = Batch(seed=SEED)

# TRAINER

trainer = Trainer(model=model, batch=batch)

# TRAINING

print("=== FIT TRAINING ===")

fit_result = trainer.fit(
  training_data=normalized_datasets["training"],
  validating_data=normalized_datasets["validation"],
  epochs=EPOCHS,
  batch_size=BATCH_SIZE,
  patience=PATIENCE,
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