# Bagian Pengelolaan Loss, yaitu:
# Angka yg digunakan untuk mengukur seberapa salah Prediksi Model

import math

# MSE — Mean Squared Error
# Menghitung seberapa besar Error

# MSE = (1/Total) × Σ(Target - Prediksi)²

def mse(targets, predictions):
  total = 0

  for target, prediction in zip(targets, predictions):
    total += (target - prediction) ** 2

  return total / len(targets)

# MSE Gradient
# Menetukan Update Weight & Bias

# Gradient = (2/Total) × (Prediksi - Target)

def mse_gradient(targets, predictions):
  gradients = []

  for target, prediction in zip(targets, predictions):
    gradients.append(2 * (prediction - target))

  return gradients

# MAE — Mean Absolute Error
# Menghitung rata² jarak absolut error

# MAE = (1/Total) × Σ|Target - Prediksi|

def mae(targets, predictions):
  total_error = 0

  for target, prediction in zip(targets, predictions):
    total_error += abs(target - prediction)

  return total_error / len(targets)

# RMSE (Root Mean Squared Error)
# Hasil MSE di-akar sehingga satuannya kembali sama dgn Target

# RMSE = √MSE

def rmse(targets, predictions):
  return math.sqrt(mse(targets, predictions))