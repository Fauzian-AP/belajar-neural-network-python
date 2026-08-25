# Bagian Metrics, yaitu:
# Ukuran yg gunakan untuk mengetahui seberapa bagus performa Model

from loss import (
  mse,
  mae,
  rmse
)

def calculate_metrics(targets, predictions):
  return {
    "mse": mse(targets, predictions),
    "mae": mae(targets, predictions),
    "rmse": rmse(targets, predictions)
  }