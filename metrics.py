# Bagian Metrics, yaitu ukuran yg gunakan untuk mengetahui seberapa bagus performa Model

from typing import TypedDict
from loss import mse, mae, rmse


# ====================
# === TYPE CHECKER ===
# ====================

class Metrics(TypedDict):
  mse: float
  mae: float
  rmse: float


# ========================
# === FUNCTION METRICS ===
# ========================

def calculate_metrics(targets: list[float], predictions: list[float]) -> Metrics:
  return {
    "mse": mse(targets, predictions),
    "mae": mae(targets, predictions),
    "rmse": rmse(targets, predictions),
  }