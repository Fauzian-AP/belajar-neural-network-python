# Bagian Metrics, yaitu ukuran yg gunakan untuk mengetahui seberapa bagus performa Model

from pydantic import validate_call

from custom_types import SequenceFloat, Metrics
from loss import MSE, MAE, RMSE

# PERHITUNGAN METRICS
@validate_call
def calculate_metrics(targets: SequenceFloat, predictions: SequenceFloat) -> Metrics:
  return {
    "MSE": MSE(targets, predictions),
    "MAE": MAE(targets, predictions),
    "RMSE": RMSE(targets, predictions),
  }