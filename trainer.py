# Bagian untuk melakukan pelatihan pd Model

import copy
from pydantic import validate_call
from typing import TypedDict

from batch import Batch
from layer import Layer
from model import Model
from loss import MSE
from metrics import calculate_metrics
from custom_types import (
  PositiveInt,
  Dataset,
  Metrics,
)

# TYPE CHECKER

class MetricsHistory(TypedDict):
  MSE: list[float]
  MAE: list[float]
  RMSE: list[float]
  
class History(TypedDict):
  epochs: list[int]
  training_metrics: MetricsHistory
  validating_metrics: MetricsHistory

class FitTestResult(TypedDict):
  history: History
  best_epoch: int
  last_epoch: int
  best_validating_mse: float

# CLASS TRAINER

class Trainer:
  # CONSTRUCTOR — Initialization
  def __init__(self, model: Model, batch: Batch) -> None:
    # Simpan Model
    self.model = model

    # Simpan Batch
    self.batch = batch

  # TRAINING — Melatih Model Selama 1 Epoch
  @validate_call
  def training(self, dataset: Dataset, batch_size: PositiveInt) -> Metrics:
    # Buat Mini-Batch
    batches = self.batch.create_batches(dataset, batch_size)

    # Proses setiap Mini-Batch
    for batch in batches:
      # Proses tiap data dlm Mini-Batch
      for inputs, targets in batch:
        # Forward
        predictions = self.model.forward(inputs)

        # Gradient Loss
        gradient_outputs = MSE.gradient(targets, predictions)
    
        # Backward
        self.model.backward(gradient_outputs)

      # Average Gradient
      self.model.average_gradient(len(batch))

      # Update Weight & Bias
      self.model.step()

      # Reset Gradient
      self.model.reset_gradient()

    # Kembalikan hasil Evaluasi Model setelah 1 Epoch
    return self.evaluating(dataset)

  # EVALUATING — Mengukur Performa Model pd Dataset
  @validate_call
  def evaluating(self, dataset: Dataset) -> Metrics:
    total_mse = total_mae = total_rmse = 0.0

    # Total data dari Dataset
    dataset_size = len(dataset)
    
    # Proses tiap Data
    for inputs, targets in dataset:
      # Forward
      predictions = self.model.forward(inputs)

      # Hitung Metrics
      metrics = calculate_metrics(targets, predictions)

      # Total masing² Metrics
      total_mse += metrics["MSE"]
      total_mae += metrics["MAE"]
      total_rmse += metrics["RMSE"]

    # Rata² Metrics seluruh Dataset
    return {
      "MSE": total_mse / dataset_size,
      "MAE": total_mae / dataset_size,
      "RMSE": total_rmse / dataset_size,
    }

  # SAVE MODEL —  Menyimpan State Model
  def save_model(self) -> list[Layer]:
    # Buat Salinan Mendalam dari Layer
    return copy.deepcopy(self.model.layers)

  # RESTORE MODEL — Mengembalikan State Model
  def restore_model(self, best_state: list[Layer]) -> None:
    # Simpan lagi Salinan Layer
    self.model.layers = copy.deepcopy(best_state)

  # FIT — Melakukan Seluruh Proses Model secara keseluruhan
  @validate_call
  def fit(
    self,
    training_data: Dataset,
    validating_data: Dataset,
    epochs: PositiveInt,
    batch_size: PositiveInt,
    patience: PositiveInt = 10,
  ) -> FitTestResult:
    # Histroy Hasil
    history: History = {
      # List nomor Epoch
      "epochs": [],

      # Menyimpan Metrics dari Training
      "training_metrics": {
        "MSE": [],
        "MAE": [],
        "RMSE": []
      },

      # Menyimpan Metrics dari Validating
      "validating_metrics": {
        "MSE": [],
        "MAE": [],
        "RMSE": []
      },
    }

    # Data MSE terbaik 
    best_validating_mse = float("inf")   # Float positif tak terhingga

    # State Model terbaik
    best_model: list[Layer] | None = None

    # Posisi Epoch terbaik
    best_epoch: int = 0

    # Counter untuk menghitung toleransi jika Epoch tdk membaik
    patience_counter: int = 0

    # Posisi terakhir Epoch dijalankan
    last_epoch: int = 0

    # Proses Training berdasarkan Epoch
    for epoch in range(1, epochs + 1):
      # Training
      training_loss = self.training(training_data, batch_size)

      # Validating
      validating_loss = self.evaluating(validating_data)

      # Simpan Nomor Epoch
      history["epochs"].append(epoch)

      # Simpan Metrics Training
      history["training_metrics"]["MSE"].append(training_loss["MSE"])
      history["training_metrics"]["MAE"].append(training_loss["MAE"])
      history["training_metrics"]["RMSE"].append(training_loss["RMSE"])

      # Simpan metrics Validating
      history["validating_metrics"]["MSE"].append(validating_loss["MSE"])
      history["validating_metrics"]["MAE"].append(validating_loss["MAE"])
      history["validating_metrics"]["RMSE"].append(validating_loss["RMSE"])

      # Ambil Validating MSE
      validating_mse = validating_loss["MSE"]

      # Jika Model Membaik
      if (validating_mse < best_validating_mse):
        # Simpan Validating MSE yg terbaik
        best_validating_mse = validating_mse

        # Simpan Model saat ini Ke State
        best_model = self.save_model()

        # Simpan Epoch terbaik
        best_epoch = epoch

        # Reset Counter
        patience_counter = 0
      else:
        # Update Counter
        patience_counter += 1

      # Log Epoch per-100 Epoch
      if epoch % 100 == 0:
        print(f"=== Epoch {epoch} ===")
        print(f"Training MSE    : {training_loss['MSE']:.6f}")
        print(f"Training MAE    : {training_loss['MAE']:.6f}")
        print(f"Training RMSE   : {training_loss['RMSE']:.6f}")
        print(f"Validating MSE  : {validating_loss['MSE']:.6f}")
        print(f"Validating MAE  : {validating_loss['MAE']:.6f}")
        print(f"Validating RMSE : {validating_loss['RMSE']:.6f}")
        print(f"Patience        : {patience_counter}/{patience}")
        print()

      # Simpan posisi Epoch paling terakhir
      last_epoch = epoch

      # Early Stopping
      if (patience_counter >= patience):
        print("=== EARLY STOPPING ===")
        print(f"Epoch               : {last_epoch}")
        print(f"Best Epoch          : {best_epoch}")
        print(f"Best Validating MSE : {best_validating_mse:.6f}")
        print()

        # Hentikan Epoch
        break

    # Restore State Best Model
    if (best_model is not None):
      self.restore_model(best_model)

    return {
      "history": history,
      "best_epoch": best_epoch,
      "last_epoch": last_epoch,
      "best_validating_mse": best_validating_mse,
    }