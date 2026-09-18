# Bagian untuk melakukan pelatihan pd Model

import copy

from .batch import Batch
from src.core import Layer, Model

from src.evaluation import (
  Loss,
  calculate_metrics,
)

from src.utils.custom_types import (
  IntPositive,
  Dataset,
  Metrics,
  FitHistory,
  FitResult,
)

class Trainer:
  # CONSTRUCTOR — Initialization
  def __init__(
    self,
    model: Model,
    batch: Batch,
    loss: Loss,
  ) -> None:
    # Simpan Model
    self.model: Model = model

    # Simpan Batch
    self.batch: Batch = batch

    # Simpan Loss Function
    self.loss: Loss = loss

  # TRAINING — Melatih Model Selama 1 Epoch
  def training(self, dataset: Dataset, batch_size: IntPositive) -> Metrics:
    # Validasi
    if not dataset:
      raise ValueError("Dataset yg digunakan tdk boleh kosong.")

    # Buat Mini-Batch
    batches = self.batch.create_batches(dataset, batch_size)

    # Proses setiap Mini-Batch
    for batch in batches:
      # Proses tiap data dlm Mini-Batch
      for inputs, targets in batch:
        # Forward
        predictions = self.model.forward(inputs)

        # Gradient Loss
        gradient_outputs = self.loss.gradient(targets, predictions)

        # Backward
        self.model.backward(gradient_outputs)

      # Average Gradient
      self.model.average_gradient(len(batch))

      # Update Weight & Bias
      self.model.step()

      # Reset Gradient
      self.model.reset_gradient()

    # Kembalikan hasil Evaluasi Model setelah 1 Epoch
    return self.evaluation(dataset)

  # EVALUATION — Mengukur Performa Model pd Dataset
  def evaluation(self, dataset: Dataset) -> Metrics:
    # Validasi
    if not dataset:
      raise ValueError("Dataset yg digunakan tdk boleh kosong.")

    # Total Metrics Initialization
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
  def fit(
    self,
    training_data: Dataset,
    validation_data: Dataset,
    epochs: IntPositive,
    batch_size: IntPositive,
    patience: IntPositive = 10,
  ) -> FitResult:
    # Histroy Hasil
    history: FitHistory = {
      # List nomor Epoch
      "epochs": [],

      # Menyimpan Metrics dari Training
      "training_metrics": {
        "MSE": [],
        "MAE": [],
        "RMSE": []
      },

      # Menyimpan Metrics dari Validation
      "validation_metrics": {
        "MSE": [],
        "MAE": [],
        "RMSE": []
      },
    }

    # Data MSE terbaik 
    best_validation_mse = float("inf")   # Float positif tak terhingga

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

      # Validation
      validation_loss = self.evaluation(validation_data)

      # Simpan Nomor Epoch
      history["epochs"].append(epoch)

      # Simpan Metrics Training
      history["training_metrics"]["MSE"].append(training_loss["MSE"])
      history["training_metrics"]["MAE"].append(training_loss["MAE"])
      history["training_metrics"]["RMSE"].append(training_loss["RMSE"])

      # Simpan metrics Validation
      history["validation_metrics"]["MSE"].append(validation_loss["MSE"])
      history["validation_metrics"]["MAE"].append(validation_loss["MAE"])
      history["validation_metrics"]["RMSE"].append(validation_loss["RMSE"])

      # Ambil Validation MSE
      validation_mse = validation_loss["MSE"]

      # Jika Model Membaik
      if (validation_mse < best_validation_mse):
        # Simpan Validation MSE yg terbaik
        best_validation_mse = validation_mse

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
      if epoch % 5 == 0:
        print(f"=== Epoch {epoch} ===")
        print(f"Training MSE    : {training_loss['MSE']:.18f}")
        print(f"Training MAE    : {training_loss['MAE']:.18f}")
        print(f"Training RMSE   : {training_loss['RMSE']:.18f}")
        print(f"Validation MSE  : {validation_loss['MSE']:.18f}")
        print(f"Validation MAE  : {validation_loss['MAE']:.18f}")
        print(f"Validation RMSE : {validation_loss['RMSE']:.18f}")
        print(f"Patience        : {patience_counter}/{patience}")
        print()

      # Simpan posisi Epoch paling terakhir
      last_epoch = epoch

      # Early Stopping
      if (patience_counter >= patience):
        print("=== EARLY STOPPING ===")
        print(f"Epoch               : {last_epoch}")
        print(f"Best Epoch          : {best_epoch}")
        print(f"Best Validation MSE : {best_validation_mse:.18f}")
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
      "best_validation_mse": best_validation_mse,
    }