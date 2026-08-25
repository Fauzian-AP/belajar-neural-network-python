# Bagian untuk melatih Model

import random
import copy

from metrics import calculate_metrics
from loss import mse, mse_gradient

class Trainer:
  # CONSTRUCTOR — Initialization
  def __init__(self, model):
    self.model = model

  # CREATE BATCH — Membuat kumpulan data yg diproses secara bersamaan dlm 1x proses NN
  def create_batches(self, dataset, batch_size):
    # Buat Duplikat dari Dataset
    shuffle_dataset = dataset.copy()

    # Shuffle data agar mengurangi ketergantungan Model pd pola urutan
    random.shuffle(shuffle_dataset)

    batches = []

    # Proses pembagian data menjadi Batch² kecil
    for i in range(0, len(shuffle_dataset), batch_size):
      # Slicing Index
      batch = shuffle_dataset[i:(i + batch_size)]

      batches.append(batch)

    return batches

  # TRAIN — Melatih Model
  def train(self, dataset, batch_size):
    # Buat Batch
    batches = self.create_batches(dataset, batch_size)

    # Proses Mini-Batch
    for batch in batches:
      for inputs, targets in batch:
        # Forward
        predictions = self.model.forward(inputs)

        # Gradient Loss
        gradient_output = mse_gradient(targets, predictions)
    
        # Backward
        self.model.backward(inputs, gradient_output)

      # Average Gradient
      self.model.average_gradient(len(batch))

      # Update Weight & Bias
      self.model.step()

      # Reset Gradient
      self.model.reset_gradient()

    # Hitung Metrics setelah Batch selesai
    training_metrics = self.evaluate(dataset)
    
    return training_metrics

  # EVALUATE — Validasi Model
  def evaluate(self, dataset):
    total_mse = total_mae = total_rmse = 0
  
    for inputs, targets in dataset:
      # Forward
      predictions = self.model.forward(inputs)

      # Metrics
      metrics = calculate_metrics(targets, predictions)

      # Kumpulkan Metrics
      total_mse += metrics["mse"]
      total_mae += metrics["mae"]
      total_rmse += metrics["rmse"]

    # Kembalikan Metrics yg telah di Average
    return {
      "mse": total_mse / len(dataset),   # Loss Core
      "mae": total_mae / len(dataset),
      "rmse": total_rmse / len(dataset),
    }

  # SAVE MODEL — State untuk menyimpan Weight & Bias Model
  def save_model(self):
    return copy.deepcopy(self.model.layers)

  # RESTORE MODEL — State untuk engembalikan Weight & Bias Model
  def restore_model(self, best_state):
    self.model.layers = copy.deepcopy(best_state)
  
  # FIT — Melakukan Test Model secara keseluruhan
  def fit(
    self,
    training_data,
    validation_data,
    epochs,
    batch_size,
    patience=10
  ):
    # Histroy Training
    history = {
      "epoch": [],
      "training_data": [],
      "validation_data": [],
    }

    # Early Stopping & Restore Best Model Initialization
    best_validation_mse = float("inf")   # Float positif tak terhingga
    best_model = None
    best_epoch = 0
    patience_counter = 0
    last_epoch = epochs

    for epoch in range(epochs):
      # Training
      training_loss = self.train(training_data, batch_size)

      # Validation
      validation_loss = self.evaluate(validation_data)

      # Simpan Histroy
      history["epoch"].append(epoch)
      history["training_data"].append(training_loss)
      history["validation_data"].append(validation_loss)

      # Ambil Validation MSE
      validation_mse = validation_loss["mse"]

      # Jika Model Membaik
      if (validation_mse < best_validation_mse):
        # Simpan Validation MSE yg terbaik
        best_validation_mse = validation_mse

        # Simpan Data Model saat ini sbg yg terbaik
        best_model = self.save_model()

        # Simpan Epoch saat ini sbg yg terbaik
        best_epoch = epoch

        # Reset patience
        patience_counter = 0
      else:
        # Update patience
        patience_counter += 1

      # Log Epochs
      if epoch % 100 == 0:
        print(f"=== Epoch {epoch} ===")
        print(f"Training MSE    : {training_loss["mse"]}")
        print(f"Training MAE    : {training_loss["mae"]}")
        print(f"Training RMSE   : {training_loss["rmse"]}")
        print(f"Validation MSE  : {validation_loss['mse']}")
        print(f"Validation MAE  : {validation_loss['mae']}")
        print(f"Validation RMSE : {validation_loss['rmse']}")
        print(f"Patience        : {patience_counter}/{patience}")
        print()

      # Early Stopping
      if (patience_counter >= patience):
        # Simpan posisi Epoch paling terakhir
        last_epoch = epoch

        print("=== EARLY STOPPING ===")
        print(f"Epoch               : {last_epoch}")
        print(f"Best Epoch          : {best_epoch}")
        print(f"Best Validation MSE : {best_validation_mse}")
        print()

        # Hentikan Epoch
        break

    # Restore Best Model
    if (best_model is not None):
      self.restore_model(best_model)
      
    return {
      "history": history,
      "best_epoch": best_epoch,
      "best_validation_mse": best_validation_mse,
      "last_epoch": last_epoch,
    }