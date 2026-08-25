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

    # Shuffle data untuk mengurangi ketergantungan Model pd pola urutan
    random.shuffle(shuffle_dataset)

    batches = []

    # Proses Mini-Batch
    for i in range(0, len(shuffle_dataset), batch_size):
      # Slicing Index
      batch = shuffle_dataset[i : (i + batch_size)]

      batches.append(batch)

    return batches

  # TRAIN — Melatih Model
  def train(self, dataset, batch_size):
    total_loss = 0

    # Buat Batch
    batches = self.create_batches(dataset, batch_size)

    for batch in batches:
      for inputs, targets in batch:
        # Forward
        predictions = self.model.forward(inputs)
    
        # Loss
        loss = mse(targets, predictions)
    
        total_loss += loss
    
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

    # Rata-rata Loss
    average_loss = total_loss / len(dataset)

    return average_loss

  # EVALUATE — Evaluasi / Validasi Model
  def evaluate(self, dataset):
    total_mse = total_mae = total_rmse = 0
  
    for inputs, targets in dataset:
      # Forward
      predictions = self.model.forward(inputs)

      # Metrics
      metrics = calculate_metrics(targets, predictions)

      total_mse += metrics["mse"]
      total_mae += metrics["mae"]
      total_rmse += metrics["rmse"]

    # Average Metrics
    average_mse = total_mse / len(dataset)
    average_mae = total_mae / len(dataset)
    average_rmse = total_rmse / len(dataset)
    
    return {
      "mse": average_mse,
      "mae": average_mae,
      "rmse": average_rmse,
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
    best_validation_mse = float("inf")   # Float positif tak terhingga
    best_model = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
      # Training
      training_loss = self.train(training_data, batch_size)

      # Validation
      validation_metrics = self.evaluate(validation_data)

      # Ambil hasil validation yg MSE saja
      validation_mse = validation_metrics["mse"]

      # Jika Model Membaik
      if (validation_mse < best_validation_mse):
        # Simpan Validation MSE
        best_validation_mse = validation_mse

        # Simpan Data Model saat ini sbg yg terbaik
        best_model = self.save_model()

        # Simpan posisi Epoch saat ini sbg yg terbaik
        best_epoch = epoch

        # Reset patience
        patience_counter = 0
      else:
        # Update patience
        patience_counter += 1

      # Log Epochs
      if epoch % 100 == 0:
        print(f"Epoch {epoch}")
        print(f"Training Loss   : {training_loss}")
        print(f"Validation MSE  : {validation_metrics['mse']}")
        print(f"Validation MAE  : {validation_metrics['mae']}")
        print(f"Validation RMSE : {validation_metrics['rmse']}")
        print(f"Patience        : {patience_counter}/{patience}")
        print()

      # Early Stopping
      if (patience_counter >= patience):
        # Simpan posisi Epoch paling terakhir
        last_epoch = epoch

        break

    # Restore Best Model
    if (best_model is not None):
      self.restore_model(best_model)
      
    return {
      "best_epoch": best_epoch,
      "best_validation_mse": best_validation_mse,
      "last_epoch": last_epoch,
    }