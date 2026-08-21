# Bagian untuk melatih Model
from loss import mse, mse_gradient   # Import Loss

class Trainer:
  # CONSTRUCTOR
  def __init__(self, model):
    self.model = model

  # CREATE BATCH — Membuat kumpulan data yg diproses secara bersamaan dlm 1x proses NN.
  def create_batches(self, dataset, batch_size):
    batches = []

    for i in range(0, len(dataset), batch_size):
      # Slicing Index
      batch = dataset[i : (i + batch_size)]

      batches.append(batch)

    return batches

  # TRAIN — Bagian melatih Model
  def train(self, dataset, batch_size):
    total_loss = 0

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

  # EVALUATE — Bagian Evaluasi / Validasi Model
  def evaluate(self, dataset):
    total_loss = 0
  
    for inputs, targets in dataset:
      # Forward
      predictions = self.model.forward(inputs)

      # Loss
      loss = mse(targets, predictions)

      total_loss += loss

    # Rata-rata Loss
    average_loss = total_loss / len(dataset)
    
    return average_loss

  # FIT — Melakukan Test Model secara keseluruhan
  def fit(
    self,
    training_data,
    validation_data,
    epochs,
    batch_size
  ):
    for epoch in range(epochs):
      # Training
      training_loss = self.train(training_data, batch_size)

      # Validation
      validation_loss = self.evaluate(validation_data)

      if epoch % 100 == 0:
        print(f"Epoch {epoch - 1}")
        print(f"Training Loss: {training_loss}")
        print(f"Validation Loss: {validation_loss}")
        print()