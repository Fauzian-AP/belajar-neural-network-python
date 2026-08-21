# Bagian untuk melatih Model
from loss import mse, mse_gradient   # Import Loss

class Trainer:
  def __init__(self, model):
    self.model = model

  # Batch adalah sekumpulan data yg diproses bersama dlm 1x proses NN.
  def create_batches(self, dataset, batch_size):
    batches = []

    for i in range(0, len(dataset), batch_size):
      # Slicing Index
      batch = dataset[i : (i + batch_size)]

      batches.append(batch)

    return batches

  def train(self, dataset):
    total_loss = 0

    for inputs, targets in dataset:
      # Forward
      predictions = self.model.forward(inputs)

      # Loss
      loss = mse(targets, predictions)

      total_loss += loss

      # Gradient Loss
      gradient_output = mse_gradient(targets, predictions)

      # Backward
      self.model.backward(inputs, gradient_output)

      # Update Weight & Bias
      self.model.step()

    # Rata-rata Loss
    average_loss = total_loss / len(dataset)

    return average_loss

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

  def fit(self, training_data, validation_data, epochs):
    for epoch in range(epochs):
      # Training
      training_loss = self.train(training_data)

      # Validation
      validation_loss = self.evaluate(validation_data)

      if (epoch - 1) % 100 == 0:
        print(f"Epoch {epoch - 1}")
        print(f"Training Loss: {training_loss}")
        print(f"Validation Loss: {validation_loss}")
        print()