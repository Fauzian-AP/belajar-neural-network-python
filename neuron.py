# Bagian paling dasar yaitu sebuah Neuron / Saraf Tiruan yg berisi
# Rumus: y = x₁w₁ + x₂w₂ + x₃w₃ + bias

import random
from activation import relu, relu_derivative

class Neuron:
  # CONSTRUCTOR
  def __init__(
    self,
    input_size,
    learning_rate,
    activation=True
  ):
    # Learning Rate (Penting)
    self.learning_rate = learning_rate

    # Cek Activation
    self.activation = activation

    # Tiap 'input_size' memiliki Weight sendiri
    self.weights = [
      random.uniform(-1, 1)
      for _ in range(input_size)
    ]

    # Bias
    self.bias = random.uniform(-1, 1)

    # Tiap 'input_size' memiliki Grandient Weight sendiri
    self.gradient_weights = [
      0
      for _ in range(input_size)
    ]

    # Gradient Bias
    self.gradient_bias = 0

    # Menyimpan Pre-Activation
    self.last_pre_activation = 0

  # FORWARD — Proses Prediksi Neuron
  def forward(self, inputs):
    pre_activation = self.bias

    # Proses Weighted Sum
    for input, weight in zip(inputs, self.weights):
      pre_activation += input * weight

    self.last_pre_activation = pre_activation

    # Proses Aktivasi (ReLU)
    if self.activation:
      return relu(pre_activation)

    return pre_activation

  # BACKWARD — Proses Cek Kesalahan
  def backward(self, inputs, gradient_output):
    # Gradient Pre-Activation (ReLU)
    if self.activation:
      gradient_pre_activation = (
        gradient_output * relu_derivative(self.last_pre_activation)
      )
  
    else:
      gradient_pre_activation = gradient_output

    # Sum Gradient Weight
    for i, input in enumerate(inputs):
      self.gradient_weights[i] += gradient_pre_activation * input

    # Sum Gradient Bias
    self.gradient_bias += gradient_pre_activation

    # Gradient Input (dikirim ke Layer sebelumnya)
    gradient_input = []

    for weight in self.weights:
      gradient_input.append(gradient_pre_activation * weight)

    return gradient_input

  # RESET GRADIENT — Proses yg dijalankan sebelum proses 1x Batch
  def reset_gradient(self):
    # Reset tiap Gradient Weight
    for i in range(len(self.gradient_weights)):
      self.gradient_weights[i] = 0

    # Reset Gradient Bias
    self.gradient_bias = 0

  # STEP — Update Weight & Bias
  def step(self):
    # Sesuaikan Weight
    for i in range(len(self.weights)):
      self.weights[i] -= (
        self.learning_rate * self.gradient_weights[i]
      )

    # Sesuaikan Bias
    self.bias -= (
      self.learning_rate * self.gradient_bias
    )