# Bagian paling dasar dari Neural Network yaitu sebuah Neuron / Saraf yg berisi
# Rumus: y = x₁w₁ + x₂w₂ + x₃w₃ + bias

import random
import math
from activation import relu, relu_derivative

class Neuron:
  # CONSTRUCTOR — Initialization
  def __init__(
    self,
    input_size,
    learning_rate,
    is_activation=True
  ):
    # Menentukan seberapa besar perubahan Weight & Bias tiap update
    self.learning_rate = learning_rate

    # Cek apakah Activation digunakan atau tdk
    self.is_activation = is_activation

    # Menentukan seberapa besar pengaruh tiap Input terhadap Output
    self.weights = [
      # He — Menentukan skala Weight awal berdasarkan jumlah inputan
      random.gauss(0, math.sqrt(2 / input_size))

      # Tiap input punya Weight sendiri
      for _ in range(input_size) 
    ]
    
    # Menentukan pergeseran nilai pd Pre-Activation
    self.bias = 0

    # Menunjukkan seberapa sensitif Loss terhadap Weight
    self.gradient_weights = [0] * input_size   # Tiap input ada

    # Menunjukkan seberapa sensitif Loss terhadap Bias
    self.gradient_bias = 0

    # Nilai terakhir sebelum Activation
    self.last_pre_activation = 0

  # FORWARD — Proses Prediksi
  def forward(self, inputs):
    pre_activation = self.bias

    # Weighted Sum
    for input, weight in zip(inputs, self.weights):
      pre_activation += input * weight

    self.last_pre_activation = pre_activation

    # Aktivasi (ReLU)
    return (
      relu(pre_activation) if (self.is_activation) else pre_activation
    )
      
  # BACKWARD — Proses Cek Kesalahan
  def backward(self, inputs, gradient_output):
    # Gradient Pre-Activation (ReLU)
    if self.is_activation:
      gradient_pre_activation = gradient_output * relu_derivative(self.last_pre_activation)
    else:
      gradient_pre_activation = gradient_output

    # Akumulasi Gradient Weight
    for i, input in enumerate(inputs):
      self.gradient_weights[i] += gradient_pre_activation * input

    # Akumulasi Gradient Bias
    self.gradient_bias += gradient_pre_activation

    # Gradient Input (dikirim ke Layer sebelumnya)
    gradient_input = []

    for weight in self.weights:
      gradient_input.append(gradient_pre_activation * weight)

    return gradient_input

  # RESET GRADIENT — Mengosongkan Gradient Weight & Bias
  def reset_gradient(self):
    # Gradient Weight
    for i in range(len(self.gradient_weights)):
      self.gradient_weights[i] = 0

    # Gradient Bias
    self.gradient_bias = 0

  # AVERAGE GRADIENT — Menghitung rata² Gradient Weight & Bias
  def average_gradient(self, batch_size):
    # Gradient Weight
    for i in range(len(self.gradient_weights)):
      self.gradient_weights[i] /= batch_size

    # Gradient Bias
    self.gradient_bias /= batch_size

  # STEP — Update Weight & Bias
  def step(self):
    # Weight
    for i in range(len(self.weights)):
      self.weights[i] -= self.learning_rate * self.gradient_weights[i]

    # Bias
    self.bias -= self.learning_rate * self.gradient_bias