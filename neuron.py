# Merupakan Bagian paling dasar dari Sistem Neural Network yaitu
# sebuah Sistem Neuron / Saraf yg berisi puluhan, ribuan, atau jutaan
# proses dibawah ini

import random
import math

from typing import Literal, Sequence, get_args
from activation import relu, relu_derivative


# ====================
# === TYPE CHECKER ===
# ====================

ActivationType = Literal["none", "ReLU"]


# =============
# === CLASS ===
# =============

class Neuron:
  # CONSTRUCTOR — Initialization
  def __init__(
    self,
    input_size: int,
    learning_rate: float,
    activation: ActivationType = "none",
  ):
    # Validasi Inputan
    if (input_size <= 0 or learning_rate <= 0):
      raise ValueError("input_size & learning_rate harus bernilai lebih besar dari 0.")

    # Validasi Pilihan Aktivasi
    if activation not in get_args(ActivationType):
      raise ValueError(f"Aktivasi '{activation}' tidak dikenali. Pilih salah satu dari: {get_args(ActivationType)}")

    # Menentukan seberapa besar perubahan Weight & Bias tiap update
    self.learning_rate: float = learning_rate

    # Menentukan jumlah Inputan data yg dpt diterima sebuah Neuron
    self.input_size: int = input_size

    # Jenis Activation yg digunakan
    self.activation: ActivationType = activation

    # Menentukan seberapa besar pengaruh tiap Input terhadap Output
    self.weights: list[float] = [
      # Inisialisasi He / Kaiming — Menentukan skala Weight awal berdasarkan jumlah inputan
      random.gauss(0.0, math.sqrt(2.0 / input_size))

      # Tiap input punya Weight sendiri
      for _ in range(input_size) 
    ]
    
    # Menentukan pergeseran nilai pd Pre-Activation
    self.bias: float = 0.0

    # Menunjukkan seberapa sensitif Loss terhadap Weight
    self.gradient_weights: list[float] = [0.0] * input_size   # Tiap input ada

    # Menunjukkan seberapa sensitif Loss terhadap Bias
    self.gradient_bias: float = 0.0

    # Nilai terakhir sebelum Activation
    self.last_pre_activation: float = 0.0

  # FORWARD — Proses Prediksi
  def forward(self, inputs: Sequence[float]) -> float:
    # Validasi Input
    if len(inputs) != self.input_size:
      raise ValueError(f"Ukuran input ({len(inputs)}) tidak sesuai dengan input_size Neuron ({self.input_size}).")

    # Proses Weighted Sum dgn Rumus: y = $z = x₁w₁ + x₂w₂ + x₃w₃ + bias
    pre_activation:float = (
      sum(input * weight for input, weight in zip(inputs, self.weights)) + self.bias
    )

    # Simpan Nilai Pre-Activation saat ini
    self.last_pre_activation = pre_activation

    # Implementasi Aktivasi
    if (self.activation == "ReLU"):
      return relu(pre_activation)

    return pre_activation

  # BACKWARD — Proses Cek Kesalahan
  def backward(self, inputs: Sequence[float], gradient_output: float) -> list[float]:
    # Validasi Input
    if len(inputs) != self.input_size:
      raise ValueError(f"Ukuran input ({len(inputs)}) tidak sesuai dengan input_size Neuron ({self.input_size}).")

    # Implementasi Aktivasi Gradient
    if self.activation == "ReLU":
      gradient_pre_activation = gradient_output * relu_derivative(self.last_pre_activation)
    else:
      gradient_pre_activation = gradient_output

    # Akumulasi Gradient Weight
    for i, input in enumerate(inputs):
      self.gradient_weights[i] += gradient_pre_activation * input

    # Akumulasi Gradient Bias
    self.gradient_bias += gradient_pre_activation

    # Gradient Input (dikirim ke Layer sebelumnya)
    gradient_input = [
      gradient_pre_activation * weight
      for weight in self.weights
    ]

    return gradient_input

  # RESET GRADIENT — Mengosongkan Gradient Weight & Bias
  def reset_gradient(self) -> None:
    self.gradient_weights = [0.0] * self.input_size
    self.gradient_bias = 0

  # AVERAGE GRADIENT — Menghitung rata² Gradient Weight & Bias
  def average_gradient(self, batch_size: int) -> None:
    # Validasi Input
    if batch_size <= 0:
      raise ValueError("batch_size harus bernilai lebih besar dari 0.")

    # Gradient Weight
    self.gradient_weights = [
      gradient_weight / batch_size
      for gradient_weight in self.gradient_weights
    ]

    # Gradient Bias
    self.gradient_bias /= batch_size

  # STEP — Update Weight & Bias
  def step(self) -> None:
    # Weight
    for i in range(len(self.weights)):
      self.weights[i] -= self.learning_rate * self.gradient_weights[i]

    # Bias
    self.bias -= self.learning_rate * self.gradient_bias