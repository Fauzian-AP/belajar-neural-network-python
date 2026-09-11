# Merupakan Bagian paling dasar dari Sistem Neural Network yaitu
# sebuah Sistem Neuron / Saraf yg berisi puluhan, ribuan, atau jutaan
# proses dibawah ini

import random
import math
from pydantic import validate_call

from activation import (
  ReLU,
  ReLU_Gradient,
  Leaky_ReLU,
  Leaky_ReLU_Gradient,
)

from custom_types import (
  ListFloat,
  SequenceFloat,
  PositiveFloat,
  PositiveInt,
  ActivationType,
  CacheNeuron,
)

class Neuron:
  # CONSTRUCTOR — Initialization
  @validate_call
  def __init__(
    self,
    input_size: PositiveInt,
    learning_rate: PositiveFloat,
    activation: ActivationType = ActivationType.NONE,
  ) -> None:
    # Menentukan jumlah Inputan yg dpt diproses oleh Neuron
    self.input_size: int = input_size

    # Menentukan seberapa besar perubahan Weight & Bias tiap update
    self.learning_rate: float = learning_rate

    # Jenis Activation yg digunakan
    self.activation: ActivationType = activation

    # Menentukan seberapa besar pengaruh tiap Input terhadap Output
    self.weights: ListFloat = [
      # Inisialisasi He / Kaiming — Menentukan skala awal berdasarkan jumlah inputan
      random.gauss(0.0, math.sqrt(2.0 / input_size))

      # Tiap input punya Weight sendiri
      for _ in range(input_size) 
    ]
    
    # Menentukan pergeseran nilai pd Pre-Activation
    self.bias: float = 0.0

    # Menunjukkan seberapa sensitif Loss terhadap Weight
    self.gradient_weights: ListFloat = [0.0] * input_size   # Tiap input ada

    # Menunjukkan seberapa sensitif Loss terhadap Bias
    self.gradient_bias: float = 0.0

    # Menyimpan Nilai² yg dibutuhkan
    self.cache: CacheNeuron | None = None

  # FORWARD — Proses Prediksi
  def forward(self, inputs: SequenceFloat) -> float:
    # Validasi Inputs
    if len(inputs) != self.input_size:
      raise ValueError(f"Panjang inputs ({len(inputs)}) tdk sesuai dgn input_size ({self.input_size}).")

    """ y = f(z) = f( ∑(xₙ × wₙ) + b ) """
    pre_activation = sum(x * w for x, w in zip(inputs, self.weights)) + self.bias

    # Simpan nilai² yg diperlukan ke Cache
    self.cache = {
      "inputs": list(inputs),
      "pre_activation": pre_activation,
    }

    # Implementasi Aktivasi
    match self.activation:
      # ReLU
      case ActivationType.RELU:
        return ReLU(pre_activation)

      # Leaky ReLU
      case ActivationType.LEAKY_RELU:
        return Leaky_ReLU(pre_activation)

      # No Activation
      case ActivationType.NONE:
        return pre_activation

  # BACKWARD — Proses Cek Kesalahan
  def backward(self, gradient_output: float) -> ListFloat:
    # Validasi
    if self.cache is None:
      raise ValueError("Backward tdk dpt dilakukan sebelum Forward.")

    # Ambil data dari Cache
    inputs = self.cache["inputs"]
    pre_activation = self.cache["pre_activation"]

    # Implementasi Aktivasi Gradient
    match self.activation:
      # ReLU
      case ActivationType.RELU:
        gradient_pre_activation = (
          gradient_output * ReLU_Gradient(pre_activation)
        )

      # Leaky ReLU
      case ActivationType.LEAKY_RELU:
        gradient_pre_activation = (
          gradient_output * Leaky_ReLU_Gradient(pre_activation)
        )

      # No Activation
      case ActivationType.NONE:
        gradient_pre_activation = gradient_output

    # Akumulasi Gradient Weight
    for i, x in enumerate(inputs):
      self.gradient_weights[i] += gradient_pre_activation * x

    # Akumulasi Gradient Bias
    self.gradient_bias += gradient_pre_activation

    # Gradient Input (dikirim ke Layer sebelumnya)
    gradient_input = [
      gradient_pre_activation * w
      for w in self.weights
    ]

    return gradient_input

  # RESET GRADIENT — Mengosongkan Gradient Weight & Bias
  def reset_gradient(self) -> None:
    # Gradient Weight
    self.gradient_weights = [0.0] * self.input_size

    # Gradient Bias
    self.gradient_bias = 0.0

  # AVERAGE GRADIENT — Menghitung rata² Gradient Weight & Bias
  @validate_call
  def average_gradient(self, batch_size: PositiveInt) -> None:
    # Gradient Weight
    self.gradient_weights = [gw / batch_size for gw in self.gradient_weights]

    # Gradient Bias
    self.gradient_bias /= batch_size

  # STEP — Update Weight & Bias
  def step(self) -> None:
    # Weight
    for i in range(self.input_size):
      self.weights[i] -= self.learning_rate * self.gradient_weights[i]

    # Bias
    self.bias -= self.learning_rate * self.gradient_bias