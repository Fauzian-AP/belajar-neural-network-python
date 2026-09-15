"""
Bagian Pengelolaan paling dasar dari Sistem Neural Network yaitu:

Neuron / Saraf yg digunakan untuk menerima beberapa Input, memberikan Weight pd tiap Input, menambahkan Bias, lalu menghasilkan Output.
"""

from .initializer import Initializer
from .activation_functions import Activation
from src.utils.custom_types import (
  FloatVector,
  FloatSeq,
  IntPositive,
  CacheNeuron,
)

class Neuron:
  # CONSTRUCTOR — Initialization
  def __init__(
    self,
    input_size: IntPositive,
    initializer: Initializer,
    activation: Activation,
  ) -> None:
    # Menentukan brp byk input yg bisa diterima Neuron
    self.input_size: IntPositive = input_size

    # Activation yg digunakan
    self.activation: Activation = activation

    # Initializer yg digunakan
    self.initializer: Initializer = initializer

    # Menentukan pengaruh tiap Input terhadap Output
    self.weights: FloatVector = initializer(input_size)

    # Menentukan pergeseran nilai Pre-Activation
    self.bias: float = 0.0

    # Menyimpan Gradient terhadap tiap Weight
    self.gradient_weights: FloatVector = [0.0] * input_size

    # Menyimpan Gradient terhadap Bias
    self.gradient_bias: float = 0.0

    # Menyimpan Nilai² yg dibutuhkan dlm Neuron ini
    self.cache: CacheNeuron | None = None

  # FORWARD — Proses Menghasilkan Prediksi
  def forward(self, inputs: FloatSeq) -> float:
    # Validasi Inputs
    if len(inputs) != self.input_size:
      raise ValueError(f"Panjang inputs ({len(inputs)}) tdk sesuai dgn input_size ({self.input_size}).")

    """ Neuron: z = ∑(xₙ × wₙ) + b """
    pre_activation = sum(x * w for x, w in zip(inputs, self.weights)) + self.bias

    # Simpan nilai² yg diperlukan ke Cache
    self.cache = {
      "inputs": list(inputs),
      "pre_activation": pre_activation,
    }

    """ Activation: y = f(z) """
    return self.activation(pre_activation)

  # BACKWARD — Menghitung Gradient yaitu nilai yg menunjukkan seberapa besar perubahan Loss
  def backward(self, gradient_output: float) -> FloatSeq:
    # Validasi Cache
    if self.cache is None:
      raise ValueError("Backward tdk dpt dilakukan sebelum Forward.")

    # Ambil data dari Cache
    inputs = self.cache["inputs"]
    pre_activation = self.cache["pre_activation"]

    """ Activation Gradient: ∂L/∂z = (∂L/∂y) × f'(z) """
    gradient_pre_activation = (
      gradient_output * self.activation.gradient(pre_activation)
    )

    """ Gradient Weight: ∂L/∂wᵢ = (∂L/∂z) × xᵢ """
    for i, x in enumerate(inputs):
      self.gradient_weights[i] += gradient_pre_activation * x

    """ Gradient Bias: ∂L/∂b = ∂L/∂z """
    self.gradient_bias += gradient_pre_activation

    """ Gradient Input: ∂L/∂xᵢ = (∂L/∂z) × wᵢ """
    gradient_input = [
      gradient_pre_activation * weight
      for weight in self.weights
    ]

    return gradient_input

  # RESET GRADIENT — Mengosongkan Gradient Weight & Bias
  def reset_gradient(self) -> None:
    # Gradient Weight
    self.gradient_weights = [0.0] * self.input_size

    # Gradient Bias
    self.gradient_bias = 0.0

  # AVERAGE GRADIENT — Menghitung Rata² Gradient Weight & Bias
  def average_gradient(self, batch_size: IntPositive) -> None:
    # Gradient Weight
    self.gradient_weights = [
      gradient / batch_size
      for gradient in self.gradient_weights
    ]

    # Gradient Bias
    self.gradient_bias /= batch_size
