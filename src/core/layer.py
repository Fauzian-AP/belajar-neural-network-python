"""
Bagian Pengelolaan Layer, yaitu:

Membuat & Mengatur sekumpulan Neuron agar dapat memproses Input secara bersamaan dan menghasilkan sekumpulan Output.
"""

from .initializer import Initializer
from .activation_functions import Activation
from .optimizer import Optimizer
from .neuron import Neuron
from src.utils.custom_types import (
  FloatVector,
  FloatSequence,
  IntPositive,
)

class Layer:
  # CONSTRUCTOR — Initialization
  def __init__(
    self,
    input_size: IntPositive,
    neuron_count: IntPositive,
    initializer: Initializer,
    activation: Activation,
    optimizer: Optimizer,
  ) -> None:
    # Menentukan brp byk input yg bisa diterima oleh tiap Neuron
    self.input_size: IntPositive = input_size

    # Menentukan jumlah Neuron yg digunakan
    self.neuron_count: IntPositive = neuron_count

    # Activation yg digunakan
    self.activation: Activation = activation

    # Initializer yg digunakan
    self.initializer: Initializer = initializer

    # Optimizer yg digunakan
    self.optimizer: Optimizer = optimizer

    # Buat & Simpan Neuron²
    self.neurons: list[Neuron] = [
      Neuron(
        input_size=input_size,
        initializer=initializer,
        activation=activation,
      )

      for _ in range(neuron_count)
    ]

  # FORWARD — Proses Menghasilkan Prediksi tiap Neuron
  def forward(self, inputs: FloatSequence) -> FloatVector:
    # Validasi Argument
    if len(inputs) != self.input_size:
      raise ValueError(f"Panjang inputs ({len(inputs)}) tdk sesuai dgn input_size ({self.input_size}).")

    # Jalankan Method Forward pd tiap Neuron
    outputs = [
      neuron.forward(inputs)
      for neuron in self.neurons
    ]

    return outputs

  # BACKWARD — Proses Menghitung Gradient tiap Neuron
  def backward(self, gradient_outputs: FloatSequence) -> FloatVector:
    # Validasi Argument
    if len(gradient_outputs) != self.neuron_count:
      raise ValueError(f"Panjang gradient_outputs ({len(gradient_outputs)}) tdk sesuai dgn neuron_count ({self.neuron_count}).")

    # Wadah Gradient Input
    gradient_input = [0.0] * self.input_size

    # Jalankan Method Backward pd tiap Neuron
    for neuron, gradient_output in zip(self.neurons, gradient_outputs):
      # Gradient yg dikirim Neuron
      neuron_gradient_input = neuron.backward(gradient_output)

      # Gabungkan Gradient tiap index
      for i, gradient in enumerate(neuron_gradient_input):
        gradient_input[i] += gradient

    return gradient_input

  # RESET GRADIENT — Proses Mengosongkan Gradient tiap Neuron
  def reset_gradient(self) -> None:
    for neuron in self.neurons:
      neuron.reset_gradient()

  # AVERAGE GRADIENT — Menghitung Rata² Gradient tiap Neuron
  def average_gradient(self, batch_size: IntPositive) -> None:
    for neuron in self.neurons:
      neuron.average_gradient(batch_size)
  
  # STEP — Proses Update Weight & Bias menggunakan Optimizer
  def step(self) -> None:
    for neuron in self.neurons:
      # Optimize, lalu kemudian Update Weight & Bias
      neuron.weights, neuron.bias = self.optimizer(
        weights=neuron.weights,
        bias=neuron.bias,
        gradient_weights=neuron.gradient_weights,
        gradient_bias=neuron.gradient_bias,
      )
