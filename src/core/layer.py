# Merupakan Bagian Lapisan yg mengatur Sekumpulan Neuron pd Sistem Neural Network

from pydantic import validate_call

from .neuron import Neuron
from .activation_functions import Activation
from .optimizer import Optimizer
from src.utils.custom_types import (
  ListFloat,
  SequenceFloat,
  PositiveInt,
)

class Layer:
  # CONSTRUCTOR — Initialization
  @validate_call(config={"arbitrary_types_allowed":True})
  def __init__(
    self,
    input_size: PositiveInt,
    neuron_count: PositiveInt,
    activation: Activation,
    optimizer: Optimizer,
  ) -> None:
    # Menentukan jumlah Inputan yg dpt diproses oleh tiap Neuron
    self.input_size: PositiveInt = input_size

    # Menentukan jumlah Neuron yg digunakan
    self.neuron_count: PositiveInt = neuron_count

    # Jenis Activation Function yg digunakan
    self.activation: Activation = activation

    # Jenis Optimizer yg digunakan untuk mengupdate Weight & Bias
    self.optimizer: Optimizer = optimizer

    # Simpan Neuron²
    self.neurons: list[Neuron] = [
      # Buat Neuron
      Neuron(
        input_size=input_size,
        activation=activation,
      )

      # Jalankan berdasarkan neuron_count 
      for _ in range(neuron_count)
    ]

  # FORWARD — Proses Prediksi tiap Neuron
  def forward(self, inputs: SequenceFloat) -> ListFloat:
    # Validasi Inputs
    if len(inputs) != self.input_size:
      raise ValueError(f"Panjang inputs ({len(inputs)}) tdk sesuai dgn input_size ({self.input_size}).")

    # Jalankan Method Forward pd tiap Neuron
    outputs = [
      neuron.forward(inputs)
      for neuron in self.neurons
    ]

    return outputs

  # BACKWARD — Proses Cek Kesalahan tiap Neuron
  def backward(self, gradient_outputs: SequenceFloat) -> ListFloat:
    # Validasi
    if len(gradient_outputs) != self.neuron_count:
      raise ValueError(f"Panjang gradient_outputs ({len(gradient_outputs)}) tdk sesuai dgn neuron_count ({self.neuron_count}).")

    # Wadah Gradient Input Layer
    gradient_input = [0.0] * self.input_size

    # Jalankan Method Backward pd tiap Neuron
    for neuron, gradient_output in zip(self.neurons, gradient_outputs):
      # Gradient yg dikirim Neuron
      neuron_gradient_input = neuron.backward(gradient_output)

      # Gabungkan Gradient dari semua Neuron
      for i, gradient in enumerate(neuron_gradient_input):
        gradient_input[i] += gradient

    return gradient_input

  # RESET GRADIENT — Proses Reset Gradient tiap Neuron
  def reset_gradient(self) -> None:
    for neuron in self.neurons:
      neuron.reset_gradient()

  # AVERAGE GRADIENT — Menghitung Rata² Gradient tiap Neuron
  @validate_call
  def average_gradient(self, batch_size: PositiveInt) -> None:
    for neuron in self.neurons:
      neuron.average_gradient(batch_size)
  
  # STEP — Proses Update Weight & Bias menggunakan Optimizer
  def step(self) -> None:
    # Jalankan pd tiap Neuron
    for neuron in self.neurons:
      # Jalankan Optimizer, kemudian Update Weight & Bias
      neuron.weights, neuron.bias = self.optimizer.update(
        weights=neuron.weights,
        bias=neuron.bias,
        gradient_weights=neuron.gradient_weights,
        gradient_bias=neuron.gradient_bias,
      )
