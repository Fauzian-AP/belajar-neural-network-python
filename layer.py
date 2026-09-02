# Merupakan Bagian Lapisan yg mengatur Sekumpulan Neuron pd Sistem Neural Network

from pydantic import validate_call

from neuron import Neuron
from custom_types import (
  ListFloat,
  SequenceFloat,
  PositiveFloat,
  PositiveInt,
  ActivationType,
)

class Layer:
  # CONSTRUCTOR — Initialization
  @validate_call
  def __init__(
    self,
    input_size: PositiveInt,
    neuron_count: PositiveInt,
    learning_rate: PositiveFloat,
    activation: ActivationType = ActivationType.NONE,
  ) -> None:
    # Menentukan jumlah Inputan yg dpt diterima sebuah Neuron
    self.input_size: int = input_size

    # Menentukan jumlah Neuron yg digunakan
    self.neuron_count: int = neuron_count

    # Menjalankan Neuron²
    self.neurons: list[Neuron] = [
      Neuron(input_size, learning_rate, activation)

      for _ in range(neuron_count)
    ]

  # FORWARD — Proses Prediksi tiap Neuron
  def forward(self, inputs: SequenceFloat) -> ListFloat:
    # Validasi Inputs
    if len(inputs) != self.input_size:
      raise ValueError(f"Ukuran inputs ({len(inputs)}) tidak sesuai dengan input_size ({self.input_size}).")

    # Jalankan Method Forward tiap Neuron
    outputs = [
      neuron.forward(inputs)
      for neuron in self.neurons
    ]

    return outputs

  # BACKWARD — Proses Cek Kesalahan tiap Neuron
  def backward(
    self,
    inputs: SequenceFloat,
    gradient_outputs: SequenceFloat
  ) -> ListFloat:
    # Validasi
    if len(inputs) != self.input_size:
      raise ValueError(f"Panjang inputs ({len(inputs)}) tdk sesuai dgn input_size ({self.input_size}).")

    if len(gradient_outputs) != self.neuron_count:
      raise ValueError(f"Panjang gradient_outputs ({len(gradient_outputs)}) tdk sesuai dgn neuron_count ({self.neuron_count}).")

    # Gradient untuk input Layer
    gradient_input = [0.0] * self.input_size

    for neuron, gradient_output in zip(self.neurons, gradient_outputs):
      # Gradient yg dikirim Neuron
      neuron_gradient_input = neuron.backward(inputs, gradient_output)

      # Gabungkan Gradient dari semua Neuron
      for i, gradient in enumerate(neuron_gradient_input):
        gradient_input[i] += gradient

    return gradient_input

  # RESET GRADIENT — Proses Reset Gradient Weight & Bias tiap Neuron
  def reset_gradient(self) -> None:
    for neuron in self.neurons:
      neuron.reset_gradient()

  # AVERAGE GRADIENT — Menghitung Rata² Gradient Weight & Bias tiap Neuron
  @validate_call
  def average_gradient(self, batch_size: PositiveInt) -> None:
    for neuron in self.neurons:
      neuron.average_gradient(batch_size)
  
  # STEP — Proses Update Weight & Bias tiap Neuron
  def step(self) -> None:
    for neuron in self.neurons:
      neuron.step()