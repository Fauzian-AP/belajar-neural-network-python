# Bagian Utama untuk membuat Model Machine Learning

from pydantic import validate_call

from .layer import Layer
from .activation_functions import Activation, Linear
from .optimizer import Optimizer

from src.utils.custom_types import (
  ListFloat,
  SequenceFloat,
  PositiveInt,
)

class Model:
  # CONSTRUCTOR
  @validate_call(config={"arbitrary_types_allowed":True})
  def __init__(
    self,
    optimizer: Optimizer,
    architecture: tuple[PositiveInt, ...] = (3, 4, 4, 2),
    activation: Activation = None,
  ) -> None:
    # Validasi jumlah Layer
    if len(architecture) < 2:
      raise ValueError("Arcitecture minimal hrs memiliki Layer Input & Output.")

    # Validasi Layer Input 
    if architecture[0] != 3:
      raise ValueError("Input Layer minimal hrs memiliki 3 Input Data.")

    # Validasi Layer Output
    if architecture[-1] != 2:
      raise ValueError("Output Layer minimal hrs memiliki 2 Input Data.")

    # Wadah seluruh Layer
    self.layers: list[Layer] = []

    # Buat Arsitektur Model
    for index in range(len(architecture) - 1):
      # Jumlah Input Data pd tiap Layer
      input_size = architecture[index]

      # Jumlah Neuron pd tiap Layer
      neuron_count = architecture[index + 1]

      # Cek apakah Layer merupakan kategori Output
      is_output_layer = index == len(architecture) - 2
    
      # Input Layer  ⟶  Activation
      # Hidden Layer  ⟶  Activation
      # Output Layer  ⟶  Linear
      layer_activation = (
        Linear() if is_output_layer else activation
      )

      # Simpan Layer
      self.layers.append(
        Layer(
          input_size=input_size,
          neuron_count=neuron_count,
          optimizer=optimizer,
          activation=layer_activation,
        )
      )

  # FORWARD — Proses Prediksi tiap Layer
  def forward(self, inputs: SequenceFloat) -> ListFloat:
    output = inputs

    # Jalankan Method Forward tiap Layer dari awal ke akhir
    for layer in self.layers:
      output = layer.forward(output)

    return output

  # BACKWARD — Proses Cek kesalahan tiap Layer
  def backward(self, gradient_outputs: SequenceFloat) -> ListFloat:
    # Gradient awal
    gradient = gradient_outputs
    
    # Jalankan Method Backward dari Layer terakhir ke awal
    for layer in reversed(self.layers):
      gradient = layer.backward(gradient)

    return gradient

  # RESET GRADIENT — Mengosongkan Gradient tiap Layer
  def reset_gradient(self) -> None:
    for layer in self.layers:
      layer.reset_gradient()

  # AVERAGE GRADIENT — Menghitung rata²
  @validate_call
  def average_gradient(self, batch_size: PositiveInt) -> None:
    for layer in self.layers:
      layer.average_gradient(batch_size)

  # STEP — Proses Update Weight & Bias tiap Layer
  def step(self) -> None:
    for layer in self.layers:
      layer.step()