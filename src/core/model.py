# Bagian Utama untuk membuat Model Machine Learning

from pydantic import validate_call

from .layer import Layer
from src.utils.custom_types import (
  ListFloat,
  SequenceFloat,
  PositiveFloat,
  PositiveInt,
  ActivationType,
)

class Model:
  # CONSTRUCTOR
  @validate_call
  def __init__(
    self,
    learning_rate: PositiveFloat,
    architecture: tuple[PositiveInt, ...] = (3, 4, 4, 2),
    activation: ActivationType = ActivationType.RELU
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

    # Buat Arsitektur Model berdasarkan Architecture
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
        ActivationType.NONE if is_output_layer else activation
      )

      # Simpan Layer
      self.layers.append(
        Layer(
          input_size=input_size,
          neuron_count=neuron_count,
          learning_rate=learning_rate,
          activation=layer_activation,
        )
      )

  # FORWARD — Proses Prediksi tiap Layer (Maju)
  def forward(self, inputs: SequenceFloat) -> ListFloat:
    output = inputs

    # Jalankan Method Forward tiap Layer sehingga Outputnya di Passing terus
    for layer in self.layers:
      output = layer.forward(output)

    return output

  # BACKWARD — Proses Cek kesalahan tiap Layer (Mundur)
  def backward(self, gradient_outputs: SequenceFloat) -> ListFloat:
    # Gradient awal
    gradient = gradient_outputs
    
    # Jalankan Method Backward dari Layer terakhir ke awal
    for i in range(len(self.layers) - 1, -1, -1):
      # Simpan Layer tiap index
      layer = self.layers[i]

      # Kirim Gradient ke Layer sebelumnya
      gradient = layer.backward(gradient)

    return gradient

  # RESET GRADIENT — Mengosongkan Gradient Weight & Bias tiap Layer
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