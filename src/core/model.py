""" Bagian Utama untuk Pengelolaan Model Neural Network """

from .layer import Layer
from .initializer import Initializer
from .activation_functions import Activation, Linear
from .optimizer import Optimizer
from src.utils.custom_types import (
  FloatVector,
  FloatSeq,
  IntPositive,
)

class Model:
  # CONSTRUCTOR — Initialization
  def __init__(
    self,
    initializer: Initializer,
    optimizer: Optimizer,
    activation: Activation,
    architecture: tuple[IntPositive, ...] = (3, 4, 4, 2),
  ) -> None:
    # Validasi jumlah Layer
    if len(architecture) < 2:
      raise ValueError("Arcitecture minimal hrs memiliki Layer Input & Output.")

    # Validasi Layer Input 
    if architecture[0] != 3:
      raise ValueError("Input Layer minimal hrs memiliki 3 Input.")

    # Validasi Layer Output
    if architecture[-1] != 2:
      raise ValueError("Output Layer minimal hrs memiliki 2 Input.")

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

      layer_activation = Linear() if is_output_layer else activation

      # Simpan Layer
      self.layers.append(
        Layer(
          input_size=input_size,
          neuron_count=neuron_count,
          initializer=initializer,
          optimizer=optimizer,
          activation=layer_activation,
        )
      )

  # FORWARD — Proses Menghasilkan Prediksi tiap Layer
  def forward(self, inputs: FloatSeq) -> FloatVector:
    # Simpan nilai awal
    output = inputs

    # Jalankan Method Forward tiap Layer dari awal ke akhir
    for layer in self.layers:
      output = layer.forward(output)

    return output

  # BACKWARD — Proses Menghitung Gradient tiap Layer
  def backward(self, gradient_outputs: FloatSeq) -> FloatVector:
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

  # AVERAGE GRADIENT — Menghitung Rata² Gradient tiap Layer
  def average_gradient(self, batch_size: IntPositive) -> None:
    for layer in self.layers:
      layer.average_gradient(batch_size)

  # STEP — Proses Update Weight & Bias tiap Layer
  def step(self) -> None:
    for layer in self.layers:
      layer.step()