# Bagian Utama untuk membuat Model Machine Learning

from pydantic import validate_call

from layer import Layer   # Import Layer
from custom_types import (
  ListFloat,
  SequenceFloat,
  PositiveFloat,
  PositiveInt,
  ActivationType,
)

class Model:
  # CONSTRUCTOR
  @validate_call
  def __init__(self, learning_rate: PositiveFloat) -> None:
    self.layers: list[Layer] = [
      # Input Layer
      Layer(
        input_size = 3,
        neuron_count = 4,
        learning_rate = learning_rate,
        activation = ActivationType.NONE,
      ),
    
      # Hidden Layer
      Layer(
        input_size = 4,
        neuron_count = 4,
        learning_rate = learning_rate,
        activation = ActivationType.NONE,
      ),
    
      # Output Layer
      Layer(
        input_size = 4,
        neuron_count = 2,
        learning_rate = learning_rate,
        activation = ActivationType.NONE,
      ),
    ]

  # FORWARD — Proses Prediksi tiap Layer (Maju)
  def forward(self, inputs: SequenceFloat) -> ListFloat:
    output = inputs

    # Forward semua Layer sehingga Outputnya di Passing terus
    for layer in self.layers:
      output = layer.forward(output)

    return output

  # BACKWARD — Proses Cek kesalahan tiap Layer (Mundur)
  def backward(
    self,
    inputs: SequenceFloat,
    gradient_outputs: SequenceFloat, 
  ) -> ListFloat:
    # Simpan Output setiap Layer
    layer_outputs: list[ListFloat] = []

    output = inputs

    # Forward tiap Layer untuk mendptkan Outputnya
    for layer in self.layers:
      output = layer.forward(output)

      layer_outputs.append(output)

    gradient = gradient_outputs
    
    # Backward dari Layer terakhir ke awal untuk mendptkan Gradientnya
    for i in range(len(self.layers) - 1, -1, -1):
      layer = self.layers[i]

      # Tentukan Mana yg Input Layer
      layer_input = inputs if (i == 0) else layer_outputs[i - 1]

      gradient = layer.backward(layer_input, gradient)

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