# Bagian yg mengatur Struktur Neuron
from neuron import Neuron

class Layer:
  # CONSTRUCTOR
  def __init__(
    self,
    input_size,
    neuron_count,
    learning_rate,
    activation=True
  ):
    self.neurons = [
      Neuron(input_size, learning_rate, activation)
      for _ in range(neuron_count)
    ]

  # FORWARD — Proses Prediksi tiap Neuron
  def forward(self, inputs):
    outputs = []

    for neuron in self.neurons:
      output = neuron.forward(inputs)

      outputs.append(output)

    return outputs

  # BACKWARD — Proses Cek Kesalahan tiap Neuron
  def backward(self, inputs, gradient_outputs):
    # Gradient untuk input Layer
    gradient_input = [
      0
      for _ in range(len(inputs))
    ]

    for neuron, gradient_output in zip(self.neurons, gradient_outputs):
      # Gradient yg dikirim Neuron
      neuron_gradient_input = neuron.backward(inputs, gradient_output)

      # Gabungkan Gradient dari semua Neuron
      for i, gradient in enumerate(neuron_gradient_input):
        gradient_input[i] += gradient

    return gradient_input

  # RESET GRADIENT — Proses Reset Gradient Weight & Bias tiap Neuron
  def reset_gradient(self):
    for neuron in self.neurons:
      neuron.reset_gradient()

  # AVERAGE GRADIENT — Menghitung Rata² Gradient Weight & Bias tiap Neuron
  def average_gradient(self, batch_size):
    for neuron in self.neurons:
      neuron.average_gradient(batch_size)
  
  # STEP — Proses Update Weight & Bias tiap Neuron
  def step(self):
    for neuron in self.neurons:
      neuron.step()