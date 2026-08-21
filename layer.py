# Bagian yg mengatur Struktur Neuron
from neuron import Neuron

class Layer:
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

  def forward(self, inputs):
    outputs = []

    for neuron in self.neurons:
      output = neuron.forward(inputs)

      outputs.append(output)

    return outputs

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

  def step(self):
    for neuron in self.neurons:
      neuron.step()