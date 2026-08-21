# Bagian Utama untuk membuat Model

from layer import Layer   # Import Neuron

class Model:
  def __init__(self, learning_rate):
    self.layers = [
      # Hidden Layer 1
      Layer(
        input_size=3,
        neuron_count=4,
        learning_rate = learning_rate,
        activation=True
      ),
    
      # Hidden Layer 2
      Layer(
        input_size=4,
        neuron_count=4,
        learning_rate = learning_rate,
        activation=True
      ),
    
      # Output Layer
      Layer(
        input_size=4,
        neuron_count=2,
        learning_rate = learning_rate,
        activation=False
      ),
    ]

  def forward(self, inputs):
    output = inputs

    # Forward semua Layer
    for layer in self.layers:
      output = layer.forward(output)

    return output

  def backward(self, inputs, gradient_output):
    # Simpan Output setiap Layer
    layer_outputs = []

    output = inputs

    # Forward tiap Layer untuk mendptkan Gradient Inputnya
    for layer in self.layers:
      output = layer.forward(output)

      layer_outputs.append(output)

    gradient = gradient_output
    
    # Backward dari Layer terakhir ke awal untuk mendptkan Gradientnya
    for i in range(len(self.layers) - 1, -1, -1):
      layer = self.layers[i]

      if (i == 0):
        layer_input = inputs
      else:
        layer_input = layer_outputs[i - 1]
  
      gradient = layer.backward(layer_input, gradient)

    return gradient

  def step(self):
    for layer in self.layers:
      layer.step()