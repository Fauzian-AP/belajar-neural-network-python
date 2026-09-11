from neuron import Neuron
from custom_types import ActivationType


# ===============================================================
# TEST LEAKY RELU PADA NEURON
# ===============================================================

neuron = Neuron(
  input_size=1,
  learning_rate=0.01,
  activation=ActivationType.LEAKY_RELU,
)

# Paksa nilai agar pre_activation = -1
neuron.weights = [-1.0]
neuron.bias = 0.0


# ===============================================================
# FORWARD
# ===============================================================

output = neuron.forward(
  [1.0],
)

print(
  f"Output          : {output}"
)


# ===============================================================
# BACKWARD
# ===============================================================

gradient_input = neuron.backward(
  1.0,
)

print(
  f"Gradient Input  : {gradient_input}"
)

print(
  f"Gradient Weight : {neuron.gradient_weights}"
)

print(
  f"Gradient Bias   : {neuron.gradient_bias}"
)