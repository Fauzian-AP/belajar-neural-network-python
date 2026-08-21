from neuron import Neuron

neuron = Neuron(
  input_size=2,
  learning_rate=0.01,
  activation=False
)

# Buat nilai Weight dan Bias tetap
# supaya hasil mudah dihitung

neuron.weights = [1, 2]
neuron.bias = 0

print("=== Gradient Awal ===")
print("Gradient Weight :", neuron.gradient_weights)
print("Gradient Bias   :", neuron.gradient_bias)

# SAMPLE 1

inputs = [2, 3]

neuron.forward(inputs)
neuron.backward(inputs, gradient_output=1)

print()
print("=== Setelah Sample 1 ===")
print("Gradient Weight :", neuron.gradient_weights)
print("Gradient Bias   :", neuron.gradient_bias)

# SAMPLE 2

inputs = [4, 5]

neuron.forward(inputs)
neuron.backward(inputs, gradient_output=1)

print()
print("=== Setelah Sample 2 ===")
print("Gradient Weight :", neuron.gradient_weights)
print("Gradient Bias   :", neuron.gradient_bias)

# RESET

neuron.reset_gradient()

print()
print("=== Setelah Reset ===")
print("Gradient Weight :", neuron.gradient_weights)
print("Gradient Bias   :", neuron.gradient_bias)