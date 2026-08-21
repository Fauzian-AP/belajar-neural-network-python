from neuron import Neuron

neuron = Neuron(
  input_size=2,
  learning_rate=0.01,
  activation=False
)

neuron.gradient_weights = [6, 12]
neuron.gradient_bias = 9

print("=== Sebelum Average ===")
print("Gradient Weight :", neuron.gradient_weights)
print("Gradient Bias   :", neuron.gradient_bias)

neuron.average_gradient(3)

print()
print("=== Setelah Average ===")
print("Gradient Weight :", neuron.gradient_weights)
print("Gradient Bias   :", neuron.gradient_bias)