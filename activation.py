# Bagian Pengelolaan Activation Function yaitu:
# Rumus MTK pd Neural Network yg menentukan apakah suatu Neuron hrs aktif & menghasilkan Output

from custom_types import Numeric

# RELU ACTIVATION

# Menghitung fungsi aktivasi ReLU (Rectified Linear Unit)
def ReLU(input: Numeric) -> float:
  """ f(x) = max(0, x) """
  return input if (input > 0) else 0

# Menghitung Gradient dari fungsi aktivasi ReLU
def ReLU_gradient(pre_activation: Numeric) -> float:
  """
  f'(x) = { 1 jika x > 0
          { 0 jika x ⩽ 0
  """
  return 1 if (pre_activation > 0) else 0