# Bagian Pengelolaan Activation Function yaitu:
# Rumus MTK pd Neural Network yg menentukan apakah suatu Neuron hrs aktif & menghasilkan Output

from typing import Final
from custom_types import Numeric


# RELU ACTIVATION

# Menghitung fungsi aktivasi ReLU (Rectified Linear Unit)
def ReLU(input: Numeric) -> float:
  """ f(x) = max(0, x) """
  return input if (input > 0.0) else 0.0

# Menghitung Gradient dari fungsi aktivasi ReLU
def ReLU_Gradient(pre_activation: Numeric) -> float:
  """
  f'(x) = { 1 jika x > 0
          { 0 jika x ⩽ 0
  """
  return 1.0 if (pre_activation > 0.0) else 0.0


# LEAKY RELU ACTIVATION

# Nilai Alpha menentukan seberapa bsr bagian negatif yg tetap dilewatkan oleh Leaky ReLU
LEAKY_RELU_ALPHA: Final = 0.01

# Menghitung fungsi aktivasi Leaky ReLU
def Leaky_ReLU(input: Numeric) -> float:
  """
  f(x) = { x  jika x > 0
         { αx jika x <= 0
  """
  return (
    input if (input > 0.0) else LEAKY_RELU_ALPHA * input
  )

# Menghitung Gradient dari fungsi aktivasi Leaky ReLU
def Leaky_ReLU_Gradient(pre_activation: Numeric) -> float:
  """
  f'(x) = { 1 jika x > 0
          { α jika x <= 0
  """
  return (
    1.0 if (pre_activation > 0.0) else LEAKY_RELU_ALPHA
  )