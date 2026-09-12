# Bagian Pengelolaan Activation Function yaitu:
# Fungsi Aktivasi pd Neural Network yg menentukan Output suatu Neuron berdasarkan nilai Pre Activation

from typing_extensions import Final
from neural_network.utils.custom_types import Numeric

# RELU ACTIVATION

# Menghitung fungsi aktivasi ReLU (Rectified Linear Unit)
def ReLU(pre_activation: Numeric) -> float:
  """ f(x) = max(0, x) """
  return (
    pre_activation if pre_activation > 0.0 else 0.0
  )

# Menghitung Gradient dari fungsi aktivasi ReLU
def ReLU_Gradient(pre_activation: Numeric) -> float:
  """
  f'(x) = { 1 jika x > 0
          { 0 jika x ⩽ 0
  """
  return (
    1.0 if pre_activation > 0.0 else 0.0
  )


# LEAKY RELU ACTIVATION

# Nilai Alpha menentukan seberapa bsr bagian negatif yg tetap dilewatkan oleh Leaky ReLU
LEAKY_RELU_ALPHA: Final = 0.01

# Menghitung fungsi aktivasi Leaky ReLU
def Leaky_ReLU(pre_activation: Numeric) -> float:
  """
  f(x) = { x  jika x > 0
         { αx jika x <= 0
  """
  return (
    pre_activation if pre_activation > 0.0 else LEAKY_RELU_ALPHA * pre_activation
  )

# Menghitung Gradient dari fungsi aktivasi Leaky ReLU
def Leaky_ReLU_Gradient(pre_activation: Numeric) -> float:
  """
  f'(x) = { 1 jika x > 0
          { α jika x <= 0
  """
  return (
    1.0 if pre_activation > 0.0 else LEAKY_RELU_ALPHA
  )