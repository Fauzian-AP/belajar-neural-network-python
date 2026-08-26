# Bagian Pengelolaan Activation Function yaitu:
# Rumus MTK pd Neural Network yg menentukan apakah suatu Neuron hrs aktif & menghasilkan Output

from typing import Union


# ====================
# === TYPE CHECKER ===
# ====================

Numeric = Union[int, float]


# ===========================
# === ACTIVATION FUNCTION ===
# ===========================

# Menghitung fungsi aktivasi ReLU (Rectified Linear Unit)
def relu(input: Numeric) -> float:
  # Rumus: f(x) = max(0, x)
  is_active = input if (input > 0) else 0

  return is_active

# Menghitung Gradient dari fungsi aktivasi ReLU
def relu_derivative(pre_activation) -> float:
  # Rumus: f'(x) = 1 jika x > 0, else 0
  is_active = 1 if (pre_activation > 0) else 0

  return is_active