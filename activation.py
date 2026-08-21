# Bagian untuk Activation Function yaitu:
# Rumus MTK pd Neural Network yg menentukan apakah suatu Neuron hrs aktif & menghasilkan keluaran.

# Menggunakan ReLU (Rectified Linear Unit)
# Rumus: f(x) = max(0, x)

# Untuk Forward
def relu(x):
  is_active = x if (x > 0) else 0

  return is_active

# Untuk Backward
def relu_derivative(x):
  is_active = 1 if (x > 0) else 0

  return is_active