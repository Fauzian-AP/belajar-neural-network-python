import numpy as np

from src.core.activation_functions import (
  Linear,
  ReLU,
  LeakyReLU,
)


# LINEAR


print("=== Linear ===")

linear = Linear()

inputs = np.array([5.0, -5.0, 0.0], dtype=np.float64)

outputs = linear(inputs)
gradients = linear.gradient(inputs)

print(f"Input    : {inputs}")
print(f"Output   : {outputs}")
print(f"Gradient : {gradients}")

assert np.array_equal(outputs, np.array([5.0, -5.0, 0.0]))
assert np.array_equal(gradients, np.array([1.0, 1.0, 1.0]))


# RELU ACTIVATION


print()
print("=== ReLU ===")

relu = ReLU()

inputs = np.array([5.0, -5.0, 0.0], dtype=np.float64)

outputs = relu(inputs)
gradients = relu.gradient(inputs)

print(f"Input    : {inputs}")
print(f"Output   : {outputs}")
print(f"Gradient : {gradients}")

assert np.array_equal(outputs, np.array([5.0, 0.0, 0.0]))
assert np.array_equal(gradients, np.array([1.0, 0.0, 0.0]))


# LEAKY RELU

print()
print("=== LeakyReLU ===")

leaky_relu = LeakyReLU(alpha=0.01)

inputs = np.array([5.0, -5.0, 0.0], dtype=np.float64)

outputs = leaky_relu(inputs)
gradients = leaky_relu.gradient(inputs)

print(f"Input    : {inputs}")
print(f"Output   : {outputs}")
print(f"Gradient : {gradients}")
print(f"Alpha    : {leaky_relu.alpha}")

assert np.allclose(outputs, np.array([5.0, -0.05, 0.0]))
assert np.allclose( gradients, np.array([1.0, 0.01, 0.01]))
assert leaky_relu.alpha == 0.01


# OUTPUT TYPE

print()
print("=== NumPy Array Validation ===")

assert isinstance(linear(inputs), np.ndarray)
assert isinstance(relu(inputs), np.ndarray)
assert isinstance(leaky_relu(inputs), np.ndarray)

print("✓ Semua Output berupa NumPy ndarray\n")


# INPUT TYPE

print("=== Input Type Validation ===")

try:
  relu("Hello")
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")
else:
  raise AssertionError("ReLU seharusnya menolak input String.")
finally:
  print()


# ALPHA VALIDATION

print("=== Alpha Validation ===")

try:
  LeakyReLU(alpha=0.0)
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")
else:
  raise AssertionError("Alpha 0.0 seharusnya ditolak.")
finally:
  print()


try:
  activation = LeakyReLU(alpha=1.0)

  print(f"Isi Alpha : {activation.alpha}")

  assert activation.alpha == 1.0
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")
finally:
  print()


try:
  LeakyReLU(alpha=1.5)
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")
else:
  raise AssertionError("Alpha 1.5 seharusnya ditolak.")
finally:
  print()


# FINISH

print("Semua Test Activation Berhasil.")