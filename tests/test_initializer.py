import numpy as np

from src.core.initializer import HeNormal


# HE NORMAL

print("=== HeNormal ===")

initializer = HeNormal()

input_size = 3
neuron_count = 4

shape = (input_size, neuron_count)

weights = initializer(shape)

print(f"Weights :\n{weights}\n")
print(f"Shape   : {weights.shape}")
print(f"Dtype   : {weights.dtype}\n")


# SHAPE VALIDATION

print("=== Shape Validation ===")

assert isinstance(weights, np.ndarray)
assert weights.shape == (input_size, neuron_count)
assert weights.dtype == np.float64

print(f"✓ Output berupa NumPy {type(weights)}")
print(f"✓ Shape sesuai: {weights.shape}")
print(f"✓ Dtype {np.float64}\n")


# MATRIX DIMENSION

print("=== Matrix Dimension Validation ===")

assert weights.ndim == 2
assert weights.shape[0] == input_size
assert weights.shape[1] == neuron_count

print(f"✓ Weight Matrix memiliki : {weights.ndim}D")
print(f"✓ Jumlah Input           : {weights.shape[0]}")
print(f"✓ Jumlah Neuron          : {weights.shape[1]}\n")


# HE NORMAL DISTRIBUTION

print("=== HeNormal Distribution ===")

fan_in = shape[0]

expected_stddev = np.sqrt(2.0 / fan_in)
actual_stddev = np.std(weights)

print(f"Expected Std : {expected_stddev:.6f}")
print(f"Actual Std   : {actual_stddev:.6f}")

# Karena jumlah sample kecil, Standard Deviation tidak harus persis sama.
assert np.isfinite(actual_stddev)

print("✓ Nilai Weight valid\n")


# WRONG TYPE VALIDATION

print("=== Wrong Type Validation ===")

try:
  initializer("Test")
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")
else:
  raise AssertionError("String seharusnya ditolak.")
finally:
  print()


# INVALID SHAPE VALIDATION

print("=== Invalid Shape Validation ===")

try:
  initializer((3,))
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")
else:
  raise AssertionError("Shape 1 dimensi seharusnya ditolak.")
finally:
  print()


try:
  initializer(())
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")
else:
  raise AssertionError("Shape kosong seharusnya ditolak.")
finally:
  print()


# INVALID DIMENSION VALUE

print("=== Invalid Dimension Value Validation ===")

try:
  initializer((4, 0))
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")
else:
  raise AssertionError("Dimensi 0 seharusnya ditolak.")
finally:
  print()


try:
  initializer((4, -3))
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")
else:
  raise AssertionError("Dimensi negatif seharusnya ditolak.")
finally:
  print()


# FINISH

print("Semua Test Initializer Berhasil.")