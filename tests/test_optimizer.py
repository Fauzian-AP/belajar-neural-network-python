import numpy as np

from src.core.optimizer import SGD


# SGD

print("=== SGD ===")

optimizer = SGD(learning_rate=0.1)


# WEIGHT & BIAS

W = np.array(
  [
    [0.5, -0.2, 0.8, 0.3],
    [0.1, 0.4, -0.6, 0.7],
    [-0.3, 0.9, 0.2, -0.5],
  ],
  dtype=np.float64,
)

b = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float64)


# GRADIENT WEIGHT & BIAS

gradient_W = np.array(
  [
    [0.4, -0.5, 0.2, 0.1],
    [-0.3, 0.6, -0.4, 0.2],
    [0.5, -0.1, 0.3, -0.7],
  ],
  dtype=np.float64,
)

gradient_b = np.array([0.3, -0.4, 0.2, 0.5], dtype=np.float64,)


# OPTIMIZER

updated_W, updated_b = optimizer(
  W=W,
  b=b,
  gradient_W=gradient_W,
  gradient_b=gradient_b,
)

print(f"W sebelum :\n{W}\n")
print(f"b sebelum : {b}\n")

print(f"Gradient W :\n{gradient_W}\n")
print(f"Gradient b : {gradient_b}\n")

print(f"W sesudah :\n{updated_W}\n")
print( f"b sesudah : {updated_b}\n")


# SGD CALCULATION

print("=== SGD Calculation Validation ===")

expected_W = W - (0.1 * gradient_W)
expected_b = b - (0.1 * gradient_b)

print(f"Update W:\n{updated_W}\n")
print(f"Update b:\n{updated_b}\n")

print(f"Expected W:\n{expected_W}\n")
print(f"Expected b:\n{expected_b}\n")

assert np.allclose(updated_W, expected_W)
assert np.allclose(updated_b, expected_b)

print("✓ SGD Weight calculation berhasil")
print("✓ SGD Bias calculation berhasil\n")


# OUTPUT TYPE

print("=== Output Type Validation ===")

assert isinstance(updated_W, np.ndarray,)
assert isinstance(updated_b, np.ndarray)

assert updated_W.dtype == np.float64
assert updated_b.dtype == np.float64

print(f"✓ Updated W berupa NumPy {type(updated_W)}")
print(f"✓ Updated b berupa NumPy {type(updated_b)}")
print(f"✓ Dtype Updated W {updated_W.dtype}")
print(f"✓ Dtype Updated b {updated_b.dtype}\n")


# OUTPUT SHAPE

print("=== Output Shape Validation ===")

assert updated_W.shape == W.shape
assert updated_b.shape == b.shape

print(f"✓ Shape W : {updated_W.shape}")
print(f"✓ Shape b : {updated_b.shape}\n")


# LEARNING RATE VALIDATION

print("=== Learning Rate Validation ===")

try:
  SGD(learning_rate=0.0)
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")
else:
  raise AssertionError("learning_rate=0 seharusnya ditolak.")
finally:
  print()


try:
  SGD(learning_rate=-0.01)
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")
else:
  raise AssertionError("learning_rate negatif seharusnya ditolak.")
finally:
  print()


# PARAMETER TYPE VALIDATION

print("=== Parameter Type Validation ===")

try:
  optimizer(
    W="invalid",
    b=b,
    gradient_W=gradient_W,
    gradient_b=gradient_b,
  )
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")
else:
  raise AssertionError("W salah tipe seharusnya ditolak.")
finally:
  print()


try:
  optimizer(
    W=W,
    b="invalid",
    gradient_W=gradient_W,
    gradient_b=gradient_b,
  )
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")
else:
  raise AssertionError("b salah tipe seharusnya ditolak.")
finally:
  print()


try:
  optimizer(
    W=W,
    b=b,
    gradient_W="invalid",
    gradient_b=gradient_b,
  )
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")
else:
  raise AssertionError("gradient_W salah tipe seharusnya ditolak.")
finally:
  print()


try:
  optimizer(
    W=W,
    b=b,
    gradient_W=gradient_W,
    gradient_b="invalid",
  )
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")
else:
  raise AssertionError("gradient_b salah tipe seharusnya ditolak.")
finally:
  print()


# FINISH

print("Semua Test Optimizer Berhasil.")