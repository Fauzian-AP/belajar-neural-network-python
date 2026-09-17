import math

from src.core.optimizer import SGD


# SGD

def test_sgd_update() -> None:
  optimizer = SGD(learning_rate=0.1)

  # Weight & Bias
  weights = [0.5, -0.2, 0.8]
  bias = 0.1

  # Gradient Weight & Bias
  gradient_weights = [0.4, -0.5, 0.2]
  gradient_bias = 0.3

  # Optimizer
  updated_weights, updated_bias = optimizer(
    weights=weights,
    bias=bias,
    gradient_weights=gradient_weights,
    gradient_bias=gradient_bias,
  )

  # Target hasil
  expected_weights = [0.46, -0.15, 0.78]
  expected_bias = 0.07

  assert all(
    math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)

    for actual, expected in zip(updated_weights, expected_weights)
  )

  assert math.isclose(updated_bias, expected_bias, rel_tol=1e-9, abs_tol=1e-9)

  print("✓ SGD calculation berhasil")
  print()


# VALIDATION

def test_learning_rate_validation() -> None:
  try:
    SGD(learning_rate=0.0)
  except Exception as error:
    print(f"Pesan Error : {error}")
  else:
    raise AssertionError("learning_rate=0 seharusnya ditolak")
  finally:
    print()

  try:
    SGD(learning_rate=-0.1)
  except Exception as error:
    print(f"Pesan Error : {error}")
  else:
    raise AssertionError("learning_rate negatif seharusnya ditolak")
  finally:
    print()


def test_parameter_validation() -> None:
  optimizer = SGD(learning_rate=0.1)

  try:
    optimizer(
      weights="invalid",
      bias=0.1,
      gradient_weights=[0.1, 0.2],
      gradient_bias=0.1,
    )
  except Exception as error:
    print(f"Pesan Error : {error}")
  else:
    raise AssertionError("weights salah tipe seharusnya ditolak")
  finally:
    print()


# MAIN

def run_tests() -> None:
  test_sgd_update()
  test_learning_rate_validation()
  test_parameter_validation()

  print("Semua test berhasil.")


if __name__ == "__main__":
  run_tests()