import math
from beartype.roar import BeartypeCallHintParamViolation

from src.core.optimizer import SGD


def test_sgd_update() -> None:
  optimizer = SGD(learning_rate=0.1)

  weights = [0.5, -0.2, 0.8]
  bias = 0.1

  gradient_weights = [0.4, -0.5, 0.2]
  gradient_bias = 0.3

  updated_weights, updated_bias = optimizer(
    weights=weights,
    bias=bias,
    gradient_weights=gradient_weights,
    gradient_bias=gradient_bias,
  )

  expected_weights = [0.46, -0.15, 0.78]
  expected_bias = 0.07

  assert all(
    math.isclose(
      actual,
      expected,
      rel_tol=1e-9,
      abs_tol=1e-9,
    )

    for actual, expected in zip(updated_weights, expected_weights)
  )

  assert math.isclose(
    updated_bias,
    expected_bias,
    rel_tol=1e-9,
    abs_tol=1e-9,
  )

  print("✓ SGD calculation berhasil")


def test_learning_rate_validation() -> None:
  try:
    SGD(learning_rate=0.0)
  except BeartypeCallHintParamViolation:
    print("✓ learning_rate=0 ditolak")
  else:
    raise AssertionError("learning_rate=0 seharusnya ditolak")

  try:
    SGD(learning_rate=-0.1)
  except BeartypeCallHintParamViolation:
    print("✓ learning_rate negatif ditolak")
  else:
    raise AssertionError("learning_rate negatif seharusnya ditolak")


def test_parameter_validation() -> None:
  optimizer = SGD(learning_rate=0.1)

  try:
    optimizer(
      weights="invalid",
      bias=0.1,
      gradient_weights=[0.1, 0.2],
      gradient_bias=0.1,
    )
  except BeartypeCallHintParamViolation:
    print("✓ weights salah tipe ditolak")
  else:
    raise AssertionError("weights salah tipe seharusnya ditolak")


def run_tests() -> None:
  test_sgd_update()
  test_learning_rate_validation()
  test_parameter_validation()

  print("\nSemua test berhasil.")


if __name__ == "__main__":
  run_tests()