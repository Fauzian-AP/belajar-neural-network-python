from beartype.roar import BeartypeCallHintParamViolation

from src.core import (
  LeakyReLU,
  HeNormal,
  Neuron,
)


# HELPER

def create_neuron() -> Neuron:
  return Neuron(
    input_size=3,
    initializer=HeNormal(),
    activation=LeakyReLU(alpha=0.01),
  )


# FORWARD

def test_neuron_forward() -> None:
  neuron = create_neuron()

  output = neuron.forward([1.0, 2.0, 3.0])

  assert isinstance(output, float)
  assert neuron.cache is not None

  print("✓ Forward berhasil")
  print(f"  Output : {output}")
  print(f"  Cache  : {neuron.cache}")


# BACKWARD

def test_neuron_backward() -> None:
  neuron = create_neuron()

  # Forward hrs dilakukan sebelum Backward
  neuron.forward([1.0, 2.0, 3.0])

  # Backward dilakukan sesudah Forward
  gradient_input = neuron.backward(1.0)

  assert len(gradient_input) == 3

  assert all(
    isinstance(gradient, float)
    for gradient in gradient_input
  )

  assert len(neuron.gradient_weights) == 3
  assert isinstance(neuron.gradient_bias, float)

  print("✓ Backward berhasil")
  print(f"  Gradient Input   : {gradient_input}")
  print(f"  Gradient Weights : {neuron.gradient_weights}")
  print(f"  Gradient Bias    : {neuron.gradient_bias}")


# RESET GRADIENT

def test_neuron_reset_gradient() -> None:
  neuron = create_neuron()

  neuron.forward([1.0, 2.0, 3.0])

  neuron.backward(1.0)

  neuron.reset_gradient()

  assert neuron.gradient_weights == [0.0, 0.0, 0.0]
  assert neuron.gradient_bias == 0.0

  print("✓ Reset Gradient berhasil")


# AVERAGE GRADIENT

def test_neuron_average_gradient() -> None:
  neuron = create_neuron()

  neuron.forward([1.0, 2.0, 3.0])

  neuron.backward(1.0)

  original_weights = neuron.gradient_weights.copy()
  original_bias = neuron.gradient_bias

  neuron.average_gradient(2)

  assert all(
    actual == expected / 2
    for actual, expected in zip(neuron.gradient_weights, original_weights)
  )

  assert neuron.gradient_bias == original_bias / 2

  print("✓ Average Gradient berhasil")


# VALIDATION

def test_backward_before_forward() -> None:
  neuron = create_neuron()

  try:
    neuron.backward(1.0)
  except ValueError:
    print("✓ Backward sebelum Forward ditolak")
  else:
    raise AssertionError("Backward sebelum Forward seharusnya ditolak")


def test_input_size_validation() -> None:
  neuron = create_neuron()

  try:
    neuron.forward([1.0, 2.0])
  except ValueError:
    print("✓ Input dengan panjang salah ditolak")
  else:
    raise AssertionError("Input dengan panjang salah seharusnya ditolak")


def test_beartype_validation() -> None:
  neuron = create_neuron()

  try:
    neuron.forward(["a", "b", "c"])
  except BeartypeCallHintParamViolation:
    print("✓ Input salah tipe ditolak")
  else:
    raise AssertionError("Input salah tipe seharusnya ditolak")


def test_empty_value_input_forward() -> None:
  neuron = create_neuron()

  try:
    neuron.forward([])
  except BeartypeCallHintParamViolation:
    print("✓ Input benar tetapi gk boleh kosong!")
  else:
    raise AssertionError("Input benar tetapi kosong seharusnya ditolak")


def test_input_size_positive_validation() -> None:
  try:
    Neuron(
      input_size=0,
      initializer=HeNormal(),
      activation=LeakyReLU(alpha=0.01),
    )
  except BeartypeCallHintParamViolation:
    print("✓ input_size=0 ditolak")
  else:
    raise AssertionError("input_size=0 seharusnya ditolak")


# MAIN

def run_tests() -> None:
  test_neuron_forward()
  test_neuron_backward()
  test_neuron_reset_gradient()
  test_neuron_average_gradient()
  test_backward_before_forward()
  test_input_size_validation()
  test_beartype_validation()
  test_empty_value_input_forward()
  test_input_size_positive_validation()

  print("\nSemua test Neuron berhasil.")


if __name__ == "__main__":
  run_tests()