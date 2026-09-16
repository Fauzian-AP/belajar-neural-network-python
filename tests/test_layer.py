from beartype.roar import BeartypeCallHintParamViolation

from src.core import (
  LeakyReLU,
  HeNormal,
  SGD,
  Layer,
)


# HELPER

def create_layer() -> Layer:
  return Layer(
    input_size=3,
    neuron_count=2,
    initializer=HeNormal(),
    activation=LeakyReLU(alpha=0.01),
    optimizer=SGD(learning_rate=0.1),
  )


# CONSTRUCTOR

def test_layer_constructor() -> None:
  layer = create_layer()

  assert layer.input_size == 3
  assert layer.neuron_count == 2
  assert len(layer.neurons) == 2

  print("✓ Constructor berhasil")
  print(f"  Input Size   : {layer.input_size}")
  print(f"  Neuron Count : {layer.neuron_count}")


# FORWARD

def test_layer_forward() -> None:
  layer = create_layer()

  outputs = layer.forward([1.0, 2.0, 3.0])

  assert len(outputs) == 2

  assert all(
    isinstance(output, float)
    for output in outputs
  )

  assert all(
    neuron.cache is not None
    for neuron in layer.neurons
  )

  print("✓ Forward berhasil")
  print(f"  Outputs : {outputs}")


# BACKWARD

def test_layer_backward() -> None:
  layer = create_layer()

  # Forward hrs dilakukan sebelum Backward
  layer.forward([1.0, 2.0, 3.0])

  gradient_input = layer.backward([1.0, 2.0])

  assert len(gradient_input) == 3

  assert all(
    isinstance(gradient, float)
    for gradient in gradient_input
  )

  # Setiap Neuron hrs memiliki Gradient
  for neuron in layer.neurons:
    assert len(neuron.gradient_weights) == 3

  print("✓ Backward berhasil")
  print(f"  Gradient Input : {gradient_input}")

  print(
    "  Gradient Weights :",
    [
      neuron.gradient_weights
      for neuron in layer.neurons
    ],
  )

  print(
    "  Gradient Bias :",
    [
      neuron.gradient_bias
      for neuron in layer.neurons
    ],
  )


# RESET GRADIENT

def test_layer_reset_gradient() -> None:
  layer = create_layer()

  layer.forward([1.0, 2.0, 3.0])

  layer.backward([1.0, 2.0])

  layer.reset_gradient()

  for neuron in layer.neurons:
    assert neuron.gradient_weights == [0.0, 0.0, 0.0]
    assert neuron.gradient_bias == 0.0

  print("✓ Reset Gradient berhasil")


# AVERAGE GRADIENT

def test_layer_average_gradient() -> None:
  layer = create_layer()

  layer.forward([1.0, 2.0, 3.0])

  layer.backward([1.0, 2.0])

  original_gradients = [
    neuron.gradient_weights.copy()
    for neuron in layer.neurons
  ]

  original_biases = [
    neuron.gradient_bias
    for neuron in layer.neurons
  ]

  layer.average_gradient(2)

  for neuron, original_weights, original_bias in zip(layer.neurons, original_gradients, original_biases):
    assert all(
      actual == expected / 2
      for actual, expected in zip(neuron.gradient_weights, original_weights)
    )

    assert neuron.gradient_bias == original_bias / 2

  print("✓ Average Gradient berhasil")


# STEP

def test_layer_step() -> None:
  layer = create_layer()

  # Forward agar Cache tersedia
  layer.forward([1.0, 2.0, 3.0])

  # Berikan Gradient secara manual agar update Weight & Bias pasti terjadi.
  for neuron in layer.neurons:
    neuron.gradient_weights = [1.0, 2.0, 3.0]

    neuron.gradient_bias = 1.0

  original_weights = [
    neuron.weights.copy()
    for neuron in layer.neurons
  ]

  original_biases = [
    neuron.bias
    for neuron in layer.neurons
  ]

  layer.step()

  for neuron, weights, bias in zip(layer.neurons, original_weights, original_biases):
    expected_weights = [
      weight - (0.1 * gradient)
      
      for weight, gradient in zip(weights, [1.0, 2.0, 3.0])
    ]

    expected_bias = bias - (0.1 * 1.0)

    assert neuron.weights == expected_weights
    assert neuron.bias == expected_bias

  print("✓ Optimizer Step berhasil")


# VALIDATION

def test_backward_before_forward() -> None:
  layer = create_layer()

  try:
    layer.backward([1.0, 2.0])
  except ValueError:
    print("✓ Backward sebelum Forward ditolak")
  else:
    raise AssertionError("Backward sebelum Forward seharusnya ditolak")


def test_input_size_validation() -> None:
  layer = create_layer()

  try:
    layer.forward([1.0, 2.0])
  except ValueError:
    print("✓ Input dengan panjang salah ditolak")
  else:
    raise AssertionError("Input dengan panjang salah seharusnya ditolak")


def test_empty_value_input_validation() -> None:
  layer = create_layer()

  try:
    layer.forward([])
  except BeartypeCallHintParamViolation as e:
    print("✓ Input benar tetapi gk boleh kosong!")
    print(f"Pesan Error: {e}")
  else:
    raise AssertionError("Input benar tetapi kosong seharusnya ditolak")


def test_gradient_output_size_validation() -> None:
  layer = create_layer()

  layer.forward([1.0, 2.0, 3.0])

  try:
    layer.backward([1.0])
  except ValueError:
    print("✓ Gradient Output dengan panjang salah ditolak")
  else:
    raise AssertionError("Gradient Output dengan panjang salah seharusnya ditolak")


def test_empty_value_gradient_input_validation() -> None:
  layer = create_layer()

  layer.forward([1.0, 2.0, 3.0])

  try:
    layer.backward((1.0,))
  except BeartypeCallHintParamViolation as e:
    print("✓ Gradient Input benar tetapi gk boleh kosong!")
    print(f"Pesan Error: {e}")
  except ValueError as e:
    print(f"Pesan Error: {e}")
  else:
    raise AssertionError("Gradient Input benar tetapi kosong seharusnya ditolak")


def test_input_beartype_validation() -> None:
  layer = create_layer()

  try:
    layer.forward(["a", "b", "c"],)
  except BeartypeCallHintParamViolation:
    print("✓ Input salah tipe ditolak")
  else:
    raise AssertionError("Input salah tipe seharusnya ditolak")


def test_gradient_output_beartype_validation() -> None:
  layer = create_layer()

  layer.forward([1.0, 2.0, 3.0])

  try:
    layer.backward(["a", "b"])
  except BeartypeCallHintParamViolation:
    print("✓ Gradient Output salah tipe ditolak")
  else:
    raise AssertionError("Gradient Output salah tipe seharusnya ditolak")


def test_input_size_positive_validation() -> None:
  try:
    Layer(
      input_size=0,
      neuron_count=2,
      initializer=HeNormal(),
      activation=LeakyReLU(alpha=0.01),
      optimizer=SGD(learning_rate=0.1),
    )
  except BeartypeCallHintParamViolation:
    print("✓ input_size=0 ditolak")
  else:
    raise AssertionError("input_size=0 seharusnya ditolak")


def test_neuron_count_positive_validation() -> None:
  try:
    Layer(
      input_size=3,
      neuron_count=0,
      initializer=HeNormal(),
      activation=LeakyReLU(alpha=0.01),
      optimizer=SGD(learning_rate=0.1),
    )
  except BeartypeCallHintParamViolation:
    print("✓ neuron_count=0 ditolak")
  else:
    raise AssertionError("neuron_count=0 seharusnya ditolak")


def test_batch_size_positive_validation() -> None:
  layer = create_layer()

  try:
    layer.average_gradient(0)
  except BeartypeCallHintParamViolation:
    print("✓ batch_size=0 ditolak")
  else:
    raise AssertionError("batch_size=0 seharusnya ditolak")


# MAIN

def run_tests() -> None:
  test_layer_constructor()
  test_layer_forward()
  test_layer_backward()
  test_layer_reset_gradient()
  test_layer_average_gradient()
  test_layer_step()

  test_backward_before_forward()
  test_input_size_validation()
  test_empty_value_input_validation()
  test_gradient_output_size_validation()
  test_empty_value_gradient_input_validation()
  test_input_beartype_validation()
  test_gradient_output_beartype_validation()
  test_input_size_positive_validation()
  test_neuron_count_positive_validation()
  test_batch_size_positive_validation()

  print("\nSemua test Layer berhasil.")


if __name__ == "__main__":
  run_tests()