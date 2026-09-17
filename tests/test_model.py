from src.core import (
  Model,
  HeNormal,
  LeakyReLU,
  SGD,
)


# HELPER

def create_model() -> Model:
  return Model(
    initializer=HeNormal(),
    optimizer=SGD(learning_rate=0.01),
    activation=LeakyReLU(alpha=0.01),
  )


# CONSTRUCTOR

def test_constructor() -> None:
  model = create_model()

  assert len(model.layers) == 3

  assert model.layers[0].input_size == 3
  assert model.layers[0].neuron_count == 4

  assert model.layers[1].input_size == 4
  assert model.layers[1].neuron_count == 4

  assert model.layers[2].input_size == 4
  assert model.layers[2].neuron_count == 2

  print("✓ Constructor berhasil")
  print(f"  Layers       : {len(model.layers)}")
  print("  Architecture : 3 → 4 → 4 → 2\n")


# ARCHITECTURE VALIDATION

def test_architecture_too_short() -> None:
  try:
    Model(
      initializer=HeNormal(),
      optimizer=SGD(learning_rate=0.01),
      activation=LeakyReLU(alpha=0.01),
      architecture=(3,),
    )
  except Exception as error:
    print(f"Pesan Error: {error}")
  else:
    raise AssertionError("Architecture terlalu pendek hrs ditolak.")
  finally:
    print()


def test_architecture_wrong_input() -> None:
  try:
    Model(
      initializer=HeNormal(),
      optimizer=SGD(learning_rate=0.01),
      activation=LeakyReLU(alpha=0.01),
      architecture=(2, 4, 2),
    )
  except Exception as error:
    print(f"Pesan Error: {error}")
  else:
    raise AssertionError("Input architecture salah hrs ditolak.")
  finally:
    print()


def test_architecture_wrong_output() -> None:
  try:
    Model(
      initializer=HeNormal(),
      optimizer=SGD(learning_rate=0.01),
      activation=LeakyReLU(alpha=0.01),
      architecture=(3, 4, 1),
    )
  except Exception as error:
    print(f"Pesan Error: {error}")
  else:
    raise AssertionError("Output architecture salah hrs ditolak.")
  finally:
    print()


# FORWARD

def test_forward() -> None:
  model = create_model()

  inputs = [1.0, 2.0, 3.0]

  outputs = model.forward(inputs)

  assert len(outputs) == 2

  assert all(
    isinstance(output, float)
    for output in outputs
  )

  print("✓ Forward berhasil")
  print(f"  Inputs  : {inputs}")
  print(f"  Outputs : {outputs}\n")


# BACKWARD

def test_backward() -> None:
  model = create_model()

  inputs = [1.0, 2.0, 3.0]

  # Forward hrs dilakukan terlebih dahulu
  model.forward(inputs)

  gradient_outputs = [1.0, 2.0]

  gradient_input = model.backward(gradient_outputs)

  assert len(gradient_input) == 3

  assert all(
    isinstance(gradient, float)
    for gradient in gradient_input
  )

  print("✓ Backward berhasil")
  print(f"  Gradient Output : {gradient_outputs}")
  print(f"  Gradient Input  : {gradient_input}\n")


# RESET GRADIENT

def test_reset_gradient() -> None:
  model = create_model()

  model.forward([1.0, 2.0, 3.0])

  model.backward([1.0, 2.0])

  model.reset_gradient()

  for layer in model.layers:
    for neuron in layer.neurons:
      assert neuron.gradient_weights == [0.0] * neuron.input_size
      assert neuron.gradient_bias == 0.0

  print("✓ Reset Gradient berhasil\n")


# AVERAGE GRADIENT

def test_average_gradient() -> None:
  model = create_model()

  model.forward([1.0, 2.0, 3.0])

  model.backward([1.0, 2.0],)

  model.average_gradient(batch_size=2)

  for layer in model.layers:
    for neuron in layer.neurons:
      assert all(
        isinstance(gradient, float)
        for gradient in neuron.gradient_weights
      )

      assert isinstance(neuron.gradient_bias, float)

  print("✓ Average Gradient berhasil\n")


# STEP

def test_step() -> None:
  model = create_model()

  # Berikan Gradient manual agar perubahan Weight & Bias dapat diuji
  for layer in model.layers:
    for neuron in layer.neurons:
      neuron.gradient_weights = [1.0] * neuron.input_size
      neuron.gradient_bias = 1.0

  # Simpan State sebelum Step
  old_weights = [
    [
      neuron.weights.copy()
      for neuron in layer.neurons
    ]
    for layer in model.layers
  ]

  old_biases = [
    [
      neuron.bias
      for neuron in layer.neurons
    ]
    for layer in model.layers
  ]

  model.step()

  learning_rate = 0.01

  for lyr_idx, layer in enumerate(model.layers):
    for n_idx, neuron in enumerate(layer.neurons):
      # Periksa Weight
      for old_weight, new_weight in zip(old_weights[lyr_idx][n_idx], neuron.weights):
        assert new_weight == old_weight - learning_rate

      # Periksa Bias
      assert neuron.bias == old_biases[lyr_idx][n_idx] - learning_rate

  print("✓ Optimizer Step berhasil\n")


# FORWARD VALIDATION

def test_forward_wrong_input_length() -> None:
  model = create_model()

  try:
    model.forward([1.0, 2.0])
  except Exception as error:
    print(f"Pesan Error: {error}")
  else:
    raise AssertionError("Input dengan panjang salah hrs ditolak.")
  finally:
    print()


def test_forward_wrong_input_type() -> None:
  model = create_model()

  try:
    model.forward(["satu", "dua", "tiga"])
  except Exception as error:
    print(f"Pesan Error: {error}")
  else:
    raise AssertionError("Input salah tipe hrs ditolak.")
  finally:
    print()


# BACKWARD VALIDATION

def test_backward_before_forward() -> None:
  model = create_model()

  try:
    model.backward([1.0, 2.0])
  except Exception as error:
    print(f"Pesan Error: {error}")
  else:
    raise AssertionError("Backward sebelum Forward hrs ditolak.")
  finally:
    print()


def test_backward_wrong_gradient_length() -> None:
  model = create_model()

  model.forward([1.0, 2.0, 3.0])

  try:
    model.backward([1.0])
  except Exception as error:
    print(f"Pesan Error: {error}")
  else:
    raise AssertionError("Gradient Output panjang salah hrs ditolak.")
  finally:
    print()


def test_empty_value_gradient_input() -> None:
  model = create_model()

  model.forward([1.0, 2.0, 3.0])

  try:
    model.backward([])
  except Exception as error:
    print(f"Pesan Error: {error}")
  else:
    raise AssertionError("Gradient Input benar tetapi kosong seharusnya ditolak")
  finally:
    print()


def test_backward_wrong_gradient_type() -> None:
  model = create_model()

  model.forward([1.0, 2.0, 3.0])

  try:
    model.backward(["satu", "dua"])
  except Exception as error:
    print(f"Pesan Error: {error}")
  else:
    raise AssertionError("Gradient Output salah tipe hrs ditolak.")
  finally:
    print()


# AVERAGE GRADIENT VALIDATION

def test_average_gradient_zero() -> None:
  model = create_model()

  try:
    model.average_gradient(batch_size=0)
  except Exception as error:
    print(f"Pesan Error: {error}")
  else:
    raise AssertionError("batch_size=0 hrs ditolak.")
  finally:
    print()


# MAIN

if __name__ == "__main__":

  test_constructor()

  test_architecture_too_short()
  test_architecture_wrong_input()
  test_architecture_wrong_output()

  test_forward()
  test_backward()

  test_reset_gradient()
  test_average_gradient()
  test_step()

  test_forward_wrong_input_length()
  test_forward_wrong_input_type()

  test_backward_before_forward()
  test_backward_wrong_gradient_length()
  test_empty_value_gradient_input()
  test_backward_wrong_gradient_type()

  test_average_gradient_zero()

  print("Semua test Model berhasil.")