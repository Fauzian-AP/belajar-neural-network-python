# Bagian untuk melakukan Diagnostic pada Neural Network

import copy
import random

from batch import Batch
from neural_network.core.model import Model
from neural_network.training.trainer import Trainer
from neural_network.evaluating.loss_functions import MSE
from neural_network.evaluating.metric_evaluation import calculate_metrics
from neural_network.utils.custom_types import (
  Dataset,
  ActivationType,
)

# CONFIGURATION

SEEDS = [1, 2, 3, 4, 5, 42]

BATCH_SIZE = 5

# HELPER

def create_diagnostic_model(learning_rate: float, seed: int) -> Model:
  random.seed(seed)

  return Model(learning_rate=learning_rate)

def create_fixed_batch(dataset: Dataset, seed: int) -> Dataset:
  batch = Batch(seed=seed)

  batches = batch.create_batches(dataset, BATCH_SIZE)

  return batches[0]

# RUN DIAGNOSTICS

def run_diagnostics(normalized_datasets, learning_rate: float, seed: int = 42) -> None:
  training_data = normalized_datasets["training"]
  validation_data = normalized_datasets["validation"]

  # DIAGNOSTIC 1

  print("=== DIAGNOSTIC — 1 BATCH ===")

  random.seed(seed)

  model = Model(learning_rate=learning_rate)

  initial_model = copy.deepcopy(model.layers)

  diagnostic_batch = create_fixed_batch(training_data, seed)

  model.reset_gradient()

  for inputs, targets in diagnostic_batch:
    predictions = model.forward(inputs)

    gradient_outputs = MSE.gradient(targets, predictions)

    model.backward(gradient_outputs)

  gradient_weights = (
    model.layers[0]
    .neurons[0]
    .gradient_weights
    .copy()
  )

  gradient_bias = (
    model.layers[0]
    .neurons[0]
    .gradient_bias
  )

  print(f"Initial Weights : {initial_model[0].neurons[0].weights}")
  print(f"Gradient Weights: {gradient_weights}")
  print(f"Gradient Bias   : {gradient_bias}")

  model.average_gradient(len(diagnostic_batch))

  model.step()

  print(f"Updated Weights : {model.layers[0].neurons[0].weights}")
  print()

  # DIAGNOSTIC 2

  print("=== DIAGNOSTIC 2 — GRADIENT PER LAYER ===")

  random.seed(seed)

  model = Model(learning_rate=learning_rate)

  diagnostic_batch = create_fixed_batch(training_data, seed)

  model.reset_gradient()

  for inputs, targets in diagnostic_batch:
    predictions = model.forward(inputs)

    gradient_outputs = MSE.gradient(targets, predictions)

    model.backward(gradient_outputs)

  model.average_gradient(len(diagnostic_batch))

  for layer_index, layer in enumerate(model.layers, start=1):
    layer_max_gradient = layer_max_update = 0.0

    for neuron in layer.neurons:
      for gradient in neuron.gradient_weights:
        update = -learning_rate * gradient

        layer_max_gradient = max(layer_max_gradient, abs(gradient))
        layer_max_update = max(layer_max_update, abs(update))

      update_bias = -learning_rate * neuron.gradient_bias

      layer_max_gradient = max(layer_max_gradient, abs(neuron.gradient_bias))
      layer_max_update = max(layer_max_update, abs(update_bias))

    print(f"Layer {layer_index}")
    print(f"  Maximum Gradient : {layer_max_gradient:.6f}")
    print(f"  Maximum Update   : {layer_max_update:.6f}")

  print()

  # DIAGNOSTIC 3

  print("=== DIAGNOSTIC 3 — EFFECT OF FIRST UPDATE ===")

  random.seed(seed)

  model = Model(learning_rate=learning_rate)

  diagnostic_batch = create_fixed_batch(training_data, seed)

  def diagnostic_mse(dataset: Dataset) -> float:
    total_mse = 0.0

    for inputs, targets in dataset:
      predictions = model.forward(inputs)

      total_mse += MSE.calculate(targets, predictions)

    return total_mse / len(dataset)

  before_training_mse = diagnostic_mse(diagnostic_batch)
  before_validation_mse = diagnostic_mse(validation_data)

  print(f"Training MSE BEFORE   : {before_training_mse}")
  print(f"Validation MSE BEFORE : {before_validation_mse}")

  model.reset_gradient()

  for inputs, targets in diagnostic_batch:
    predictions = model.forward(inputs)

    gradient_outputs = MSE.gradient(targets, predictions)

    model.backward(gradient_outputs)

  model.average_gradient(len(diagnostic_batch))

  model.step()

  after_training_mse = diagnostic_mse(diagnostic_batch)
  after_validation_mse = diagnostic_mse(validation_data)

  print(f"Training MSE AFTER    : {after_training_mse}")
  print(f"Validation MSE AFTER  : {after_validation_mse}")
  print(f"Training MSE CHANGE   : {after_training_mse - before_training_mse}")
  print(f"Validation MSE CHANGE : {after_validation_mse - before_validation_mse}")
  print()

  # DIAGNOSTIC 4

  print("=== DIAGNOSTIC 4 — WEIGHT vs GRADIENT ===")

  random.seed(seed)

  model = Model(learning_rate=learning_rate)

  diagnostic_batch = create_fixed_batch(training_data, seed)

  model.reset_gradient()

  for inputs, targets in diagnostic_batch:
    predictions = model.forward(inputs)

    gradient_outputs = MSE.gradient(targets, predictions)

    model.backward(gradient_outputs)

  model.average_gradient(len(diagnostic_batch))

  for layer_index, layer in enumerate(model.layers, start=1):
    max_weight = max_gradient = 0.0

    for neuron in layer.neurons:
      for weight in neuron.weights:
        max_weight = max(max_weight, abs(weight))

      for gradient in neuron.gradient_weights:
        max_gradient = max(max_gradient, abs(gradient))

      max_gradient = max(max_gradient, abs(neuron.gradient_bias),)

    print(f"Layer {layer_index}")
    print(f"  Maximum |Weight|   : {max_weight:.6f}")
    print(f"  Maximum |Gradient| : {max_gradient:.6f}")

    if max_weight != 0:
      print(f"  Gradient / Weight  : {max_gradient / max_weight:.6f}")

  print()

  # DIAGNOSTIC 5 — GRADIENT FLOW

  print("=== DIAGNOSTIC 5 — GRADIENT FLOW ===")

  random.seed(seed)

  model = Model(learning_rate=learning_rate)

  diagnostic_batch = create_fixed_batch(training_data, seed)

  # Analisis Gradient Flow pada setiap data
  for sample_index, (inputs, targets) in enumerate(diagnostic_batch, start=1):
    # Pastikan Gradient kosong
    model.reset_gradient()

    # Forward
    predictions = model.forward(inputs)

    # Output GENERATE
    gradient = MSE.gradient(targets, predictions)

    max_output_gradient = max(
      abs(value)
      for value in gradient
    )

    print(f"Sample {sample_index}")
    print(f"  Gradient Output : {max_output_gradient:.6f}")

    # Backward Flow
    for layer_index in range(len(model.layers) - 1, -1, -1):
      layer = model.layers[layer_index]

      gradient = layer.backward(gradient)

      max_gradient = max(
        abs(value)
        for value in gradient
      )

      print(f"  After L{layer_index + 1}       : {max_gradient:.6f}")

    print()

  # DIAGNOSTIC 6 — SEED SENSITIVITY

  print("=== DIAGNOSTIC 6 — SEED SENSITIVITY ===")

  for diagnostic_seed in SEEDS:
    # Initial Model
    model = create_diagnostic_model(learning_rate, diagnostic_seed)

    # -------------------------------------------------------------
    # FIXED BATCH
    # -------------------------------------------------------------

    diagnostic_batch = create_fixed_batch(
      training_data,
      diagnostic_seed,
    )

    # -------------------------------------------------------------
    # MSE BEFORE
    # -------------------------------------------------------------

    mse_before = 0.0

    for inputs, targets in training_data:

      predictions = model.forward(
        inputs,
      )

      mse_before += MSE.calculate(
        targets,
        predictions,
      )

    mse_before /= len(training_data)

    # -------------------------------------------------------------
    # FORWARD + BACKWARD
    # -------------------------------------------------------------

    model.reset_gradient()

    for inputs, targets in diagnostic_batch:

      predictions = model.forward(
        inputs,
      )

      gradient_outputs = MSE.gradient(
        targets,
        predictions,
      )

      model.backward(
        gradient_outputs,
      )

    # -------------------------------------------------------------
    # MAX GRADIENT
    # -------------------------------------------------------------

    max_gradient = 0.0

    for layer in model.layers:

      for neuron in layer.neurons:

        for gradient in neuron.gradient_weights:

          max_gradient = max(
            max_gradient,
            abs(gradient),
          )

        max_gradient = max(
          max_gradient,
          abs(neuron.gradient_bias),
        )

    # -------------------------------------------------------------
    # AVERAGE GRADIENT
    # -------------------------------------------------------------

    model.average_gradient(
      len(diagnostic_batch),
    )

    # -------------------------------------------------------------
    # MAX UPDATE
    # -------------------------------------------------------------

    max_update = 0.0

    for layer in model.layers:

      for neuron in layer.neurons:

        for gradient in neuron.gradient_weights:

          update = (
            learning_rate * gradient
          )

          max_update = max(
            max_update,
            abs(update),
          )

        update_bias = (
          learning_rate
          * neuron.gradient_bias
        )

        max_update = max(
          max_update,
          abs(update_bias),
        )

    # -------------------------------------------------------------
    # UPDATE
    # -------------------------------------------------------------

    model.step()

    # -------------------------------------------------------------
    # MSE AFTER
    # -------------------------------------------------------------

    mse_after = 0.0

    for inputs, targets in training_data:

      predictions = model.forward(
        inputs,
      )

      mse_after += MSE.calculate(
        targets,
        predictions,
      )

    mse_after /= len(training_data)

    # -------------------------------------------------------------
    # MSE CHANGE
    # -------------------------------------------------------------

    mse_change = (
      mse_after - mse_before
    )

    # -------------------------------------------------------------
    # RESULT
    # -------------------------------------------------------------

    print(
      f"Seed {diagnostic_seed:<2} → "
      f"Max Gradient: {max_gradient:.6f} → "
      f"Max Update: {max_update:.6f} → "
      f"First-step MSE Change: {mse_change:.6f}"
    )

  print()


  # ===============================================================
  # DIAGNOSTIC 7 — INITIALIZATION SENSITIVITY
  # ===============================================================

  print("=== DIAGNOSTIC 7 — INITIALIZATION SENSITIVITY ===")

  # Gunakan satu Batch yang sama untuk semua Seed.
  fixed_batch = create_fixed_batch(
    training_data,
    seed,
  )

  for diagnostic_seed in SEEDS:

    # -------------------------------------------------------------
    # INITIAL MODEL
    # -------------------------------------------------------------

    model = create_diagnostic_model(
      learning_rate,
      diagnostic_seed,
    )

    # -------------------------------------------------------------
    # MSE BEFORE
    # -------------------------------------------------------------

    mse_before = 0.0

    for inputs, targets in training_data:

      predictions = model.forward(
        inputs,
      )

      mse_before += MSE.calculate(
        targets,
        predictions,
      )

    mse_before /= len(training_data)

    # -------------------------------------------------------------
    # FORWARD + BACKWARD
    # -------------------------------------------------------------

    model.reset_gradient()

    for inputs, targets in fixed_batch:

      predictions = model.forward(
        inputs,
      )

      gradient_outputs = MSE.gradient(
        targets,
        predictions,
      )

      model.backward(
        gradient_outputs,
      )

    # -------------------------------------------------------------
    # MAX GRADIENT
    # -------------------------------------------------------------

    max_gradient = 0.0

    for layer in model.layers:

      for neuron in layer.neurons:

        for gradient in neuron.gradient_weights:

          max_gradient = max(
            max_gradient,
            abs(gradient),
          )

        max_gradient = max(
          max_gradient,
          abs(neuron.gradient_bias),
        )

    # -------------------------------------------------------------
    # AVERAGE GRADIENT
    # -------------------------------------------------------------

    model.average_gradient(
      len(fixed_batch),
    )

    # -------------------------------------------------------------
    # MAX UPDATE
    # -------------------------------------------------------------

    max_update = 0.0

    for layer in model.layers:

      for neuron in layer.neurons:

        for gradient in neuron.gradient_weights:

          update = (
            learning_rate * gradient
          )

          max_update = max(
            max_update,
            abs(update),
          )

        update_bias = (
          learning_rate
          * neuron.gradient_bias
        )

        max_update = max(
          max_update,
          abs(update_bias),
        )

    # -------------------------------------------------------------
    # UPDATE
    # -------------------------------------------------------------

    model.step()

    # -------------------------------------------------------------
    # MSE AFTER
    # -------------------------------------------------------------

    mse_after = 0.0

    for inputs, targets in training_data:

      predictions = model.forward(
        inputs,
      )

      mse_after += MSE.calculate(
        targets,
        predictions,
      )

    mse_after /= len(training_data)

    # -------------------------------------------------------------
    # MSE CHANGE
    # -------------------------------------------------------------

    mse_change = (
      mse_after - mse_before
    )

    # -------------------------------------------------------------
    # RESULT
    # -------------------------------------------------------------

    print(
      f"Seed {diagnostic_seed:<2} → "
      f"Initial MSE: {mse_before:.6f} → "
      f"Max Gradient: {max_gradient:.6f} → "
      f"Max Update: {max_update:.6f} → "
      f"MSE Change: {mse_change:.6f}"
    )

  print()


# =================================================================
  # ===============================================================
  # DIAGNOSTIC 8 — GRADIENT FLOW PER SEED
  # ===============================================================

  print("=== DIAGNOSTIC 8 — GRADIENT FLOW PER SEED ===")

  # Gunakan satu data yang sama untuk semua Seed.
  fixed_batch = create_fixed_batch(
    training_data,
    seed,
  )

  inputs, targets = fixed_batch[0]

  for diagnostic_seed in SEEDS:

    # -------------------------------------------------------------
    # INITIAL MODEL
    # -------------------------------------------------------------

    model = create_diagnostic_model(
      learning_rate,
      diagnostic_seed,
    )

    # -------------------------------------------------------------
    # FORWARD
    # -------------------------------------------------------------

    predictions = model.forward(
      inputs,
    )

    # -------------------------------------------------------------
    # OUTPUT GRADIENT
    # -------------------------------------------------------------

    gradient = MSE.gradient(
      targets,
      predictions,
    )

    max_output_gradient = max(
      abs(value)
      for value in gradient
    )

    print(
      f"Seed {diagnostic_seed:<2} | "
      f"Output: {max_output_gradient:.6f}"
    )

    # -------------------------------------------------------------
    # BACKWARD FLOW
    # -------------------------------------------------------------

    for layer_index in range(
      len(model.layers) - 1,
      -1,
      -1,
    ):

      gradient = model.layers[
        layer_index
      ].backward(
        gradient,
      )

      max_gradient = max(
        abs(value)
        for value in gradient
      )

      print(
        f"          | "
        f"After L{layer_index + 1}: "
        f"{max_gradient:.6f}"
      )

    print()

  # ===============================================================
  # DIAGNOSTIC 9 — WEIGHT vs GRADIENT PER SEED
  # ===============================================================

  print("=== DIAGNOSTIC 9 — WEIGHT vs GRADIENT PER SEED ===")

  fixed_batch = create_fixed_batch(
    training_data,
    seed,
  )

  for diagnostic_seed in SEEDS:

    model = create_diagnostic_model(
      learning_rate,
      diagnostic_seed,
    )

    model.reset_gradient()

    for inputs, targets in fixed_batch:

      predictions = model.forward(
        inputs,
      )

      gradient_outputs = MSE.gradient(
        targets,
        predictions,
      )

      model.backward(
        gradient_outputs,
      )

    max_weight = 0.0
    max_gradient = 0.0

    for layer in model.layers:

      for neuron in layer.neurons:

        for weight in neuron.weights:
          max_weight = max(
            max_weight,
            abs(weight),
          )

        for gradient in neuron.gradient_weights:
          max_gradient = max(
            max_gradient,
            abs(gradient),
          )

        max_gradient = max(
          max_gradient,
          abs(neuron.gradient_bias),
        )

    model.average_gradient(
      len(fixed_batch),
    )

    max_update = 0.0

    for layer in model.layers:

      for neuron in layer.neurons:

        for gradient in neuron.gradient_weights:

          update = (
            learning_rate * gradient
          )

          max_update = max(
            max_update,
            abs(update),
          )

        update_bias = (
          learning_rate
          * neuron.gradient_bias
        )

        max_update = max(
          max_update,
          abs(update_bias),
        )

    print(
      f"Seed {diagnostic_seed:<2} → "
      f"Max Weight: {max_weight:.6f} → "
      f"Max Gradient: {max_gradient:.6f} → "
      f"Max Update: {max_update:.6f}"
    )

  print()


  # ===============================================================
  # DIAGNOSTIC 10 — OUTPUT GRADIENT vs PARAMETER GRADIENT
  # ===============================================================

  print("=== DIAGNOSTIC 10 — OUTPUT vs PARAMETER GRADIENT ===")

  fixed_batch = create_fixed_batch(
    training_data,
    seed,
  )

  for diagnostic_seed in SEEDS:

    model = create_diagnostic_model(
      learning_rate,
      diagnostic_seed,
    )

    inputs, targets = fixed_batch[0]

    predictions = model.forward(
      inputs,
    )

    gradient_outputs = MSE.gradient(
      targets,
      predictions,
    )

    max_output_gradient = max(
      abs(value)
      for value in gradient_outputs
    )

    model.reset_gradient()

    model.backward(
      gradient_outputs,
    )

    max_parameter_gradient = 0.0

    for layer in model.layers:

      for neuron in layer.neurons:

        for gradient in neuron.gradient_weights:

          max_parameter_gradient = max(
            max_parameter_gradient,
            abs(gradient),
          )

        max_parameter_gradient = max(
          max_parameter_gradient,
          abs(neuron.gradient_bias),
        )

    print(
      f"Seed {diagnostic_seed:<2} → "
      f"Output Gradient: {max_output_gradient:.6f} → "
      f"Parameter Gradient: "
      f"{max_parameter_gradient:.6f}"
    )

  print()


  # ===============================================================
  # DIAGNOSTIC 11 — MAX ACTIVATION PER LAYER
  # ===============================================================

  print("=== DIAGNOSTIC 11 — MAX ACTIVATION PER LAYER ===")

  model = create_diagnostic_model(
    learning_rate,
    seed,
  )

  max_activations = [
    0.0
    for _ in model.layers
  ]

  initial_mse = 0.0

  for inputs, targets in training_data:

    output = inputs

    for layer_index, layer in enumerate(
      model.layers
    ):

      output = layer.forward(
        output,
      )

      layer_max = max(
        abs(value)
        for value in output
      )

      max_activations[layer_index] = max(
        max_activations[layer_index],
        layer_max,
      )

    predictions = output

    initial_mse += MSE.calculate(
      targets,
      predictions,
    )

  initial_mse /= len(training_data)

  for layer_index, activation in enumerate(
    max_activations
  ):

    print(
      f"L{layer_index + 1} "
      f"Maximum Activation: "
      f"{activation:.6f}"
    )

  print(
    f"Initial MSE: {initial_mse:.6f}"
  )

  print()


  # ===============================================================
  # DIAGNOSTIC 12 — ACTIVATION PATTERN
  # ===============================================================

  print("=== DIAGNOSTIC 12 — ACTIVATION PATTERN ===")

  model = create_diagnostic_model(
    learning_rate,
    seed,
  )

  for inputs, targets in training_data:

    model.forward(
      inputs,
    )

  for layer_index, layer in enumerate(
    model.layers
  ):

    print(
      f"Layer {layer_index + 1}"
    )

    if layer.neurons[0].activation != ActivationType.RELU:

      print(
        "  Activation : LINEAR"
      )

      print()

      continue

    for neuron_index, neuron in enumerate(
      layer.neurons
    ):

      pre_activation = neuron.cache[
        "pre_activation"
      ]

      if pre_activation > 0:
        status = "ACTIVE"
      else:
        status = "INACTIVE"

      print(
        f"  N{neuron_index + 1} : "
        f"{status} "
        f"({pre_activation:.6f})"
      )

    print()

  # ===============================================================
  # DIAGNOSTIC 13 — ACTIVATION RATE
  # ===============================================================

  print("=== DIAGNOSTIC 13 — ACTIVATION RATE ===")

  model = create_diagnostic_model(
    learning_rate,
    seed,
  )

  activation_counts = [
    [0] * layer.neuron_count
    for layer in model.layers
  ]

  sample_count = len(training_data)

  for inputs, targets in training_data:

    model.forward(
      inputs,
    )

    for layer_index, layer in enumerate(
      model.layers
    ):

      if layer.neurons[0].activation != ActivationType.RELU:
        continue

      for neuron_index, neuron in enumerate(
        layer.neurons
      ):

        pre_activation = neuron.cache[
          "pre_activation"
        ]

        if pre_activation > 0:

          activation_counts[
            layer_index
          ][neuron_index] += 1

  for layer_index, layer in enumerate(
    model.layers
  ):

    print(
      f"Layer {layer_index + 1}"
    )

    if layer.neurons[0].activation != ActivationType.RELU:

      print(
        "  Activation : LINEAR"
      )

      print()

      continue

    for neuron_index in range(
      layer.neuron_count
    ):

      active_count = activation_counts[
        layer_index
      ][neuron_index]

      activation_rate = (
        active_count / sample_count
      ) * 100

      print(
        f"  N{neuron_index + 1} : "
        f"{activation_rate:.2f}%"
      )

    print()


  # ===============================================================
  # DIAGNOSTIC 14 — CURRENT MODEL ACTIVATION
  # ===============================================================

  print("=== DIAGNOSTIC 14 — CURRENT MODEL ACTIVATION ===")

  model = create_diagnostic_model(
    learning_rate,
    seed,
  )

  for inputs, targets in training_data:

    model.forward(
      inputs,
    )

  for layer_index, layer in enumerate(
    model.layers
  ):

    print(
      f"Layer {layer_index + 1}"
    )

    if layer.neurons[0].activation != ActivationType.RELU:

      print(
        "  Activation : LINEAR"
      )

      print()

      continue

    for neuron_index, neuron in enumerate(
      layer.neurons
    ):

      pre_activation = neuron.cache[
        "pre_activation"
      ]

      status = (
        "ACTIVE"
        if pre_activation > 0
        else "INACTIVE"
      )

      print(
        f"  N{neuron_index + 1} : "
        f"{status}"
      )

    print()

  # ===============================================================
  # DIAGNOSTIC 15 — ACTIVATION PATTERN DURING TRAINING
  # ===============================================================

  print("=== DIAGNOSTIC 15 — ACTIVATION PATTERN DURING TRAINING ===")

  model = create_diagnostic_model(
    learning_rate,
    seed,
  )

  epochs = 26

  for epoch in range(
    1,
    epochs + 1,
  ):

    batches = create_fixed_batch(
      training_data,
      seed,
    )

    model.reset_gradient()

    for inputs, targets in batches:

      predictions = model.forward(
        inputs,
      )

      gradient_outputs = MSE.gradient(
        targets,
        predictions,
      )

      model.backward(
        gradient_outputs,
      )

    model.average_gradient(
      len(batches),
    )

    model.step()

    if epoch in (
      1,
      5,
      10,
      15,
      20,
      26,
    ):

      print(
        f"Epoch {epoch}"
      )

      for layer_index, layer in enumerate(
        model.layers
      ):

        if layer.neurons[0].activation != ActivationType.RELU:

          print(
            f"  L{layer_index + 1} : LINEAR"
          )

          continue

        active_neurons = []

        for neuron_index, neuron in enumerate(
          layer.neurons
        ):

          active_count = 0

          for inputs, targets in training_data:

            model.forward(
              inputs,
            )

            pre_activation = neuron.cache[
              "pre_activation"
            ]

            if pre_activation > 0:
              active_count += 1

          rate = (
            active_count
            / len(training_data)
          ) * 100

          active_neurons.append(
            f"N{neuron_index + 1}={rate:.0f}%"
          )

        print(
          f"  L{layer_index + 1} : "
          + ", ".join(active_neurons)
        )

      print()


  # ===============================================================
  # DIAGNOSTIC 16 — PARAMETER CHANGE
  # ===============================================================

  print("=== DIAGNOSTIC 16 — PARAMETER CHANGE ===")

  model = create_diagnostic_model(
    learning_rate,
    seed,
  )

  initial_layers = copy.deepcopy(
    model.layers,
  )

  fixed_batch = create_fixed_batch(
    training_data,
    seed,
  )

  model.reset_gradient()

  for inputs, targets in fixed_batch:

    predictions = model.forward(
      inputs,
    )

    gradient_outputs = MSE.gradient(
      targets,
      predictions,
    )

    model.backward(
      gradient_outputs,
    )

  model.average_gradient(
    len(fixed_batch),
  )

  model.step()

  for layer_index, (
    initial_layer,
    updated_layer,
  ) in enumerate(
    zip(
      initial_layers,
      model.layers,
    ),
    start=1,
  ):

    print(
      f"Layer {layer_index}"
    )

    for neuron_index, (
      initial_neuron,
      updated_neuron,
    ) in enumerate(
      zip(
        initial_layer.neurons,
        updated_layer.neurons,
      ),
      start=1,
    ):

      max_weight_change = max(
        abs(
          updated_weight
          - initial_weight
        )
        for initial_weight, updated_weight
        in zip(
          initial_neuron.weights,
          updated_neuron.weights,
        )
      )

      bias_change = abs(
        updated_neuron.bias
        - initial_neuron.bias
      )

      print(
        f"  N{neuron_index} → "
        f"Weight Change: "
        f"{max_weight_change:.6f} → "
        f"Bias Change: "
        f"{bias_change:.6f}"
      )

    print()

  # ===============================================================
  # DIAGNOSTIC 17 — GRADIENT PER PARAMETER
  # ===============================================================

  print("=== DIAGNOSTIC 17 — GRADIENT PER PARAMETER ===")

  model = create_diagnostic_model(
    learning_rate,
    seed,
  )

  fixed_batch = create_fixed_batch(
    training_data,
    seed,
  )

  model.reset_gradient()

  for inputs, targets in fixed_batch:

    predictions = model.forward(
      inputs,
    )

    gradient_outputs = MSE.gradient(
      targets,
      predictions,
    )

    model.backward(
      gradient_outputs,
    )

  print(
    f"Gradient dikumpulkan dari "
    f"{len(fixed_batch)} sample "
    f"(BELUM di-average)"
  )

  for layer_index, layer in enumerate(
    model.layers,
    start=1,
  ):

    print(
      f"Layer {layer_index}"
    )

    for neuron_index, neuron in enumerate(
      layer.neurons,
      start=1,
    ):

      max_weight_gradient = max(
        abs(gradient)
        for gradient
        in neuron.gradient_weights
      )

      print(
        f"  N{neuron_index} → "
        f"Max Weight Gradient: "
        f"{max_weight_gradient:.6f} → "
        f"Bias Gradient: "
        f"{abs(neuron.gradient_bias):.6f}"
      )

    print()

  print()

  # ===============================================================
  # DIAGNOSTIC 18 — DEAD RELU
  # ===============================================================

  print("=== DIAGNOSTIC 18 — DEAD RELU ===")

  model = create_diagnostic_model(
    learning_rate,
    seed,
  )

  epochs = 26

  # Menyimpan jumlah activation setiap neuron
  # selama seluruh proses training.
  activation_counts = [
    [0] * layer.neuron_count
    for layer in model.layers
  ]

  # Menyimpan jumlah pemeriksaan
  # setiap neuron.
  total_checks = [
    [0] * layer.neuron_count
    for layer in model.layers
  ]

  diagnostic_batch = create_fixed_batch(
    training_data,
    seed,
  )

  for epoch in range(
    1,
    epochs + 1,
  ):

    # -------------------------------------------------------------
    # TRAIN SATU EPOCH
    # -------------------------------------------------------------

    model.reset_gradient()

    for inputs, targets in diagnostic_batch:

      predictions = model.forward(
        inputs,
      )

      gradient_outputs = MSE.gradient(
        targets,
        predictions,
      )

      model.backward(
        gradient_outputs,
      )

    model.average_gradient(
      len(diagnostic_batch),
    )

    model.step()

    # -------------------------------------------------------------
    # CEK AKTIVASI
    # -------------------------------------------------------------

    for inputs, targets in training_data:

      model.forward(inputs)

      for layer_index, layer in enumerate(
        model.layers
      ):

        if layer.neurons[0].activation != ActivationType.RELU:
          continue

        for neuron_index, neuron in enumerate(
          layer.neurons
        ):

          pre_activation = neuron.cache[
            "pre_activation"
          ]

          total_checks[
            layer_index
          ][neuron_index] += 1

          if pre_activation > 0:

            activation_counts[
              layer_index
            ][neuron_index] += 1

  # ---------------------------------------------------------------
  # HASIL
  # ---------------------------------------------------------------

  for layer_index, layer in enumerate(
    model.layers
  ):

    print(
      f"Layer {layer_index + 1}"
    )

    if layer.neurons[0].activation != ActivationType.RELU:

      print(
        "  Activation : LINEAR"
      )

      print()

      continue

    for neuron_index in range(
      layer.neuron_count
    ):

      active_count = activation_counts[
        layer_index
      ][neuron_index]

      total_check = total_checks[
        layer_index
      ][neuron_index]

      activation_rate = (
        active_count
        / total_check
      ) * 100

      if active_count == 0:
        status = "DEAD"
      elif active_count == total_check:
        status = "ALWAYS ACTIVE"
      else:
        status = "ACTIVE / INACTIVE"

      print(
        f"  N{neuron_index + 1} : "
        f"{status} | "
        f"Activation: "
        f"{activation_rate:.2f}%"
      )

    print()

  # ===============================================================
  # DIAGNOSTIC 19 — PENYEBAB DEAD RELU
  # ===============================================================

  print("=== DIAGNOSTIC 19 — PENYEBAB DEAD RELU ===")

  model = create_diagnostic_model(
    learning_rate,
    seed,
  )

  diagnostic_batch = create_fixed_batch(
    training_data,
    seed,
  )

  epochs = 26

  for epoch in range(
    1,
    epochs + 1,
  ):

    # -------------------------------------------------------------
    # TRAIN SATU EPOCH
    # -------------------------------------------------------------

    model.reset_gradient()

    for inputs, targets in diagnostic_batch:

      predictions = model.forward(
        inputs,
      )

      gradient_outputs = MSE.gradient(
        targets,
        predictions,
      )

      model.backward(
        gradient_outputs,
      )

    model.average_gradient(
      len(diagnostic_batch),
    )

    model.step()

    # -------------------------------------------------------------
    # ANALISIS PADA EPOCH TERTENTU
    # -------------------------------------------------------------

    if epoch in (1, 5, 10, 15, 20, 26):

      print(f"Epoch {epoch}")

      for layer_index, layer in enumerate(
        model.layers,
        start=1,
      ):

        if layer.neurons[0].activation != ActivationType.RELU:
          print(
            f"  L{layer_index} : LINEAR"
          )
          continue

        print(
          f"  L{layer_index}"
        )

        # Cari min dan max pre-activation
        # setiap neuron pada seluruh training data.
        for neuron_index, neuron in enumerate(
          layer.neurons,
          start=1,
        ):

          minimum = float("inf")
          maximum = float("-inf")

          for inputs, targets in training_data:

            # Forward satu kali untuk mendapatkan
            # cache seluruh layer.
            model.forward(inputs)

            pre_activation = neuron.cache[
              "pre_activation"
            ]

            minimum = min(
              minimum,
              pre_activation,
            )

            maximum = max(
              maximum,
              pre_activation,
            )

          print(
            f"    N{neuron_index} | "
            f"Min: {minimum:.6f} | "
            f"Max: {maximum:.6f}"
          )

      print()

  # ===============================================================
  # DIAGNOSTIC 20 — LEAKY RELU
  # ===============================================================

  print("=== DIAGNOSTIC 20 — LEAKY RELU ===")

  def leaky_relu(
    value: float,
    alpha: float = 0.01,
  ) -> float:

    if value > 0:
      return value

    return alpha * value


  def leaky_relu_gradient(
    value: float,
    alpha: float = 0.01,
  ) -> float:

    if value > 0:
      return 1.0

    return alpha


  # ---------------------------------------------------------------
  # PERBANDINGAN GRADIENT
  # ---------------------------------------------------------------

  print("=== GRADIENT COMPARISON ===")

  test_values = [
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0,
  ]

  for value in test_values:

    relu_gradient = (
      1.0
      if value > 0
      else 0.0
    )

    leaky_gradient = leaky_relu_gradient(
      value,
    )

    print(
      f"x = {value:>5.1f} | "
      f"ReLU Gradient = {relu_gradient:.2f} | "
      f"Leaky ReLU Gradient = {leaky_gradient:.2f}"
    )

  print()

  # ---------------------------------------------------------------
  # NILAI AKTIVASI
  # ---------------------------------------------------------------

  print("=== ACTIVATION COMPARISON ===")

  for value in test_values:

    relu_value = max(
      0.0,
      value,
    )

    leaky_value = leaky_relu(
      value,
    )

    print(
      f"x = {value:>5.1f} | "
      f"ReLU = {relu_value:>6.3f} | "
      f"Leaky ReLU = {leaky_value:>6.3f}"
    )

  print()


def compare_relu_leaky_relu(
  normalized_datasets: dict,
  learning_rate: float,
  seed: int,
) -> None:

  print("=== D21: RELU VS LEAKY RELU ===")

  results = {}

  for activation_name in [
    ActivationType.RELU,
    ActivationType.LEAKY_RELU,
  ]:

    random.seed(seed)

    model = Model(
      learning_rate=learning_rate,
      activation=activation_name,
    )

    batch = Batch(seed=seed)

    trainer = Trainer(
      model=model,
      batch=batch,
    )

    fit_result = trainer.fit(
      training_data=normalized_datasets["training"],
      validating_data=normalized_datasets["validation"],
      epochs=360,
      batch_size=5,
      patience=5,
    )

    testing_metrics = trainer.evaluating(
      normalized_datasets["test"]
    )

    results[activation_name] = {
      "best_epoch": fit_result["best_epoch"],
      "validation_mse": fit_result["best_validating_mse"],
      "testing_mse": testing_metrics["MSE"],
    }

  for activation, result in results.items():

    print(
      f"{activation.value:<12} | "
      f"Best Epoch: {result['best_epoch']:<4} | "
      f"Validation MSE: {result['validation_mse']:.10f} | "
      f"Testing MSE: {result['testing_mse']:.10f}"
    )

def compare_gradient_parameter_change(
  normalized_datasets: dict,
  learning_rate: float,
  seed: int,
) -> None:

  print("=== D21B: GRADIENT / PARAMETER CHANGE ===")

  dataset = normalized_datasets["training"]

  # Gunakan batch yang sama untuk kedua Model.
  diagnostic_batch = create_fixed_batch(
    dataset,
    seed,
  )

  for activation in (
    ActivationType.RELU,
    ActivationType.LEAKY_RELU,
  ):

    # Seed sama → Weight awal sama.
    random.seed(seed)

    model = Model(
      learning_rate=learning_rate,
      activation=activation,
    )

    # -------------------------------------------------------------
    # FORWARD + BACKWARD
    # -------------------------------------------------------------

    model.reset_gradient()

    for inputs, targets in diagnostic_batch:

      predictions = model.forward(
        inputs,
      )

      gradient_outputs = MSE.gradient(
        targets,
        predictions,
      )

      model.backward(
        gradient_outputs,
      )

    # -------------------------------------------------------------
    # SIMPAN GRADIENT SEBELUM UPDATE
    # -------------------------------------------------------------

    print()
    print(activation.value)

    for layer_index, layer in enumerate(
      model.layers,
      start=1,
    ):

      if layer_index == 3:
        print(
          " L3 LINEAR"
        )

      for neuron_index, neuron in enumerate(
        layer.neurons,
        start=1,
      ):

        max_gradient_weight = max(
          abs(gradient)
          for gradient in neuron.gradient_weights
        )

        gradient_bias = abs(
          neuron.gradient_bias
        )

        print(
          f" L{layer_index} "
          f"N{neuron_index} → "
          f"Weight Gradient: "
          f"{max_gradient_weight:.10f} | "
          f"Bias Gradient: "
          f"{gradient_bias:.10f}"
        )

    # -------------------------------------------------------------
    # AVERAGE GRADIENT
    # -------------------------------------------------------------

    model.average_gradient(
      len(diagnostic_batch),
    )

    # -------------------------------------------------------------
    # SIMPAN PARAMETER SEBELUM UPDATE
    # -------------------------------------------------------------

    before_weights = [
      [
        list(neuron.weights)
        for neuron in layer.neurons
      ]
      for layer in model.layers
    ]

    before_biases = [
      [
        neuron.bias
        for neuron in layer.neurons
      ]
      for layer in model.layers
    ]

    # -------------------------------------------------------------
    # UPDATE PARAMETER
    # -------------------------------------------------------------

    model.step()

    # -------------------------------------------------------------
    # HITUNG PARAMETER CHANGE
    # -------------------------------------------------------------

    print()

    for layer_index, layer in enumerate(
      model.layers,
      start=1,
    ):

      if layer_index == 3:
        print(
          " L3 LINEAR"
        )

      for neuron_index, neuron in enumerate(
        layer.neurons,
        start=1,
      ):

        max_weight_change = max(
          abs(
            new_weight - old_weight
          )
          for new_weight, old_weight
          in zip(
            neuron.weights,
            before_weights[layer_index - 1][neuron_index - 1],
          )
        )

        bias_change = abs(
          neuron.bias
          - before_biases[layer_index - 1][neuron_index - 1]
        )

        print(
          f" L{layer_index} "
          f"N{neuron_index} → "
          f"Weight Change: "
          f"{max_weight_change:.10f} | "
          f"Bias Change: "
          f"{bias_change:.10f}"
        )