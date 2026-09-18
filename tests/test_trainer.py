# Test untuk Bagian Trainer

from src.core import Model
from src.core import LeakyReLU
from src.core import HeNormal
from src.core import SGD

from src.evaluation import MSE

from src.training import Batch
from src.training.trainer import Trainer

from src.utils.custom_types import (
  Dataset,
)


# DATA

training_data: Dataset = [
  ([0.0, 0.0, 0.0], [0.0, 0.0]),
  ([1.0, 0.0, 0.0], [1.0, 0.0]),
  ([0.0, 1.0, 0.0], [0.0, 1.0]),
  ([1.0, 1.0, 0.0], [1.0, 1.0]),
]

validation_data: Dataset = [
  ([0.5, 0.5, 0.0], [0.5, 0.5]),
  ([0.8, 0.2, 0.0], [0.8, 0.2]),
]


# FACTORY

def create_model() -> Model:
  return Model(
    initializer=HeNormal(),
    optimizer=SGD(learning_rate=0.01),
    activation=LeakyReLU(alpha=0.01),
  )


def create_trainer() -> Trainer:
  return Trainer(
    model=create_model(),
    batch=Batch(seed=42),
    loss=MSE(),
  )


# CONSTRUCTOR

trainer = create_trainer()

assert isinstance(trainer.model, Model)
assert isinstance(trainer.batch, Batch)
assert isinstance(trainer.loss, MSE)

print("✓ Constructor berhasil\n")


# EVALUATION 

metrics = trainer.evaluation(training_data)

assert set(metrics.keys()) == {
  "MSE",
  "MAE",
  "RMSE",
}

assert all(
  isinstance(value, float)
  for value in metrics.values()
)

assert all(
  value >= 0.0
  for value in metrics.values()
)

print("✓ Evaluation berhasil\n")


# EMPTY DATASET

try:
  trainer.evaluation([])

  raise AssertionError("Evaluation seharusnya menolak Dataset kosong.")
except ValueError as error:
  print(f"Pesan Error: {error}")
finally:
  print()

print("✓ Evaluation Dataset kosong berhasil\n")


# TRAINING

training_metrics = trainer.training(
  training_data,
  batch_size=2,
)

assert set(training_metrics.keys()) == {
  "MSE",
  "MAE",
  "RMSE",
}

assert all(
  isinstance(value, float)
  for value in training_metrics.values()
)

assert all(
  value >= 0.0
  for value in training_metrics.values()
)

print("✓ Training berhasil\n")


# SAVE / RESTORE MODEL

trainer = create_trainer()

original_state = trainer.save_model()

# Ubah Parameter Model
trainer.model.layers[0].neurons[0].weights[0] += 100.0
trainer.model.layers[0].neurons[0].bias += 100.0

modified_state = trainer.save_model()

assert modified_state != original_state

# Restore
trainer.restore_model(original_state)

restored_state = trainer.save_model()

# Bandingkan isi Weight & Bias
for restored_layer, original_layer in zip(restored_state, original_state):
  for restored_neuron, original_neuron in zip(restored_layer.neurons, original_layer.neurons):
    assert restored_neuron.weights == original_neuron.weights
    assert restored_neuron.bias == original_neuron.bias

print("✓ Save / Restore Model berhasil\n")


# DEEP COPY 

trainer = create_trainer()

saved_state = trainer.save_model()

# Simpan nilai awal
original_weight = (
  trainer.model.layers[0]
  .neurons[0]
  .weights[0]
)

# Ubah Model setelah State disimpan
trainer.model.layers[0].neurons[0].weights[0] += 50.0

# State yang telah disimpan tidak boleh ikut berubah
assert (
  saved_state[0]
  .neurons[0]
  .weights[0]
  == original_weight
)

print("✓ Deep Copy Model berhasil\n")


# FIT
trainer = create_trainer()

fit_result = trainer.fit(
  training_data=training_data,
  validation_data=validation_data,
  epochs=5,
  batch_size=2,
  patience=3,
)

assert set(fit_result.keys()) == {
  "history",
  "best_epoch",
  "last_epoch",
  "best_validation_mse",
}

assert fit_result["best_epoch"] >= 1
assert fit_result["last_epoch"] >= fit_result["best_epoch"]
assert fit_result["best_validation_mse"] >= 0.0

print("✓ Fit berhasil\n")


# HISTORY

history = fit_result["history"]

assert history["epochs"] == list(
  range(1, fit_result["last_epoch"] + 1)
)

assert len(history["training_metrics"]["MSE"]) == fit_result["last_epoch"]
assert len(history["training_metrics"]["MAE"]) == fit_result["last_epoch"]
assert len(history["training_metrics"]["RMSE"]) == fit_result["last_epoch"]
assert len(history["validation_metrics"]["MSE"]) == fit_result["last_epoch"]
assert len(history["validation_metrics"]["MAE"]) == fit_result["last_epoch"]
assert len(history["validation_metrics"]["RMSE"]) == fit_result["last_epoch"]

print("✓ History berhasil\n")


# BEST MODEL

class ControlledTrainer(Trainer):

  def __init__(
    self,
    model: Model,
    batch: Batch,
    loss: MSE,
    validation_mse_values: list[float],
  ) -> None:
    super().__init__(
      model=model,
      batch=batch,
      loss=loss,
    )

    self.validation_mse_values = validation_mse_values

    self.validation_index = 0


  def training(self, dataset: Dataset, batch_size: int) -> dict[str, float]:
    # Ubah parameter Model setiap Epoch.
    self.model.layers[0].neurons[0].bias += 1.0

    return {
      "MSE": 1.0,
      "MAE": 1.0,
      "RMSE": 1.0,
    }


  def evaluation(self, dataset: Dataset) -> dict[str, float]:
    value = self.validation_mse_values[self.validation_index]

    self.validation_index += 1

    return {
      "MSE": value,
      "MAE": value,
      "RMSE": value,
    }


controlled_trainer = ControlledTrainer(
  model=create_model(),
  batch=Batch(seed=42),
  loss=MSE(),
  validation_mse_values=[0.50, 0.30, 0.10, 0.20, 0.40],
)

controlled_result = controlled_trainer.fit(
  training_data=training_data,
  validation_data=validation_data,
  epochs=5,
  batch_size=2,
  patience=10,
)

assert controlled_result["best_epoch"] == 3

assert (
  controlled_result["best_validating_mse"]
  if "best_validating_mse"
  in controlled_result
  else controlled_result["best_validation_mse"]
) == 0.10

print("✓ Best Model berhasil\n")


# EARLY STOPPING

early_stop_trainer = ControlledTrainer(
  model=create_model(),
  batch=Batch(seed=42),
  loss=MSE(),
  validation_mse_values=[0.10, 0.20, 0.30, 0.40, 0.50],
)

early_stop_result = early_stop_trainer.fit(
  training_data=training_data,
  validation_data=validation_data,
  epochs=10,
  batch_size=2,
  patience=2,
)

assert early_stop_result["best_epoch"] == 1
assert early_stop_result["last_epoch"] == 3
assert early_stop_result["best_validation_mse"] == 0.10

print("✓ Early Stopping berhasil\n")


# RESTORE BEST MODEL

controlled_trainer = ControlledTrainer(
  model=create_model(),
  batch=Batch(seed=42),
  loss=MSE(),
  validation_mse_values=[0.50, 0.30, 0.10, 0.20, 0.40],
)

initial_bias = (
  controlled_trainer
  .model.layers[0]
  .neurons[0]
  .bias
)

controlled_trainer.fit(
  training_data=training_data,
  validation_data=validation_data,
  epochs=5,
  batch_size=2,
  patience=10,
)

restored_bias = (
  controlled_trainer
  .model.layers[0]
  .neurons[0]
  .bias
)

# Training menaikkan bias 1 setiap Epoch.
# Best Model terjadi pada Epoch 3.
assert restored_bias == (initial_bias + 3.0)

print("✓ Restore Best Model berhasil\n")


# BEARTYPE VALIDATION

trainer = create_trainer()


# batch_size harus > 0
try:
  trainer.training(training_data, 0)

  raise AssertionError("batch_size=0 seharusnya ditolak.")
except Exception as error:
  print(f"Pesan Error: {error}")
finally:
  print()


# batch_size harus int
try:
  trainer.training(training_data, "2")

  raise AssertionError("batch_size='2' seharusnya ditolak.")
except Exception as error:
  print(f"Pesan Error: {error}")
finally:
  print()


# Dataset harus sesuai type
try:
  trainer.evaluation("dataset")

  raise AssertionError("Dataset string seharusnya ditolak.")
except Exception as error:
  print(f"Pesan Error: {error}")
finally:
  print()


print("✓ Validasi Beartype berhasil\n")


# RESULT

print("Semua Test Trainer Berhasil.")