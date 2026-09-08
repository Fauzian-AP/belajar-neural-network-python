import copy
from model import Model
from trainer import Trainer
from batch import Batch
from preprocessing import Preprocessor
from custom_types import PositiveInt, Dataset, Metrics
from pydantic import validate_call

# ============================================================
# TEST MODEL + PREPROCESSOR + DENORMALIZATION
# ============================================================

preprocessor = Preprocessor()

training_data = [
  ([10.0, 20.0, 30.0], [100.0, 200.0]),
  ([20.0, 30.0, 40.0], [200.0, 300.0]),
  ([30.0, 40.0, 50.0], [300.0, 400.0]),
]

preprocessor.fit_dataset(training_data)

model = Model(learning_rate=0.001,)

# ------------------------------------------------------------
# ORIGINAL INPUT
# ------------------------------------------------------------

original_inputs = [
  20.0,
  30.0,
  40.0,
]

# ------------------------------------------------------------
# NORMALIZE INPUT
# ------------------------------------------------------------

normalized_inputs = (
  preprocessor.normalize_inputs(original_inputs )
)


# ------------------------------------------------------------
# MODEL FORWARD
# ------------------------------------------------------------

normalized_predictions = model.forward(
  normalized_inputs,
)


# ------------------------------------------------------------
# DENORMALIZE PREDICTION
# ------------------------------------------------------------

original_predictions = (
  preprocessor.denormalize_targets(
    normalized_predictions,
  )
)


# ------------------------------------------------------------
# VERIFY
# ------------------------------------------------------------

assert len(original_predictions) == 2

assert all(
  isinstance(
    prediction,
    float,
  )
  for prediction in original_predictions
)

print(
  "Normalized Prediction :",
  normalized_predictions,
)

print(
  "Original Prediction   :",
  original_predictions,
)

print(
  "✅ TEST MODEL + PREPROCESSOR "
  "DENORMALIZATION — PASS"
)