# Bagian Pengelolaan Type Code pd Project Neural Network

from beartype.vale import Is
from enum import Enum
from typing_extensions import (
  Annotated,
  Sequence,
  TypedDict,
  TypeAlias,
)

# BASIC TYPES

Numeric: TypeAlias = int | float

IntVector: TypeAlias = list[int]
FloatVector: TypeAlias = list[float]

# NUMERIC CONSTRAINTS

IntPositive: TypeAlias = Annotated[int, Is[lambda value: value > 0]]
FloatPositive: TypeAlias = Annotated[float, Is[lambda value: value > 0.0]]

FloatSequence: TypeAlias = Annotated[Sequence[float], Is[lambda value: bool(value)]]

AlphaRange: TypeAlias = Annotated[float, Is[lambda value: 0.0 < value < 1.0]]

# DATASET TYPES

Dataset: TypeAlias = list[tuple[FloatVector, FloatVector]]
ListDataset: TypeAlias = list[Dataset]
DatasetType: TypeAlias = dict[str, Dataset]

# EVALUATING TYPES

class Metrics(TypedDict):
  MSE: float
  MAE: float
  RMSE: float

EvaluatingType: TypeAlias = dict[str, Metrics]

# CACHE NEURON TYPE

class CacheNeuron(TypedDict):
  inputs: FloatVector
  pre_activation: float

# TRAINING HISTORY TYPES

class MetricsHistory(TypedDict):
  MSE: FloatVector
  MAE: FloatVector
  RMSE: FloatVector

class FitHistory(TypedDict):
  epochs: IntVector
  training_metrics: MetricsHistory
  validating_metrics: MetricsHistory

class FitResult(TypedDict):
  history: FitHistory
  best_epoch: int
  last_epoch: int
  best_validating_mse: float

# Enum Jenis² Activation

class ActivationType(str, Enum):
  NONE = "none"
  RELU = "ReLU"
  LEAKY_RELU = "Leaky_ReLU"

# Enum Jenis² Bentuk Data

class ScaleType(str, Enum):
  ORIGINAL = "original"
  NORMALIZE = "normalize"