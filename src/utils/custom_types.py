""" Bagian Pengelolaan Type Code pd Project Neural Network """

from enum import Enum
from typing import (
  Annotated,
  NamedTuple,
  Sequence,
  TypedDict,
  TypeAlias,
)

import numpy as np
from beartype.vale import Is


# BASIC TYPES

IntVector: TypeAlias = list[int]
FloatVector: TypeAlias = list[float]


# NUMERIC CONSTRAINTS

IntPositive: TypeAlias = Annotated[int, Is[lambda value: value > 0]]
FloatPositive: TypeAlias = Annotated[float, Is[lambda value: value > 0.0]]

FloatSequence: TypeAlias = Annotated[Sequence[float], Is[lambda value: bool(value)]]

FloatArray: TypeAlias = Annotated[np.ndarray, Is[lambda arr: arr.dtype.type == np.float64]]

AlphaRange: TypeAlias = Annotated[float, Is[lambda value: 0.0 < value < 1.0]]


# DATASET TYPES

class DataSample(NamedTuple):
  inputs: FloatVector   # Data Input
  targets: FloatVector   # Data Target / Ground Truth

Dataset: TypeAlias = list[DataSample]   # Kumpulan Sample
DatasetList: TypeAlias = list[Dataset]   # Kumpulan Dataset
DatasetType: TypeAlias = dict[str, Dataset]   # Dataset berdasarkan Key


# EVALUATING TYPES

Metrics: TypeAlias = dict[str, float]

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


# ENUM TYPES

# Jenis² Bentuk Data

class ScaleType(str, Enum):
  ORIGINAL = "original"
  NORMALIZE = "normalize"