# Bagian Pengelolaan Type Checking Project Neural Network

from enum import Enum
from pydantic import Field
from typing import (
  Annotated,
  Sequence,
  Union, 
  TypedDict, 
  TypeAlias,
)

# Type Union
Numeric: TypeAlias = Union[int, float]

# Aliases Number
ListInt: TypeAlias = list[int]
SequenceInt: TypeAlias = Sequence[int]

# Aliases Float
ListFloat: TypeAlias = list[float]
SequenceFloat: TypeAlias = Sequence[float]

# Aliases Dataset
Dataset: TypeAlias = list[tuple[ListFloat, ListFloat]]
ListDataset: TypeAlias = list[Dataset]
DatasetType: TypeAlias = dict[str, Dataset]

# Type Metrics
class Metrics(TypedDict):
  MSE: float
  MAE: float
  RMSE: float

# Aliases Plot
EvaluatingType: TypeAlias = dict[str, Metrics]

# Type Cache Neuron
class CacheNeuron(TypedDict):
  inputs: ListFloat
  pre_activation: float

# Type Metrics History Trainer
class MetricsHistory(TypedDict):
  MSE: list[float]
  MAE: list[float]
  RMSE: list[float]

# Type History Fit Trainer
class FitHistory(TypedDict):
  epochs: list[int]
  training_metrics: MetricsHistory
  validating_metrics: MetricsHistory

# Type Result Fit Trainer
class FitResult(TypedDict):
  history: FitHistory
  best_epoch: int
  last_epoch: int
  best_validating_mse: float

# Pydantic Validasi
PositiveInt = Annotated[int, Field(gt=0, description="Angka bulat hrs > 0")]
PositiveFloat = Annotated[float, Field(gt=0, description="Angka desimal hrs > 0")]

# Enum Jenis² Activation
class ActivationType(str, Enum):
  NONE = "none"
  RELU = "ReLU"
  LEAKY_RELU = "LeakyReLU"

# Enum Jenis² Bentuk Data
class ScaleType(str, Enum):
  ORIGINAL = "original"
  NORMALIZE = "normalize"