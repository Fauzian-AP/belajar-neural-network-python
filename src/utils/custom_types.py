"""Bagian Pengelolaan Type Code pd Project Neural Network"""

import numpy as np

from numpy.typing import NDArray
from enum import Enum

from typing import (
  Any,
  Annotated,
  Literal,
  TypeAlias,
  TypedDict,
)

from pydantic import (
  StrictInt,
  StrictFloat,
  StrictBool,
  BaseModel,
  BeforeValidator,
  ConfigDict,
  Field,
)


# ===================
# === BASIC TYPES ===
# ===================

Int: TypeAlias = StrictInt
Float: TypeAlias = StrictFloat
Bool: TypeAlias = StrictBool

IntList: TypeAlias = list[int]
FloatList: TypeAlias = list[float]


# =============================
# === NUMPY ARRAY VALIDATOR ===
# =============================

def _validate_vector(value: Any) -> NDArray[np.float64]:
  # Konversi menjadi NumPy Array float64
  array = np.asarray(value, dtype=np.float64)

  # Vector hrs 1D
  if array.ndim != 1:
    # Bangkitkan
    raise ValueError(f"Dimensi Vector hrs 1D, sedangkan pd value {array.ndim}D.")

  return array


def _validate_matrix(value: Any) -> NDArray[np.float64]:
  # Konversi menjadi NumPy Array float64
  array = np.asarray(value, dtype=np.float64)

  # Matrix harus 2D
  if array.ndim != 2:
    # Bangkitkan Error
    raise ValueError(f"Dimensi Matrix hrs 2D, sedangkan pd value {array.ndim}D.")

  return array


def _validate_array(value: Any) -> NDArray[np.float64]:
  # Konversi menjadi NumPy Array float64
  array = np.asarray(value, dtype=np.float64)

  # Array hanya boleh 1D atau 2D
  if array.ndim not in (1, 2):
    # Bangkitkan Error
    raise ValueError(f"Dimensi Array hrs 1D atau 2D, sedangkan pd value {array.ndim}D.")

  return array


# ===================
# === NUMPY TYPES ===
# ===================

FloatVector: TypeAlias = Annotated[NDArray[np.float64], BeforeValidator(_validate_vector)]
FloatMatrix: TypeAlias = Annotated[NDArray[np.float64], BeforeValidator(_validate_matrix)]
FloatArray: TypeAlias = Annotated[NDArray[np.float64], BeforeValidator(_validate_array)]


# ===========================
# === NUMERIC CONSTRAINTS ===
# ===========================

IntPositive: TypeAlias = Annotated[Int, Field(gt=0,)]
FloatPositive: TypeAlias = Annotated[Float, Field(gt=0.0)]
AlphaRange: TypeAlias = Annotated[Float, Field(gt=0.0, lt=1.0,)]


# ===================
# === DATA SAMPLE ===
# ===================

class DataSample(BaseModel):
  # NumPy Array merupakan Arbitrary Type
  model_config = ConfigDict(arbitrary_types_allowed=True)

  # Input Features
  inputs: FloatVector

  # Target / Expected Output
  targets: FloatVector


# ===============
# === DATASET ===
# ===============

DatasetName: TypeAlias = Literal[
  "training",
  "validation",
  "testing",
  "generalization",
]


Dataset: TypeAlias = list[DataSample]

DatasetList: TypeAlias = list[Dataset]

DatasetType: TypeAlias = dict[DatasetName, Dataset]


# ==================
# === EVALUATING ===
# ==================

MetricsName: TypeAlias = Literal["MSE", "MAE", "RMSE"]

Metrics: TypeAlias = dict[MetricsName, Float]

EvaluatingType: TypeAlias = dict[DatasetName, Metrics]


# =================================================================
# === CACHE NEURON =================================================
# =================================================================

class CacheNeuron(TypedDict):

  # Input yg digunakan ketika Forward
  inputs: FloatVector

  # Nilai sebelum Activation Function
  pre_activation: Float


# ========================
# === TRAINING HISTORY ===
# ========================

class MetricsHistory(TypedDict):
  MSE: FloatVector
  MAE: FloatVector
  RMSE: FloatVector


class FitHistory(TypedDict):
  # Nomor Epoch
  epochs: IntList

  # Metrics Training
  training_metrics: MetricsHistory

  # Metrics Validation
  validating_metrics: MetricsHistory


class FitResult(TypedDict):
  # Seluruh History Training
  history: FitHistory

  # Epoch dengan Validation MSE terbaik
  best_epoch: int

  # Epoch terakhir yg dijalankan
  last_epoch: int

  # Validation MSE terbaik
  best_validating_mse: float


# =================================================================
# === ENUM TYPES ==================================================
# =================================================================

class ScaleType(str, Enum):

  ORIGINAL = "original"

  NORMALIZE = "normalize"