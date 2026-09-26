"""Bagian Pengelolaan Type Code pd Project Neural Network"""

import numpy as np

from numpy.typing import NDArray

from enum import Enum
from collections.abc import Sequence

from typing import (
  Any,
  Literal,
  Annotated,
  TypedDict,
)

from pydantic import (
  StrictInt,
  StrictFloat,
  StrictBool,
  BaseModel,
  ConfigDict,
  Field,
  BeforeValidator,
)


# ===================
# === BASIC TYPES ===
# ===================

type Numeric = StrictInt | StrictFloat

type Int = StrictInt
type Float = StrictFloat
type Bool = StrictBool


# =============================
# === NUMPY ARRAY VALIDATOR ===
# =============================

def _validate_vector(value: Any) -> NDArray[np.float64]:
  # Konversi menjadi NumPy Array float64
  array = np.asarray(value, dtype=np.float64)

  # Vector hrs 1D
  if array.ndim != 1:
    # Bangkitkan Error
    raise ValueError(f"Dimensi Vector hrs 1D, sedangkan pd value {array.ndim}D.")

  # Vector hrs memiliki isi
  if array.size == 0:
    # Bangkitkan Error
    raise ValueError(f"Vector tdk boleh kosong.")

  return array


def _validate_matrix(value: Any) -> NDArray[np.float64]:
  # Konversi menjadi NumPy Array float64
  array = np.asarray(value, dtype=np.float64)

  # Matrix harus 2D
  if array.ndim != 2:
    # Bangkitkan Error
    raise ValueError(f"Dimensi Matrix hrs 2D, sedangkan pd value {array.ndim}D.")

  # Matrix hrs memiliki isi
  if array.size == 0:
    # Bangkitkan Error
    raise ValueError(f"Matrix tdk boleh kosong.")

  return array


def _validate_array(value: Any) -> NDArray[np.float64]:
  # Konversi menjadi NumPy Array float64
  array = np.asarray(value, dtype=np.float64)

  # Array hanya boleh 1D atau 2D
  if array.ndim not in (1, 2):
    # Bangkitkan Error
    raise ValueError(f"Dimensi Array hrs 1D atau 2D, sedangkan pd value {array.ndim}D.")

  # Array hrs memiliki isi
  if array.size == 0:
    # Bangkitkan Error
    raise ValueError(f"Array tdk boleh kosong.")

  return array


# ===================
# === NUMPY TYPES ===
# ===================

type FloatVector = Annotated[NDArray[np.float64], BeforeValidator(_validate_vector)]
type FloatMatrix = Annotated[NDArray[np.float64], BeforeValidator(_validate_matrix)]
type FloatArray = Annotated[NDArray[np.float64], BeforeValidator(_validate_array)]


# ===========================
# === NUMERIC CONSTRAINTS ===
# ===========================

type IntPositive = Annotated[Int, Field(gt=0,)]
type FloatPositive = Annotated[Float, Field(gt=0.0)]

type NumSequence = Annotated[Sequence[Numeric], Field(min_length=1)]

type IntList = Annotated[list[Int], Field(min_length=1)]
type FloatList = Annotated[list[Float], Field(min_length=1)]

type AlphaRange = Annotated[Float, Field(gt=0.0, lt=1.0,)]


# ===================
# === DATA SAMPLE ===
# ===================

class DataSample(BaseModel):
  # NumPy Array merupakan Arbitrary Type
  model_config = ConfigDict(arbitrary_types_allowed=True)

  # Input Features
  inputs: FloatVector

  # Expected Output
  targets: FloatVector


# ========================
# === LIST DATA SAMPLE ===
# ========================

type DataSampleList = Annotated[list[DataSample], Field(min_length=1)]


# ===============
# === DATASET ===
# ===============

type DatasetName = Literal["training", "validation", "testing", "generalization"]

class Dataset(BaseModel):
  # Hanya 4 jenis Dataset yg diperbolehkan.
  model_config = ConfigDict(extra="forbid")

  # Dataset Training
  training: DataSampleList

  # Dataset Validation
  validation: DataSampleList

  # Dataset Testing
  testing: DataSampleList

  # Dataset Generalization
  generalization: DataSampleList


type DatasetList = list[Dataset]


# ==================
# === EVALUATING ===
# ==================

type MetricsName = Literal["MSE", "MAE", "RMSE"]

type Metrics = dict[MetricsName, Float]

type EvaluatingType = dict[DatasetName, Metrics]


# ====================
# === CACHE NEURON ===
# ====================

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


# ==================
# === ENUM TYPES ===
# ==================

class ScaleType(str, Enum):

  ORIGINAL = "original"

  NORMALIZE = "normalize"