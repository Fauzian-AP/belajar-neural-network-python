import numpy as np

from src.validation.runtime import (
  RuntimeFunctionError,
  RuntimeMethodError,
  runtime_function,
  runtime_method,
)

from src.utils.custom_types import (
  Int,
  Float,
  IntList,
  FloatList,
  FloatVector,
  FloatMatrix,
  FloatArray,
  IntPositive,
  FloatPositive,
  AlphaRange,
  DataSample,
  Dataset,
  Metrics,
)


# =====================
# === TYPING MANUAL ===
# =====================

print("=== TYPING MANUAL ===\n")


@runtime_function(strict=True)
def manual_int(value: Int) -> Int:
  return value


@runtime_function(strict=True)
def manual_float(value: Float) -> Float:
  return value


@runtime_function(strict=True)
def manual_int_list(values: IntList) -> IntList:
  return values


@runtime_function(strict=True)
def manual_float_list(values: FloatList) -> FloatList:
  return values


print(f"Int : {manual_int(10)}")
print(f"Float : {manual_float(0.5)}")
print(f"IntList: {manual_int_list([1, 2, 3])}")
print(f"FloatList: {manual_float_list([1.0, 2.0, 3.0])}\n")


# ===========================
# === TYPING MANUAL ERROR ===
# ===========================

print("=== TYPING MANUAL ERROR ===\n")


try:
  manual_int("10")

except RuntimeFunctionError as error:
  print(f"{error}\n")


try:
  manual_float(10)

except RuntimeFunctionError as error:
  print(f"{error}\n")


try:
  manual_int_list([1, "salah", 3])

except RuntimeFunctionError as error:
  print(f"{error}\n")


try:
  manual_float_list([1.0, "salah", 3.0])

except RuntimeFunctionError as error:
  print(f"{error}\n")


# ====================
# === INT POSITIVE ===
# ====================

print("=== INT POSITIVE ===\n")


@runtime_function(strict=True)
def create_batch_size(value: IntPositive) -> IntPositive:
  return value


print(f"Valid: {create_batch_size(5)}\n")


try:
  create_batch_size(0)

except RuntimeFunctionError as error:
  print(f"{error}\n")


try:
  create_batch_size(-5)

except RuntimeFunctionError as error:
  print(f"{error}\n")


# ======================
# === FLOAT POSITIVE ===
# ======================

print("=== FLOAT POSITIVE ===\n")


@runtime_function(strict=True)
def create_learning_rate(value: FloatPositive) -> FloatPositive:
  return value


print(f"Valid: {create_learning_rate(0.01)}\n")


try:
  create_learning_rate(0.0)

except RuntimeFunctionError as error:
  print(f"{error}\n")


try:
  create_learning_rate(-0.01)

except RuntimeFunctionError as error:
  print(f"{error}\n")


# ===================
# === ALPHA RANGE ===
# ===================

print("=== ALPHA RANGE ===\n")


@runtime_function(strict=True)
def create_alpha(value: AlphaRange) -> AlphaRange:
  return value


print(f"Valid: {create_alpha(0.01)}\n")


try:
  create_alpha(0.0)

except RuntimeFunctionError as error:
  print(f"{error}\n")


try:
  create_alpha(1.0)

except RuntimeFunctionError as error:
  print(f"{error}\n")


try:
  create_alpha(1.5)

except RuntimeFunctionError as error:
  print(f"{error}\n")


# ====================
# === FLOAT VECTOR ===
# ====================

print("=== FLOAT VECTOR ===\n")


@runtime_function(strict=True)
def vector_function(value: FloatVector) -> FloatVector:
  return value


vector = vector_function(
  np.array([1.0, 2.0, 3.0], dtype=np.float64)
)

print(f"Valid: {vector}\n")


try:
  vector_function(
    np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
  )

except RuntimeFunctionError as error:
  print(f"{error}\n")


# ====================
# === FLOAT MATRIX ===
# ====================

print("=== FLOAT MATRIX ===\n")


@runtime_function(strict=True)
def matrix_function(value: FloatMatrix) -> FloatMatrix:
  return value


matrix = matrix_function(
  np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
)

print(f"Valid:\n{matrix}\n")


try:
  matrix_function(
    np.array([1.0, 2.0, 3.0], dtype=np.float64)
  )

except RuntimeFunctionError as error:
  print(f"{error}\n")


# ===================
# === FLOAT ARRAY ===
# ===================

print("=== FLOAT ARRAY ===\n")


@runtime_function(strict=True)
def array_function(value: FloatArray) -> FloatArray:
  return value


array_1d = array_function(
  np.array([1.0, 2.0, 3.0], dtype=np.float64)
)

array_2d = array_function(
  np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
)

print(f"1D: {array_1d}")
print(f"2D:\n{array_2d}\n")


try:
  array_function(
    np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)
  )

except RuntimeFunctionError as error:
  print(f"{error}\n")


# ===================
# === DATA SAMPLE ===
# ===================

print("=== DATA SAMPLE ===\n")


@runtime_function(strict=True)
def create_sample(sample: DataSample) -> DataSample:
  return sample


sample = create_sample(
  DataSample(
    inputs=np.array([1.0, 2.0, 3.0], dtype=np.float64),
    targets=np.array([4.0, 5.0], dtype=np.float64),
  )
)

print(f"Inputs  : {sample.inputs}")
print(f"Targets : {sample.targets}\n")


# ===============
# === DATASET ===
# ===============

print("=== DATASET ===\n")


@runtime_function(strict=True)
def create_dataset(dataset: Dataset) -> Dataset:
  return dataset


dataset = create_dataset(
  [
    DataSample(
      inputs=np.array([1.0, 2.0, 3.0], dtype=np.float64),
      targets=np.array([4.0, 5.0], dtype=np.float64),
    ),
  ]
)

print(f"Dataset: {dataset}\n")


# ===============
# === METRICS ===
# ===============

print("=== METRICS ===\n")


@runtime_function(strict=True)
def create_metrics(metrics: Metrics) -> Metrics:
  return metrics


metrics = create_metrics({"MSE": 0.01, "MAE": 0.02, "RMSE": 0.1})

print(f"Metrics: {metrics}\n")


try:
  create_metrics({"MSE": 0.01, "MAE": 0.02, "SALAH": 0.1})

except RuntimeFunctionError as error:
  print(f"{error}\n")


# ===============================
# === FUNCTION ARGUMENT ERROR ===
# ===============================

print("=== FUNCTION ARGUMENT ERROR ===\n")


@runtime_function(strict=True)
def tambah(a: int, b: int) -> int:
  return a + b


print(f"Valid: {tambah(10, 20)}\n")


try:
  tambah("salah", 20)

except RuntimeFunctionError as error:
  print(f"{error}\n")


# =============================
# === FUNCTION RETURN ERROR ===
# =============================

print("=== FUNCTION RETURN ERROR ===\n")


@runtime_function(strict=True)
def salah_return() -> int:
  return "salah"


try:
  salah_return()

except RuntimeFunctionError as error:
  print(f"{error}\n")


# =============================
# === METHOD ARGUMENT ERROR ===
# =============================

print("=== METHOD ARGUMENT ERROR ===\n")


class Calculator:
  # TAMBAH — Menggabungkan 2 nilai Argument
  @runtime_method(strict=True)
  def tambah(self, a: int, b: int) -> int:
    return a + b


calculator = Calculator()


print(f"Valid: {calculator.tambah(10, 20)}\n")


try:
  calculator.tambah("salah", 20)

except RuntimeMethodError as error:
  print(f"{error}\n")


# ===========================
# === METHOD RETURN ERROR ===
# ===========================

print("=== METHOD RETURN ERROR ===\n")


class BrokenCalculator:
  # SALAH RETURN — Method yg sengaja dibuat error untuk test
  @runtime_method(strict=True)
  def salah_return(self) -> int:
    return "salah"


broken_calculator = BrokenCalculator()


try:
  broken_calculator.salah_return()

except RuntimeMethodError as error:
  print(f"{error}\n")


# =========================
# === RETURN NONE ERROR ===
# =========================

print("=== RETURN NONE ERROR ===\n")


@runtime_function(strict=True)
def return_none() -> int:
  return None


try:
  return_none()

except RuntimeFunctionError as error:
  print(f"{error}\n")


# ===================
# === FINISH TEST ===
# ===================

print("=== TEST SELESAI ===")