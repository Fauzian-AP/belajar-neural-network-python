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


# =================================================================
# === TYPING MANUAL ===============================================
# =================================================================

print("=== TYPING MANUAL ===")


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


print(
  "IntList:",
  manual_int_list([1, 2, 3]),
)

print(
  "FloatList:",
  manual_float_list([1.0, 2.0, 3.0]),
)


# =================================================================
# === TYPING MANUAL ERROR =========================================
# =================================================================

print()
print("=== TYPING MANUAL ERROR ===")


try:
  manual_int("10")

except RuntimeFunctionError as error:
  print(error)


try:
  manual_float(10)

except RuntimeFunctionError as error:
  print(error)


try:
  manual_int_list([1, "salah", 3])

except RuntimeFunctionError as error:
  print(error)


try:
  manual_float_list([1.0, "salah", 3.0])

except RuntimeFunctionError as error:
  print(error)


# =================================================================
# === INT POSITIVE ================================================
# =================================================================

print()
print("=== INT POSITIVE ===")


@runtime_function(strict=True)
def create_batch_size(
  value: IntPositive,
) -> IntPositive:
  return value


print(
  "Valid:",
  create_batch_size(5),
)


try:
  create_batch_size(0)

except RuntimeFunctionError as error:
  print(error)


try:
  create_batch_size(-5)

except RuntimeFunctionError as error:
  print(error)


# =================================================================
# === FLOAT POSITIVE ==============================================
# =================================================================

print()
print("=== FLOAT POSITIVE ===")


@runtime_function(strict=True)
def create_learning_rate(
  value: FloatPositive,
) -> FloatPositive:
  return value


print(
  "Valid:",
  create_learning_rate(0.01),
)


try:
  create_learning_rate(0.0)

except RuntimeFunctionError as error:
  print(error)


try:
  create_learning_rate(-0.01)

except RuntimeFunctionError as error:
  print(error)


# =================================================================
# === ALPHA RANGE =================================================
# =================================================================

print()
print("=== ALPHA RANGE ===")


@runtime_function(strict=True)
def create_alpha(
  value: AlphaRange,
) -> AlphaRange:
  return value


print(
  "Valid:",
  create_alpha(0.01),
)


try:
  create_alpha(0.0)

except RuntimeFunctionError as error:
  print(error)


try:
  create_alpha(1.0)

except RuntimeFunctionError as error:
  print(error)


try:
  create_alpha(1.5)

except RuntimeFunctionError as error:
  print(error)


# =================================================================
# === FLOAT VECTOR ================================================
# =================================================================

print()
print("=== FLOAT VECTOR ===")


@runtime_function(strict=True)
def vector_function(
  value: FloatVector,
) -> FloatVector:
  return value


vector = vector_function(
  np.array(
    [1.0, 2.0, 3.0],
    dtype=np.float64,
  )
)

print(
  "Valid:",
  vector,
)


try:
  vector_function(
    np.array(
      [
        [1.0, 2.0],
        [3.0, 4.0],
      ],
      dtype=np.float64,
    )
  )

except RuntimeFunctionError as error:
  print(error)


# =================================================================
# === FLOAT MATRIX ================================================
# =================================================================

print()
print("=== FLOAT MATRIX ===")


@runtime_function(strict=True)
def matrix_function(
  value: FloatMatrix,
) -> FloatMatrix:
  return value


matrix = matrix_function(
  np.array(
    [
      [1.0, 2.0],
      [3.0, 4.0],
    ],
    dtype=np.float64,
  )
)

print(
  "Valid:",
  matrix,
)


try:
  matrix_function(
    np.array(
      [1.0, 2.0, 3.0],
      dtype=np.float64,
    )
  )

except RuntimeFunctionError as error:
  print(error)


# =================================================================
# === FLOAT ARRAY =================================================
# =================================================================

print()
print("=== FLOAT ARRAY ===")


@runtime_function(strict=True)
def array_function(
  value: FloatArray,
) -> FloatArray:
  return value


array_1d = array_function(
  np.array(
    [1.0, 2.0, 3.0],
    dtype=np.float64,
  )
)

array_2d = array_function(
  np.array(
    [
      [1.0, 2.0],
      [3.0, 4.0],
    ],
    dtype=np.float64,
  )
)

print(
  "1D:",
  array_1d,
)

print(
  "2D:",
  array_2d,
)


try:
  array_function(
    np.array(
      [
        [
          [1.0, 2.0],
          [3.0, 4.0],
        ],
      ],
      dtype=np.float64,
    )
  )

except RuntimeFunctionError as error:
  print(error)


# =================================================================
# === DATA SAMPLE =================================================
# =================================================================

print()
print("=== DATA SAMPLE ===")


@runtime_function(strict=True)
def create_sample(
  sample: DataSample,
) -> DataSample:
  return sample


sample = create_sample(
  DataSample(
    inputs=np.array(
      [1.0, 2.0, 3.0],
      dtype=np.float64,
    ),
    targets=np.array(
      [4.0, 5.0],
      dtype=np.float64,
    ),
  )
)

print(
  "Inputs:",
  sample.inputs,
)

print(
  "Targets:",
  sample.targets,
)


# =================================================================
# === DATASET =====================================================
# =================================================================

print()
print("=== DATASET ===")


@runtime_function(strict=True)
def create_dataset(
  dataset: Dataset,
) -> Dataset:
  return dataset


dataset = create_dataset(
  [
    DataSample(
      inputs=np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
      ),
      targets=np.array(
        [4.0, 5.0],
        dtype=np.float64,
      ),
    ),
  ]
)

print(
  "Dataset:",
  dataset,
)


# =================================================================
# === METRICS =====================================================
# =================================================================

print()
print("=== METRICS ===")


@runtime_function(strict=True)
def create_metrics(
  metrics: Metrics,
) -> Metrics:
  return metrics


metrics = create_metrics(
  {
    "MSE": 0.01,
    "MAE": 0.02,
    "RMSE": 0.1,
  }
)

print(
  "Metrics:",
  metrics,
)


try:
  create_metrics(
    {
      "MSE": 0.01,
      "MAE": 0.02,
      "SALAH": 0.1,
    }
  )

except RuntimeFunctionError as error:
  print(error)


# =================================================================
# === FUNCTION ARGUMENT ERROR ====================================
# =================================================================

print()
print("=== FUNCTION ARGUMENT ERROR ===")


@runtime_function(strict=True)
def tambah(
  a: int,
  b: int,
) -> int:
  return a + b


print(
  "Valid:",
  tambah(10, 20),
)


try:
  tambah(
    "salah",
    20,
  )

except RuntimeFunctionError as error:
  print(error)


# =================================================================
# === FUNCTION RETURN ERROR ======================================
# =================================================================

print()
print("=== FUNCTION RETURN ERROR ===")


@runtime_function(strict=True)
def salah_return() -> int:
  return "salah"


try:
  salah_return()

except RuntimeFunctionError as error:
  print(error)


# =================================================================
# === METHOD ARGUMENT ERROR =======================================
# =================================================================

print()
print("=== METHOD ARGUMENT ERROR ===")


class Calculator:

  @runtime_method(strict=True)
  def tambah(
    self,
    a: int,
    b: int,
  ) -> int:
    return a + b


calculator = Calculator()


print(
  "Valid:",
  calculator.tambah(10, 20),
)


try:
  calculator.tambah(
    "salah",
    20,
  )

except RuntimeMethodError as error:
  print(error)


# =================================================================
# === METHOD RETURN ERROR =========================================
# =================================================================

print()
print("=== METHOD RETURN ERROR ===")


class BrokenCalculator:

  @runtime_method(strict=True)
  def salah_return(
    self,
  ) -> int:
    return "salah"


broken_calculator = BrokenCalculator()


try:
  broken_calculator.salah_return()

except RuntimeMethodError as error:
  print(error)


# =================================================================
# === RETURN NONE ERROR ===========================================
# =================================================================

print()
print("=== RETURN NONE ERROR ===")


@runtime_function(strict=True)
def return_none() -> int:
  return None


try:
  return_none()

except RuntimeFunctionError as error:
  print(error)


print()
print("=== TEST SELESAI ===")