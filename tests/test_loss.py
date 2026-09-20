import numpy as np

from src.evaluation.loss_functions import (
  MSE,
  MAE,
)


# MSE

def test_mse() -> None:

  loss = MSE()

  targets = np.array([1.0, 2.0, 3.0], dtype=np.float64)
  predictions = np.array([1.0, 3.0, 5.0], dtype=np.float64,)

  result = loss(
    targets,
    predictions,
  )

  # Error:
  # 0² + (-1)² + (-2)² = 5
  #
  # MSE:
  # 5 / 3
  expected = 5.0 / 3.0

  assert np.isclose(
    result,
    expected,
  )

  print("✓ MSE berhasil")
  print("  Result :", result)
  print()


# =================================================================
# === MSE GRADIENT ================================================
# =================================================================

def test_mse_gradient() -> None:

  loss = MSE()

  targets = np.array(
    [1.0, 2.0, 3.0],
    dtype=np.float64,
  )

  predictions = np.array(
    [2.0, 4.0, 1.0],
    dtype=np.float64,
  )

  gradients = loss.gradient(
    targets,
    predictions,
  )

  expected = np.array(
    [
      2.0 / 3.0,
      4.0 / 3.0,
      -4.0 / 3.0,
    ],
    dtype=np.float64,
  )

  assert np.allclose(
    gradients,
    expected,
  )

  print("✓ MSE Gradient berhasil")
  print("  Gradient :", gradients)
  print()


# =================================================================
# === MAE =========================================================
# =================================================================

def test_mae() -> None:

  loss = MAE()

  targets = np.array(
    [1.0, 2.0, 3.0],
    dtype=np.float64,
  )

  predictions = np.array(
    [1.0, 3.0, 5.0],
    dtype=np.float64,
  )

  result = loss(
    targets,
    predictions,
  )

  # Error:
  # |0| + |-1| + |-2| = 3
  #
  # MAE:
  # 3 / 3 = 1
  expected = 1.0

  assert np.isclose(
    result,
    expected,
  )

  print("✓ MAE berhasil")
  print("  Result :", result)
  print()


# =================================================================
# === MAE GRADIENT ================================================
# =================================================================

def test_mae_gradient() -> None:

  loss = MAE()

  targets = np.array(
    [1.0, 2.0, 3.0],
    dtype=np.float64,
  )

  predictions = np.array(
    [2.0, 2.0, 1.0],
    dtype=np.float64,
  )

  gradients = loss.gradient(
    targets,
    predictions,
  )

  # sign(prediction - target):
  #
  # [1, 0, -1]
  #
  # Karena MAE menggunakan Mean:
  #
  # [1/3, 0, -1/3]

  expected = np.array(
    [
      1.0 / 3.0,
      0.0,
      -1.0 / 3.0,
    ],
    dtype=np.float64,
  )

  assert np.allclose(
    gradients,
    expected,
  )

  print("✓ MAE Gradient berhasil")
  print("  Gradient :", gradients)
  print()


# =================================================================
# === OUTPUT TYPE VALIDATION ======================================
# =================================================================

def test_output_type() -> None:

  loss = MSE()

  targets = np.array(
    [1.0, 2.0, 3.0],
    dtype=np.float64,
  )

  predictions = np.array(
    [2.0, 3.0, 4.0],
    dtype=np.float64,
  )

  result = loss(
    targets,
    predictions,
  )

  gradients = loss.gradient(
    targets,
    predictions,
  )

  assert isinstance(
    result,
    float,
  )

  assert isinstance(
    gradients,
    np.ndarray,
  )

  assert gradients.dtype == np.float64

  print("✓ MSE Result berupa Python float")
  print("✓ MSE Gradient berupa NumPy ndarray")
  print("✓ MSE Gradient dtype float64")
  print()


# =================================================================
# === OUTPUT SHAPE VALIDATION =====================================
# =================================================================

def test_output_shape() -> None:

  loss = MAE()

  targets = np.array(
    [1.0, 2.0, 3.0],
    dtype=np.float64,
  )

  predictions = np.array(
    [2.0, 3.0, 4.0],
    dtype=np.float64,
  )

  gradients = loss.gradient(
    targets,
    predictions,
  )

  assert gradients.shape == targets.shape

  print("✓ MAE Gradient shape :", gradients.shape)
  print()


# =================================================================
# === EMPTY INPUT VALIDATION ======================================
# =================================================================

def test_empty_targets() -> None:

  loss = MSE()

  try:
    loss(
      np.array(
        [],
        dtype=np.float64,
      ),
      np.array(
        [1.0],
        dtype=np.float64,
      ),
    )

  except Exception as error:

    print(
      f"Pesan Error "
      f"({type(error).__name__}) : "
      f"{error}"
    )

  else:

    raise AssertionError(
      "Empty targets hrs ditolak."
    )

  finally:
    print()


def test_empty_predictions() -> None:

  loss = MSE()

  try:
    loss(
      np.array(
        [1.0],
        dtype=np.float64,
      ),
      np.array(
        [],
        dtype=np.float64,
      ),
    )

  except Exception as error:

    print(
      f"Pesan Error "
      f"({type(error).__name__}) : "
      f"{error}"
    )

  else:

    raise AssertionError(
      "Empty predictions hrs ditolak."
    )

  finally:
    print()


def test_both_empty() -> None:

  loss = MSE()

  try:
    loss(
      np.array(
        [],
        dtype=np.float64,
      ),
      np.array(
        [],
        dtype=np.float64,
      ),
    )

  except Exception as error:

    print(
      f"Pesan Error "
      f"({type(error).__name__}) : "
      f"{error}"
    )

  else:

    raise AssertionError(
      "Input kosong hrs ditolak."
    )

  finally:
    print()


# =================================================================
# === SHAPE VALIDATION =============================================
# =================================================================

def test_mismatched_shape() -> None:

  loss = MSE()

  try:
    loss(
      np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
      ),
      np.array(
        [1.0, 2.0],
        dtype=np.float64,
      ),
    )

  except Exception as error:

    print(
      f"Pesan Error "
      f"({type(error).__name__}) : "
      f"{error}"
    )

  else:

    raise AssertionError(
      "Shape input berbeda hrs ditolak."
    )

  finally:
    print()


def test_mismatched_shape_mae() -> None:

  loss = MAE()

  try:
    loss.gradient(
      np.array(
        [1.0, 2.0],
        dtype=np.float64,
      ),
      np.array(
        [1.0],
        dtype=np.float64,
      ),
    )

  except Exception as error:

    print(
      f"Pesan Error "
      f"({type(error).__name__}) : "
      f"{error}"
    )

  else:

    raise AssertionError(
      "MAE hrs menolak shape input berbeda."
    )

  finally:
    print()


# =================================================================
# === TYPE VALIDATION ==============================================
# =================================================================

def test_wrong_target_type() -> None:

  loss = MSE()

  try:
    loss(
      "invalid",
      np.array(
        [1.0, 2.0],
        dtype=np.float64,
      ),
    )

  except Exception as error:

    print(
      f"Pesan Error "
      f"({type(error).__name__}) : "
      f"{error}"
    )

  else:

    raise AssertionError(
      "Target salah tipe hrs ditolak."
    )

  finally:
    print()


def test_wrong_prediction_type() -> None:

  loss = MSE()

  try:
    loss(
      np.array(
        [1.0, 2.0],
        dtype=np.float64,
      ),
      "invalid",
    )

  except Exception as error:

    print(
      f"Pesan Error "
      f"({type(error).__name__}) : "
      f"{error}"
    )

  else:

    raise AssertionError(
      "Prediction salah tipe hrs ditolak."
    )

  finally:
    print()


def test_wrong_gradient_target_type() -> None:

  loss = MAE()

  try:
    loss.gradient(
      "invalid",
      np.array(
        [1.0],
        dtype=np.float64,
      ),
    )

  except Exception as error:

    print(
      f"Pesan Error "
      f"({type(error).__name__}) : "
      f"{error}"
    )

  else:

    raise AssertionError(
      "Gradient target salah tipe hrs ditolak."
    )

  finally:
    print()


# =================================================================
# === MSE / MAE ZERO ERROR =========================================
# =================================================================

def test_mse_zero_error() -> None:

  loss = MSE()

  values = np.array(
    [1.0, 2.0, 3.0],
    dtype=np.float64,
  )

  result = loss(
    values,
    values,
  )

  assert result == 0.0

  print("✓ MSE zero error berhasil")
  print()


def test_mae_zero_error() -> None:

  loss = MAE()

  values = np.array(
    [1.0, 2.0, 3.0],
    dtype=np.float64,
  )

  result = loss(
    values,
    values,
  )

  assert result == 0.0

  print("✓ MAE zero error berhasil")
  print()


# =================================================================
# === MAIN ========================================================
# =================================================================

if __name__ == "__main__":

  test_mse()
  test_mse_gradient()

  test_mae()
  test_mae_gradient()

  test_output_type()
  test_output_shape()

  test_empty_targets()
  test_empty_predictions()
  test_both_empty()

  test_mismatched_shape()
  test_mismatched_shape_mae()

  test_wrong_target_type()
  test_wrong_prediction_type()
  test_wrong_gradient_target_type()

  test_mse_zero_error()
  test_mae_zero_error()

  print("Semua Test Loss Function Berhasil.")