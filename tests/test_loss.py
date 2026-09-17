from src.evaluation.loss_functions import (
  MSE,
  MAE,
)


# MSE

def test_mse() -> None:
  loss = MSE()

  targets = [1.0, 2.0, 3.0]
  predictions = [1.0, 3.0, 5.0]

  result = loss(targets, predictions)

  # Error : 0² + (-1)² + (-2)² = 5
  # MSE   : 5 / 3
  expected = 5.0 / 3.0

  assert result == expected

  print("✓ MSE berhasil")
  print("  Result :", result)
  print()


# MSE GRADIENT

def test_mse_gradient() -> None:
  loss = MSE()

  targets = [1.0, 2.0, 3.0]
  predictions = [2.0, 4.0, 1.0]

  gradients = loss.gradient(targets, predictions)

  expected = [
    2.0 / 3.0,
    4.0 / 3.0,
    -4.0 / 3.0,
  ]

  assert gradients == expected

  print("✓ MSE Gradient berhasil")
  print("  Gradient :", gradients)
  print()


# MAE

def test_mae() -> None:
  loss = MAE()

  targets = [1.0, 2.0, 3.0]
  predictions = [1.0, 3.0, 5.0]

  result = loss(targets, predictions)

  # Error:
  # |0| + |-1| + |-2| = 3
  # MAE = 3 / 3
  expected = 1.0

  assert result == expected

  print("✓ MAE berhasil")
  print("  Result :", result)
  print()


# MAE GRADIENT

def test_mae_gradient() -> None:
  loss = MAE()

  targets = [1.0, 2.0, 3.0]
  predictions = [2.0, 2.0, 1.0]

  gradients = loss.gradient(targets, predictions)

  expected = [1.0, 0.0, -1.0]

  assert gradients == expected

  print("✓ MAE Gradient berhasil")
  print("  Gradient :", gradients)
  print()


# EMPTY INPUT VALIDATION

def test_empty_targets() -> None:
  loss = MSE()

  try:
    loss([], [1.0])
  except Exception as error:
    print(f"Pesan Error : {error}")
  else:
    raise AssertionError("Empty targets hrs ditolak.")
  finally:
    print()



def test_empty_predictions() -> None:
  loss = MSE()

  try:
    loss([1.0], [])
  except Exception as error:
    print(f"Pesan Error : {error}")
  else:
    raise AssertionError("Empty predictions hrs ditolak.")
  finally:
    print()


def test_both_empty() -> None:
  loss = MSE()

  try:
    loss([], [])
  except Exception as error:
    print(f"Pesan Error : {error}")
  else:
    raise AssertionError("Input kosong hrs ditolak.")
  finally:
    print()


# LENGTH VALIDATION

def test_mismatched_length() -> None:
  loss = MSE()

  try:
    loss(
      [1.0, 2.0, 3.0],
      [1.0, 2.0],
    )
  except Exception as error:
    print(f"Pesan Error : {error}")
  else:
    raise AssertionError("Panjang input berbeda hrs ditolak.")
  finally:
    print()


def test_mismatched_length_mae() -> None:
  loss = MAE()

  try:
    loss.gradient([1.0, 2.0], [1.0])
  except Exception as error:
    print(f"Pesan Error : {error}")
  else:
    raise AssertionError("MAE hrs menolak panjang input berbeda.")
  finally:
    print()


# TYPE VALIDATION

def test_wrong_target_type() -> None:
  loss = MSE()

  try:
    loss(["1.0", "2.0"], [1.0, 2.0])
  except Exception as error:
    print(f"Pesan Error : {error}")
  else:
    raise AssertionError("Target salah tipe hrs ditolak.")
  finally:
    print()


def test_wrong_prediction_type() -> None:
  loss = MSE()

  try:
    loss([1.0, 2.0], ["1.0", "2.0"])
  except Exception as error:
    print(f"Pesan Error : {error}")
  else:
    raise AssertionError("Prediction salah tipe hrs ditolak.")
  finally:
    print()


def test_wrong_gradient_target_type() -> None:
  loss = MAE()

  try:
    loss.gradient(["1.0"], [1.0],)
  except Exception as error:
    print(f"Pesan Error : {error}")
  else:
    raise AssertionError("Gradient target salah tipe hrs ditolak.")
  finally:
    print()


# MSE / MAE ZERO ERROR

def test_mse_zero_error() -> None:
  loss = MSE()

  result = loss(
    [1.0, 2.0, 3.0],
    [1.0, 2.0, 3.0],
  )

  assert result == 0.0

  print("✓ MSE zero error berhasil")
  print()


def test_mae_zero_error() -> None:
  loss = MAE()

  result = loss(
    [1.0, 2.0, 3.0],
    [1.0, 2.0, 3.0],
  )

  assert result == 0.0

  print("✓ MAE zero error berhasil")
  print()


# MAIN

if __name__ == "__main__":

  test_mse()
  test_mse_gradient()

  test_mae()
  test_mae_gradient()

  test_empty_targets()
  test_empty_predictions()
  test_both_empty()

  test_mismatched_length()
  test_mismatched_length_mae()

  test_wrong_target_type()
  test_wrong_prediction_type()
  test_wrong_gradient_target_type()

  test_mse_zero_error()
  test_mae_zero_error()

  print()
  print("Semua test Loss Function berhasil.")