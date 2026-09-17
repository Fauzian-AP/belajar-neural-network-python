from src.evaluation.metric_evaluation import (
  MSE,
  MAE,
  RMSE,
  calculate_metrics,
)


# TEST DATA

targets = [1.0, 2.0, 4.0]
predictions = [2.0, 2.0, 2.0,]


# TEST MSE

mse = MSE()(targets, predictions)

assert mse == 5.0 / 3.0

print("✓ MSE berhasil")
print(f"  Result : {mse}")
print()


# TEST MAE

mae = MAE()(targets, predictions)

assert mae == 1.0

print("✓ MAE berhasil")
print(f"  Result : {mae}")
print()


# TEST RMSE

rmse = RMSE()(targets, predictions)

assert rmse == (5.0 / 3.0) ** 0.5

print("✓ RMSE berhasil")
print(f"  Result : {rmse}")
print()


# TEST CALCULATE METRICS

metrics = calculate_metrics(targets, predictions)

assert metrics["MSE"] == mse
assert metrics["MAE"] == mae
assert metrics["RMSE"] == rmse

print("✓ Calculate Metrics berhasil")
print(f"  Metrics : {metrics}")
print()


# TEST EMPTY TARGETS

try:
  MSE()([], predictions)
except Exception as error:
  print(f"Pesan Error: {error}")
finally:
  print()


# TEST EMPTY PREDICTIONS

try:
  MSE()(targets, [])
except Exception as error:
  print(f"Pesan Error: {error}")
finally:
  print()


# TEST LENGTH MISMATCH

try:
  MSE()(
    [1.0, 2.0, 3.0],
    [1.0, 2.0],
  )
except ValueError as error:
  print(f"Pesan Error: {error}")
finally:
  print()


# TEST WRONG TYPE

try:
  MSE()(
    ["1.0", "2.0"],
    predictions,
  )
except Exception as error:
  print(f"Pesan Error: {error}")
finally:
  print()


# TEST ZERO ERROR

zero_targets = [1.0, 2.0, 3.0]
zero_predictions = [1.0, 2.0, 3.0]

zero_metrics = calculate_metrics(
  zero_targets,
  zero_predictions,
)

assert zero_metrics["MSE"] == 0.0
assert zero_metrics["MAE"] == 0.0
assert zero_metrics["RMSE"] == 0.0

print("✓ Zero Error berhasil")


# FINISH

print()
print("Semua test Metric Evaluation berhasil.")