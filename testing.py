import math

from pydantic import ValidationError

from loss import MAE, MSE, MSE_gradient, RMSE
from metrics import calculate_metrics


print("=" * 65)
print("=== 1. PENGUJIAN KALKULASI MATEMATIS LOSS FUNCTION ===")
print("=" * 65)

targets = [3.0, -0.5, 2.0, 7.0]
predictions = [2.5, 0.0, 2.0, 8.0]

# Perhitungan Manual (N = 4):
# Error (Prediksi - Target) : [-0.5, 0.5, 0.0, 1.0]
# Squared Error             : [0.25, 0.25, 0.0, 1.0] -> Total = 1.5
# MSE                       : 1.5 / 4 = 0.375
# MAE                       : (0.5 + 0.5 + 0.0 + 1.0) / 4 = 0.5
# RMSE                      : √0.375 ≈ 0.6123724356957945
# MSE Grad                  : (2/4) * (Prediksi - Target) = 0.5 * [-0.5, 0.5, 0.0, 1.0]
#                           = [-0.25, 0.25, 0.0, 0.5]

calc_mse = MSE(targets, predictions)
calc_mae = MAE(targets, predictions)
calc_rmse = RMSE(targets, predictions)
calc_grad = MSE_gradient(targets, predictions)

print(f"Target      : {targets}")
print(f"Predictions : {predictions}\n")

print(f"[MSE]  Hasil: {calc_mse} (Ekspektasi: 0.375)")
assert math.isclose(calc_mse, 0.375), "Pengujian MSE Gagal!"

print(f"[MAE]  Hasil: {calc_mae} (Ekspektasi: 0.5)")
assert math.isclose(calc_mae, 0.5), "Pengujian MAE Gagal!"

print(f"[RMSE] Hasil: {calc_rmse:.6f} (Ekspektasi: 0.612372)")
assert math.isclose(calc_rmse, math.sqrt(0.375)), "Pengujian RMSE Gagal!"

print(f"[GRAD] Hasil: {calc_grad} (Ekspektasi: [-0.25, 0.25, 0.0, 0.5])")
assert calc_grad == [-0.25, 0.25, 0.0, 0.5], "Pengujian MSE_gradient Gagal!"


print("\n" + "=" * 65)
print("=== 2. PENGUJIAN ERROR HANDLING & VALIDASI LOSS FUNCTION ===")
print("=" * 65)

def test_exception(description: str, func) -> None:
  try:
    func()
    print(f"[FAIL] {description} -> TIDAK melempar error!")
  except (ValueError, ValidationError) as e:
    error_msg = str(e).split("\n")[0]
    print(f"[PASS] {description}")
    print(f'       Pesan Error Ditangkap: "{error_msg}"')

# 1. Target Kosong
test_exception("MSE dengan target kosong ([])", lambda: MSE([], [1.0, 2.0]))

# 2. Panjang Tidak Cocok
test_exception("MSE dengan panjang beda ([1.0, 2.0] vs [1.0])", lambda: MSE([1.0, 2.0], [1.0]),)

# 3. MSE Gradient Kosong
test_exception("MSE_gradient dengan input kosong", lambda: MSE_gradient([], []))

# 4. MAE Panjang Tidak Cocok
test_exception("MAE dengan panjang beda ([1.0] vs [1.0, 2.0])", lambda: MAE([1.0], [1.0, 2.0]),)

print("\n" + "=" * 65)
print("Hasil: Seluruh pengujian matematika & validasi loss.py LULUS!")
print("=" * 65)



print("=" * 65)
print("=== 1. PENGUJIAN KALKULASI METRICS ===")
print("=" * 65)

targets = [3.0, -0.5, 2.0, 7.0]
predictions = [2.5, 0.0, 2.0, 8.0]

metrics_res = calculate_metrics(targets, predictions)

print(f"Target      : {targets}")
print(f"Predictions : {predictions}\n")
print("Hasil Dictionary Metrics:")
for k, v in metrics_res.items():
  print(f"  - {k}: {v}")

assert math.isclose(metrics_res["MSE"], 0.375), "Metrics MSE salah!"
assert math.isclose(metrics_res["MAE"], 0.5), "Metrics MAE salah!"
assert math.isclose(metrics_res["RMSE"], math.sqrt(0.375)), "Metrics RMSE salah!"


print("\n" + "=" * 65)
print("=== 2. PENGUJIAN ERROR HANDLING METRICS ===")
print("=" * 65)

def test_exception(description: str, func) -> None:
  try:
    func()
    print(f"[FAIL] {description} -> TIDAK melempar error!")
  except (ValueError, ValidationError) as e:
    error_msg = str(e).split("\n")[0]
    print(f"[PASS] {description}")
    print(f'       Pesan Error Ditangkap: "{error_msg}"')

# 1. Input Kosong
test_exception("Calculate Metrics dengan list kosong", lambda: calculate_metrics([], []))

# 2. Panjang Beda
test_exception("Calculate Metrics dengan panjang beda",lambda: calculate_metrics([1.0, 2.0], [1.0]),)

print("\n" + "=" * 65)
print("Hasil: Seluruh pengujian metrics.py LULUS!")
print("=" * 65)