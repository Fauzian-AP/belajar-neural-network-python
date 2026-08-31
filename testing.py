# =================================================================
# === IMPORT ======================================================
# =================================================================

import math

from pydantic import ValidationError

from loss import MSE
from metrics import MAE, RMSE, calculate_metrics
from normalizer import (
  Normalizer,
  normalize_list,
  denormalize_list,
)

# =================================================================
# === HELPER ERROR HANDLING =======================================
# =================================================================

def test_exception(description: str, func) -> None:
  # Jalankan fungsi yang diharapkan menghasilkan Error
  try:
    func()

    # Jika tidak menghasilkan Error
    print(f"[FAIL] {description} -> TIDAK melempar error!")

  # Tangkap Error dari validasi
  except (ValueError, ValidationError) as e:
    # Ambil baris pertama pesan Error
    error_msg = str(e).split("\n")[0]

    print(f"[PASS] {description}")
    print(f'       Pesan Error Ditangkap: "{error_msg}"')


# =================================================================
# === 1. PENGUJIAN KALKULASI LOSS FUNCTION =======================
# =================================================================

print("=" * 65)
print("=== 1. PENGUJIAN KALKULASI MATEMATIS LOSS FUNCTION ===")
print("=" * 65)

# DATA
targets = [3.0, -0.5, 2.0, 7.0]
predictions = [2.5, 0.0, 2.0, 8.0]

# =================================================================
# === PERHITUNGAN MANUAL ==========================================
# =================================================================

# Jumlah data:
# N = 4

# Error (Prediksi - Target):
# [-0.5, 0.5, 0.0, 1.0]

# Squared Error:
# [0.25, 0.25, 0.0, 1.0]

# Total Squared Error:
# 1.5

# MSE:
# 1.5 / 4 = 0.375

# MAE:
# (0.5 + 0.5 + 0.0 + 1.0) / 4 = 0.5

# RMSE:
# √0.375 ≈ 0.612372

# MSE Gradient:
# (2 / 4) × (Prediksi - Target)
#
# = 0.5 × [-0.5, 0.5, 0.0, 1.0]
#
# = [-0.25, 0.25, 0.0, 0.5]


# =================================================================
# === CALCULATE ==================================================
# =================================================================

calc_mse = MSE.calculate(
  targets,
  predictions,
)

calc_mae = MAE.calculate(
  targets,
  predictions,
)

calc_rmse = RMSE.calculate(
  targets,
  predictions,
)

calc_grad = MSE.gradient(
  targets,
  predictions,
)


# =================================================================
# === TAMPILKAN HASIL =============================================
# =================================================================

print(f"Target      : {targets}")
print(f"Predictions : {predictions}\n")


# =================================================================
# === TEST MSE ====================================================
# =================================================================

print(
  f"[MSE]  Hasil: {calc_mse} "
  "(Ekspektasi: 0.375)"
)

assert math.isclose(
  calc_mse,
  0.375,
), "Pengujian MSE Gagal!"


# =================================================================
# === TEST MAE ====================================================
# =================================================================

print(
  f"[MAE]  Hasil: {calc_mae} "
  "(Ekspektasi: 0.5)"
)

assert math.isclose(
  calc_mae,
  0.5,
), "Pengujian MAE Gagal!"


# =================================================================
# === TEST RMSE ===================================================
# =================================================================

print(
  f"[RMSE] Hasil: {calc_rmse:.6f} "
  "(Ekspektasi: 0.612372)"
)

assert math.isclose(
  calc_rmse,
  math.sqrt(0.375),
), "Pengujian RMSE Gagal!"


# =================================================================
# === TEST MSE GRADIENT ===========================================
# =================================================================

expected_grad = [
  -0.25,
  0.25,
  0.0,
  0.5,
]

print(
  f"[GRAD] Hasil: {calc_grad} "
  f"(Ekspektasi: {expected_grad})"
)

assert all(
  math.isclose(actual, expected)
  for actual, expected
  in zip(calc_grad, expected_grad)
), "Pengujian MSE Gradient Gagal!"


# =================================================================
# === 2. PENGUJIAN ERROR HANDLING LOSS FUNCTION ===================
# =================================================================

print("\n" + "=" * 65)
print("=== 2. PENGUJIAN ERROR HANDLING LOSS FUNCTION ===")
print("=" * 65)


# =================================================================
# === TEST TARGET KOSONG ==========================================
# =================================================================

test_exception(
  "MSE dengan target kosong",
  lambda: MSE.calculate(
    [],
    [1.0, 2.0],
  ),
)


# =================================================================
# === TEST PANJANG TIDAK COCOK ====================================
# =================================================================

test_exception(
  "MSE dengan panjang berbeda",
  lambda: MSE.calculate(
    [1.0, 2.0],
    [1.0],
  ),
)


# =================================================================
# === TEST MSE GRADIENT KOSONG =====================================
# =================================================================

test_exception(
  "MSE Gradient dengan input kosong",
  lambda: MSE.gradient(
    [],
    [],
  ),
)


# =================================================================
# === TEST MAE PANJANG TIDAK COCOK ================================
# =================================================================

test_exception(
  "MAE dengan panjang berbeda",
  lambda: MAE.calculate(
    [1.0],
    [1.0, 2.0],
  ),
)


print("\n" + "=" * 65)
print("Hasil: Seluruh pengujian Loss Function LULUS!")
print("=" * 65)


# =================================================================
# === 3. PENGUJIAN KALKULASI METRICS ==============================
# =================================================================

print("\n" + "=" * 65)
print("=== 3. PENGUJIAN KALKULASI METRICS ===")
print("=" * 65)


# DATA
targets = [3.0, -0.5, 2.0, 7.0]
predictions = [2.5, 0.0, 2.0, 8.0]


# =================================================================
# === HITUNG METRICS ==============================================
# =================================================================

metrics_result = calculate_metrics(
  targets,
  predictions,
)


# =================================================================
# === TAMPILKAN HASIL =============================================
# =================================================================

print(f"Target      : {targets}")
print(f"Predictions : {predictions}\n")

print("Hasil Dictionary Metrics:")

for key, value in metrics_result.items():
  print(f"  - {key}: {value}")


# =================================================================
# === TEST MSE ====================================================
# =================================================================

assert math.isclose(
  metrics_result["MSE"],
  0.375,
), "Metrics MSE salah!"


# =================================================================
# === TEST MAE ====================================================
# =================================================================

assert math.isclose(
  metrics_result["MAE"],
  0.5,
), "Metrics MAE salah!"


# =================================================================
# === TEST RMSE ===================================================
# =================================================================

assert math.isclose(
  metrics_result["RMSE"],
  math.sqrt(0.375),
), "Metrics RMSE salah!"


# =================================================================
# === 4. PENGUJIAN ERROR HANDLING METRICS =========================
# =================================================================

print("\n" + "=" * 65)
print("=== 4. PENGUJIAN ERROR HANDLING METRICS ===")
print("=" * 65)


# =================================================================
# === TEST INPUT KOSONG ===========================================
# =================================================================

test_exception(
  "Calculate Metrics dengan list kosong",
  lambda: calculate_metrics(
    [],
    [],
  ),
)


# =================================================================
# === TEST PANJANG BERBEDA ========================================
# =================================================================

test_exception(
  "Calculate Metrics dengan panjang berbeda",
  lambda: calculate_metrics(
    [1.0, 2.0],
    [1.0],
  ),
)


print("\n" + "=" * 65)
print("Hasil: Seluruh pengujian Metrics LULUS!")
print("=" * 65)


# =================================================================
# === 5. PENGUJIAN MIN-MAX NORMALIZATION =========================
# =================================================================

print("\n" + "=" * 65)
print("=== 5. PENGUJIAN MIN-MAX NORMALIZATION ===")
print("=" * 65)


# =================================================================
# === DATA ========================================================
# =================================================================

values = [
  10.0,
  20.0,
  30.0,
  40.0,
  50.0,
]


# =================================================================
# === BUAT NORMALIZER =============================================
# =================================================================

normalizer = Normalizer()


# =================================================================
# === FIT =========================================================
# =================================================================

normalizer.fit(values)


print(f"Data    : {values}")
print(f"Minimum : {normalizer.minimum}")
print(f"Maximum : {normalizer.maximum}")


# =================================================================
# === TEST MINIMUM & MAXIMUM ======================================
# =================================================================

assert normalizer.minimum == 10.0
assert normalizer.maximum == 50.0


# =================================================================
# === 6. PENGUJIAN NORMALIZE ======================================
# =================================================================

print("\n" + "=" * 65)
print("=== 6. PENGUJIAN NORMALIZE ===")
print("=" * 65)


# =================================================================
# === NORMALIZE ===================================================
# =================================================================

normalized = normalize_list(
  values,
  normalizer,
)


print(f"Original  : {values}")
print(f"Normalized: {normalized}")


# =================================================================
# === HASIL YANG DIHARAPKAN =======================================
# =================================================================

expected_normalized = [
  0.0,
  0.25,
  0.5,
  0.75,
  1.0,
]


# =================================================================
# === TEST NORMALIZATION ==========================================
# =================================================================

assert all(
  math.isclose(actual, expected)
  for actual, expected
  in zip(
    normalized,
    expected_normalized,
  )
), "Pengujian Normalisasi Gagal!"


# =================================================================
# === 7. PENGUJIAN DENORMALIZE ====================================
# =================================================================

print("\n" + "=" * 65)
print("=== 7. PENGUJIAN DENORMALIZE ===")
print("=" * 65)


# =================================================================
# === DENORMALIZE =================================================
# =================================================================

denormalized = denormalize_list(
  normalized,
  normalizer,
)


print(f"Normalized  : {normalized}")
print(f"Denormalized: {denormalized}")


# =================================================================
# === TEST DENORMALIZATION ========================================
# =================================================================

assert all(
  math.isclose(actual, expected)
  for actual, expected
  in zip(
    denormalized,
    values,
  )
), "Pengujian Denormalisasi Gagal!"


# =================================================================
# === 8. PENGUJIAN ERROR HANDLING NORMALIZER ======================
# =================================================================

print("\n" + "=" * 65)
print("=== 8. PENGUJIAN ERROR HANDLING NORMALIZER ===")
print("=" * 65)


# =================================================================
# === TEST FIT DENGAN DATA KOSONG =================================
# =================================================================

test_exception(
  "Normalizer.fit dengan values kosong",
  lambda: Normalizer().fit([]),
)


# =================================================================
# === TEST FIT DENGAN NILAI SAMA ==================================
# =================================================================

test_exception(
  "Normalizer.fit dengan semua nilai sama",
  lambda: Normalizer().fit(
    [5.0, 5.0, 5.0],
  ),
)


# =================================================================
# === TEST NORMALIZE SEBELUM FIT ==================================
# =================================================================

test_exception(
  "Normalizer.normalize sebelum fit",
  lambda: Normalizer().normalize(5.0),
)


# =================================================================
# === TEST DENORMALIZE SEBELUM FIT ================================
# =================================================================

test_exception(
  "Normalizer.denormalize sebelum fit",
  lambda: Normalizer().denormalize(0.5),
)


print("\n" + "=" * 65)
print("Hasil: Seluruh pengujian Normalizer LULUS!")
print("=" * 65)


# =================================================================
# === FINAL =======================================================
# =================================================================

print("\n" + "=" * 65)
print("=== SEMUA PENGUJIAN BERHASIL ===")
print("=" * 65)

print("Loss Function : LULUS")
print("Metrics       : LULUS")
print("Normalizer    : LULUS")

print("=" * 65)