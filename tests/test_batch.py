# Test untuk Batch

from src.training.batch import Batch


# =================================================================
# === DATASET =====================================================
# =================================================================

dataset = [
  ([1.0, 2.0, 3.0], [4.0, 5.0]),
  ([2.0, 3.0, 4.0], [5.0, 6.0]),
  ([3.0, 4.0, 5.0], [6.0, 7.0]),
  ([4.0, 5.0, 6.0], [7.0, 8.0]),
  ([5.0, 6.0, 7.0], [8.0, 9.0]),
  ([6.0, 7.0, 8.0], [9.0, 10.0]),
  ([7.0, 8.0, 9.0], [10.0, 11.0]),
  ([8.0, 9.0, 10.0], [11.0, 12.0]),
  ([9.0, 10.0, 11.0], [12.0, 13.0]),
  ([10.0, 11.0, 12.0], [13.0, 14.0]),
]


# =================================================================
# === SPLIT DATASET ===============================================
# =================================================================

batch_1 = Batch(seed=42)

result_1 = batch_1.split_dataset(
  dataset,
  training_ratio=0.6,
  validating_ratio=0.2,
  testing_ratio=0.2,
)

assert len(result_1["training"]) == 6
assert len(result_1["validating"]) == 2
assert len(result_1["testing"]) == 2

assert (
  len(result_1["training"])
  + len(result_1["validating"])
  + len(result_1["testing"])
  == len(dataset)
)

print("✓ Split Dataset berhasil")


# =================================================================
# === REPRODUCIBILITY =============================================
# =================================================================

batch_2 = Batch(seed=42)

result_2 = batch_2.split_dataset(
  dataset,
  training_ratio=0.6,
  validating_ratio=0.2,
  testing_ratio=0.2,
)

assert result_1 == result_2

print("✓ Reproducibility berhasil")


# =================================================================
# === ORIGINAL DATASET ============================================
# =================================================================

original_dataset = list(dataset)

Batch(seed=42).split_dataset(
  dataset,
  training_ratio=0.6,
  validating_ratio=0.2,
  testing_ratio=0.2,
)

assert dataset == original_dataset

print("✓ Dataset asli tidak berubah")


# =================================================================
# === CREATE BATCHES ==============================================
# =================================================================

batches = Batch(seed=42).create_batches(
  dataset,
  batch_size=3,
)

assert [len(batch) for batch in batches] == [
  3,
  3,
  3,
  1,
]

assert (
  sum(len(batch) for batch in batches)
  == len(dataset)
)

print("✓ Create Batches berhasil")


# =================================================================
# === BATCH REPRODUCIBILITY =======================================
# =================================================================

batches_1 = Batch(seed=42).create_batches(
  dataset,
  batch_size=3,
)

batches_2 = Batch(seed=42).create_batches(
  dataset,
  batch_size=3,
)

assert batches_1 == batches_2

print("✓ Batch Reproducibility berhasil")


# =================================================================
# === INVALID BATCH SIZE ==========================================
# =================================================================

try:
  Batch(seed=42).create_batches(
    dataset,
    batch_size=0,
  )

except Exception as error:
  print(
    f"Pesan Error: {error}"
  )


# =================================================================
# === INVALID RATIO ===============================================
# =================================================================

try:
  Batch(seed=42).split_dataset(
    dataset,
    training_ratio=0.5,
    validating_ratio=0.3,
    testing_ratio=0.3,
  )

except ValueError as error:
  print(
    f"Pesan Error: {error}"
  )


# =================================================================
# === INVALID POSITIVE RATIO ======================================
# =================================================================

try:
  Batch(seed=42).split_dataset(
    dataset,
    training_ratio=0.0,
    validating_ratio=0.5,
    testing_ratio=0.5,
  )

except Exception as error:
  print(
    f"Pesan Error: {error}"
  )


# =================================================================
# === INVALID TYPE ================================================
# =================================================================

try:
  Batch(seed=42).create_batches(
    dataset,
    batch_size="3",
  )

except Exception as error:
  print(
    f"Pesan Error: {error}"
  )


print()
print("Semua test Batch berhasil.")