from src.preprocessing.normalization import (
  Normalizer,
  MinMaxNormalizer,
)


# FIT

normalizer = MinMaxNormalizer()

normalizer.fit(
  [10.0, 20.0, 30.0, 40.0, 50.0]
)

assert normalizer.minimum == 10.0
assert normalizer.maximum == 50.0
assert normalizer._is_fitted is True

print("✓ Fit MinMaxNormalizer berhasil\n")


# NORMALIZE

normalized = normalizer.normalize(30.0)

assert normalized == 0.5

print("✓ Normalize Berhasil\n")


# DENORMALIZE

original = normalizer.denormalize(0.5)

assert original == 30.0

print("✓ Denormalize Berhasil\n")


# BOUNDARY

assert normalizer.normalize(10.0) == 0.0
assert normalizer.normalize(50.0) == 1.0

assert normalizer.denormalize(0.0) == 10.0
assert normalizer.denormalize(1.0) == 50.0

print("✓ Boundary Min-Max Berhasil\n")


# NOT FITTED

not_fitted = MinMaxNormalizer()

try:
  not_fitted.normalize(10.0)
except Exception as error:
  print(f"Pesan Error: {error}")
finally:
  print()

try:
  not_fitted.denormalize(0.5)
except Exception as error:
  print(f"Pesan Error: {error}")
finally:
  print()


# EMPTY VALUES

try:
  MinMaxNormalizer().fit([])
except Exception as error:
  print(f"Pesan Error: {error}")
finally:
  print()


# SAME MINIMUM & MAXIMUM

try:
  MinMaxNormalizer().fit(
    [5.0, 5.0, 5.0]
  )
except Exception as error:
  print(f"Pesan Error: {error}")
finally:
  print()


# INVALID TYPE

try:
  MinMaxNormalizer().fit(
    ["10.0", "50.0"]
  )
except Exception as error:
  print(f"Pesan Error: {error}")
finally:
  print()


# NONE MINIMUM & MAXIMUM

none_fitted = MinMaxNormalizer()

# Kosongkan dgn Sengaja
none_fitted._is_fitted = True
none_fitted.minimum = 10.0
none_fitted.maximum = None

try:
  none_fitted.normalize(15.0)
except Exception as error:
  print(f"Pesan Error: {error}")
finally:
  print()

try:
  none_fitted.denormalize(0.125)
except Exception as error:
  print(f"Pesan Error: {error}")
finally:
  print()


# ABSTRACT CLASS

try:
  Normalizer()
except Exception as error:
  print(f"Pesan Error: {error}")
finally:
  print()


# RESULT

print("Semua Test Normalizer Berhasil.")