from beartype.roar import BeartypeCallHintParamViolation

from src.core import (
  Linear,
  ReLU,
  LeakyReLU,
)


# LINEAR

linear = Linear()

print("=== Linear ===")

print(linear(5.0))
print(linear.gradient(5.0))


# RELU

relu = ReLU()

print()
print("=== ReLU ===")

print(relu(5.0))
print(relu(-5.0))

print(relu.gradient(5.0))
print(relu.gradient(-5.0))


# LEAKY RELU

leaky_relu = LeakyReLU(alpha=0.01)

print()
print("=== LeakyReLU ===")

print(leaky_relu(5.0))
print(leaky_relu(-5.0))

print(leaky_relu.gradient(5.0))
print(leaky_relu.gradient(-5.0))


# VALIDATION

print()
print("=== Type Validation ===")

try:
  relu = ReLU()

  relu("Hello")
except BeartypeCallHintParamViolation:
  print("✓ Tipe yg dimasukan ke Argument tdk sesuai")


print()
print("=== Alpha Validation ===")

try:
  LeakyReLU(alpha=0.0)
except BeartypeCallHintParamViolation as e:
  print(f"Pesan Error : {e}")
  print()


try:
  activation = LeakyReLU(alpha=1.0)

  print(f"Isi Alpha: {activation.alpha}")
except BeartypeCallHintParamViolation as e:
  print(f"Pesan Error : {e}")
  print()


try:
  LeakyReLU(alpha=1.5)
except BeartypeCallHintParamViolation as e:
  print(f"Pesan Error : {e}")
  print()
