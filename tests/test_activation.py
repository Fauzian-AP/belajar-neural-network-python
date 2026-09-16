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


# ALPHA VALIDATION

print()
print("=== Alpha Validation ===")

try:
  LeakyReLU(alpha=0.0)
except Exception as error:
  print(type(error).__name__)

try:
  lr = LeakyReLU(alpha=1.0)

  print(f"Isi Alpha: {lr.alpha}")
except Exception as error:
  print(type(error).__name__)

try:
  LeakyReLU(alpha=1.5)
except Exception as error:
  print(type(error).__name__)