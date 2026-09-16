from beartype.roar import BeartypeCallHintParamViolation

from src.core import HeNormal


# HE NORMAL

initializer = HeNormal()

print("=== HeNormal ===")

weights = initializer(3)

print(f"Weights : {weights}")
print(f"Length  : {len(weights)}")


# VALIDATION

print("=== Wrong Type Validation ===")

try:
  initializer("Test")
except BeartypeCallHintParamViolation as e:
  print(f"Pesan Error : {e}")
  print()


print("=== Input Size Validation ===")

try:
  initializer(0)
except BeartypeCallHintParamViolation as e:
  print(f"Pesan Error : {e}")
  print()


try:
  initializer(-3)
except BeartypeCallHintParamViolation as e:
  print(f"Pesan Error : {e}")
  print()