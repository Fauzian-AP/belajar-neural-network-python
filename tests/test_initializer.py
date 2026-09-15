from src.core import HeNormal

# HE NORMAL

initializer = HeNormal()

print("=== HeNormal ===")

weights = initializer(3)

print(f"Weights : {weights}")
print(f"Length  : {len(weights)}")

# INPUT SIZE VALIDATION

print()
print("=== Input Size Validation ===")

try:
  initializer(0)
except Exception as error:
  print(type(error).__name__)

try:
  initializer(-3)
except Exception as error:
  print(type(error).__name__)