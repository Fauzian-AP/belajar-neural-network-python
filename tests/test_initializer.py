from src.core.initializer import HeNormal


# HE NORMAL

initializer = HeNormal()

print("=== HeNormal ===")

weights = initializer(3)

print(f"Weights : {weights}")
print(f"Length  : {len(weights)}")


# VALIDATION

print()
print("=== Wrong Type Validation ===")

try:
  initializer("Test")
except Exception as error:
  print(f"Pesan Error : {error}")
finally:
  print()


print("=== Input Size Validation ===")

try:
  initializer(0)
except Exception as error:
  print(f"Pesan Error : {error}")
finally:
  print()


try:
  initializer(-3)
except Exception as error:
  print(f"Pesan Error : {error}")
finally:
  print()