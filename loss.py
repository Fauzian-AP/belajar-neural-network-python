# Bagian untuk menghitung Loss, yaitu:
# Nilai numeric yg digunakan untukbmengukur seberapa besar tingkat kesalahan dlm prediksi model

# Gunakan MSE (Mean Squared Error)

def mse(targets, predictions):
  total = 0

  for target, prediction in zip(targets, predictions):
    total += (target - prediction) ** 2

  return total / len(targets)

def mse_gradient(targets, predictions):
  gradients = []

  for target, prediction in zip(targets, predictions):
    gradients.append(
      # Rumus Gradient MSE
      2 * (prediction - target)
    )

  return gradients