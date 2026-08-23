from model import Model
from trainer import Trainer

# =========================
# MODEL
# =========================

model = Model(
  learning_rate=0.001
)

trainer = Trainer(model)

# =========================
# DATASET KECIL
# =========================

training_data = [
  ([0.2, 0.4, 0.6], [0.3, 0.7]),
  ([0.4, 0.6, 0.8], [0.5, 0.9]),
]

# Validation sengaja dibuat sama
# dengan training data.
#
# Tujuannya bukan menguji generalisasi,
# tetapi melihat apakah model mampu
# menghafal data.

validation_data = training_data.copy()

# =========================
# TRAINING
# =========================

print("=== OVERFITTING TEST ===")

trainer.fit(
  training_data,
  validation_data,
  epochs=5000,
  batch_size=2
)

# =========================
# FINAL PREDICTION
# =========================

print("=== FINAL ===")

for inputs, targets in training_data:

  prediction = model.forward(inputs)

  print()
  print("Input      :", inputs)
  print("Prediction :", prediction)
  print("Target     :", targets)