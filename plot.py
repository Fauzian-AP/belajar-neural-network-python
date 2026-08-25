# Menampilkan hasil Training / Validation Model dalam bentuk Grafik

import matplotlib.pyplot as plt   # Library Matplotlib
from pathlib import Path

# Buat path spesifik untuk menyimpan file² image
PLOT_DIR = Path(__file__).parent / "plots"

# Buat otomatis folder 'plots' jika tdk ada
PLOT_DIR.mkdir(exist_ok=True)

def plot_history(
  epoch,
  training_data,
  validation_data
):
  # AMBIL DATA² TRAINING METRICS

  training_mse = [
    metrics["mse"]
    for metrics in training_data
  ]

  training_mae = [
    metrics["mae"]
    for metrics in training_data
  ]

  training_rmse = [
    metrics["rmse"]
    for metrics in training_data
  ]

  # AMBIL DATA² VALIDATION METRICS

  validation_mse = [
    metrics["mse"]
    for metrics in validation_data
  ]

  validation_mae = [
    metrics["mae"]
    for metrics in validation_data
  ]

  validation_rmse = [
    metrics["rmse"]
    for metrics in validation_data
  ]

  # PLOT MSE
  
  plt.figure(figsize=(10, 6))

  # Training
  plt.plot(
    epoch,
    training_mse,
    color="purple",
    label="Training MSE"
  )

  # Validation
  plt.plot(
    epoch,
    validation_mse,
    color="darkblue",
    label="Validation MSE"
  )

  plt.xlabel("Epoch")
  plt.ylabel("MSE")
  plt.title("Training vs Validation — MSE")

  plt.legend()
  plt.grid()

  # Save & Replace File Image
  plt.savefig(
    PLOT_DIR / "training_mse.png",
    dpi=150,
    bbox_inches="tight"
  )

  plt.close()

  # PLOT MAE

  plt.figure(figsize=(10, 6))

  # Training
  plt.plot(
    epoch,
    training_mae,
    color="purple",
    label="Training MAE"
  )

  # Validation
  plt.plot(
    epoch,
    validation_mae,
    color="darkblue",
    label="Validation MAE"
  )

  plt.xlabel("Epoch")
  plt.ylabel("MAE")
  plt.title("Training vs Validation — MAE")

  plt.legend()
  plt.grid()

  # Save & Replace File Image
  plt.savefig(
    PLOT_DIR / "training_mae.png",
    dpi=150,
    bbox_inches="tight"
  )

  plt.close()

  # PLOT RMSE

  plt.figure(figsize=(10, 6))

  # Training
  plt.plot(
    epoch,
    training_rmse,
    color="purple",
    label="Training RMSE"
  )

  # Validation
  plt.plot(
    epoch,
    validation_rmse,
    color="darkblue",
    label="Validation RMSE"
  )

  plt.xlabel("Epoch")
  plt.ylabel("RMSE")
  plt.title("Training vs Validation — RMSE")

  plt.legend()
  plt.grid()

  # Save & Replace File Image
  plt.savefig(
    PLOT_DIR / "training_rmse.png",
    dpi=150,
    bbox_inches="tight"
  )

  plt.close()