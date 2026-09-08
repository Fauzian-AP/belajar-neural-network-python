# Menampilkan hasil Pelatihan² Model dalam bentuk Grafik

import matplotlib.pyplot as plt   # Library Matplotlib
from pathlib import Path
from pydantic import validate_call

from custom_types import (
  FitHistory,
  ScaleType,
  EvaluatingType,
)

# Buat path spesifik untuk menyimpan file² image
PLOT_DIR = Path(__file__).parent / "plots"

# Buat otomatis folder 'plots' jika tdk ada
PLOT_DIR.mkdir(exist_ok=True)


# PLOT TRAINING — Grafik Proses Belajar

@validate_call
def plot_training(history: FitHistory, scale: ScaleType) -> None:
  # Ambil Epochs
  epochs = history["epochs"]

  # Ambil metrics
  training_metrics = history["training_metrics"]
  validating_metrics = history["validating_metrics"]

  # Ambil nama² Metrics
  metric_names = list(training_metrics.keys())

  # Jenis Scale Data yg dipakai
  scale_name = scale.value

  # Buat Plot pd tiap jenis Metrics
  for metric_name in metric_names:
    # Ambil data² Training
    training_values = (training_metrics[metric_name])
    
    # Ambil data² Validating
    validating_values = (validating_metrics[metric_name])
    
    plt.figure(figsize=(10, 6))
    
    # Training
    plt.plot(
      epochs,
      training_values,
      color="purple",
      label=f"Training {metric_name}"
    )
    
    # Validation
    plt.plot(
      epochs,
      validating_values,
      color="darkblue",
      label=f"Validating {metric_name}"
    )
    
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Training vs Validation — {metric_name} ({scale_name})")
    
    plt.legend()
    plt.grid()
    
    # Save & Replace File Image
    plt.savefig(
      PLOT_DIR / f"training_{metric_name.lower()}_{scale_name}.png",
      dpi=150,
      bbox_inches="tight",
    )
    
    plt.close()


# PLOT EVALUATING — Grafik Perbandingan Hasil Akhir

@validate_call
def plot_evaluating(evaluating_metrics: EvaluatingType, scale: ScaleType) -> None:
  # Jenis Scale Data yg dipakai
  scale_name = scale.value

  # Ambil nama² Dataset
  dataset_names = list(evaluating_metrics.keys())

  # Ambil isi data Evaluating yg pertama
  first_metrics = next(iter(evaluating_metrics.values()))

  # Ambil nama² Metrics
  metric_names = list(first_metrics.keys())

  # Buat Plot pd tiap jenis Metrics
  for metric_name in metric_names:
    # Ambil nilai² tiap Dataset
    values = [
      evaluating_metrics[name][metric_name]
      for name in dataset_names
    ]
  
    plt.figure(figsize=(10, 6))
  
    plt.bar(dataset_names, values)
  
    plt.xlabel("Dataset")
    plt.ylabel(metric_name)
  
    plt.title(f"Final Evaluating — {metric_name} ({scale_name})")
  
    plt.grid(axis="y")
  
    # Save & Replace File Image
    plt.savefig(
      PLOT_DIR / f"evaluating_{metric_name.lower()}_{scale_name}.png",
      dpi=150,
      bbox_inches="tight",
    )
  
    plt.close()