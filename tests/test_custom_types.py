import numpy as np

from src.utils.custom_types import (
  IntVector,
  FloatVector,
  FloatArray,
  IntPositive,
  FloatPositive,
  FloatSequence,
  AlphaRange,
  DataSample,
  Dataset,
  DatasetList,
  DatasetType,
  Metrics,
  EvaluatingType,
  CacheNeuron,
  MetricsHistory,
  FitHistory,
  FitResult,
)

try:
  # TYPES

  int_vector: IntVector = [1, 2, 3]
  float_vector: FloatVector = [1.0, 2.0, 3.0]

  print("✓ Basic Types berhasil\n")


  # NUMPY FLOAT ARRAY

  float_array: FloatArray = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

  assert isinstance(float_array, np.ndarray)
  assert float_array.dtype == np.float64
  assert float_array.shape == (3,)

  print("✓ FloatArray NumPy berhasil\n")


  # DATA SAMPLE

  sample = DataSample(
    inputs=[1.0, 2.0, 3.0],
    targets=[4.0, 5.0],
  )

  assert sample.inputs == [1.0, 2.0, 3.0]
  assert sample.targets == [4.0, 5.0]

  print("✓ DataSample berhasil\n")


  # DATA SAMPLE ATTRIBUTE ACCESS

  assert sample.inputs[0] == 1.0
  assert sample.targets[0] == 4.0

  print("✓ DataSample Attribute Access berhasil\n")


  # DATA SAMPLE UNPACKING 

  inputs, targets = sample

  assert inputs == [1.0, 2.0, 3.0]
  assert targets == [4.0, 5.0]

  print("✓ DataSample Unpacking berhasil\n")


  # DATASET

  dataset: Dataset = [
    DataSample(
      inputs=[1.0, 2.0, 3.0],
      targets=[4.0, 5.0],
    ),
    DataSample(
      inputs=[6.0, 7.0, 8.0],
      targets=[9.0, 10.0],
    ),
  ]

  assert len(dataset) == 2

  for inputs, targets in dataset:
    assert len(inputs) == 3
    assert len(targets) == 2

  print("✓ Dataset berhasil\n")


  # DATASET TYPE

  datasets: DatasetType = {
    "training": dataset,
    "validation": dataset,
    "testing": dataset,
  }

  assert "training" in datasets
  assert "validation" in datasets
  assert "testing" in datasets

  list_dataset: DatasetList = [dataset, dataset]

  assert len(list_dataset) == 2

  print("✓ DatasetType & ListDataset berhasil\n")


  # EVALUATING TYPES

  metrics: Metrics = {
    "MSE": 0.1,
    "MAE": 0.2,
    "RMSE": 0.3,
  }

  evaluating: EvaluatingType = {
    "training": metrics,
    "validation": metrics,
  }

  assert evaluating["training"]["MSE"] == 0.1

  print("✓ Evaluating Types berhasil\n")


  # CACHE NEURON

  cache: CacheNeuron = {
    "inputs": [1.0, 2.0, 3.0],
    "pre_activation": 4.5,
  }

  assert cache["inputs"] == [1.0, 2.0, 3.0]
  assert cache["pre_activation"] == 4.5

  print("✓ CacheNeuron berhasil\n")


  # HISTORY TYPES

  metrics_history: MetricsHistory = {
    "MSE": [0.5, 0.3, 0.1],
    "MAE": [0.4, 0.2, 0.1],
    "RMSE": [0.7, 0.5, 0.3],
  }

  fit_history: FitHistory = {
    "epochs": [1, 2, 3],
    "training_metrics": metrics_history,
    "validating_metrics": metrics_history,
  }

  fit_result: FitResult = {
    "history": fit_history,
    "best_epoch": 3,
    "last_epoch": 3,
    "best_validating_mse": 0.1,
  }

  assert fit_result["best_epoch"] == 3
  assert fit_result["best_validating_mse"] == 0.1

  print("✓ Training History Types berhasil\n")


  # NUMERIC CONSTRAINT TYPES

  positive_int: IntPositive = 10
  positive_float: FloatPositive = 0.5
  float_sequence: FloatSequence = [1.0, 2.0, 3.0]
  alpha: AlphaRange = 0.01

  assert positive_int > 0
  assert positive_float > 0.0
  assert bool(float_sequence)
  assert 0.0 < alpha < 1.0

  print("✓ Numeric Constraint Types berhasil\n")


  # FINISH

  print("Semua Test Custom Types Berhasil.")
except Exception as error:
  print(f"Pesan Error ({type(error).__name__}) : {error}")