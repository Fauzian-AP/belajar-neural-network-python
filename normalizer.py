# Normalisasi Data merupakan proses mengatur dan menstrukturkan data agar lebih rapi

# Teknik Min Max Normalization
# Sehingga hasilnya itu merupakan rentan nilai antara 0 sampai 1

# normalized = (x - min) / (max - min)

class Normalizer:
  # CONSTRUCTOR — Initialization
  def __init__(self):
    self.minimum = None
    self.maximum = None

  # FIT — Menetukan nilai batas dari kumpulan data
  def fit(self, values):
    self.minimum = min(values)
    self.maximum = max(values)

  # NORMALIZE — Menormalisasikan sebuah data
  def normalize(self, value):
    return (
      (value - self.minimum) / (self.maximum - self.minimum)
    )

  # DENORMALIZE — Mengembalikan sebuah data yg telah di Normalisasi
  def denormalize(self, value):
    return (
      value * (self.maximum - self.minimum) + self.minimum
    )

# Fungsi Helper Normalisasi List Data
def normalize_list(values, normalizer):
  return [
    normalizer.normalize(value)
    for value in values
  ]

# Fungsi Helper Denomalisasi List Data
def denormalize_list(values, normalizer):
  return [
    normalizer.denormalize(value)
    for value in values
  ]

# Fungsi Utama Normalisasi sebuah Dataset
def normalize_dataset(
  dataset,
  input_normalizer,
  target_normalizer,
):
  normalized_dataset = []

  for inputs, targets in dataset:
    normalized_inputs = normalize_list(inputs, input_normalizer)
    normalized_targets = normalize_list(targets, target_normalizer)

    normalized_dataset.append((normalized_inputs, normalized_targets))

  return normalized_dataset

# Fungsi Utama Normalisasi byk Dataset sekaligus
def normalize_datasets(
  datasets,
  input_normalizer,
  target_normalizer,
):
  normalized_datasets = {}

  for name, dataset in datasets.items():
    normalized_datasets[name] = (
      normalize_dataset(dataset, input_normalizer, target_normalizer)
    )

  return normalized_datasets