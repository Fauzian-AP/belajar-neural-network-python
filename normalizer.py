# Normalisasi Data merupakan proses mengatur dan menstrukturkan data agar lebih rapi.

# Teknik: Min Max Normalization
# Rumus: normalized = (x - min) / (max - min)
# Sehingga hasilnya itu merupakan rentan nilai antara 0 dgn 1

class Normalizer:
  def __init__(self):
    self.minimum = None
    self.maximum = None

  def fit(self, values):
    self.minimum = min(values)
    self.maximum = max(values)

  def normalize(self, value):
    return (
      (value - self.minimum) / (self.maximum - self.minimum)
    )
    
  def denormalize(self, value):
    return (
      value * (self.maximum - self.minimum) + self.minimum
    )

# Penanganan jika berbentuk normalize_list

def normalize_list(values, normalizer):
  return [
    normalizer.normalize(value)
    for value in values
  ]

def denormalize_list(values, normalizer):
  return [
    normalizer.denormalize(value)
    for value in values
  ]