# Bagian Pengelolaan Data² yg akan digunakan dlm melatih Model


# ====================
# === TYPE CHECKER ===
# ====================

Dataset = list[
  tuple[
    list[float],
    list[float],
  ]
]


# ============
# === DATA ===
# ============

# TRAINING --> Model belajar dari data ini
training_data: Dataset = [
  ([1, 2, 3], [10, 20]),
  ([2, 4, 6], [20, 40]),
  ([3, 6, 9], [30, 60]),
  ([4, 8, 12], [40, 80]),
  ([5, 10, 15], [50, 100]),
  ([6, 12, 18], [60, 120]),
  ([7, 14, 21], [70, 140]),
  ([8, 16, 24], [80, 160]),
  ([9, 18, 27], [90, 180]),
  ([10, 20, 30], [100, 200]),
]

# VALIDATION --> Mengecek perkembangan model selama eksperimen
validation_data: Dataset = [
  ([11, 22, 33], [110, 220]),
  ([12, 24, 36], [120, 240]),
  ([13, 26, 39], [130, 260]),
]

# TESTING --> Evaluasi final pd data yg tdk digunakan untuk mengambil keputusan
test_data: Dataset = [
  ([14, 28, 42], [140, 280]),
  ([15, 30, 45], [150, 300]),
  ([16, 32, 48], [160, 320]),
]

# GENERALIZATION --> Menguji kemampuan Model pd input baru yg masih mengikuti pola data
generalization_data: Dataset = [
  ([5, 10, 15], [50, 100]),
  ([5.5, 11, 16.5], [55, 110]),
  ([7.5, 15, 22.5], [75, 150]),
  ([9.5, 19, 28.5], [95, 190]),
]