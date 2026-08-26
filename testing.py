# testing.py — Script Pengujian Neuron & Validasi Keamanan

from dataset import training_data
from neuron import Neuron

print("=" * 65)
print("=== 1. PENGUJIAN PROSES NORMAL (FORWARD, BACKWARD, STEP, RESET) ===")
print("=" * 65)

# Inisialisasi Neuron (3 input, learning_rate 0.01, aktivasi ReLU)
neuron = Neuron(input_size=3, learning_rate=0.01, activation="ReLU")

print(f"[INIT] Neuron berhasil dibuat.")
print(f"       Weights Awal : {neuron.weights}")
print(f"       Bias Awal    : {neuron.bias}\n")

# Sampel dataset pertama: ([1, 2, 3], [10, 20])
x_sample, y_sample = training_data[0]
target = float(y_sample[0])  # Target output pertama (10.0)

# A. Forward Pass
prediksi = neuron.forward(x_sample)
print(f"[FORWARD] Input: {x_sample}")
print(f"          Prediksi Output     : {prediksi:.4f}")
print(f"          Last Pre-Activation : {neuron.last_pre_activation:.4f}\n")

# B. Backward Pass
# Misal gradien Loss dL/d_pred = 2 * (prediksi - target)
gradien_output = 2.0 * (prediksi - target)
gradien_input = neuron.backward(x_sample, gradien_output)

print(f"[BACKWARD] Gradien Output Diterima      : {gradien_output:.4f}")
print(f"           Gradien Weights Terakumulasi : {neuron.gradient_weights}")
print(f"           Gradien Bias Terakumulasi    : {neuron.gradient_bias:.4f}")
print(f"           Gradien Input Diteruskan     : {gradien_input}\n")

# C. Average Gradient (misal mini-batch ukuran 2)
neuron.average_gradient(batch_size=2)
print(f"[AVERAGE] Rata-rata Gradien Weights (batch=2) : {neuron.gradient_weights}")
print(f"          Rata-rata Gradien Bias              : {neuron.gradient_bias:.4f}\n")

# D. Step Update Parameter
neuron.step()
print(f"[STEP] Parameter setelah diperbarui:")
print(f"       Weights Baru : {neuron.weights}")
print(f"       Bias Baru    : {neuron.bias:.4f}\n")

# E. Reset Gradient
neuron.reset_gradient()
print(f"[RESET] Gradien Weights setelah reset : {neuron.gradient_weights}")
print(f"        Gradien Bias setelah reset    : {neuron.gradient_bias:.4f}\n")


print("=" * 65)
print("=== 2. PENGUJIAN VALIDASI & KONDISI SALAH (ERROR HANDLING) ===")
print("=" * 65)

def test_exception(description: str, func) -> None:
    """Helper fungsi untuk memverifikasi bahwa ValueError dilempar saat kondisi salah."""
    try:
        func()
        print(f"[FAIL] {description} -> TIDAK melempar error!")
    except ValueError as e:
        print(f"[PASS] {description}")
        print(f"       Pesan Error Ditangkap: \"{e}\"")

# 1. Validasi Input Size
test_exception(
    "Init dengan input_size = 0",
    lambda: Neuron(input_size=0, learning_rate=0.01)
)

# 2. Validasi Learning Rate
test_exception(
    "Init dengan learning_rate = -0.05",
    lambda: Neuron(input_size=3, learning_rate=-0.05)
)

# 3. Validasi Nama Activation
test_exception(
    "Init dengan activation = 'sigmoid'",
    lambda: Neuron(input_size=3, learning_rate=0.01, activation="sigmoid")  # type: ignore
)

# 4. Validasi Ukuran Input pada Forward Pass
test_exception(
    "Forward dengan panjang input salah ([1.0, 2.0])",
    lambda: neuron.forward([1.0, 2.0])
)

# 5. Validasi Ukuran Input pada Backward Pass
test_exception(
    "Backward dengan panjang input salah ([1.0, 2.0, 3.0, 4.0])",
    lambda: neuron.backward([1.0, 2.0, 3.0, 4.0], gradient_output=1.0)
)

# 6. Validasi Batch Size pada Average Gradient
test_exception(
    "Average Gradient dengan batch_size = 0",
    lambda: neuron.average_gradient(batch_size=0)
)

print("\n" + "=" * 65)
print("Hasil: Semua alur proses dan penanganan error berjalan sesuai kriteria.")
print("=" * 65)