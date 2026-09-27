"""
Bagian Pengelolaan Activation pada Project Neural Network.

Activation Function merupakan fungsi matematis yang digunakan untuk mengubah nilai Pre-Activation (z) menjadi Output Activation.

Activation digunakan setelah proses:
  z = Σ(xᵢwᵢ) + b

Kemudian:
  a = f(z)

dengan:
  z = Pre-Activation
  a = Output Activation
  x = Input
  w = Weight
  b = Bias
  f = Activation Function
"""

import numpy as np

from abc import ABC, abstractmethod

from src.validation import runtime_method

from src.utils.custom_types import (
  Float,
  FloatArray,
  AlphaRange,
)


# ============================
# === BLUEPRINT ACTIVATION ===
# ============================

class Activation(ABC):
  # CALL — Menghitung Output Activation Function berdasarkan nilai Pre-Activation.
  @abstractmethod
  def __call__(self, pre_activation: FloatArray) -> FloatArray:
    # Bangkitkan Error
    raise NotImplementedError("Subclass harus mengimplementasikan method __call__().")

  # GRADIENT — Menghitung turunan Activation Function terhadap Pre-Activation.
  @abstractmethod
  def gradient(self, pre_activation: FloatArray) -> FloatArray:
    # Bangkitkan Error
    raise NotImplementedError("Subclass harus mengimplementasikan method gradient().")


# ==========================
# === METODE² ACTIVATION ===
# ==========================


class Linear(Activation):
  """
  Linear Activation Function.

  Karakteristik:
    Meneruskan nilai Pre-Activation tanpa perubahan sehingga tidak memberikan non-linearitas.

  Penggunaan:
    - Output Regression.
    - Model yang membutuhkan Output kontinu tanpa batas.

  Kelebihan:
    - Sederhana.
    - Tidak mengalami Saturation.
    - Gradient selalu konstan sehingga tidak mengalami Vanishing Gradient akibat fungsi activation.

  Kekurangan:
    - Tidak menghasilkan non-linearitas.
    - Jika digunakan pada seluruh Hidden Layer, kemampuan Model tetap bersifat linear.

  Rumus:
    f(z) = z

  Keterangan:
    f(z) = Output Activation
    z    = Pre-Activation
  """

  @runtime_method()
  def __call__(self, pre_activation: FloatArray) -> FloatArray:
    """
    Menghasilkan Output Linear.

    Rumus:
      a = f(z) = z

    Dengan:
      a = Output Activation
      z = Pre-Activation

    Linear tidak mengubah nilai maupun bentuk data, sehingga dapat langsung dikembalikan.
    """

    return pre_activation

  @runtime_method()
  def gradient(self, pre_activation: FloatArray) -> FloatArray:
    """
    Menghasilkan Gradient Linear.

    Rumus:
      f'(z) = 1

    Dengan:
      z = Pre-Activation

    Gradient selalu bernilai 1 pada setiap element.
    """

    return np.ones_like(pre_activation)


class ReLU(Activation):
  """
  Rectified Linear Unit.

  Karakteristik:
    Meneruskan nilai positif dan mengubah nilai negatif menjadi 0.

  Penggunaan:
    Hidden Layer pada Neural Network, terutama ketika dibutuhkan fungsi activation non-linear yang sederhana dan efisien.

  Kelebihan:
    - Sangat sederhana dan cepat.
    - Tidak mengalami Saturation pada sisi positif.
    - Membantu menghasilkan representasi non-linear.

  Kekurangan:
    - Dapat mengalami Dying ReLU.
    - Neuron yang terus menerima nilai negatif dapat memiliki Gradient 0.

  Rumus:
    f(z) = max(0, z)

  Keterangan:
    z = Pre-Activation
    f(z) = Output Activation
  """

  @runtime_method()
  def __call__(self, pre_activation: FloatArray) -> FloatArray:
    """
    Menghasilkan Output ReLU.

    Rumus:
      f(z) = max(0, z)

    Nilai positif diteruskan, sedangkan nilai negatif menjadi 0.
    """

    return np.maximum(pre_activation, 0.0)

  @runtime_method()
  def gradient(self, pre_activation: FloatArray) -> FloatArray:
    """
    Menghasilkan Gradient ReLU.

    Rumus:
      f'(z) = { 1, jika z > 0
              { 0, jika z <= 0

    Dengan:
      z = Pre-Activation

    Pada z = 0, implementasi ini menetapkan Gradient = 0.
    """

    return np.where(pre_activation > 0.0, 1.0, 0.0)


class LeakyReLU(Activation):
  """
  Leaky Rectified Linear Unit (Leaky ReLU).

  Karakteristik:
    Merupakan pengembangan ReLU yang tetap meneruskan sebagian kecil nilai negatif melalui parameter alpha.

  Penggunaan:
    Hidden Layer ketika ingin mengurangi masalah Dying ReLU.

  Kelebihan:
    - Nilai negatif tidak langsung menjadi 0.
    - Gradient pada sisi negatif tetap tersedia.
    - Mengurangi risiko neuron berhenti belajar akibat Gradient = 0 pada seluruh area negatif.

  Kekurangan:
    - Memperkenalkan hyperparameter alpha.
    - Nilai negatif tetap dapat menjadi sangat kecil sehingga kontribusinya terbatas.

  Rumus:
      f(z) = { z,  jika z > 0
             { αz, jika z <= 0

  Keterangan:
    z = Pre-Activation
    α = Faktor kemiringan pada sisi negatif
    f(z) = Output Activation
  """

  # INIT — Constructor untuk Initialization.
  @runtime_method()
  def __init__(self, alpha: AlphaRange = 0.01) -> None:
    # Nilai kemiringan pada nilai negatif.
    self._alpha: Float = alpha

  # ALPHA — Getter nilai alpha yang digunakan.
  @property
  def alpha(self) -> float:
    return self._alpha

  @runtime_method()
  def __call__(self, pre_activation: FloatArray) -> FloatArray:
    """
    Menghasilkan Output Leaky ReLU.

    Rumus:
      f(z) = { z,  jika z > 0
             { αz, jika z <= 0

    Dengan:
      z = Pre-Activation
      α = Alpha
      f(z) = Output Activation
    """

    return np.where(
      pre_activation > 0.0,   # Condition
      pre_activation,   # Success
      self._alpha * pre_activation,   # Failed
    )

  @runtime_method()
  def gradient(self, pre_activation: FloatArray) -> FloatArray:
    """
    Menghasilkan Gradient Leaky ReLU.

    Rumus:
      f'(z) = { 1, jika z > 0
              { α, jika z <= 0

    Dengan:
      z = Pre-Activation
      α = Alpha
    """

    return np.where(pre_activation > 0.0, 1.0, self._alpha)


class Sigmoid(Activation):
  """
  Sigmoid / Logistic Activation Function.

  Karakteristik:
    Mengubah nilai input menjadi rentang 0 hingga 1.

  Penggunaan:
    - Binary Classification.
    - Output yang dapat ditafsirkan sebagai probabilitas untuk kelas biner.

  Kelebihan:
    - Output berada pada rentang 0 sampai 1.
    - Mudah ditafsirkan sebagai probabilitas.

  Kekurangan:
    - Mengalami Saturation pada nilai sangat positif atau sangat negatif.
    - Dapat menyebabkan Vanishing Gradient.

  Rumus:
    σ(z) = 1 / (1 + e^(-z))

  Keterangan:
    z = Pre-Activation
    e = Bilangan Euler
    σ(z) = Output Sigmoid
  """

  @runtime_method()
  def __call__(self, pre_activation: FloatArray) -> FloatArray:
    """
    Menghasilkan Output Sigmoid.

    Rumus:
      σ(z) = 1 / (1 + e^(-z))

    Untuk menjaga stabilitas numerik, nilai z dibatasi sebelum operasi exponential sehingga tidak mengalami Overflow pada nilai ekstrem.
    """

    clipped = np.clip(pre_activation, -709.0, 709.0)

    return 1.0 / (1.0 + np.exp(-clipped))

  @runtime_method()
  def gradient(self, pre_activation: FloatArray) -> FloatArray:
    """
    Menghasilkan Gradient Sigmoid.

    Rumus:
      σ'(z) = σ(z)(1 - σ(z))

    Dengan:
      σ(z) = Output Sigmoid
      z = Pre-Activation

    Output Sigmoid diperoleh kembali melalui __call__() agar satu implementasi rumus digunakan secara konsisten.
    """

    activated = self(pre_activation)

    return activated * (
      1.0 - activated
    )


class Tanh(Activation):

  """
  Hyperbolic Tangent (Tanh).

  Karakteristik:
      Menghasilkan output pada rentang (-1, 1)
      dan berpusat di sekitar 0.

  Digunakan untuk:
      Hidden Layer ketika representasi yang berpusat
      di sekitar 0 diperlukan.

  Kelebihan:
      - Zero-centered.
      - Memiliki output yang simetris terhadap 0.

  Kekurangan:
      - Mengalami Saturation pada nilai ekstrem.
      - Dapat mengalami Vanishing Gradient.

  Rumus:
      tanh(z) =
          (e^z - e^(-z))
          ----------------
          (e^z + e^(-z))

  Keterangan:
      z = Pre-Activation
      e = Bilangan Euler
  """

  @runtime_method()
  def __call__(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:

    """
    Menghasilkan Output Tanh.

    Rumus:
        a = tanh(z)

    Output berada pada rentang:
        -1 < a < 1
    """

    return np.tanh(
      pre_activation,
    )

  @runtime_method()
  def gradient(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:

    """
    Menghasilkan Gradient Tanh.

    Rumus:
        tanh'(z) = 1 - tanh²(z)

    Dengan:
        z = Pre-Activation
        tanh(z) = Output Tanh
    """

    activated = self(pre_activation)

    return 1.0 - (
      activated ** 2
    )


class SoftMax(Activation):

  """
  Softmax Activation Function.

  Karakteristik:
      Mengubah sekumpulan nilai logit menjadi distribusi
      probabilitas.

      Untuk setiap sample:
          Σ pᵢ = 1

      dan:
          0 < pᵢ < 1

  Digunakan untuk:
      - Multi-Class Classification.
      - Output Layer ketika setiap sample harus menghasilkan
        distribusi probabilitas antar kelas.

  Kelebihan:
      - Output mudah ditafsirkan sebagai probabilitas.
      - Seluruh probabilitas dalam satu sample berjumlah 1.
      - Cocok untuk output multi-class.

  Kekurangan:
      - Sensitif terhadap stabilitas numerik bila exponential
        dihitung langsung pada logit besar.
      - Gradient sebenarnya berbentuk Jacobian sehingga lebih
        kompleks dibanding Activation element-wise.

  Rumus:
      pᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)

  Dengan:
      zᵢ = Pre-Activation / logit pada neuron ke-i
      zⱼ = Logit pada neuron ke-j
      e = Bilangan Euler
      pᵢ = Probabilitas pada neuron ke-i
      Σⱼ = Penjumlahan seluruh neuron output dalam satu sample

  Catatan Dimensi:
      - 1D → seluruh elemen dianggap satu sample.
      - 2D → setiap baris dianggap satu sample.
  """

  @runtime_method()
  def __call__(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:

    """
    Menghasilkan probabilitas Softmax.

    Rumus:
        pᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)

    Untuk stabilitas numerik digunakan bentuk:

        pᵢ = e^(zᵢ - z_max)
             ------------------
             Σⱼ e^(zⱼ - z_max)

    Mengurangi risiko Overflow pada np.exp().
    """

    # FloatArray hanya memiliki dimensi 1D atau 2D.
    #
    # 1D:
    #   axis = 0 → seluruh elemen merupakan satu sample.
    #
    # 2D:
    #   axis = 1 → setiap baris merupakan satu sample.
    axis = 0 if pre_activation.ndim == 1 else 1

    # Kurangi seluruh logit dengan nilai maksimum
    # pada dimensi output untuk menjaga stabilitas numerik.
    shifted = (
      pre_activation
      - np.max(
        pre_activation,
        axis=axis,
        keepdims=True,
      )
    )

    # Hitung exponential dari nilai yang telah digeser.
    exponentials = np.exp(
      shifted,
    )

    # Normalisasi sehingga seluruh probabilitas
    # pada setiap sample berjumlah 1.
    return exponentials / np.sum(
      exponentials,
      axis=axis,
      keepdims=True,
    )

  @runtime_method()
  def gradient(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:

    """
    Menghasilkan Gradient Softmax versi Element-Wise.

    Rumus diagonal Jacobian:
        ∂pᵢ/∂zᵢ = pᵢ(1 - pᵢ)

    Dengan:
        pᵢ = Output Softmax neuron ke-i
        zᵢ = Pre-Activation neuron ke-i

    Catatan penting:
        Gradient Softmax yang lengkap sebenarnya adalah Jacobian:

            ∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ - pⱼ)

        dengan:
            δᵢⱼ = 1 jika i = j
            δᵢⱼ = 0 jika i ≠ j

        Jacobian lengkap memiliki dimensi tambahan.
        Karena contract Activation saat ini mengharuskan
        hasil tetap berupa FloatArray 1D/2D, method ini hanya
        mengembalikan bagian diagonalnya.

    Implementasi ini sesuai bila backward engine kamu
    memperlakukan gradient activation secara element-wise.

    Untuk Multi-Class Classification dengan Cross-Entropy,
    biasanya turunan gabungan Softmax + Cross-Entropy
    dapat disederhanakan menjadi:

        ∂L/∂z = p - y

    sehingga Jacobian Softmax tidak perlu dihitung secara
    eksplisit.
    """

    probabilities = self(
      pre_activation,
    )

    # Bagian diagonal Jacobian:
    #     pᵢ(1 - pᵢ)
    return probabilities * (
      1.0 - probabilities
    )