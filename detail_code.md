kode - kode yg diberikan ini adalah urutan ekesekusi / perbaikan yg akan kita lakukan sebuah implementasi Runtime Custom untuk Argument, return type & Pydantic untuk Variable termasuk Self jg bila ada di dlm method.

Berikut Struktur project:

**Root Path:** `e:\Project C Zii\belajar-neural-network-python`

```
├── 📁 examples
│   └── 🐍 main.py
├── 📁 plots
│   └── ⚙️ .gitkeep
├── 📁 src
│   ├── 📁 core
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 activation_functions.py
│   │   ├── 🐍 initializer.py
│   │   ├── 🐍 layer.py
│   │   ├── 🐍 model.py
│   │   ├── 🐍 neuron.py
│   │   └── 🐍 optimizer.py
│   ├── 📁 data
│   │   └── 🐍 dataset.py
│   ├── 📁 evaluation
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 loss_functions.py
│   │   └── 🐍 metric_evaluation.py
│   ├── 📁 preprocessing
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 normalization.py
│   │   └── 🐍 preprocessing.py
│   ├── 📁 training
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 batch.py
│   │   └── 🐍 trainer.py
│   ├── 📁 utils
│   │   └── 🐍 custom_types.py
│   ├── 📁 validation
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 runtime.py
│   └── 📁 visualization
│       ├── 🐍 __init__.py
│       └── 🐍 plot.py
├── 📁 tests
│   ├── 🐍 diagnostics.py
│   ├── 🐍 test_activation.py
│   ├── 🐍 test_batch.py
│   ├── 🐍 test_custom_types.py
│   ├── 🐍 test_initializer.py
│   ├── 🐍 test_layer.py
│   ├── 🐍 test_loss.py
│   ├── 🐍 test_metrics.py
│   ├── 🐍 test_model.py
│   ├── 🐍 test_neuron.py
│   ├── 🐍 test_normalization.py
│   ├── 🐍 test_optimizer.py
│   ├── 🐍 test_preprocessing.py
│   ├── 🐍 test_trainer.py
│   └── 🐍 test_validation.py
├── ⚙️ .gitignore
├── 📝 README.md
└── 📄 requirements.txt
```


Lalu kode yg saya berikan ini jg berhubungan dgn posisi struktur Project yg telah diberikan sebelumnya dan sekaligus urutan perbaikan impelemntasi vectorized & runtime Custom & Pydantic, oke

1. `src/validation/runtime.py` ini sudah SELESAI dan TIDAK DIUBAH lagi, oke

```
"""
Bagian Pengelolaan Runtime Validation pada Project Neural Network.

Menyediakan Runtime Validation untuk:

1. Function
2. Method

Pydantic digunakan sebagai Engine Validation.

Module ini juga mengubah ValidationError menjadi Error Message yang lebih spesifik dan mudah dibaca.


=======================
=== CARA PENGGUNAAN ===
=======================

1. Validasi Function

    @runtime_function()
    def add(a: int, b: int) -> int:
      return a + b

    add(1, 2)        # OK
    add(1, "abc")    # RuntimeFunctionError (Argument "b" tidak valid)


2. Validasi Method

    class Layer:
      @runtime_method()
      def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        return inputs * 2

    Layer().forward(np.ones(3))     # OK
    Layer().forward([1.0, 2.0])     # RuntimeMethodError


3. Mode Strict (tanpa konversi tipe otomatis)

    @runtime_function(strict=True)
    def square(x: int) -> int:
      return x * x

    square(3)      # OK
    square("3")    # RuntimeFunctionError (di mode biasa "3" akan dikonversi)


4. Menangkap Error

    try:
      add(1, "abc")
    except RuntimeValidationError as error:   # induk dari Function & Method
      print(error)


ATURAN PENTING:

- Urutan Decorator untuk classmethod / staticmethod:

    @classmethod
    @runtime_method()          # Decorator runtime harus DI BAWAH classmethod
    def build(cls, size: int) -> "Layer": ...

- Semua Type Annotation harus sudah bisa di-resolve saat Decorator dipasang (get_type_hints dipanggil saat itu).
  Forward Reference ke Class yang sedang didefinisikan (mis. `-> "Layer"` di dalam Class Layer sendiri) akan menimbulkan NameError.

- Belum mendukung `async def` (Return yang divalidasi adalah coroutine-nya, bukan hasil akhirnya).


=====================
=== STRUKTUR FILE ===
=====================

1. Type Variable & Constant
2. Exception              -> RuntimeValidationError dan turunannya
3. ErrorFormatter         -> Class untuk membuat pesan Error yang rapi
4. Adapter Helper         -> Membuat TypeAdapter untuk validasi Return
5. RuntimeValidator       -> Class Decorator (mesin utama)
6. Public API             -> runtime_function() dan runtime_method()
"""


from collections.abc import Callable
from dataclasses import dataclass, is_dataclass
from functools import wraps
from inspect import signature
from typing import (
  Any,
  Final,
  TypeVar,
  cast,
  get_type_hints,
  is_typeddict,
)

import numpy as np

from numpy.typing import NDArray

from pydantic import (
  BaseModel,
  ConfigDict,
  TypeAdapter,
  ValidationError,
  validate_call,
)

from pydantic_core import ErrorDetails


# ======================
# === TYPE VARIABLES ===
# ======================

# F mewakili "Function apa pun". 
# Dgn TypeVar, Decorator mengembalikan tipe yg SAMA dgn Function asli sehingga Autocomplete / Type Checker tdk kehilangan Signature Function.

F = TypeVar("F", bound=Callable[..., Any])


# =================
# === CONSTANTS ===
# =================

# Lebar Label pada Pesan Error agar tanda ":" sejajar.

LABEL_WIDTH: Final[int] = 8


# =================
# === EXCEPTION ===
# =================

class RuntimeValidationError(Exception):
  """Base Error seluruh Runtime Validation."""


class RuntimeFunctionError(RuntimeValidationError):
  """Error pada Function."""


class RuntimeMethodError(RuntimeValidationError):
  """Error pada Method."""


# ===========================
# === ERROR FORMATTER ===
# ===========================

# ERROR FORMATTER — Kumpulan logika untuk mengubah ValidationError milik Pydantic menjadi teks yg mudah dibaca.

# Kenapa dijadikan Class?
# Semua fungsi format berbagi 1 pengaturan yg sama (label_width) dan berkaitan erat satu sama lain.
# Dgn Class, semuanya berkumpul di 1 tempat dan pengaturannya bisa diganti tanpa mengubah Constant global.

# frozen=True membuat Object tidak bisa diubah setelah dibuat (aman dibagi).

@dataclass(frozen=True)
class ErrorFormatter:
  label_width: int = LABEL_WIDTH

  # --------------------
  # --- Helper Kecil ---
  # --------------------

  # FORMAT VALUE — Memformat Value agar informasi penting tetap terlihat.
  @staticmethod
  def format_value(value: Any) -> str:
    # Tipe Array NumPy: tampilkan shape & dtype
    if isinstance(value, np.ndarray):
      # Beri tahu Type Checker bahwa ini NDArray
      array = cast(NDArray[Any], value)

      return f"{type(array).__name__}(shape={array.shape}, dtype={array.dtype})"

    # Tipe lain: tampilkan value dgn representasi stirng
    return repr(value)

  # FORMAT TYPE — Memformat nama tipe dari sebuah Value.
  @staticmethod
  def format_type(value: Any) -> str:
    # Jika Value adalah Class, tampilkan nama Class itu sendiri
    if isinstance(value, type):
      return value.__name__

    # Jika Value adalah Instance, tampilkan nama tipe Instance-nya
    return type(value).__name__

  # GET ERROR MESSAGE — Mengambil pesan Error dari detail Pydantic.
  @staticmethod
  def get_error_message(detail: ErrorDetails) -> str:
    # Ambil Conteks yg ada bila Validator Custom melempar Exception
    context: dict[str, Any] | None = detail.get("ctx")

    if context is not None:
      original_error: Any = context.get("error")

      # Jika ada Exception asli, pakai pesannya
      if isinstance(original_error, Exception):
        return str(original_error)

    # Jika tidak ada, gunakan pesan bawaan Pydantic
    return str(detail.get("msg", "Nilai tidak valid."))

  # RESOLVE ARGUMENT NAME — Menentukan nama Argument dari lokasi Error.
  @staticmethod
  def _resolve_argument_name(
    location: tuple[int | str, ...],
    argument_names: tuple[str, ...],
    positional_offset: int,
  ) -> str:
    # Pydantic memberi lokasi ("loc") dalam 2 bentuk:
    #   - str  -> sudah berupa nama Argument (kasus umum)
    #   - int  -> urutan Positional Argument, sehingga harus diterjemahkan ke nama

    # Tdk ada informasi lokasi sama sekali
    if not location:
      return "unknown"

    # Ambil lokasi pertama
    first: int | str = location[0]

    # Jika lokasi berupa index, terjemahkan menjadi nama Argument
    if isinstance(first, int):
      # Sesuaikan index terhadap self/cls pada Method
      index: int = first - positional_offset

      # Jika index msh berada dlm jangkauan, kembalikan nama Argument yg sesuai
      if 0 <= index < len(argument_names):
        return argument_names[index]

    # Jika lokasi sdh berupa nama atau index berada di luar jangkauan, tampilkan lokasi tersebut apa adanya
    return str(first)

  # ----------------------
  # --- Format 1 Error ---
  # ----------------------

  # ROW — Membuat 1 baris Label dgn lebar yang tetap.
  def _row(self, label: str, content: str) -> str:
    return f"{label:<{self.label_width}} : {content}"   # Contoh: "Label    : Isi"

  # FORMAT ARGUMENT DETAIL — Memformat 1 Error Argument.
  def format_argument_detail(
    self,
    detail: ErrorDetails,
    argument_names: tuple[str, ...],
    positional_offset: int,
  ) -> str:
    # Buat nama Argument
    argument_name: str = self._resolve_argument_name(
      detail.get("loc", ()),
      argument_names,
      positional_offset,
    )

    # Value yg ditolak oleh Pydantic
    input_value: Any = detail.get("input")

    # Format Detail
    return "\n".join((
      self._row("Argument", argument_name),
      self._row("Received", self.format_value(input_value)),
      self._row("Type", self.format_type(input_value)),
      self._row("Message", self.get_error_message(detail)),
    ))

  # FORMAT RETURN DETAIL — Memformat 1 Error Return.
  def format_return_detail(self, detail: ErrorDetails, result: Any) -> str:
    # Format Detail
    return "\n".join((
      self._row("Return", self.format_value(result)),
      self._row("Type", self.format_type(result)),
      self._row("Message", self.get_error_message(detail)),
    ))

  # ----------------------------
  # --- Format Seluruh Error ---
  # ----------------------------

  # FORMAT ARGUMENT ERRORS — Memformat seluruh Error Argument.
  def format_argument_errors(
    self,
    error: ValidationError,
    argument_names: tuple[str, ...],
    positional_offset: int,
  ) -> str:
    # Buat Detail pada setiap Error Argument
    details: list[str] = [
      self.format_argument_detail(detail, argument_names, positional_offset)
      for detail in error.errors()
    ]

    # Antar Error dipisahkan 1 baris kosong
    return "\n\n".join(details)

  # FORMAT RETURN ERRORS — Memformat seluruh Error Return.
  def format_return_errors(self, error: ValidationError, result: Any) -> str:
    # Buat Detail pada setiap Error Return
    details: list[str] = [
      self.format_return_detail(detail, result)
      for detail in error.errors()
    ]

    # Antar Error dipisahkan 1 baris kosong
    return "\n\n".join(details)


# ======================
# === ADAPTER HELPER ===
# ======================

# Pydantic menolak (PydanticUserError) bila Config diberikan ke TypeAdapter untuk Type yg sdh punya Config sendiri, yaitu:
#   1. Class turunan BaseModel
#   2. TypedDict
#   3. Dataclass

# Catatan:
#   - Poin Dataclass adalah tambahan dari kode awal, yg hanya menangani BaseModel & TypedDict.
#   - Tanpa ini, Function dgn Return Dataclass akan Error saat Decorator dipasang.

# HAS OWN CONFIG — Menentukan apakah sebuah Type sudah membawa Config sendiri.
def _has_own_config(annotation: object) -> bool:
  # BaseModel: hrs berupa Class dulu sebelum issubclass() dipakai
  if isinstance(annotation, type) and issubclass(annotation, BaseModel):
    return True

  # TypedDic: hanya valid pada class/type, bukan instance atau object umum
  if isinstance(annotation, type) and is_typeddict(cast(Any, annotation)):
    return True

  # Cek apakah Object sesuai dgn Tipe Instance
  return isinstance(annotation, type) and is_dataclass(annotation)


# CREATE RETURN ADAPTER — Membuat Adapter untuk validasi Return Value.
def _create_return_adapter(return_type: object, config: ConfigDict) -> TypeAdapter[Any]:
  # Pyright tdk selalu bisa menginfer Type Annotation, karena itu diperlakukan sebagai Any
  annotation: Any = return_type

  # Type dgn Config sendiri, jgn beri Config tambahan
  if _has_own_config(annotation):
    return TypeAdapter(annotation)

  # Type umum/custom, beri Config dari Decorator
  return TypeAdapter(annotation, config=config)


# =============================
# === RUNTIME VALIDATOR ===
# =============================

# RUNTIME VALIDATOR — Mesin utama untuk Object dari Class Decorator.
class RuntimeValidator:
  # CONSTRUCTOR — initialization
  def __init__(
    self,
    error_type: type[RuntimeValidationError],
    label: str,
    *,
    strict: bool = False,
    skip_first_parameter: bool = False,
    formatter: ErrorFormatter | None = None,
  ) -> None:
    # Jenis Exception yg dibangkitkan saat validasi gagal
    self._error_type: type[RuntimeValidationError] = error_type

    # Label pada Pesan Error
    self._label: str = label

    # Opsi apakah tipe harus persis sama atau konversi otomatis
    self._strict: bool = strict

    # Opsi apakah Parameter self/cls digunakan atau tdk digunakan
    self._skip_first_parameter: bool = skip_first_parameter

    # Formatter Pesan Error
    self._formatter: ErrorFormatter = (
      formatter if formatter is not None else ErrorFormatter()
    )

    # Aturan validasi Pydantic
    self._config: ConfigDict = ConfigDict(
      strict=strict,
      arbitrary_types_allowed=True,
    )

  # BUILD ERROR — Membangun Exception dari Detail Error yg sudah diformat.
  def _build_error(self, function: Callable[..., Any], details: str) -> RuntimeValidationError:
    # Buat Format Detail Error
    return self._error_type(
      f"Error ({self._error_type.__name__}):\n\n"
      f"{self._label} : {function.__name__}\n\n"
      f"{details}"
    )

  # CALL — Dipanggil saat Decorator telah dipasang pada sebuah Function/Method.
  def __call__(self, function: F) -> F:
    # --- Persiapan (dijalankan SEKALI saat Decorator dipasang) ---

    # Nama seluruh Parameter Function
    all_names: tuple[str, ...] = tuple(signature(function).parameters)

    # Penentuan posisi awal Argument
    argument_names: tuple[str, ...] = (
      all_names[1:] if self._skip_first_parameter else all_names
    )

    positional_offset: int = 1 if self._skip_first_parameter else 0

    # Return Type dari Annotation
    return_type: Any = (
      get_type_hints(function, include_extras=True).get("return", Any)
    )

    # Validator Argument
    argument_validator: Callable[..., Any] = (
      validate_call(config=self._config, validate_return=False)(function)
    )

    # Validator Return
    return_adapter: TypeAdapter[Any] = (
      _create_return_adapter(return_type, self._config)
    )

    # --- Wrapper (dijalankan SETIAP Function dipanggil) ---

    # @wraps menyalin nama, docstring, dan annotation Function asli ke wrapper agar tdk "hilang" setelah didekorasi
    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
      # Validasi Argument
      try:
        # Jalankan Validasi
        result: Any = argument_validator(*args, **kwargs)

      except ValidationError as error:
        # Buat detail Errors
        details: str = self._formatter.format_argument_errors(error, argument_names, positional_offset)

        # Bangkitkan Error
        raise self._build_error(function, details) from error

      # Validasi Return
      try:
        # Jalankan Validasi
        return_adapter.validate_python(result, strict=self._strict)

      except ValidationError as error:
        # Buat detail Errors
        details = self._formatter.format_return_errors(error, result)

        # Bangkitkan Error
        raise self._build_error(function, details) from error

      return result

    # Untuk Type Checke,: wrapper diperlakukan sbg tipe Function asli
    return cast(F, wrapper)


# ==================
# === PUBLIC API ===
# ==================

# RUNTIME FUNCTION — Decorator Runtime Validation untuk Function.
def runtime_function(*, strict: bool = False) -> RuntimeValidator:
  return RuntimeValidator(RuntimeFunctionError, "Function", strict=strict)


# RUNTIME METHOD — Decorator Runtime Validation untuk Method.
def runtime_method(*, strict: bool = False) -> RuntimeValidator:
  return RuntimeValidator(RuntimeMethodError, "Method", strict=strict, skip_first_parameter=True)
```


2. `src/utils/custom_types.py` ini sudah SELESAI, tetap akan tetap diupdate terus bila ada tipe yg memang cocok dimasukan ke sini di urtan perbaikan kode berikutnya

```
"""Bagian Pengelolaan Type Code pd Project Neural Network"""

import numpy as np

from numpy.typing import NDArray
from enum import Enum

from typing import (
  Any,
  Annotated,
  Literal,
  TypeAlias,
  TypedDict,
)

from pydantic import (
  StrictInt,
  StrictFloat,
  StrictBool,
  BaseModel,
  BeforeValidator,
  ConfigDict,
  Field,
)


# ===================
# === BASIC TYPES ===
# ===================

Numeric: TypeAlias = StrictInt | StrictFloat

Int: TypeAlias = StrictInt
Float: TypeAlias = StrictFloat
Bool: TypeAlias = StrictBool


# =============================
# === NUMPY ARRAY VALIDATOR ===
# =============================

def _validate_vector(value: Any) -> NDArray[np.float64]:
  # Konversi menjadi NumPy Array float64
  array = np.asarray(value, dtype=np.float64)

  # Vector hrs 1D
  if array.ndim != 1:
    # Bangkitkan Error
    raise ValueError(f"Dimensi Vector hrs 1D, sedangkan pd value {array.ndim}D.")

  # Vector hrs memiliki isi
  if array.size == 0:
    # Bangkitkan Error
    raise ValueError(f"Vector tdk boleh kosong.")

  return array


def _validate_matrix(value: Any) -> NDArray[np.float64]:
  # Konversi menjadi NumPy Array float64
  array = np.asarray(value, dtype=np.float64)

  # Matrix harus 2D
  if array.ndim != 2:
    # Bangkitkan Error
    raise ValueError(f"Dimensi Matrix hrs 2D, sedangkan pd value {array.ndim}D.")

  # Matrix hrs memiliki isi
  if array.size == 0:
    # Bangkitkan Error
    raise ValueError(f"Matrix tdk boleh kosong.")

  return array


def _validate_array(value: Any) -> NDArray[np.float64]:
  # Konversi menjadi NumPy Array float64
  array = np.asarray(value, dtype=np.float64)

  # Array hanya boleh 1D atau 2D
  if array.ndim not in (1, 2):
    # Bangkitkan Error
    raise ValueError(f"Dimensi Array hrs 1D atau 2D, sedangkan pd value {array.ndim}D.")

  # Array hrs memiliki isi
  if array.size == 0:
    # Bangkitkan Error
    raise ValueError(f"Array tdk boleh kosong.")

  return array


# ===================
# === NUMPY TYPES ===
# ===================

FloatVector: TypeAlias = Annotated[NDArray[np.float64], BeforeValidator(_validate_vector)]
FloatMatrix: TypeAlias = Annotated[NDArray[np.float64], BeforeValidator(_validate_matrix)]
FloatArray: TypeAlias = Annotated[NDArray[np.float64], BeforeValidator(_validate_array)]


# ===========================
# === NUMERIC CONSTRAINTS ===
# ===========================

IntPositive: TypeAlias = Annotated[Int, Field(gt=0,)]
FloatPositive: TypeAlias = Annotated[Float, Field(gt=0.0)]

IntList: TypeAlias = Annotated[list[Int], Field(min_length=1)]
FloatList: TypeAlias = Annotated[list[Float], Field(min_length=1)]

AlphaRange: TypeAlias = Annotated[Float, Field(gt=0.0, lt=1.0,)]


# ===================
# === DATA SAMPLE ===
# ===================

class DataSample(BaseModel):
  # NumPy Array merupakan Arbitrary Type
  model_config = ConfigDict(arbitrary_types_allowed=True)

  # Input Features
  inputs: FloatVector

  # Target / Expected Output
  targets: FloatVector


# ====================
# === DATASET NAME ===
# ====================

DatasetName: TypeAlias = Literal["training", "validation", "testing", "generalization"]

# ===============
# === DATASET ===
# ===============

class Dataset(BaseModel):
  # Hanya 4 jenis Dataset yg diperbolehkan.
  model_config = ConfigDict(extra="forbid")

  # Dataset Training.
  training: Annotated[list[DataSample], Field(min_length=1)]

  # Dataset Validation.
  validation: Annotated[list[DataSample], Field(min_length=1)]

  # Dataset Testing.
  testing: Annotated[list[DataSample], Field(min_length=1)]

  # Dataset Generalization.
  generalization: Annotated[list[DataSample], Field(min_length=1)]

DatasetList: TypeAlias = list[Dataset]


# ==================
# === EVALUATING ===
# ==================

MetricsName: TypeAlias = Literal["MSE", "MAE", "RMSE"]

Metrics: TypeAlias = dict[MetricsName, Float]

EvaluatingType: TypeAlias = dict[DatasetName, Metrics]


# ====================
# === CACHE NEURON ===
# ====================

class CacheNeuron(TypedDict):

  # Input yg digunakan ketika Forward
  inputs: FloatVector

  # Nilai sebelum Activation Function
  pre_activation: Float


# ========================
# === TRAINING HISTORY ===
# ========================

class MetricsHistory(TypedDict):
  MSE: FloatVector
  MAE: FloatVector
  RMSE: FloatVector


class FitHistory(TypedDict):
  # Nomor Epoch
  epochs: IntList

  # Metrics Training
  training_metrics: MetricsHistory

  # Metrics Validation
  validating_metrics: MetricsHistory


class FitResult(TypedDict):
  # Seluruh History Training
  history: FitHistory

  # Epoch dengan Validation MSE terbaik
  best_epoch: int

  # Epoch terakhir yg dijalankan
  last_epoch: int

  # Validation MSE terbaik
  best_validating_mse: float


# ==================
# === ENUM TYPES ===
# ==================

class ScaleType(str, Enum):

  ORIGINAL = "original"

  NORMALIZE = "normalize"
```


3. `src/data/dataset.py` Kode bisa disesuaikan agar helpernya lbh clean dan tdk terlalu panjang ketika digunakan, eksplisit Type, dan mudah di pahami. Dibawah ini bentuk pola Dataset yg baru tolong seusiakan ke kode-kode yg berikutnya, oke

```
# Bagian Pengelolaan Data² yang akan digunakan dalam melatih Model

import numpy as np

from collections.abc import Sequence
from src.validation import runtime_function

from src.utils.custom_types import (
  Numeric,
  FloatVector,
  DataSample,
  Dataset,
)


# ==============
# === HELPER ===
# ==============

# CREATE FLOAT VECTOR — Membuat NumPy Vector dgn dtype float64.

@runtime_function()
def create_float_vector(values: Sequence[Numeric]) -> FloatVector:
  # Konversi Values menjadi Numpy Array float64
  return np.asarray(values, dtype=np.float64)


# ===============
# === Dataset ===
# ===============

dataset = Dataset(
  # TRAINING — Model belajar dari data ini.
  training=[
    DataSample(
      inputs=create_float_vector([1, 2, 3]),
      targets=create_float_vector([10, 20]),
    ),
    DataSample(
      inputs=create_float_vector([2, 4, 6]),
      targets=create_float_vector([20, 40]),
    ),
    DataSample(
      inputs=create_float_vector([3, 6, 9]),
      targets=create_float_vector([30, 60]),
    ),
    DataSample(
      inputs=create_float_vector([4, 8, 12]),
      targets=create_float_vector([40, 80]),
    ),
    DataSample(
      inputs=create_float_vector([5, 10, 15]),
      targets=create_float_vector([50, 100]),
    ),
    DataSample(
      inputs=create_float_vector([6, 12, 18]),
      targets=create_float_vector([60, 120]),
    ),
    DataSample(
      inputs=create_float_vector([7, 14, 21]),
      targets=create_float_vector([70, 140]),
    ),
    DataSample(
      inputs=create_float_vector([8, 16, 24]),
      targets=create_float_vector([80, 160]),
    ),
    DataSample(
      inputs=create_float_vector([9, 18, 27]),
      targets=create_float_vector([90, 180]),
    ),
    DataSample(
      inputs=create_float_vector([10, 20, 30]),
      targets=create_float_vector([100, 200]),
    ),
  ],

  # VALIDATION — Mengecek perkembangan Model selama eksperimen.
  validation=[
    DataSample(
      inputs=create_float_vector([11, 22, 33]),
      targets=create_float_vector([110, 220]),
    ),
    DataSample(
      inputs=create_float_vector([12, 24, 36]),
      targets=create_float_vector([120, 240]),
    ),
    DataSample(
      inputs=create_float_vector([13, 26, 39]),
      targets=create_float_vector([130, 260]),
    ),
  ],

  # TESTING — Evaluasi final pada data yang tidak digunakan untuk mengambil keputusan selama training.
  testing=[
    DataSample(
      inputs=create_float_vector([14, 28, 42]),
      targets=create_float_vector([140, 280]),
    ),
    DataSample(
      inputs=create_float_vector([15, 30, 45]),
      targets=create_float_vector([150, 300]),
    ),
    DataSample(
      inputs=create_float_vector([16, 32, 48]),
      targets=create_float_vector([160, 320]),
    ),
  ],

  # GENERALIZATION — Menguji kemampuan Model pada input baru yg masih mengikuti pola data.
  generalization=[
    DataSample(
      inputs=create_float_vector([5, 10, 15]),
      targets=create_float_vector([50, 100]),
    ),
    DataSample(
      inputs=create_float_vector([5.5, 11, 16.5]),
      targets=create_float_vector([55, 110]),
    ),
    DataSample(
      inputs=create_float_vector([7.5, 15, 22.5]),
      targets=create_float_vector([75, 150]),
    ),
    DataSample(
      inputs=create_float_vector([9.5, 19, 28.5]),
      targets=create_float_vector([95, 190]),
    ),
  ],
)
```


4. `src/core/activation_function.py` ini bisa disesuaikan Konsep Blutprintnya dgn Vectorized dan Runtimenya, tetapi saya punya saran kalau memang Method prosesnya itu memang lbh baik return single value maka bisa dibua, tetapi kalau lbh cleang menggunakan vectorized maka lbh baik menggunakan vectorized saja, oke

```
"""
Bagian Pengelolaan 'Activation Function', yaitu:

Metode dari Fungsi Aktivasi yg digunakan untuk menentukan Output suatu Neuron berdasarkan nilai Pre-Activation.
"""

import numpy as np

from abc import ABC, abstractmethod

from src.utils.custom_types import(
  FloatArray,
  AlphaRange,
)

# ============================
# === BLUEPRINT ACTIVATION ===
# ============================

class Activation(ABC):
  # DUNDER — Menghitung Aktivasi Fungsi setelah Initialization
  @abstractmethod
  def __call__(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    # Lempar Error
    raise NotImplementedError("Sub Class hrs mengimplementasikan method __call__().")

  # GRADIENT — Menghitung Gradient dari Activation
  @abstractmethod
  def gradient(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    # Lempar Error
    raise NotImplementedError("Sub Class hrs mengimplementasikan method gradient().")

# ==========================
# === METODE² ACTIVATION ===
# ==========================

# Linear — Meneruskan nilai apa adanya tanpa non-linearitas.

class Linear(Activation):
  def __call__(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    """
    f(z) = z
    """
    return np.asarray(pre_activation, dtype=np.float64)

  def gradient(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    """
    f'(z) = 1
    """
    return np.ones_like(pre_activation, dtype=np.float64)


# Rectified Linear Unit — Meneruskan nilai positif dan mengubah nilai negatif menjadi 0.

class ReLU(Activation):
  def __call__(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    """
    f(z) = max(0, z)
    """
    return np.maximum(0.0, pre_activation)

  def gradient(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    """
    f'(z) = { 1, jika z > 0
            { 0, jika z ⩽ 0
    """
    return np.where(pre_activation > 0.0, 1.0, 0.0)


# Leaky Rectified Linear Unit — Nilai positif diteruskan dan sebagian kecil nilai negatif tetap diteruskan.

class LeakyReLU(Activation):
  # CONSTRUCTOR — Initialization
  def __init__(self, alpha: AlphaRange = 0.01) -> None:
    # Nilai Alpha menentukan seberapa bsr nilai negatif yg tetap dpt dilewati
    self._alpha: AlphaRange = alpha

  # ALPHA — Getter untuk mendapatkan nilai Alpha
  @property
  def alpha(self) -> float:
    return self._alpha

  def __call__(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    """
    f(z) = { z,  jika z > 0
           { αz, jika z ⩽ 0
    """
    return np.where(
      pre_activation > 0.0,
      pre_activation,
      self._alpha * pre_activation,
    )

  def gradient(
    self,
    pre_activation: FloatArray,
  ) -> FloatArray:
    """
    f'(z) = { 1, jika z > 0
            { α, jika z ⩽ 0
    """
    return np.where(pre_activation > 0.0, 1.0, self._alpha)
```


5. `src/core/initializer.py` Ini msh di seusiakan krn konsep Blueprint sama seperti activation kalau memang lbh clean langsung menghasilkan byk weight ketimbang single return value maka lbh baik menggunakan vectorized, oke

```
"""
Bagian Pengelolaan 'Initialization', yaitu:

Menentukan Metode dan menghasilkan nilai awal Weight yg akan digunakan oleh Neuron.
"""

import numpy as np

from abc import ABC, abstractmethod

from src.utils.custom_types import (
  FloatArray,
  IntPositive,
)

# =============================
# === BLUEPRINT INITIALIZER ===
# =============================

class Initializer(ABC):
  # DUNDER — Generate Weights setelah Initialization
  @abstractmethod
  def __call__(
    self,
    shape: tuple[IntPositive, ...],   # Pola: (Jumlah_input, jumlah_neuron)
  ) -> FloatArray:
    # Lempar Error
    raise NotImplementedError("Sub Class hrs mengimplementasi method __call__().")

# ===========================
# === METODE² INITIALIZER ===
# ===========================

# He/Kaiming Normal — Menentukan nilai Weight awal menggunakan Distribusi Gaussian.

class HeNormal(Initializer):
  def __call__(
    self,
    shape: tuple[IntPositive, ...],
  ) -> FloatArray:
    """
    σ = √(2 / fan_in)
    """

    # Validasi Argument
    if len(shape) < 2:
      raise ValueError("Shape Weight minimal hrs memiliki 2 dimensi.")

    # Ambil jumlah Input yg msk ke tiap Neuron 
    fan_in = shape[0]

    # Menghitung Standard Deviation
    stddev = np.sqrt(2.0 / fan_in)

    # Gunakan Generator Random
    rng = np.random.default_rng()

    # Generate Weight berupa skala Gaussian dgn Mean 0.0 dan Standar Deviasi
    return rng.normal(loc=0.0, scale=stddev, size=shape)
```


6. `src/core/optimizer.py` ini jg sama seperti activation & intializer sesuaikan jg dan pastikan eksplisit jg untuk kode type dan runtime nya, oke

```
"""
Bagian pengelolaan 'Optimizer', yaitu:

Menyesuaikan Weight & Bias secara berulang-ulang untuk meminimalkan nilai Loss berdasarkan Gradient.
"""

from abc import ABC, abstractmethod

from src.utils.custom_types import (
  FloatArray,
  FloatPositive,
)

# ===========================
# === BLUEPRINT OPTIMIZER ===
# ===========================

class Optimizer(ABC):
  # UPDATE — Proses penyesuaian Weight & Bias berdasarkan Gradient
  @abstractmethod
  def __call__(
    self,
    W: FloatArray,
    b: FloatArray,
    gradient_W: FloatArray,
    gradient_b: FloatArray,
  ) -> tuple[FloatArray, FloatArray]:
    # Lempar Error
    raise NotImplementedError("Sub Class hrs mengimplementasikan method __call__().")

# =========================
# === METODE² OPTIMIZER ===
# =========================

# Stochastic Gradient Descent — Memperbarui Weight & Bias dgn bergerak berlawanan arah terhadap Gradient Loss agar nilai Loss cenderung menurun.

class SGD(Optimizer):
  # CONSTRUCTOR — Initialization
  def __init__(self, learning_rate: FloatPositive) -> None:
    # Hyper Parameter untuk menentukan seberapa bsr perubahan Weight & Bias dlm tiap Update
    self.learning_rate: FloatPositive = learning_rate

  def __call__(
    self,
    W: FloatArray,
    b: FloatArray,
    gradient_W: FloatArray,
    gradient_b: FloatArray,
  ) -> tuple[FloatArray, FloatArray]:
    """
    Update Weight: w_new = w - η × ∂L/∂w
    """
    updated_W = W - (self.learning_rate * gradient_W)

    """
    Update Bias: b_new = b - η × ∂L/∂b
    """
    updated_b = b - (self.learning_rate * gradient_b)

    return (updated_W, updated_b)
```


7. `src/core/neuron.py` ini seperti rencana vectorized sebelumnya, yaitu akan dialihkan ke Layer, tetapi tolong ketika pindah ke layer tetap clean sesuai dgn plan vectorized, lalu beritahu secara eksplisit bagian-bagian proses yg akan dilakukan oleh Neuron dan layer, oke

```
"""
Bagian Pengelolaan paling dasar dari Sistem Neural Network yaitu:

Neuron / Saraf yg digunakan untuk menerima beberapa Input, memberikan Weight pd tiap Input, menambahkan Bias, lalu menghasilkan Output.
"""

from .initializer import Initializer
from .activation_functions import Activation
from src.utils.custom_types import (
  FloatVector,
  FloatSequence,
  IntPositive,
  CacheNeuron,
)

class Neuron:
  # CONSTRUCTOR — Initialization
  def __init__(
    self,
    input_size: IntPositive,
    initializer: Initializer,
    activation: Activation,
  ) -> None:
    # Menentukan brp byk input yg bisa diterima Neuron
    self.input_size: IntPositive = input_size

    # Activation yg digunakan
    self.activation: Activation = activation

    # Initializer yg digunakan
    self.initializer: Initializer = initializer

    # Menentukan pengaruh tiap Input terhadap Output
    self.weights: FloatVector = initializer(input_size)

    # Menentukan pergeseran nilai Pre-Activation
    self.bias: float = 0.0

    # Menyimpan Gradient terhadap tiap Weight
    self.gradient_weights: FloatVector = [0.0] * input_size

    # Menyimpan Gradient terhadap Bias
    self.gradient_bias: float = 0.0

    # Menyimpan Nilai² yg dibutuhkan dlm Neuron ini
    self.cache: CacheNeuron | None = None

  # FORWARD — Proses Menghasilkan Prediksi
  def forward(self, inputs: FloatSequence) -> float:
    # Validasi Argument
    if len(inputs) != self.input_size:
      raise ValueError(f"Panjang inputs ({len(inputs)}) tdk sesuai dgn input_size ({self.input_size}).")

    """ Neuron: z = ∑(xᵢ × wᵢ) + b """
    pre_activation = sum(x * w for x, w in zip(inputs, self.weights)) + self.bias

    # Simpan ke Cache
    self.cache = {
      'inputs': list(inputs),
      'pre_activation': pre_activation,
    }

    """ Activation: y = f(z) """
    return self.activation(pre_activation)

  # BACKWARD — Menghitung Gradient yaitu nilai yg menunjukkan seberapa bsr perubahan Loss
  def backward(self, gradient_output: float) -> FloatVector:
    # Validasi Cache
    if self.cache is None:
      raise ValueError("Backward tdk dpt dilakukan sebelum Forward.")

    # Ambil data² dari Cache
    inputs = self.cache['inputs']
    pre_activation = self.cache['pre_activation']

    """ Activation Gradient: ∂L/∂z = (∂L/∂y) × f'(z) """
    gradient_pre_activation = (
      gradient_output * self.activation.gradient(pre_activation)
    )

    """ Gradient Weight: ∂L/∂wᵢ = (∂L/∂z) × xᵢ """
    for i, x in enumerate(inputs):
      self.gradient_weights[i] += gradient_pre_activation * x

    """ Gradient Bias: ∂L/∂b = ∂L/∂z """
    self.gradient_bias += gradient_pre_activation

    """ Gradient Input: ∂L/∂xᵢ = (∂L/∂z) × wᵢ """
    gradient_input = [
      gradient_pre_activation * weight
      for weight in self.weights
    ]

    return gradient_input

  # RESET GRADIENT — Mengosongkan Gradient Weight & Bias
  def reset_gradient(self) -> None:
    # Gradient Weight
    self.gradient_weights = [0.0] * self.input_size

    # Gradient Bias
    self.gradient_bias = 0.0

  # AVERAGE GRADIENT — Menghitung Rata² Gradient Weight & Bias
  def average_gradient(self, batch_size: IntPositive) -> None:
    """ Gradient Weight: ḡwᵢ = gwᵢ / B """
    self.gradient_weights = [
      gradient / batch_size
      for gradient in self.gradient_weights
    ]

    """ Gradient Bias: ḡb = gb / B """
    self.gradient_bias /= batch_size
```


8. `src/core/layer.py` ini merupakan pusatnya sesuai rencana sebelumnya yaitu neuron dipindahkan kesini dan ingat pd point di neuron bahwa tetap eksplisit mana bagian yg merupakan proses Neuron & layer. Lalu tolong buat juga agar Layer dibuat Meiliki Nama nya sendiri dan menjadi konsep Blueprint agar ketika dimasa depan akan memulai fokus impelementasi Image Clasificationn seperti CNN, Flatten layer, ddl ini sdh siap, oke

```
"""
Bagian Pengelolaan Layer, yaitu:

Membuat & Mengatur sekumpulan Neuron agar dapat memproses Input secara bersamaan dan menghasilkan sekumpulan Output.
"""

from .initializer import Initializer
from .activation_functions import Activation
from .optimizer import Optimizer
from .neuron import Neuron
from src.utils.custom_types import (
  FloatVector,
  FloatSequence,
  IntPositive,
)

class Layer:
  # CONSTRUCTOR — Initialization
  def __init__(
    self,
    input_size: IntPositive,
    neuron_count: IntPositive,
    initializer: Initializer,
    activation: Activation,
    optimizer: Optimizer,
  ) -> None:
    # Menentukan brp byk input yg bisa diterima oleh tiap Neuron
    self.input_size: IntPositive = input_size

    # Menentukan jumlah Neuron yg digunakan
    self.neuron_count: IntPositive = neuron_count

    # Activation yg digunakan
    self.activation: Activation = activation

    # Initializer yg digunakan
    self.initializer: Initializer = initializer

    # Optimizer yg digunakan
    self.optimizer: Optimizer = optimizer

    # Buat & Simpan Neuron²
    self.neurons: list[Neuron] = [
      Neuron(
        input_size=input_size,
        initializer=initializer,
        activation=activation,
      )

      for _ in range(neuron_count)
    ]

  # FORWARD — Proses Menghasilkan Prediksi tiap Neuron
  def forward(self, inputs: FloatSequence) -> FloatVector:
    # Validasi Argument
    if len(inputs) != self.input_size:
      raise ValueError(f"Panjang inputs ({len(inputs)}) tdk sesuai dgn input_size ({self.input_size}).")

    # Jalankan Method Forward pd tiap Neuron
    outputs = [
      neuron.forward(inputs)
      for neuron in self.neurons
    ]

    return outputs

  # BACKWARD — Proses Menghitung Gradient tiap Neuron
  def backward(self, gradient_outputs: FloatSequence) -> FloatVector:
    # Validasi Argument
    if len(gradient_outputs) != self.neuron_count:
      raise ValueError(f"Panjang gradient_outputs ({len(gradient_outputs)}) tdk sesuai dgn neuron_count ({self.neuron_count}).")

    # Wadah Gradient Input
    gradient_input = [0.0] * self.input_size

    # Jalankan Method Backward pd tiap Neuron
    for neuron, gradient_output in zip(self.neurons, gradient_outputs):
      # Gradient yg dikirim Neuron
      neuron_gradient_input = neuron.backward(gradient_output)

      # Gabungkan Gradient tiap index
      for i, gradient in enumerate(neuron_gradient_input):
        gradient_input[i] += gradient

    return gradient_input

  # RESET GRADIENT — Proses Mengosongkan Gradient tiap Neuron
  def reset_gradient(self) -> None:
    for neuron in self.neurons:
      neuron.reset_gradient()

  # AVERAGE GRADIENT — Menghitung Rata² Gradient tiap Neuron
  def average_gradient(self, batch_size: IntPositive) -> None:
    for neuron in self.neurons:
      neuron.average_gradient(batch_size)
  
  # STEP — Proses Update Weight & Bias menggunakan Optimizer
  def step(self) -> None:
    for neuron in self.neurons:
      # Optimize, lalu kemudian Update Weight & Bias
      neuron.weights, neuron.bias = self.optimizer(
        weights=neuron.weights,
        bias=neuron.bias,
        gradient_weights=neuron.gradient_weights,
        gradient_bias=neuron.gradient_bias,
      )
```

9. `src/core/model.py` ini juga masih perlu implementasi vectorized & runtime nya. Lalu sama seperti layer kalau misalakan standar implementasi nyata / di library untuk Deep learinng / Machine learning sebuah Model hrs memailiki NAMA nya sendiri ketika digunaakan, maka tolong implementasikan, oke (jika tdk, abaikan saja)

```
""" Bagian Utama untuk Pengelolaan Model Neural Network """

from .initializer import Initializer
from .optimizer import Optimizer
from .activation_functions import Activation, Linear
from .layer import Layer
from src.utils.custom_types import (
  FloatVector,
  FloatSequence,
  IntPositive,
)

class Model:
  # CONSTRUCTOR — Initialization
  def __init__(
    self,
    initializer: Initializer,
    optimizer: Optimizer,
    activation: Activation,
    architecture: tuple[IntPositive, ...] = (3, 4, 4, 2),
  ) -> None:
    # Validasi jumlah Layer
    if len(architecture) < 2:
      raise ValueError("Architecture minimal hrs memiliki Layer Input & Output.")

    # Validasi Layer Input 
    if architecture[0] != 3:
      raise ValueError("Input Layer minimal hrs memiliki 3 Neuron.")

    # Validasi Layer Output
    if architecture[-1] != 2:
      raise ValueError("Output Layer minimal hrs memiliki 2 Neuron.")

    # Wadah seluruh Layer
    self.layers: list[Layer] = []

    # Buat Arsitektur Model
    for i in range(len(architecture) - 1):
      # Jumlah Input Data pd tiap Layer
      input_size = architecture[i]

      # Jumlah Neuron pd tiap Layer
      neuron_count = architecture[i + 1]

      # Cek apakah Layer merupakan kategori Output
      is_output_layer = i == len(architecture) - 2
    
      # Input Layer  ⟶  Activation
      # Hidden Layer  ⟶  Activation
      # Output Layer  ⟶  Linear

      layer_activation = Linear() if is_output_layer else activation

      # Simpan Layer
      self.layers.append(
        Layer(
          input_size=input_size,
          neuron_count=neuron_count,
          initializer=initializer,
          optimizer=optimizer,
          activation=layer_activation,
        )
      )

  # FORWARD — Proses Menghasilkan Prediksi tiap Layer
  def forward(self, inputs: FloatSequence) -> FloatVector:
    # Simpan nilai awal
    output = inputs

    # Jalankan Method Forward tiap Layer dari awal ke akhir
    for layer in self.layers:
      output = layer.forward(output)

    return output

  # BACKWARD — Proses Menghitung Gradient tiap Layer
  def backward(self, gradient_outputs: FloatSequence) -> FloatVector:
    # Gradient awal
    gradient = gradient_outputs

    # Jalankan Method Backward dari Layer terakhir ke awal
    for layer in reversed(self.layers):
      gradient = layer.backward(gradient)

    return gradient

  # RESET GRADIENT — Mengosongkan Gradient tiap Layer
  def reset_gradient(self) -> None:
    for layer in self.layers:
      layer.reset_gradient()

  # AVERAGE GRADIENT — Menghitung Rata² Gradient tiap Layer
  def average_gradient(self, batch_size: IntPositive) -> None:
    for layer in self.layers:
      layer.average_gradient(batch_size)

  # STEP — Proses Update Weight & Bias tiap Layer
  def step(self) -> None:
    for layer in self.layers:
      layer.step()
```


10. `src/evaluation/loss_function.py` ini sesuaikan juga dengan vectorized dan buat jg kalau memang lbh baik vectorized maka implementasikan, oke

```
"""
Bagian Pengelolaan Loss Function, yaitu:

Angka yg digunakan untuk mengukur seberapa Error (salah) yg dihasilkan pd sebuah Prediksi Model.
"""

import numpy as np

from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

from src.utils.custom_types import (
  FloatArray,
)

P = ParamSpec("P")
R = TypeVar("R")

# ======================
# === BLUEPRINT LOSS ===
# ======================

class Loss(ABC):
  # VALIDATE ARGUMENS — Decorator yg memvalidasi hubungan antara Argument
  @staticmethod
  def validate_arguments(func: Callable[P, R]) -> Callable[P, R]:
    # Membungkus Function asli dgn proses tambahan
    @wraps(func)
    def wrapper(
      self: Any,
      targets: FloatArray,
      predictions: FloatArray
    ) -> R:
      # Validasi Shape Argument
      if targets.shape != predictions.shape:
        # Lempar Error
        raise ValueError(
          f"Shape targets ({targets.shape}) dgn"
          f"Shape predictions ({predictions.shape}) tdk cocok."
        )

      # Jalankan Function asli setelah validasi berhasil
      return func(self, targets, predictions)

    # Kembalikan Function yg sdh dibungkus validasi
    return wrapper

  # DUNDER — Menghitung Loss setelah Initialization
  @abstractmethod
  def __call__(
    self, 
    targets: FloatArray, 
    predictions: FloatArray,
  ) -> float:
    # Lempar Error
    raise NotImplementedError("Sub Class hrs mengimplementasikan method __call__().")

  # GRADIENT — Menghitung Loss pd Gradient
  @abstractmethod
  def gradient(
    self, 
    targets: FloatArray, 
    predictions: FloatArray,
  ) -> FloatArray:
    # Lempar Error
    raise NotImplementedError("Sub Class hrs mengimplementasikan method gradient().")

# ====================
# === METODE² LOSS ===
# ====================

# Mean Squared Error — Menghitung rata² kuadrat selisih antara Target dgn Prediction sehingga Error yg bsr diberi Penalti yg lbh bsr

class MSE(Loss):
  @Loss.validate_arguments
  def __call__(self, targets: FloatArray, predictions: FloatArray) -> float:  
    """ MSE = (1/n) × Σ(y - ŷ)² """

    errors = targets - predictions

    return float(np.mean(errors ** 2))

  @Loss.validate_arguments
  def gradient(self, targets: FloatArray, predictions: FloatArray) -> FloatArray:
    """
    ∂MSE/∂ŷ = (2/n) × (ŷ - y)
    """
    return 2.0 * (predictions - targets) / targets.size


# Mean Absolute Error — Menghitung rata² jarak absolut antara Target dgn Prediction sehingga lbh tahan terhadap Outlier

class MAE(Loss):
  @Loss.validate_arguments
  def __call__(self, targets: FloatArray, predictions: FloatArray) -> float:  
    """ MAE = (1/n) × Σ|y - ŷ| """

    errors = targets - predictions

    return float(np.abs(errors).mean())

  @Loss.validate_arguments
  def gradient(self, targets: FloatArray, predictions: FloatArray) -> FloatArray:
    """
    ∂MAE/∂ŷ = sign(ŷ - y) / n
    """
    return np.sign(predictions - targets) / targets.size
```


11. `src/evaluation/metric_evaluation.py` sepertinya nanti saat mencapai perbaikan file ini akan direname saja menjadi `metrics.py` krn dgn kata ini saja sdh dpt menjelasakan makna isi nya, dan seperti sebelumnya sesuaikann dgn implementasii Vectorized dan runtime, oke

```
"""
Bagian Pengelolaan Metric, yaitu:

Ukuran yg digunakan untuk mengetahui seberapa bagus Performa sebuah Model berdasarkan hasil Prediksi terhadap Target.
"""

import math

from abc import ABC, abstractmethod
from functools import wraps
from typing_extensions import Any, Callable

from src.utils.custom_types import (
  FloatSequence,
  Metrics,
)

# ========================
# === BLUEPRINT METRIC ===
# ========================

class Metric(ABC):
  # VALIDATE ARGUMENS — Decorator yg memvalidasi hubungan antara Argument
  @staticmethod
  def validate_arguments(func: Callable[..., Any]) -> Callable[..., Any]:
    # Melanjutkan informasi function yg dibungkus
    @wraps(func)
    def wrapper(self, targets: FloatSequence, predictions: FloatSequence) -> Any:
      # Validasi panjang input
      if len(targets) != len(predictions):
        raise ValueError(f"Panjang targets ({len(targets)}) dgn predictions ({len(predictions)}) tdk cocok.")

      return func(self, targets, predictions)

    return wrapper

  # DUNDER — Menjalankan Method setelah Initialization yaitu menghitung Metric
  @abstractmethod
  def __call__(self, targets: FloatSequence, predictions: FloatSequence) -> float:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method __call__().")

# ======================
# === METODE² METRIC ===
# ======================

#  Mean Squared Error — Menghitung rata² kuadrat selisih antara Target dgn Prediction sehingga Error yg bsr diberi Penalti yg lbh bsr

class MSE(Metric):
  @Metric.validate_arguments
  def __call__(self, targets: FloatSequence, predictions: FloatSequence) -> float:  
    """ MSE = (1/n) × Σ(y - ŷ)² """
    total = sum(
      (target - prediction) ** 2
      for target, prediction in zip(targets, predictions)
    )
  
    return total / len(targets)


# Mean Absolute Error — Mengukur rata² jarak absolut Error Prediksi Model sehingga mdh diinterpretasikan dlm satuan Target

class MAE(Metric):
  @Metric.validate_arguments
  def __call__(self, targets: FloatSequence, predictions: FloatSequence) -> float:
    """ MAE = (1/n) × Σ|y - ŷ| """
    total_error = sum(
      abs(target - prediction)
      for target, prediction in zip(targets, predictions)
    )
  
    return total_error / len(targets)


# Root Mean Squared Error — Hasil MSE di-akar sehingga satuannya kembali sama dgn Target

class RMSE(Metric):
  @Metric.validate_arguments
  def __call__(self, targets: FloatSequence, predictions: FloatSequence) -> float:
    """ RMSE = √MSE """
    rmse = math.sqrt(
      MSE()(targets, predictions)
    )

    return rmse


# =========================
# === CALCULATE METRICS ===
# =========================

def calculate_metrics(targets: FloatSequence, predictions: FloatSequence) -> Metrics:
  return {
    metric.__name__:
      metric()(targets, predictions)

    # Ambil otomatis tiap Sub Class dari Metric
    for metric in Metric.__subclasses__()
  }
```


12. `src/training/batch.py` kode ini jg sesuaikan dgn impelentasi runtime dan vectorized, oke 


```
"""
Bagian Pengelolaan Batch, yaitu:

Kumpulan data² yg diproses secara bersamaan.
"""

import math
import random

from src.utils.custom_types import (
  IntPositive,
  FloatPositive,
  Dataset,
  ListDataset,
  DatasetType,
)

class Batch:
  # CONSTRUCTOR — Initialization
  def __init__(self, seed: int | None = None) -> None:
    # Simpan Random Generator untuk Reproducibility
    self.random = random.Random(seed)

  # SPLIT DATASET — Membagi Dataset menjadi Sesi² seperti Training, Validation, dan Testing
  def split_dataset(
    self,
    dataset: Dataset,
    training_ratio: FloatPositive,
    validation_ratio: FloatPositive,
    testing_ratio: FloatPositive,
  ) -> DatasetType:
    # Hitung total Rasio
    total_ratio = training_ratio + validation_ratio + testing_ratio

    # Validasi Total Ratio adalah 1
    if not math.isclose(total_ratio, 1.0):
      raise ValueError("Total Rasio hrs sama dgn 1.")

    # Salin Dataset agar yg asli tdk berubah
    shuffle_dataset = list(dataset)

    # Acak data agar mengurangi ketergantungan Model pd pola urutan
    self.random.shuffle(shuffle_dataset)

    # Hitung total data
    dataset_size = len(shuffle_dataset)
    training_size = int(dataset_size * training_ratio)
    validation_size = int(dataset_size * validation_ratio)

    # Tentukan Index
    training_end = training_size
    validation_end = training_end + validation_size

    # Pisahkan Dataset
    training_data = shuffle_dataset[:training_end]
    validation_data = shuffle_dataset[training_end:validation_end]
    testing_data = shuffle_dataset[validation_end:]

    # Kembalikan Dataset yg telah dipisah
    return {
      "training": training_data,
      "validation": validation_data,
      "testing": testing_data,
    }

  # CREATE BATCH — Membagi Dataset menjadi beberapa Mini-Batch
  def create_batches(self, dataset: Dataset, batch_size: IntPositive) -> ListDataset:
    # Salin Dataset agar yg asli tdk berubah
    shuffle_dataset = list(dataset)

    # Acak data agar mengurangi ketergantungan Model pd pola urutan
    self.random.shuffle(shuffle_dataset)

    # Siapkan Wadah Batch
    batches: ListDataset = []

    # Proses pembagian Dataset berdasarkan pembagian ukuran Batch
    for i in range(0, len(shuffle_dataset), batch_size):
      # Slicing Index
      batch = shuffle_dataset[i : i + batch_size]

      # Simpan Mini-Batch
      batches.append(batch)

    return batches
```

13. `src/preprocessing/normalization.py` ini jg akan diakan di rename menjadi `normalizer.py` dan krn konsep blueprint maka sesuaikan dgn impelementasi Vectorized dan runtime, oke

```
"""
Bagian Pengelolaan Normalisasi Data, yaitu:

Proses mengubah skala Data agar lebih sesuai untuk digunakan oleh Model Neural Network.
"""

from abc import ABC, abstractmethod

from src.utils.custom_types import (
  FloatSequence,
)

# ============================
# === BLUEPRINT NORMALIZER ===
# ============================

class Normalizer(ABC):
  # CONSTRUCTOR — Initialization
  def __init__(self) -> None:
    # Menentukan apakah Normalizer sdh di-Fit
    self._is_fitted: bool = False

  # IS FITTED — Getter untuk mengetahui apakah sdh melakukan Fit
  @property
  def is_fitted(self) -> bool:
    return self._is_fitted

  # VALIDATE FITTED — Memastikan Normalizer telah di-Fit
  def _validate_fitted(self) -> None:
    if not self._is_fitted:
      raise ValueError("Normalizer hrs menjalankan fit() dahulu.")

  # FIT — Mempelajari parameter berdasarkan Data
  @abstractmethod
  def fit(self, values: FloatSequence) -> None:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method fit().")

  # NORMALIZE — Membuat sebuah nilai menjadi skala Normalisasi
  @abstractmethod
  def normalize(self, value: float) -> float:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method normalize().")

  # DENORMALIZE — Mengembalikan sebuah nilai ke skala Original
  @abstractmethod
  def denormalize(self, value: float) -> float:
    raise NotImplementedError("Sub Class hrs mengimplementasikan method denormalize().")

# ==========================
# === METODE² NORMALIZER ===
# ==========================

# Min Max — Menghasilkan nilai dgn rentan antara 0 sampai 1 selama berada dlm nilai minimum & maximum

class MinMaxNormalizer(Normalizer):
  # CONSTRUCTOR — Initialization
  def __init__(self) -> None:
    # Jalankan Constructor Normalizer
    super().__init__()

    # Nilai Minimum Data
    self.minimum: float | None = None

    # Nilai Maximum Data
    self.maximum: float | None = None

  # CHECK MIN MAX — Memastikan Property Minimum & Maximum ada
  def _validate_min_max(self) -> None:
    if (
      self.minimum is None or
      self.maximum is None
    ):
      raise ValueError("Nilai Min & Max tdk tersedia.")

  # FIT — Menetukan nilai Minimum & Maximum dari kumpulan data
  def fit(self, values: FloatSequence) -> None:
    # Cari nilai Min & Max dari kumpulan data
    min_value = min(values)
    max_value = max(values)

    # Validasi Min & Max tdk boleh sama
    if min_value == max_value:
      raise ValueError("Minimum dgn Maximum tdk boleh sama.")

    # Simpan nilai Min & Max
    self.minimum = min_value
    self.maximum = max_value

    # Set bahwa Normalizer sdh di-Fit
    self._is_fitted = True

  def normalize(self, value: float) -> float:
    # Validasi Cek Fit
    self._validate_fitted()

    # Validasi Cek Min & Max
    self._validate_min_max()

    """ x' = (x - x_min) / (x_max - x_min) """
    return (value - self.minimum) / (self.maximum - self.minimum)

  def denormalize(self, value: float) -> float:
    # Validasi Cek Fit
    self._validate_fitted()

    # Validasi Cek Min & Max
    self._validate_min_max()

    """ x = x' × (x_max - x_min) + x_min """
    return value * (self.maximum - self.minimum) + self.minimum
```


14. `src/preprocessing/preprocessing.py` ini lbh baik saat mencapai kode ini direname menjadi `preprocessor.py`, kemudian seperti yg lain implementasikan vectorized dan runtime. Lalu bisakah dibuat lbh clena yg dimana sy melihat byk fungsi yg hanya melakukan  looping satu tinglat beda method saya ingin supaya jgn terlalu byk dan sesuaikan dgn Bentuk dataset yg baru di custom, oke, oke

```
""" Bagian Preprocessing Data sebelum Digunakan oleh Model """

from .normalization import (
  Normalizer,
  MinMaxNormalizer
)

from src.utils.custom_types import (
  FloatVector,
  FloatSequence,
  Dataset,
  DatasetType,
)

class Preprocessor:
  # CONSTRUCTOR — Initialization
  def __init__(
    self,
    input_scaler: Normalizer | None = None,
    target_scaler: Normalizer | None = None,
  ) -> None:
    # Simpan Normalizer untuk Input
    self.input_scaler: Normalizer = input_scaler or MinMaxNormalizer()

    # Simpan Normalizer untuk Target
    self.target_scaler: Normalizer = target_scaler or MinMaxNormalizer()

  # NORMALIZE INPUTS — Normalisasi Kumpulan Nilai Input
  def normalize_inputs(self, values: FloatSequence) -> FloatVector:
    return [
      # Normalisasikan tiap nilai Input Scaler
      self.input_scaler.normalize(value)
  
      # Jalankan berdasarkan panjang values
      for value in values
    ]

  # NORMALIZE TARGET — Normalisasi Kumpulan Nilai Target
  def normalize_targets(self, values: FloatSequence) -> FloatVector:
    return [
      # Normalisasikan tiap nilai Target Scaler
      self.target_scaler.normalize(value)
  
      # Jalankan berdasarkan panjang values
      for value in values
    ]

  # NORMALIZE DATASET — Normalisasi sebuah Dataset
  def normalize_dataset(self, dataset: Dataset) -> Dataset:
    # Siapkan Wadah Dataset
    normalized_dataset: Dataset = []
  
    # Proses tiap Data
    for inputs, targets in dataset:
      # Normalisasi Input
      normalized_inputs = self.normalize_inputs(inputs)
  
      # Normalisasi Target
      normalized_targets = self.normalize_targets(targets)
  
      # Simpan Data yg sdh di Normalisasi
      normalized_dataset.append(
        (normalized_inputs, normalized_targets)
      )
  
    return normalized_dataset
  
  # NORMALIZE DATASETS — Normalisasi byk Dataset
  def normalize_datasets(self, datasets: DatasetType) -> DatasetType:
    # Siapkan Wadah Dataset
    normalized_datasets: DatasetType = {}
  
    # Proses tiap Dataset
    for name, dataset in datasets.items():
      # Normalisasikan Dataset
      normalized_datasets[name] = self.normalize_dataset(dataset)
  
    return normalized_datasets

  # DENORMALIZE INPUTS — Nengembalikan Kumpulan Nilai Input ke Skala Asli
  def denormalize_inputs(self, values: FloatSequence) -> FloatVector:
    return [
      # Denormalisasikan tiap nilai Input Scaler
      self.input_scaler.denormalize(value)
  
      # Jalankan berdasarkan panjang values
      for value in values
    ]
  
  # DENORMALIZE TARGETS — Nengembalikan Kumpulan Nilai Target ke Skala Asli
  def denormalize_targets(self, values: FloatSequence) -> FloatVector:
    return [
      # Denormalisasikan tiap nilai Target Scaler
      self.target_scaler.denormalize(value)
  
      # Jalankan berdasarkan panjang values
      for value in values
    ]
  
  # DENORMALIZE DATASET — Nengembalikan sebuah Dataset ke Skala Asli
  def denormalize_dataset(self, dataset: Dataset) -> Dataset:
    # Siapkan Wadah Dataset
    denormalized_dataset: Dataset = []
  
    # Proses tiap Data
    for inputs, targets in dataset:
      # Denormalisasi Input
      denormalized_inputs = self.denormalize_inputs(inputs)
  
      # Denormalisasi Target
      denormalized_targets = self.denormalize_targets(targets)
  
      # Simpan Data yg sdh di kembalikan
      denormalized_dataset.append(
        (denormalized_inputs, denormalized_targets)
      )
  
    return denormalized_dataset
  
  # DENORMALIZE DATASETS — Mengembalikan byk Dataset ke Skala Asli
  def denormalize_datasets(self, datasets: DatasetType) -> DatasetType:
    # Siapkan Wadah Dataset
    denormalized_datasets: DatasetType = {}
  
    # Proses tiap Dataset
    for name, dataset in datasets.items():
      # Denormalisasikan Dataset
      denormalized_datasets[name] = self.denormalize_dataset(dataset)
  
    return denormalized_datasets    

  # FIT DATASET — Mencari nilai minimum & maximum dari training data
  def fit_dataset(self, dataset: Dataset) -> None:
    # Wadah seluruh nilai Input
    input_values: FloatVector = []

    # Wadah seluruh nilai Target
    target_values: FloatVector = []

    # Ambil Data dari Dataset
    for inputs, targets in dataset:
      # Gabungkan seluruh data Input
      input_values.extend(inputs)

      # Gabungkan seluruh data Target
      target_values.extend(targets)

    # Fit Gabungan Input ke Normalizer
    self.input_scaler.fit(input_values)

    # Fit Gabungan Target ke Normalizer
    self.target_scaler.fit(target_values)

  # PREPROCESS — Menjalankan seluruh Proses Preprocessor
  def preprocess(self, datasets: DatasetType) -> DatasetType:
    # Fit Normalizer
    self.fit_dataset(datasets["training"])

    # Normalize Datasets
    return self.normalize_datasets(datasets)
```


15. `src/training/trainer.py` ini jg seusiakan dgn impelementasi vectorized & runtime dan sepertinya kalau memang lbh baik sesuikan jg, oke

```
# Bagian untuk melakukan pelatihan pd Model

import copy

from .batch import Batch
from src.core import Layer, Model

from src.evaluation import (
  Loss,
  calculate_metrics,
)

from src.utils.custom_types import (
  IntPositive,
  Dataset,
  Metrics,
  FitHistory,
  FitResult,
)

class Trainer:
  # CONSTRUCTOR — Initialization
  def __init__(
    self,
    model: Model,
    batch: Batch,
    loss: Loss,
  ) -> None:
    # Simpan Model
    self.model: Model = model

    # Simpan Batch
    self.batch: Batch = batch

    # Simpan Loss Function
    self.loss: Loss = loss

  # TRAINING — Melatih Model Selama 1 Epoch
  def training(self, dataset: Dataset, batch_size: IntPositive) -> Metrics:
    # Validasi
    if not dataset:
      raise ValueError("Dataset yg digunakan tdk boleh kosong.")

    # Buat Mini-Batch
    batches = self.batch.create_batches(dataset, batch_size)

    # Proses setiap Mini-Batch
    for batch in batches:
      # Proses tiap data dlm Mini-Batch
      for inputs, targets in batch:
        # Forward
        predictions = self.model.forward(inputs)

        # Gradient Loss
        gradient_outputs = self.loss.gradient(targets, predictions)

        # Backward
        self.model.backward(gradient_outputs)

      # Average Gradient
      self.model.average_gradient(len(batch))

      # Update Weight & Bias
      self.model.step()

      # Reset Gradient
      self.model.reset_gradient()

    # Kembalikan hasil Evaluasi Model setelah 1 Epoch
    return self.evaluation(dataset)

  # EVALUATION — Mengukur Performa Model pd Dataset
  def evaluation(self, dataset: Dataset) -> Metrics:
    # Validasi
    if not dataset:
      raise ValueError("Dataset yg digunakan tdk boleh kosong.")

    # Total Metrics Initialization
    total_mse = total_mae = total_rmse = 0.0

    # Total data dari Dataset
    dataset_size = len(dataset)

    # Proses tiap Data
    for inputs, targets in dataset:
      # Forward
      predictions = self.model.forward(inputs)

      # Hitung Metrics
      metrics = calculate_metrics(targets, predictions)

      # Total masing² Metrics
      total_mse += metrics["MSE"]
      total_mae += metrics["MAE"]
      total_rmse += metrics["RMSE"]

    # Rata² Metrics seluruh Dataset
    return {
      "MSE": total_mse / dataset_size,
      "MAE": total_mae / dataset_size,
      "RMSE": total_rmse / dataset_size,
    }

  # SAVE MODEL —  Menyimpan State Model
  def save_model(self) -> list[Layer]:
    # Buat Salinan Mendalam dari Layer
    return copy.deepcopy(self.model.layers)

  # RESTORE MODEL — Mengembalikan State Model
  def restore_model(self, best_state: list[Layer]) -> None:
    # Simpan lagi Salinan Layer
    self.model.layers = copy.deepcopy(best_state)

  # FIT — Melakukan Seluruh Proses Model secara keseluruhan
  def fit(
    self,
    training_data: Dataset,
    validation_data: Dataset,
    epochs: IntPositive,
    batch_size: IntPositive,
    patience: IntPositive = 10,
  ) -> FitResult:
    # Histroy Hasil
    history: FitHistory = {
      # List nomor Epoch
      "epochs": [],

      # Menyimpan Metrics dari Training
      "training_metrics": {
        "MSE": [],
        "MAE": [],
        "RMSE": []
      },

      # Menyimpan Metrics dari Validation
      "validation_metrics": {
        "MSE": [],
        "MAE": [],
        "RMSE": []
      },
    }

    # Data MSE terbaik 
    best_validation_mse = float("inf")   # Float positif tak terhingga

    # State Model terbaik
    best_model: list[Layer] | None = None

    # Posisi Epoch terbaik
    best_epoch: int = 0

    # Counter untuk menghitung toleransi jika Epoch tdk membaik
    patience_counter: int = 0

    # Posisi terakhir Epoch dijalankan
    last_epoch: int = 0

    # Proses Training berdasarkan Epoch
    for epoch in range(1, epochs + 1):
      # Training
      training_loss = self.training(training_data, batch_size)

      # Validation
      validation_loss = self.evaluation(validation_data)

      # Simpan Nomor Epoch
      history["epochs"].append(epoch)

      # Simpan Metrics Training
      history["training_metrics"]["MSE"].append(training_loss["MSE"])
      history["training_metrics"]["MAE"].append(training_loss["MAE"])
      history["training_metrics"]["RMSE"].append(training_loss["RMSE"])

      # Simpan metrics Validation
      history["validation_metrics"]["MSE"].append(validation_loss["MSE"])
      history["validation_metrics"]["MAE"].append(validation_loss["MAE"])
      history["validation_metrics"]["RMSE"].append(validation_loss["RMSE"])

      # Ambil Validation MSE
      validation_mse = validation_loss["MSE"]

      # Jika Model Membaik
      if (validation_mse < best_validation_mse):
        # Simpan Validation MSE yg terbaik
        best_validation_mse = validation_mse

        # Simpan Model saat ini Ke State
        best_model = self.save_model()

        # Simpan Epoch terbaik
        best_epoch = epoch

        # Reset Counter
        patience_counter = 0
      else:
        # Update Counter
        patience_counter += 1

      # Log Epoch per-100 Epoch
      if epoch % 5 == 0:
        print(f"=== Epoch {epoch} ===")
        print(f"Training MSE    : {training_loss['MSE']:.18f}")
        print(f"Training MAE    : {training_loss['MAE']:.18f}")
        print(f"Training RMSE   : {training_loss['RMSE']:.18f}")
        print(f"Validation MSE  : {validation_loss['MSE']:.18f}")
        print(f"Validation MAE  : {validation_loss['MAE']:.18f}")
        print(f"Validation RMSE : {validation_loss['RMSE']:.18f}")
        print(f"Patience        : {patience_counter}/{patience}")
        print()

      # Simpan posisi Epoch paling terakhir
      last_epoch = epoch

      # Early Stopping
      if (patience_counter >= patience):
        print("=== EARLY STOPPING ===")
        print(f"Epoch               : {last_epoch}")
        print(f"Best Epoch          : {best_epoch}")
        print(f"Best Validation MSE : {best_validation_mse:.18f}")
        print()

        # Hentikan Epoch
        break

    # Restore State Best Model
    if (best_model is not None):
      self.restore_model(best_model)

    return {
      "history": history,
      "best_epoch": best_epoch,
      "last_epoch": last_epoch,
      "best_validation_mse": best_validation_mse,
    }
```


Oke itu urutan perbaikan utama yg kita akan fokuskan satu persatu per topik 1 pembaruan dan kemudian jika kode sdh bagus baru Testing sesuai urutan masing masing, oke


Note:
- Tolong buat agar alur Kode kita ini sesuai dgn Algoritma (Rumus) NN / Deep learning termasuk Implelemntasi Di library utama nya,
- Tetapi ingat krn belajar sesuaikan dgn versi belajar, dan Typeing yg eksplisist agar ketika di lihat mudah dimengerti, oke