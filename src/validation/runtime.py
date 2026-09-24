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
