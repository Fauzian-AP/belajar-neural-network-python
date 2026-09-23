"""
Bagian Pengelolaan Runtime Validation pd Project Neural Network.

Menyediakan Runtime Validation untuk:

1. Function
2. Method

Pydantic digunakan sbg Engine Validation.

Module ini jg mengubah ValidationError menjadi Error Message yg lbh spesifik dan mdh dibaca.
"""

import numpy as np

from functools import wraps
from inspect import signature
from pydantic_core import ErrorDetails
from numpy.typing import NDArray
from typing import (
  Any,
  Final,
  Callable,
  TypeGuard,
  TypeVar,
  cast,
  get_type_hints,
  is_typeddict,
)

from pydantic import (
  BaseModel,
  ConfigDict,
  TypeAdapter,
  ValidationError,
  validate_call,
)


# ======================
# === TYPE VARIABLES ===
# ======================

F = TypeVar("F", bound=Callable[..., Any])


# =================
# === CONSTANTS ===
# =================

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


# ==============
# === HELPER ===
# ==============

# FORMAT VALUE — Memformat Value agar informasi penting tetap terlihat.

def _format_value(value: Any) -> str:
  # Nilai Array NumPy
  if isinstance(value, np.ndarray):
    # Tandai Value sbg NDArray agar informasi shape & dtype dpt ditampilkan dgn jelas
    value = cast(NDArray[Any], value)

    return f"{type(value).__name__}(shape={value.shape}, dtype={value.dtype})"

  # Nilai String
  if isinstance(value, str):
    return repr(value)

  # Nilai tipe lain
  return repr(value)


# FORMAT TYPE — Memformat Nama Tipe dari sebuah Value.

def _format_type(value: Any) -> str:
  # Jika Value adalah Class, tampilkan nama Class
  if isinstance(value, type):
    return value.__name__

  # Jika Value adalah Instance, tampilkan nama tipe Instance
  return type(value).__name__


# ERROR MESSAGE — Mengambil Pesan Error dari Pydantic.

def _get_error_message(detail: ErrorDetails) -> str:
  # Ambil Context Error
  context = detail.get("ctx")

  if context is not None:
    # Ambil Error asli dari Custom Validator
    original_error = context.get("error")

    # Jika Error asli berupa Exception, gunakan pesan dari Error tersebut
    if isinstance(original_error, Exception):
      return str(original_error)

  # Jika bkn, gunakan pesan bawaan Pydantic.
  return str(detail.get("msg", "Nilai tidak valid."))


# FORMAT ARGUMENT DETAIL — Memformat 1 Error Argument.

def _format_argument_detail(
  detail: ErrorDetails,
  parameter_names: tuple[str, ...],
  positional_offset: int,
) -> str:
  # Ambil Lokasi Error
  location = detail.get("loc", ())

  # Ambil nilai yang divalidasi
  input_value = detail.get("input")

  # Ambil pesan Error
  message = _get_error_message(detail)

  # Ambil index Positional Argument
  raw_index = (
    location[0]
    if location and isinstance(location[0], int)
    else None
  )

  # Sesuaikan index terhadap Self / Cls
  # jika Function yg divalidasi adalah Method
  index = (
    raw_index - positional_offset
    if raw_index is not None
    else None
  )

  # Tentukan nama Argument

  if (
    index is not None
    and 0 <= index < len(parameter_names)
  ):
    argument_name = parameter_names[index]

  elif location:
    argument_name = str(location[0])

  else:
    argument_name = "unknown"

  # Format Detail
  return (
    f"{'Argument':<{LABEL_WIDTH}} : {argument_name}\n"
    f"{'Received':<{LABEL_WIDTH}} : {_format_value(input_value)}\n"
    f"{'Type':<{LABEL_WIDTH}} : {_format_type(input_value)}\n"
    f"{'Message':<{LABEL_WIDTH}} : {message}"
  )


# FORMAT RETURN DETAIL — Memformat 1 Error Return.

def _format_return_detail(detail: ErrorDetails, result: Any) -> str:
  # Ambil pesan Error
  message = _get_error_message(detail)

  # Format Detail
  return (
    f"{'Return':<{LABEL_WIDTH}} : {_format_value(result)}\n"
    f"{'Type':<{LABEL_WIDTH}} : {_format_type(result)}\n"
    f"{'Message':<{LABEL_WIDTH}} : {message}"
  )


# FORMAT ARGUMENT ERRORS — Memformat seluruh Error Argument.

def _format_argument_errors(
  error: ValidationError,
  parameter_names: tuple[str, ...],
  positional_offset: int,
) -> str:
  # Format setiap Error Argument
  details = [
    _format_argument_detail(
      detail,
      parameter_names,
      positional_offset,
    )
    for detail in error.errors()
  ]

  return "\n\n".join(details)


# FORMAT RETURN ERRORS — Memformat seluruh Error Return.

def _format_return_errors(error: ValidationError, result: Any) -> str:
  # Format setiap Error Return.
  details = [
    _format_return_detail(
      detail,
      result,
    )
    for detail in error.errors()
  ]

  return "\n\n".join(details)


# IS PYDANTIC MODEL TYPE — Menentukan apakah Value merupakan Class turunan BaseModel.
def _is_pydantic_model_type(
  value: object,
) -> TypeGuard[type[BaseModel]]:

  return (
    isinstance(value, type)
    and issubclass(value, BaseModel)
  )


# IS TYPED DICT TYPE — Menentukan apakah Value merupakan Class TypedDict.
def _is_typed_dict_type(
  value: object,
) -> bool:

  return is_typeddict(value)


# CREATE TYPE ADAPTER — Membuat TypeAdapter dari
# Type Annotation yang diperoleh secara Runtime.
def _create_type_adapter(
  annotation: object,
  config: ConfigDict | None = None,
) -> TypeAdapter[Any]:

  # Type Annotation berasal dari Runtime sehingga
  # tidak selalu dapat diinfer secara statis oleh Pyright.
  runtime_annotation = cast(
    Any,
    annotation,
  )

  # BaseModel / TypedDict tidak membutuhkan Config tambahan.
  if config is None:
    return TypeAdapter[Any](
      runtime_annotation,
    )

  # Type biasa / custom type menggunakan Config.
  return TypeAdapter[Any](
    runtime_annotation,
    config=config,
  )


# CREATE RETURN ADAPTER — Membuat Adapter Validasi Return
# dengan Penanganan Khusus untuk BaseModel dan TypedDict.
def _create_return_adapter(
  return_type: Any,
  config: ConfigDict,
) -> TypeAdapter[Any]:

  # Normalisasi menjadi object agar Type Checker
  # tidak membawa Any | type[Unknown].
  annotation: object = return_type

  # BaseModel memiliki Schema dan Configurasi sendiri.
  if _is_pydantic_model_type(annotation):
    return _create_type_adapter(annotation)

  # TypedDict memiliki Schema sendiri.
  if _is_typed_dict_type(annotation):
    return _create_type_adapter(annotation)

  # Tipe umum / custom menggunakan Configurasi.
  return _create_type_adapter(
    annotation,
    config,
  )


# =============================
# === RUNTIME ERROR BUILDER ===
# =============================

# BUILD RUNTIME ERROR — Membangun RuntimeFunctionError
# atau RuntimeMethodError dari Detail Error yang sudah diformat.
def _build_runtime_error(
  error_type: type[RuntimeValidationError],
  callable_object: Callable[..., Any],
  label: str,
  details: str,
) -> RuntimeValidationError:

  return error_type(
    f"Error ({error_type.__name__}):\n\n"
    f"{label} : {callable_object.__name__}\n\n"
    f"{details}"
  )


# ==========================
# === RUNTIME VALIDATION ===
# ==========================

# RUNTIME CALLABLE — Mesin utama untuk membuat Decorator
# yang menangani Validasi Function maupun Method.
def _runtime_callable(
  error_type: type[RuntimeValidationError],
  label: str,
  *,
  strict: bool = False,
  skip_first_parameter: bool = False,
) -> Callable[[F], F]:

  # DECORATOR — Menerima Function/Method yang akan
  # diberi Runtime Validation.
  def decorator(function: F) -> F:

    # Ambil semua nama Parameter dari Function/Method.
    parameter_names = tuple(
      signature(function).parameters
    )

    # Method memiliki self/cls sebagai parameter pertama.
    # Parameter tersebut tidak ditampilkan sebagai Argument
    # milik user.
    positional_offset = 0

    if skip_first_parameter:
      parameter_names = parameter_names[1:]
      positional_offset = 1

    # Ambil Return Type dari Annotation Function.
    return_type = get_type_hints(
      function,
      include_extras=True,
    ).get(
      "return",
      Any,
    )

    # Tentukan aturan validasi Pydantic.
    config = ConfigDict(
      strict=strict,
      arbitrary_types_allowed=True,
    )

    # VALIDATOR ARGUMENT — Hanya melakukan Validasi
    # terhadap Argument Function.
    argument_validator = validate_call(
      config=config,
      validate_return=False,
    )(function)

    # VALIDATOR RETURN — Membuat Adapter khusus
    # untuk melakukan Validasi Return Value.
    return_adapter = _create_return_adapter(
      return_type,
      config,
    )

    # WRAPPER — Function pengganti yang dipanggil
    # oleh user setelah decorator diterapkan.
    @wraps(function)
    def wrapper(
      *args: Any,
      **kwargs: Any,
    ) -> Any:

      # =========================
      # === ARGUMENT VALIDATION ===
      # =========================

      try:

        # Jalankan Validasi Argument.
        result = argument_validator(
          *args,
          **kwargs,
        )

      except ValidationError as error:

        # Format Detail Error Argument.
        details = _format_argument_errors(
          error,
          parameter_names,
          positional_offset,
        )

        # Bangkitkan Runtime Error khusus.
        raise _build_runtime_error(
          error_type,
          function,
          label,
          details,
        ) from error

      # ========================
      # === RETURN VALIDATION ===
      # ========================

      try:

        # Jalankan Validasi Return.
        #
        # Nilai hasil validasi sengaja tidak dikembalikan.
        # Project tetap mengembalikan result asli seperti
        # perilaku kode sebelumnya.
        return_adapter.validate_python(
          result,
          strict=strict,
        )

      except ValidationError as error:

        # Format Detail Error Return.
        details = _format_return_errors(
          error,
          result,
        )

        # Bangkitkan Runtime Error khusus.
        raise _build_runtime_error(
          error_type,
          function,
          label,
          details,
        ) from error

      # Kembalikan result asli dari Function.
      return result

    # Kembalikan Wrapper sebagai Function generik F.
    return cast(F, wrapper)

  # Kembalikan Decorator dengan Konfigurasi Runtime.
  return decorator


# RUNTIME FUNCTION — Decorator untuk Runtime Validation
# pada sebuah Function.
def runtime_function(
  *,
  strict: bool = False,
) -> Callable[[F], F]:

  # Gunakan Mesin Decorator Runtime.
  return _runtime_callable(
    RuntimeFunctionError,
    "Function",
    strict=strict,
  )


# RUNTIME METHOD — Decorator untuk Runtime Validation
# pada sebuah Method.
def runtime_method(
  *,
  strict: bool = False,
) -> Callable[[F], F]:

  # Gunakan Mesin Decorator Runtime.
  return _runtime_callable(
    RuntimeMethodError,
    "Method",
    strict=strict,
    skip_first_parameter=True,
  )