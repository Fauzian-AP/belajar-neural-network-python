# Bagian Pengelolaan Type Checking Project Neural Network

from enum import Enum
from typing import Annotated, Sequence, Union, TypedDict, TypeAlias
from pydantic import Field

# Aliases Number
Numeric = Union[int, float]
ListFloat = list[float]
SequenceFloat = Sequence[float]

# Aliases Dataset

Dataset: TypeAlias = list[
  tuple[ListFloat, ListFloat]
]
ListDataset: TypeAlias = list[Dataset]
DictDataset: TypeAlias = dict[str, Dataset]

# Struktur Metrics
class Metrics(TypedDict):
  MSE: float
  MAE: float
  RMSE: float

# Pydantic Validated
PositiveInt = Annotated[int, Field(gt=0, description="Angka bulat harus > 0")]
PositiveFloat = Annotated[float, Field(gt=0, description="Angka desimal harus > 0")]

# Enum Jenis² Activation
class ActivationType(str, Enum):
  NONE = "none"
  RELU = "ReLU"