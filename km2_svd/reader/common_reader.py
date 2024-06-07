from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from itertools import islice
from logging import getLogger
from pathlib import Path
from typing import (
    Generator,
    Generic,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    overload,
)

import numpy as np


class BaseReader(ABC):
    @property
    @abstractmethod
    def times(self) -> np.ndarray[float]:
        """timeを返却します。"""

    @property
    @abstractmethod
    def power(self) -> np.ndarray[float]:
        """powerを返却します。"""

    @property
    @abstractmethod
    def degree(self) -> np.ndarray[float]:
        """degreeを返却します。"""

    @abstractmethod
    def __len__(self) -> int:
        """"""
