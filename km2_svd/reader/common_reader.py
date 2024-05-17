from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from itertools import islice
from logging import getLogger
from pathlib import Path
from typing import Generator, Generic, Iterator, Mapping, Optional, Sequence, Tuple, TypeVar, overload

import numpy as np

class BaseReader(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass