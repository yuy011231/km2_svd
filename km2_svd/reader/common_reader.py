from abc import ABC, abstractmethod

import numpy as np


class CommonReader(ABC):
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
