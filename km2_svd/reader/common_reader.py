from abc import ABC, abstractmethod

import numpy as np


class CommonReader(ABC):
    @property
    @abstractmethod
    def split_times(self):
        pass

    @property
    @abstractmethod
    def split_powers(self):
        pass

    @property
    @abstractmethod
    def split_degrees(self):
        pass

    @property
    @abstractmethod
    def split_count(self):
        pass
    