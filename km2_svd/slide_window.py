import numpy as np
import pandas as pd
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
from km2_svd.reader.common_reader import BaseReader


class SlideWindow:
    def __init__(self, reader: BaseReader, s_window_size: int = 50, s_window_step=1):
        self.__reader = reader
        self.__s_window_size = s_window_size
        self.__s_window_step = s_window_step

    @property
    def partial_time_count(self):
        return int(
            ((len(self.__reader.times) - self.__s_window_size) / self.__s_window_step)
            + 1
        )

    @property
    def partial_time_series_data(self) -> np.ndarray[np.ndarray[float]]:
        """スライドウィンドウで分割したデータを返却。

        Returns:
            np.ndarray[np.ndarray[float]]: 時系列分割データ
        """
        return np.array(
            [
                self.__reader.power[offset : offset + self.__s_window_size]
                for offset in range(0, self.partial_time_count, self.__s_window_step)
            ]
        )
    
    @property
    def s_window_size(self):
        return self.__s_window_size
