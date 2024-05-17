import numpy as np
from typing import (
    Iterator,
)


class SlideWindow:
    def __init__(
        self, target: Iterator[float], s_window_size: int = 10, s_window_step=1
    ):
        self.__target = target
        self.__s_window_size = s_window_size
        self.__s_window_step = s_window_step

    @property
    def partial_time_count(self):
        return int(((len(self.__target) - self.__s_window_size)) + 1)

    @property
    def partial_time_series_data(self) -> np.ndarray[np.ndarray[float]]:
        """スライドウィンドウで分割したデータを返却。

        Returns:
            np.ndarray[np.ndarray[float]]: 時系列分割データ
        """
        # TODO: partial_time_count＜0を考慮したい = 窓サイズがでかすぎる
        return np.array(
            [
                self.__target[offset : offset + self.__s_window_size]
                for offset in range(0, self.partial_time_count, self.__s_window_step)
            ]
        )

    @property
    def s_window_size(self):
        return self.__s_window_size
