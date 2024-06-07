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
from km2_svd.slide_window import SlideWindow


class SvdCalculator:
    def __init__(self, s_window: SlideWindow):
        self.__s_window = s_window
        u, s, vh = np.linalg.svd(s_window.partial_time_series_data, full_matrices=True)
        self.__left_singular_vectors = u
        self.__singular_vectors = s
        self.__right_singular_vectors_transpose = vh

    @property
    def singular_vectors(self) -> np.ndarray:
        """特異値ベクトルを返却します

        Returns:
            np.ndarray: 特異値ベクトル
        """
        return self.__singular_vectors

    @property
    def left_singular_vectors(self) -> np.ndarray:
        """左特異ベクトルを返却します。

        Returns:
            np.ndarray: 左特異ベクトル
        """
        return self.__left_singular_vectors

    @property
    def right_singular_vectors_transpose(self) -> np.ndarray:
        """右特異ベクトルの転置を返却します。

        Returns:
            np.ndarray: 右特異ベクトルの転置
        """
        return self.__right_singular_vectors_transpose
    
    @property
    def s_window_size(self) -> int:
        return self.__s_window.s_window_size
