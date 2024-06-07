import numpy as np
import pandas as pd
from scipy import integrate
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
from km2_svd.svd_calculator import SvdCalculator


class PeakNoiseControl:
    def __init__(self, calculator: SvdCalculator, peak_threshold: int = 5):
        self.__calculator = calculator
        self.__peak_threshold = peak_threshold
        self.__peak_component = sum(
            calculator.left_singular_vectors[:, i].reshape(-1, 1)
            * calculator.singular_vectors[i]
            * calculator.right_singular_vectors_transpose[i, :]
            for i in range(peak_threshold)
        )
        self.__noise_component = sum(
            calculator.left_singular_vectors[:, i].reshape(-1, 1)
            * calculator.singular_vectors[i]
            * calculator.right_singular_vectors_transpose[i, :]
            for i in range(peak_threshold, calculator.s_window_size)
        )
    
    @property
    def peak_component(self)->np.ndarray:
        return self.__peak_component
    
    @property
    def noise_component(self)->np.ndarray:
        return self.__noise_component
