import numpy as np
from scipy import integrate
from km2_svd.svd_calculator import SvdCalculator


class PeakNoiseControl:
    def __init__(self, calculator: SvdCalculator, peak_threshold: int = 4):
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
    def peak_component(self) -> np.ndarray:
        return self.__peak_component

    @property
    def noise_component(self) -> np.ndarray:
        return self.__noise_component

    def integrated_difference(self, times_s_window) -> np.ndarray:
        """ピークとノイズの積分値の差を返却します。"""
        return np.array(
            integrate.simps(
                self.peak_component,
                np.linspace(
                    times_s_window[0], times_s_window[-1], len(self.peak_component)
                ),
            )
            - integrate.simps(
                self.noise_component,
                np.linspace(
                    times_s_window[0], times_s_window[-1], len(self.noise_component)
                ),
            )
        )
