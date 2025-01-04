import numpy as np
from matplotlib.axes import Axes
from typing import Iterator
import pandas as pd
from scipy import integrate
from km2_svd.reader.common_reader import CommonReader

from km2_svd.plotter.itc_plotter import PowerPlotter


class SvdCalculator:
    def __init__(
        self,
        reader: CommonReader,
        peak_threshold: int,
        s_window_size: int,
        s_window_step: int = 1,
    ):
        self._reader = reader
        self._peak_threshold = peak_threshold
        self._s_window_size = self._correction_s_window(s_window_size)
        self._s_window_step = s_window_step

    def _correction_s_window(self, s_window_size):
        """窓サイズの補正(データ数<窓サイズになる可能性を考慮)

        Args:
            s_window_size : 希望する窓サイズ

        Returns:
            窓サイズ
        """
        power_lens = np.array([len(s_power) for s_power in self._reader.split_powers])
        if s_window_size > np.min(power_lens):
            return np.min(power_lens)
        else:
            return s_window_size

    def _slice_by_window(self, t):
        """窓サイズでスライスする

        Args:
            t : スライス対象

        Returns:
            List[List[Any]]
        """
        return np.array(
            [
                t[i : i + self._s_window_size]
                for i in range(0, len(t) - self._s_window_size + 1, self._s_window_step)
            ]
        )

    def _svd(self):
        result = []
        for split_power in self._reader.split_powers:
            # 分割(滴定)ごとの計算を行う
            slice_power = self._slice_by_window(split_power)
            U, s, V = np.linalg.svd(slice_power, full_matrices=True)
            result.append((U, s, V))
        return result

    def _peak_reproduction(self, u, s, v):
        return sum(u[0][i] * s[i] * v[i] for i in range(self._peak_threshold))

    def _noise_reproduction(self, u, s, v, split_power):
        if len(split_power) > 2 * self._s_window_size:
            return sum(
                u[0][i] * s[i] * v[i]
                for i in range(self._peak_threshold, self._s_window_size)
            )
        else:
            return sum(
                u[0][i] * s[i] * v[i]
                for i in range(self._peak_threshold, (split_power - self.s_window_size))
            )

    def calculation_peak_noise(self):
        peaks = []
        noises = []
        for (u, s, v), split_power in zip(self._svd(), self._reader.split_powers):
            peaks.append(self._peak_reproduction(u, s, v))
            noises.append(self._noise_reproduction(u, s, v, split_power))
        return peaks, noises

    def calculation_peak_noise_diff(self):
        peaks, noises = self.calculation_peak_noise()
        times = self._reader.split_times
        diff_peak_noise = []
        for peak, noise, time in zip(peaks, noises, times):
            x_axis = np.linspace(time[0], time[-1], len(peak))
            diff_peak_noise.append(
                integrate.simps(peak, x_axis) - integrate.simps(noise, x_axis)
            )
        return diff_peak_noise
    
    def get_power_plotter(self, ax: Axes=None):
        diff=self.calculation_peak_noise_diff()
        return PowerPlotter(pd.DataFrame({"count":range(self._reader.split_count-1), "diff":diff[1:]}), ax)
