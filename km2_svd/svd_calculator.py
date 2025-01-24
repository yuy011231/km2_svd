import numpy as np
from matplotlib.axes import Axes
from typing import Iterator
import pandas as pd
from scipy.integrate import simpson
from km2_svd.reader.common_reader import CommonReader

from km2_svd.plotter.itc_plotter import PowerPlotter


class SvdCalculator:
    def __init__(
        self,
        reader: CommonReader,
        s_window_size: int,
        s_window_step: int = 1,
    ):
        self._reader = reader
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
    
    def _reproduction(self, u, s, v, threshold: int):
        k = np.sum(s >= threshold)
        s_denoised = np.zeros_like(s)
        s_denoised[:k] = s[:k]
        # s_denoised_matrix = np.diag(s_denoised)
        s_denoised_matrix = np.zeros((u.shape[1], v.shape[0]))  # 必要に応じてゼロで埋める
        np.fill_diagonal(s_denoised_matrix, s_denoised[:min(u.shape[1], v.shape[0])])
        denoised = np.dot(u, np.dot(s_denoised_matrix, v))
        return denoised

    def _peak_reproduction(self, u, s, v, peak_threshold: int):
        return sum(u[0][i] * s[i] * v[i] for i in range(peak_threshold))

    def _noise_reproduction(self, u, s, v, split_power, peak_threshold:int):
        if len(split_power) > 2 * self._s_window_size:
            return sum(
                u[0][i] * s[i] * v[i]
                for i in range(peak_threshold, self._s_window_size)
            )
        else:
            return sum(
                u[0][i] * s[i] * v[i]
                for i in range(peak_threshold, (split_power - self.s_window_size))
            )

    def _calculation_peak_noise(self, peak_threshold:int):
        peaks = []
        noises = []
        for (u, s, v), split_power in zip(self._svd(), self._reader.split_powers):
            peaks.append(self._peak_reproduction(u, s, v, peak_threshold))
            noises.append(self._noise_reproduction(u, s, v, split_power, peak_threshold))
        return peaks, noises

    def calculation_peak_noise_diff(self, peak_threshold:int):
        peaks, noises = self._calculation_peak_noise(peak_threshold)
        times = self._reader.split_times
        diff_peak_noise = []
        for peak, noise, time in zip(peaks, noises, times):
            x_axis = np.linspace(time[0], time[-1], len(peak))
            diff_peak_noise.append(
                simpson(y=peak, x=x_axis) - simpson(y=noise, x=x_axis)
            )
        return diff_peak_noise
    
    def get_power_plotter(self, ax: Axes=None):
        # TODO: ピークの閾値を変えられるようにする
        diff=self.calculation_peak_noise_diff(4)
        return PowerPlotter(pd.DataFrame({"count":range(self._reader.split_count-1), "diff":diff[1:]}), ax)

    def _reconstruct_from_windows(self, windows: list, origin_len: int):
        reconstructed = np.zeros(
            (origin_len,), dtype=np.float64
        )
        counts = np.zeros_like(reconstructed)
        for i, window in enumerate(windows):
            start = i * self._s_window_step
            end = start + self._s_window_size
            reconstructed[start:end] += window
            counts[start:end] += 1
        return reconstructed / np.maximum(counts, 1)
    
    def get_noise_removal_power(self, peak_threshold: int):
        result = []
        for (u, s, v), split_time, split_power in zip(
            self._svd(), self._reader.split_times, self._reader.split_powers
        ):
            reconstructed_power = self._reconstruct_from_windows(self._reproduction(u, s, v, peak_threshold), len(split_power))
            df = pd.DataFrame({
                "time": split_time,
                "power": reconstructed_power,
            })
            result.append(df)
        return result
