from typing import Any
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.integrate import simpson


class SvdCalculator:
    def __init__(self, data_df: pd.DataFrame, slide_window_size: int, slide_window_step: int, threshold: int):
        self.data_df = data_df
        self.slide_window_size = self._correction_slide_window(slide_window_size)
        self.slide_window_step = slide_window_step
        self.threshold = threshold
        
    def _correction_slide_window(self, slide_window_size: int):
        if slide_window_size > len(self.data_df):
            return len(self.data_df)
        else:
            return slide_window_size
    
    def _slice_by_window(self, powers: np.ndarray):
        return np.array(
            [
                powers[i : i + self.slide_window_size]
                for i in range(0, len(powers) - self.slide_window_size + 1, self.slide_window_step)
            ]
        )
    
    def svd(self):
        window_slice_power = self._slice_by_window(self.data_df["power"])
        U, s, V = scipy.linalg.svd(window_slice_power, full_matrices=False)
        return U, s, V
    
    def get_reproduction_peak_df(self):
        u, s, v = self.svd()
        peak = self._reconstruct_from_windows(self._reproduction_peak(u, s, v), len(self.data_df))
        return pd.DataFrame({"time": self.data_df["time"], "peak": peak})
    
    def get_reproduction_noise_df(self):
        u, s, v = self.svd()
        noise = self._reconstruct_from_windows(self._reproduction_noise(u, s, v), len(self.data_df))
        return pd.DataFrame({"time": self.data_df["time"], "noise": noise})
    
    def _reproduction_peak(self, U: np.ndarray[Any], S: np.ndarray[Any], Vt: np.ndarray[Any]):
        if self.threshold > len(S):
            raise ValueError("threshold is larger than the number of singular values")        
        S[self.threshold:] = 0
        return U @ np.diag(S) @ Vt
    
    def _reproduction_noise(self, U: np.ndarray[Any], S: np.ndarray[Any], Vt: np.ndarray[Any]):
        if self.threshold > len(S):
            raise ValueError("threshold is larger than the number of singular values")        
        S[:self.threshold] = 0
        return U @ np.diag(S) @ Vt
    
    def _reconstruct_from_windows(self, data: list, origin_len: int):
        reconstructed = np.zeros(
            (origin_len,), dtype=np.float64
        )
        counts = np.zeros_like(reconstructed)
        for i, window in enumerate(data):
            start = i * self.slide_window_step
            end = start + self.slide_window_size
            reconstructed[start:end] += window
            counts[start:end] += 1
        return reconstructed / np.maximum(counts, 1)

    def calculation_peak_noise_diff(self) -> float:
        u, s, v = self.svd()
        peak = self._reconstruct_from_windows(self._reproduction_peak(u, s, v), len(self.data_df))
        noise = self._reconstruct_from_windows(self._reproduction_noise(u, s, v), len(self.data_df))
        time = self.data_df["time"].reset_index(drop=True)
        x_axis = np.linspace(time.iloc[0], time.iloc[-1], len(peak))
        return simpson(y=peak, x=x_axis) - simpson(y=noise, x=x_axis)
