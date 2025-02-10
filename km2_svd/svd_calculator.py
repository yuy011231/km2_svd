from typing import Any, Tuple
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.integrate import simpson
import scipy.signal as signal
from scipy.optimize import curve_fit


class SvdCalculator:
    def __init__(self, data_df: pd.DataFrame, slide_window_size: int, slide_window_step: int, threshold: int, *, is_linear: bool = True):
        self.data_df = data_df
        self.slide_window_size = self._correction_slide_window(slide_window_size)
        self.slide_window_step = slide_window_step
        self.threshold = threshold
        self.is_linear = is_linear
        self.start_idx = 0
        self.end_idx = 0
        self.detect_peak_range()
        
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
    
    def detect_peak_range(self)->Tuple[int, int]:
        self.start_idx = 0
        self.end_idx = 35
    
    def set_start_idx(self, start_idx: int):
        self.start_idx = start_idx
    
    def set_end_idx(self, end_idx: int):
        self.end_idx = end_idx
    
    def get_baseline_df(self):
        peak_df= self.get_reproduction_peak_df()
        
        if self.start_idx < 2:
            self.start_idx = 0
            pre_peak_baseline = None
        else:
            pre_peak_baseline= self.get_range_baseline(0, self.start_idx)
        post_peak_baseline = self.get_range_baseline(self.end_idx, -1)
        peak_baseline = self.get_peak_baseline(self.start_idx, self.end_idx, peak_df)
        
        return pd.DataFrame({
                "time": self.data_df["time"], 
                "pre_peak_baseline": pre_peak_baseline, 
                "post_peak_baseline": post_peak_baseline, 
                "peak_baseline": peak_baseline
            })   
    
    def get_peak_baseline(self, start_idx: int, end_idx: int, peak_df: pd.DataFrame):
        def linear(x, a, b):
            return a * x + b
        x1, x2 = self.data_df["time"].iloc[start_idx], self.data_df["time"].iloc[end_idx]
        y1, y2 = peak_df["peak"].iloc[start_idx], peak_df["peak"].iloc[end_idx]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        
        return linear(self.data_df["time"], m, b)
    
    def get_range_baseline(self, start_idx: int, end_idx: int):
        def linear(x, a, b):
            return a * x + b
        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c
        time = self.data_df["time"]
        power = self.data_df["power"]
        start_time, end_time = time.iloc[start_idx], time.iloc[end_idx]
        mask = (time >= start_time) & (time <= end_time)
        time_range = time[mask]
        power_range = power[mask]
        params_lin, _ = curve_fit(linear, time_range, power_range)
        params_quad, _ = curve_fit(quadratic, time_range, power_range)
        baseline_linear = linear(time, *params_lin)
        baseline_quadratic = quadratic(time, *params_quad)
        if self.is_linear:
            return baseline_linear
        else:
            return baseline_quadratic
    
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
    
    def calculation_peak_baseline_diff(self) -> float:
        u, s, v = self.svd()
        peak = self._reconstruct_from_windows(self._reproduction_peak(u, s, v), len(self.data_df))
        time = self.data_df["time"].reset_index(drop=True)
        x_axis = np.linspace(time.iloc[0], time.iloc[-1], len(peak))
        
        baseline_df = self.get_baseline_df()
        pre_peak_baseline = baseline_df["pre_peak_baseline"].to_numpy()
        peak_baseline = baseline_df["peak_baseline"].to_numpy()
        post_peak_baseline = baseline_df["post_peak_baseline"].to_numpy()
        
        if self.start_idx < 2:
            integral_pre_peak_baseline = 0
        else:
            integral_pre_peak_baseline = simpson(y=pre_peak_baseline[:self.start_idx], x=x_axis[:self.start_idx])
        if self.end_idx - self.start_idx < 2:
            integral_peak_baseline = 0
        else:
            integral_peak_baseline = simpson(y=peak_baseline[self.start_idx:self.end_idx], x=x_axis[self.start_idx:self.end_idx])  
        if len(post_peak_baseline) - self.end_idx < 2:
            integral_post_peak_baseline = 0
        else:
            integral_post_peak_baseline = simpson(y=post_peak_baseline[self.end_idx:], x=x_axis[self.end_idx:])
        return simpson(y=peak, x=x_axis) - (integral_pre_peak_baseline + integral_peak_baseline + integral_post_peak_baseline)
