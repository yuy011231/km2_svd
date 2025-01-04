from typing import Sequence
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from km2_svd.plotter.common_plotter import CommonPlotter
from scipy.optimize import curve_fit


class ITCPlotter(CommonPlotter):
    def __init__(self, target_df: pd.DataFrame, ax: Axes):
        super().__init__(target_df, ax)

    def axis_setting(self):
        """グラフの軸設定を行います。"""
        self.ax.set_xlabel("Time[sec]", size="large")
        self.ax.set_ylabel("μcal/sec", size="large")
        self.ax.minorticks_on()
        self.ax.grid(which="major", color="black", alpha=0.5)
        self.ax.grid(which="minor", color="gray", linestyle=":")

    def plot(self):
        """指定データをプロットします。"""
        sns.lineplot(x="time", y="power", data=self.target_df, ax=self.ax)


class TitrationPlotters:
    def __init__(self, target_dfs: Sequence[pd.DataFrame]):
        self.plotters = [TitrationPlotter(df) for df in target_dfs]

    def plot(self):
        for plotter in self.plotters:
            plotter.plot()

    def save_fig(self, output_path: Path):
        output_path.mkdir(parents=True, exist_ok=True)
        for index, plotter in enumerate(self.plotters):
            plotter.save_fig(output_path / f"titration-{index}.png")


class TitrationPlotter(CommonPlotter):
    def __init__(self, target_df: pd.DataFrame, ax: Axes):
        super().__init__(target_df, ax)

    def axis_setting(self):
        """グラフの軸設定を行います。"""
        self.ax.set_xlabel("Time[sec]", size="large")
        self.ax.set_ylabel("μcal/sec", size="large")
        self.ax.minorticks_on()
        self.ax.grid(which="major", color="black", alpha=0.5)
        self.ax.grid(which="minor", color="gray", linestyle=":")

    def plot(self):
        """指定データをプロットします。"""
        sns.lineplot(x="time", y="power", data=self.target_df, ax=self.ax)


class PowerPlotter(CommonPlotter):
    def __init__(self, target_df: pd.DataFrame, ax: Axes):
        super().__init__(target_df, ax)

    def axis_setting(self):
        """グラフの軸設定を行います。"""
        self.ax.set_xlabel("molar ratio", size="large")
        self.ax.set_ylabel("kJ/mol", size="large")
        self.ax.minorticks_on()
        self.ax.grid(which="major", color="black", alpha=0.5)
        self.ax.grid(which="minor", color="gray", linestyle=":")

    def plot(self):
        """指定データをプロットします。"""
        def model(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))
        
        sns.scatterplot(x="count", y="diff", data=self.target_df, ax=self.ax)

        # coeffs = np.polyfit(self.target_df["count"], self.target_df["diff"], deg=7)  # 3次多項式
        # polynomial = np.poly1d(coeffs)
        # x_fit = np.linspace(min(self.target_df["count"]), max(self.target_df["count"]), 100)
        # y_fit = polynomial(x_fit)
        x_data=self.target_df["count"]
        y_data=self.target_df["diff"]
        popt, _ = curve_fit(model, x_data, y_data, p0=[max(y_data), 1, np.median(x_data)], maxfev=5000)
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = model(x_fit, *popt)

        # self.ax.plot(x_fit, y_fit, color='red', label='Fitted Curve')
        # self.ax.legend()
