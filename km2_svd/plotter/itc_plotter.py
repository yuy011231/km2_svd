from typing import Sequence
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from km2_svd.plotter.common_plotter import CommonPlotter
from km2_svd.plotter.axis_settings import itc_axis_setting, titration_axis_setting, power_axis_setting
from scipy.optimize import curve_fit


class ITCPlotter(CommonPlotter):
    def __init__(self, target_df: pd.DataFrame, ax: Axes):
        super().__init__(target_df, ax)
        self.target_df = target_df

    def axis_setting(self):
        """グラフの軸設定を行います。"""
        itc_axis_setting(self.ax)

    def plot(self):
        """指定データをプロットします。"""
        sns.lineplot(x="time", y="power", data=self.target_df, ax=self.ax)
