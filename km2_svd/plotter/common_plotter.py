from abc import ABC, abstractmethod
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path


class CommonPlotter(ABC):
    def __init__(self, ax: Axes):
        self.ax = ax
        self.axis_setting()

    @abstractmethod
    def axis_setting(self):
        """axisの設定を行います。"""

    @abstractmethod
    def plot(self):
        """グラフを描画します。"""

    def replot(self):
        self.ax.clear()
        self.axis_setting()
        self.plot()

    def save_fig(self, output_path: Path):
        self.ax.figure.savefig(output_path)
