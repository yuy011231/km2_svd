from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path


class CommonPlotter(ABC):
    def __init__(self, target_df: pd.DataFrame):
        self.fig = plt.figure(figsize=(12, 7))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.target_df = target_df
        self.axis_setting()

    @abstractmethod
    def axis_setting(self):
        """axisの設定を行います。"""

    @abstractmethod
    def plot(self):
        """グラフを描画します。"""

    def replot(self):
        self.plot()

    def save_fig(self, output_path: Path):
        self.fig.savefig(output_path)
