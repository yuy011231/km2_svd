
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from km2_svd.plotter.common_plotter import CommonPlotter


class ITCPlotter(CommonPlotter):
    def __init__(self, target_df:pd.DataFrame):
        super().__init__(target_df)
    
    def axis_setting(self):
        """グラフの軸設定を行います。
        """
        self.ax.set_xlabel("Time[sec]", size="large")
        self.ax.set_ylabel("μcal/sec", size="large")
        self.ax.minorticks_on()
        self.ax.grid(which="major", color="black", alpha=0.5)
        self.ax.grid(which="minor", color="gray", linestyle=":")

    def plot(self):
        """指定データをプロットします。
        """
        sns.lineplot(x="time", y="power", data=self.target_df, ax=self.ax)
