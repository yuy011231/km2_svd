from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import seaborn as sns
from km2_svd.plotter.common_plotter import CommonPlotter
from km2_svd.svd_calculator import SvdCalculator
from km2_svd.plotter.axis_settings import singular_value_axis_setting, peak_noise_axis_setting


class SvdPlotter:
    def __init__(self, svd_calculator: SvdCalculator, singular_value_ax: Axes, peak_ax: Axes, noise_ax: Axes):
        self.svd_calculator = svd_calculator
        self.singular_value_plotter = SingularValuePlotter(self.singular_value_df(), singular_value_ax)
        self.peak_plotter = PeakPlotter(self.svd_calculator.get_reproduction_peak_df(), peak_ax)
        self.noise_plotter = NoisePlotter(self.svd_calculator.get_reproduction_noise_df(), noise_ax)
    
    def singular_value_df(self):
        _, s, _ = self.svd_calculator.svd()
        return pd.DataFrame({"rank": range(1, len(s)+1), "singular_value": s})
    
    def singular_value_plot(self):
        self.singular_value_plotter.plot()
    
    def peak_plot(self):
        self.peak_plotter.target_df = self.svd_calculator.get_reproduction_peak_df()
        self.peak_plotter.plot()
        
    def noise_plot(self):
        self.noise_plotter.target_df = self.svd_calculator.get_reproduction_noise_df()
        self.noise_plotter.plot()


class SingularValuePlotter(CommonPlotter):
    def __init__(self, target_df: pd.DataFrame, ax: Axes):
        super().__init__(target_df, ax)

    def axis_setting(self):
        self.ax.set_yscale("log")
        singular_value_axis_setting(self.ax)

    def plot(self):
        sns.scatterplot(
            x="rank",
            y="singular_value",
            data=self.target_df,
            ax=self.ax,
            s=100,
            color="blue",
        )
        self.axis_setting()


class PeakPlotter(CommonPlotter):
    def __init__(self, target_df: pd.DataFrame, ax: Axes):
        super().__init__(target_df, ax)

    def axis_setting(self):
        peak_noise_axis_setting(self.ax)

    def plot(self):
        sns.lineplot(x="time", y="peak", data=self.target_df, ax=self.ax)


class NoisePlotter(CommonPlotter):
    def __init__(self, target_df: pd.DataFrame, ax: Axes):
        super().__init__(target_df, ax)

    def axis_setting(self):
        peak_noise_axis_setting(self.ax)

    def plot(self):
        sns.lineplot(x="time", y="noise", data=self.target_df, ax=self.ax)

class PeakNoiseDiffPlotter(CommonPlotter):
    def __init__(self, svd_calculators: list[SvdCalculator], ax: Axes):
        self.svd_calculators = svd_calculators
        self.peak_noise_diff_list = [svd_calculator.calculation_peak_noise_diff() for svd_calculator in svd_calculators]
        self.counts = range(1, len(svd_calculators)+1)
        target_df = pd.DataFrame({"count": self.counts, "peak_noise_diff": self.peak_noise_diff_list})
        super().__init__(target_df, ax)

    def axis_setting(self):
        peak_noise_axis_setting(self.ax)

    def plot(self):
        sns.lineplot(x="count", y="peak_noise_diff", data=self.target_df, ax=self.ax)
