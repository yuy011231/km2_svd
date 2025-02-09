from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import seaborn as sns
from km2_svd.plotter.common_plotter import CommonPlotter
from km2_svd.svd_calculator import SvdCalculator
from km2_svd.plotter.axis_settings import (
    raw_axis_setting,
    singular_value_axis_setting,
    peak_noise_axis_setting,
)


class SvdPlotter:
    def __init__(
        self,
        svd_calculator: SvdCalculator,
        singular_value_ax: Axes,
        peak_ax: Axes,
        noise_ax: Axes,
        peak_baseline_ax: Axes,
    ):
        self.svd_calculator = svd_calculator
        self.singular_value_plotter = SingularValuePlotter(
            self.singular_value_df(), singular_value_ax
        )
        self.peak_plotter = PeakPlotter(
            self.svd_calculator.get_reproduction_peak_df(), peak_ax
        )
        self.noise_plotter = NoisePlotter(
            self.svd_calculator.get_reproduction_noise_df(), noise_ax
        )
        self.peak_baseline_plotter = PeakBaselinePlotter(
            self.svd_calculator.get_reproduction_peak_df(),
            self.svd_calculator.get_baseline_df(),
            peak_baseline_ax,
        )

    def singular_value_df(self):
        _, s, _ = self.svd_calculator.svd()
        return pd.DataFrame({"rank": range(1, len(s) + 1), "singular_value": s})

    def singular_value_plot(self):
        self.singular_value_plotter.plot()

    def peak_plot(self):
        self.peak_plotter.target_df = self.svd_calculator.get_reproduction_peak_df()
        self.peak_plotter.replot()

    def noise_plot(self):
        self.noise_plotter.target_df = self.svd_calculator.get_reproduction_noise_df()
        self.noise_plotter.replot()

    def peak_baseline_plot(self):
        self.peak_baseline_plotter.peak_df = (
            self.svd_calculator.get_reproduction_peak_df()
        )
        self.peak_baseline_plotter.baseline_df = self.svd_calculator.get_baseline_df()
        self.peak_baseline_plotter.replot()

    def save_fig(self, path: Path):
        self.singular_value_plotter.save_fig(path / "singular_value.png")
        self.peak_plotter.save_fig(path / "peak.png")
        self.noise_plotter.save_fig(path / "noise.png")
        self.peak_baseline_plotter.save_fig(path / "peak_baseline.png")

    def save_csv(self, path: Path):
        self.svd_calculator.get_reproduction_peak_df().to_csv(
            path / "peak.csv", index=False
        )
        self.svd_calculator.get_reproduction_noise_df().to_csv(path / "noise.csv")
        self.svd_calculator.get_baseline_df().to_csv(path / "baseline.csv")
        self.singular_value_df().to_csv(path / "singular_value.csv")


class SingularValuePlotter(CommonPlotter):
    def __init__(self, target_df: pd.DataFrame, ax: Axes):
        super().__init__(ax)
        self.target_df = target_df

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
        super().__init__(ax)
        self.target_df = target_df

    def axis_setting(self):
        peak_noise_axis_setting(self.ax)

    def plot(self):
        sns.lineplot(x="time", y="peak", data=self.target_df, ax=self.ax)


class NoisePlotter(CommonPlotter):
    def __init__(self, target_df: pd.DataFrame, ax: Axes):
        super().__init__(ax)
        self.target_df = target_df

    def axis_setting(self):
        peak_noise_axis_setting(self.ax)

    def plot(self):
        sns.lineplot(x="time", y="noise", data=self.target_df, ax=self.ax)


class PeakBaselinePlotter(CommonPlotter):
    def __init__(self, peak_df: pd.DataFrame, baseline_df: pd.DataFrame, ax: Axes):
        super().__init__(ax)
        self.peak_df = peak_df
        self.baseline_df = baseline_df

    def axis_setting(self):
        peak_noise_axis_setting(self.ax)

    def plot(self):
        sns.lineplot(
            x="time",
            y="peak",
            data=self.peak_df,
            ax=self.ax,
            label="Peak",
            color="blue",
        )
        sns.lineplot(
            x="time",
            y="pre_peak_baseline",
            data=self.baseline_df,
            ax=self.ax,
            label="Baseline",
            color="green",
        )
        sns.lineplot(
            x="time",
            y="post_peak_baseline",
            data=self.baseline_df,
            ax=self.ax,
            label="Baseline",
            color="green",
        )
        sns.lineplot(
            x="time",
            y="peak_baseline",
            data=self.baseline_df,
            ax=self.ax,
            label="Baseline",
            color="red",
        )
        self.ax.legend()


class PowerPlotter(CommonPlotter):
    def __init__(self, target_df: pd.DataFrame, ax: Axes):
        super().__init__(ax)
        self.target_df = target_df

    def axis_setting(self):
        peak_noise_axis_setting(self.ax)

    def plot(self):
        sns.lineplot(x="time", y="power", data=self.target_df, ax=self.ax)


class PeakNoiseDiffPlotter(CommonPlotter):
    def __init__(self, svd_calculators: list[SvdCalculator], ax: Axes):
        self.svd_calculators = svd_calculators
        self.peak_noise_diff_list = [
            svd_calculator.calculation_peak_noise_diff()
            for svd_calculator in svd_calculators
        ]
        self.counts = range(1, len(svd_calculators) + 1)
        self.target_df = pd.DataFrame(
            {"count": self.counts, "peak_noise_diff": self.peak_noise_diff_list}
        )
        super().__init__(ax)

    def axis_setting(self):
        peak_noise_axis_setting(self.ax)

    def plot(self):
        sns.lineplot(x="count", y="peak_noise_diff", data=self.target_df, ax=self.ax)


class RawDataPlotter(CommonPlotter):
    def __init__(self, target_df: pd.DataFrame, ax: Axes):
        super().__init__(ax)
        self.target_df = target_df

    def axis_setting(self):
        raw_axis_setting(self.ax)

    def plot(self):
        sns.lineplot(x="time", y="power", data=self.target_df, ax=self.ax)

    def save_csv(self, path: Path):
        self.target_df.to_csv(path / "raw_data.csv", index=False)
