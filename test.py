from pathlib import Path

from matplotlib import pyplot as plt
from km2_svd.plotter.plotter import SvdPlotter, PeakNoiseDiffPlotter, PowerPlotter
from km2_svd.reader.common_reader import CommonReader
from km2_svd.reader.itc_reader import ItcReader
from km2_svd.svd_calculator import SvdCalculator
#読み込み、整形
reader = ItcReader("data/210107C.ITC")

svd_calculators = [SvdCalculator(reader.get_titration_df(i), 10, 1, 4) for i in range(1, reader.split_count)]

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(1, 1, 1)
plotter = PowerPlotter(reader.get_titration_df(5), ax)
plotter.plot()
plotter.save_fig("power.png")

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(1, 1, 1)
fig2 = plt.figure(figsize=(12, 7))
ax2 = fig2.add_subplot(1, 1, 1)
fig3 = plt.figure(figsize=(12, 7))
ax3 = fig3.add_subplot(1, 1, 1)
fig4 = plt.figure(figsize=(12, 7))
ax4 = fig4.add_subplot(1, 1, 1)
fig5 = plt.figure(figsize=(12, 7))
ax5 = fig5.add_subplot(1, 1, 1)

svd_calculator = SvdCalculator(reader.get_titration_df(5), 10, 1, 3)
svd_plotter = SvdPlotter(svd_calculator, ax, ax2, ax3, ax5)
svd_plotter.singular_value_plotter.plot()
svd_plotter.singular_value_plotter.save_fig("singular_value.png")

svd_plotter.peak_plot()
svd_plotter.peak_plotter.save_fig("peak1.png")

svd_plotter.noise_plot()
svd_plotter.noise_plotter.save_fig("noise1.png")

svd_plotter.peak_noise_plot()
svd_plotter.peak_noise_plotter.save_fig("peak_noise1.png")

svd_calculator.threshold=0

svd_plotter.peak_plot()
svd_plotter.peak_plotter.save_fig("peak2.png")

svd_plotter.noise_plot()
svd_plotter.noise_plotter.save_fig("noise2.png")

peak_noise_diff_plotter = PeakNoiseDiffPlotter(svd_calculators, ax4)
peak_noise_diff_plotter.plot()
peak_noise_diff_plotter.save_fig("peak_noise_diff.png")
