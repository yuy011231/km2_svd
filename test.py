from pathlib import Path

from matplotlib import pyplot as plt
from km2_svd.reader.common_reader import CommonReader
from km2_svd.reader.itc_reader import ItcReader
from km2_svd.svd_calculator import SvdCalculator
from km2_svd.plotter.titration_plotter import TitrationPlotters
#読み込み、整形
reader = ItcReader("data/210107C.ITC")

print(len(reader.split_powers[0]))
print(reader.split_powers)
print(len(reader.split_times))

svd_calculator = SvdCalculator(reader.get_titration_df(5), 10, 1, 3)

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(1, 1, 1)
fig1 = plt.figure(figsize=(12, 7))
ax2 = fig1.add_subplot(1, 1, 1)
fig2 = plt.figure(figsize=(12, 7))
ax3 = fig2.add_subplot(1, 1, 1)

svd_plotter = SvdPlotter(svd_calculator, ax, ax2, ax3)
svd_plotter.singular_value_plotter.plot()
svd_plotter.singular_value_plotter.save_fig("singular_value.png")

svd_plotter.peak_plotter.plot()
svd_plotter.peak_plotter.save_fig("peak.png")

svd_plotter.noise_plotter.plot()
svd_plotter.noise_plotter.save_fig("noise.png")
