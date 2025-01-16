from pathlib import Path
from km2_svd.reader.common_reader import CommonReader
from km2_svd.reader.itc_reader import ItcReader
from km2_svd.svd_calculator import SvdCalculator
from km2_svd.plotter.titration_plotter import TitrationPlotters
#読み込み、整形
#reader = ItcReader("data/210315C.ITC")
reader = ItcReader("data/210107C.ITC")
svd=SvdCalculator(reader, 10)

print(len(reader.split_powers[0]))
print(len(reader.split_times))

print(svd.calculation_peak_noise_diff(4))

plotter=TitrationPlotters(svd.get_noise_removal_power(0))
plotter.plot()
plotter.save_fig(Path("./"))

plotter=reader.get_itc_plotter()
plotter.plot()
plotter.save_fig("output.png")

plotter_svd=svd.get_power_plotter()
plotter_svd.plot()
plotter_svd.save_fig("output_1.png")
