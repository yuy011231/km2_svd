from pathlib import Path
from km2_svd.reader.common_reader import CommonReader
from km2_svd.reader.itc_reader import ItcReader
from km2_svd.svd_calculator import SvdCalculator

#読み込み、整形
#reader = ItcReader("data/210315C.ITC")
reader = ItcReader("data/210107C.ITC")
svd=SvdCalculator(reader, 4, 10)

print(len(reader.split_powers[0]))
print(len(reader.split_times))

print(svd.calculation_peak_noise_diff())

plotter=reader.get_itc_plotter()
plotter.plot()
plotter.save_fig("output.png")

plotter_svd=svd.get_power_plotter()
plotter_svd.plot()
plotter_svd.save_fig("output_1.png")
