from km2_svd.reader.common_reader import CommonReader
from km2_svd.reader.itc_reader import ItcReader
from km2_svd.slide_window import SlideWindow
from km2_svd.svd_calculator import MultiSvdCalculator
from km2_svd.peak_noise_control import PeakNoiseControl

#読み込み、整形
reader = ItcReader("data/210315C.ITC")
#reader = ItcReader("data/210107C.ITC")
print(len(reader.split_power))
print(len(reader.split_times))
print(len(reader.split_times[0]))
reader.plot_fig()

#スライド窓
power_s_windows = [SlideWindow(target = powers) for powers in reader.split_power]
time_s_windows = [SlideWindow(target = times) for times in reader.split_times]
#SVD
cal = MultiSvdCalculator(power_s_windows)
print(cal[0].left_singular_vectors.shape)
print(cal[0].singular_vectors.shape)
print(cal[0].right_singular_vectors_transpose.shape)

#ピークとノイズを操作
sp = [PeakNoiseControl(c) for c in cal]

print(sp[0].integrated_difference(time_s_windows[0].partial_time_series_data))
