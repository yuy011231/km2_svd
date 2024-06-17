from km2_svd.reader.common_reader import BaseReader
from km2_svd.reader.itc_reader import ItcReader
from km2_svd.slide_window import SlideWindow
from km2_svd.svd_calculator import MultiSvdCalculator
from km2_svd.peak_noise_control import PeakNoiseControl

#読み込み、整形
reader = ItcReader("data/210315C.ITC")
# reader = ItcReader("data/210107C.ITC")
print(len(reader.split_power))
print(len(reader.split_times))

#スライド窓
power_s_windows = [SlideWindow(target = powers) for powers in reader.split_power]
#SVD
cal = MultiSvdCalculator(power_s_windows)
print(cal[0].left_singular_vectors.shape)
print(cal[0].singular_vectors.shape)
print(cal[0].right_singular_vectors_transpose.shape)

#ピークとノイズを操作
sp = [PeakNoiseControl(c) for c in cal]

print(sp[0].integrated_difference(reader.split_times))
