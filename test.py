from km2_svd.reader.common_reader import BaseReader
from km2_svd.reader.itc_reader import SingleItcReader
from km2_svd.slide_window import SlideWindow
from km2_svd.svd_calculator import SvdCalculator
from km2_svd.peak_noise_control import PeakNoiseControl

reader = SingleItcReader("data/210315C.ITC")
print(len(reader.times))
print(len(reader.power))
print(len(reader.degree))

s_window = SlideWindow(reader)
print(s_window.partial_time_count)
print(s_window.partial_time_series_data)

cal = SvdCalculator(s_window)
print(cal.right_singular_vectors_transpose)
print(cal.left_singular_vectors)
print(cal.singular_vectors)

sp = PeakNoiseControl(cal)
print(sp.peak_component)
print(sp.noise_component)
