from matplotlib.axes import Axes


def raw_axis_setting(ax: Axes):
    """RawDataグラフの軸設定を行います。"""
    ax.set_xlabel("Time[sec]", size="large")
    ax.set_ylabel("μcal/sec", size="large")
    ax.minorticks_on()
    ax.grid(which="major", color="black", alpha=0.5)
    ax.grid(which="minor", color="gray", linestyle=":")


def titration_axis_setting(ax: Axes):
    """titrationグラフの軸設定を行います。"""
    ax.set_xlabel("Time[sec]", size="large")
    ax.set_ylabel("μcal/sec", size="large")
    ax.minorticks_on()
    ax.grid(which="major", color="black", alpha=0.5)
    ax.grid(which="minor", color="gray", linestyle=":")


def power_axis_setting(ax: Axes):
    """powerグラフの軸設定を行います。"""
    ax.set_xlabel("molar ratio", size="large")
    ax.set_ylabel("kJ/mol", size="large")
    ax.minorticks_on()
    ax.grid(which="major", color="black", alpha=0.5)
    ax.grid(which="minor", color="gray", linestyle=":")


def singular_value_axis_setting(ax: Axes):
    """singular_valueグラフの軸設定を行います。"""
    ax.set_xlabel("k", size="large")
    ax.set_ylabel("sk", size="large")
    ax.minorticks_on()


def peak_noise_axis_setting(ax: Axes):
    """peakグラフの軸設定を行います。"""
    ax.set_xlabel("Time[sec]", size="large")
    # ax.set_ylabel("μcal/sec", size="large")
    ax.minorticks_on()
    ax.grid(which="major", color="black", alpha=0.5)
    ax.grid(which="minor", color="gray", linestyle=":")
