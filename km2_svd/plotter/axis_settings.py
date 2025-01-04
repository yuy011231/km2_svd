from matplotlib.axes import Axes

def itc_axis_setting(ax: Axes):
    """ITCグラフの軸設定を行います。"""
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
