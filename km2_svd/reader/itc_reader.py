from pathlib import Path
from matplotlib.axes import Axes
import pandas as pd
from km2_svd.reader.common_reader import CommonReader


class ItcReader(CommonReader):
    def __init__(self, path: Path):
        self._path = path
        with open(path, mode="rt", encoding="utf-8") as file:
            read_data = file.readlines()
            self._data_header = [
                s.replace(" ", "").replace("\n", "") for s in read_data[:31]
            ]
            data_body = [
                s.replace(" ", "").replace("\n", "") for s in read_data[31:]
            ]

        split_count = -1
        rows = []
        for data in data_body:
            if "@" in data:
                split_count += 1
                continue
            row = list(map(float, data.split(",")))
            row.insert(0, split_count)
            rows.append(row)
        self.data_df = pd.DataFrame(rows, columns=self.COLUMNS)

    @property
    def split_times(self):
        return self._get_split_column("time")

    @property
    def split_powers(self):
        return [split_power-split_power[0] for split_power in self._get_split_column("power")]

    @property
    def split_degrees(self):
        return self._get_split_column("degree")

    @property
    def split_count(self) -> int:
        return len(self.data_df["titration"].unique())
    
    def get_titration_df(self, titration_count: int) -> pd.DataFrame:
        return self.data_df[self.data_df["titration"] == titration_count]

    def get_itc_plotter(self, ax: Axes=None):
        return ITCPlotter(self.data_df, ax)
