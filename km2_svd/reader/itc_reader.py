from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from km2_svd.plotter.itc_plotter import ITCPlotter, TitrationPlotters
from km2_svd.reader.common_reader import CommonReader


class ItcReader(CommonReader):
    COLUMNS = ["titration", "time", "power", "degree"]

    def __init__(self, path: Path):
        self._path = path
        with open(path, mode="rt", encoding="utf-8") as file:
            read_data = file.readlines()
            self._data_header = [
                s.replace(" ", "").replace("\n", "") for s in read_data[:31]
            ]
            self._data_body = [
                s.replace(" ", "").replace("\n", "") for s in read_data[31:]
            ]

        split_count = -1
        rows = []
        for data in self._data_body:
            if "@" in data:
                split_count += 1
                continue
            row = list(map(float, data.split(",")))
            row.insert(0, split_count)
            rows.append(row)
        self.data_body = pd.DataFrame(rows, columns=self.COLUMNS)

    @property
    def split_times(self):
        return self._get_split_column("time")

    @property
    def split_powers(self):
        return [split_power-split_power[0] for split_power in self._get_split_column("power")]

    @property
    def split_degrees(self):
        return self._get_split_column("degree")

    def _get_split_column(self, key: str):
        return [
            self.data_body[self.data_body["titration"] == split_count][
                key
            ].to_numpy()
            for split_count in range(self.split_count)
        ]

    def _get_split_df(self):
        return [
            self.data_body[self.data_body["titration"] == split_count]
            for split_count in range(self.split_count)
        ]

    @property
    def split_count(self) -> int:
        """滴定回数を返却します。

        Returns:
            int: 滴定回数
        """
        return len(self.data_body["titration"].unique())

    def get_titration_plotter(self, ax=None):
        return TitrationPlotters(self._get_split_df(), ax)

    def get_itc_plotter(self, ax=None):
        return ITCPlotter(self.data_body, ax)
