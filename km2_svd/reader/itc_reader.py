from typing import Iterator

import numpy as np
from km2_svd.reader.common_reader import CommonReader


class Titration:
    def __init__(self, data: Iterator[str]):
        split_data = [row.split(",") for row in data]
        columns = np.array(list(zip(*split_data)))
        self.__column_data_body = np.array(
            [[float(item) for item in col] for col in columns]
        )

    @property
    def times(self) -> np.ndarray[float]:
        """時間データを返却します。

        Returns:
            np.NDArray[float]: 時間データ
        """
        return self.__column_data_body[0]

    @property
    def power(self) -> np.ndarray[float]:
        """(ITCの場合)電力データを返却します。

        Returns:
            np.NDArray[float]: 電力データ
        """
        return self.__column_data_body[1]

    @property
    def degree(self) -> np.ndarray[float]:
        """degreeを返却します。

        Returns:
            np.NDArray[float]: degreeデータ
        """
        return self.__column_data_body[2]


class ItcReader(CommonReader):
    def __init__(self, path):
        self.__path = path
        with open(path, mode="rt", encoding="utf-8") as file:
            read_data = file.readlines()
            self.__data_header = [
                s.replace(" ", "").replace("\n", "") for s in read_data[:31]
            ]
            self.__data_body = [
                s.replace(" ", "").replace("\n", "") for s in read_data[31:]
            ]

        columns, column = [], []
        for data in self.__data_body:
            if "@" in data or self.__data_body.index(data) == len(self.__data_body) - 1:
                if "@0" not in data:
                    columns.append(Titration(column))
                    column = []
                continue
            column.append(data)
        self.__titrations: Iterator[Titration] = columns

        split_data = [row.split(",") for row in self.remove_at_data_body]
        all_columns = np.array(list(zip(*split_data)))
        self.__column_data_body = np.array(
            [[float(item) for item in col] for col in all_columns]
        )

    def __len__(self) -> int:
        return len(self.__data_body)

    @property
    def times(self) -> np.ndarray[float]:
        """時間データを返却します。

        Returns:
            np.NDArray[float]: 時間データ
        """
        return self.__column_data_body[0]

    @property
    def power(self) -> np.ndarray[float]:
        """(ITCの場合)電力データを返却します。

        Returns:
            np.NDArray[float]: 電力データ
        """
        return self.__column_data_body[1]

    @property
    def degree(self) -> np.ndarray[float]:
        """degreeを返却します。

        Returns:
            np.NDArray[float]: degreeデータ
        """
        return self.__column_data_body[2]

    @property
    def split_times(self):
        return [titration.times for titration in self.__titrations]

    @property
    def split_power(self):
        return [titration.power for titration in self.__titrations]

    @property
    def split_degree(self):
        return [titration.degree for titration in self.__titrations]

    @property
    def remove_at_data_body(self):
        """滴定回数の区切りデータ(@を含む行)を削除したデータを返却します。

        Returns:
            _type_: _description_
        """
        return list([data for data in self.__data_body if "@" not in data])

    @property
    def titration_count(self) -> int:
        """滴定回数を返却します。

        Returns:
            int: 滴定回数
        """
        return len(self.__titrations)
