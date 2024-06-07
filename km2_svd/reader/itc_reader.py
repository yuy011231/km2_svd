from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from itertools import islice
from logging import getLogger
from pathlib import Path
from typing import (
    Generator,
    Generic,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    overload,
)

import numpy as np
from km2_svd.reader.common_reader import BaseReader


class SingleItcReader(BaseReader):
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
        split_data = [row.split(",") for row in self.remove_at_data_body]
        columns = np.array(list(zip(*split_data)))
        self.__column_data_body = np.array(
            [[float(item) for item in col] for col in columns]
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
        return len([data for data in self.__data_body if "@" in data]) - 1
