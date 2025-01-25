from abc import ABC, abstractmethod

import numpy as np


class CommonReader(ABC):
    COLUMNS = ["titration", "time", "power", "degree"]
    
    @property
    @abstractmethod
    def split_times(self):
        """滴定回数ごとの時間を返却します。

        Returns:
            list[np.ndarray]: 滴定回数ごとの時間
        """

    @property
    @abstractmethod
    def split_powers(self):
        """滴定回数ごとのパワーを返却します。

        Returns:
            list[np.ndarray]: 滴定回数ごとのパワー
        """

    @property
    @abstractmethod
    def split_degrees(self):
        """滴定回数ごとの基準温度を返却します。

        Returns:
            list[np.ndarray]: 滴定回数ごとの温度
        """

    @property
    @abstractmethod
    def split_count(self):
        """滴定回数を返却します。

        Returns:
            int: 滴定回数
        """
    
    def _get_split_column(self, key: str):
        return [
            self.data_df[self.data_df["titration"] == split_count][
                key
            ].to_numpy()
            for split_count in range(self.split_count)
        ]
