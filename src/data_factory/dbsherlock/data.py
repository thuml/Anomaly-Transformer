import functools
from typing import *

import hkkang_utils.data as data_utils
import numpy as np


@data_utils.dataclass
class AnomalyData:
    cause: str  # the name of each performance anomaly
    attributes: List[str]  # list of attribute names
    values: List[List[float]]  # shape: (time, attribute)
    normal_regions: List[int]  # list of normal region indices
    abnormal_regions: List[int]  # list of abnormal region indices

    @functools.cached_property
    def values_as_np(self) -> np.ndarray:
        return np.array(self.values)

    @functools.cached_property
    def valid_normal_regions(self) -> List[int]:
        """Get all region size"""
        if self.normal_regions:
            return self.normal_regions
        return [
            i
            for i in range(len(self.values))
            if i not in self.abnormal_regions and self.values[i][1] > 0
        ]

    @functools.cached_property
    def valid_abnormal_regions(self) -> List[int]:
        """Get all region size"""
        return self.abnormal_regions

    @functools.cached_property
    def valid_attributes(self) -> List[str]:
        return [self.attributes[i] for i in range(2, len(self.attributes))]

    @functools.cached_property
    def valid_values(self) -> np.ndarray:
        """Get all values"""
        tmp = []
        for values_in_time in self.values:
            tmp.append([values_in_time[i] for i in range(2, len(self.attributes))])
        return tmp

    @functools.cached_property
    def valid_values_as_np(self) -> np.ndarray:
        """Get all values"""
        return np.array(self.valid_values)

    @functools.cached_property
    def valid_normal_values(self) -> List[List[float]]:
        return [self.values[i] for i in self.valid_normal_regions]

    @functools.cached_property
    def valid_abnormal_values(self) -> List[List[float]]:
        return [self.values[i] for i in self.valid_abnormal_regions]

    @functools.cached_property
    def training_data(self) -> np.ndarray:
        """Get training data"""
        valid_regions = self.valid_normal_regions + self.abnormal_regions
        training_indices = [i for i in range(len(self.values)) if i in valid_regions]
        return self.values_as_np[training_indices:]


@data_utils.dataclass
class AnomalyDataset:
    causes: List[str] = data_utils.field(default_factory=list)
    data: List[AnomalyData] = data_utils.field(default_factory=list)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> AnomalyData:
        return self.data[idx]

    def get_data_of_cause(self, cause: str) -> List[AnomalyData]:
        return [data for data in self.data if data.cause == cause]
