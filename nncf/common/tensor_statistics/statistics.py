# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from abc import abstractmethod
from collections import Counter
from typing import Dict, List, TypeVar

from nncf.common.tensor import NNCFTensor

TensorType = TypeVar("TensorType")


class TensorStatistic(ABC):
    """Base class that stores statistic data"""

    @staticmethod
    def tensor_eq(tensor1: NNCFTensor, tensor2: NNCFTensor, rtol=1e-6) -> bool:
        backend = tensor1.backend
        return backend.isclose_all(tensor1, tensor2, rtol=rtol)

    @abstractmethod
    def __eq__(self, other):
        pass


class MinMaxTensorStatistic(TensorStatistic):
    MIN_STAT = "min_values"
    MAX_STAT = "max_values"

    def __init__(self, min_values: NNCFTensor, max_values: NNCFTensor):
        self.min_values = min_values
        self.max_values = max_values

    def __eq__(self, other: "MinMaxTensorStatistic") -> bool:
        return self.tensor_eq(self.min_values, other.min_values) and self.tensor_eq(self.max_values, other.max_values)

    def __repr__(self):
        return f"min: {repr(self.min_values.tensor)}, max: {repr(self.max_values.tensor)}"

    @staticmethod
    def from_stat(statistic: TensorStatistic) -> "MinMaxTensorStatistic":
        if isinstance(statistic, MinMaxTensorStatistic):
            return statistic
        if isinstance(statistic, MedianMADTensorStatistic):
            # Using three-sigma approach to estimate min and max
            # Constant factor depends on the distribution form - assuming normal and the factor is 1.4826
            return MinMaxTensorStatistic(
                statistic.median_values - 3 * 1.4826230 * statistic.mad_values,
                statistic.median_values + 3 * 1.4826230 * statistic.mad_values,
            )
        if isinstance(statistic, PercentileTensorStatistic):
            if len(statistic.percentile_vs_values_dict.keys()) < 2:
                raise ValueError("Cannot create a min-max statistic for less than 2 percentile values")
            min_pct = min(statistic.percentile_vs_values_dict.keys())
            max_pct = max(statistic.percentile_vs_values_dict.keys())
            return MinMaxTensorStatistic(
                statistic.percentile_vs_values_dict[min_pct], statistic.percentile_vs_values_dict[max_pct]
            )
        raise ValueError("Unknown TensorStatistic to generate min-max stat from!")

class MeanTensorStatistic(TensorStatistic):
    MEAN_STAT = "mean_values"
    SHAPE_STAT = "shape"

    """
    Base class for the statistics that collects as mean per-axis
    """

    def __init__(self, mean_values: NNCFTensor, shape: List[int]):
        """
        :param mean_values: Collected mean per-axis values.
        :param shape: The shape of the collected statistics.
        """
        self.mean_values = mean_values
        self.observed_shape = shape

    def __eq__(self, other: "MeanTensorStatistic") -> bool:
        return self.mean_values == other.mean_values and self.observed_shape == other.observed_shape


class MedianMADTensorStatistic(TensorStatistic):
    def __init__(self, median_values: NNCFTensor, mad_values: NNCFTensor):
        self.median_values = median_values
        self.mad_values = mad_values

    def __eq__(self, other: "MedianMADTensorStatistic") -> bool:
        return self.tensor_eq(self.median_values, other.median_values) and self.tensor_eq(
            self.mad_values, other.mad_values
        )


class PercentileTensorStatistic(TensorStatistic):
    def __init__(self, percentile_vs_values_dict: Dict[float, NNCFTensor]):
        self.percentile_vs_values_dict = percentile_vs_values_dict

    def __eq__(self, other: "PercentileTensorStatistic", rtol=1e-9) -> bool:
        if Counter(self.percentile_vs_values_dict.keys()) != Counter(other.percentile_vs_values_dict.keys()):
            return False
        for pct in self.percentile_vs_values_dict.keys():
            if not self.tensor_eq(self.percentile_vs_values_dict[pct], other.percentile_vs_values_dict[pct]):
                return False
        return True


class RawTensorStatistic(TensorStatistic):
    VALUES_STATS = "values"

    """
    Base class for the raw statistics, without any aggregation.
    """

    def __init__(self, values: List[NNCFTensor]):
        """
        :param values: Collected raw values.
        """
        self.values = values

    def __eq__(self, other: "RawTensorStatistic") -> bool:
        return self.tensor_eq(self.values, other.values)
