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

from typing import Any, Callable, Deque, List, Optional, Union

import numpy as np
import tensorflow as tf

from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import TensorElementsType
from nncf.common.tensor_statistics.collectors import MeanMinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import MeanPercentileStatisticCollector
from nncf.common.tensor_statistics.collectors import MedianMADStatisticCollector
from nncf.common.tensor_statistics.collectors import MinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import MixedMinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import PercentileStatisticCollector
from nncf.common.tensor_statistics.reduction import np_percentile_reduce_like
from nncf.tensorflow.tensor import TFNNCFTensor
from nncf.tensorflow.tensor_statistics.reduction import convert_rs_to_pt_type
from nncf.tensorflow.tensor_statistics.statistics import TFMedianMADTensorStatistic
from nncf.tensorflow.tensor_statistics.statistics import TFMinMaxTensorStatistic
from nncf.tensorflow.tensor_statistics.statistics import TFPercentileTensorStatistic


class TFMinMaxStatisticCollector(MinMaxStatisticCollector):
    def _register_input(self, x: tf.Tensor):
        self._register_input_common(TFNNCFTensor(x))

    def _get_statistics(self) -> TFMinMaxTensorStatistic:
        return TFMinMaxTensorStatistic(self._min_values.tensor, self._max_values.tensor)


class TFMixedMinMaxStatisticCollector(MixedMinMaxStatisticCollector):
    def _register_input(self, x: tf.Tensor):
        self._register_input_common(TFNNCFTensor(x))

    def _get_statistics(self) -> TFMinMaxTensorStatistic:
        return TFMinMaxTensorStatistic(self._min_aggregate().tensor, self._max_aggregate().tensor)


class TFMeanMinMaxStatisticCollector(MeanMinMaxStatisticCollector):
    def _register_input(self, x: tf.Tensor):
        self._register_input_common(TFNNCFTensor(x))

    def _get_statistics(self) -> TFMinMaxTensorStatistic:
        return TFMinMaxTensorStatistic(self._min_aggregate().tensor, self._max_aggregate().tensor)


class TFMedianMADStatisticCollector(MedianMADStatisticCollector):
    def _register_input(self, x: tf.Tensor):
        self._samples.append(x.numpy())

    def _get_statistics(self) -> TFMedianMADTensorStatistic:
        self._reduction_shape = convert_rs_to_pt_type(self._samples[0].shape, self._reduction_shape)
        numpy_median, numpy_mad = self._prepare_statistics()
        median_tensor = tf.convert_to_tensor(np.array(numpy_median), dtype=tf.float32)
        mad_tensor = tf.convert_to_tensor(np.array(numpy_mad), dtype=tf.float32)
        return TFMedianMADTensorStatistic(median_tensor, mad_tensor)


class TFPercentileStatisticCollector(PercentileStatisticCollector):
    def _register_input(self, x: tf.Tensor):
        self._samples.append(x.numpy())

    def _get_statistics(self) -> TFPercentileTensorStatistic:
        self._reduction_shape = convert_rs_to_pt_type(self._samples[0].shape, self._reduction_shape)
        percentile_vs_values_dict = self._prepare_statistics()
        for key, val in percentile_vs_values_dict.items():
            percentile_vs_values_dict[key] = tf.convert_to_tensor(val, dtype=tf.float32)
        return TFPercentileTensorStatistic(percentile_vs_values_dict)


class TFMeanPercentileStatisticCollector(MeanPercentileStatisticCollector):
    def _register_input(self, x: tf.Tensor):
        x_np = x.numpy()
        for pct, values in self._all_pct_values.items():
            np_vals = np_percentile_reduce_like(x_np, convert_rs_to_pt_type(x_np.shape, self._reduction_shape), pct)
            tf_vals = tf.convert_to_tensor(np_vals, dtype=tf.float32)
            values.append(tf_vals)

    def _get_statistics(self) -> TFPercentileTensorStatistic:
        mean_percentile_values = {}
        for pct, values in self._all_pct_values.items():
            stacked_pct_vals = tf.stack(values)
            mean_percentile_values[pct] = tf.math.reduce_mean(stacked_pct_vals, axis=0)
        return TFPercentileTensorStatistic(mean_percentile_values)
