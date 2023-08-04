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

from typing import List, Optional

from nncf.common.tensor_statistics.statistics import MeanTensorStatistic
from nncf.common.tensor_statistics.statistics import RawTensorStatistic
from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import AbsQuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import BatchMeanReducer
from nncf.experimental.common.tensor_statistics.collectors import InplaceInsertionFNType
from nncf.experimental.common.tensor_statistics.collectors import MaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanPerChReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MinReducer
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
from nncf.experimental.common.tensor_statistics.collectors import NoopReducer
from nncf.experimental.common.tensor_statistics.collectors import QuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import ShapeAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.graph.node_utils import get_inplace_batch_mean_op
from nncf.openvino.graph.node_utils import get_inplace_max_op
from nncf.openvino.graph.node_utils import get_inplace_mean_op
from nncf.openvino.graph.node_utils import get_inplace_mean_per_ch
from nncf.openvino.graph.node_utils import get_inplace_min_op
from nncf.openvino.graph.node_utils import get_reducer_output_node_names
from nncf.openvino.graph.node_utils import get_result_node_name
from nncf.quantization.advanced_parameters import StatisticsType


class OVNoopReducer(NoopReducer):
    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return [get_result_node_name(target_node_name, port_id)]


class OVMinReducer(MinReducer):
    def get_inplace_fn(self):
        return get_inplace_min_op(self.name, reduction_axes=self._reduction_axes, channel_axis=self._channel_axis)

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.name, target_node_name, port_id, self.output_port_id, self.inplace)


class OVMaxReducer(MaxReducer):
    def get_inplace_fn(self):
        return get_inplace_max_op(
            self.name, reduction_axes=self._reduction_axes, channel_axis=self._channel_axis, use_abs_max=False
        )

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.name, target_node_name, port_id, self.output_port_id, self.inplace)


class OVAbsMaxReducer(AbsMaxReducer):
    def get_inplace_fn(self):
        return get_inplace_max_op(
            self.name, reduction_axes=self._reduction_axes, channel_axis=self._channel_axis, use_abs_max=True
        )

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.name, target_node_name, port_id, self.output_port_id, self.inplace)


class OVMeanReducer(MeanReducer):
    def get_inplace_fn(self):
        return get_inplace_mean_op(self.name, reduction_axes=self._reduction_axes, channel_axis=self._channel_axis)

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.name, target_node_name, port_id, self.output_port_id, self.inplace)


class OVBatchMeanReducer(BatchMeanReducer):
    def get_inplace_fn(self):
        return get_inplace_batch_mean_op(self.name)

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.name, target_node_name, port_id, self.output_port_id, self.inplace)


class OVMeanPerChanelReducer(MeanPerChReducer):
    def get_inplace_fn(self):
        return get_inplace_mean_per_ch(self.name, self._channel_axis)

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.name, target_node_name, port_id, self.output_port_id, self.inplace)


class OVQuantileReducer(QuantileReducer):
    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        return None

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.name, target_node_name, port_id, self.output_port_id, self.inplace)


class OVAbsQuantileReducer(AbsQuantileReducer):
    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        return None

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.name, target_node_name, port_id, self.output_port_id, self.inplace)


def get_mean_statistic_collector(
    num_samples: int, channel_axis: int, window_size: Optional[int] = None, inplace: bool = True
) -> TensorCollector:
    """
    Mean statistic collector builder.

    :param num_samples: Maximum number of samples to collect.
    :param channel_axis: Channel axis to use during reduction phase.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :param inplace: Whether the mean reducer should be calculated inplace or out of place.
    :return: Mean statistic collector.
    """
    # TODO(dlyakhov): use inplace OVBatchMeanReducer and OVMeanPerChanelReducer
    # after migration on openvino-dev=2023.0
    inplace = False
    if channel_axis is None:
        reducer = OVBatchMeanReducer(inplace)
    else:
        reducer = OVMeanPerChanelReducer(channel_axis=channel_axis, inplace=inplace)
    noop_reducer = OVNoopReducer()

    kwargs = {
        "use_per_sample_stats": False,
        "num_samples": num_samples,
        "window_size": window_size,
    }
    aggregate_mean = MeanAggregator(**kwargs)
    aggregate_shape = ShapeAggregator()

    collector = TensorCollector(MeanTensorStatistic)
    collector.register_statistic_branch(MeanTensorStatistic.MEAN_STAT, reducer, aggregate_mean)
    collector.register_statistic_branch(MeanTensorStatistic.SHAPE_STAT, noop_reducer, aggregate_shape)
    return collector


def get_raw_stat_collector(num_samples, inplace=False):
    reducer = OVNoopReducer()
    aggregator = NoopAggregator(num_samples)

    collector = TensorCollector(RawTensorStatistic)
    collector.register_statistic_branch(RawTensorStatistic.VALUES_STATS, reducer, aggregator)
    return collector


OV_REDUCERS_MAP = {
    StatisticsType.MIN: OVMinReducer,
    StatisticsType.MAX: OVMaxReducer,
    StatisticsType.ABS_MAX: OVAbsMaxReducer,
    StatisticsType.MEAN: OVMeanReducer,
    StatisticsType.QUANTILE: OVQuantileReducer,
    StatisticsType.ABS_QUANTILE: OVAbsQuantileReducer,
}
