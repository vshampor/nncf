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

import numpy as np
import pytest

from nncf.common.tensor_statistics.statistics import MeanTensorStatistic
from nncf.common.tensor_statistics.statistics import MedianMADTensorStatistic
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.tensor_statistics.statistics import PercentileTensorStatistic
from nncf.common.tensor_statistics.statistics import RawTensorStatistic
from nncf.experimental.common.tensor_statistics.collectors import MergedTensorCollector
from nncf.experimental.common.tensor_statistics.collectors import OfflineAggregatorBase
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.experimental.tensor import Tensor


class DummyTensorReducer(TensorReducerBase):
    def __init__(self, output_name: str, inplace: bool = False, inplace_mock=None):
        super().__init__(reduction_axes=tuple(), inplace=inplace)
        self._output_name = output_name
        self._inplace_mock = inplace_mock

    def _reduce_out_of_place(self, x: List[Tensor]):
        return x

    def get_inplace_fn(self):
        return self._inplace_mock

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return [self._output_name]

    def _get_processor(self):
        return None


class DummyTensorReducerA(DummyTensorReducer):
    pass


class DummyTensorAggregator(OfflineAggregatorBase):
    def _aggregate_stacked_samples(self, stacked_samples: Tensor) -> Tensor:
        return stacked_samples

    def __init__(self, num_samples: Optional[int] = None):
        super().__init__(None, num_samples=num_samples)

    def _register_reduced_input_impl(self, x: Tensor):
        return self._samples.append(x)

    def _aggregate_impl(self) -> Tensor:
        return self._samples[0]


class DummyTensorAggregatorA(DummyTensorAggregator):
    pass


def test_aggregator_enabled_and_reset():
    collector = TensorCollector()
    reducer = DummyTensorReducer("Dummy")
    aggregator = DummyTensorAggregator(5)
    collector.register_statistic_branch("A", reducer, aggregator)
    input_name = "input_name"
    inputs = TensorCollector.get_tensor_collector_inputs(
        {input_name: Tensor(np.array(100))}, [(hash(reducer), [input_name])]
    )

    for _ in range(3):
        collector.register_inputs(inputs)
    assert len(aggregator._samples) == 3
    assert aggregator._collected_samples == 3

    collector.disable()

    for _ in range(3):
        collector.register_inputs(inputs)
    assert len(aggregator._samples) == 3
    assert aggregator._collected_samples == 3

    collector.enable()

    for _ in range(3):
        collector.register_inputs(inputs)
    assert len(aggregator._samples) == 5
    assert aggregator._collected_samples == 5

    collector.reset()
    assert len(aggregator._samples) == 0
    assert aggregator._collected_samples == 0


def test_duplicated_statistics_are_merged():
    collector = TensorCollector()
    reducer = DummyTensorReducer("Dummy")
    reducer_a = DummyTensorReducerA("A")
    keys = "ABC"
    aggregators = []
    for key in keys:
        aggregator = DummyTensorAggregator(5)
        collector.register_statistic_branch(key, reducer, aggregator)
        aggregators.append(aggregator)
    aggregator_a = DummyTensorAggregatorA(1)
    aggregator_b = DummyTensorAggregator(100)
    collector.register_statistic_branch("D", reducer, aggregator_a)
    collector.register_statistic_branch("E", reducer_a, aggregator_b)
    reducer_inplace = DummyTensorReducer("Dummy_inplace", True)
    aggregator_for_inplace = DummyTensorAggregator(4)
    collector.register_statistic_branch("F", reducer_inplace, aggregator_for_inplace)

    # Check reducers and aggregators are merged
    assert len(collector._reducers) == 3
    assert len(collector._aggregators) == 4
    assert collector.num_samples == 100

    output_info = collector.get_output_info(None, None)
    # Check output info
    assert sorted(output_info) == sorted(
        [(hash(reducer_inplace), ["Dummy_inplace"]), (hash(reducer_a), ["A"]), (hash(reducer), ["Dummy"])]
    )

    outputs = {
        "Dummy": Tensor(np.array(5)),
        "A": Tensor(np.array(0)),
        "Dummy_inplace": Tensor(np.array(6)),
    }
    target_inputs = TensorCollector.get_tensor_collector_inputs(outputs, output_info)
    collector.register_inputs(target_inputs)

    # Check aggregators recieved inputs as expected
    assert aggregators[0]._collected_samples == 1
    for aggregator in aggregators[1:]:
        assert aggregator._collected_samples == 0
    assert aggregator_a._collected_samples == 1
    assert aggregator_b._collected_samples == 1
    assert aggregator_for_inplace._collected_samples == 1

    statistics = collector.get_statistics()

    # Check aggregators recieved correct inputs
    assert len(statistics) == 6
    for k in "ABC":
        assert statistics[k] == Tensor(np.array(5))
    assert statistics["D"] == Tensor(np.array(5))
    assert statistics["E"] == Tensor(np.array(0))
    assert statistics["F"] == Tensor(np.array(6))


def test_inplace_param():
    inplace_op = lambda: 0
    collector = TensorCollector()
    reducer_out_of_place = DummyTensorReducer("Dummy")
    reducer_inplace = DummyTensorReducer("Dummy", True, inplace_op)
    reducer_other = DummyTensorReducerA("Dummy")
    aggregator_inplace = DummyTensorAggregator(5)
    aggregator_out_of_place = DummyTensorAggregator(5)
    aggregator_other = DummyTensorAggregator(5)

    collector.register_statistic_branch("out_of_place", reducer_out_of_place, aggregator_out_of_place)
    collector.register_statistic_branch("inplace", reducer_inplace, aggregator_inplace)
    collector.register_statistic_branch("other", reducer_other, aggregator_other)
    assert len(collector._reducers) == 3
    assert len(collector._aggregators) == 3
    assert collector.get_inplace_fn_info()[0][0] == inplace_op
    assert collector.any_stat_out_of_place()


def test_merged_tensor_collector():
    num_collectors = 4
    collectors = [TensorCollector() for _ in range(num_collectors)]
    for idx, collector in enumerate(collectors):
        reducer_common = DummyTensorReducer("common_input")
        aggregator_common = DummyTensorAggregator(5)
        reducer_unique = type(DummyTensorReducer.__name__ + str(idx), (DummyTensorReducer,), {})(f"input_{idx + 1}")
        aggregator_unique = type(DummyTensorAggregator.__name__ + str(idx), (DummyTensorAggregator,), {})(5)
        collector.register_statistic_branch("common", reducer_common, aggregator_common)
        collector.register_statistic_branch("unique", reducer_unique, aggregator_unique)

    collectors[-1].disable()
    merged_collector = MergedTensorCollector(collectors)

    # Check reducers and aggregators are merged correctly
    assert len(merged_collector._reducers) == num_collectors
    assert len(merged_collector._aggregators) == num_collectors

    # Check aggregators were replaced correctly
    common_branch_key = (hash(reducer_common), 0, hash(aggregator_common))
    common_aggregator = merged_collector._aggregators[common_branch_key]
    for collector in collectors[:-1]:
        assert collector.aggregators[common_branch_key] is common_aggregator

    output_info = merged_collector.get_output_info(None, None)
    outputs = {"common_input": Tensor(np.array(0))}
    outputs.update({f"input_{idx + 1}": Tensor(np.array(idx + 1)) for idx, _ in enumerate(collectors[:-1])})
    target_inputs = TensorCollector.get_tensor_collector_inputs(outputs, output_info)
    merged_collector.register_inputs(target_inputs)

    # Check statistics are collected in a correct way
    for idx, collector in enumerate(collectors[:-1]):
        for aggregator in collector._aggregators.values():
            assert aggregator._collected_samples == 1

        statistic = collector.get_statistics()
        assert len(statistic) == 2
        assert statistic["common"] == Tensor(np.array(0))
        assert statistic["unique"] == Tensor(np.array(idx + 1))


def test_ambigous_container_key():
    collector = TensorCollector()
    reducer = DummyTensorReducer("Dummy")
    aggregator = DummyTensorAggregator(5)
    collector.register_statistic_branch("A", reducer, aggregator)
    with pytest.raises(RuntimeError):
        collector.register_statistic_branch("A", reducer, aggregator)


def test_ambiguous_branches():
    collector = TensorCollector()
    reducer = DummyTensorReducer("Dummy")
    aggregator = DummyTensorAggregator(5)
    collector.register_statistic_branch("A", reducer, aggregator)
    with pytest.raises(RuntimeError):
        collector.register_statistic_branch("B", reducer, aggregator)


class DummyMultipleInpOutTensorReducer(DummyTensorReducer):
    NUM_INPUTS = 3
    NUM_OUTPUTS = 2

    def _reduce_out_of_place(self, x: List[Tensor]):
        return x[: self.NUM_OUTPUTS]

    def get_output_names(self, target_node_name: str, port_id: int) -> str:
        return [f"{target_node_name}_{port_id}_{self._output_name}_{i}" for i in range(self.NUM_INPUTS)]


def test_multiple_branch_reducer():
    reducer_output_name = "reducer_output_name"
    target_node_name = "target_node_name"
    collector = TensorCollector()
    reducer = DummyMultipleInpOutTensorReducer(reducer_output_name)

    for i in range(reducer.NUM_OUTPUTS):
        aggregator = DummyTensorAggregator(None)
        collector.register_statistic_branch(str(i), reducer, aggregator, i)

    ref_output_info = [
        (
            hash(reducer),
            [
                "target_node_name_0_reducer_output_name_0",
                "target_node_name_0_reducer_output_name_1",
                "target_node_name_0_reducer_output_name_2",
            ],
        )
    ]
    inputs = {name: Tensor(np.array(i)) for i, name in enumerate(ref_output_info[0][1])}

    output_info = collector.get_output_info(target_node_name, 0)
    assert output_info == ref_output_info

    target_inputs = collector.get_tensor_collector_inputs(inputs, output_info)
    collector.register_inputs(target_inputs)

    ref_stats = {"0": Tensor(np.array(0)), "1": Tensor(np.array(1))}
    stats = collector.get_statistics()
    assert len(ref_stats) == len(stats)
    for key, value in ref_stats.items():
        assert value == stats[key]


def test_register_unnamed_statistics(mocker):
    tensor_collector = TensorCollector()
    reducer_hashes = []
    for reducer_cls, key in zip([DummyTensorReducer, DummyTensorReducerA], "AB"):
        reducer = reducer_cls(f"Dummy{key}")
        tensor_collector.register_statistic_branch(key, reducer, DummyTensorAggregator(None))
        reducer_hashes.append(hash(reducer))

    tensor_collector.register_inputs = mocker.MagicMock()
    inputs_ = Tensor(np.ones(5))
    tensor_collector.register_input_for_all_reducers(inputs_)

    tensor_collector.register_inputs.assert_called_once()
    args = tensor_collector.register_inputs.call_args[0][0]
    assert len(args) == 2
    for k, v in args.items():
        assert k in reducer_hashes
        assert len(v) == 1
        assert all(v[0] == inputs_)


def test_wrong_statistic_container_class():
    class BadStatContainer:
        pass

    tensor_collector = TensorCollector(BadStatContainer)
    tensor_collector.register_statistic_branch("A", DummyTensorReducer("A"), DummyTensorAggregator())
    tensor_collector.register_input_for_all_reducers(Tensor(np.ndarray([1])))
    with pytest.raises(RuntimeError):
        tensor_collector.get_statistics()


class TemplateTestStatisticCollector:
    @pytest.mark.parametrize("inplace", [False, True])
    @pytest.mark.parametrize("any_not_empty", [False, True])
    def test_empty_tensors_register(self, inplace, any_not_empty):
        collector = TensorCollector()
        reducer = DummyTensorReducer("Dummy", inplace=inplace)
        aggregator = DummyTensorAggregator(5)
        collector.register_statistic_branch("A", reducer, aggregator)
        input_name = "input_name"
        full_inputs = TensorCollector.get_tensor_collector_inputs(
            {input_name: Tensor(np.array([100]))}, [(hash(reducer), [input_name])]
        )
        empty_inputs = TensorCollector.get_tensor_collector_inputs(
            {input_name: Tensor(np.array([]))}, [(hash(reducer), [input_name])]
        )

        stats = collector.get_statistics()
        assert len(stats) == 1
        assert stats["A"] is None

        inputs = [full_inputs, empty_inputs, full_inputs] if any_not_empty else [empty_inputs, empty_inputs]
        for input_ in inputs:
            collector.register_inputs(input_)

        if any_not_empty:
            assert len(aggregator._samples) == 2
            assert aggregator._collected_samples == 2
            stats = collector.get_statistics()
            assert len(stats) == 1
            assert stats["A"] == Tensor([100])
            return

        assert len(aggregator._samples) == 0
        assert aggregator._collected_samples == 0
        stats = collector.get_statistics()
        assert len(stats) == 1
        assert stats["A"] is None

    def test_min_max_stat_building(self):
        tensor_collector = TensorCollector(MinMaxTensorStatistic)
        tensor_collector.register_statistic_branch(
            MinMaxTensorStatistic.MIN_STAT, DummyTensorReducer("A"), DummyTensorAggregator()
        )
        tensor_collector.register_statistic_branch(
            MinMaxTensorStatistic.MAX_STAT, DummyTensorReducer("B"), DummyTensorAggregator()
        )
        tensor_collector.register_input_for_all_reducers(Tensor(np.array([1])))
        statistic = tensor_collector.get_statistics()
        assert isinstance(statistic, MinMaxTensorStatistic)
        assert statistic.min_values == statistic.max_values == Tensor(np.array([1]))

    def test_mean_max_stat_building(self):
        tensor_collector = TensorCollector(MeanTensorStatistic)
        tensor_collector.register_statistic_branch(
            MeanTensorStatistic.MEAN_STAT, DummyTensorReducer("A"), DummyTensorAggregator()
        )
        tensor_collector.register_statistic_branch(
            MeanTensorStatistic.SHAPE_STAT, DummyTensorReducer("B"), DummyTensorAggregator()
        )
        tensor_collector.register_input_for_all_reducers(Tensor(np.array([1])))
        statistic = tensor_collector.get_statistics()
        assert isinstance(statistic, MeanTensorStatistic)
        assert statistic.mean_values == Tensor(np.array([1]))
        assert statistic.observed_shape == Tensor(np.array([1]))

    def test_median_mad_stat_building(self):
        class DummyMADPercentileAggregator(DummyTensorAggregator):
            def _aggregate_impl(self):
                return {
                    MedianMADTensorStatistic.MEDIAN_VALUES_STAT: self._samples[0],
                    MedianMADTensorStatistic.MAD_VALUES_STAT: self._samples[0],
                }

        tensor_collector = TensorCollector(MedianMADTensorStatistic)
        tensor_collector.register_statistic_branch(
            MedianMADTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY,
            DummyTensorReducer("A"),
            DummyMADPercentileAggregator(),
        )
        tensor_collector.register_input_for_all_reducers(Tensor(np.array([1])))
        statistic = tensor_collector.get_statistics()
        assert isinstance(statistic, MedianMADTensorStatistic)
        assert statistic.median_values == statistic.mad_values == Tensor(np.array([1]))

    def test_percentile_max_stat_building(
        self,
    ):
        class DummyPercentileTensorAggregator(DummyTensorAggregator):
            def _aggregate_impl(self):
                return {0.5: self._samples[0]}

        tensor_collector = TensorCollector(PercentileTensorStatistic)
        tensor_collector.register_statistic_branch(
            PercentileTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY,
            DummyTensorReducer("A"),
            DummyPercentileTensorAggregator(),
        )
        tensor_collector.register_input_for_all_reducers(Tensor(np.array([1])))
        statistic = tensor_collector.get_statistics()
        assert isinstance(statistic, PercentileTensorStatistic)
        assert statistic.percentile_vs_values_dict[0.5] == Tensor(np.array([1]))

        tensor_collector = TensorCollector(PercentileTensorStatistic)
        qs = [0.3, 0.5, 0.7]
        for q in qs:
            tensor_collector.register_statistic_branch(
                (PercentileTensorStatistic.PERCENTILE_VS_VALUE_DICT, q),
                DummyTensorReducer(f"A{q}"),
                DummyTensorAggregator(),
            )
        tensor_collector.register_input_for_all_reducers(Tensor(np.array([1])))
        statistic = tensor_collector.get_statistics()
        assert isinstance(statistic, PercentileTensorStatistic)
        assert len(statistic.percentile_vs_values_dict) == len(qs)
        for q in qs:
            assert statistic.percentile_vs_values_dict[q] == Tensor(np.array([1]))

    def test_raw_max_stat_building(self):
        tensor_collector = TensorCollector(RawTensorStatistic)
        tensor_collector.register_statistic_branch(
            RawTensorStatistic.VALUES_STATS, DummyTensorReducer("A"), DummyTensorAggregator()
        )
        tensor_collector.register_input_for_all_reducers(Tensor(np.array([1])))
        statistic = tensor_collector.get_statistics()
        assert isinstance(statistic, RawTensorStatistic)
        assert statistic.values == Tensor(np.array([1]))
