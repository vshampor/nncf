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

from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.tensor import Tensor
from nncf.onnx.graph.node_utils import get_bias_value
from nncf.onnx.graph.node_utils import is_any_weight_quantized
from nncf.onnx.graph.node_utils import is_node_with_bias
from nncf.onnx.graph.transformations.command_creation import create_bias_correction_command
from nncf.onnx.graph.transformations.commands import ONNXBiasCorrectionCommand
from nncf.onnx.graph.transformations.commands import ONNXModelExtractionCommand
from nncf.onnx.graph.transformations.commands import ONNXNullBiasInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.statistics.collectors import ONNXMeanStatisticCollector
from nncf.quantization.algorithms.fast_bias_correction.backend import FastBiasCorrectionAlgoBackend


class ONNXFastBiasCorrectionAlgoBackend(FastBiasCorrectionAlgoBackend):
    @property
    def types_to_insert_bias(self):
        return []

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_bias_insertion_command(node: NNCFNode) -> ONNXNullBiasInsertionCommand:
        return ONNXNullBiasInsertionCommand(node)

    @staticmethod
    def create_bias_correction_command(
        node: NNCFNode, bias_value: Tensor, nncf_graph: NNCFGraph
    ) -> ONNXBiasCorrectionCommand:
        return create_bias_correction_command(node, bias_value.data)

    @staticmethod
    def model_extraction_command(inputs: List[str], outputs: List[str]) -> ONNXModelExtractionCommand:
        return ONNXModelExtractionCommand(inputs, outputs)

    @staticmethod
    def mean_statistic_collector(
        channel_axis: int,
        inplace: bool,
        num_samples: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> ONNXMeanStatisticCollector:
        return ONNXMeanStatisticCollector(channel_axis, num_samples, window_size)

    @staticmethod
    def get_sub_input_output_names(subgraph: onnx.ModelProto) -> Tuple[str, str]:
        return subgraph.graph.input[0].name, subgraph.graph.output[0].name

    @staticmethod
    def create_input_data(
        shape: Tuple[int], data: List[Tensor], input_name: str, channel_axis: int
    ) -> Dict[str, np.array]:
        blob = np.zeros(shape, dtype=data[0].data.dtype)
        for j, idx in enumerate(np.ndindex(blob.shape[channel_axis])):
            index = tuple(slice(None) if i != channel_axis else idx for i in range(blob.ndim))
            blob[index] = data[j].data
        input_data = {input_name: blob}
        return input_data

    @staticmethod
    def get_bias_value(node: NNCFNode, nncf_graph: NNCFGraph, model: onnx.ModelProto) -> Tensor:
        return Tensor(get_bias_value(node, model))

    @staticmethod
    def get_activation_port_ids_for_bias_node(node: NNCFNode) -> Tuple[int, int]:
        return 0, 0

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> Tensor:
        return Tensor(raw_data[output_name])

    @staticmethod
    def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_any_weight_quantized(node, nncf_graph)

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_node_with_bias(node)

    @staticmethod
    def get_node_names_for_input_output_statistics(node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[str, str]:
        return node.node_name, node.node_name
