#  Copyright (c) 2023 Intel Corporation
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import inspect
from pathlib import Path
from typing import Callable
from typing import List

import networkx as nx
import torch
from torch._dynamo import register_backend
from torch.fx import GraphModule
from torch.fx import Node
from torch.nn import Conv2d
from torch.nn import Linear

from nncf.common.utils.dot_file_rw import write_dot_graph


def get_node_id(node: torch.fx.Node, idx: int) -> str:
    return f"{idx} {node.op} {node.target if not inspect.isfunction(node.target) else node.target.__name__}"


def visualize_fx_graph(fx_graph: torch.fx.Graph, dot_path: Path):
    node_list: List[Node] = list(fx_graph.nodes)
    nx_graph = nx.DiGraph()
    node_vs_node_id = {}
    for idx, node in enumerate(node_list):
        node_id = get_node_id(node, idx)
        nx_graph.add_node(node_id)
        node_vs_node_id[node] = node_id

    for idx, node in enumerate(node_list):
        node_id = node_vs_node_id[node]
        user_node_ids = [node_vs_node_id[n] for n in node.users]
        for user_node_id in user_node_ids:
            nx_graph.add_edge(node_id, user_node_id)

    write_dot_graph(nx_graph, dot_path)


def quantize_weight_and_activation_for_basic_module(graph_module: torch.fx.GraphModule,
                                                    basic_module_call_node: torch.fx.Node,
                                                    count: int):
    graph = graph_module.graph
    target_basic_module_accessor = basic_module_call_node.target
    target_basic_module = getattr(graph_module, target_basic_module_accessor)
    module_input = next(iter(basic_module_call_node.all_input_nodes))

    fq_a_module_name = f'nncf_quantizers.fq_a_{count}'
    fq_w_module_name = f'nncf_quantizers.fq_w_{count}'
    graph_module.add_submodule(fq_a_module_name, torch.ao.quantization.FakeQuantize(quant_min=0, quant_max=255))
    graph_module.add_submodule(fq_w_module_name, torch.ao.quantization.FakeQuantize(quant_min=0, quant_max=255))

    with graph.inserting_before(basic_module_call_node):
        weight_node = graph.create_node('get_attr', target_basic_module_accessor + '.weight', name=f'conv2d_weight_{count}')

    with graph.inserting_before(basic_module_call_node):
        bias_node = graph.create_node('get_attr', target_basic_module_accessor + '.bias', name=f'conv2d_bias_{count}')

    with graph.inserting_after(bias_node):
        call_fq_w_node = graph.call_module(fq_w_module_name, args=(weight_node,))

    with graph.inserting_after(module_input):
        call_fq_a_node = graph.call_module(fq_a_module_name, args=(module_input,))
        for node in graph.nodes:
            if node is call_fq_a_node:
                continue
            node.replace_input_with(module_input, call_fq_a_node)

    with graph.inserting_after(call_fq_w_node):
        if isinstance(target_basic_module, Conv2d):
            call_basic_module_op_node = graph.call_function(torch.nn.functional.conv2d, kwargs={
                'input': call_fq_a_node,
                'weight': call_fq_w_node,
                'bias': bias_node,
                'stride': target_basic_module.stride,
                'padding': target_basic_module.padding,
                'dilation': target_basic_module.dilation,
                'groups': target_basic_module.groups
            })
        elif isinstance(target_basic_module, Linear):
            call_basic_module_op_node = graph.call_function(torch.nn.functional.linear, kwargs={
                'input': call_fq_a_node,
                'weight': call_fq_w_node,
                'bias': bias_node
            })
        else:
            raise NotImplementedError
    basic_module_call_node.replace_all_uses_with(call_basic_module_op_node)
    graph.erase_node(basic_module_call_node)
    return call_basic_module_op_node


@register_backend(name="nncf")
def embedding_backend(gm: GraphModule, inputs) -> Callable:
    visualize_fx_graph(gm.graph, Path('before.dot'))
    original_nodes: List[Node] = list(gm.graph.nodes)

    # Have to lower the original call_module[conv2d] node to the call_function ops,
    # because cannot setup a pre-op for FakeQuantizing the weight in the torch.fx.Graph domain
    for idx, node in enumerate(original_nodes):
        if node.op == 'call_module':
            target_module = getattr(gm, node.target)
            if isinstance(target_module, (Conv2d, Linear)):
                quantize_weight_and_activation_for_basic_module(gm, node, idx)

    print(gm.graph)
    visualize_fx_graph(gm.graph, Path('after.dot'))
    gm.recompile()
    return gm