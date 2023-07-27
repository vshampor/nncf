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
from collections import Counter
from collections import defaultdict
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

from nncf.common.graph import NNCFGraph
from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.utils.dot_file_rw import write_dot_graph
from nncf.torch.dynamic_graph.context import get_compile_output
from nncf.torch.dynamic_graph.context import get_compression_state
from nncf.torch.dynamic_graph.context import set_compile_output
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.quantization.algo import QuantizationController
from nncf.torch.quantization.layers import PTQuantizationPoint
from nncf.torch.quantization.layers import PTQuantizerSetup
from nncf.torch.quantization.strip import convert_to_torch_fakequantizer


def get_node_id(node: torch.fx.Node, idx: int) -> str:
    return f"{idx} {node.op} {node.name}" #{node.target if not inspect.isfunction(node.target) else node.target.__name__}"


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


def quantize_weight_for_basic_module(graph_module: torch.fx.GraphModule,
                                     basic_module_call_node: torch.fx.Node,
                                     uid: str,
                                     fq_module: torch.nn.Module = None):
    graph = graph_module.graph
    target_basic_module_accessor = basic_module_call_node.target
    target_basic_module = getattr(graph_module, target_basic_module_accessor)
    module_input = next(iter(basic_module_call_node.all_input_nodes))
    fq_w_module_name = f'nncf_quantizers_fq_w_{uid}'

    if fq_module is None:
        fq_module = torch.ao.quantization.FakeQuantize(quant_min=0, quant_max=255)

    graph_module.add_submodule(fq_w_module_name, fq_module)

    with graph.inserting_before(basic_module_call_node):
        weight_node = graph.create_node('get_attr', target_basic_module_accessor + '.weight',
                                        name=f'{target_basic_module.__class__.__name__}_weight_{uid}')

    if target_basic_module.bias is not None:
        with graph.inserting_before(basic_module_call_node):
            bias_node = graph.create_node('get_attr', target_basic_module_accessor + '.bias',
                                          name=f'{target_basic_module.__class__.__name__}_bias_{uid}')
    else:
        bias_node = None

    next_node = bias_node if bias_node is not None else weight_node
    with graph.inserting_after(next_node):
        call_fq_w_node = graph.call_module(fq_w_module_name, args=(weight_node,))

    with graph.inserting_after(call_fq_w_node):
        # Have to lower the original call_module[conv2d] node to the call_function ops,
        # because cannot setup a pre-op for FakeQuantizing the weight in the torch.fx.Graph domain
        if isinstance(target_basic_module, Conv2d):
            call_basic_module_op_node = graph.call_function(torch.nn.functional.conv2d, kwargs={
                'input': module_input,
                'weight': call_fq_w_node,
                'bias': bias_node,
                'stride': target_basic_module.stride,
                'padding': target_basic_module.padding,
                'dilation': target_basic_module.dilation,
                'groups': target_basic_module.groups
            })
        elif isinstance(target_basic_module, Linear):
            call_basic_module_op_node = graph.call_function(torch.nn.functional.linear, kwargs={
                'input': module_input,
                'weight': call_fq_w_node,
                'bias': bias_node
            })
        else:
            raise NotImplementedError
    basic_module_call_node.replace_all_uses_with(call_basic_module_op_node)
    graph.erase_node(basic_module_call_node)
    return call_basic_module_op_node


def quantize_activation_after(graph_module: torch.fx.GraphModule, node_to_quantize: torch.fx.Node, uid: str,
                              fq_module: torch.nn.Module = None) -> torch.fx.Node:
    graph = graph_module.graph
    fq_a_module_name = f'nncf_quantizers_fq_a_{uid}'
    if fq_module is None:
        fq_module = torch.ao.quantization.FakeQuantize(quant_min=0, quant_max=255)
    graph_module.add_submodule(fq_a_module_name, fq_module)
    with graph.inserting_after(node_to_quantize):
        call_fq_a_node = graph.call_module(fq_a_module_name, args=(node_to_quantize,))
        for node in graph.nodes:
            if node is call_fq_a_node:
                continue
            node.replace_input_with(node_to_quantize, call_fq_a_node)
    return call_fq_a_node


def quantize_activation_before(graph_module: torch.fx.GraphModule, node_to_quantize: torch.fx.Node,
                               input_port_id: int,
                               uid: str,
                               fq_module: torch.nn.Module = None) -> torch.fx.Node:
    graph = graph_module.graph
    fq_a_module_name = f'nncf_quantizers_fq_a_{uid}'
    if fq_module is None:
        fq_module = torch.ao.quantization.FakeQuantize(quant_min=0, quant_max=255)
    graph_module.add_submodule(fq_a_module_name, fq_module)
    with graph.inserting_before(node_to_quantize):
        port_input = node_to_quantize.all_input_nodes[input_port_id]
        call_fq_a_node = graph.call_module(fq_a_module_name, args=(port_input,))
        node_to_quantize.replace_input_with(port_input, call_fq_a_node)
    return call_fq_a_node


def quantize_weight_and_activation_for_basic_module(graph_module: torch.fx.GraphModule,
                                                    basic_module_call_node: torch.fx.Node,
                                                    count: int) -> torch.fx.Node:
    graph = graph_module.graph
    target_basic_module_accessor = basic_module_call_node.target
    target_basic_module = getattr(graph_module, target_basic_module_accessor)
    module_input = next(iter(basic_module_call_node.all_input_nodes))

    fq_a_module_name = f'nncf_quantizers.fq_a_{count}'
    fq_w_module_name = f'nncf_quantizers.fq_w_{count}'
    graph_module.add_submodule(fq_a_module_name, torch.ao.quantization.FakeQuantize(quant_min=0, quant_max=255))
    graph_module.add_submodule(fq_w_module_name, torch.ao.quantization.FakeQuantize(quant_min=0, quant_max=255))

    with graph.inserting_before(basic_module_call_node):
        weight_node = graph.create_node('get_attr', target_basic_module_accessor + '.weight',
                                        name=f'{target_basic_module.__class__.__name__}_weight_{count}')

    with graph.inserting_before(basic_module_call_node):
        bias_node = graph.create_node('get_attr', target_basic_module_accessor + '.bias',
                                      name=f'{target_basic_module.__class__.__name__}_bias_{count}')

    with graph.inserting_after(bias_node):
        call_fq_w_node = graph.call_module(fq_w_module_name, args=(weight_node,))

    with graph.inserting_after(module_input):
        call_fq_a_node = graph.call_module(fq_a_module_name, args=(module_input,))
        for node in graph.nodes:
            if node is call_fq_a_node:
                continue
            node.replace_input_with(module_input, call_fq_a_node)

    with graph.inserting_after(call_fq_w_node):
        # Have to lower the original call_module[conv2d] node to the call_function ops,
        # because cannot setup a pre-op for FakeQuantizing the weight in the torch.fx.Graph domain
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


def translate_compression(gm: GraphModule, state) -> GraphModule:
    visualize_fx_graph(gm.graph, Path('before.dot'))
    original_nodes: List[Node] = list(gm.graph.nodes)
    qsetup = PTQuantizerSetup.from_state(state['builder_state']['quantization']['quantizer_setup'])

    nncf_graph = state['graph']  # type: NNCFGraph
    qctrl = state['ctrl']  # type: QuantizationController
    nncf_node_name_vs_fx_target_name = {}  # type: Dict[OperationAddress, str]
    op_counts = Counter()
    for nncf_node in nncf_graph.get_all_nodes():
        target_name = gm.graph._target_to_str(nncf_node.node_type)
        op_counts[target_name] += 1
        if op_counts[target_name] > 1:
            target_name += f'_{op_counts[target_name]}'
        nncf_node_name_vs_fx_target_name[nncf_node.node_name] = target_name

    target_node_name_vs_qp = defaultdict(list)

    for qp in qsetup.quantization_points.values():
        assert isinstance(qp, PTQuantizationPoint)
        op_address = OperationAddress.from_str(qp.target_point.target_node_name)
        scope = op_address.scope_in_model
        if scope.scope_elements:
            calling_field_elements = [se.calling_field_name for se in scope.scope_elements[1:]]
            if any([x is None for x in calling_field_elements]):
                fx_module_path = None
            else:
                underscored_path = '_'.join(calling_field_elements)
                fx_module_path = f'self_{underscored_path}'
        else:
            fx_module_path = None
        if qp.is_weight_quantization_point():
            target_node_name = fx_module_path
        elif qp.is_activation_quantization_point():
            if fx_module_path is not None and hasattr(gm, fx_module_path):
                # QP is for one of the standard modules
                target_node_name = fx_module_path
                if op_address.call_order > 0:
                    target_node_name += f"_{op_address.call_order}"
            elif op_address.operator_name == MODEL_INPUT_OP_NAME:
                target_node_name = 'x'  # TODO (vshampor): better input matching
            else:
                # QP is for a free function
                target_node_name = nncf_node_name_vs_fx_target_name[qp.target_point.target_node_name]
        target_node_name_vs_qp[target_node_name].append(qp)

    # fx_node_name_counter = defaultdict(int)
    # fx_node_vs_canonical_name = {}
    # for node in original_nodes:
    #     fx_node_name_counter[node.name] += 1
    #     if fx_node_name_counter[node.name] > 1:
    #         fx_node_vs_canonical_name[node] = node.name + f'_{fx_node_name_counter[node.name]}'
    #     else:
    #         fx_node_vs_canonical_name[node] = node.name

    print(f"QPs in total: {len(qsetup.quantization_points)}")
    processed_qps = 0
    for idx, node in enumerate(original_nodes):
        node_name = node.name
        if node_name in target_node_name_vs_qp:
            qps = target_node_name_vs_qp[node_name]
            for qp in qps:
                print(f"QP: {qp.target_point}")
                processed_qps += 1
                if qp.is_weight_quantization_point():
                    found = False
                    for qinfo in qctrl.weight_quantizers.values():
                        for qtp in qinfo.affected_insertions:
                            if qtp == qp.target_point:
                                found = True
                                break
                        if found:
                            break

                    if not found:
                        raise RuntimeError("No fitting FQ found for a weight")
                    fq = convert_to_torch_fakequantizer(qinfo.quantizer_module_ref)

                    target_module = getattr(gm, node.target)
                    assert isinstance(target_module, (Conv2d, Linear))
                    quantize_weight_for_basic_module(gm, node, node_name, fq)
                elif qp.is_activation_quantization_point():
                    found = False
                    for qinfo in qctrl.non_weight_quantizers.values():
                        for qtp in qinfo.affected_insertions:
                            if qtp == qp.target_point:
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        raise RuntimeError("No fitting FQ found for a weight")
                    fq = convert_to_torch_fakequantizer(qinfo.quantizer_module_ref)
                    if qp.target_point.input_port_id is None:
                        quantize_activation_after(gm, node, node_name, fq)
                    else:
                        quantize_activation_before(gm, node, qp.target_point.input_port_id, node_name, fq)

    # for idx, node in enumerate(original_nodes):
    #     if node.op == 'call_module':
    #         target_module = getattr(gm, node.target)
    #         if isinstance(target_module, (Conv2d, Linear)):
    #             new_quantized_module_node = quantize_weight_for_basic_module(gm, node, idx)
    #             module_input = next(iter(new_quantized_module_node.all_input_nodes))
    #             quantize_activation_after(gm, module_input, idx)
    #             # quantize_weight_and_activation_for_basic_module(gm, node, idx)

    print(f"QPs processed: {processed_qps}")

    actual_fq_calls = 0
    for node in gm.graph.nodes:
        if node.op == 'call_module' and isinstance(getattr(gm, node.target), torch.ao.quantization.FakeQuantize):
            actual_fq_calls += 1
    print(f"Actual FQ calls in graph: {actual_fq_calls}")

    print(gm.graph)
    visualize_fx_graph(gm.graph, Path('after.dot'))
    gm.graph.lint()
    gm.recompile()
    # prev = get_compile_output()  # DOES NOT WORK - torch.compile calls the optimizing backend on all subgraphs repeatedly
    # if prev is None:
    #     set_compile_output(gm)
    return gm

@register_backend(name="_nncf_internal")
def compression_translator_compile_backend(gm: GraphModule, inputs) -> Callable:
    state = get_compression_state()
    return translate_compression(gm, state)
