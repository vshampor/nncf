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
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import networkx as nx
import torch
from torch._dynamo import register_backend
from torch.fx import GraphModule
from torch.fx import Node
from torch.nn import Conv2d
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import Embedding

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


def _snake_case(s: str) -> str:
    # Copied as-is from torch code at torch.fx.graph._snake_case
    """
    Transforms the given string ``s`` to a Python-style variable name

    Examples:
        ``mod.snake_case`` -> ``mod.snake_case``
        ``mod.pascalCase``-> ``mod.pascal_case``
        ``mod.ALL_CAPS`` -> ``mod.all_caps``
    """
    chars = []
    prev_lower = False
    for c in s:
        if prev_lower and c.isupper():
            chars.append('_')
        chars.append(c.lower())
        prev_lower = c.islower()
    return ''.join(chars)

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


def get_call_fq_fn_node(fq_module: torch.ao.quantization.FakeQuantize,
                        input_node: Node,
                        scale_node: Node,
                        zero_point_node: Node,
                        graph: torch.fx.Graph) -> Node:

    if fq_module.is_per_channel:
        return graph.call_function(torch.fake_quantize_per_channel_affine,
                                   args=(input_node,
                                         scale_node,
                                         zero_point_node,
                                         fq_module.ch_axis,
                                         fq_module.quant_min,
                                         fq_module.quant_max))
    else:
        return graph.call_function(torch.fake_quantize_per_tensor_affine,
                                   args=(input_node,
                                         scale_node,
                                         zero_point_node,
                                         fq_module.quant_min,
                                         fq_module.quant_max))


def get_scale_zp_nodes(fq_qualname: str, graph: torch.fx.Graph) -> Tuple[Node, Node]:
    scale_node = graph.get_attr(f"{fq_qualname}.scale")
    zero_point_node = graph.get_attr(f"{fq_qualname}.zero_point")
    scale_node.append(zero_point_node)  # to ensure the order of scale/zero_point nodes for future processing
    return scale_node, zero_point_node


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

    if getattr(target_basic_module, 'bias', None) is not None:
        with graph.inserting_before(basic_module_call_node):
            bias_node = graph.create_node('get_attr', target_basic_module_accessor + '.bias',
                                          name=f'{target_basic_module.__class__.__name__}_bias_{uid}')
    else:
        bias_node = None

    next_node = bias_node if bias_node is not None else weight_node

    with graph.inserting_after(next_node):
        scale_node, zp_node = get_scale_zp_nodes(fq_w_module_name, graph)

    with graph.inserting_after(zp_node):
        call_fq_w_node = get_call_fq_fn_node(fq_module, weight_node, scale_node, zp_node, graph)

    with graph.inserting_after(call_fq_w_node):
        # Have to lower the original call_module[conv2d] node to the call_function ops,
        # because cannot setup a pre-op for FakeQuantizing the weight in the torch.fx.Graph domain
        if isinstance(target_basic_module, Conv1d):
            call_basic_module_op_node = graph.call_function(torch.nn.functional.conv1d, kwargs={
                'input': module_input,
                'weight': call_fq_w_node,
                'bias': bias_node,
                'stride': target_basic_module.stride,
                'padding': target_basic_module.padding,
                'dilation': target_basic_module.dilation,
                'groups': target_basic_module.groups
            })
        elif isinstance(target_basic_module, Conv2d):
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
        elif isinstance(target_basic_module, Embedding):
            call_basic_module_op_node = graph.call_function(torch.nn.functional.embedding, kwargs={
                'input': module_input,
                'weight': call_fq_w_node,
            })
        else:
            raise NotImplementedError
    basic_module_call_node.replace_all_uses_with(call_basic_module_op_node)
    graph.erase_node(basic_module_call_node)
    return call_basic_module_op_node


def quantize_weight_node_directly(graph_module: torch.fx.GraphModule, node_to_quantize: torch.fx.Node, uid: str,
                                  fq_module: torch.nn.Module = None) -> torch.fx.Node:
    fq_w_module_name = f'nncf_quantizers_fq_w_{uid}'

    if fq_module is None:
        fq_module = torch.ao.quantization.FakeQuantize(quant_min=0, quant_max=255)
    return _quantize_after(graph_module, node_to_quantize, fq_w_module_name, fq_module)


def quantize_activation_after(graph_module: torch.fx.GraphModule, node_to_quantize: torch.fx.Node, uid: str,
                              fq_module: torch.nn.Module = None) -> torch.fx.Node:
    fq_a_module_name = f'nncf_quantizers_fq_a_{uid}'
    if fq_module is None:
        fq_module = torch.ao.quantization.FakeQuantize(quant_min=0, quant_max=255)
    return _quantize_after(graph_module, node_to_quantize, fq_a_module_name, fq_module)


def _quantize_after(graph_module: torch.fx.GraphModule,
                    node_to_quantize: torch.fx.Node, fq_module_name: str,
                    fq_module: torch.nn.Module) -> torch.fx.Node:
    graph = graph_module.graph
    graph_module.add_submodule(fq_module_name, fq_module)

    with graph.inserting_after(node_to_quantize):
        scale_node, zp_node = get_scale_zp_nodes(fq_module_name, graph)

    with graph.inserting_after(zp_node):
        call_fq_node = get_call_fq_fn_node(fq_module, node_to_quantize, scale_node, zp_node, graph)
        for node in graph.nodes:
            if node is call_fq_node:
                continue
            node.replace_input_with(node_to_quantize, call_fq_node)
    return call_fq_node


def quantize_activation_before(graph_module: torch.fx.GraphModule, node_to_quantize: torch.fx.Node,
                               input_port_id: int,
                               uid: str,
                               fq_module: torch.nn.Module = None) -> torch.fx.Node:
    graph = graph_module.graph
    fq_a_module_name = f'nncf_quantizers_fq_a_{uid}_input_port{input_port_id}'
    if fq_module is None:
        fq_module = torch.ao.quantization.FakeQuantize(quant_min=0, quant_max=255)
    graph_module.add_submodule(fq_a_module_name, fq_module)
    with graph.inserting_before(node_to_quantize):
        scale_node, zp_node = get_scale_zp_nodes(fq_a_module_name, graph)

    with graph.inserting_after(zp_node):  # when inserting_before as above, scale_node will follow the zp node
        input_for_port = node_to_quantize.all_input_nodes[input_port_id]
        call_fq_a_node = get_call_fq_fn_node(fq_module, input_for_port, scale_node, zp_node, graph)
        node_to_quantize.replace_input_with(input_for_port, call_fq_a_node)
    return call_fq_a_node


CleanedFXModuleAttrName = str
FXNodeName = str
FXModuleAttrName = str


def _cleanup_fx_node_name(fx_module_attr_name: FXModuleAttrName) -> CleanedFXModuleAttrName:
    """
    Brings the FX module attr name to a normalized representation to which the NNCF's `Scope` objects can be mapped.
    This is done by removing non-informative parts of the `fx_module_attr_name`, leaving only the sequence of
    calling field names in the module hierarchy, and replaces any dots from the state_dict-like hierarchy syntax by
    underscores.
    :param fx_module_attr_name:
    :return:
    """

    # There doesn't seem to be any guarantees or rules specified by the torchdynamo tracing backend:
    # see comments in torch._guards.Guard
    split = fx_module_attr_name.split(sep='_')
    split = [x for x in split if x]  # split returns empty strings for repeated separators
    if len(split) > 1:
        if split[0] == "getattr":
            del split[0]
    if len(split) > 2:
        if split[0].lower() == "l" and split[1] == "self":
            del split[1]
            del split[0]
    return '_'.join(split).replace('.', '_')


def get_normalized_module_name_map(gm: GraphModule) -> Dict[FXModuleAttrName, CleanedFXModuleAttrName]:
    retval = {}
    for name, module in gm.named_modules():
        cleaned_name = _cleanup_fx_node_name(name)
        retval[cleaned_name] = name
    return retval


def _get_underscored_calling_field_sequence(scope: Scope) -> Optional[str]:
    if not scope.scope_elements:
        return None

    calling_field_elements = [se.calling_field_name for se in scope.scope_elements[1:]]
    if any([x is None for x in calling_field_elements]):
        return None  # TODO (vshampor): this means that the module was created ad-hoc, but how does compile represent this in graph?
    underscored_path = '_'.join(calling_field_elements)
    return underscored_path


def get_fx_module_path_for_scope(scope: Scope, clean_module_names_map: Dict[FXModuleAttrName, CleanedFXModuleAttrName]) -> Optional[FXModuleAttrName]:
    underscored_calling_field_sequence = _get_underscored_calling_field_sequence(scope)
    if underscored_calling_field_sequence is None:
        return None
    if underscored_calling_field_sequence not in clean_module_names_map:
        return None
    return clean_module_names_map[underscored_calling_field_sequence]


def get_possible_weight_node_name_for_weighted_module_scope(scope: Scope, possible_weight_node_names: Set[FXNodeName]) -> Optional[FXNodeName]:
    underscored_calling_field_sequence = _get_underscored_calling_field_sequence(scope)

    # TODO(vshampor) improve performance, don't search in this set each time
    node_candidates = []

    weight_suffix = "_weight"
    for possible_node_name in possible_weight_node_names:
        assert possible_node_name.endswith(weight_suffix)
        consumer_node_name = possible_node_name[:-len(weight_suffix)]
        if consumer_node_name.endswith(underscored_calling_field_sequence):
            node_candidates.append(possible_node_name)

    if len(node_candidates) > 1:
        print(f"More than one weight node candidate for scope {str(scope)}, lost the weight quantizer.")
        return None
    if not node_candidates:
        print(f"No weight node candidates for scope {str(scope)}, lost the weight quantizer.")
        return None

    return node_candidates[0]


def translate_compression(gm: GraphModule, state) -> GraphModule:
    visualize_fx_graph(gm.graph, Path('before.dot'))
    original_nodes: List[Node] = list(gm.graph.nodes)
    qsetup = PTQuantizerSetup.from_state(state['builder_state']['quantization']['quantizer_setup'])

    nncf_graph: NNCFGraph = state['graph']
    qctrl: QuantizationController = state['ctrl']
    nncf_node_name_vs_fx_node_name: Dict[str, str] = {}
    op_counts = Counter()
    for nncf_node in nncf_graph.get_all_nodes():
        target_name = gm.graph._target_to_str(nncf_node.node_type)
        op_counts[target_name] += 1
        if op_counts[target_name] > 1:
            target_name += f'_{op_counts[target_name] - 1}'
        nncf_node_name_vs_fx_node_name[nncf_node.node_name] = target_name

    target_node_name_vs_qp: Dict[str, List[PTQuantizationPoint]] = defaultdict(list)
    qp_to_target_node_name: Dict[PTQuantizationPoint, str] = {}
    clean_module_attr_name_map = get_normalized_module_name_map(gm)
    possible_weight_node_names = {n.name for n in original_nodes if n.name.endswith("weight")}

    # Need to associate "placeholder" nodes, which denote inputs in fx.Graph and are identified by the
    # argname, and the NNCF input nodes, which are identified by positions of tensor args.
    # Looks like the "placeholder" nodes in the fx.Graph will have order equivalent to the arg order
    # in the original forward function.
    placeholder_node_names = []
    for node in original_nodes:
        if node.op == "placeholder": # and torch.fx.node._type_repr(node.type) == "torch.Tensor":
            placeholder_node_names.append(node.name)

    for qp in qsetup.quantization_points.values():
        assert isinstance(qp, PTQuantizationPoint)
        op_address = OperationAddress.from_str(qp.target_point.target_node_name)
        scope = op_address.scope_in_model
        fx_module_path = get_fx_module_path_for_scope(scope, clean_module_attr_name_map)
        if qp.is_weight_quantization_point():
            if fx_module_path is None:
                # The original module could have been lowered by dynamo into `call_function` nodes, but the
                # weight-accessing node will still be named based on the original module's hierarchy
                possible_node_name = get_possible_weight_node_name_for_weighted_module_scope(scope,
                                                                                             possible_weight_node_names)
                if possible_node_name is None:
                    # could not find the module, the QP will be lost
                    continue
                target_node_name = possible_node_name
            else:
                target_node_name = _snake_case(fx_module_path)
        elif qp.is_activation_quantization_point():
            if fx_module_path is not None and hasattr(gm, fx_module_path):
                # QP is for one of the standard modules
                target_node_name = _snake_case(fx_module_path)
                if op_address.call_order > 0:
                    target_node_name += f"_{op_address.call_order}"
            elif op_address.operator_name == MODEL_INPUT_OP_NAME:
                target_node_name = placeholder_node_names[op_address.call_order]
            else:
                # QP is for a free function
                target_node_name = nncf_node_name_vs_fx_node_name[qp.target_point.target_node_name]
        target_node_name_vs_qp[target_node_name].append(qp)
        qp_to_target_node_name[qp] = target_node_name

    # fx_node_name_counter = defaultdict(int)
    # fx_node_vs_canonical_name = {}
    # for node in original_nodes:
    #     fx_node_name_counter[node.name] += 1
    #     if fx_node_name_counter[node.name] > 1:
    #         fx_node_vs_canonical_name[node] = node.name + f'_{fx_node_name_counter[node.name]}'
    #     else:
    #         fx_node_vs_canonical_name[node] = node.name

    print(f"QPs in total: {len(qsetup.quantization_points)}")
    processed_qps = set()
    for idx, node in enumerate(original_nodes):
        node_name = node.name
        if node_name in target_node_name_vs_qp:
            qps = target_node_name_vs_qp[node_name]
            for qp in qps:
                print(f"QP: {qp.target_point}")
                processed_qps.add(qp)
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

                    target = getattr(gm, node.target)
                    if isinstance(target, torch.nn.Module):
                        assert isinstance(target, (Conv1d, Conv2d, Linear, Embedding))
                        quantize_weight_for_basic_module(gm, node, node_name, fq)
                    elif isinstance(target, str) and node.type == "get_attr":  # op was lowered already
                        quantize_activation_after(gm, node, node_name, fq)
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
                        raise RuntimeError("No fitting FQ found for an activation")
                    fq = convert_to_torch_fakequantizer(qinfo.quantizer_module_ref)
                    if qp.target_point.input_port_id is None:
                        quantize_activation_after(gm, node, node_name, fq)
                    else:
                        quantize_activation_before(gm, node, qp.target_point.input_port_id, node_name, fq)

    print(f"QPs processed: {len(processed_qps)}")
    missed_qps = set(qsetup.quantization_points.values()) - processed_qps
    print(f"Missed QPs: {len(missed_qps)}")
    linesep = '\n'
    missed_qp_str_list = []
    for q in missed_qps:
        missed_qp_str_list.append(f"{str(q.target_point)}")
    print(f"Missed QP locations: {linesep.join(missed_qp_str_list)}")

    actual_fq_calls = 0
    for node in gm.graph.nodes:
        fq_fns = [torch.fake_quantize_per_channel_affine, torch.fake_quantize_per_tensor_affine]
        if node.op == 'call_function' and node.target in fq_fns:
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


GRAPH_COUNT = 0

@register_backend
def _test(gm, inputs):
    print(f"Backend called, graph length: {len(list(gm.graph.nodes))}")
    global GRAPH_COUNT
    visualize_fx_graph(gm.graph, Path(f"fx_graph_{GRAPH_COUNT}.dot"))
    GRAPH_COUNT += 1
    return gm
