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
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch.fx import GraphModule
from torch.fx import Tracer
from torch.fx.proxy import ScopeContextManager

from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.layers import NNCFBatchNorm1d
from nncf.torch.layers import NNCFBatchNorm2d
from nncf.torch.layers import NNCFBatchNorm3d


class NNCFTracer(Tracer):
    def is_leaf_module(self, m: torch.nn.Module, qualified_name: str):
        if isinstance(m, (NNCFBatchNorm1d, NNCFBatchNorm2d, NNCFBatchNorm3d)):
            return True
        return False


def symbolic_trace(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = None,
) -> GraphModule:
    tracer = NNCFTracer()
    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    return GraphModule(tracer.root, graph, name)