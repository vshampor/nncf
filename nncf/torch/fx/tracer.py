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


class NNCFTracer(Tracer):
    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        module_qualified_name = self.path_of_module(m)
        with ScopeContextManager(self.scope, Scope(module_qualified_name, type(m))) as _scope:
            # module_stack is an ordered dict so writing then deleting the
            # entry is equivalent to push/pop on a list
            self.module_stack[_scope.module_path] = _scope.module_type
            if not self.is_leaf_module(m, module_qualified_name):
                ret_val = forward(*args, **kwargs)
            else:
                ret_val = self.create_proxy("call_module", module_qualified_name, args, kwargs)
            key, _ = self.module_stack.popitem(last=True)
            assert key == _scope.module_path, f" Unexpected key {key}"

        return ret_val


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