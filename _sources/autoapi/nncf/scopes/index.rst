:py:mod:`nncf.scopes`
=====================

.. py:module:: nncf.scopes

.. autoapi-nested-parse::

   Copyright (c) 2023 Intel Corporation
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.




Classes
~~~~~~~

.. autoapisummary::

   nncf.scopes.IgnoredScope




.. py:class:: IgnoredScope(names: Optional[List[str]] = None, patterns: Optional[List[str]] = None, types: Optional[List[str]] = None)

   Dataclass that contains description of the ignored scope.

   The ignored scope defines model sub-graphs that should be excluded from
   the compression process such as quantization, pruning and etc.

   Examples:

   ```
   import nncf

   # Exclude by node name:
   node_names = ['node_1', 'node_2', 'node_3']
   ignored_scope = nncf.IgnoredScope(names=node_names)

   # Exclude using regular expressions:
   patterns = ['node_\d']
   ignored_scope = nncf.IgnoredScope(patterns=patterns)

   # Exclude by operation type:

   # OpenVINO opset https://docs.openvino.ai/latest/openvino_docs_ops_opset.html
   operation_types = ['Multiply', 'GroupConvolution', 'Interpolate']
   ignored_scope = nncf.IgnoredScope(types=operation_types)

   # ONNX opset https://github.com/onnx/onnx/blob/main/docs/Operators.md
   operation_types = ['Mul', 'Conv', 'Resize']
   ignored_scope = nncf.IgnoredScope(types=operation_types)

   ...

   ```

   **Note** Operation types must be specified according to the model framework.


