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

from typing import List, Optional, Union
from typing import Type

import numpy as np

from nncf.common.pruning.symbolic_mask import SymbolicMask
from nncf.common.pruning.tensor_processor import NNCFPruningBaseTensorProcessor
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import TensorDtype
from nncf.common.tensor_impl_np import NPNNCFTensor


class NPNNCFTensorProcessor(NNCFPruningBaseTensorProcessor):
    @classmethod
    def concatenate(cls, tensors: List[NNCFTensor], axis: int) -> NNCFTensor:
        for tensor in tensors[1:]:
            assert tensors[0].device == tensor.device

        ret_tensor = np.concatenate([t.tensor for t in tensors], axis=axis)
        return NPNNCFTensor(ret_tensor)

    @classmethod
    def ones(cls, shape: Union[int, List[int]], device: Optional[str]) -> NNCFTensor:
        return NPNNCFTensor(np.ones(shape))

    @classmethod
    def assert_allclose(cls, tensors: List[np.ndarray]) -> None:
        for input_mask in tensors[1:]:
            np.testing.assert_allclose(tensors[0], input_mask)

    @classmethod
    def repeat(cls, tensor: NNCFTensor, repeats: int) -> NNCFTensor:
        ret_tensor = np.repeat(tensor.tensor, repeats)
        return NPNNCFTensor(ret_tensor)

    @classmethod
    def elementwise_mask_propagation(cls, input_masks: List[SymbolicMask]) -> SymbolicMask:
        cls.assert_allclose([np.asarray(im.shape) for im in input_masks])
        return input_masks[0]

    @classmethod
    def split(cls, tensor: NNCFTensor, output_shapes: List[int]) -> List[NNCFTensor]:
        chunks = len(output_shapes)
        ret_tensors = np.split(tensor.tensor, chunks)
        return [NPNNCFTensor(ret_tensor) for ret_tensor in ret_tensors]
