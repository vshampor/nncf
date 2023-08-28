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
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
from numpy import dtype
from numpy.ma import MaskedArray
from numpy.ma.core import MaskedConstant

from nncf import TargetDevice
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import NNCFTensorBackend
from nncf.common.tensor import TensorDtype


_DTYPE_MAP: Dict[TensorDtype, Any] = {
    TensorDtype.FLOAT32: np.float32,
    TensorDtype.INT64: np.int64
}

_INV_DTYPE_MAP = {v: k for k, v in _DTYPE_MAP.items()}
_INV_DTYPE_MAP[dtype('float32')] = TensorDtype.FLOAT32


class WrappingIterator:
    def __init__(self, orig_iter: Iterator):
        self._orig_iter = orig_iter

    def __iter__(self):
        return self

    def __next__(self):
        retval = next(self._orig_iter)
        return NPNNCFTensor(retval)


class NPNNCFTensor(NNCFTensor[np.ndarray]):
    """
    Implementation of NNCFTensor over NumPy.
    """

    def _is_native_bool(self, bool_result: Any) -> bool:
        return isinstance(bool_result, np.bool_)

    @property
    def ndim(self) -> int:
        return self._tensor.ndim

    @property
    def size(self) -> int:
        return self._tensor.size

    def __iter__(self) -> Iterator:
        return WrappingIterator(iter(self._tensor))

    def dot(self, other: "NPNNCFTensor") -> "NPNNCFTensor":
        return self.__class__(self._tensor.dot(other._tensor))

    def astype(self, dtype: TensorDtype) -> "NPNNCFTensor":
        return self.__class__(self._tensor.astype(_DTYPE_MAP[dtype]))

    @property
    def dtype(self) -> TensorDtype:
        local_dtype = self._tensor.dtype
        return _INV_DTYPE_MAP[local_dtype]

    def any(self) -> bool:
        return self._tensor.any()

    def all(self) -> bool:
        return self._tensor.all()

    def to_numpy(self) -> np.ndarray:
        return self._tensor

    @property
    def shape(self) -> List[int]:
        return self.tensor.shape

    @property
    def backend(self) -> Type[NNCFTensorBackend]:
        return NPNNCFTensorBackend

    def mean(self, axis: int, keepdims: bool = None) -> "NPNNCFTensor":
        if keepdims is None:
            keepdims = np._NoValue
        return self.__class__(np.mean(self.tensor, axis=axis, keepdims=keepdims))

    def median(self, axis: int, keepdims: bool = False) -> "NNCFTensor":
        return self.__class__(np.median(self.tensor, axis=axis, keepdims=keepdims))

    @property
    def device(self):
        return TargetDevice.CPU.value

    def is_empty(self) -> bool:
        return self.tensor.size == 0

    def reshape(self, *shape: Tuple[int, ...]) -> "NPNNCFTensor":
        return self.__class__(self.tensor.reshape(*shape))

    def min(self) -> float:
        return self._tensor.min()

    def max(self) -> float:
        return self._tensor.max()


class NPNNCFTensorBackend(NNCFTensorBackend):
    inf = np.inf

    @staticmethod
    def isclose_all(tensor1: NPNNCFTensor, tensor2: NPNNCFTensor, rtol=1e-05, atol=1e-08) -> bool:
        return bool(np.isclose(tensor1.tensor, tensor2.tensor, rtol, atol))

    @staticmethod
    def stack(tensor_list: List[NPNNCFTensor]) -> NPNNCFTensor:
        return NPNNCFTensor(np.stack([nt.tensor for nt in tensor_list]))

    @staticmethod
    def count_nonzero(mask: NPNNCFTensor) -> NPNNCFTensor:
        return NPNNCFTensor(np.count_nonzero(mask.tensor))

    @staticmethod
    def abs(tensor: NPNNCFTensor) -> NPNNCFTensor:
        return NPNNCFTensor(np.abs(tensor.tensor))

    @staticmethod
    def min(tensor: NPNNCFTensor, axis: int = None) -> NPNNCFTensor:
        return NPNNCFTensor(np.min(tensor.tensor, axis=axis))

    @staticmethod
    def max(tensor: NPNNCFTensor, axis: int = None) -> NPNNCFTensor:
        return NPNNCFTensor(np.max(tensor.tensor, axis=axis))

    @staticmethod
    def min_of_list(tensor_list: List[NNCFTensor], axis: int = None) -> NNCFTensor:
        return NPNNCFTensor(np.min([t.tensor for t in tensor_list], axis=axis))

    @staticmethod
    def max_of_list(tensor_list: List[NNCFTensor], axis: int = None) -> NNCFTensor:
        return NPNNCFTensor(np.max([t.tensor for t in tensor_list], axis=axis))

    @staticmethod
    def expand_dims(tensor: NPNNCFTensor, axes: List[int]) -> NPNNCFTensor:
        return NPNNCFTensor(np.expand_dims(tensor.tensor, axis=axes))

    @staticmethod
    def sum(tensor: NPNNCFTensor, axes: List[int]) -> NPNNCFTensor:
        return NPNNCFTensor(np.sum(tensor.tensor, axis=tuple(axes)))

    @staticmethod
    def transpose(tensor: NPNNCFTensor, axes: List[int]) -> NPNNCFTensor:
        return NPNNCFTensor(np.transpose(tensor.tensor, axes=axes))

    @staticmethod
    def eps(dtype: TensorDtype) -> float:
        return np.finfo(_DTYPE_MAP[dtype]).eps

    @staticmethod
    def median(tensor: NPNNCFTensor) -> NPNNCFTensor:
        return NPNNCFTensor(np.median(tensor.tensor))

    @staticmethod
    def clip(tensor: NPNNCFTensor, min_val: float, max_val: Optional[float] = None) -> NPNNCFTensor:
        return NPNNCFTensor(np.clip(tensor.tensor, a_min=min_val, a_max=max_val))

    @staticmethod
    def ones(shape: Union[int, List[int]], dtype: TensorDtype) -> NPNNCFTensor:
        return NPNNCFTensor(np.ones(shape, dtype=_DTYPE_MAP[dtype]))

    @staticmethod
    def squeeze(tensor: NPNNCFTensor) -> NPNNCFTensor:
        return NPNNCFTensor(np.squeeze(tensor.tensor))

    @staticmethod
    def power(tensor: NPNNCFTensor, pwr: float) -> NPNNCFTensor:
        return NPNNCFTensor(np.power(tensor.tensor, pwr))

    @staticmethod
    def quantile(tensor: NPNNCFTensor, quantile: Union[float, List[float]], axis: Union[int, List[int]] = None, keepdims: bool = False) -> Union[float, NNCFTensor]:
        retval = np.quantile(tensor.tensor, quantile, axis=axis, keepdims=keepdims)
        if not isinstance(quantile, list):
            return retval
        else:
            return NPNNCFTensor(retval)

    @staticmethod
    def mean_of_list(tensor_list: List[NPNNCFTensor], axis: int) -> NPNNCFTensor:
        return NPNNCFTensor(np.mean([x.tensor for x in tensor_list], axis=axis))

    @staticmethod
    def mean(x: "NPNNCFTensor", axis: int, keepdims: bool = False) -> NPNNCFTensor:
        return NPNNCFTensor(np.mean(x.tensor, axis, keepdims))

    @staticmethod
    def moveaxis(x: "NPNNCFTensor", src: int, dst: int) -> NPNNCFTensor:
        return NPNNCFTensor(np.moveaxis(x.tensor, src, dst))

    @staticmethod
    def logical_or(tensor1: NPNNCFTensor, tensor2: NPNNCFTensor) -> NPNNCFTensor:
        return NPNNCFTensor(np.logical_or(tensor1.tensor, tensor2.tensor))

    @staticmethod
    def masked_mean(tensor: NPNNCFTensor, mask: NPNNCFTensor, axis: int = None, keepdims: bool = False) -> NPNNCFTensor:
        masked_x = np.ma.array(tensor.tensor, mask=mask.tensor)
        result = np.ma.mean(masked_x, axis=axis, keepdims=False)
        if isinstance(result, (MaskedConstant, MaskedArray)):
            result = result.data
        return NPNNCFTensor(result)

    @staticmethod
    def masked_median(tensor: NPNNCFTensor, mask: NPNNCFTensor, axis: int = None, keepdims: bool = False) -> NPNNCFTensor:
        masked_x = np.ma.array(tensor.tensor, mask=mask.tensor)
        result = np.ma.median(masked_x, axis=axis, keepdims=False)
        if isinstance(result, (MaskedConstant, MaskedArray)):
            result = result.data
        return NPNNCFTensor(result.data)

    @staticmethod
    def concatenate(tensor_list: List[NPNNCFTensor]) -> NPNNCFTensor:
        return NPNNCFTensor(np.concatenate([t.tensor for t in tensor_list]))

    @staticmethod
    def amin(tensor: NPNNCFTensor, axis: List[int], keepdims: bool = None) -> NPNNCFTensor:
        return NPNNCFTensor(np.amin(tensor.tensor, axis, keepdims))

    @staticmethod
    def amax(tensor: NPNNCFTensor, axis: List[int], keepdims: bool = None) -> NPNNCFTensor:
        return NPNNCFTensor(np.amax(tensor.tensor, axis, keepdims))

    @staticmethod
    def minimum(tensor1: NPNNCFTensor, tensor2: NPNNCFTensor) -> NPNNCFTensor:
        return NPNNCFTensor(np.minimum(tensor1.tensor, tensor2.tensor))

    @staticmethod
    def maximum(tensor1: NPNNCFTensor, tensor2: NPNNCFTensor) -> NPNNCFTensor:
        return NPNNCFTensor(np.maximum(tensor1.tensor, tensor2.tensor))

    @staticmethod
    def unstack(tensor: NPNNCFTensor, axis: int = 0) -> List[NPNNCFTensor]:
        return [NPNNCFTensor(np.squeeze(e, axis)) for e in np.split(tensor.tensor, tensor.tensor.shape[axis], axis=axis)]
