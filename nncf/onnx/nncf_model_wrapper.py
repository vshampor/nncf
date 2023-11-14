#  Copyright (c) 2023 Intel Corporation
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import onnx

from nncf.common.nncf_model import BasicNNCFModelInterface
from nncf.common.nncf_model import NNCFModel
from nncf.common.nncf_model import NNCFModelInterface


class ONNXNNCFModelWrapper(NNCFModel):
    def __init__(self, onnx_model: onnx.ModelProto):
        super().__setattr__("_model_proto",  onnx_model)
        super().__setattr__("_nncf_model_interface",  BasicNNCFModelInterface(self._model_proto))

    @property
    def nncf(self) -> BasicNNCFModelInterface:
        return self._nncf_model_interface

    def __getattr__(self, item):
        if item == "nncf":
            return super().__getattr__(self, "_nncf_model_interface")
        return getattr(self._model_proto, item)

    def __setattr__(self, item, value):
        setattr(self._model_proto, item, value)
