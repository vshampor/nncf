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
import abc
from abc import abstractmethod
from typing import Generic, Protocol, TypeVar

from nncf.common.graph import NNCFGraph

ModelType = TypeVar("ModelType")


class NNCFModelInterface(abc.ABC, Generic[ModelType]):
    @property
    @abstractmethod
    def graph(self) -> NNCFGraph:
        pass

    @abstractmethod
    def release(self) -> ModelType:
        pass

    def rebuild_graph(self) -> NNCFGraph:
        pass


class BasicNNCFModelInterface(NNCFModelInterface, Generic[ModelType]):
    def __init__(self, model: ModelType, graph: NNCFGraph = None):
        self._model_ref = model
        if graph is not None:
            self._graph = graph
        else:
            self._graph = self.rebuild_graph()

    @property
    def graph(self) -> NNCFGraph:
        return self._graph

    def release(self) -> ModelType:
        if hasattr(self._model_ref, "nncf"):
            delattr(self._model_ref, "nncf")
        return self._model_ref

    def rebuild_graph(self) -> NNCFGraph:
        from nncf.common.factory import NNCFGraphFactory

        self._graph = NNCFGraphFactory.create(self._model_ref)
        return self._graph


class NNCFModel(Protocol):
    @property
    @abstractmethod
    def nncf(self) -> NNCFModelInterface:
        pass


def wrap_model(model: ModelType, dataset=None) -> NNCFModel:
    # TODO: (vshampor) better model type dispatch, maybe use nncf.torch.wrap_model but extend it to all backends
    # in the fashion below?
    try:
        import onnx

        if isinstance(model, onnx.ModelProto):
            from nncf.onnx.nncf_model_wrapper import ONNXNNCFModelWrapper

            return ONNXNNCFModelWrapper(model)

        import torch

        if isinstance(model, torch.nn.Module):
            if dataset is None:
                raise RuntimeError("Wrapping torch models requires dataset to be set")
            from nncf.torch.dynamic_graph.io_handling import ExampleInputInfo
            from nncf.torch.nncf_network import NNCFNetwork

            if isinstance(model, NNCFNetwork):
                return model
            return NNCFNetwork(model, input_info=ExampleInputInfo.from_nncf_dataset(dataset))
    except ModuleNotFoundError:
        print("Wrong backend, fix me")

    model.nncf = BasicNNCFModelInterface(model)
    return model
