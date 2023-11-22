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
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import GraphProto
from onnx import NodeProto

from nncf.onnx.engine import ONNXEngine
from nncf.onnx.nncf_model_wrapper import ONNXNNCFModelWrapper
from tests.onnx.models import LinearModel


@pytest.fixture
def wrapped_onnx_model() -> ONNXNNCFModelWrapper:
    model = LinearModel().onnx_model
    wrapped_model = ONNXNNCFModelWrapper(model)
    return wrapped_model


def test_nncfmodel_preserves_onnx_modelproto_attrs():
    model = LinearModel().onnx_model
    wrapped_model = ONNXNNCFModelWrapper(model)
    missing_attrs = []
    for attr in dir(model):
        if not hasattr(wrapped_model, attr):
            missing_attrs.append(attr)
    assert missing_attrs == ["Extensions"]  # e.g. all attrs are accessible except for "Extensions"

    # some sanity checks
    assert isinstance(wrapped_model.graph, GraphProto)
    assert isinstance(next(iter(wrapped_model.graph.node)), NodeProto)


def test_nncfmodel_is_serializable(tmp_path: Path, wrapped_onnx_model: ONNXNNCFModelWrapper):
    file_path = tmp_path / "tmp.onnx"
    onnx.save_model(wrapped_onnx_model, tmp_path / "tmp.onnx")
    assert file_path.exists()


def test_can_set_modelproto_fields(wrapped_onnx_model: ONNXNNCFModelWrapper):
    assert wrapped_onnx_model.producer_name != "foo"
    wrapped_onnx_model.producer_name = "foo"
    assert wrapped_onnx_model.producer_name == "foo"


def test_wrapped_model_is_inferrable(wrapped_onnx_model: ONNXNNCFModelWrapper):
    engine = ONNXEngine(wrapped_onnx_model)
    engine.infer({LinearModel.INPUT_NAME: np.ndarray([1, 3, 32, 32], dtype=np.float32)})
