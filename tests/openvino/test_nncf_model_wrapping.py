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
import openvino as ov

from nncf.common.nncf_model import wrap_model
from tests.openvino.native.models import LinearModel


def test_nncf_model_wrapping_does_not_change_object_id_and_class():
    model = LinearModel().ov_model
    wrapped_model = wrap_model(model)
    assert wrapped_model is model
    assert wrapped_model.__class__ is model.__class__


def test_can_serialize_compile_and_infer_nncf_model(tmp_path: Path):
    model = LinearModel().ov_model
    wrapped_model = wrap_model(model)
    serialize_path = tmp_path / "tmp.xml"
    ov.save_model(wrapped_model, serialize_path)
    compiled_model = ov.compile_model(serialize_path)
    compiled_model(np.ndarray([1, 3, 4, 2], dtype=np.float32))
