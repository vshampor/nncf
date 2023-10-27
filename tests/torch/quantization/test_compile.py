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
import time
from typing import Callable

import pytest
import torch
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18

from nncf import NNCFConfig


def _get_compiled_resnet18(with_openvino: bool = False, with_nncf: bool = False) -> Callable:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if with_nncf:
        from nncf.torch import create_compressed_model
        _, model = create_compressed_model(model, NNCFConfig.from_dict({"input_info": {"sample_size": [1, 3, 224, 224]},
                                                                        "compression": {"algorithm": "quantization"}}))
        model = model.nncf.strip()
    if with_openvino:
        return torch.compile(model, backend="openvino")
    return torch.compile(model)


def _compile_and_run_resnet18(with_openvino: bool = False) -> torch.Tensor:
    compiled_model = _get_compiled_resnet18(with_openvino=with_openvino)
    return compiled_model(torch.ones([1, 3, 224, 224]))

def _compile_and_time_resnet18(with_openvino: bool = False, with_nncf: bool = False) -> float:
    compiled_model = _get_compiled_resnet18(with_openvino=with_openvino, with_nncf=with_nncf)
    compiled_model(torch.ones([1, 3, 224, 224]))
    compiled_model(torch.ones([1, 3, 224, 224]))
    start = time.time()
    for i in range(100):
        compiled_model(torch.ones([1, 3, 224, 224]))
    finish = time.time()
    return finish - start


# TODO (vshampor): move to isolated tests
def test_compile_works_with_nncf():
    before_nncf = _compile_and_run_resnet18()
    from nncf.torch import create_compressed_model
    after_nncf = _compile_and_run_resnet18()
    assert torch.allclose(before_nncf, after_nncf)

@pytest.fixture()
def ov_available():
    try:
        import openvino.frontend.pytorch.torchdynamo.backend
    except ImportError:
        pytest.skip("OV installation not found")


def test_compile_works_with_nncf_via_openvino(ov_available):
    before_nncf = _compile_and_run_resnet18(with_openvino=True)
    from nncf.torch import create_compressed_model
    after_nncf = _compile_and_run_resnet18(with_openvino=True)
    assert torch.allclose(before_nncf, after_nncf)


def test_ov_compile_is_faster_than_regular(ov_available):
    regular_time = _compile_and_time_resnet18()
    ov_time = _compile_and_time_resnet18(with_openvino=True)
    speedup = regular_time / ov_time
    print(f"Speedup due to OV compile: {speedup}")
    assert speedup > 1.1


def test_ov_int8_compile_is_faster_than_ov_compile(ov_available):
    ov_time = _compile_and_time_resnet18(with_openvino=True)
    ov_int8_time = _compile_and_time_resnet18(with_openvino=True, with_nncf=True)
    speedup = ov_time / ov_int8_time
    print(f"Speedup due to NNCF INT8 through OV compile: {speedup}")
    assert speedup > 1.1
