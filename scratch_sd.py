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

from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import time
import logging
import sys
import os
import openvino.frontend.pytorch.torchdynamo.backend
import time

from nncf import NNCFConfig
from nncf.torch import create_compressed_model
from nncf.torch import register_module

from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
register_module()(LoRACompatibleConv)
register_module()(LoRACompatibleLinear)

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.safety_checker = None
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

_, compressed_unet = create_compressed_model(pipe.unet, NNCFConfig.from_dict({
    "input_info": [{"sample_size": [2, 4, 64, 64]},
                   {"sample_size": [2], "filler": "zeros"},
                   {"sample_size": [2, 77, 768]}],
    "compression": {"algorithm": "quantization"}
}))
# pipe.unet = compressed_unet.nncf.strip()
pipe.unet = compressed_unet

pipe.unet = torch.compile(pipe.unet, backend="openvino")

prompt = "cat on sofa"
batch_size = 1
generator = torch.Generator(device="cpu").manual_seed(1024)

# Warmup run
image = pipe(prompt, num_inference_steps=1, generator=generator, num_images_per_prompt=batch_size, width=512, height=512).images[0]

# Inference
start = time.time()
image = pipe(prompt, num_inference_steps=20, generator=generator, num_images_per_prompt=batch_size, width=512, height=512).images[0]
end = time.time()
print("Inference time: ", end - start)

image.save("cat_on_sofa.png")
