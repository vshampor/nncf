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

# GLIGEN inference with PyTorch with and without OpenVINO backend
# Usage: python gligen.py {ov|pytorch} {inpainting|generation}"
# Note: the script uses 20 inference steps to compare performance. Use 50 steps for better image quality

import os
import sys
import time

import openvino.torch
import torch
from diffusers import StableDiffusionGLIGENPipeline
from diffusers.utils import load_image

from nncf import NNCFConfig
from nncf.torch import create_compressed_model

assert len(sys.argv) == 3, "Usage: python gligen.py {ov|pytorch} {inpainting|generation}"

USE_OPENVINO = sys.argv[1] == "ov"
task = sys.argv[2]
assert task in ["inpainting", "generation"]

print("Framework: ", "OpenVINO" if USE_OPENVINO else "PyTorch")
if USE_OPENVINO:
    print(f"OpenVINO caching: {os.environ.get('OPENVINO_TORCH_MODEL_CACHING')}")

if task == "inpainting":
    # Insert objects described by text at the region defined by bounding boxes
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        "masterful/gligen-1-4-inpainting-text-box",  # variant="fp16", torch_dtype=torch.float16
    )

    # _, compressed_unet = create_compressed_model(pipe.unet, NNCFConfig.from_dict({
    #     "input_info": [{"sample_size": [2, 9, 64, 64]},
    #                    {"sample_size": [2], "filler": "zeros"},
    #                    {"sample_size": [2, 77, 768]}],
    #     "compression": {"algorithm": "quantization"}
    # }))
    #
    # _, compressed_te = create_compressed_model(pipe.text_encoder, NNCFConfig.from_dict({
    #     "input_info": [{"sample_size": [1, 5], "keyword": "input_ids", "type": "long"},
    #                    {"sample_size": [1, 5], "keyword": "attention_mask", "type": "long"}],
    #     "compression": {"algorithm": "quantization"}
    # }))
    #
    _, compressed_vae_enc = create_compressed_model(pipe.vae.encoder, NNCFConfig.from_dict({
        "input_info": {"sample_size": [1, 3, 512, 512]},
        "compression": {"algorithm": "quantization"}
    }))

    _, compressed_vae_dec = create_compressed_model(pipe.vae.decoder, NNCFConfig.from_dict({
        "input_info": {"sample_size": [1, 4, 64, 64]},
        "compression": {"algorithm": "quantization"}
    }))
    #
    # pipe.unet = compressed_unet.nncf.strip()
    # pipe.text_encoder = compressed_te.nncf.strip()
    pipe.vae.encoder = compressed_vae_enc.nncf.strip()
    pipe.vae.decoder = compressed_vae_dec.nncf.strip()

    if USE_OPENVINO:
        pipe.unet = torch.compile(pipe.unet, backend="openvino")
        pipe.text_encoder = torch.compile(pipe.text_encoder, backend="openvino")
        pipe.vae = torch.compile(pipe.vae, backend="openvino")

    input_image = load_image(
        "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/livingroom_modern.png"
    )
    prompts = ["a birthday cake", "a candle"]
    boxes = [[0.2676, 0.6088, 0.4773, 0.7183]]
    phrases = ["a birthday cake", "a candle"]

    for i in range(2):
        start = time.perf_counter()
        images = pipe(
            prompt=prompts[i],
            gligen_phrases=phrases[i],
            gligen_inpaint_image=input_image,
            gligen_boxes=boxes,
            gligen_scheduled_sampling_beta=1,
            output_type="pil",
            num_inference_steps=20,
        ).images
        end = time.perf_counter()
        print(f"Inpainting duration {i}: {end-start:.2f} seconds")
        images[0].save(f"./gligen-1-4-inpainting-text-box-{i}.jpg")
    print("Trying to save_pretrained:")
    try:
        pipe.save_pretrained("/tmp/gligen_saved")
    except Exception as e:
        print(f"Failed - {e}")


    print("Trying to torch.onnx.export the VAE encoder:")
    try:
        torch.onnx.export(pipe.vae.encoder, (torch.ones([1, 3, 512, 512]), ), "/tmp/gligen_exported.onnx")
    except Exception as e:
        print(f"Failed - {e}")



elif task == "generation":
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        "masterful/gligen-1-4-generation-text-box",
    )

    if USE_OPENVINO:
        pipe.unet = torch.compile(pipe.unet, backend="openvino")
        pipe.text_encoder = torch.compile(pipe.text_encoder, backend="openvino")
        pipe.vae = torch.compile(pipe.vae, backend="openvino")

    prompts = [
        "a waterfall and a modern high speed train running through the tunnel in a beautiful forest with fall foliage",
        "a river and an antique car running through the tunnel in a beautiful forest with spring foliage",
    ]
    boxes = [[0.1387, 0.2051, 0.4277, 0.7090], [0.4980, 0.4355, 0.8516, 0.7266]]
    phrases = [["a waterfall", "a modern high speed train running through the tunnel"],["a river", "an antique car"]]

    for i, prompt in enumerate(prompts):
        start = time.perf_counter()
        images = pipe(
            prompt=prompt,
            gligen_phrases=phrases[i],
            gligen_boxes=boxes,
            gligen_scheduled_sampling_beta=1,
            output_type="pil",
            num_inference_steps=20,
        ).images
        end = time.perf_counter()
        images[0].save(f"./gligen-1-4-generation-text-box-{i}.jpg")
        print(f"Generation duration {i}: {end-start:.2f} seconds")