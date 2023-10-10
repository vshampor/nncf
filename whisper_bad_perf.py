#!/usr/bin/env python
# coding: utf-8
from copy import deepcopy
from openvino import Core
from scipy.io import wavfile
import whisper
from contextlib import contextmanager
from functools import partial
import openvino as ov
from typing import Optional
import torch
import nncf


MODEL_TYPE = "encoder"  # set this to "decoder" to process decoder part of the whisper model

base_model = whisper.load_model("base").to("cpu").eval()

COLLECT_CALIBRATION_DATA = False
calibration_data = []

@contextmanager
def calibration_data_collection():
    global COLLECT_CALIBRATION_DATA
    try:
        COLLECT_CALIBRATION_DATA = True
        yield
    finally:
        COLLECT_CALIBRATION_DATA = False


def encoder_forward(orig_forward, module, mel: torch.Tensor):
    if COLLECT_CALIBRATION_DATA:
        calibration_data.append(mel)
    return orig_forward(mel)

def decoder_forward(orig_forward, module, x: torch.Tensor, xa: torch.Tensor, kv_cache: Optional[dict] = None):
    feed_dict = {'x': x, 'xa': xa}
    if COLLECT_CALIBRATION_DATA:
        calibration_data.append(feed_dict)
    res = orig_forward(**feed_dict)
    return res

if MODEL_TYPE == "encoder":
    target_model = base_model.encoder
    replaced_forward = encoder_forward
    sqa = 0.5
elif MODEL_TYPE == "decoder":
    target_model = base_model.decoder
    replaced_forward = decoder_forward
    sqa = 0.95


orig_forward = target_model.forward
target_model.forward = partial(replaced_forward, orig_forward, target_model)


task = "translate"

from datasets import load_dataset
from tqdm import tqdm

CALIBRATION_DATASET_SIZE = 1 # 30
calibration_dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True).take(CALIBRATION_DATASET_SIZE)

with calibration_data_collection():
    for data_item in tqdm(calibration_dataset, desc="Collecting calibration data", total=CALIBRATION_DATASET_SIZE):
        base_model.transcribe(data_item["audio"]["array"].astype("float32"), task=task)


del target_model.__dict__["forward"]

target_model_fp32 = deepcopy(target_model)

if MODEL_TYPE == "encoder":
    calibration_dataloader = nncf.Dataset(calibration_data)
    input_infos = { "sample_size": list(calibration_data[0].shape) }
elif MODEL_TYPE == "decoder":
    calibration_dataloader = nncf.Dataset(calibration_data, transform_func=lambda x: (x['x'], x['xa']))
    input_infos = [{ "sample_size": list(calibration_data[0]['x'].shape) , "type": "long"}, { "sample_size": list(calibration_data[0]['xa'].shape) } ]


print(f"Quantizing {MODEL_TYPE}...")
#quantized_model = nncf.quantize(
#    model=target_model,
#    calibration_dataset=calibration_dataloader,
#    subset_size=1, #len(calibration_data),
#    model_type=nncf.ModelType.TRANSFORMER,
#    advanced_parameters=nncf.AdvancedQuantizationParameters(
#    smooth_quant_alpha=sqa      # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
#    )
#)
from nncf import NNCFConfig
from nncf.torch import create_compressed_model
_, quantized_model = create_compressed_model(target_model, NNCFConfig.from_dict({"input_info": input_infos, "compression": {"algorithm" : "quantization"}}))


import time
import numpy as np

def calculate_call_inference_time(model, dataset):
    inference_time = []
    start = time.perf_counter()
    for data_item in tqdm(dataset[:100], desc="Measuring performance"):
        model(*data_item)

    end = time.perf_counter()
    delta = end - start
    return delta

if MODEL_TYPE == "encoder":
    inference_data = [(torch.ones_like(calibration_data[0]),) for i in range(100)]
elif MODEL_TYPE == "decoder":
    inference_data = [(torch.ones_like(calibration_data[0]['x']), torch.ones_like(calibration_data[0]['xa'])) for i in range(100)]


import openvino.frontend.pytorch.torchdynamo.backend

print("VSHAMPOR: compiling FP32 model")
compiled_ov_fp32_model = torch.compile(target_model_fp32, backend="openvino")

print("VSHAMPOR: executing FP32 model")
# measure twice to exclude warmup effects
time_fp32 = calculate_call_inference_time(compiled_ov_fp32_model, inference_data)
time_fp32 = calculate_call_inference_time(compiled_ov_fp32_model, inference_data)

print("VSHAMPOR: compiling INT8 model")
compiled_ov_quantized_model = torch.compile(quantized_model, backend="openvino")

print("VSHAMPOR: executing INT8 model")
time_int8 = calculate_call_inference_time(compiled_ov_quantized_model, inference_data)
time_int8 = calculate_call_inference_time(compiled_ov_quantized_model, inference_data)

print(f"Performance speedup ({MODEL_TYPE}): {time_fp32 / time_int8:.3f}")
