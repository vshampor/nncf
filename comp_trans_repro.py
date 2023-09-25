import time

import torch
torch.set_num_threads(1)

from torchvision.models.resnet import resnet18


def profile(model, inputs):
    model(inputs)  # to trigger compilation if needed
    start = time.time()
    for i in range(1000):
        model(inputs)
    duration = time.time() - start
    print(f"Inference took {duration}s")


#model = resnet18().cpu()
#input_size = [1, 3, 224, 224]
#inputs = torch.ones(input_size)

import transformers
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
input_size = [1, 128]
inputs = torch.ones(input_size, dtype=torch.long)

from copy import deepcopy
another_model = deepcopy(model)


import openvino.frontend.pytorch.torchdynamo.backend
print("Compiling FP32 model with OV backend...")
ov_compiled_model = torch.compile(model, backend="openvino")
print("Profiling FP32 OV-compiled model...")
profile(ov_compiled_model, inputs)

from nncf.torch import create_compressed_model
from nncf import NNCFConfig

ctrl, compressed_model = create_compressed_model(another_model, NNCFConfig.from_dict({
    "input_info": {"sample_size": input_size, "type": "long"},
    "compression": {"algorithm": "quantization"}
}))

print("Compiling NNCF-INT8 model with OV backend...")
ov_compiled_compressed_model = torch.compile(compressed_model, backend="openvino")
print("Profiling NNCF-INT8 OV-compiled model...")
profile(ov_compiled_compressed_model, inputs)

