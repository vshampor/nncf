import numpy as np
from pyinstrument import Profiler

profiler = Profiler()

from nncf.onnx.tensor import ONNXNNCFTensor


profiler.start()
backend_tensor = np.ones([1])
for _ in range(1000000):
    common_tensor = ONNXNNCFTensor(backend_tensor)
    common_tensor.mean(0)
profiler.stop()
profiler.print()
