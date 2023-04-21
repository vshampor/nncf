:py:mod:`nncf`
==============

.. py:module:: nncf

.. autoapi-nested-parse::

   Copyright (c) 2019-2023 Intel Corporation
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.



Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   common/index.rst
   config/index.rst
   tensorflow/index.rst
   torch/index.rst



Classes
~~~~~~~

.. autoapisummary::

   nncf.NNCFConfig
   nncf.Dataset
   nncf.IgnoredScope
   nncf.ModelType
   nncf.TargetDevice
   nncf.QuantizationPreset



Functions
~~~~~~~~~

.. autoapisummary::

   nncf.quantize
   nncf.quantize_with_accuracy_control



.. py:class:: NNCFConfig(*args, **kwargs)

   Bases: :py:obj:`dict`

   A regular dictionary object extended with some utility functions.

   .. py:method:: from_dict(nncf_dict)
      :classmethod:

      Load NNCF config from dict;
      The dict must contain only json supported primitives.


   .. py:method:: get_redefinable_global_param_value_for_algo(param_name: str, algo_name: str) -> Optional

      Some parameters can be specified both on the global NNCF config .json level (so that they apply
      to all algos), and at the same time overridden in the algorithm-specific section of the .json.
      This function returns the value that should apply for a given algorithm name, considering the
      exact format of this config.

      :param param_name: The name of a parameter in the .json specification of the NNCFConfig, that may
        be present either at the top-most level of the .json, or at the top level of the algorithm-specific
        subdict.
      :param algo_name: The name of the algorithm (among the allowed algorithm names in the .json) for which
        the resolution of the redefinable parameter should occur.
      :return: The value of the parameter that should be applied for the algo specified by `algo_name`.



.. py:class:: Dataset(data_source: Iterable[DataItem], transform_func: Optional[Callable[[DataItem], ModelInput]] = None)

   Bases: :py:obj:`Generic`\ [\ :py:obj:`DataItem`\ , :py:obj:`ModelInput`\ ]

   The `nncf.Dataset` class defines the interface by which compression algorithms
   retrieve data items from the passed data source object. These data items are used
   for different purposes, for example, model inference and model validation. It depends
   on the compression algorithm.

   If the data item has been returned from the data source per iteration and it cannot be
   used as input for model inference, the transformation function is used to extract the
   model's input from this data item. For example, in supervised learning, the data item
   usually contains both examples and labels. So transformation function should extract
   the examples from the data item.

   .. py:method:: get_data(indices: Optional[List[int]] = None) -> Iterable[DataItem]

      Returns the iterable object that contains selected data items from the data source as-is.

      :param indices: The zero-based indices of data items that should be selected from
          the data source. The indices should be sorted in ascending order. If indices are
          not passed all data items are selected from the data source.
      :return: The iterable object that contains selected data items from the data source as-is.


   .. py:method:: get_inference_data(indices: Optional[List[int]] = None) -> Iterable[ModelInput]

      Returns the iterable object that contains selected data items from the data source, for which
      the transformation function was applied. The item, which was returned per iteration from this
      iterable, can be used as the model's input for model inference.

      :param indices: The zero-based indices of data items that should be selected from
          the data source. The indices should be sorted in ascending order. If indices are
          not passed all data items are selected from the data source.
      :return: The iterable object that contains selected data items from the data source, for which
          the transformation function was applied.



.. py:class:: IgnoredScope

   Dataclass that contains description of the ignored scope.

   The ignored scope defines model sub-graphs that should be excluded from
   the compression process such as quantization, pruning and etc.

   Examples:

   ```
   import nncf

   # Exclude by node name:
   node_names = ['node_1', 'node_2', 'node_3']
   ignored_scope = nncf.IgnoredScope(names=node_names)

   # Exclude using regular expressions:
   patterns = ['node_\d']
   ignored_scope = nncf.IgnoredScope(patterns=patterns)

   # Exclude by operation type:

   # OpenVINO opset https://docs.openvino.ai/latest/openvino_docs_ops_opset.html
   operation_types = ['Multiply', 'GroupConvolution', 'Interpolate']
   ignored_scope = nncf.IgnoredScope(types=operation_types)

   # ONNX opset https://github.com/onnx/onnx/blob/main/docs/Operators.md
   operation_types = ['Mul', 'Conv', 'Resize']
   ignored_scope = nncf.IgnoredScope(types=operation_types)

   ...

   ```

   **Note** Operation types must be specified according to the model framework.

   :param names: List of ignored node names.
   :param patterns: List of regular expressions that define patterns for names of
       ignored nodes.
   :param types: List of ignored operation types.


.. py:class:: ModelType

   Bases: :py:obj:`enum.Enum`

   Describes the model type the specificity of which will be taken into
   account during compression.

   :param TRANSFORMER: Transformer-based models
       (https://arxiv.org/pdf/1706.03762.pdf)


.. py:class:: TargetDevice

   Bases: :py:obj:`enum.Enum`

   Describes the target device the specificity of which will be taken
   into account while compressing in order to obtain the best performance
   for this type of device.

   :param ANY:
   :param CPU:
   :param GPU:
   :param VPU:


.. py:class:: QuantizationPreset

   Bases: :py:obj:`enum.Enum`

   Generic enumeration.

   Derive from this class to define new enumerations.


.. py:function:: quantize(model: nncf.api.compression.TModel, calibration_dataset: nncf.data.Dataset, preset: nncf.common.quantization.structs.QuantizationPreset = QuantizationPreset.PERFORMANCE, target_device: nncf.parameters.TargetDevice = TargetDevice.ANY, subset_size: int = 300, fast_bias_correction: bool = True, model_type: Optional[nncf.parameters.ModelType] = None, ignored_scope: Optional[nncf.scopes.IgnoredScope] = None) -> nncf.api.compression.TModel

   Applies post-training quantization algorithm to provided model.

   :param model: A model to be quantized.
   :param calibration_dataset: A representative dataset for the
       calibration process.
   :param preset: A preset that controls the quantization mode
       (symmetric and asymmetric). It can take the following values:
       - `performance`: Symmetric quantization of weights and activations.
       - `mixed`: Symmetric quantization of weights and asymmetric
         quantization of activations.
   :param target_device: A target device the specificity of which will be taken
       into account while compressing in order to obtain the best performance
       for this type of device.
   :param subset_size: Size of a subset to calculate activations
       statistics used for quantization.
   :param fast_bias_correction: Setting this option to `False` enables a different
       bias correction method which is more accurate, in general, and takes
       more time but requires less memory.
   :param model_type: Model type is needed to specify additional patterns
       in the model. Supported only `transformer` now.
   :param ignored_scope: An ignored scope that defined the list of model control
       flow graph nodes to be ignored during quantization.
   :return: The quantized model.


.. py:function:: quantize_with_accuracy_control(model: nncf.api.compression.TModel, calibration_dataset: nncf.data.Dataset, validation_dataset: nncf.data.Dataset, validation_fn: Callable[[Any, Iterable[Any]], float], max_drop: float = 0.01, preset: nncf.common.quantization.structs.QuantizationPreset = QuantizationPreset.PERFORMANCE, target_device: nncf.parameters.TargetDevice = TargetDevice.ANY, subset_size: int = 300, fast_bias_correction: bool = True, model_type: Optional[nncf.parameters.ModelType] = None, ignored_scope: Optional[nncf.scopes.IgnoredScope] = None) -> nncf.api.compression.TModel

   Applies post-training quantization algorithm with accuracy control to provided model.

   :param model: A model to be quantized.
   :param calibration_dataset: A representative dataset for the calibration process.
   :param validation_dataset: A dataset for the validation process.
   :param validation_fn: A validation function to validate the model. It should take
       two argumets:
       - `model`: model to be validate.
       - `validation_dataset`: dataset that provides data items to
             validate the provided model.
       The function should return the value of the metric with the following meaning:
       A higher value corresponds to better performance of the model.
   :param max_drop: The maximum absolute accuracy drop that should be achieved after the quantization.
   :param preset: A preset that controls the quantization mode.
   :param target_device: A target device the specificity of which will be taken
       into account while compressing in order to obtain the best performance
       for this type of device.
   :param subset_size: Size of a subset to calculate activations
       statistics used for quantization.
   :param fast_bias_correction: Setting this option to `False` enables a different
       bias correction method which is more accurate, in general, and takes
       more time but requires less memory.
   :param model_type: Model type is needed to specify additional patterns
       in the model. Supported only `transformer` now.
   :param ignored_scope: An ignored scope that defined the list of model control
       flow graph nodes to be ignored during quantization.
   :return: The quantized model.


