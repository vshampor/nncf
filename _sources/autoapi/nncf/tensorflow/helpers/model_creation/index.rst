:py:mod:`nncf.tensorflow.helpers.model_creation`
================================================

.. py:module:: nncf.tensorflow.helpers.model_creation

.. autoapi-nested-parse::

   Copyright (c) 2023 Intel Corporation
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.





Functions
~~~~~~~~~

.. autoapisummary::

   nncf.tensorflow.helpers.model_creation.create_compressed_model



.. py:function:: create_compressed_model(model: tensorflow.keras.Model, config: nncf.NNCFConfig, compression_state: Optional[Dict[str, Any]] = None) -> Tuple[nncf.api.compression.CompressionAlgorithmController, tensorflow.keras.Model]

   The main function used to produce a model ready for compression fine-tuning
   from an original TensorFlow Keras model and a configuration object.

   :param model: The original model. Should have its parameters already loaded
       from a checkpoint or another source.
   :param config: A configuration object used to determine the exact compression
       modifications to be applied to the model.
   :param compression_state: compression state to unambiguously restore the compressed model.
       Includes builder and controller states. If it is specified, trainable parameter initialization will be skipped
       during building.
   :return: A tuple (compression_ctrl, compressed_model) where
       - compression_ctrl: The controller of the compression algorithm.
       - compressed_model: The model with additional modifications
           necessary to enable algorithm-specific compression during fine-tuning.


