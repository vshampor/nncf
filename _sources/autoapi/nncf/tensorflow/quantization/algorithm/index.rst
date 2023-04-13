:py:mod:`nncf.tensorflow.quantization.algorithm`
================================================

.. py:module:: nncf.tensorflow.quantization.algorithm

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




Classes
~~~~~~~

.. autoapisummary::

   nncf.tensorflow.quantization.algorithm.QuantizationController




.. py:class:: QuantizationController(target_model, config, op_names: List[str])

   Bases: :py:obj:`nncf.common.compression.BaseCompressionAlgorithmController`

   Contains the implementation of the basic functionality of the compression controller.

   .. py:method:: strip_model(model: tensorflow.keras.Model, do_copy: bool = False) -> tensorflow.keras.Model

      Strips auxiliary layers that were used for the model compression, as it's
      only needed for training. The method is used before exporting the model
      in the target format.

      :param model: The compressed model.
      :param do_copy: Modify copy of the model, defaults to False.
      :return: The stripped model.


   .. py:method:: statistics(quickly_collected_only: bool = False) -> nncf.common.statistics.NNCFStatistics

      Returns a `Statistics` class instance that contains compression algorithm statistics.

      :param quickly_collected_only: Enables collection of the statistics that
          don't take too much time to compute. Can be helpful for the case when
          need to keep track of statistics on each training batch/step/iteration.
      :return: A `Statistics` class instance that contains compression algorithm statistics.


   .. py:method:: compression_stage() -> nncf.api.compression.CompressionStage

      Returns the compression stage. Should be used on saving best checkpoints
      to distinguish between uncompressed, partially compressed, and fully
      compressed models.

      :return: The compression stage of the target model.



