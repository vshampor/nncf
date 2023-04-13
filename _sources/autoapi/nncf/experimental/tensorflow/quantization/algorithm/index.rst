:py:mod:`nncf.experimental.tensorflow.quantization.algorithm`
=============================================================

.. py:module:: nncf.experimental.tensorflow.quantization.algorithm

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

   nncf.experimental.tensorflow.quantization.algorithm.QuantizationControllerV2




.. py:class:: QuantizationControllerV2(target_model, config, op_names: List[str])

   Bases: :py:obj:`nncf.tensorflow.quantization.algorithm.QuantizationController`

   Contains the implementation of the basic functionality of the compression controller.

   .. py:method:: strip_model(model: nncf.experimental.tensorflow.nncf_network.NNCFNetwork, do_copy: bool = False) -> nncf.experimental.tensorflow.nncf_network.NNCFNetwork

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


   .. py:method:: prepare_for_export() -> None

      Prepare the compressed model for deployment.



