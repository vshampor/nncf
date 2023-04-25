:py:mod:`nncf.torch.quantization.algo`
======================================

.. py:module:: nncf.torch.quantization.algo

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




Classes
~~~~~~~

.. autoapisummary::

   nncf.torch.quantization.algo.QuantizationController




.. py:class:: QuantizationController(target_model, config, debug_interface, weight_quantizers, non_weight_quantizers, groups_of_adjacent_quantizers, quantizers_input_shapes, build_time_metric_info = None, build_time_range_init_params = None)

   Bases: :py:obj:`QuantizationControllerBase`

   Serves as a handle to the additional modules, parameters and hooks inserted
   into the original uncompressed model in order to enable algorithm-specific compression.
   Hosts entities that are to be used during the training process, such as compression scheduler and
   compression loss.

   .. py:method:: prepare_for_export()

      Prepare the compressed model for deployment.


   .. py:method:: distributed()

      Should be called when distributed training with multiple training processes
      is going to be used (i.e. after the model is wrapped with DistributedDataParallel).
      Any special preparations for the algorithm to properly support distributed training
      should be made inside this function.


   .. py:method:: compression_stage()

      Returns the compression stage. Should be used on saving best checkpoints
      to distinguish between uncompressed, partially compressed, and fully
      compressed models.

      :return: The compression stage of the target model.


   .. py:method:: init_precision(precision_init_type, precision_init_params, precision_constraints)

      Precision initialization happens based on an measure of layer sensitivity to perturbations. The measure is
      calculated by average Hessian trace estimation for each layer using Hutchinson algorithm.


   .. py:method:: init_range(range_init_params = None)

      Tracks input statistics for quantizers in the model and sets ranges of the quantizers to correspond to
      minimum and maximum input tensor levels observed.
      :param range_init_params: specifies parameters for this range initialization call; if None, the parameters
      that were used during compressed model creation will be used.


   .. py:method:: statistics(quickly_collected_only=False)

      Returns a `Statistics` class instance that contains compression algorithm statistics.

      :param quickly_collected_only: Enables collection of the statistics that
          don't take too much time to compute. Can be helpful for the case when
          need to keep track of statistics on each training batch/step/iteration.
      :return: A `Statistics` class instance that contains compression algorithm statistics.


   .. py:method:: strip_model(model, do_copy = False)

      Strips auxiliary layers that were used for the model compression, as it's
      only needed for training. The method is used before exporting the model
      in the target format.

      :param model: The compressed model.
      :param do_copy: Modify copy of the model, defaults to False.
      :return: The stripped model.



