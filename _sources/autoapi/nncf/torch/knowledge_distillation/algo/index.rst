:py:mod:`nncf.torch.knowledge_distillation.algo`
================================================

.. py:module:: nncf.torch.knowledge_distillation.algo

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

   nncf.torch.knowledge_distillation.algo.KnowledgeDistillationController




.. py:class:: KnowledgeDistillationController(target_model: nncf.torch.nncf_network.NNCFNetwork, original_model: torch.nn.Module, kd_type: str, scale: float, temperature: float)

   Bases: :py:obj:`nncf.torch.compression_method_api.PTCompressionAlgorithmController`

   Serves as a handle to the additional modules, parameters and hooks inserted
   into the original uncompressed model in order to enable algorithm-specific compression.
   Hosts entities that are to be used during the training process, such as compression scheduler and
   compression loss.

   .. py:method:: compression_stage() -> nncf.api.compression.CompressionStage

      Returns level of compression. Should be used on saving best checkpoints to distinguish between
      uncompressed, partially compressed and fully compressed models.


   .. py:method:: statistics(quickly_collected_only: bool = False) -> nncf.common.statistics.NNCFStatistics

      Returns a `Statistics` class instance that contains compression algorithm statistics.

      :param quickly_collected_only: Enables collection of the statistics that
          don't take too much time to compute. Can be helpful for the case when
          need to keep track of statistics on each training batch/step/iteration.
      :return: A `Statistics` class instance that contains compression algorithm statistics.


   .. py:method:: distributed()

      Should be called when distributed training with multiple training processes
      is going to be used (i.e. after the model is wrapped with DistributedDataParallel).
      Any special preparations for the algorithm to properly support distributed training
      should be made inside this function.



