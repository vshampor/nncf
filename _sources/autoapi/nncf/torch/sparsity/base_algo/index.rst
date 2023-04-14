:py:mod:`nncf.torch.sparsity.base_algo`
=======================================

.. py:module:: nncf.torch.sparsity.base_algo

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

   nncf.torch.sparsity.base_algo.BaseSparsityAlgoController




.. py:class:: BaseSparsityAlgoController(target_model: nncf.torch.nncf_network.NNCFNetwork, sparsified_module_info: List[SparseModuleInfo])

   Bases: :py:obj:`nncf.torch.compression_method_api.PTCompressionAlgorithmController`, :py:obj:`nncf.common.sparsity.controller.SparsityController`

   Serves as a handle to the additional modules, parameters and hooks inserted
   into the original uncompressed model in order to enable algorithm-specific compression.
   Hosts entities that are to be used during the training process, such as compression scheduler and
   compression loss.

   .. py:method:: disable_scheduler()

      Disables current compression scheduler during training by changing
      it to a dummy one that does not change the compression rate.


   .. py:method:: compression_stage() -> nncf.api.compression.CompressionStage

      Returns the compression stage. Should be used on saving best checkpoints
      to distinguish between uncompressed, partially compressed, and fully
      compressed models.

      :return: The compression stage of the target model.


   .. py:method:: strip_model(model: nncf.torch.nncf_network.NNCFNetwork, do_copy: bool = False) -> nncf.torch.nncf_network.NNCFNetwork

      Strips auxiliary layers that were used for the model compression, as it's
      only needed for training. The method is used before exporting the model
      in the target format.

      :param model: The compressed model.
      :param do_copy: Modify copy of the model, defaults to False.
      :return: The stripped model.



