:py:mod:`nncf.torch.pruning.filter_pruning.algo`
================================================

.. py:module:: nncf.torch.pruning.filter_pruning.algo

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

   nncf.torch.pruning.filter_pruning.algo.FilterPruningController




.. py:class:: FilterPruningController(target_model: nncf.torch.nncf_network.NNCFNetwork, prunable_types: List[str], pruned_module_groups: nncf.common.pruning.clusterization.Clusterization[nncf.torch.pruning.structs.PrunedModuleInfo], pruned_norms_operators: List[Tuple[nncf.common.graph.NNCFNode, nncf.torch.pruning.filter_pruning.layers.FilterPruningMask, torch.nn.Module]], config: nncf.NNCFConfig)

   Bases: :py:obj:`nncf.torch.pruning.base_algo.BasePruningAlgoController`

   Serves as a handle to the additional modules, parameters and hooks inserted
   into the original uncompressed model in order to enable algorithm-specific compression.
   Hosts entities that are to be used during the training process, such as compression scheduler and
   compression loss.

   .. py:method:: get_mask(minfo: nncf.torch.pruning.structs.PrunedModuleInfo) -> torch.Tensor
      :staticmethod:

      Returns pruning mask for minfo.module.


   .. py:method:: statistics(quickly_collected_only: bool = False) -> nncf.common.statistics.NNCFStatistics

      Returns a `Statistics` class instance that contains compression algorithm statistics.

      :param quickly_collected_only: Enables collection of the statistics that
          don't take too much time to compute. Can be helpful for the case when
          need to keep track of statistics on each training batch/step/iteration.
      :return: A `Statistics` class instance that contains compression algorithm statistics.


   .. py:method:: set_pruning_level(pruning_level: Union[float, Dict[int, float]], run_batchnorm_adaptation: bool = False) -> None

      Set the global or groupwise pruning level in the model.
      If pruning_level is a float, the correspoding global pruning level is set in the model,
      either in terms of the percentage of filters pruned or as the percentage of flops
      removed, the latter being true in case the "prune_flops" flag of the controller is
      set to True.
      If pruning_level is a dict, the keys should correspond to layer group id's and the
      values to groupwise pruning level to be set in the model.


   .. py:method:: prepare_for_export()

      Applies pruning masks to layer weights before exporting the model to ONNX.


   .. py:method:: compression_stage() -> nncf.api.compression.CompressionStage

      Returns the compression stage. Should be used on saving best checkpoints
      to distinguish between uncompressed, partially compressed, and fully
      compressed models.

      :return: The compression stage of the target model.


   .. py:method:: disable_scheduler()

      Disables current compression scheduler during training by changing
      it to a dummy one that does not change the compression rate.


   .. py:method:: strip_model(model: nncf.torch.nncf_network.NNCFNetwork, do_copy: bool = False) -> nncf.torch.nncf_network.NNCFNetwork

      Strips auxiliary layers that were used for the model compression, as it's
      only needed for training. The method is used before exporting the model
      in the target format.

      :param model: The compressed model.
      :param do_copy: Modify copy of the model, defaults to False.
      :return: The stripped model.



