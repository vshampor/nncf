:py:mod:`nncf.torch.quantization.quantize`
==========================================

.. py:module:: nncf.torch.quantization.quantize

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

   nncf.torch.quantization.quantize.CalibrarionDataLoader




.. py:class:: CalibrarionDataLoader(data_loader: nncf.data.Dataset)

   Bases: :py:obj:`nncf.torch.initialization.PTInitializingDataLoader`

   This class wraps the nncf.Dataset.

   This is required for proper initialization of certain compression algorithms.

   .. py:method:: __iter__()

      Creates an iterator for the elements of a custom data source.
      The returned iterator implements the Python Iterator protocol.

      :return: An iterator for the elements of a custom data source.


   .. py:method:: get_inputs(dataloader_output: Any) -> Tuple[Tuple, Dict]

      Returns (args, kwargs) for the current model call to be made during the initialization process



