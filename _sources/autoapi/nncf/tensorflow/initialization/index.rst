:py:mod:`nncf.tensorflow.initialization`
========================================

.. py:module:: nncf.tensorflow.initialization

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

   nncf.tensorflow.initialization.register_default_init_args



.. py:function:: register_default_init_args(nncf_config: nncf.config.NNCFConfig, data_loader: tensorflow.data.Dataset, batch_size: int, device: str = None) -> nncf.config.NNCFConfig

   Register extra structures in the NNCFConfig. Initialization of some
   compression algorithms requires certain extra structures.

   :param nncf_config: An instance of the NNCFConfig class without extra structures.
   :param data_loader: Dataset used for initialization.
   :param batch_size: Batch size used for initialization.
   :param device: Device to perform initialization. If `device` is `None` then the device
       of the model parameters will be used.
   :return: An instance of the NNCFConfig class with extra structures.


