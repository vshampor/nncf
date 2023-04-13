:py:mod:`nncf.config.config`
============================

.. py:module:: nncf.config.config

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

   nncf.config.config.NNCFConfig




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



