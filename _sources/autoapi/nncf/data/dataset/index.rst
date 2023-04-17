:py:mod:`nncf.data.dataset`
===========================

.. py:module:: nncf.data.dataset

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

   nncf.data.dataset.Dataset




.. py:class:: Dataset(data_source: Iterable[DataItem], transform_func: Optional[Callable[[DataItem], ModelInput]] = None)

   Bases: :py:obj:`Generic`\ [\ :py:obj:`DataItem`\ , :py:obj:`ModelInput`\ ]

   The `nncf.Dataset` class defines the interface by which compression algorithms
   retrieve data items from the passed data source object. These data items are used
   for different purposes, for example, model inference and model validation. It depends
   on the compression algorithm.

   If the data item has been returned from the data source per iteration and it cannot be
   used as input for model inference, the transformation function is used to extract the
   model's input from this data item. For example, in supervised learning, the data item
   usually contains both examples and labels. So transformation function should extract
   the examples from the data item.

   .. py:method:: get_data(indices: Optional[List[int]] = None) -> Iterable[DataItem]

      Returns the iterable object that contains selected data items from the data source as-is.

      :param indices: The zero-based indices of data items that should be selected from
          the data source. The indices should be sorted in ascending order. If indices are
          not passed all data items are selected from the data source.
      :return: The iterable object that contains selected data items from the data source as-is.


   .. py:method:: get_inference_data(indices: Optional[List[int]] = None) -> Iterable[ModelInput]

      Returns the iterable object that contains selected data items from the data source, for which
      the transformation function was applied. The item, which was returned per iteration from this
      iterable, can be used as the model's input for model inference.

      :param indices: The zero-based indices of data items that should be selected from
          the data source. The indices should be sorted in ascending order. If indices are
          not passed all data items are selected from the data source.
      :return: The iterable object that contains selected data items from the data source, for which
          the transformation function was applied.



