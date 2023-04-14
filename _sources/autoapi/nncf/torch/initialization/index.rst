:orphan:

:py:mod:`nncf.torch.initialization`
===================================

.. py:module:: nncf.torch.initialization



Classes
~~~~~~~

.. autoapisummary::

   nncf.torch.initialization.PTInitializingDataLoader
   nncf.torch.initialization.DefaultInitializingDataLoader




.. py:class:: PTInitializingDataLoader(data_loader: torch.utils.data.DataLoader)

   Bases: :py:obj:`nncf.common.initialization.dataloader.NNCFDataLoader`

   This class wraps the torch.utils.data.DataLoader class,
   and defines methods to parse the general data loader output to
   separate the input to the compressed model and the ground truth target
   for the neural network. This is required for proper initialization of
   certain compression algorithms.

   .. py:method:: __iter__()

      Creates an iterator for the elements of a custom data source.
      The returned iterator implements the Python Iterator protocol.

      :return: An iterator for the elements of a custom data source.


   .. py:method:: get_inputs(dataloader_output: Any) -> Tuple[Tuple, Dict]
      :abstractmethod:

      Returns (args, kwargs) for the current model call to be made during the initialization process


   .. py:method:: get_target(dataloader_output: Any) -> Any
      :abstractmethod:

      Parses the generic data loader output and returns a structure to be used as
      ground truth in the loss criterion.

      :param dataloader_output - the (args, kwargs) tuple returned by the __next__ method.



.. py:class:: DefaultInitializingDataLoader(data_loader: torch.utils.data.DataLoader)

   Bases: :py:obj:`PTInitializingDataLoader`

   This class wraps the torch.utils.data.DataLoader class,
   and defines methods to parse the general data loader output to
   separate the input to the compressed model and the ground truth target
   for the neural network. This is required for proper initialization of
   certain compression algorithms.

   .. py:method:: get_inputs(dataloader_output: Any) -> Tuple[Tuple, Dict]

      Returns (args, kwargs) for the current model call to be made during the initialization process


   .. py:method:: get_target(dataloader_output: Any)

      Parses the generic data loader output and returns a structure to be used as
      ground truth in the loss criterion.

      :param dataloader_output - the (args, kwargs) tuple returned by the __next__ method.



