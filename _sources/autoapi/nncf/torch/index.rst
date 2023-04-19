:py:mod:`nncf.torch`
====================

.. py:module:: nncf.torch

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



Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   knowledge_distillation/index.rst
   pruning/index.rst
   quantization/index.rst
   sparsity/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   structures/index.rst




Functions
~~~~~~~~~

.. autoapisummary::

   nncf.torch.create_compressed_model
   nncf.torch.load_state
   nncf.torch.disable_tracing



.. py:function:: create_compressed_model(model: torch.nn.Module, config: nncf.config.NNCFConfig, compression_state: Optional[Dict[str, Any]] = None, dummy_forward_fn: Callable[[torch.nn.Module], Any] = None, wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None, wrap_outputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None, dump_graphs=True) -> Tuple[nncf.api.compression.CompressionAlgorithmController, nncf.torch.nncf_network.NNCFNetwork]

   The main function used to produce a model ready for compression fine-tuning from an original PyTorch
   model and a configuration object.
   dummy_forward_fn
   :param model: The original model. Should have its parameters already loaded from a checkpoint or another
   source.
   :param config: A configuration object used to determine the exact compression modifications to be applied
   to the model
   :param compression_state: representation of the entire compression state to unambiguously restore
   the compressed model. Includes builder and controller states.
   :param dummy_forward_fn: if supplied, will be used instead of a *forward* function call to build
   the internal graph representation via tracing. Specifying this is useful when the original training pipeline
   has special formats of data loader output or has additional *forward* arguments other than input tensors.
   Otherwise, the *forward* call of the model during graph tracing will be made with mock tensors according
   to the shape specified in the config object. The dummy_forward_fn code MUST contain calls to nncf.nncf_model_input
   functions made with each compressed model input tensor in the underlying model's args/kwargs tuple, and these
   calls should be exactly the same as in the wrap_inputs_fn function code (see below); if dummy_forward_fn is
   specified, then wrap_inputs_fn also must be specified.
   :param wrap_inputs_fn: if supplied, will be used on the module's input arguments during a regular, non-dummy
   forward call before passing the inputs to the underlying compressed model. This is required if the model's input
   tensors that are important for compression are not supplied as arguments to the model's forward call directly, but
   instead are located in a container (such as list), and the model receives the container as an argument.
   wrap_inputs_fn should take as input two arguments - the tuple of positional arguments to the underlying
   model's forward call, and a dict of keyword arguments to the same. The function should wrap each tensor among the
   supplied model's args and kwargs that is important for compression (e.g. quantization) with an nncf.nncf_model_input
   function, which is a no-operation function and marks the tensors as inputs to be traced by NNCF in the internal
   graph representation. Output is the tuple of (args, kwargs), where args and kwargs are the same as were supplied in
   input, but each tensor in the original input. Must be specified if dummy_forward_fn is specified.
   :param wrap_outputs_fn: same as `wrap_inputs_fn`, but applies to model outputs
   :param dump_graphs: Whether to dump the internal graph representation of the
   original and compressed models in the .dot format into the log directory.
   :return: A controller for the compression algorithm (or algorithms, in which case the controller
   is an instance of CompositeCompressionController) and the model ready for compression parameter training wrapped
   as an object of NNCFNetwork.


.. py:function:: load_state(model: torch.nn.Module, state_dict_to_load: dict, is_resume: bool = False, keys_to_ignore: List[str] = None) -> int

   Used to load a checkpoint containing a compressed model into an NNCFNetwork object, but can
   be used for any PyTorch module as well. Will do matching of state_dict_to_load parameters to
   the model's state_dict parameters while discarding irrelevant prefixes added during wrapping
   in NNCFNetwork or DataParallel/DistributedDataParallel objects, and load the matched parameters
   from the state_dict_to_load into the model's state dict.
   :param model: The target module for the state_dict_to_load to be loaded to.
   :param state_dict_to_load: A state dict containing the parameters to be loaded into the model.
   :param is_resume: Determines the behavior when the function cannot do a successful parameter match
   when loading. If True, the function will raise an exception if it cannot match the state_dict_to_load
   parameters to the model's parameters (i.e. if some parameters required by model are missing in
   state_dict_to_load, or if state_dict_to_load has parameters that could not be matched to model parameters,
   or if the shape of parameters is not matching). If False, the exception won't be raised.
   Usually is_resume is specified as False when loading uncompressed model's weights into the model with
   compression algorithms already applied, and as True when loading a compressed model's weights into the model
   with compression algorithms applied to evaluate the model.
   :param keys_to_ignore: A list of parameter names that should be skipped from matching process.
   :return: The number of state_dict_to_load entries successfully matched and loaded into model.


.. py:function:: disable_tracing(method)

   Patch a method so that it will be executed within no_nncf_trace context
   :param method: A method to patch.


