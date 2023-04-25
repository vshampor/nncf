:py:mod:`nncf.tensorflow`
=========================

.. py:module:: nncf.tensorflow

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

   pruning/index.rst
   quantization/index.rst
   sparsity/index.rst



Classes
~~~~~~~

.. autoapisummary::

   nncf.tensorflow.AdaptiveCompressionTrainingLoop
   nncf.tensorflow.EarlyExitCompressionTrainingLoop



Functions
~~~~~~~~~

.. autoapisummary::

   nncf.tensorflow.create_compressed_model
   nncf.tensorflow.create_compression_callbacks
   nncf.tensorflow.register_default_init_args



.. py:function:: create_compressed_model(model, config, compression_state = None)

   The main function used to produce a model ready for compression fine-tuning
   from an original TensorFlow Keras model and a configuration object.

   :param model: The original model. Should have its parameters already loaded
       from a checkpoint or another source.
   :param config: A configuration object used to determine the exact compression
       modifications to be applied to the model.
   :type config: nncf.NNCFConfig
   :param compression_state: compression state to unambiguously restore the compressed model.
       Includes builder and controller states. If it is specified, trainable parameter initialization will be skipped
       during building.
   :return: A tuple (compression_ctrl, compressed_model) where
       - compression_ctrl: The controller of the compression algorithm.
       - compressed_model: The model with additional modifications
           necessary to enable algorithm-specific compression during fine-tuning.


.. py:function:: create_compression_callbacks(compression_ctrl, log_tensorboard=True, log_text=True, log_dir=None)


.. py:function:: register_default_init_args(nncf_config, data_loader, batch_size, device = None)

   Register extra structures in the NNCFConfig. Initialization of some
   compression algorithms requires certain extra structures.

   :param nncf_config: An instance of the NNCFConfig class without extra structures.
   :param data_loader: Dataset used for initialization.
   :param batch_size: Batch size used for initialization.
   :param device: Device to perform initialization. If `device` is `None` then the device
       of the model parameters will be used.
   :return: An instance of the NNCFConfig class with extra structures.


.. py:class:: AdaptiveCompressionTrainingLoop(nncf_config, compression_controller, uncompressed_model_accuracy, lr_updates_needed=True, verbose=True, minimal_compression_rate=0.0, maximal_compression_rate=0.95, dump_checkpoints=True)

   Bases: :py:obj:`BaseEarlyExitCompressionTrainingLoop`

   Adaptive compression training loop allows an accuracy-aware training process whereby
   the compression rate is automatically varied during training to reach the maximal
   possible compression rate with a positive accuracy budget
   (the maximal allowed accuracy degradation criterion is satisfied).

   .. py:method:: run(model, train_epoch_fn, validate_fn, configure_optimizers_fn=None, dump_checkpoint_fn=None, load_checkpoint_fn=None, early_stopping_fn=None, tensorboard_writer=None, log_dir=None, update_learning_rate_fn=None)

      Implements the custom logic to run a training loop for model fine-tuning
      by using the provided `train_epoch_fn`, `validate_fn` and `configure_optimizers_fn` methods.
      The passed methods are registered in the `TrainingRunner` instance and the training logic
      is implemented by calling the corresponding `TrainingRunner` methods

      :param model: The model instance before fine-tuning
      :param train_epoch_fn: a method to fine-tune the model for a single epoch
      (to be called inside the `train_epoch` of the TrainingRunner)
      :param validate_fn: a method to evaluate the model on the validation dataset
      (to be called inside the `train_epoch` of the TrainingRunner)
      :param configure_optimizers_fn: a method to instantiate an optimizer and a learning
      rate scheduler (to be called inside the `configure_optimizers` of the TrainingRunner)
      :param dump_checkpoint_fn: a method to dump a checkpoint
      :param load_checkpoint_fn: a method to load a checkpoint
      :param early_stopping_fn: a method to check for an early stopping condition
      :return: The fine-tuned model



.. py:class:: EarlyExitCompressionTrainingLoop(nncf_config, compression_controller, uncompressed_model_accuracy, lr_updates_needed = True, verbose = True, dump_checkpoints = True)

   Bases: :py:obj:`BaseEarlyExitCompressionTrainingLoop`

   Adaptive compression training loop allows an accuracy-aware training process
   to reach the maximal accuracy drop
   (the maximal allowed accuracy degradation criterion is satisfied).


