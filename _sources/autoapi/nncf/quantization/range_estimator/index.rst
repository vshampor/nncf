:py:mod:`nncf.quantization.range_estimator`
===========================================

.. py:module:: nncf.quantization.range_estimator



Classes
~~~~~~~

.. autoapisummary::

   nncf.quantization.range_estimator.StatisticsType
   nncf.quantization.range_estimator.AggregatorType
   nncf.quantization.range_estimator.StatisticsCollectorParameters
   nncf.quantization.range_estimator.RangeEstimatorParameters




.. py:class:: StatisticsType

   Bases: :py:obj:`enum.Enum`

   Enumeration of different types of statistics that are used to collect per sample
   statistics for activations and weights of the model.

   :param MAX: The maximum value in a tensor.
   :param MIN: The minimum value in a tensor.
   :param ABS_MAX: The maximum absolute value in a tensor.
   :param QUANTILE: A specific quantile value in a tensor.
   :param ABS_QUANTILE: A specific quantile value in the absolute tensor.
   :param MEAN: The mean value of a tensor.


.. py:class:: AggregatorType

   Bases: :py:obj:`enum.Enum`

   Enumeration of different types of aggregators that are used to aggregate per sample
   statistics for activations and weights of the model.

   :param MEAN: The mean value of a set of tensors.
   :param MAX: The maximum value of a set of tensors.
   :param MIN: The minimum value of a set of tensors.
   :param MEDIAN: The median value of a set of tensors.
   :param MEAN_NO_OUTLIERS: The mean value of a set of tensors with outliers removed.
   :param MEDIAN_NO_OUTLIERS: The median value of a set of tensors with outliers removed.


.. py:class:: StatisticsCollectorParameters

   Contains parameters for collecting statistics for activations and weights of the model.

   :param statistics_type: The type of per sample statistics to collect.
   :type statistics_type: Optional[nncf.quantization.range_estimator.StatisticsType]
   :param aggregator_type: The type of aggregator of per sample statistics.
   :type aggregator_type: Optional[nncf.quantization.range_estimator.AggregatorType]
   :param clipping_value: The value to use for clipping the input tensors before
       collecting statistics.
   :type clipping_value: Optional[float]
   :param quantile_outlier_prob: The outlier probability for quantile statistics.
   :type quantile_outlier_prob: float


.. py:class:: RangeEstimatorParameters

   Contains parameters for estimating the range of activations and weights of the model.

   :param min: The parameters for estimating the lower bound of the range.
   :type min: nncf.quantization.range_estimator.StatisticsCollectorParameters
   :param max: The Parameters for estimating the upper bound of the range.
   :type max: nncf.quantization.range_estimator.StatisticsCollectorParameters


