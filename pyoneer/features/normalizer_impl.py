from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops.resource_variable_ops import ResourceVariable as Variable

from pyoneer.math import logical_ops
from pyoneer.features import normalization_ops
from pyoneer.features import array_ops as parray_ops


class StatelessNormalizer(keras.Model):

    def __init__(self, mean, std, center=True, scale=True):
        super(StatelessNormalizer, self).__init__()
        self.mean = mean
        self.std = std
        self.center = center
        self.scale = scale

    def call(self, inputs, weights=1.):
        return normalization_ops.select_weighted_normalize(
            inputs, self.mean, self.std, self.center, self.scale, weights)

    def inverse(self, inputs, weights=1.):
        return normalization_ops.select_weighted_denormalize(
            inputs, self.mean, self.std, self.center, self.scale, weights)


class StatefulNormalizer(keras.Model):

    def __init__(self, shape, center=True, scale=True):
        super(StatefulNormalizer, self).__init__()

        self.center = center
        self.scale = scale

        self.count = Variable(0., dtype=dtypes.float32, trainable=False)
        self.mean = Variable(
            array_ops.zeros(shape=shape, dtype=dtypes.float32), trainable=False)
        self.var_sum = Variable(
            array_ops.zeros(shape=shape, dtype=dtypes.float32), trainable=False)

    @property
    def std(self):
        raise NotImplementedError()

    def update_mean(self, mean_deltas, inputs, weights=1.):
        raise NotImplementedError()

    def call(self, inputs, weights=1., training=False):
        """Compute the normalization of the inputs.

        Args:
            inputs: inversely normalized input `Tensor`.
            training: bool if the mean and stddev should be updated.

        Returns:
            normalized inputs.
        """
        inputs = ops.convert_to_tensor(inputs)

        if training:
            input_shape = inputs.get_shape().as_list()
            input_dims = len(input_shape)
            mean_shape = self.mean.get_shape().as_list()
            mean_dims = len(mean_shape)

            reduce_axes = list(range(input_dims - mean_dims + 1))
            reduce_shape = input_shape[:-mean_dims]

            assert input_dims > mean_dims
            count = math_ops.cast(math_ops.reduce_prod(reduce_shape), dtypes.float32)
            self.count.assign_add(count)

            mean_deltas = math_ops.reduce_sum(
                inputs - self.mean, axis=reduce_axes)

            new_mean = self.update_mean(mean_deltas, inputs, weights)

            var_deltas = (inputs - self.mean) * (inputs - new_mean)
            new_var_sum = parray_ops.weighted_mask(
                self.var_sum,
                self.var_sum + math_ops.reduce_sum(var_deltas, axis=reduce_axes),
                weights)
            self.mean.assign(new_mean)
            self.var_sum.assign(new_var_sum)

        return normalization_ops.select_weighted_normalize(
            inputs, self.mean, self.std, self.center, self.scale, weights)

    def inverse(self, inputs, weights=1.):
        """Compute the inverse normalization of the inputs.

        Args:
        inputs: normalized input `Tensor`.

        Returns:
        inversely normalized inputs.
        """
        inputs = ops.convert_to_tensor(inputs)
        return normalization_ops.select_weighted_denormalize(
            inputs, self.mean, self.std, self.center, self.scale, weights)


class HighLowNormalizer(StatelessNormalizer):

    def __init__(self, high, low, center=True, scale=True):
        mean, std = normalization_ops.high_low_mean_and_stddev(
            ops.convert_to_tensor(high), ops.convert_to_tensor(low))
        super(HighLowNormalizer, self).__init__(mean, std, center=center, scale=scale)
        

class SampleAverageNormalizer(StatefulNormalizer):

    @property
    def std(self):
        return math_ops.sqrt(gen_math_ops.maximum(self.var_sum / self.count, 0))

    def update_mean(self, mean_deltas, inputs, weights=1.):
        return array_ops.where(
            self.count > 1.,
            parray_ops.weighted_mask(
                self.mean, 
                self.mean + (mean_deltas / self.count), 
                weights),
            inputs[tuple([0] * (len(inputs.shape) - 1))])


class WeightedAverageNormalizer(StatefulNormalizer):

    def __init__(self, shape, alpha, center=True, scale=True):
        super(WeightedAverageNormalizer, self).__init__(
            shape, center=center, scale=scale)
        self.alpha = alpha

    @property
    def std(self):
        return math_ops.sqrt(gen_math_ops.maximum(self.var_sum * self.alpha, 0))

    def update_mean(self, mean_deltas, inputs, weights=1.):
        return parray_ops.weighted_mask(
            self.mean, 
            self.mean + (mean_deltas * self.alpha), 
            weights)
