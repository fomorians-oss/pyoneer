from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops.resource_variable_ops import ResourceVariable as Variable

from pyoneer.math import logical_ops
from pyoneer.math import normalization_ops

from pyoneer.manip import array_ops as parray_ops


class StatelessNormalizer(keras.Model):

    def __init__(self, loc, scale_, center=True, scale=True):
        super(StatelessNormalizer, self).__init__()
        self.loc = loc
        self.scale = scale_
        self.center = center
        self.scale_ = scale

    @property
    def shape(self):
        return self.loc.shape

    def call(self, inputs, weights=1.):
        return normalization_ops.select_weighted_normalize(
            inputs, self.loc, self.scale, self.center, self.scale_, weights)

    def inverse(self, inputs, weights=1.):
        return normalization_ops.select_weighted_denormalize(
            inputs, self.loc, self.scale, self.center, self.scale_, weights)


class StatefulNormalizer(keras.Model):

    def __init__(self, shape, center=True, scale=True):
        super(StatefulNormalizer, self).__init__()

        self.center = center
        self.scale_ = scale

        self.count = Variable(0., dtype=dtypes.float32, trainable=False)
        self.loc = Variable(
            array_ops.zeros(shape=shape, dtype=dtypes.float32), trainable=False)
        self.var_sum = Variable(
            array_ops.zeros(shape=shape, dtype=dtypes.float32), trainable=False)

    @property
    def shape(self):
        return self.loc.shape

    @property
    def scale(self):
        raise NotImplementedError()

    def update_loc(self, loc_deltas, inputs, weights=1.):
        raise NotImplementedError()

    def call(self, inputs, weights=1., training=False):
        """Compute the normalization of the inputs.

        Args:
            inputs: inversely normalized input `Tensor`.
            training: bool if the loc and scale should be updated.

        Returns:
            normalized inputs.
        """
        inputs = ops.convert_to_tensor(inputs)

        if training:
            input_shape = inputs.get_shape().as_list()
            input_dims = len(input_shape)
            loc_shape = self.loc.get_shape().as_list()
            loc_dims = len(loc_shape)

            reduce_axes = list(range(input_dims - loc_dims + 1))
            reduce_shape = input_shape[:-loc_dims]

            assert input_dims > loc_dims
            count = math_ops.cast(math_ops.reduce_prod(reduce_shape), dtypes.float32)
            self.count.assign_add(count)

            loc_deltas = math_ops.reduce_sum(
                inputs - self.loc, axis=reduce_axes)

            new_loc = self.update_loc(loc_deltas, inputs, weights)

            var_deltas = (inputs - self.loc) * (inputs - new_loc)
            new_var_sum = parray_ops.weighted_mask(
                self.var_sum,
                self.var_sum + math_ops.reduce_sum(var_deltas, axis=reduce_axes),
                weights)
            self.loc.assign(new_loc)
            self.var_sum.assign(new_var_sum)

        return normalization_ops.select_weighted_normalize(
            inputs, self.loc, self.scale, self.center, self.scale_, weights)

    def inverse(self, inputs, weights=1.):
        """Compute the inverse normalization of the inputs.

        Args:
        inputs: normalized input `Tensor`.

        Returns:
        inversely normalized inputs.
        """
        inputs = ops.convert_to_tensor(inputs)
        return normalization_ops.select_weighted_denormalize(
            inputs, self.loc, self.scale, self.center, self.scale_, weights)


class HighLowNormalizer(StatelessNormalizer):

    def __init__(self, high, low, center=True, scale=True):
        loc, scale_ = normalization_ops.high_low_loc_and_scale(
            ops.convert_to_tensor(high), ops.convert_to_tensor(low))
        super(HighLowNormalizer, self).__init__(loc, scale_, center=center, scale=scale)
        

class SampleAverageNormalizer(StatefulNormalizer):

    @property
    def scale(self):
        return math_ops.sqrt(gen_math_ops.maximum(self.var_sum / self.count, 0))

    def update_loc(self, loc_deltas, inputs, weights=1.):
        return array_ops.where(
            self.count > 1.,
            parray_ops.weighted_mask(
                self.loc, 
                self.loc + (loc_deltas / self.count), 
                weights),
            inputs[tuple([0] * (len(inputs.shape) - 1))])


class WeightedAverageNormalizer(StatefulNormalizer):

    def __init__(self, shape, alpha, center=True, scale=True):
        super(WeightedAverageNormalizer, self).__init__(
            shape, center=center, scale=scale)
        self.alpha = alpha

    @property
    def scale(self):
        return math_ops.sqrt(gen_math_ops.maximum(self.var_sum * self.alpha, 0))

    def update_loc(self, loc_deltas, inputs, weights=1.):
        return parray_ops.weighted_mask(
            self.loc, 
            self.loc + (loc_deltas * self.alpha), 
            weights)
