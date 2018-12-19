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
    """
    Normalizes the inputs by subtracting a mean and dividing by a standard deviation. 

    `StatelessNormalizer` treats the `loc` and `scale` as constant parameters.
    """

    def __init__(self, loc, scale_, center=True, scale=True, dtype=None):
        """Creates a new StatelessNormalizer model.

        Args:
            loc: The mean to use for normalization.
            scale_: The standard deviation to use for normalization.
            center: Center using the mean with this flag.
            scale: Scale using the standard deviation with this flag.
        """
        super(StatelessNormalizer, self).__init__()
        self.loc = ops.convert_to_tensor(loc, dtype=dtype)
        self.scale = ops.convert_to_tensor(scale_, dtype=dtype)
        self.center = center
        self.scale_ = scale

    @property
    def shape(self):
        """Returns the event shape of the normalizer.
        
        Returns:
            the TensorShape of the `loc`.
        """
        return self.loc.shape

    def call(self, inputs, weights=1., **kwargs):
        """Normalizes an input.
        
        Args:
            inputs: a possibly un-normalized Tensor no less than 2-D.
            weights: mask to apply the operation.
            **kwargs: unused keyword arguments.

        Returns:
            A normalized Tensor.
        """
        del kwargs  # unused
        inputs = ops.convert_to_tensor(inputs)
        inputs = math_ops.cast(inputs, self.loc.dtype)
        return normalization_ops.select_weighted_normalize(
            inputs, self.loc, self.scale, self.center, self.scale_, weights)

    def inverse(self, inputs, weights=1., **kwargs):
        """Un-normalizes an input.
        
        Args:
            inputs: a possibly normalized Tensor no less than 2-D.
            weights: mask to apply the operation.
            **kwargs: unused keyword arguments.

        Returns:
            An un-normalized Tensor.
        """
        del kwargs  # unused
        inputs = ops.convert_to_tensor(inputs)
        inputs = math_ops.cast(inputs, self.loc.dtype)
        return normalization_ops.select_weighted_denormalize(
            inputs, self.loc, self.scale, self.center, self.scale_, weights)


class StatefulNormalizer(keras.Model):
    """
    Normalizes the inputs by subtracting a mean and dividing by a standard deviation. 

    `StatefulNormalizer` treats the `loc` and `scale` as variable parameters. 
    A subclass must implement the property getter `scale` and method `update_loc`.
    """

    def __init__(self, shape, center=True, scale=True, dtype=dtypes.float32):
        """Creates a new StatefulNormalizer.

        Args:
            shape: The shape of the mean/standard deviation to use for normalization.
            center: Center using the mean with this flag.
            scale: Scale using the standard deviation with this flag.
        """
        super(StatefulNormalizer, self).__init__()

        self.center = center
        self.scale_ = scale

        self.count = Variable(0., dtype=dtype, trainable=False)
        self.loc = Variable(
            array_ops.zeros(shape=shape, dtype=dtype), trainable=False)
        self.var_sum = Variable(
            array_ops.zeros(shape=shape, dtype=dtype), trainable=False)

    @property
    def shape(self):
        """Returns the event shape of the normalizer.
        
        Returns:
            the TensorShape of the `loc`.
        """
        return self.loc.shape

    @property
    def scale(self):
        raise NotImplementedError()

    def update_loc(self, loc_deltas, inputs, weights=1.):
        """Updates the `loc` variable.
        
        Args:
            loc_deltas: the change in `loc`.
            inputs: a possibly un-normalized Tensor no less than 2-D.
            weights: mask to apply the operation.

        Returns:
            A new `loc`.
        """
        raise NotImplementedError()

    def call(self, inputs, weights=1., training=False, **kwargs):
        """Normalizes an input.
        
        Args:
            inputs: a possibly un-normalized Tensor no less than 2-D.
            weights: mask to apply the operation.
            training: bool used for determining if the loc and scale should 
                be updated.
            **kwargs: unused keyword arguments.

        Returns:
            A normalized Tensor.
        """
        del kwargs
        inputs = ops.convert_to_tensor(inputs)
        inputs = math_ops.cast(inputs, self.loc.dtype)

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

    def inverse(self, inputs, weights=1., **kwargs):
        """Un-normalizes an input.
        
        Args:
            inputs: a possibly normalized Tensor no less than 2-D.
            weights: mask to apply the operation.
            **kwargs: unused keyword arguments.

        Returns:
            An un-normalized Tensor.
        """
        del kwargs
        inputs = ops.convert_to_tensor(inputs)
        inputs = math_ops.cast(inputs, self.loc.dtype)
        return normalization_ops.select_weighted_denormalize(
            inputs, self.loc, self.scale, self.center, self.scale_, weights)


class HighLowNormalizer(StatelessNormalizer):
    """Infers `loc` and `scale` from `high` and `low` parameters."""

    def __init__(self, high, low, center=True, scale=True, dtype=None):
        """Creates a new HighLowNormalizer.

        Args:
            high: the high parameter.
            low: the low parameter.
            center: Center using the mean with this flag.
            scale: Scale using the standard deviation with this flag.
        """
        loc, scale_ = normalization_ops.min_max_loc_and_scale(
            ops.convert_to_tensor(low, dtype), ops.convert_to_tensor(high, dtype))
        super(HighLowNormalizer, self).__init__(
            loc, scale_, center=center, scale=scale, dtype=dtype)
        

class SampleAverageNormalizer(StatefulNormalizer):
    """Compute the moving loc and scale according to the input sample count."""

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
    """Compute the moving loc and scale according to the supplied alpha scalar."""

    def __init__(self, shape, alpha, center=True, scale=True, dtype=dtypes.float32):
        """Creates a new WeightedAverageNormalizer.

        Args:
            shape: The shape of the mean/standard deviation to use for normalization.
            alpha: the scalar used to scale the mean updates instead of a sample average.
            center: Center using the mean with this flag.
            scale: Scale using the standard deviation with this flag.
        """
        super(WeightedAverageNormalizer, self).__init__(
            shape, center=center, scale=scale, dtype=dtype)
        self.alpha = alpha

    @property
    def scale(self):
        return math_ops.sqrt(gen_math_ops.maximum(self.var_sum * self.alpha, 0))

    def update_loc(self, loc_deltas, inputs, weights=1.):
        return parray_ops.weighted_mask(
            self.loc, 
            self.loc + (loc_deltas * self.alpha), 
            weights)
