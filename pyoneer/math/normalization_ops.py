from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn_impl

from pyoneer.math import logical_ops
from pyoneer.math import math_ops as pmath_ops
from pyoneer.manip import array_ops as parray_ops


def high_low_loc_and_scale(high, low):
    """Compute the loc and scale from high and low bounds.

    Args:
        high: scalar or Tensor.
        low: scalar or Tensor.

    Returns:
        tuple containing (loc, scale).
    """
    high = ops.convert_to_tensor(high)
    low = ops.convert_to_tensor(low)
    loc = (high + low) / 2.
    scale = (high - low) / 2.
    scale = array_ops.where(
        logical_ops.isclose(scale, 0.), array_ops.ones_like(scale), scale)
    return loc, scale


def normalize(x, loc, scale):
    """Normalizes an input.

    Args:
        x: a possibly un-normalized Tensor.
        loc: expected loc.
        scale: expected scale.

    Returns:
        A normalized Tensor.
    """
    x = ops.convert_to_tensor(x, loc.dtype)
    return pmath_ops.safe_divide((x - loc), scale)


def denormalize(x, loc, scale):
    """De-normalizes an input.

    Args:
        x: a possibly normalized Tensor.
        loc: expected loc.
        scale: expected scale.

    Returns:
        A de-normalized Tensor.
    """
    x = ops.convert_to_tensor(x, loc.dtype)
    return (x * scale) + loc


def rescale(x, oldmin, oldmax, newmin, newmax):
    """
    Rescale from [oldmin..oldmax] to [newmin..newmax].
    """
    x = (x - oldmin) / (oldmax - oldmin)
    x = (x * (newmax - newmin)) + newmin
    return x


def weighted_normalize(x, loc, scale, weights):
    """Normalizes an input.

    Args:
        x: a possibly un-normalized Tensor.
        loc: expected loc.
        scale: expected scale.
        weights: mask for computing this op.

    Returns:
        A normalized Tensor.
    """
    x = ops.convert_to_tensor(x, loc.dtype)
    return parray_ops.weighted_mask(x, normalize(x, loc, scale), weights)


def weighted_denormalize(x, loc, scale, weights):
    """De-normalizes an input.

    Args:
        x: a possibly normalized Tensor.
        loc: expected loc.
        scale: expected scale.
        weights: mask for computing this op.

    Returns:
        A de-normalized Tensor.
    """
    x = ops.convert_to_tensor(x, loc.dtype)
    return parray_ops.weighted_mask(x, denormalize(x, loc, scale), weights)


def high_low_normalize(x, high, low):
    """Normalizes an input according to high and low.

    Args:
        x: a possibly un-normalized Tensor.
        high: high boundary.
        low: low boundary.

    Returns:
        A normalized Tensor.
    """
    x = ops.convert_to_tensor(x, high.dtype)
    loc, scale = high_low_loc_and_scale(high, low)
    return normalize(x, loc, scale)


def weighted_high_low_normalize(x, high, low, weights):
    """Normalizes an input according to high and low.

    Args:
        x: a possibly un-normalized Tensor.
        high: high boundary.
        low: low boundary.
        weights: mask for computing this op.

    Returns:
        A normalized Tensor.
    """
    x = ops.convert_to_tensor(x, high.dtype)
    loc, scale = high_low_loc_and_scale(high, low)
    return weighted_normalize(x, loc, scale, weights)


def high_low_denormalize(x, high, low):
    """De-normalizes an input according to high and low.

    Args:
        x: a possibly normalized Tensor.
        high: high boundary.
        low: low boundary.

    Returns:
        A de-normalized Tensor.
    """
    x = ops.convert_to_tensor(x, high.dtype)
    loc, scale = high_low_loc_and_scale(high, low)
    return denormalize(x, loc, scale)


def weighted_high_low_denormalize(x, high, low, weights):
    """De-normalizes an input according to high and low.

    Args:
        x: a possibly normalized Tensor.
        high: high boundary.
        low: low boundary.
        weights: mask for computing this op.

    Returns:
        A de-normalized Tensor.
    """
    x = ops.convert_to_tensor(x, high.dtype)
    loc, scale = high_low_loc_and_scale(high, low)
    return weighted_denormalize(x, loc, scale, weights)


def moments_normalize(x, axes=[0, 1], epsilon=1e-7):
    """Normalizes an input according to the input moments.

    Args:
        x: a possibly un-normalized Tensor.
        axes: axes to compute moments.
        epsilon: precision epsilon.

    Returns:
        A normalized Tensor.
    """
    x = ops.convert_to_tensor(x)
    x_loc, x_variance = nn_impl.moments(x, axes=axes, keep_dims=True)
    x_scale = math_ops.sqrt(gen_math_ops.maximum(x_variance, epsilon))
    return normalize(x, x_loc, x_scale)


def weighted_moments_normalize(x, weights, axes=[0, 1], epsilon=1e-7):
    """Normalizes an input according to the input moments.

    Args:
        x: a possibly un-normalized Tensor.
        axes: axes to compute moments.
        epsilon: precision epsilon.
        weights: mask for computing the moments in this op.

    Returns:
        A normalized Tensor.
    """
    x = ops.convert_to_tensor(x)
    x_loc, x_variance = nn_impl.weighted_moments(
        x, axes=axes, frequency_weights=weights, keep_dims=True)
    x_scale = math_ops.sqrt(gen_math_ops.maximum(x_variance, epsilon))
    return weighted_normalize(x, x_loc, x_scale, weights)


def select_weighted_normalize(inputs, loc, scale_, center, scale, weights):
    """Normalizes an input according the center and scale.

    Args:
        inputs: a possibly de-normalized Tensor.
        loc: expected loc.
        scale_: expected scale.
        center: flag for inputs to be centered by loc.
        scale: flag for inputs to be scaled by scale_.
        weights: mask for computing this op.

    Returns:
        A normalized Tensor.
    """
    inputs = ops.convert_to_tensor(inputs, loc.dtype)
    outputs = inputs
    if center and scale:
        outputs = weighted_normalize(inputs, loc, scale_, weights)
        outputs = gen_array_ops.check_numerics(outputs,
                                               'select_weighted_normalize')
        return outputs
    if center:
        outputs = parray_ops.weighted_mask(inputs, inputs - loc, weights)
        outputs = gen_array_ops.check_numerics(outputs,
                                               'select_weighted_normalize')
        return outputs
    if scale:
        outputs = parray_ops.weighted_mask(
            inputs, pmath_ops.safe_divide(inputs, scale_), weights)
        outputs = gen_array_ops.check_numerics(outputs,
                                               'select_weighted_normalize')
        return outputs
    outputs = gen_array_ops.check_numerics(outputs,
                                           'select_weighted_normalize')
    return outputs


def select_weighted_denormalize(inputs, loc, scale_, center, scale, weights):
    """De-normalizes an input according the center and scale.

    Args:
        inputs: a possibly normalized Tensor.
        loc: expected loc.
        scale_: expected scale.
        center: flag for inputs to be centered by loc.
        scale: flag for inputs to be scaled by scale_.
        weights: mask for computing this op.

    Returns:
        A de-normalized Tensor.
    """
    inputs = ops.convert_to_tensor(inputs, loc.dtype)
    outputs = inputs
    if center and scale:
        outputs = weighted_denormalize(inputs, loc, scale_, weights)
        outputs = gen_array_ops.check_numerics(outputs,
                                               'select_weighted_denormalize')
        return outputs
    if center:
        outputs = parray_ops.weighted_mask(inputs, inputs + loc, weights)
        outputs = gen_array_ops.check_numerics(outputs,
                                               'select_weighted_denormalize')
        return outputs
    if scale:
        outputs = parray_ops.weighted_mask(
            inputs,
            inputs * array_ops.where(
                logical_ops.isclose(scale_, 1e-6), array_ops.ones_like(scale_),
                scale_), weights)
        outputs = gen_array_ops.check_numerics(outputs,
                                               'select_weighted_denormalize')
        return outputs
    return outputs