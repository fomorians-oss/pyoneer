from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn_impl

from pyoneer.math import logical_ops
from pyoneer.math import math_ops as pmath_ops
from pyoneer.manip import array_ops as parray_ops



def min_max_loc_and_scale(minval, maxval):
    """Compute the loc and scale from high and low bounds.

    Args:
        high: scalar or Tensor.
        low: scalar or Tensor.
    
    Returns:
        tuple containing (loc, scale).
    """
    maxval = ops.convert_to_tensor(maxval)
    minval = ops.convert_to_tensor(minval)
    loc = (maxval + minval) / 2.
    scale = (maxval - minval) / 2.
    scale = array_ops.where(
        logical_ops.isclose(scale, 0.), 
        array_ops.ones_like(scale), 
        scale)
    return loc, scale


def normalize(x, loc, scale, weights=1.):
    """Normalizes an input.
    
    Args:
        x: a possibly un-normalized Tensor.
        loc: expected loc.
        scale: expected scale.
        weights: optional mask for computing this op

    Returns:
        A normalized Tensor.
    """

    x = ops.convert_to_tensor(x, loc.dtype)
    weights = ops.convert_to_tensor(weights, x.dtype)
    x_normed = pmath_ops.safe_divide((x - loc), scale)

    return parray_ops.weighted_mask(x, x_normed, weights)


def denormalize(x, loc, scale, weights=1.):
    """De-normalizes an input.
    
    Args:
        x: a possibly normalized Tensor.
        loc: expected loc.
        scale: expected scale.
        weights: optional mask for computing this op

    Returns:
        A de-normalized Tensor.
    """
    x = ops.convert_to_tensor(x, loc.dtype)
    weights = ops.convert_to_tensor(weights, x.dtype)
    x_denormed = (x * scale) + loc

    return parray_ops.weighted_mask(x, x_denormed, weights)

def normalize_by_moments(x, weights=1., axes=[0, 1], epsilon=1e-7):
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
    return normalize(x, x_loc, x_scale, weights)


def select_weighted_normalize(inputs, loc, scale_, center, scale, weights=1.):
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
    weights = ops.convert_to_tensor(weights, inputs.dtype)
    outputs = inputs
    if center and scale:
        outputs = normalize(inputs, loc, scale_, weights)
        outputs = gen_array_ops.check_numerics(outputs, 'select_normalize')
        return outputs
    if center:
        outputs = parray_ops.weighted_mask(
            inputs,
            inputs - loc, 
            weights)
        outputs = gen_array_ops.check_numerics(outputs, 'select_normalize')
        return outputs
    if scale:
        outputs = parray_ops.weighted_mask(
            inputs,
            pmath_ops.safe_divide(inputs, scale_), 
            weights)
        outputs = gen_array_ops.check_numerics(outputs, 'select_normalize')
        return outputs
    outputs = gen_array_ops.check_numerics(outputs, 'select_normalize')
    return outputs


def select_weighted_denormalize(inputs, loc, scale_, center, scale, weights=1.):
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
    weights = ops.convert_to_tensor(weights, inputs.dtype)
    outputs = inputs
    if center and scale:
        outputs = denormalize(inputs, loc, scale_, weights)
        outputs = gen_array_ops.check_numerics(outputs, 'select_weighted_denormalize')
        return outputs
    if center:
        outputs = parray_ops.weighted_mask(
            inputs,
            inputs + loc, 
            weights)
        outputs = gen_array_ops.check_numerics(outputs, 'select_weighted_denormalize')
        return outputs
    if scale:
        outputs = parray_ops.weighted_mask(
            inputs,
            inputs * array_ops.where(
                logical_ops.isclose(scale_, 1e-6), 
                array_ops.ones_like(scale_), 
                scale_), 
            weights)
        outputs = gen_array_ops.check_numerics(outputs, 'select_weighted_denormalize')
        return outputs
    return outputs