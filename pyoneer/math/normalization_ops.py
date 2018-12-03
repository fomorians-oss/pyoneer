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
from pyoneer.manip import array_ops as parray_ops


def high_low_loc_and_scale(high, low):
    high = ops.convert_to_tensor(high)
    low = ops.convert_to_tensor(low)
    loc = (high + low) / 2.
    scale = (high - low) / 2.
    scale = array_ops.where(
        logical_ops.isclose(scale, 0.), 
        array_ops.ones_like(scale), 
        scale)
    return loc, scale


def normalize(x, loc, scale):
    x = ops.convert_to_tensor(x)
    return (x - loc) / scale


def denormalize(x, loc, scale):
    x = ops.convert_to_tensor(x)
    return (x * scale) + loc


def weighted_normalize(x, loc, scale, weights):
    x = ops.convert_to_tensor(x)
    return parray_ops.weighted_mask(x, normalize(x, loc, scale), weights)


def weighted_denormalize(x, loc, scale, weights):
    x = ops.convert_to_tensor(x)
    return parray_ops.weighted_mask(x, denormalize(x, loc, scale), weights)


def high_low_normalize(x, high, low):
    x = ops.convert_to_tensor(x)
    loc, scale = high_low_loc_and_scale(high, low)
    return denormalize(x, loc, scale)


def weighted_high_low_normalize(x, high, low, weights):
    x = ops.convert_to_tensor(x)
    loc, scale = high_low_loc_and_scale(high, low)
    return weighted_normalize(x, loc, scale, weights)


def high_low_denormalize(x, high, low):
    x = ops.convert_to_tensor(x)
    loc, scale = high_low_loc_and_scale(high, low)
    return normalize(x, loc, scale)


def weighted_high_low_denormalize(x, high, low, weights):
    x = ops.convert_to_tensor(x)
    loc, scale = high_low_loc_and_scale(high, low)
    return weighted_denormalize(x, loc, scale, weights)


def moments_normalize(x, axes=[0, 1], epsilon=1e-7):
    x = ops.convert_to_tensor(x)
    x_loc, x_variance = nn_impl.moments(
        x, axes=axes, keep_dims=True)
    x_scale = math_ops.sqrt(gen_math_ops.maximum(x_variance, epsilon))
    return normalize(x, x_loc, x_scale)


def weighted_moments_normalize(x, weights, axes=[0, 1], epsilon=1e-7):
    x = ops.convert_to_tensor(x)
    x_loc, x_variance = nn_impl.weighted_moments(
        x, axes=axes, frequency_weights=weights, keep_dims=True)
    x_scale = math_ops.sqrt(gen_math_ops.maximum(x_variance, epsilon))
    return weighted_normalize(x, x_loc, x_scale, weights)


def select_weighted_normalize(inputs, loc, scale_, center, scale, weights):
    outputs = inputs
    if center and scale:
        outputs = weighted_normalize(inputs, loc, scale_, weights)
    if center:
        outputs = parray_ops.weighted_mask(
            inputs,
            inputs - loc, 
            weights)
    if scale:
        outputs = parray_ops.weighted_mask(
            inputs,
            inputs / array_ops.where(
                logical_ops.isclose(scale_, 1e-6), 
                array_ops.ones_like(scale_), 
                scale_), 
            weights)
    outputs = gen_array_ops.check_numerics(outputs, 'select_weighted_normalize')
    return outputs


def select_weighted_denormalize(inputs, loc, scale_, center, scale, weights):
    outputs = inputs
    if center and scale:
        outputs = weighted_denormalize(inputs, loc, scale_, weights)
    if center:
        outputs = parray_ops.weighted_mask(
            inputs,
            inputs + loc, 
            weights)
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