from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn_impl

from pyoneer.math import logical_ops
from pyoneer.manip import array_ops as parray_ops


def high_low_mean_and_stddev(high, low):
    high = ops.convert_to_tensor(high)
    low = ops.convert_to_tensor(low)
    mean = (high + low) / 2.
    stddev = (high - low) / 2.
    stddev = array_ops.where(
        logical_ops.isclose(stddev, 0.), 
        array_ops.ones_like(stddev), 
        stddev)
    return mean, stddev


def normalize(x, mean, stddev):
    x = ops.convert_to_tensor(x)
    return (x - mean) / stddev


def denormalize(x, mean, stddev):
    x = ops.convert_to_tensor(x)
    return (x * stddev) + mean


def weighted_normalize(x, mean, stddev, weights):
    x = ops.convert_to_tensor(x)
    return parray_ops.weighted_mask(x, normalize(x, mean, stddev), weights)


def weighted_denormalize(x, mean, stddev, weights):
    x = ops.convert_to_tensor(x)
    return parray_ops.weighted_mask(x, denormalize(x, mean, stddev), weights)


def high_low_normalize(x, high, low):
    x = ops.convert_to_tensor(x)
    mean, stddev = high_low_mean_and_stddev(high, low)
    return denormalize(x, mean, stddev)


def weighted_high_low_normalize(x, high, low, weights):
    x = ops.convert_to_tensor(x)
    mean, stddev = high_low_mean_and_stddev(high, low)
    return weighted_normalize(x, mean, stddev, weights)


def high_low_denormalize(x, high, low):
    x = ops.convert_to_tensor(x)
    mean, stddev = high_low_mean_and_stddev(high, low)
    return normalize(x, mean, stddev)


def weighted_high_low_denormalize(x, high, low, weights):
    x = ops.convert_to_tensor(x)
    mean, stddev = high_low_mean_and_stddev(high, low)
    return weighted_denormalize(x, mean, stddev, weights)


def moments_normalize(x, axes=[0, 1], epsilon=1e-7):
    x = ops.convert_to_tensor(x)
    x_mean, x_variance = nn_impl.moments(
        x, axes=axes, keep_dims=True)
    x_stddev = math_ops.sqrt(gen_math_ops.maximum(x_variance, epsilon))
    return normalize(x, x_mean, x_stddev)


def weighted_moments_normalize(x, weights, axes=[0, 1], epsilon=1e-7):
    x = ops.convert_to_tensor(x)
    x_mean, x_variance = nn_impl.weighted_moments(
        x, axes=axes, frequency_weights=weights, keep_dims=True)
    x_stddev = math_ops.sqrt(gen_math_ops.maximum(x_variance, epsilon))
    return weighted_normalize(x, x_mean, x_stddev, weights)


def select_weighted_normalize(inputs, mean, std, center, scale, weights):
    outputs = inputs
    if center and scale:
        outputs = weighted_normalize(inputs, mean, std, weights)
    if center:
        outputs = parray_ops.weighted_mask(
            inputs,
            inputs - mean, 
            weights)
    if scale:
        outputs = parray_ops.weighted_mask(
            inputs,
            inputs / array_ops.where(
                logical_ops.isclose(std, 1e-6), 
                array_ops.ones_like(std), 
                std), 
            weights)
    outputs = gen_array_ops.check_numerics(outputs, 'select_weighted_normalize')
    return outputs


def select_weighted_denormalize(inputs, mean, std, center, scale, weights):
    outputs = inputs
    if center and scale:
        outputs = weighted_denormalize(inputs, mean, std, weights)
    if center:
        outputs = parray_ops.weighted_mask(
            inputs,
            inputs + mean, 
            weights)
    if scale:
        outputs = parray_ops.weighted_mask(
            inputs,
            inputs * array_ops.where(
                logical_ops.isclose(std, 1e-6), 
                array_ops.ones_like(std), 
                std), 
            weights)
    outputs = gen_array_ops.check_numerics(outputs, 'select_weighted_denormalize')
    return outputs