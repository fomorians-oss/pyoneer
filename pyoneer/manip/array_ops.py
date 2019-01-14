from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def pad_or_truncate(tensor, maxsize, axis=-1, constant_values=0.0):
    """
    Pad or truncate tensor according to a max size.

    Args:
        tensor: Tensor to pad or truncate.
        maxsize: maximum size for tensor. If tensor.shape[axis] is greater than
            maxsize, the values will be trucated, else they will be padded with
            constant_values.
        axis: where to pad or truncate tensor.
        constant_values: value to pad tensor.shape[axis] to `maxsize`.

    Returns:
        padded or truncated Tensor of `shape[axis] = maxsize`
    """
    ndims = tensor.shape.ndims
    size = tensor.shape[axis]
    value_padding = [[0, 0]] * ndims
    value_padding[axis] = [0, maxsize - size]

    index_padding = [slice(None)] * ndims
    index_padding[axis] = slice(0, maxsize)
    index_padding = tuple(index_padding)

    def truncate():
        return tensor[index_padding]

    def pad():
        return tf.pad(
            tensor,
            value_padding,
            mode='CONSTANT',
            constant_values=constant_values)

    return tf.cond(size > maxsize, truncate, pad)


def shift(x, axis=1, shift=1, constant_values=None):
    """
    Shift the dimension according to `axis` of `x` right by `shift`.

    This is similar to `tf.manip.roll`, expect it fills the rolled
    Tensors with `constant_values`.

    Args:
        x: Tensor to be shifted.
        axis: dimension to shift.
        shift: number of shifts to compute.
        constant_values: value to pad where shifts took place.

    Returns:
        A shifted Tensor.
    """
    x = tf.convert_to_tensor(x)
    direction = abs(shift)
    is_right = direction == shift
    ndims = x.shape.ndims

    index_padding = [slice(None)] * ndims
    index_padding[axis] = slice(0, -1) if is_right else slice(1, None)
    index_padding = tuple(index_padding)

    if constant_values is None:
        value_padding = [[0, 0]] * ndims
        value_padding[axis] = [direction, 0] if is_right else [0, direction]
        return tf.pad(x, value_padding)[index_padding]

    padded = [constant_values, x] if is_right else [x, constant_values]
    return tf.concat(padded, axis=axis)[index_padding]
