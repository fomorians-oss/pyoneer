from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import control_flow_ops


def pad_or_truncate(x, maxsize, axis=-1, pad_value=0):
    """Pad or truncate x according to a max size.
    
    Args:
        x: Tensor to pad or truncate.
        maxsize: maximum size for x. If x.shape[axis] is greater than maxsize, 
            the values will be trucated, else they will be padded with pad_value.
        axis: where to pad or truncate x.
        pad_value: value to pad x.shape[axis] to `maxsize`.

    Returns:
        padded or truncated Tensor of `shape[axis] = maxsize`
    """
    rank = len(x.shape)
    size = array_ops.shape(x)[axis]
    value_padding = [[0, 0]] * rank
    value_padding[axis] = [0, maxsize - size]

    pad = lambda: array_ops.pad(
        x, value_padding,
        mode="CONSTANT",
        constant_values=pad_value)
    index_padding = [slice(None)] * rank
    index_padding[axis] = slice(0, maxsize)
    index_padding = tuple(index_padding)

    truncate = lambda: x[index_padding]
    return control_flow_ops.cond(size > maxsize, truncate, pad)


def weighted_mask(x, true_x, weights):
    """Replace `x` with `true_x` according to `weights`.
    
    Args:
        x: values to possibly replace.
        true_x: values to put in place of x.
        weights: mask determining where to put true_x.
    
    Returns:
        `x` with `true_x` values filled in where `weights` is `1.`.
    """
    return array_ops.where(
        gen_math_ops.equal(
            gen_array_ops.broadcast_to(
                weights, array_ops.shape(x)), 1.), 
        true_x, x)


def swap_time_major(x):
    """Swap x.shape[0] <-> x.shape[1].
    
    Args:
        x: Tensor with no less than 2-D.

    Returns:
        x with swapped time and batch dimension.
    """
    return array_ops.transpose(
        x, [1, 0] + list(range(x.shape.ndims))[2:])


def expand_to(x, ndims):
    """Expand x to n-dimensions.
    
    Args:
        x: Tensor to expand to ndims.

    Returns:
        Expanded Tensor with `ndims`-D
    
    Raises:
        `ValueError` when `x.shape.ndims` is less than `ndims`.
    """
    x_ndims = x.shape.ndims
    diff = x_ndims - ndims
    if diff > 0:
        raise ValueError('`x.shape.ndims` must be at most `ndims`.')
    diff = abs(diff)
    if x_ndims == ndims:
        return x
    return gen_array_ops.reshape(x, x.shape.as_list() + [1] * diff)


def shift(x, axis=1, rotations=1, pad_value=None):
    """Shift the dimension according to `axis` of `x` right by `rotations`.

    This is similar to `tf.manip.roll`, expect it fills the rolled 
    Tensors with `pad_value`.

    Args:
        x: Tensor to be shifted.
        axis: dimension to shift.
        rotations: number of shifts to compute.
        pad_value: value to pad where shifts took place.
    
    Returns:
        A shifted Tensor.
    """
    x = ops.convert_to_tensor(x)
    direction = abs(rotations)
    is_right = direction == rotations
    rank = len(x.shape)
    index_padding = [slice(None)] * rank
    index_padding[axis] = slice(0, -1) if is_right else slice(1, None)
    index_padding = tuple(index_padding)

    if pad_value is None:
        value_padding = [[0, 0]] * rank
        value_padding[axis] = [direction, 0] if is_right else [0, direction]
        return array_ops.pad(x, value_padding)[index_padding]

    padded = [pad_value, x] if is_right else [x, pad_value]
    return array_ops.concat(padded, axis=axis)[index_padding]


def flatten(inputs):
    """Flatten into 1-D or 2-D depending on the input dimension.
    
    Args:
        inputs: Tensor to flatten dimensions.
    
    Returns:
        2-D Tensor if inputs.shape.ndims > 2, else 1-D Tensor.
    """
    if len(inputs.shape) > 2:
        return gen_array_ops.reshape(inputs, [-1, array_ops.shape(inputs)[-1]])
    return gen_array_ops.reshape(inputs, [-1])
