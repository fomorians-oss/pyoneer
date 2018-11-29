from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import control_flow_ops


def pad_or_truncate(x, maxsize, axis=-1, pad_value=0):
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
    return array_ops.where(gen_math_ops.equal(gen_array_ops.broadcast_to(weights, array_ops.shape(x)), 1.), true_x, x)


def swap_time_major(x):
    dims = list(range(len(x.shape)))
    tmp = dims[0]
    dims[0] = dims[1]
    dims[1] = tmp
    return array_ops.transpose(x, dims)


def shift(x, axis=1, rotations=1, pad_value=None):
    """Shift the dimension according to `axis` of `x` right by `rotations`.
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
    """Flatten `inputs` into 1 or 2 dimensions depending on the `inputs` dimensions.
    """
    if len(inputs.shape) > 2:
        return gen_array_ops.reshape(inputs, [-1, array_ops.shape(inputs)[-1]])
    return gen_array_ops.reshape(inputs, [-1])
