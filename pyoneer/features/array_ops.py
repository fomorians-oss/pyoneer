from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import control_flow_ops


def pad_or_truncate(x, maxsize, axis=-1, pad_value=0):
    """Pad or truncate the dimension according to `axis` of x by `maxsize`, with `pad_value`."""
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
    return array_ops.where(
        gen_math_ops.equal(
                gen_array_ops.broadcast_to(
                    weights, array_ops.shape(x)), 1.),
        true_x, x)