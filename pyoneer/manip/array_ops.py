from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def flatten(tensor):
    """
    Flatten a tensor by reshaping into a vector.

    Args:
        tensor: Tensor to flatten.

    Returns:
        Flattened tensor.
    """
    return tf.reshape(tensor, [-1])


def pad_or_truncate(tensor, sizes, mode="CONSTANT", constant_values=0.0):
    """
    Pad or truncate a tensor. This is useful for ensuring sequences have the
    same minimum shape when the sequence length is not known in advance.

    Examples:

    sentences = [['apple', 'is', 'red'], ['i', 'dont', 'like', 'apples']]
    sentences = tf.stack([
        pynr.manip.pad_or_truncate(sentences, sizes=[2, 3], '<PAD>')
        for s in sentences], axis=0)
    >>> [['apple', 'is', 'red'], ['i', 'dont', 'like']]

    sentences = [['apple', 'is', 'red'], ['i', 'dont', 'like', 'apples']]
    sentences = tf.stack([
        pynr.manip.pad_or_truncate(sentences, sizes=[2, 4], '<PAD>')
        for s in sentences], axis=0)
    >>> [['apple', 'is', 'red', '<PAD>'], ['i', 'dont', 'like', 'apples']]

    Args:
        tensor: Tensor to pad or truncate.
        sizes: Maximum sizes for tensor along each axis. If tensor.shape is
            greater than sizes along any axis, the axis will be trucated,
            otherwise they will be padded according to `mode` and
            `constant_values`.
        mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive).
        constant_values: In "CONSTANT" mode, the padding value to use.

    Returns:
        Padded or truncated tensor.
    """
    tensor = tf.convert_to_tensor(tensor)

    paddings = [
        [0, max(sizes[axis] - tensor.shape[axis], 0)]
        for axis in range(tensor.shape.ndims)
    ]

    slices = [slice(0, sizes[axis]) for axis in range(tensor.shape.ndims)]

    tensor = tf.pad(
        tensor=tensor, paddings=paddings, mode=mode, constant_values=constant_values
    )
    tensor = tensor[slices]

    return tensor


def shift(inputs, shift, axis, padding_values=0.0):
    """
    Shifts the elements of a tensor along an axis.

    This is similar to `tf.manip.roll`, except that it optionally fills the
    rolled Tensors with `padding_values`.

    Examples:

        Convert states to next states and fill last step with zeros:

            next_state = pynr.manip.shift(states, shift=-1, axis=1)

        Convert values to next values and use bootstrap values for last step:

            next_values = pynr.manip.shift(values, shift=-1, axis=1,
                padding_values=bootstrap_values[:, None])

        Convert actions and rewards to previous actions and rewards for RL^2:

            previous_actions = pynr.manip.shift(actions, shift=1, axis=1)
            previous_rewards = pynr.manip.shift(rewards, shift=1, axis=1)

    Args:
        inputs: A `Tensor` to shift.
        shift:  A `Tensor`. Must be one of the following types: int32, int64.
            Dimension must be 0-D or 1-D. shift[i] specifies the number of
            places by which elements are shifted positively (towards larger
            indices) along the dimension specified by axis[i]. Negative shifts
            will roll the elements in the opposite direction.
        axis: A `Tensor`. Must be one of the following types: int32, int64.
            Dimension must be 0-D or 1-D. axis[i] specifies the dimension that
            the shift shift[i] should occur. If the same axis is referenced
            more than once, the total shift for that axis will be the sum of
            all the shifts that belong to that axis.
        padding_values: A tensor to use for padding.

    Returns:
        A shifted Tensor.
    """
    inputs = tf.convert_to_tensor(inputs)

    # slice inputs
    slices = [slice(None)] * inputs.shape.ndims

    if shift >= 0:
        slices[axis] = slice(0, -shift)
    else:
        slices[axis] = slice(-shift, None)

    sliced = inputs[slices]

    # pad inputs
    paddings = [[0, 0]] * inputs.shape.ndims
    paddings[axis] = [max(shift, 0), max(-shift, 0)]
    padding_slices = [slice(None)] * inputs.shape.ndims

    if shift >= 0:
        padding_slices[axis] = slice(-shift, None)
        padded = tf.concat(
            [padding_values * tf.ones_like(inputs[padding_slices]), sliced], axis=axis
        )
    else:
        padding_slices[axis] = slice(0, -shift)
        padded = tf.concat(
            [sliced, tf.ones_like(inputs[padding_slices]) * padding_values], axis=axis
        )

    return padded
