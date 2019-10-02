from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pyoneer.layers import layers_impl


class SelfAttention(tf.keras.layers.Layer):

    def __init__(self, key_size, value_size, num_heads, **kwargs):
        """Creates a new SelfAttention.

        Args:
            key_size: The size of the query and key dimension.
            value_size: The size of the value dimension.
            num_heads: Number of attention values.
            **kwargs: Keyword arguments for the Q, K, V computation.
        """
        super().__init__()
        self._num_heads = num_heads
        self._key_size = key_size
        self._value_size = value_size
        self._qkv = layers_impl.BatchApply(
            tf.keras.layers.Dense(
                self._num_heads * (
                    self._key_size + self._key_size + self._value_size),
                **kwargs))

    def call(self, inputs):
        """Applies multi-headed dot-product attention to the inputs.

        Args:
            inputs: Tensor of shape: [Batch x Memory Size x D].

        Returns:
            Attended outputs tensor of shape:
                [Batch x Memory Size x (Value Size * Num Heads)]
        """
        batch_size = tf.shape(inputs)[0]

        # Compute Q, K, V in parallel.
        qkv = self._qkv(inputs)
        qkv_reshape = tf.reshape(
            qkv,
            [batch_size, -1, self._num_heads, self._key_size + self._key_size + self._value_size])

        # Treat each head as a batch of attention views.
        qkv_t = tf.transpose(qkv_reshape, [0, 2, 1, 3])
        q, k, v = tf.split(
            qkv_t,
            [self._key_size, self._key_size, self._value_size],
            axis=-1)

        # Compute attention scores.
        attention = tf.matmul(q, k, transpose_b=True)
        attention *= tf.sqrt(tf.cast(self._key_size, tf.dtypes.float32))
        attention_weights = tf.nn.softmax(attention, axis=-1)
        attention_output = tf.matmul(attention_weights, v)

        # Restore heads.
        attention_output_t = tf.transpose(attention_output, [0, 2, 1, 3])

        # Flatten head values.
        output = tf.reshape(
            attention_output_t,
            [batch_size, -1, self._value_size * self._num_heads])
        return output
