import tensorflow as tf
from tensorflow.python.framework import tensor_shape

from tensorflow_probability.python import distributions


class GumbleSoftmax(distributions.RelaxedOneHotCategorical):
    """GumbleSoftmax distribution wrapper for the `RelaxedOneHotCategorical` distribution.

    Implements `d.mode()` and `d.entropy()`.

    Example:
        >>> import tensorflow as tf
        >>> tf.enable_eager_execution()
        >>> import pyoneer as pynr
        >>> d = pynr.distributions.GumbleSoftmax(1., probs=[.9, .1])
        >>> d.sample()
        <tf.Tensor: id=0, shape=(), dtype=int32, numpy=0>
        >>> d.mode()
        <tf.Tensor: id=1, shape=(), dtype=int32, numpy=0>
        >>> d.entropy()
        <tf.Tensor: id=2, shape=(), dtype=float32, numpy=0.32508302>
    """

    def __init__(
        self,
        temperature,
        logits=None,
        probs=None,
        validate_args=False,
        allow_nan_stats=True,
        name="GumbleSoftmax"):
        super(GumbleSoftmax, self).__init__(
            temperature,
            logits=logits,
            probs=probs,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name)

    @property
    def temperature(self):
        return self._distribution.temperature

    @temperature.setter
    def temperature(self, temp):
        self._distribution.temperature = temp

    @property
    def logits(self):
        return self._distribution.logits

    @property
    def probs(self):
        return self._distribution.probs
    
    @property
    def dtype(self):
        return self._distribution.dtype

    def sample(self, sample_shape=(), seed=None):
        sample = super(GumbleSoftmax, self).sample(sample_shape=sample_shape, seed=seed)
        return self._categorical_mode(sample)

    def _entropy(self):
        return -tf.reduce_sum(
            tf.nn.log_softmax(self.logits / self.temperature) * self.probs, axis=-1)

    def _categorical_mode(self, x):
        ret = tf.argmax(x, axis=-1)
        ret = tf.cast(ret, tf.int32)
        return ret

    def _mode(self):
        return self._categorical_mode(self.logits)
