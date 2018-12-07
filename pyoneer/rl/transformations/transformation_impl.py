from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Transformation(tf.keras.Model):
    """Base class for common RL data transformations.

    This class defines the API to add apply transformations. You never use this 
    class directly, but instead instantiate one of its subclasses such as
    `DistillationBonusTransformation`.
    """

    def __init__(self):
        """Creates a new Transformation."""
        super(Transformation, self).__init__()

    def call(self, *args, **kwargs):
        """Transform Tensors according to the underlying transformation.

        Args:
            *args: positional arguments required by and passed to the 
                underlying implementation.
            **kwargs: keyword arguments required by and passed to the 
                underlying implementation.

        Returns:
            Transformed Tensors.
        """
        raise NotImplementedError('`Transformation` must implement `call`.')