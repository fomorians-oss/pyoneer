from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.eager import backprop


class Agent(keras.Model):
    """Base class for agents.

    This class defines the API to add ops to compute losses and gradients.  
    You never use this class directly, but instead instantiate one of its 
    subclasses such as `AdvantageActorCriticAgent`.
    """

    def __init__(self, optimizer):
        """Create a new `Agent`.

        This must be called by the constructors of subclasses.

        Args:
            optimizer: Instance of `tf.train.Optimizer`.
        """
        super(Agent, self).__init__()
        self.optimizer = optimizer

    def _least_fit(self, tensor, base_shape):
        """Broadcasts tensor into a least fitting tensor compatible with base_shape."""
        tensor = ops.convert_to_tensor(tensor)
        tensor_shape = tensor.shape
        if tensor_shape.ndims == 0:
            return gen_array_ops.tile(
                array_ops.expand_dims(tensor, axis=-1), 
                [base_shape[-1]])
        if tensor_shape.ndims == 1:
            tensor_shape.assert_is_compatible_with([base_shape[-1]])
        if tensor_shape.ndims == 2:
            return gen_array_ops.tile(
                array_ops.expand_dims(tensor, axis=-1), 
                [1, 1, base_shape[-1]])
        if tensor_shape.ndims == 3:
            tensor_shape.assert_is_compatible_with(base_shape)
        raise ValueError('`tensor.shape.ndims` must be at most 3.')

    @property
    def loss(self):
        """Access recent losses computed after `compute_loss(...)` is called.

        Returns:
            tuple associated with the loss, must _atleast_ supply 
                the `total_loss` as the last element.
        """
        raise NotImplementedError()

    def compute_loss(self, *args, **kwargs):
        """Compute losses implemented by the underlying algorithm.
        
        This method is meant to update the `loss` property with the `total_loss` 
        property.

        Args:
            *args: positional arguments required by the underlying 
                algorithm.
            **kwargs: positional arguments required by the underlying 
                algorithm.

        Returns:
            the total loss Tensor computed by the underlying algorithm.
        """
        raise NotImplementedError()

    def estimate_gradients(self, *args, **kwargs):
        """Estimate gradients using the `compute_loss(...)` and `optimizer` property.
        
        This method calls `compute_loss` and returns the associated 
        `grads_and_vars` list.

        Args:
            *args: positional arguments required passed to `compute_loss(...)`.
            **kwargs: keyword arguments passed to `compute_loss(...)`

        Returns:
            list of tuples: `[(grads, variable), ...]`
        """
        with backprop.GradientTape(persistent=True) as tape:
            _ = self.compute_loss(*args, **kwargs)
        vars_ = self.trainable_variables
        grads = tape.gradient(self.total_loss, vars_)
        return list(zip(grads, vars_))

    def fit(self, *args, **kwargs):
        """Short cut for calling optimizing w.r.t estimated gradients.
        
        This method calls `estimate_gradients` and passes the result to 
        `optimizer.apply_gradients(...)`.

        Args:
            *args: positional arguments required passed to `estimate_gradients(...)`.
            **kwargs: keyword arguments passed to `estimate_gradients(...)`

        Returns:
            Op returned by `optimizer.apply_gradients(...)`
        """
        grads_and_vars = self.estimate_gradients(*args, **kwargs)
        return self.optimizer.apply_gradients(grads_and_vars)