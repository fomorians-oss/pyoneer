from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python import keras
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops


class RNN(keras.Model):
    """Implements an RNN wrapper.

    Wrapper for classes that inherit from `tf.nn.rnn_cell.RNNCell` 
    as a `tf.keras.Model`.
    """

    def __init__(self, cell):
        """Creates a new RNN.
        
        Args: 
            cell: class that possibly inherits from `tf.nn.rnn_cell.RNNCell`. 
                This means that `cell` should implement `cell.zero_state(shape, dtype)` 
                and is callable that takes a 2-D Tensor, and a state Tensor of the 
                same shape as `cell.zero_state(...)` and returns a 2-D Tensor and 
                a state Tensor of the same shape as `cell.zero_state(...)`.
        """
        super(RNN, self).__init__()
        self.cell = cell
        self.state = None

    def call(self, inputs, reset_state=True, **kwargs):
        """Calls the internal `cell` over the time distrbuted `inputs`.

        Args:
            inputs: 2-D or 3-D Tensor passed to `cell` each iteration of the 
                time dimension.
            reset_state: Flag determining if the `cell` state should set to 
                `cell.zero_state(...)`. 
            **kwargs: Optional keyward arguments passed to `cell(..)`.
        
        Returns:
            2-D or 3-D Tensor depending on the shape of the `inputs`.
        """
        no_time = inputs.shape.ndims < 3
        if no_time:
            inputs = gen_array_ops.reshape(inputs, [-1, 1, inputs.shape[-1]])
        if reset_state:
            self.state = self.cell.zero_state(inputs.shape[0], dtypes.float32)
        outputs = []
        inputs = array_ops.unstack(inputs, num=inputs.shape[1], axis=1)
        for i, inputs_i in enumerate(inputs):
            output, self.state = self.cell(inputs_i, self.state, **kwargs)
            outputs.append(output)
        outputs = array_ops.stack(outputs, axis=1)
        if no_time:
            outputs = array_ops.squeeze(outputs, axis=1)
        return outputs
