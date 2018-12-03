from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python import keras
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops


class RNN(keras.Model):

    def __init__(self, cell):
        super(RNN, self).__init__()
        self.cell = cell
        self.state = None

    def call(self, inputs, reset_state=True, **kwargs):
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
