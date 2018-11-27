from tensorflow.python import keras
from tensorflow.python.ops import array_ops


class RNN(keras.Model):

    def __init__(self, cell):
        super(RNN, self).__init__()
        self.cell = cell
        self.state = None

    def call(self, inputs, training=False, reset_state=True):
        if reset_state:
            self.state = self.cell.zero_state(inputs.shape[0], tf.float32)
        outputs = []
        inputs = array_ops.unstack(inputs, num=inputs.shape[1], axis=1)
        for i, inputs_i in enumerate(inputs):
            output, self.state = self.cell(inputs_i, self.state, training=training)
            outputs.append(output)
        return array_ops.stack(outputs, axis=1)
