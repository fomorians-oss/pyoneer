import tensorflow as tf


class RNN(tf.keras.Model):

    def __init__(self, cell):
        super(RNN, self).__init__()
        self.cell = cell
        self.state = None

    def call(self, inputs, training=False, reset_state=True):
        if reset_state:
            self.state = self.cell.zero_state(inputs.shape[0], tf.float32)
        outputs = []
        inputs = tf.unstack(inputs, num=inputs.shape[1], axis=1)
        for i, inputs_i in enumerate(inputs):
            output, self.state = self.cell(inputs_i, self.state, training=training)
            outputs.append(output)
        return tf.stack(outputs, axis=1)
