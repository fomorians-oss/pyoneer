import tensorflow as tf


class RNN(tf.keras.Model):

    def __init__(self, cell):
        super(RNN, self).__init__()
        self.cell = cell
        self.state = None

    def call(self, inputs, training=False, reset_state=True):
        if reset_state:
            self.state = self.cell.zero_state(inputs.shape[0], tf.float32)
        outputs, self.state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=self.state)
        return outputs
