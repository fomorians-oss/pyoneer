from tensorflow.python import keras


class Agent(keras.Model):

    def __init__(self, optimizer):
        super(Agent, self).__init__()
        self.optimizer = optimizer

    @property
    def loss(self):
        raise NotImplementedError()

    def compute_loss(self, rollouts, **kwargs):
        raise NotImplementedError()

    def estimate_gradients(self, rollouts, **kwargs):
        return self.optimizer.compute_gradients(
            lambda: self.compute_loss(rollouts, **kwargs), 
            self.trainable_variables)