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
        grads = self.optimizer.compute_gradients(
            lambda: self.compute_loss(rollouts, **kwargs), 
            self.trainable_variables)
        return zip(grads, self.trainable_variables)

    def fit(self, rollouts, **kwargs):
        grads_and_vars = self.estimate_gradients(rollouts, **kwargs)
        return self.optimizer.apply_gradients(grads_and_vars)