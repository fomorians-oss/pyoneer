from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
        grads_and_vars = self.optimizer.compute_gradients(
            lambda: self.compute_loss(rollouts, **kwargs), 
            self.trainable_variables)
        return grads_and_vars

    def fit(self, rollouts, **kwargs):
        grads_and_vars = self.estimate_gradients(rollouts, **kwargs)
        return self.optimizer.apply_gradients(grads_and_vars)