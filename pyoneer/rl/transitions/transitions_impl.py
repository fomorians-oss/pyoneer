from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class TransitionBuffer(object):
    def __init__(self, max_size=None):
        self.max_size = max_size
        self.transitions = None

    @property
    def size(self):
        if self.transitions is None:
            size = 0
        else:
            for key in self.transitions.keys():
                size = self.transitions[key].shape[0]
                break
        return size

    def __len__(self):
        return self.size

    def append(self, transitions):
        def append_transitions(elem_old, elem_new):
            return np.concatenate([elem_old, elem_new], axis=0)

        def truncate_transitions(elem):
            return elem[:self.max_size]

        # append the new transitions
        if self.transitions is not None:
            self.transitions = tf.nest.map_structure(
                append_transitions, self.transitions, transitions
            )
        else:
            self.transitions = transitions

        # truncate the transitions to the maximum size
        self.transitions = tf.nest.map_structure(
            truncate_transitions, self.transitions
        )

    def sample(self, batch_size):
        sample_indices = np.random.choice(self.size, batch_size)

        def sample_transitions(elem):
            return elem[sample_indices]

        transitions = tf.nest.map_structure(sample_transitions, self.transitions)
        return transitions
