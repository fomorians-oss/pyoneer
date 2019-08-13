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

    def __getitem__(self, key):
        return self.transitions[key]

    def __contains__(self, key):
        return key in self.transitions

    def __len__(self):
        return self.size

    def append(self, transitions):
        # append the new transitions
        def append_transitions(elem_old, elem_new):
            return np.concatenate([elem_new, elem_old], axis=0)

        if self.transitions is not None:
            self.transitions = tf.nest.map_structure(
                append_transitions, self.transitions, transitions
            )
        else:
            self.transitions = transitions

        # truncate the transitions to the maximum size
        def truncate_transitions(elem):
            return elem[: self.max_size]

        if self.max_size is not None:
            self.transitions = tf.nest.map_structure(
                truncate_transitions, self.transitions
            )

    def update(self, indices, updates):
        def update_transitions(elem_old, elem_new):
            elem_old[indices] = elem_new

        tf.nest.map_structure(update_transitions, self.transitions, updates)

    def sample(self, size, p=None, return_indices=False):
        indices = np.random.choice(self.size, size=size, p=p, replace=False)

        def sample_transitions(elem):
            return elem[indices]

        transitions = tf.nest.map_structure(sample_transitions, self.transitions)

        if return_indices:
            return transitions, indices
        else:
            return transitions

    def save(self, path):
        np.savez_compressed(path, **self.transitions)

    def restore(self, path):
        with np.load(path) as data:
            self.transitions = dict(data)
