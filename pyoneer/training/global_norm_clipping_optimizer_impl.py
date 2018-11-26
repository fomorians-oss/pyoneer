import numpy as np

from tensorflow.python.training import optimizer
from tensorflow.python.ops import clip_ops


class GlobalNormClippingOptimizer(optimizer.Optimizer):
    """Optimizer that clips gradients by global norm."""

    def __init__(self,
                 opt,
                 clip_norm,
                 use_locking=False,
                 name="GlobalNormClippingOptimizer"):
        super(GlobalNormClippingOptimizer, self).__init__(use_locking, name)
        self._opt = opt
        self._clip_norm = clip_norm

    def compute_gradients(self, *args, **kwargs):
        return self._opt.compute_gradients(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        if self._clip_norm == np.inf:
            return self._opt.apply_gradients(grads_and_vars, *args, **kwargs)
        grads, vars_ = zip(*grads_and_vars)
        clipped_grads, _ = clip_ops.clip_by_global_norm(grads, self._clip_norm)
        return self._opt.apply_gradients(zip(clipped_grads, vars_), *args, **kwargs)
