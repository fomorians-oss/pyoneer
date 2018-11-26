from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class MockSpace(object):
    
    def __init__(self, dtype):
        self.dtype = dtype


class CounterEnv(object):

    def __init__(self):
        self.observation_space = MockSpace(dtypes.float32)
        self.action_space = MockSpace(dtypes.float32)
        self._step = 0

    def reset(self, size=None):
        self._step = 1
        if size:
            return array_ops.zeros(size, dtype=self.observation_space.dtype)
        return math_ops.cast(0, dtype=self.observation_space.dtype)

    def step(self, action):
        zero = array_ops.zeros_like(action)
        state = math_ops.cast(zero, self.observation_space.dtype) + self._step
        self._step += 1
        return state, math_ops.cast(zero, dtypes.float32), math_ops.cast(zero, dtypes.bool), {}
