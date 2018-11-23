from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def isclose(x, y, epsilon=1e-7):
    x = ops.convert_to_tensor(x)
    y = ops.convert_to_tensor(y)
    epsilon = ops.convert_to_tensor(epsilon)
    return math_ops.abs(x - y) <= (epsilon + epsilon * 2e-7)
