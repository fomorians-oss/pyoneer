from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gym.core import Space
from gym.spaces import box
from gym.spaces import dict_space
from gym.spaces import tuple_space

from pyoneer.math import normalization_ops
from pyoneer.features import normalizer_impl


def high_low_normalizer_from_gym_space(space, clip_inf=None):
    assert isinstance(space, Space), '`space` must be an instance of `gym.Space`'

    if isinstance(space, box.Box):
        high, low = space.high, space.low
        if clip_inf:
            high = np.clip(high, -clip_inf, clip_inf)
            low = np.clip(low, -clip_inf, clip_inf)
        loc, scale = normalization_ops.high_low_loc_and_scale(high, low)
        return normalizer_impl.HighLowNormalizer(loc, scale)

    elif isinstance(space, tuple_space.Tuple):
        return tuple(high_low_normalizer_from_gym_space(val) 
                     for val in space.spaces)

    elif isinstance(space, dict_space.Dict):
        return {key: high_low_normalizer_from_gym_space(val) 
                for key, val in space.spaces.items()}

    raise TypeError('`space` not supported: {}'.format(type(space)))
