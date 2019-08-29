from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyoneer.rl.targets.reward_ops import (
    n_step_discounted_bootstrap_values,
    n_step_discounted_returns,
    discounted_returns,
    temporal_difference,
    v_trace_returns,
    generalized_advantage_estimate,
    NstepDiscountedReturns,
    DiscountedReturns,
    VtraceReturns,
    GeneralizedAdvantageEstimation
)

__all__ = [
    "n_step_discounted_bootstrap_values",
    "n_step_discounted_returns",
    "discounted_returns",
    "temporal_difference",
    "v_trace_returns",
    "generalized_advantage_estimate",
    "NstepDiscountedReturns",
    "DiscountedReturns",
    "VtraceReturns",
    "GeneralizedAdvantageEstimation"
]
