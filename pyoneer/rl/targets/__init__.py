from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyoneer.rl.targets.reward_ops import (
    discounted_returns,
    lambda_returns,
    v_trace_returns,
    generalized_advantage_estimation,
    advantage_estimation_weights,
    DiscountedReturns,
    LambdaReturns,
    VtraceReturns,
    GeneralizedAdvantageEstimation
)

__all__ = [
    "discounted_returns",
    "lambda_returns",
    "v_trace_returns",
    "generalized_advantage_estimation",
    "advantage_estimation_weights",
    "DiscountedReturns",
    "LambdaReturns",
    "VtraceReturns",
    "GeneralizedAdvantageEstimation"
]
