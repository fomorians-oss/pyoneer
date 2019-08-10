from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyoneer.rl.targets.target_ops import DiscountedReturns, GeneralizedAdvantages
from pyoneer.rl.targets.sequence_ops import scan_discounted_sum, multistep_forward_view

__all__ = ["DiscountedReturns", "GeneralizedAdvantages"]
