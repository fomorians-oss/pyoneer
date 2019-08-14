from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.utils import losses_utils


def policy_gradient(log_probs, advantages):
    """
    Computes the Vanilla policy gradient loss.

    Args:
        log_probs: Log probabilities of taking actions under a policy.
        advantages: Advantage estimation.

    Returns:
        Tensor of losses.
    """
    advantages = tf.stop_gradient(advantages)
    losses = -log_probs * advantages
    losses = tf.debugging.check_numerics(losses, "losses")
    return losses


def clipped_policy_gradient(
    log_probs, log_probs_anchor, advantages, epsilon_clipping=0.2
):
    """
    Computes the clipped surrogate objective found in
    [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) based on
    clipped probability ratios.

    Args:
        log_probs: Log probabilities of taking actions under a policy.
        log_probs_anchor: Log probabilities of taking actions under an anchor
            policy which is updated less frequently.
        advantages: Advantage estimation.
        epsilon_clipping: Scalar for clipping the policy ratio.
        sample_weight: Optional tensor for weighting the losses.

    Returns:
        Tensor of losses.
    """
    log_probs_anchor = tf.stop_gradient(log_probs_anchor)
    advantages = tf.stop_gradient(advantages)

    ratio = tf.exp(log_probs - log_probs_anchor)
    ratio_clipped = tf.clip_by_value(ratio, 1 - epsilon_clipping, 1 + epsilon_clipping)

    surrogate1 = ratio * advantages
    surrogate2 = ratio_clipped * advantages

    surrogate_min = tf.minimum(surrogate1, surrogate2)

    losses = -surrogate_min
    losses = tf.debugging.check_numerics(losses, "losses")
    return losses


def policy_entropy(entropy):
    """
    Computes the policy entropy loss.

    Args:
        entropy: Entropy of the policy distribution.

    Returns:
        Tensor of losses.
    """
    losses = -entropy
    losses = tf.debugging.check_numerics(losses, "losses")
    return losses


def soft_policy_gradient(log_probs, targets, alpha=1.0):
    """
    Computes the soft policy gradient loss.

    Args:
        log_probs: Log probabilities of taking actions under a policy.
        targets: Compute the target action values.

    Returns:
        Tensor of losses.
    """
    targets = tf.stop_gradient(targets)
    losses = alpha * log_probs - targets
    losses = tf.debugging.check_numerics(losses, "losses")
    return losses


def soft_policy_entropy(log_alpha, log_probs, target_entropy):
    """
    Computes the soft policy entropy loss to dynamically adjust the temperature.

    Args:
        log_alpha: Log alpha parameter.
        log_probs: Log probabilities of taking actions under a policy.
        target_entropy: Target entropy for the policy.

    Returns:
        Tensor of losses.
    """
    log_probs = tf.stop_gradient(log_probs)
    losses = -log_alpha * (log_probs + target_entropy)
    losses = tf.debugging.check_numerics(losses, "losses")
    return losses


class PolicyGradient(tf.keras.losses.Loss):
    """
    Computes the Vanilla policy gradient loss.

    Attributes:
        reduction: a tf.keras.losses.Reduction method.
        name: name of the loss.
    """

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name=None):
        self.reduction = reduction
        self.name = name

    def __call__(self, log_probs, advantages, sample_weight=None):
        """
        Computes the Vanilla policy gradient loss.

        Args:
            log_probs: Log probabilities of taking actions under a policy.
            advantages: Advantage estimation.

        Returns:
            Scalar loss tensor.
        """
        losses = policy_gradient(log_probs, advantages)
        return losses_utils.compute_weighted_loss(
            losses, sample_weight, reduction=self.reduction
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {"reduction": self.reduction, "name": self.name}


class ClippedPolicyGradient(tf.keras.losses.Loss):
    """
    Computes the clipped surrogate objective found in
    [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) based on
    clipped probability ratios.

    Attributes:
        epsilon_clipping: epsilon parameter of the clipped surrogate objective.
        reduction: a tf.keras.losses.Reduction method.
        name: name of the loss.
    """

    def __init__(
        self, epsilon_clipping=0.2, reduction=tf.keras.losses.Reduction.AUTO, name=None
    ):
        self.epsilon_clipping = epsilon_clipping
        self.reduction = reduction
        self.name = name

    def __call__(self, log_probs, log_probs_anchor, advantages, sample_weight=None):
        """
        Computes the clipped surrogate objective found in
        [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) based on
        clipped probability ratios.

        Args:
            log_probs: Log probabilities of taking actions under a policy.
            log_probs_anchor: Log probabilities of taking actions under an anchor
                policy which is updated less frequently.
            advantages: Advantage estimation.
            epsilon_clipping: Scalar for clipping the policy ratio.
            sample_weight: Optional tensor for weighting the losses.

        Returns:
            Scalar loss tensor.
        """
        losses = clipped_policy_gradient(log_probs, log_probs_anchor, advantages)
        return losses_utils.compute_weighted_loss(
            losses, sample_weight, reduction=self.reduction
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {
            "epsilon_clipping": self.epsilon_clipping,
            "reduction": self.reduction,
            "name": self.name,
        }


class PolicyEntropy(tf.keras.losses.Loss):
    """
    Computes the policy entropy loss.

    Attributes:
        reduction: a tf.keras.losses.Reduction method.
        name: name of the loss.
    """

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name=None):
        self.reduction = reduction
        self.name = name

    def __call__(self, entropy, sample_weight=None):
        """
        Computes the policy entropy loss.

        Args:
            entropy: Entropy of the policy distribution.

        Returns:
            Scalar loss tensor.
        """
        losses = policy_entropy(entropy)
        return losses_utils.compute_weighted_loss(
            losses, sample_weight, reduction=self.reduction
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {"reduction": self.reduction, "name": self.name}


class SoftPolicyGradient(tf.keras.losses.Loss):
    """
    Computes the soft policy gradient loss.

    Attributes:
        reduction: a tf.keras.losses.Reduction method.
        name: name of the loss.
    """

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name=None):
        self.reduction = reduction
        self.name = name

    def __call__(self, log_probs, targets, alpha=1.0, sample_weight=None):
        """
        Computes the soft policy gradient loss.

        Args:
            log_probs: Log probabilities of taking actions under a policy.
            targets: Compute the target action values.

        Returns:
            Scalar loss tensor.
        """
        losses = soft_policy_gradient(log_probs, targets, alpha)
        return losses_utils.compute_weighted_loss(
            losses, sample_weight, reduction=self.reduction
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {"reduction": self.reduction, "name": self.name}


class SoftPolicyEntropy(tf.keras.losses.Loss):
    """
    Computes the soft policy entropy loss to dynamically adjust the temperature.

    Attributes:
        reduction: a tf.keras.losses.Reduction method.
        name: name of the loss.
    """

    def __init__(
        self, target_entropy, reduction=tf.keras.losses.Reduction.AUTO, name=None
    ):
        self.target_entropy = target_entropy
        self.reduction = reduction
        self.name = name

    def __call__(self, log_alpha, log_probs, sample_weight=None):
        """
        Computes the soft policy entropy loss to dynamically adjust the temperature.

        Args:
            log_alpha: Log alpha parameter.
            log_probs: Log probabilities of taking actions under a policy.
            target_entropy: Target entropy for the policy.

        Returns:
            Scalar loss tensor.
        """
        losses = soft_policy_entropy(log_alpha, log_probs, self.target_entropy)
        return losses_utils.compute_weighted_loss(
            losses, sample_weight, reduction=self.reduction
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {
            "target_entropy": self.target_entropy,
            "reduction": self.reduction,
            "name": self.name,
        }
