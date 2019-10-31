from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


@tf.function
def discounted_returns(rewards,
                       bootstrap_value=None,
                       discounts=0.99,
                       weights=1.0,
                       time_major=False,
                       back_prop=False,
                       name=None):
    """Compute discounted returns.

    Args:
        rewards: tensor of shape [Batch x Time], [Batch x Time x ...]
        bootstrap_value: The last value [t+n]. A tensor of shape [],
            [...], [Batch], [Batch, ...], [Batch x Time], [Batch x Time x ...].
        discounts: tensor of shape [], [...], [Batch], [Batch, ...],
            [Batch x Time], [Batch x Time x ...].
        weights: tensor of shape [], [...], [Batch], [Batch, ...],
            [Batch x Time], [Batch x Time x ...].
        time_major: flag if tensors are time_major.
            Batch and Time are transposed in this doc.
        back_prop: allow back_prop through the calculation.
        name: optional op name.

    Returns:
        tensor of shape [Batch x Time x ...] or [Time x Batch x ...]
    """
    with tf.name_scope(name or 'DiscountedReturns'):
        batch_axis = int(time_major)

        # Check rewards shape.
        rewards = tf.convert_to_tensor(rewards)
        rewards_rank = rewards.shape.rank
        assert rewards_rank >= 2, 'rewards must be atleast rank 2.'
        rewards_shape = tf.shape(rewards)

        # Compute the batch shape of rewards.
        if time_major:
            reward_shape_no_time = rewards_shape[1:]
        else:
            reward_shape_no_time = rewards_shape[:1]
            if rewards_rank > 2:
                reward_shape_no_time = tf.concat([reward_shape_no_time,
                                                  rewards_shape[2:]], axis=-1)

        # Check discounts shape, broadcast to rewards shape.
        discounts = tf.convert_to_tensor(discounts)
        assert discounts.shape.rank <= rewards.shape.rank, (
            'discounts rank must be <= rewards rank.')
        discounts = tf.broadcast_to(discounts, rewards_shape)

        # Check weights shape, broadcast to weights shape.
        weights = tf.convert_to_tensor(weights)
        assert weights.shape.rank <= rewards.shape.rank, (
            'weights rank must be <= rewards rank.')
        weights = tf.broadcast_to(weights, rewards_shape)

        # Check if bootstrap values are supplied. If bootstrap values exist,
        # we want them to be the same shape as the batch.
        if bootstrap_value is None:
            bootstrap_value = tf.zeros(reward_shape_no_time, rewards.dtype)
        else:
            bootstrap_value = tf.convert_to_tensor(bootstrap_value)
            bootstrap_value = tf.broadcast_to(bootstrap_value,
                                              reward_shape_no_time)

        def reduce_fn(agg, cur):
            next_agg = cur[0] + cur[1] * agg
            return next_agg

        if not time_major:
            rank_list = list(range(2, rewards_rank))
            rewards = tf.transpose(rewards, [1, 0] + rank_list)
            discounts = tf.transpose(discounts, [1, 0] + rank_list)

        returns = tf.scan(fn=reduce_fn,
                          elems=[rewards, discounts],
                          initializer=bootstrap_value,
                          parallel_iterations=1,  # chronological.
                          back_prop=back_prop,
                          reverse=True)

        if not time_major:
            rank_list = list(range(2, rewards_rank))
            returns = tf.transpose(returns, [1, 0] + rank_list)

        returns = returns * weights
        returns = tf.debugging.check_numerics(returns, 'Returns')
        if not back_prop:
            returns = tf.stop_gradient(returns)
        return returns


@tf.function
def lambda_returns(rewards,
                   values,
                   bootstrap_value=None,
                   discounts=.99,
                   lambdas=.975,
                   weights=1.,
                   time_major=False,
                   back_prop=False,
                   name=None):
    """Computes lambda-returns.

    For example, this can be used to compute TD(lambda):

        ```
        td_lambda = lambda_returns(r(t), V(t+1)) - V(t)
        ```

    Args:
        rewards: tensor of shape [Batch x Time], [Batch x Time x ...].
        values: tensor of shape [Batch x Time], [Batch x Time x ...].
        bootstrap_value: tensor of shape [], [Batch], [Batch x ...].
        discounts: tensor of shape [], [...], [Batch], [Batch, ...],
            [Batch x Time], [Batch x Time x ...].
        lambdas: tensor of shape [], [...], [Batch], [Batch, ...],
            [Batch x Time], [Batch x Time x ...].
        weights: tensor of shape [], [...], [Batch], [Batch, ...],
            [Batch x Time], [Batch x Time x ...].
        time_major: flag if tensors are time_major.
            Batch and Time are transposed in this doc.
        back_prop: allow back_prop through the calculation.
        name: optional op name.

    Returns:
        tensor of shape [Batch x Time] or [Time x Batch]
    """
    with tf.name_scope(name or 'GeneralizedAdvantageEstimate'):
        batch_axis = int(time_major)
        time_axis = int(not time_major)

        # Check rewards shape.
        rewards = tf.convert_to_tensor(rewards)
        rewards_rank = rewards.shape.rank
        assert rewards_rank >= 2, 'rewards must be atleast rank 2.'
        rewards_shape = tf.shape(rewards)

        # Check discounts shape, broadcast to rewards shape.
        discounts = tf.convert_to_tensor(discounts)
        assert discounts.shape.rank <= rewards.shape.rank, (
            'discounts rank must be <= rewards rank.')
        discounts = tf.broadcast_to(discounts, rewards_shape)

        # Check lambdas shape, broadcast to rewards shape.
        lambdas = tf.convert_to_tensor(lambdas)
        assert lambdas.shape.rank <= rewards.shape.rank, (
            'lambdas rank must be <= rewards rank.')
        lambdas = tf.broadcast_to(lambdas, rewards_shape)

        # Check weights shape, broadcast to weights shape.
        weights = tf.convert_to_tensor(weights)
        assert weights.shape.rank <= rewards.shape.rank, (
            'weights rank must be <= rewards rank.')
        weights = tf.broadcast_to(weights, rewards_shape)

        if bootstrap_value is None:
            if time_major:
                bootstrap_value = tf.zeros_like(values[-1, :])
            else:
                bootstrap_value = tf.zeros_like(values[:, -1])
        else:
            bootstrap_value = tf.convert_to_tensor(bootstrap_value)

        bootstrap_value_t = tf.expand_dims(bootstrap_value, axis=time_axis)

        # Shift values to tp1.
        if time_major:
            values_tp1 = values[1:, :]
        else:
            values_tp1 = values[:, 1:]

        # Pad the values with the last value.
        values_tp1 = tf.concat([values_tp1, bootstrap_value_t],
                               axis=time_axis)

        delta = rewards + discounts * values_tp1 * (1. - lambdas)

        def reduce_fn(agg, cur):
            next_agg = cur[0] + cur[1] * agg
            return next_agg

        if time_major:
            elements = (delta * weights,
                        lambdas * discounts * weights)
        else:
            rank_list = list(range(2, rewards_rank))
            elements = (tf.transpose(delta * weights, [1, 0] + rank_list),
                        tf.transpose(lambdas * discounts * weights,
                                     [1, 0] + rank_list))

        returns = tf.scan(
            fn=reduce_fn,
            elems=elements,
            initializer=tf.zeros_like(bootstrap_value),
            parallel_iterations=1,
            back_prop=back_prop,
            reverse=True)

        if not time_major:
            rank_list = list(range(2, rewards_rank))
            returns = tf.transpose(returns, [1, 0] + rank_list)
        returns = returns * weights
        returns = tf.debugging.check_numerics(returns, 'Returns')
        if not back_prop:
            returns = tf.stop_gradient(returns)
        return returns


@tf.function
def v_trace_returns(rewards,
                    values,
                    log_probs,
                    log_probs_old,
                    bootstrap_value=None,
                    discounts=.99,
                    weights=1.,
                    time_major=False,
                    back_prop=False,
                    name=None):
    """Computes v-trace returns.

    Args:
        rewards: tensor of shape [Batch x Time], [Batch x Time x ...].
        values: tensor of shape [Batch x Time], [Batch x Time x ...].
        log_probs: tensor of shape [Batch x Time], [Batch x Time x ...].
        log_probs_old: tensor of shape [Batch x Time], [Batch x Time x ...].
        bootstrap_value: tensor of shape [], [Batch], [Batch x ...].
        discounts: tensor of shape [], [...], [Batch], [Batch, ...],
            [Batch x Time], [Batch x Time x ...].
        weights: tensor of shape [], [...], [Batch], [Batch, ...],
            [Batch x Time], [Batch x Time x ...].
        time_major: flag if tensors are time_major.
            Batch and Time are transposed in this doc.
        back_prop: allow back_prop through the calculation.
        name: optional op name.

    Returns:
        tensor of shape [Batch x Time] or [Time x Batch]
    """
    with tf.name_scope(name or 'VTraceReturns'):
        batch_axis = int(time_major)
        time_axis = int(not time_major)

        # Check rewards shape.
        rewards = tf.convert_to_tensor(rewards)
        rewards_rank = rewards.shape.rank
        assert rewards_rank >= 2, 'rewards must be atleast rank 2.'
        rewards_shape = tf.shape(rewards)

        # Check discounts shape, broadcast to rewards shape.
        discounts = tf.convert_to_tensor(discounts)
        assert discounts.shape.rank <= rewards.shape.rank, (
            'discounts rank must be <= rewards rank.')
        discounts = tf.broadcast_to(discounts, rewards_shape)

        # Check weights shape, broadcast to weights shape.
        weights = tf.convert_to_tensor(weights)
        assert weights.shape.rank <= rewards.shape.rank, (
            'weights rank must be <= rewards rank.')
        weights = tf.broadcast_to(weights, rewards_shape)

        if bootstrap_value is None:
            if time_major:
                bootstrap_value = tf.zeros_like(values[-1, :])
            else:
                bootstrap_value = tf.zeros_like(values[:, -1])
        else:
            bootstrap_value = tf.convert_to_tensor(bootstrap_value)

        clipped_ratios = tf.minimum(1., tf.exp(log_probs - log_probs_old))

        bootstrap_value_t = tf.expand_dims(bootstrap_value, axis=time_axis)

        # Shift values to tp1.
        if time_major:
            values_tp1 = values[1:, :]
        else:
            values_tp1 = values[:, 1:]

        # Pad the values with the last value.
        values_tp1 = tf.concat([values_tp1, bootstrap_value_t],
                               axis=time_axis)

        advantages = rewards + discounts * values_tp1 - values
        delta = clipped_ratios * advantages

        def reduce_fn(agg, cur):
            next_agg = cur[0] + cur[1] * agg
            return next_agg

        if time_major:
            elements = (delta * weights,
                        clipped_ratios * discounts * weights)
        else:
            rank_list = list(range(2, rewards_rank))
            elements = (tf.transpose(delta * weights, [1, 0] + rank_list),
                        tf.transpose(clipped_ratios * discounts * weights,
                                     [1, 0] + rank_list))

        v_trace_values = tf.scan(fn=reduce_fn,
                                 elems=elements,
                                 initializer=tf.zeros_like(bootstrap_value),
                                 parallel_iterations=1,
                                 back_prop=back_prop,
                                 reverse=True)

        if not time_major:
            rank_list = list(range(2, rewards_rank))
            v_trace_values = tf.transpose(v_trace_values, [1, 0] + rank_list)

        returns = v_trace_values + values
        returns = tf.debugging.check_numerics(returns, 'Returns')
        if not back_prop:
            returns = tf.stop_gradient(returns)
        return returns


@tf.function
def generalized_advantage_estimation(rewards,
                                     values,
                                     bootstrap_value=None,
                                     discounts=.99,
                                     lambdas=.975,
                                     weights=1.,
                                     time_major=False,
                                     back_prop=False,
                                     name=None):
    """Computes Generalized Advantage Estimation.

    Args:
        rewards: tensor of shape [Batch x Time], [Batch x Time x ...].
        values: tensor of shape [Batch x Time], [Batch x Time x ...].
        bootstrap_value: tensor of shape [], [Batch], [Batch x ...].
        discounts: tensor of shape [], [...], [Batch], [Batch, ...],
            [Batch x Time], [Batch x Time x ...].
        lambdas: tensor of shape [], [...], [Batch], [Batch, ...],
            [Batch x Time], [Batch x Time x ...].
        weights: tensor of shape [], [...], [Batch], [Batch, ...],
            [Batch x Time], [Batch x Time x ...].
        time_major: flag if tensors are time_major.
            Batch and Time are transposed in this doc.
        back_prop: allow back_prop through the calculation.
        name: optional op name.

    Returns:
        tensor of shape [Batch x Time] or [Time x Batch]
    """
    with tf.name_scope(name or 'GeneralizedAdvantageEstimate'):
        batch_axis = int(time_major)
        time_axis = int(not time_major)

        # Check rewards shape.
        rewards = tf.convert_to_tensor(rewards)
        rewards_rank = rewards.shape.rank
        assert rewards_rank >= 2, 'rewards must be atleast rank 2.'
        rewards_shape = tf.shape(rewards)

        # Check discounts shape, broadcast to rewards shape.
        discounts = tf.convert_to_tensor(discounts)
        assert discounts.shape.rank <= rewards.shape.rank, (
            'discounts rank must be <= rewards rank.')
        discounts = tf.broadcast_to(discounts, rewards_shape)

        # Check lambdas shape, broadcast to rewards shape.
        lambdas = tf.convert_to_tensor(lambdas)
        assert lambdas.shape.rank <= rewards.shape.rank, (
            'lambdas rank must be <= rewards rank.')
        lambdas = tf.broadcast_to(lambdas, rewards_shape)

        # Check weights shape, broadcast to weights shape.
        weights = tf.convert_to_tensor(weights)
        assert weights.shape.rank <= rewards.shape.rank, (
            'weights rank must be <= rewards rank.')
        weights = tf.broadcast_to(weights, rewards_shape)

        if bootstrap_value is None:
            if time_major:
                bootstrap_value = tf.zeros_like(values[-1, :])
            else:
                bootstrap_value = tf.zeros_like(values[:, -1])
        else:
            bootstrap_value = tf.convert_to_tensor(bootstrap_value)

        bootstrap_value_t = tf.expand_dims(bootstrap_value, axis=time_axis)

        # Shift values to tp1.
        if time_major:
            values_tp1 = values[1:, :]
        else:
            values_tp1 = values[:, 1:]

        # Pad the values with the last value.
        values_tp1 = tf.concat([values_tp1, bootstrap_value_t],
                               axis=time_axis)

        delta = rewards + discounts * values_tp1 - values

        def reduce_fn(agg, cur):
            next_agg = cur[0] + cur[1] * agg
            return next_agg

        if time_major:
            elements = (delta * weights,
                        lambdas * discounts * weights)
        else:
            rank_list = list(range(2, rewards_rank))
            elements = (tf.transpose(delta * weights, [1, 0] + rank_list),
                        tf.transpose(lambdas * discounts * weights,
                                     [1, 0] + rank_list))

        advantages = tf.scan(
            fn=reduce_fn,
            elems=elements,
            initializer=tf.zeros_like(bootstrap_value),
            parallel_iterations=1,
            back_prop=back_prop,
            reverse=True)

        if not time_major:
            rank_list = list(range(2, rewards_rank))
            advantages = tf.transpose(advantages, [1, 0] + rank_list)
        advantages = advantages * weights
        advantages = tf.debugging.check_numerics(advantages, 'Advantages')
        if not back_prop:
            advantages = tf.stop_gradient(advantages)
        return advantages


def advantage_estimation_weights(advantage, beta, w_max):
    """Advantage weighted regression scores."""
    return tf.minimum(tf.exp(advantage / beta), w_max)


class Returns(object):
    """Base class for computing returns."""

    def __init__(self, time_major):
        self.time_major = time_major

    def bootstrap_value(self, value_tp1, t, terminal):
        value_tp1_rev = tf.reverse_sequence(
            value_tp1, t, seq_axis=int(not self.time_major))
        terminal_rev = tf.reverse_sequence(
            terminal, t, seq_axis=int(not self.time_major))

        if self.time_major:
            bootstrap_value = value_tp1_rev[0]
            bootstrap_terminal = terminal_rev[0]
        else:
            bootstrap_value = value_tp1_rev[:, 0]
            bootstrap_terminal = terminal_rev[:, 0]

        return bootstrap_value * tf.cast(
            ~bootstrap_terminal, bootstrap_value.dtype)


class DiscountedReturns(Returns):

    def __init__(self, time_major=False, back_prop=False, name=None):
        """Creates a new DiscountedReturns

        Args:
            time_major: flag if tensors are time_major.
                Batch and Time are transposed in this doc.
            back_prop: allow back_prop through the calculation.
            name: optional op name.
        """
        super().__init__(time_major)
        self.back_prop = back_prop
        self.name = name

    def __call__(self,
                 rewards,
                 bootstrap_value=None,
                 discounts=0.99,
                 sample_weight=1.0):
        """Compute discounted returns.

        Args:
            rewards: tensor of shape [Batch x Time], [Batch x Time x ...]
            bootstrap_value: The discounted n-step value. A tensor of shape [],
                [...], [Batch], [Batch, ...], [Batch x Time], [Batch x Time x ...].
            discounts: tensor of shape [], [...], [Batch], [Batch, ...],
                [Batch x Time], [Batch x Time x ...].
            sample_weight: tensor of shape [], [...], [Batch], [Batch, ...],
                [Batch x Time], [Batch x Time x ...].

        Returns:
            tensor of shape [Batch x Time x ...] or [Time x Batch x ...]
        """
        return discounted_returns(
            rewards,
            bootstrap_value=bootstrap_value,
            discounts=discounts,
            weights=sample_weight,
            time_major=self.time_major,
            back_prop=self.back_prop,
            name=self.name)


class LambdaReturns(Returns):

    def __init__(self, time_major=False, back_prop=False, name=None):
        """Creates a new LambdaReturns

        Args:
            time_major: flag if tensors are time_major.
                Batch and Time are transposed in this doc.
            back_prop: allow back_prop through the calculation.
            name: optional op name.
        """
        super().__init__(time_major)
        self.back_prop = back_prop
        self.name = name

    def __call__(self,
                 rewards,
                 values,
                 bootstrap_value=None,
                 discounts=.99,
                 lambdas=0.95,
                 sample_weight=1.0):
        """Computes lambda-returns.

        For example, this can be used to compute TD(lambda):

            ```
            td_lambda = lambda_returns(r(t), V(t+1)) - V(t)
            ```

        Args:
            rewards: tensor of shape [Batch x Time], [Batch x Time x ...].
            values: tensor of shape [Batch x Time], [Batch x Time x ...].
            bootstrap_value: tensor of shape [], [Batch], [Batch x ...].
            discounts: tensor of shape [], [...], [Batch], [Batch, ...],
                [Batch x Time], [Batch x Time x ...].
            lambdas: tensor of shape [], [...], [Batch], [Batch, ...],
                [Batch x Time], [Batch x Time x ...].
            sample_weight: tensor of shape [], [...], [Batch], [Batch, ...],
                [Batch x Time], [Batch x Time x ...].

        Returns:
            tensor of shape [Batch x Time] or [Time x Batch]
        """
        return lambda_returns(
            rewards,
            values,
            bootstrap_value=bootstrap_value,
            discounts=discounts,
            lambdas=lambdas,
            weights=sample_weight,
            time_major=self.time_major,
            back_prop=self.back_prop,
            name=self.name)


class VtraceReturns(Returns):

    def __init__(self, time_major=False, back_prop=False, name=None):
        """Creates a new VtraceReturns.

        Args:
            time_major: flag if tensors are time_major.
                Batch and Time are transposed in this doc.
            back_prop: allow back_prop through the calculation.
            name: optional op name.
        """
        super().__init__(time_major)
        self.back_prop = back_prop
        self.name = name

    def __call__(self,
                 rewards,
                 values,
                 log_probs,
                 behavioral_log_probs,
                 bootstrap_value=None,
                 discounts=0.99,
                 sample_weight=1.0):
        """Computes v-trace returns.

        Args:
            rewards: tensor of shape [Batch x Time], [Batch x Time x ...].
            values: tensor of shape [Batch x Time], [Batch x Time x ...].
            log_probs: tensor of shape [Batch x Time], [Batch x Time x ...].
            log_probs_old: tensor of shape [Batch x Time], [Batch x Time x ...].
            bootstrap_value: tensor of shape [], [Batch], [Batch x ...].
            discounts: tensor of shape [], [...], [Batch], [Batch, ...],
                [Batch x Time], [Batch x Time x ...].
            weights: tensor of shape [], [...], [Batch], [Batch, ...],
                [Batch x Time], [Batch x Time x ...].
            time_major: flag if tensors are time_major.
                Batch and Time are transposed in this doc.
            back_prop: allow back_prop through the calculation.
            name: optional op name.

        Returns:
            tensor of shape [Batch x Time] or [Time x Batch]
        """
        return v_trace_returns(
            rewards,
            values,
            log_probs,
            behavioral_log_probs,
            bootstrap_value=bootstrap_value,
            discounts=discounts,
            weights=sample_weight,
            time_major=self.time_major,
            back_prop=self.back_prop,
            name=self.name)


class GeneralizedAdvantageEstimation(object):

    def __init__(self, time_major=False, back_prop=False, name=None):
        """Creates a new GeneralizedAdvantageEstimation

        Args:
            time_major: flag if tensors are time_major.
                Batch and Time are transposed in this doc.
            back_prop: allow back_prop through the calculation.
            name: optional op name.
        """
        self.back_prop = back_prop
        self.time_major = time_major
        self.name = name

    def __call__(self,
                 rewards,
                 values,
                 bootstrap_value=None,
                 discounts=.99,
                 lambdas=0.95,
                 sample_weight=1.0):
        """Computes Generalized Advantage Estimation.

        Args:
            rewards: tensor of shape [Batch x Time], [Batch x Time x ...].
            values: tensor of shape [Batch x Time], [Batch x Time x ...].
            bootstrap_value: tensor of shape [], [Batch], [Batch x ...].
            discounts: tensor of shape [], [...], [Batch], [Batch, ...],
                [Batch x Time], [Batch x Time x ...].
            lambdas: tensor of shape [], [...], [Batch], [Batch, ...],
                [Batch x Time], [Batch x Time x ...].
            sample_weight: tensor of shape [], [...], [Batch], [Batch, ...],
                [Batch x Time], [Batch x Time x ...].

        Returns:
            tensor of shape [Batch x Time] or [Time x Batch]
        """
        return generalized_advantage_estimation(
            rewards,
            values,
            bootstrap_value=bootstrap_value,
            discounts=discounts,
            lambdas=lambdas,
            weights=sample_weight,
            time_major=self.time_major,
            back_prop=self.back_prop,
            name=self.name)
