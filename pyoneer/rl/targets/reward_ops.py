from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


@tf.function
def n_step_discounted_bootstrap_values(values,
                                       n_step,
                                       seq_lens,
                                       discounts,
                                       time_major=False):
    """Compute n-step discounted bootstrap values.

    Args:
        values: tensor of shape [Batch x Time], [Batch x Time x ...].
        n_step: tensor of shape [], [Batch].
        seq_lens: tensor of shape [], [Batch].
        time_major: flag if tensors are time_major.
            Batch and Time are transposed in this doc.
    """
    batch_axis = int(time_major)
    time_axis = int(not time_major)

    # Check values shape.
    values = tf.convert_to_tensor(values)
    values_rank = values.shape.rank
    assert values_rank >= 2, 'values must be atleast rank 2.'
    values_shape = tf.shape(values)

    # Check n_step shape and tile to [Batch].
    n_step = tf.convert_to_tensor(n_step)
    assert n_step.shape.rank < 2, 'n_step must be less than rank 2.'
    n_step_shape = tf.shape(n_step)
    if n_step.shape.rank == 1:
        tf.debugging.assert_equal(values_shape[batch_axis],
                                  n_step_shape[0])
    else:
        n_step = tf.tile(tf.reshape(n_step, [-1]),
                         [values_shape[batch_axis]])
    # Offset the n_step index to correct for values being values t + 1.
    n_step = n_step - 1

    # Check seq_lens shape and tile to [Batch].
    seq_lens = tf.convert_to_tensor(seq_lens)
    assert seq_lens.shape.rank < 2, 'seq_lens must be rank 0 or 1.'
    seq_lens_shape = tf.shape(seq_lens)
    if seq_lens.shape.rank == 1:
        tf.debugging.assert_equal(values_shape[batch_axis],
                                  seq_lens_shape[0])
    else:
        seq_lens = tf.tile(tf.reshape(seq_lens, [-1]),
                           [values_shape[batch_axis]])

    # Check discounts shape, broadcast to values shape.
    discounts = tf.convert_to_tensor(discounts)
    assert discounts.shape.rank <= values.shape.rank, (
        'discounts rank must be <= values rank.')
    discounts = tf.broadcast_to(discounts, values_shape)

    # Increasing sequence lengths, clipped at n_step.
    max_seq_range = tf.expand_dims(tf.range(tf.reduce_max(seq_lens)), axis=0)
    max_seq_range = tf.tile(max_seq_range, [values_shape[batch_axis], 1])
    seq_range = tf.minimum(max_seq_range, tf.expand_dims(seq_lens, axis=1))
    n_step_range = tf.minimum(seq_range, tf.expand_dims(n_step, axis=1))

    if time_major:
        rank_list = list(range(2, values_rank))
        values = tf.transpose(values, [1, 0] + rank_list)
        discounts = tf.transpose(discounts, [1, 0] + rank_list)

    # Reverse the values to get a normalized index.
    values_rev = tf.reverse_sequence(values, seq_lens,
                                     batch_axis=0, seq_axis=1)

    # Gather the offset bootstrap values.
    seq_index = seq_range - n_step_range
    bootstrap_values_rev = tf.gather(values_rev, seq_index, batch_dims=1)

    # Discount the values.
    bootstrap_values_rev *= (
        discounts ** tf.cast(n_step_range + 1, tf.float32))

    # Reverse the offset bootstrap values back to the original index.
    bootstrap_values = tf.reverse_sequence(bootstrap_values_rev, seq_lens,
                                           seq_axis=1)

    if time_major:
        rank_list = list(range(2, values_rank))
        bootstrap_values = tf.transpose(bootstrap_values, [1, 0] + rank_list)

    return bootstrap_values


@tf.function
def _concat_right(x):
    """Shift x by 1 in the 'time' dimension."""
    return tf.concat([x[1:], tf.zeros_like(x[:1])], axis=0)


@tf.function
def _nstep_returns(args):
    """Compute n-step returns."""
    rewards, n_step, discounts, bootstrap_values = args
    # Compute n_step returns.
    returns = bootstrap_values
    for _ in tf.range(n_step):
        returns += rewards

        # Shift rewards up by one time step.
        rewards_tp1 = _concat_right(rewards)

        # Compute the one step return and set it equal to the next
        # "rewards".
        rewards = discounts * rewards_tp1
    return returns


@tf.function
def n_step_discounted_returns(rewards,
                              n_step,
                              bootstrap_values=None,
                              discounts=0.99,
                              weights=1.0,
                              time_major=False,
                              back_prop=False,
                              name=None):
    """Compute n-step discounted returns.

    Args:
        rewards: tensor of shape [Batch x Time], [Batch x Time x ...].
        n_step: tensor of shape [], [Batch].
        bootstrap_value: The discounted n-step value. A tensor of shape [],
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
    with tf.name_scope(name or 'NStepDiscountedReturns'):
        batch_axis = int(time_major)
        time_axis = int(not time_major)

        # Check rewards shape.
        rewards = tf.convert_to_tensor(rewards)
        rewards_rank = rewards.shape.rank
        assert rewards_rank >= 2, 'rewards must be atleast rank 2.'
        rewards_shape = tf.shape(rewards)

        # Check n_step shape and tile to [Batch].
        n_step = tf.convert_to_tensor(n_step)
        assert n_step.shape.rank < 2, 'n_step must be less than rank 2.'
        n_step_shape = tf.shape(n_step)
        if n_step.shape.rank == 1:
            tf.debugging.assert_equal(rewards_shape[batch_axis],
                                      n_step_shape[0])
        else:
            n_step = tf.tile(tf.reshape(n_step, [-1]),
                             [rewards_shape[batch_axis]])

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
        # we want them to be the same shape as the rewards.
        if bootstrap_values is None:
            bootstrap_values = tf.zeros_like(rewards)
        else:
            bootstrap_values = tf.convert_to_tensor(bootstrap_values)
            bootstrap_values = tf.broadcast_to(bootstrap_values, rewards_shape)

        if time_major:
            rank_list = list(range(2, rewards_rank))
            rewards = tf.transpose(rewards, [1, 0] + rank_list)
            discounts = tf.transpose(discounts, [1, 0] + rank_list)
            bootstrap_values = tf.transpose(bootstrap_values,
                                            [1, 0] + rank_list)

        returns = tf.map_fn(_nstep_returns,
                            (rewards, n_step, discounts, bootstrap_values),
                            dtype=rewards.dtype,
                            back_prop=back_prop)

        if time_major:
            rank_list = list(range(2, rewards_rank))
            returns = tf.transpose(returns, [1, 0] + rank_list)

        returns = returns * weights
        returns = tf.debugging.check_numerics(returns, 'Returns')
        if not back_prop:
            returns = tf.stop_gradient(returns)
        return returns


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
        bootstrap_value: The discounted n-step value. A tensor of shape [],
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
def temporal_difference(returns, values, back_prop=False, name=None):
    """Computes the temporal difference.

    Args:
        returns: tensor of shape [Batch x Time], [Batch x Time x ...]
        values: tensor of shape [Batch x Time], [Batch x Time x ...]
        back_prop: allow back_prop through the calculation.
        name: optional op name.

    Returns:
        tensor of shape [Batch x Time]
    """
    with tf.name_scope(name or 'BaselineAdvantageEstimate'):
        # Check returns shape.
        returns = tf.convert_to_tensor(returns)
        assert returns.shape.rank >= 2, 'returns must be atleast rank 2.'

        # Check values shape.
        values = tf.convert_to_tensor(values)
        assert values.shape.rank >= 2, 'values must be atleast rank 2.'

        assert returns.shape.rank <= values.shape.rank, (
            'values rank must be == returns rank.')

        advantages = returns - values
        advantages = tf.debugging.check_numerics(advantages, 'Advantages')
        if not back_prop:
            advantages = tf.stop_gradient(advantages)
        return advantages


@tf.function
def v_trace_returns(rewards,
                    values,
                    log_probs,
                    log_probs_old,
                    last_value=None,
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
        last_value: tensor of shape [], [Batch], [Batch x ...].
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

        if last_value is None:
            if time_major:
                last_value = tf.zeros_like(values[-1, :])
            else:
                last_value = tf.zeros_like(values[:, -1])
        else:
            last_value = tf.convert_to_tensor(last_value)

        clipped_ratios = tf.minimum(1., tf.exp(log_probs - log_probs_old))

        last_value_t = tf.expand_dims(last_value, axis=time_axis)

        # Shift values to tp1.
        if time_major:
            values_tp1 = values[1:, :]
        else:
            values_tp1 = values[:, 1:]

        # Pad the values with the last value.
        values_tp1 = tf.concat([values_tp1, last_value_t],
                               axis=time_axis)

        advantages = temporal_difference(rewards + discounts * values_tp1,
                                         values, back_prop=back_prop)
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
                                 initializer=tf.zeros_like(last_value),
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
def generalized_advantage_estimate(rewards,
                                   values,
                                   last_value=None,
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
        last_value: tensor of shape [], [Batch], [Batch x ...].
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

        if last_value is None:
            if time_major:
                last_value = tf.zeros_like(values[-1, :])
            else:
                last_value = tf.zeros_like(values[:, -1])
        else:
            last_value = tf.convert_to_tensor(last_value)

        last_value_t = tf.expand_dims(last_value, axis=time_axis)

        # Shift values to tp1.
        if time_major:
            values_tp1 = values[1:, :]
        else:
            values_tp1 = values[:, 1:]

        # Pad the values with the last value.
        values_tp1 = tf.concat([values_tp1, last_value_t],
                               axis=time_axis)

        delta = temporal_difference(rewards + discounts * values_tp1,
                                    values, back_prop=back_prop)

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
            initializer=tf.zeros_like(last_value),
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


class NstepDiscountedReturns(object):

    def __init__(self, time_major=False, back_prop=False, name=None):
        """Creates a new NstepDiscountedReturns

        Args:
            time_major: flag if tensors are time_major.
                Batch and Time are transposed in this doc.
            back_prop: allow back_prop through the calculation.
            name: optional op name.
        """
        self.back_prop = back_prop
        self.time_major = time_major
        self.name = name

    def __call__(self, rewards, n_step, bootstrap_values=None, discounts=0.99, sample_weight=1.0):
        """Compute n-step discounted returns.

        Args:
            rewards: tensor of shape [Batch x Time], [Batch x Time x ...].
            n_step: tensor of shape [], [Batch].
            bootstrap_value: The discounted n-step value. A tensor of shape [],
                [...], [Batch], [Batch, ...], [Batch x Time], [Batch x Time x ...].
            discounts: tensor of shape [], [...], [Batch], [Batch, ...],
                [Batch x Time], [Batch x Time x ...].
            sample_weight: tensor of shape [], [...], [Batch], [Batch, ...],
                [Batch x Time], [Batch x Time x ...].

        Returns:
            tensor of shape [Batch x Time x ...] or [Time x Batch x ...]
        """
        return n_step_discounted_returns(
            rewards,
            n_step,
            bootstrap_values=bootstrap_values,
            discounts=discounts,
            weights=sample_weight,
            time_major=self.time_major,
            back_prop=self.back_prop,
            name=self.name)

    def discounted_bootstrap_values(self, values, n_step, seq_lens, discounts):
        """Compute n-step discounted bootstrap values.

        Args:
            values: tensor of shape [Batch x Time], [Batch x Time x ...].
            n_step: tensor of shape [], [Batch].
            seq_lens: tensor of shape [], [Batch].
            time_major: flag if tensors are time_major.
                Batch and Time are transposed in this doc.
        """
        return n_step_discounted_bootstrap_values(
            values,
            n_step,
            seq_lens,
            discounts,
            time_major=self.time_major)


class DiscountedReturns(object):

    def __init__(self, time_major=False, back_prop=False, name=None):
        """Creates a new DiscountedReturns

        Args:
            time_major: flag if tensors are time_major.
                Batch and Time are transposed in this doc.
            back_prop: allow back_prop through the calculation.
            name: optional op name.
        """
        self.back_prop = back_prop
        self.time_major = time_major
        self.name = name

    def __call__(self, rewards, bootstrap_value=None, discounts=0.99, sample_weight=1.0):
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


class VtraceReturns(object):

    def __init__(self, time_major=False, back_prop=False, name=None):
        """Creates a new VtraceReturns.

        Args:
            time_major: flag if tensors are time_major.
                Batch and Time are transposed in this doc.
            back_prop: allow back_prop through the calculation.
            name: optional op name.
        """
        self.back_prop = back_prop
        self.time_major = time_major
        self.name = name

    def __call__(self, rewards, values, log_probs, behavioral_log_probs, last_value=None, discounts=0.99, sample_weight=1.0):
        """Computes v-trace returns.

        Args:
            rewards: tensor of shape [Batch x Time], [Batch x Time x ...].
            values: tensor of shape [Batch x Time], [Batch x Time x ...].
            log_probs: tensor of shape [Batch x Time], [Batch x Time x ...].
            log_probs_old: tensor of shape [Batch x Time], [Batch x Time x ...].
            last_value: tensor of shape [], [Batch], [Batch x ...].
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
            last_value=last_value,
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

    def __call__(self, rewards, values, last_value=None, discounts=.99, lambdas=0.95, sample_weight=1.0):
        """Computes Generalized Advantage Estimation.

        Args:
            rewards: tensor of shape [Batch x Time], [Batch x Time x ...].
            values: tensor of shape [Batch x Time], [Batch x Time x ...].
            last_value: tensor of shape [], [Batch], [Batch x ...].
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
            last_value=last_value,
            discounts=discounts,
            lambdas=self.lambdas,
            weights=sample_weight,
            time_major=self.time_major,
            back_prop=self.back_prop,
            name=self.name)
