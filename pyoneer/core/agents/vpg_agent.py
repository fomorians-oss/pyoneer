import tensorflow as tf

from pyoneer.core.losses import vpg_loss
from pyoneer.core.normalizers import weighted


class VPGAgent(tf.keras.Model):

    def __init__(self,
                 policy,
                 baseline=None,
                 entropy_scale=.2):
        super(VPGAgent, self).__init__()
        self.policy = policy
        self.baseline = baseline
        self.entropy_scale = entropy_scale

    def call(self, states, exploring=False, training=False, reset_state=True):
        dist = self.policy(states, exploring=exploring, training=training, reset_state=reset_state)
        output = tf.cond(exploring, dist.sample, dist.mean)
        return output

    def compute_loss(self, states, actions, returns, weights, global_step=0, record_summaries=True):
        policy = self.policy(states, training=True)

        if self.baseline:
            advantages = returns - self.baseline(states, returns, training=True)
        else:
            advantages = returns

        advantages = tf.stop_gradient(weighted.weighted_normalize(advantages, weights))

        policy_loss, entropy_loss = vpg_loss.vpg_loss(
            policy.log_prob(actions), 
            advantages,
            policy.entropy(), 
            weights=weights,
            entropy_scale=self.entropy_scale)

        if record_summaries:
            with tf.contrib.summary.always_record_summaries():
                global_step = tf.cast(global_step, tf.int64)
                tf.contrib.summary.scalar('policy_loss', policy_loss, step=global_step)
                tf.contrib.summary.scalar('entropy_loss', entropy_loss, step=global_step)
        return policy_loss, entropy_loss

