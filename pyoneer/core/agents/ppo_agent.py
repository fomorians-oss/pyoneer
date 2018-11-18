import tensorflow as tf

from pyoneer.core.losses import ppo_loss
from pyoneer.core.advantages import discounted_lambda
from pyoneer.core.normalizers import weighted


class PPOAgent(tf.keras.Model):

    def __init__(self,
                 policy,
                 behavioral_policy,
                 value,
                 entropy_scale=.2,
                 discount=.999,
                 lam=1.,
                 epsilon=.2):
        super(PPOAgent, self).__init__()
        self.policy = policy
        self.behavioral_policy = behavioral_policy
        self.value = value
        self.entropy_scale = entropy_scale
        self.discount = discount
        self.lam = lam
        self.epsilon = epsilon

    def call(self, states, exploring=False, training=False, reset_state=True):
        dist = self.behavioral_policy(states, exploring=exploring, training=training, reset_state=reset_state)
        output = tf.cond(exploring, dist.sample, dist.mean)
        return output

    def compute_loss(self, states, actions, rewards, returns, weights, global_step=0, record_summaries=True):
        policy = self.policy(states, training=True)
        behavioral_policy = self.behavioral_policy(states)
        values = tf.squeeze(self.value(states, training=True), axis=-1)

        advantages = discounted_lambda.discounted_lambda_advantages(
            rewards, values, 
            discount=self.discount, 
            lam=self.lam, 
            weights=weights)

        advantages = tf.stop_gradient(weighted.weighted_normalize(advantages, weights))

        policy_loss, value_loss, entropy_loss = ppo_loss.ppo_loss(
            ratio=tf.exp(policy.log_prob(actions) - tf.stop_gradient(behavioral_policy.log_prob(actions))), 
            values=values, 
            returns=returns, 
            advantages=advantages, 
            entropy=policy.entropy(), 
            weights=weights,
            epsilon=self.epsilon, 
            entropy_scale=self.entropy_scale)

        if record_summaries:
            with tf.contrib.summary.always_record_summaries():
                global_step = tf.cast(global_step, tf.int64)
                tf.contrib.summary.scalar('policy_loss', policy_loss, step=global_step)
                tf.contrib.summary.scalar('value_loss', value_loss, step=global_step)
                tf.contrib.summary.scalar('entropy_loss', entropy_loss, step=global_step)
        return policy_loss, entropy_loss

