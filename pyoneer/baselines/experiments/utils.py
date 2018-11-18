import tensorflow as tf


def copy(src):
    return [tf.identity(s) for s in src]


def assign_list(dst, src):
    for tv, fv in zip(dst, src):
        tv.assign(fv)


def sample_trajectories(agent, exploring=False):
    def action_fn(step, state, action, reward, done, is_initial_state):
        action = agent(
            tf.expand_dims(state, axis=1),
            exploring=exploring,
            reset_state=is_initial_state)
        return tf.squeeze(action, axis=1)
    return action_fn