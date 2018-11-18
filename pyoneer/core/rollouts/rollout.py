import collections


class Rollout(collections.namedtuple(
        'Rollout', ['states', 'actions', 'rewards', 'weights'])):
    """Holder for states, actions, rewards, and weights."""
    pass