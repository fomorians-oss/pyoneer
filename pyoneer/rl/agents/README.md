# pyoneer.rl.agents

RL algorithms in a simplified Keras-like API.

## Status:
- [x] Advantage Actor-Critic (A2C)
- [x] Proximal Policy Optimization (PPO)
- [x] Vanilla Policy Gradient (VPG)
- [x] V-Trace Actor-Critic (IMPALA)
- [] Trust Region Policy Optimization (TRPO) - _Proposed_
- [] Deterministic Policy Gradient (DPG) - *WIP*
- [] Q - *WIP*
- [] Double Q (DQ) - *WIP*
- [] Q(lambda) - *WIP*
- [] Double Q(lambda) - *WIP*

# TODO:
+ Clean interface for action-value algorithms that are not temporally compatible, such as Q and original DDPG.
    + add compatible interface into `rollout_impl` that supports a simplified `{S, A, R, S'}` tuple instead of `{S, A, R}`.
+ Documentation for individual algorithms, citations and easy interaction.
+ Add more tests for edge-cases other than convergence.
+ Performance improvements?