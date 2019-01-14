# pyoneer

Tensor utilities, reinforcement learning, and more! Designed to make research easier with lower-level abstractions for common operations.

## API

In general the API tries to adhere to TensorFlow 2.0's API.

### Initializers ([`pynr.initializers`](pyoneer/initializers))

- `pynr.initializers.SoftplusInverse`

### Layers ([`pynr.layers`](pyoneer/layers))

- `pynr.layers.Normalizer`
- `pynr.layers.OneHotEncoder`
- `pynr.layers.AngleEncoder`
- `pynr.layers.DictFeaturizer`
- `pynr.layers.ListFeaturizer`
- `pynr.layers.VecFeaturizer`

### Math ([`pynr.math`](pyoneer/math))

- `pynr.math.to_radians`
- `pynr.math.to_degrees`
- `pynr.math.to_cartesian`
- `pynr.math.to_polar`
- `pynr.math.RADIANS_TO_DEGREES`
- `pynr.math.DEGREES_TO_RADIANS`
- `pynr.math.isclose`
- `pynr.math.safe_divide`
- `pynr.math.rescale`
- `pynr.math.normalize`
- `pynr.math.denormalize`

### Metrics ([`pynr.metrics`](pyoneer/metrics))

- `pynr.metrics.mape`
- `pynr.metrics.smape`
- `pynr.metrics.MAPE`
- `pynr.metrics.SMAPE`

### Neural Networks ([`pynr.nn`](pyoneer/nn))

- `pynr.nn.swish`
- `pynr.nn.moments_from_range`
- `pynr.nn.StreamingMoments`

### Reinforcement Learning ([`pynr.rl`](pyoneer/rl))

Utilities for reinforcement learning.

#### envs ([`pynr.rl.envs`](pyoneer/rl/envs))

- `pynr.rl.envs.BatchEnv`
- `pynr.rl.envs.ProcessEnv`

#### Losses ([`pynr.rl.losses`](pyoneer/rl/losses))

- `pynr.rl.losses.policy_gradient_loss`
- `pynr.rl.losses.clipped_policy_gradient_loss`

#### Strategies ([`pynr.rl.strategies`](pyoneer/rl/strategies))

- `pynr.rl.strategies.EpsilonGreedyStrategy`
- `pynr.rl.strategies.ModeStrategy`
- `pynr.rl.strategies.SampleStrategy`

### Training ([`pynr.training`](pyoneer/training))

- `pynr.training.CyclicSchedule`

## Installation

There are a few options of installing:

1. Install with `pipenv`:

       pipenv install pyoneer

2. Install with `pip`:

       pip install pyoneer

3. Install locally for development with `pipenv`:

       git clone https://github.com/fomorians/pyoneer.git
       cd pyoneer
       pipenv install
       pipenv shell

4. Install locally for development with `pip`:

       git clone https://github.com/fomorians/pyoneer.git
       cd pyoneer
       pip install -e .

## Usage

```
import pyoneer as pynr
```

### Examples

- [Proximal Policy Optimization](https://github.com/fomorians/ppo)

## Testing

There are a few options for testing:

1. Run all tests:

       python -m unittest discover -p '*_test.py'

2. Run specific tests:

       python -m pyoneer.math.logical_ops_test

## Contributing

File an issue following the `ISSUE_TEMPLATE`, then submit a pull request from a branch describing the feature. This will eventually get merged into `master`.
