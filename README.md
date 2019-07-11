# pyoneer

Tensor utilities, reinforcement learning, and more! Designed to make research easier with low-level abstractions for common operations.

## Usage

For the top-level utilities, import like so:

    import pyoneer as pynr
    pynr.math.rescale(...)

For the larger sub-modules, such as reinforcement learning, we recommend:

    import pyoneer.rl as pyrl
    loss_fn = pyrl.losses.PolicyGradient(...)

In general, the Pyoneer API tries to adhere to the TensorFlow 2.0 API.

### Examples

- [Proximal Policy Optimization with Pyoneer and TF 2.0](https://github.com/fomorians/ppo)

## API

### Activations ([`pynr.activations`](pyoneer/activations))

- `pynr.activations.swish`

### Debugging ([`pynr.debugging`](pyoneer/debugging))

- `pynr.debugging.Stopwatch`

### Distributions ([`pynr.distributions`](pyoneer/distributions))

- `pynr.distributions.MultiCategorical`

### Initializers ([`pynr.initializers`](pyoneer/initializers))

- `pynr.initializers.SoftplusInverse`

### Layers ([`pynr.layers`](pyoneer/layers))

- `pynr.layers.Swish`
- `pynr.layers.OneHotEncoder`
- `pynr.layers.AngleEncoder`
- `pynr.layers.Nest`

### Tensor Manipulation ([`pynr.manip`](pyoneer/manip))

- `pynr.manip.flatten`
- `pynr.manip.batched_index`
- `pynr.manip.pad_or_truncate`
- `pynr.manip.shift`

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

### Moments ([`pynr.moments`](pyoneer/moments))

- `pynr.moments.range_moments`
- `pynr.moments.StaticMoments`
- `pynr.moments.StreamingMoments`
- `pynr.moments.ExponentialMovingMoments`

### Learning Rate Schedules ([`pynr.schedules`](pyoneer/schedules))

- `pynr.schedules.CyclicSchedule`

### Reinforcement Learning ([`pynr.rl`](pyoneer/rl))

Utilities for reinforcement learning.

#### Losses ([`pynr.rl.losses`](pyoneer/rl/losses))

- `pynr.rl.losses.policy_gradient`
- `pynr.rl.losses.policy_entropy`
- `pynr.rl.losses.clipped_policy_gradient`
- `pynr.rl.losses.PolicyGradient`
- `pynr.rl.losses.PolicyEntropy`
- `pynr.rl.losses.ClippedPolicyGradient`

#### Targets ([`pynr.rl.targets`](pyoneer/rl/targets))

- `pynr.rl.targets.DiscountedReturns`
- `pynr.rl.targets.GeneralizedAdvantages`

#### Strategies ([`pynr.rl.strategies`](pyoneer/rl/strategies))

- `pynr.rl.strategies.EpsilonGreedy`
- `pynr.rl.strategies.Mode`
- `pynr.rl.strategies.Sample`

#### Wrappers ([`pynr.rl.wrappers`](pyoneer/rl/wrappers))

- `pynr.rl.wrappers.ObservationCoordinates`
- `pynr.rl.wrappers.ObservationNormalization`
- `pynr.rl.wrappers.Batch`
- `pynr.rl.wrappers.Process`

## Installation

There are a few options for installation:

1. (Recommended) Install with `pipenv`:

        pipenv install fomoro-pyoneer

2. Install locally for development with `pipenv`:

        git clone https://github.com/fomorians/pyoneer.git
        cd pyoneer
        pipenv install
        pipenv shell

## Testing

There are a few options for testing:

1. Run all tests:

        python -m unittest discover -bfp '*_test.py'

2. Run specific tests:

        python -m pyoneer.math.logical_ops_test

## Contributing

File an issue following the `ISSUE_TEMPLATE`. If the issue discussion warrants implementation, then submit a pull request from a branch describing the feature. This will eventually get merged into `master` after a few rounds of code review.
