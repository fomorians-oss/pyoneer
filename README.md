# pyoneer

## Installation

There are a few options of installing:

1. Install with `pipenv`:

    pipenv install -e git+git@github.com:fomorians/pyoneer.git#egg=pyoneer

2. Install with `pip`:

    pip install git+git://github.com/fomorians/pyoneer.git#egg=pyoneer

3. Install locally for development with `pipenv`:

    git clone https://github.com/fomorians/pyoneer.git
    cd pyoneer
    pipenv install
    pipenv shell

4. Install locally for development with `pip`:

    git clone https://github.com/fomorians/pyoneer.git
    cd pyoneer
    pip install .

## Testing

There are a few options for testing:

1. Run all tests:

    python -m unittest discover -p '*_test.py'

2. Run specific tests:

    python -m pyoneer.math.logical_ops_test

## Contributing

File an issue following the `ISSUE_TEMPLATE`, then submit a pull request from a branch describing the feature. This will eventually get merged into `master`.
