# pyoneer

## Installation

There are a few options of installing:

```bash
# install with pipenv
pipenv install git+git://github.com/fomorians/pyoneer.git#egg=pyoneer
# or install locally with pipenv
git clone https://github.com/fomorians/pyoneer.git
(cd pyoneer; pipenv install)
# or install with pip
pip install git+git://github.com/fomorians/pyoneer.git#egg=pyoneer
# or install locally with pip
git clone https://github.com/fomorians/pyoneer.git
(cd pyoneer; pip install .)
```

## Testing

There are a few options for testing:

```
# run all tests
python -m unittest discover -p '*_test.py'
# or run an individual test
python -m pyoneer.math.logical_ops_test
```

## Contributing

File an issue following the `ISSUE_TEMPLATE`, then submit a pull request from a branch describing the feature. This will eventually get merged into `master`.