from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

def funk_svd_solve(matrix, k, lr, max_epochs=200, l2_scale=0,
                   batch_size=None, tol=1e-6, n_epochs_no_change=10):
    """Factorizes input `matrix` of size `[M, N]` into two matrices: `x` of
    size `[M, k]`, and `y` of size `[k, N]`. Factorization is done through SGD.

    As originally proposed by Simon Funk: https://sifter.org/simon/journal/20061211.html

    Args:
        matrix: 2D `Tensor` of shape `[M, N]`. Should be a `float`.
        k: Dimensionality of the two factorized matrices, as an `int`.
        lr: Learning rate. Can be a callable that takes no inputs and returns
            the learning rate to use.
        max_epochs: maximum number of epochs over the input `matrix`. If
            `tol` is not `None` (in which case it should be a `float`) and the
            factorization loss has not improved for `n_epochs_no_change`
            epochs, then training stops before `max_epochs` is reached.
        l2_scale: L2 penalty (regularization term) parameter as a `float`.
        batch_size: number of rows of `matrix` to train on each step. Set to
            `None` if you want to feed the entire matrix all at once each step
            (row shuffling will still be performed).
        tol: Tolerance for the optimization. When the factorization loss has
            not improved by at least `tol` for `n_epochs_no_change` epochs,
            then training automatically stops. Set to `None` if you don't want
            any early stopping.
        n_epochs_no_change: Maximum number of consecutive epochs allowed where
            the factorization loss does not improve by at least `tol`. Does
            nothing when `tol` is `None`.

    Returns:
        A 2-tuple `(x, y)`, where `x` is a 2D `Tensor` of size `[M, k]`, and
            `y` is a 2D `Tensor` of size `[k, N]`.
    """
    M, N = matrix.shape.as_list()
    x = tf.random.uniform(
        (M, k), minval=-np.sqrt(6/(M+k)), maxval=np.sqrt(6/(M+k)))
    y = tf.random.uniform(
        (k, N), minval=-np.sqrt(6/(N+k)), maxval=np.sqrt(6/(N+k)))

    for e in range(max_epochs):
        products = x @ y
        err = matrix - products

        x_update = (err @ tf.transpose(y)) / (N * k)
        y_update = (tf.transpose(x) @ err) / (M * k)
        
        _lr = lr() if callable(lr) else lr
        x += _lr * (x_update - l2_scale * x)
        y += _lr * (y_update - l2_scale * y)
    
    return x, y
