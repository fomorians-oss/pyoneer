from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from pyoneer.losses import get_l2_loss

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

"""

        optimizer: A `tf.train.Optimizer` instance.
        step: Optional `Variable` used by `optimizer` to track optimization
            steps. Incremented by one each update.

    dtype = matrix.dtype
    x = tfe.Variable(
        tf.random.uniform(
            (M, k), minval=-np.sqrt(6/(M+k)), maxval=np.sqrt(6/(M+k))),
        dtype=dtype,
        trainable=True,
    )
    y = tfe.Variable(
        tf.random.uniform(
            (k, N), minval=-np.sqrt(6/(N+k)), maxval=np.sqrt(6/(N+k))),
        dtype=dtype,
        trainable=True,
    )

    data = tf.data.Dataset.from_tensor_slices(matrix)
    data = data.shuffle(M).batch(batch_size or M)
    epochs_w_no_change = 0
    for e in range(max_epochs):
        epoch_losses = []
        for batch_targets in data:
            with tf.GradientTape() as tape:
                tape.watch(x)
                tape.watch(y)

                products = x @ y
                mse_loss = tf.losses.mean_squared_error(
                    labels=batch_targets, predictions=products
                )
                l2_loss = get_l2_loss(l2_scale, [x, y])
                loss = mse_loss + l2_loss
            grads = tape.gradient(loss, [x, y])
            optimizer.apply_gradients(zip(grads, [x, y]), global_step=step)
            epoch_losses.append(loss)
        epoch_loss = float(tf.reduce_mean(epoch_losses))
        if tol is not None:
            if epoch_loss < tol:
                epochs_w_no_change += 1
                if epochs_w_no_change >= n_epochs_no_change:
                    break
            else:
                epochs_w_no_change = 0
    print(epoch_loss)
    return tfe.Variable(x, trainable=False), tfe.Variable(y, trainable=False)
"""
    
