# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import tensorflow as tf
# import tensorflow.contrib.eager as tfe

# class StatelessNormalizer(tf.keras.Model):
#     """
#     Normalizes the inputs by subtracting a mean and dividing by a standard
#     deviation.

#     `StatelessNormalizer` treats the `loc` and `scale` as constant parameters.

#     Args:
#         loc: Mean to use for normalization.
#         scale: Standard deviation to use for normalization.
#         should_center: Center by subtracting the mean.
#         should_scale: Scale by the standard deviation.
#     """

#     def __init__(self,
#                  loc,
#                  scale,
#                  should_center=True,
#                  should_scale=True,
#                  dtype=None,
#                  **kwargs):
#         super(StatelessNormalizer, self).__init__(**kwargs)
#         self.loc = tf.convert_to_tensor(loc, dtype=dtype)
#         self.scale = tf.convert_to_tensor(scale, dtype=dtype)
#         self.should_center = should_center
#         self.should_scale = should_scale

#     def call(self, inputs, weights=1.0):
#         """
#         Normalizes inputs.

#         Args:
#             inputs: A tensor to normalize.
#             weights: Optional weights for the normalization.
#             **kwargs: unused keyword arguments.

#         Returns:
#             A normalized Tensor.
#         """
#         inputs = tf.convert_to_tensor(inputs)
#         if self.should_center:
#             inputs = inputs - self.loc
#         if self.should_scale:
#             inputs = inputs / self.scale
#         inputs = inputs * weights
#         return inputs

#     def inverse(self, inputs, weights=1.0):
#         """
#         Denormalizes inputs.

#         Args:
#             inputs: a possibly normalized Tensor no less than 2-D.
#             weights: mask to apply the operation.
#             **kwargs: unused keyword arguments.

#         Returns:
#             A denormalized Tensor.
#         """
#         inputs = tf.convert_to_tensor(inputs)
#         if self.should_scale:
#             inputs = inputs * self.scale
#         if self.should_center:
#             inputs = inputs + self.loc
#         inputs = inputs * weights
#         return inputs

# class StatefulNormalizer(tf.keras.Model):
#     """
#     Normalizes the inputs by subtracting a mean and dividing by a standard
#     deviation.

#     `StatefulNormalizer` treats the `loc` and `scale` as variable parameters.
#     A subclass must implement the property getter `scale` and method
#     `update_loc`.
#     """

#     def __init__(self, shape, center=True, scale=True, dtype=tf.float32):
#         """Creates a new StatefulNormalizer.

#         Args:
#             shape: The shape of the mean/standard deviation to use for
#                 normalization.
#             center: Center using the mean with this flag.
#             scale: Scale using the standard deviation with this flag.
#         """
#         super(StatefulNormalizer, self).__init__()

#         self.center = center
#         self.scale_ = scale

#         self.count = tfe.Variable(0, dtype=tf.int64, trainable=False)
#         self.loc = tfe.Variable(
#             tf.zeros(shape=shape, dtype=dtype), trainable=False)
#         self.var_sum = tfe.Variable(
#             tf.zeros(shape=shape, dtype=dtype), trainable=False)

#     @property
#     def shape(self):
#         """Returns the event shape of the normalizer.

#         Returns:
#             the TensorShape of the `loc`.
#         """
#         return self.loc.shape

#     @property
#     def scale(self):
#         raise NotImplementedError()

#     def update_loc(self, loc_deltas, inputs, weights=1.0):
#         """Updates the `loc` variable.

#         Args:
#             loc_deltas: the change in `loc`.
#             inputs: a possibly un-normalized Tensor no less than 2-D.
#             weights: mask to apply the operation.

#         Returns:
#             A new `loc`.
#         """
#         raise NotImplementedError()

#     def call(self, inputs, weights=1.0, training=False):
#         """Normalizes an input.

#         Args:
#             inputs: a possibly un-normalized Tensor no less than 2-D.
#             weights: mask to apply the operation.
#             training: bool used for determining if the loc and scale should
#                 be updated.
#             **kwargs: unused keyword arguments.

#         Returns:
#             A normalized Tensor.
#         """
#         inputs = tf.convert_to_tensor(inputs)
#         inputs = tf.cast(inputs, self.loc.dtype)

#         if training:
#             input_shape = inputs.get_shape().as_list()
#             input_dims = len(input_shape)
#             loc_shape = self.loc.get_shape().as_list()
#             loc_dims = len(loc_shape)

#             reduce_axes = list(range(input_dims - loc_dims + 1))
#             reduce_shape = input_shape[:-loc_dims]

#             assert input_dims > loc_dims
#             count = tf.cast(tf.reduce_prod(reduce_shape), tf.float32)
#             self.count.assign_add(count)

#             loc_deltas = tf.reduce_sum(inputs - self.loc, axis=reduce_axes)

#             new_loc = self.update_loc(loc_deltas, inputs, weights)

#             var_deltas = (inputs - self.loc) * (inputs - new_loc)
#             new_var_sum = parray_ops.weighted_mask(
#                 self.var_sum,
#                 self.var_sum + tf.reduce_sum(var_deltas, axis=reduce_axes),
#                 weights)
#             self.loc.assign(new_loc)
#             self.var_sum.assign(new_var_sum)

#         return normalization_ops.select_weighted_normalize(
#             inputs, self.loc, self.scale, self.center, self.scale_, weights)

#     def inverse(self, inputs, weights=1.0):
#         """Un-normalizes an input.

#         Args:
#             inputs: a possibly normalized Tensor no less than 2-D.
#             weights: mask to apply the operation.
#             **kwargs: unused keyword arguments.

#         Returns:
#             An un-normalized Tensor.
#         """
#         inputs = tf.convert_to_tensor(inputs)
#         inputs = tf.cast(inputs, self.loc.dtype)
#         return normalization_ops.select_weighted_denormalize(
#             inputs, self.loc, self.scale, self.center, self.scale_, weights)

# class SampleAverageNormalizer(StatefulNormalizer):
#     """Compute the moving loc and scale according to the input sample count."""

#     @property
#     def scale(self):
#         return tf.sqrt(tf.maximum(self.var_sum / self.count, 0))

#     def update_loc(self, loc_deltas, inputs, weights=1.0):
#         return tf.where(
#             self.count > 1,
#             parray_ops.weighted_mask(
#                 self.loc, self.loc + (loc_deltas / self.count), weights),
#             inputs[tuple([0] * (len(inputs.shape) - 1))])

# class WeightedAverageNormalizer(StatefulNormalizer):
#     """Compute the moving loc and scale according to the supplied alpha scalar."""

#     def __init__(self, shape, alpha, center=True, scale=True,
#                  dtype=tf.float32):
#         """Creates a new WeightedAverageNormalizer.

#         Args:
#             shape: The shape of the mean/standard deviation to use for normalization.
#             alpha: the scalar used to scale the mean updates instead of a sample average.
#             center: Center using the mean with this flag.
#             scale: Scale using the standard deviation with this flag.
#         """
#         super(WeightedAverageNormalizer, self).__init__(
#             shape, center=center, scale=scale, dtype=dtype)
#         self.alpha = alpha

#     @property
#     def scale(self):
#         return tf.sqrt(tf.maximum(self.var_sum * self.alpha, 0))

#     def update_loc(self, loc_deltas, inputs, weights=1.0):
#         return parray_ops.weighted_mask(
#             self.loc, self.loc + (loc_deltas * self.alpha), weights)
