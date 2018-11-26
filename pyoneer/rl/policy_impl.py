from tensorflow.python import keras


class Policy(keras.Model):
    """
    Policy is a parameterized distribution on a behavioral-level.
    """

    @staticmethod
    def from_parameters(*args, **kwargs):
        raise NotImplementedError()