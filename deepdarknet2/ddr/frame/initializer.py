from abc import abstractmethod
from .context import lib


class Initializer():
    pass


class Gaussian(Initializer):
    """Gaussian distribution initializer
    It usually used for kernels.
    """
    def __init__(self, mean=0, std=.001):
        self._mean = mean
        self._std = std

    def sample(self, shape):
        return lib.random.randn(*shape).astype(lib.GLOBAL_DTYPE) * self._std + self._mean


class Constant(Initializer):
    """Constant initializer
    It usually used for biases which is activated by ReLU.
    """
    def __init__(self, c=.1):
        self._c = c

    def sample(self, shape):
        return lib.tile(self._c, shape).astype(lib.GLOBAL_DTYPE)


class Uniform(Initializer):
    """Uniform initializer"""
    def __init__(self, a=0, b=1):
        self._intv = b - a
        self._off = a

    def sample(self, shape):
        return lib.random.rand(*shape).astype(lib.GLOBAL_DTYPE) * self._intv + self._off
