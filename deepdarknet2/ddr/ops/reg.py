from .context import lib
from .base import Operator


class L1Penalty(Operator):
    def __init__(self, lamb):
        assert isinstance(lamb, (int, float))
        self._lambda = lamb

    def forward(self, *args, index):
        self._cache = args[0]
        tmp = lib.copy(args[0])
        tmp[tmp < 0] *= -1
        return self._lambda * lib.sum(tmp)

    def backward(self, *grads, index):
        tmp = lib.ones_like(self._cache)
        tmp[self._cache < 0] *= -1
        return self._lambda * grads[0] * tmp


class L2Penalty(Operator):
    def __init__(self, lamb):
        assert isinstance(lamb, (int, float))
        self._lambda = lamb

    def forward(self, *args, index):
        self._cache = args[0]
        return self._lambda * lib.sum(args[0] ** 2)

    def backward(self, *grads, index):
        return 2 * self._lambda * self._cache * grads[0]
    

class DropoutElemWise(Operator):
    def __init__(self, keep_prob=0.5):
        self._keep_prob = keep_prob

    def forward(self, *args, index):
        mask = lib.random.rand(*args[0].shape) > self._keep_prob
        self._cache = mask
        args[0][mask] = 0
        return args[0] / self._keep_prob
    
    def backward(self, *grads, index):
        grads[0][self._cache] = 0
        return grads[0] / self._keep_prob


class Dropout2D(Operator):
    def __init__(self, feature_axis, keep_prob=0.5):
        self._feature_axis = feature_axis
        self._keep_prob = keep_prob

    def forward(self, *args, index):
        prev = args[0]

        if len(prev.shape) != 2:
            raise RuntimeError

        mask = lib.random.rand(
            prev.shape[self._feature_axis]) > self._keep_prob
        mask = lib.tile(mask, prev.shape[not self._feature_axis])

        if self._feature_axis == 1:
            mask = lib.reshape(mask, prev.shape)
        elif self._feature_axis == 0:
            mask = lib.reshape(mask, list(reversed(prev.shape))).T
        self._cache = mask

        prev[mask] = 0
        return prev / self._keep_prob

    def backward(self, *grads, index):
        grads[0][self._cache] = 0
        return grads[0] / self._keep_prob
