from .context import lib
from .base import Operator

import numpy as np


class SumAbsoluteError(Operator):
    def __init__(self, sample_axis):
        self.axis = sample_axis

    def forward(self, *args, index):
        self._cache = (args[0] < args[1], args[0].shape)
        return lib.sum(lib.abs(args[0] - args[1]))
    
    def backward(self, *grads, index):
        true_map, shape = self._cache
        if index == 1:
            tmp = lib.full(shape, grads[0], dtype=lib.GLOBAL_DTYPE)
            tmp[~true_map] *= -1
            return tmp
        else:
            tmp = lib.full(shape, grads[0], dtype=lib.GLOBAL_DTYPE)
            tmp[true_map] *= -1
            return tmp


class MeanAbsoluteError(Operator):
    def __init__(self, sample_axis):
        self.axis = sample_axis

    def forward(self, *args, index):
        self._cache = (args[0] < args[1], args[0].shape)
        return lib.mean(lib.abs(args[0] - args[1]))
    
    def backward(self, *grads, index):
        true_map, shape = self._cache
        if index == 1:
            tmp = lib.full(shape, grads[0] / np.prod(shape), dtype=lib.GLOBAL_DTYPE)
            tmp[~true_map] *= -1
            return tmp
        else:
            tmp = lib.full(shape, grads[0] / np.prod(shape), dtype=lib.GLOBAL_DTYPE)
            tmp[true_map] *= -1
            return tmp


class MeanSquaredError(Operator):
    def __init__(self, sample_axis):
        self._axis = sample_axis

    def forward(self, *args, index):
        self._cache = args
        return lib.mean((args[0] - args[1]) ** 2)

    def backward(self, *grads, index):
        if index == 1:
            m = lib.shape(self._cache[0])[self._axis]
            return (self._cache[1] - self._cache[0]) / m * (grads[0] + grads[0])
        else:
            raise RuntimeError


class CrossEntropy(Operator):
    def __init__(self, feature_axis: int, is_reduced: bool = False):
        assert isinstance(feature_axis, int) and feature_axis in (0, 1)
        self._axis = feature_axis
        self._is_reduced = is_reduced

    def forward(self, *args, index):
        left, right = args
        stack = (lib.vstack, lib.hstack)[self._axis]

        if self._is_reduced:
            tmp = lib.sum(left, self._axis, keepdims=True)
            left_value = stack([left, 1 - tmp])
            tmp = lib.sum(right, self._axis, keepdims=True)
            right_value = stack([right, 1 - tmp])
        else:
            left_value = left
            right_value = right

        self._cache = args

        return -lib.sum(left_value * lib.log(right_value + 1e-8), 
                       keepdims=True)

    def backward(self, *grads, index):
        if index == 0:
            raise RuntimeError
        left, right = self._cache
        tmp = - left / (right + 1e-8)
        if self._is_reduced:
            tmp += ((1 - lib.sum(left, self._axis, keepdims=True))
                / (1 - lib.sum(right, self._axis, keepdims=True)))
        return grads[0] * tmp
