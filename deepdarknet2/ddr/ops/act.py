from .context import lib
from .base import Operator


class Relu(Operator):
    def forward(self, *args, index):
        self._cache = args[0] < 0
        tmp = lib.copy(args[0])
        tmp[self._cache] = 0
        return tmp
    
    def backward(self, *grads, index):
        tmp = lib.copy(grads[0])
        tmp[self._cache] = 0
        return tmp
    

class Gelu(Operator):
    def forward(self, *args, index):
        self._cache = lib.copy(args[0])
        return (1 + lib.erf(args[0] / lib.sqrt(2))) * args[0] / 2
    
    def backward(self, *grads, index):
        x = self._cache
        gain = ((1+lib.erf(x/lib.sqrt(2)))/2 
                + x*lib.exp(-x**2/2)/lib.sqrt(2*lib.pi))
        return gain * grads[0]


class Softplus(Operator):
    def forward(self, *args, index):
        self._cache = args[0]
        return lib.log(1 + lib.exp(args[0]))

    def backward(self, *grads, index):
        return grads[0] / (1 + lib.exp(-self._cache))


class Sigmoid(Operator):
    def forward(self, *args, index):
        self._cache = 1 / (1 + lib.exp(-args[0]))
        return lib.copy(self._cache)

    def backward(self, *grads, index):
        return grads[0] * self._cache * (1 - self._cache)


class Softmax(Operator):
    def __init__(self, feature_axis):
        self._axis = feature_axis

    def forward(self, *args, index):
        tmp = lib.exp(args[0])
        tmp /= lib.sum(tmp, axis=self._axis, keepdims=True)
        self._cache = lib.copy(tmp)
        return tmp

    def backward(self, *grads, index):
        tmp = lib.sum(grads[0] * self._cache, self._axis, keepdims=True)
        return self._cache * (grads[0] - tmp)
