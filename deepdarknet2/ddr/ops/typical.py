from .context import lib
from .base import Operator
import numpy as np


class UnFlatten(Operator):
    def __init__(self, dim: int, split: tuple):
        """将dim维度进行拆分, 按照split进行拆分"""
        assert isinstance(split, tuple)
        self.dim = dim
        self.split = split

    def forward(self, *args, index):
        self._cache = args[0].shape
        shape = (self._cache[:self.dim]
                 + self.split + self._cache[self.dim+1:])
        return args[0].reshape(*shape)
    
    def backward(self, *grads, index):
        return grads[0].reshape(*self._cache)


class Flatten(Operator):
    def __init__(self, start_dim, end_dim):
        """将[start_dim, end_dim]区间展平"""
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, *args, index):
        self._cache = args[0].shape
        shape = (self._cache[:self.start_dim]
                 + (np.prod(self._cache[self.start_dim: self.end_dim+1]),)
                 + self._cache[self.end_dim+1:])
        return args[0].reshape(*shape)
    
    def backward(self, *grads, index):
        return grads[0].reshape(*self._cache)


class Split(Operator):
    def __init__(self, axis, start, stop, step):
        """单输入单输出"""
        self.axis = axis
        self.start = start
        self.stop = stop
        self.step = step

    def forward(self, *args, index):
        s = [slice(None)] * args[0].ndim
        s[self.axis] = slice(self.start, self.stop, self.step)
        self._cache = args[0].shape[self.axis]
        return args[0][tuple(s)]
    
    def backward(self, *grads, index):
        shape = list(grads[0].shape)
        shape[self.axis] = self._cache
        tmp = lib.zeros(shape, dtype=lib.GLOBAL_DTYPE)
        s = [slice(None)] * grads[0].ndim
        s[self.axis] = slice(self.start, self.stop, self.step)
        tmp[tuple(s)] = grads[0]
        return tmp


class Tile(Operator):
    def __init__(self, rep):
        self.rep = rep

    def forward(self, *args, index):
        self._cache = args[0].shape
        return lib.tile(args[0], self.rep)
    
    def backward(self, *grads, index):
        shape = self._cache
        tmp = grads[0]
        for i, (before, after) in enumerate(zip(shape, self.rep)):
            if before != after:
                s = [slice(None)] * len(shape)
                s[i] = slice(0,before)
                tmp = tmp[tuple(s)]
        return tmp * np.prod(self.rep)


class Transpose(Operator):
    def __init__(self, *axes):
        self._axes1 = axes
        self._axes2 = []
        for i in range(len(axes)):
            self._axes2.append(axes.index(i))

    def forward(self, *args, index):
        return lib.transpose(args[0], self._axes1)
    
    def backward(self, *grads, index):
        return lib.transpose(grads[0], self._axes2)


class Deconcatenate(Operator):
    def __init__(self, axis, plan: tuple):
        """单输入多输出"""
        self._axis = axis
        self._cache = np.cumsum(((0,) + plan))
    
    def forward(self, *args, index):
        s = [slice(None)] * args[0].ndim
        s[self._axis] = slice(self._cache[index], self._cache[index+1])
        return args[0][tuple(s)]

    def backward(self, *grads, index):
        return lib.concatenate(grads, self._axis)


class Concatenate(Operator):
    def __init__(self, axis):
        """多输入单输出"""
        self._axis = axis
    
    def forward(self, *args, index):
        self._cache = [arg.shape[self._axis] for arg in args]
        self._cache.insert(0, 0)
        self._cache = np.cumsum(self._cache)
        return lib.concatenate(args, self._axis)

    def backward(self, *grads, index):
        s = [slice(None)] * grads[0].ndim
        s[self._axis] = slice(self._cache[index], self._cache[index+1])
        return grads[0][tuple(s)]


class Reshape(Operator):
    def __init__(self, *shape):
        self._shape = shape
    
    def forward(self, *args, index):
        self._cache = args[0].shape
        return lib.reshape(args[0], self._shape)

    def backward(self, *grads, index):
        return grads[0].reshape(self._cache)


class Matmul(Operator):
    def forward(self, *args, index):
        self._cache = args
        return args[0] @ args[1]

    def backward(self, *grads, index):
        if index == 0:
            return grads[0] @ self._cache[1].T
        elif index == 1:
            if self._cache[0].ndim == 2 and grads[0].ndim == 2:
                return self._cache[0].T @ grads[0]
            else:
                return (self._cache[0].reshape(-1, self._cache[0].shape[-1]).T 
                        @ grads[0].reshape(-1, grads[0].shape[-1])
                ).reshape(*self._cache[1].shape)
        else:
            raise RuntimeError
        

class Bmm(Operator):
    def forward(self, *args, index):
        self._cache = args
        values = []
        for a, b in zip(args[0], args[1]):
            values.append(a @ b)
        return lib.stack(values)
    
    def backward(self, *grads, index):
        results = []
        if index == 0:
            for g, b in zip(grads[0], self._cache[1]):
                results.append(g @ b.T)
        elif index == 1:
            for g, a in zip(grads[0], self._cache[0]):
                results.append(a.T @ g)
        else:
            raise RuntimeError
        return lib.stack(results)


class Dot(Operator):
    def forward(self, *args, index):
        self._cache = args
        return args[0] * args[1]

    def backward(self, *grads, index):
        ans = lib.copy(grads[0])
        for i in range(len(self._cache)):
            if i != index:
                ans = ans * self._cache[i]
        if grads[0].shape != self._cache[index].shape:
            # 存在广播行为
            for i, (after, before) in (
                    enumerate(zip(grads[0].shape, self._cache[index].shape))):
                if before == 1 and after != 1:
                    ans = lib.sum(ans, axis=i, keepdims=True)
        return ans.reshape(self._cache[index].shape)

class Plus(Operator):
    def forward(self, *args, index):
        ans = lib.zeros_like(args[0])
        self._cache = []  # 缓存了每个双亲的变量尺寸
        for arg in args:
            ans = ans + arg
            self._cache.append(arg.shape)
        return ans

    def backward(self, *grads, index):
        ans = lib.copy(grads[0])
        shape = self._cache[index]
        if grads[0].ndim != len(shape):
            shape = (1,) * (grads[0].ndim - len(shape)) + shape
        if grads[0].shape != shape:
            # 存在广播行为
            for i, (after, before) in (
                    enumerate(zip(grads[0].shape, shape))):
                if before == 1 and after != 1:
                    # 找到被广播的维度，使用sum进行压缩
                    ans = lib.sum(ans, axis=i, keepdims=True)
        return ans.reshape(shape)

class Minus(Operator):
    def forward(self, *args, index):
        self._cache = (args[0].shape, args[1].shape)  # 缓存结点尺寸备用
        return args[0] - args[1]

    def backward(self, *grads, index):
        ans = lib.copy(grads[0])
        shape = self._cache[index]
        if grads[0].shape != shape:
            # 存在广播行为
            for i, (after, before) in (
                    enumerate(zip(grads[0].shape, shape))):
                if before == 1 and after != 1:
                    # 找到被广播的维度，使用sum进行信息整合
                    ans = lib.sum(ans, axis=i, keepdims=True)
        if index == 0:
            return ans.reshape(shape)
        elif index == 1:
            return -ans.reshape(shape)
        else:
            raise RuntimeError

class Divide(Operator):
    def forward(self, *args, index):
        self._cache = (args[0], args[1])
        return args[0] / args[1]

    def backward(self, *grads, index):
        a, b = self._cache
        if index == 0:
            tmp = grads[0] / b  # self._cache[3]是除数
            for i, (ix, iy) in enumerate(zip(a.shape, tmp.shape)):
                if ix == 1 and iy > 1:
                    tmp = lib.sum(tmp, axis=i, keepdims=True)
        elif index == 1:
            tmp = -grads[0]*a / b**2
            for i, (ix, iy) in enumerate(zip(b.shape, tmp.shape)):
                if ix == 1 and iy > 1:
                    tmp = lib.sum(tmp, axis=i, keepdims=True)
        return tmp


class Power(Operator):
    def __init__(self, power):
        self._pwr = power
        super().__init__()

    def forward(self, *args, index):
        self._cache = args[0]
        return args[0] ** self._pwr

    def backward(self, *grads, index):
        return self._pwr * self._cache ** (self._pwr - 1) * grads[0]

class Abs(Operator):
    def forward(self, *args, index):
        self._cache = args[0] < 0
        result = lib.copy(args[0])
        result[self._cache] = -result[self._cache]
        return result

    def backward(self, *grads, index):
        result = lib.copy(grads[0])
        result[self._cache] = -result[self._cache]
        return result

class Neg(Operator):
    def forward(self, *args, index):
        return lib.copy(-args[0])

    def backward(self, *grads, index):
        return -lib.copy(grads[0])


class Max(Operator):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, *args, index):
        inds = lib.argmax(args[0], axis=self.axis, keepdims=True)
        self._cache = (list(args[0].shape), inds)
        return lib.max(args[0], axis=self.axis, keepdims=True)

    def backward(self, *grads, index):
        shape, inds = self._cache
        shape1, shape2 = shape.copy(), shape.copy()
        shape1[not self.axis] = 1
        shape2[self.axis] = 1
        return lib.tile(grads[0], shape1) * (lib.tile(lib.arange(
            shape1[self.axis], dtype='int32').reshape(
            *shape1), shape2) == inds)


class Min(Operator):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, *args, index):
        inds = lib.argmin(args[0], axis=self.axis, keepdims=True)
        self._cache = (list(args[0].shape), inds)
        return lib.min(args[0], axis=self.axis, keepdims=True)

    def backward(self, *grads, index):
        shape, inds = self._cache
        shape1, shape2 = shape.copy(), shape.copy()
        shape1[not self.axis] = 1
        shape2[self.axis] = 1
        return lib.tile(grads[0], shape1) * (lib.tile(lib.arange(
            shape1[self.axis], dtype='int32').reshape(
            *shape1), shape2) == inds)

class Sum(Operator):
    def __init__(self, axis=None):
        self._axis = axis
    
    def forward(self, *args, index):
        shape = list(args[0].shape)
        if self._axis is not None: 
            self._dim = args[0].shape[self._axis]
            shape[self._axis] = 1
        else:
            shape = (1, 1)
        self._cache = args[0].shape
        return lib.reshape(lib.sum(args[0], self._axis), shape)

    def backward(self, *grads, index):
        if self._axis is None:
            return grads[0] * lib.ones(self._cache)
        elif self._axis == 0:
            result = lib.tile(lib.ravel(grads[0]), self._dim)
            return lib.reshape(result, self._cache)
        elif self._axis == 1:
            result = lib.tile(lib.ravel(grads[0]), self._dim)
            shape = list(self._cache)
            shape[0], shape[1] = shape[1], shape[0]
            return lib.reshape(result, shape).T
        
class Exp(Operator):
    def forward(self, *args, index):
        self._cache = args[0]
        return lib.exp(args[0])
    
    def backward(self, *grads, index):
        return grads[0] * lib.exp(self._cache)

class Log(Operator):
    def __init__(self, fix: bool = True):
        assert isinstance(fix, bool)
        self._eps = 1e-8 if fix is True else 0

    def forward(self, *args, index):
        self._cache = args[0]
        return lib.log(args[0] + self._eps)
    
    def backward(self, *grads, index):
        return grads[0] / (self._cache + self._eps)

# class BatchNorm1D(Operator):
#     def __init__(self, sample_axis):
#         assert (isinstance(sample_axis, int) 
#                 and sample_axis in (0, 1))
#         self._axis = sample_axis

#     def forward(self, *args, index):
#         arg = args[0]
#         mu = lib.mean(arg, self._axis, keepdims=True)
#         std = lib.sqrt(lib.mean((arg - mu) ** 2, self._axis, keepdims=True))
#         self._cache = (mu, std, arg)
#         return (arg - mu) / std

#     def backward(self, *grads, index):
#         mu, std, arg = self._cache
#         m = arg.shape[self._axis]

#         a = 1 / std / m
#         b = a / std / std
#         c = b * lib.sum(arg - mu, self._axis, keepdims=True) / m
#         d = b * (arg - mu)
#         e = c - d

#         return (
#             -(mu * e + a) * lib.sum(grad, self._axis, keepdims=True) 
#             + e * lib.sum(grad * arg, self._axis, keepdims=True)
#             + grad / std
#             )

class BatchNorm1D(Operator):
    def __init__(self, sample_axis):
        assert (isinstance(sample_axis, int) 
                and sample_axis in (0, 1))
        self._axis = sample_axis

    def forward(self, *args, index):
        arg = args[0]
        mu = lib.mean(arg, self._axis, keepdims=True)
        std = lib.sqrt(lib.mean((arg - mu) ** 2, self._axis, keepdims=True))
        res = (arg - mu) / std
        self._cache = (arg.shape[self._axis], std, res)
        return res

    def backward(self, *grads, index):
        m, sigma, y = self._cache
        
        coef = -lib.sum(y, axis=self._axis, keepdims=True) / m + y
        tmp1 = lib.sum(grads[0] * y, axis=self._axis, keepdims=True)
        tmp2 = lib.sum(grads[0], axis=self._axis, keepdims=True)

        tmp = coef * tmp1 + tmp2
        return (-tmp / m + grads[0]) / sigma



class BatchNorm2D(Operator):
    def __init__(self):
        self._bn = BatchNorm1D(sample_axis=1)
    
    def forward(self, *args, index):
        feature_map = args[0]
        im, ic, ih, iw = feature_map.shape
        feature_map = feature_map.transpose(1, 0, 2, 3
            ).reshape(ic, im * ih * iw)
        
        return self._bn.forward(feature_map, index=0).reshape(ic, im, ih, iw
            ).transpose(1, 0, 2, 3)

    def backward(self, *grads, index):
        im, ic, ih, iw = grads[0].shape
        tmp = grads[0].transpose(1, 0, 2, 3).reshape(ic, im * ih * iw)

        return self._bn.backward(tmp, index=0).reshape(ic, im, ih, iw
            ).transpose(1, 0, 2, 3)


class LayerNorm(Operator):
    def __init__(self):
        self._bn = BatchNorm1D(sample_axis=1)

    def forward(self, *args, index):
        shape = args[0].shape
        self._cache = shape
        return self._bn.forward(args[0].reshape(-1,np.prod(shape[1:])), 
                                index=0).reshape(shape)
    
    def backward(self, *grads, index):
        shape = self._cache
        return self._bn.backward(grads[0].reshape(-1,np.prod(shape[1:])), 
                                 index=0).reshape(shape)
