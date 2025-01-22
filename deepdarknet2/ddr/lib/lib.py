from numpy import (
    sum, mean, dot, log, exp, max, min, argmax, argmin, sqrt, pad, tensordot,
    reshape, transpose, expand_dims, ravel, shape, ndarray, 
    copy, stack, zeros_like, ones_like, tile, vstack, hstack, 
    zeros, ones, arange, full, array, random, concatenate,
    cumsum, tile, abs, nonzero, meshgrid, maximum, minimum, isnan)
from numpy import pi
from scipy.special import erf
from numpy.lib.stride_tricks import as_strided


GLOBAL_DTYPE = 'float32'


def get_cpu_array(array):
    return array


def set_global_dtype(dtype):
    assert dtype in ('float', 'float32', 'float64')
    GLOBAL_DTYPE = dtype
