from .context import lib


def clip_grad_nan(grad_vector, grad_norm=0.1):
    if lib.nonzero(lib.isnan(grad_vector))[0].size > 0:
        grad_vector = lib.random.randn(grad_vector.size, 1)
        grad_vector /= lib.sqrt(lib.sum(grad_vector ** 2))
        grad_vector *= grad_norm
    return grad_vector


# def clip_grad_norm(grad_vector, grad_norm=.1):
#     if lib.nonzero(lib.isnan(grad_vector))[0].size > 0:
#         grad_vector = lib.random.randn(grad_vector.size, 1)
#         grad_vector /= lib.sqrt(lib.sum(grad_vector ** 2))
#         grad_vector *= grad_norm
#     elif norm := lib.sqrt(lib.sum(grad_vector ** 2)) > grad_norm:
#         grad_vector *= grad_norm / norm
#     return grad_vector


def label_smoothing(label, eps, feature_axis):
    k = lib.maximum(2, lib.shape(label)[feature_axis])
    return (label - eps) / (label * k + 1 - k)


def convert_to_onehot(labels: lib.ndarray, label_axis: int, nlabels: int, 
        start: int = 0, dtype=lib.GLOBAL_DTYPE):
    """convert the labels to onehot encoding

    Parameters
    ----------
    labels : the original label numpy 2d array
        one axis is the sample axis and the other axis is the
        label axis
    label_axis : to point out which axis is the label axis
    nlabels : how many labels are there in the array
    """
    assert isinstance(label_axis, int)
    assert isinstance(nlabels, int)
    assert isinstance(labels, lib.ndarray)
    assert labels.ndim == 2 and labels.shape[label_axis] == 1
    
    temp = lib.arange(nlabels) + start
    if label_axis == 0:
        temp = lib.reshape(temp, (-1, 1))
    elif label_axis == 1:
        temp = lib.reshape(temp, (1, -1))

    return (labels == temp).astype(dtype)
