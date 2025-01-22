from .context import lib, frame
from .base import Operator
from .typical import BatchNorm1D, Dot, Plus
from .act import Relu, Sigmoid
import math


def _im2col_for_conv(image: lib.ndarray, kernel_shape, strides):
    """a complete inplementation of _im2col_for_conv

    Parameters
    ----------
    image : input image or last layer's output feature map.

    kernel_shape : kernel's shape, which usually be a list or a tuple.

    strides : the stride used to traverse the image, a tuple or a list.

    Returns
    -------
    col : unfolded input.
    
    """
    assert (isinstance(image, lib.ndarray)
        and isinstance(kernel_shape, (tuple, list)) 
        and len(image.shape) == 4
        and len(kernel_shape) == 4)

    im, ic, ih, iw = image.shape
    kc, kh, kw, km = kernel_shape
    sh, sw = strides

    return lib.as_strided(
        image, shape=(im, int((ih - kh) / sh) + 1, int((iw - kw) / sw) + 1, kc, kh, kw),
        strides=(image.strides[0],
                 image.strides[2] * sh,
                 image.strides[3] * sw,
                 image.strides[1],
                 image.strides[2],
                 image.strides[3])
        ).reshape(im, (int((ih - kh) / sh) + 1) * (int((iw - kw) / sw) + 1), kh * kw * kc)


def _im2col_for_conv_backprop(image: lib.ndarray, kernel_shape, strides=(1,1)):

    assert (isinstance(image, lib.ndarray)
        and isinstance(kernel_shape, (tuple, list)) 
        and len(image.shape) == 4
        and len(kernel_shape) == 4)

    sh, sw = strides
    im, ic, ih, iw = image.shape
    kc, kh, kw, km = kernel_shape

    return lib.as_strided(
        image, shape = (im, ic, kh, kw, (ih-kh)//sh+1, (iw-kw)//sw+1),
        strides=(image.strides[0],
                 image.strides[1],
                 image.strides[2],
                 image.strides[3],
                 image.strides[2] * sh,
                 image.strides[3] * sw)
        ).reshape(
            im, (ic * kh * kw), ((ih-kh)//sh+1) * ((iw-kw)//sw+1)
            ).transpose(1, 0, 2).reshape(
                (ic * kh * kw), im * ((ih-kh)//sh+1) * ((iw-kw)//sw+1))


def _im2col_for_pooling(feature_map: lib.ndarray, window_shape, strides):
    assert (isinstance(feature_map, lib.ndarray)
        and isinstance(window_shape, (tuple, list))
        and isinstance(strides, (tuple, list)))

    im, ic, ih, iw = feature_map.shape
    kh, kw = window_shape
    sh, sw = strides

    zh, zw = (ih - kh) // sh + 1, (iw - kw) // sw + 1

    return lib.as_strided(
        feature_map, shape=(im, ic, zh, zw, kh, kw),
        strides=(feature_map.strides[0],
                 feature_map.strides[1],
                 feature_map.strides[2] * sh,
                 feature_map.strides[3] * sw,
                 feature_map.strides[2],
                 feature_map.strides[3])
            ).reshape(im * ic * zh * zw, kh * kw)


def _col2im_for_pooling(col: lib.ndarray, feature_map_shape, window_shape, strides):
    assert (isinstance(col, lib.ndarray)
        and isinstance(feature_map_shape, (tuple, list))
        and isinstance(window_shape, (tuple, list))
        and isinstance(strides, (tuple, list)))

    if not strides == window_shape:
        raise NotImplementedError('the strides must be the same '
                                  'as the window_shape, yet.')

    im, ic, ih, iw = feature_map_shape
    kh, kw = window_shape
    sh, sw = strides
    zh, zw = (ih - kh) // sh + 1, (iw - kw) // sw + 1

    col = col.reshape(im, ic, zh, zw, kh, kw)

    return col.transpose(0, 1, 2, 4, 3, 5).reshape(im, ic, zh * kh, zw * kw)


def _ker2col_for_conv(kernel: lib.ndarray):
    """unfold kernels

    Parameters
    ----------
    kernel : the objective kernel.

    Returns
    -------
    col : unfolded version of the input kernel.
    
    """
    assert (isinstance(kernel, lib.ndarray)
        and len(kernel.shape) == 4)

    kc, kh, kw, km = kernel.shape
    return kernel.transpose(3, 0, 1 ,2).reshape(km, kc * kh * kw).T


def _ker2col_for_conv_backprop(gradient: lib.ndarray):
    gm, gc, gh, gw = gradient.shape
    gradient_transposed = gradient.transpose(0, 2, 3, 1)
    return gradient_transposed.reshape(gm * gh * gw, gc)


def _conv(image: lib.ndarray, kernel: lib.ndarray, strides=(1,1)):
    """convolution operation

    Parameters
    ----------
    image : the input image or the last layer's output feature map.

    kernel : convolution kernel.

    Returns
    -------
    feature_map : the convolution's result.
    
    """
    unfolded_image = _im2col_for_conv(image, kernel.shape, strides)
    unfolded_kernel = _ker2col_for_conv(kernel)
    return lib.tensordot(unfolded_image, unfolded_kernel, 1)


def _conv_backward(toward_image: bool = True, toward_kernel: bool = True, **kwargs):
    """backward propagate the gradient

    Parameters
    ----------
    toward_image : a boolean refers to whether to compute the 
        gradient of input image or not.

    toward_kernel : a boolean refers to whether to compute the gradient
        of the kernel or not.

    kernel : input kernel.

    image : input image.

    gradient : the gradient of the feature map of this convolution
        operation.

    Returns
    -------
    iter : an iterator
        If you computed both the gradients of image and the gradient of 
        kernel, then the iterator will contain two elements,

        If you only computed one gradient of either image or kernel,
        then the itertor will just contain one element, which is the 
        result graident.

    Examples
    --------
    >>> image_grad = next(_conv_backward(toward_kernel=False,
    ...     kernel=kernel,
    ...     gradient=gradient))

    >>> image_grad, kernel_grad = tuple(_conv_backward(
    ...     kernel=kernel,
    ...     image=image,
    ...     gradient=gradient))
    """
    gradient = kwargs.get('gradient') # 获取gradient
    sh, sw = kwargs.get('strides')    # 获取strides

    if toward_image is True:
        kernel = kwargs.get('kernel')
        gm, gc, gh, gw = gradient.shape
        im, ic, ih, iw = kwargs.get('image_shape')
        kc, kh, kw, km = kernel.shape
        # 对梯度进行upsample
        gradient = lib.pad(gradient.reshape(gm, gc, gh*gw, 1), 
            ((0,0),(0,0),(0,0),(0,sh*sw-1))
            ).reshape(gm, gc, gh, gw, sh, sw
            ).transpose(0,1,2,4,3,5
            ).reshape(gm, gc, gh*sh, gw*sw)
        # 对梯度进行padding
        gradient_padded = lib.pad(gradient, 
            ((0, 0), (0, 0), 
            (kh-1, ih-sh*gh), (kw-1, iw-sw*gw)))
        # 将内核进行对称操作
        kernel_flipped = kernel[:, ::-1, ::-1, :].transpose(3, 1, 2, 0)
        # 进行im2col操作，将梯度变换为矩阵
        unfolded_gradient_padded = _im2col_for_conv(gradient_padded,
            kernel_flipped.shape, (1,1))
        # 进行ker2col操作，将卷积核变换为矩阵
        unfolded_kernel_flipped = _ker2col_for_conv(kernel_flipped)
        # 进行卷积（张量乘）
        image_grad = lib.tensordot(unfolded_gradient_padded,
            unfolded_kernel_flipped, 1)
        yield image_grad
    
    if toward_kernel is True:
        # 获取参数
        image = kwargs.get('image')
        kernel_shape = kwargs.get('kernel_shape')
        # 将image进行im2col展开
        image_unfolded = _im2col_for_conv_backprop(image, kernel_shape, 
            strides=(sh,sw))
        # 将梯度进行ker2col展开
        gradient_unfolded = _ker2col_for_conv_backprop(gradient)
        # 将image和梯度进行卷积（张量乘）
        kernel_grad = lib.tensordot(image_unfolded, gradient_unfolded, 1)
        yield kernel_grad


class Padding(Operator):
    def __init__(self, height: int, width: int):
        self.h = height
        self.w = width

    def forward(self, *args, index):
        return lib.pad(args[0], pad_width=((0, 0), (0, 0), 
            (self.h,self.h), (self.w,self.w)))

    def backward(self, *grads, index):
        sh = slice(self.h,-self.h) if self.h > 0 else slice(None)
        sw = slice(self.w,-self.w) if self.w > 0 else slice(None)
        return grads[0][:,:,sh,sw]


class Conv2D(Operator):
    def __init__(self, strides=(1,1), padding=(0,0)):
        self._pad = Padding(*padding)
        self._strides = strides

    def forward(self, *args, index):
        image, kernel = args
        sh, sw = self._strides
        # 首先进行padding
        image_pad = self._pad.forward(image, index=0)
        # 获取图像与卷积核的维度
        im, ic, ih, iw = image_pad.shape
        kc, kh, kw, km = kernel.shape
        # 判断卷积核的通道数与图像通道数是否相符
        if ic != kc:
            raise RuntimeError(f"ic != kc: {ic} != {kc}")
        # 设置一个缓存
        self._cache = (image_pad, kernel)
        # 返回卷积前向传播的结果
        return _conv(image_pad, kernel, strides=self._strides).reshape(
            im, (ih-kh)//sh + 1, (iw-kw)//sw + 1, km
            ).transpose(0, 3, 1, 2)

    def backward(self, *grads, index):
        left, right = self._cache

        if index == 0:
            # 获取图像的维度信息
            im, ic, ih, iw = left.shape
            # 计算图像的梯度
            image_grad = next(
                _conv_backward(toward_kernel=False,
                               image_shape=left.shape,
                               kernel=right,
                               gradient=grads[0],
                               strides=self._strides)
                ).reshape(im, ih, iw, ic).transpose(0, 3, 1, 2)
            # 返回unpadding之后的结果
            return self._pad.backward(image_grad, index=0)
        elif index == 1:
            # 获取卷积核的维度信息
            kc, kh, kw, km = right.shape
            # 计算卷积核的梯度并返回
            return next(
                _conv_backward(toward_image=False,
                               image=left,
                               kernel_shape=right.shape,
                               gradient=grads[0],
                               strides=self._strides)
                ).reshape(kc, kh, kw, km)
        else:
            # 没有匹配的index
            raise RuntimeError
        

class AdaptivePooling(Operator):
    # 直通式操作，原地工作
    def __init__(self, obj_grid: tuple = (1,1), method: str = 'max',):
        self.pooling = None
        self.method = method
        self.obj_grid = obj_grid

    def forward(self, *args, index):
        h, w = args[0].shape[-2:]
        sh = int(math.ceil(h / self.obj_grid[0]))
        sw = int(math.ceil(w / self.obj_grid[1]))
        
        self._cache = [Pooling(window_shape=(sh,sw), strides=(sh,sw), 
                               method=self.method, auto_padding=True)]
        result = self._cache[0].forward(*args, index=index)
        if result.shape[-2:] != self.obj_grid:
            dh, dw = self.obj_grid[0] - result.shape[2], self.obj_grid[1] - result.shape[3]
            self._cache.append((dh, dw))
            result = lib.pad(result, ((0,0),(0,0),(0,dh),(0,dw)))
        return result
    
    def backward(self, *grads, index):
        if len(self._cache) == 1:
            return self._cache[0].backward(*grads, index=index)
        else:
            dh, dw = self._cache[1]
            grad = grads[0][:,:,:-dh,:-dw]
            return self._cache[0].backward(grad, index=index)


class Pooling(Operator):
    # 直通式操作，原地工作
    def __init__(self, window_shape: tuple,
                 strides: tuple, method: str = 'max',
                 auto_padding: bool = False):
        """pooling operation for cnn
        
        Parameters
        ----------
        window_shape : the shape of the pooling window

        strides : the strides that window move
            at present, only pooling with the same strides and the window
            shape is supported.
            i.e. window_shape == strides

        method : the algorithm of pooling, max pooling supported only.

        auto_padding : whether to apply padding
            if pass 'True' and the feature map's height (width) can't be 
            divide by the height stride (width stride) exactly, then 
            apply zero padding automatically to make it be divided exactly.
        """
        assert (isinstance(window_shape, (tuple, list))
            and isinstance(strides, (tuple, list)))

        if method not in ('max', 'avg'):
            raise NotImplementedError(f'pooling method {method} '
                                      f'is not developed yet.')

        self._method = method
        self._window_shape = window_shape
        self._strides = strides
        self._auto_padding = auto_padding

    def forward(self, *args, index):
        feature_map = args[0]

        im, ic, ih, iw = args[0].shape
        kh, kw = self._window_shape
        sh, sw = self._strides
        zh, zw = (ih - kh) // sh + 1, (iw - kw) // sw + 1

        self._auto_padded = False

        cond1 = self._auto_padding
        cond2 = (ih - kh) % sh != 0 or (iw - kw) % sw != 0

        if cond1 and cond2:
            dh, dw = (zh + 1) * sh - ih, (zw + 1) * sw - iw
            feature_map_pad = lib.pad(feature_map, ((0, 0), (0, 0), (0, dh), (0, dw)))
            ih += dh; iw += dw; zh += 1; zw += 1
            self._auto_padded = True
            self._delta_shapes = (dh, dw)
        else:
            feature_map_pad = feature_map
            self._delta_shapes = tuple()

        self._cache = (im, ic, ih, iw, kh, kw, sh, sw, zh, zw)
        self._cache += self._delta_shapes

        feature_map_unfolded = _im2col_for_pooling(feature_map_pad,
            self._window_shape, self._strides)

        if self._method == 'max':
            self._indices = lib.argmax(feature_map_unfolded, axis=1)
            return lib.max(feature_map_unfolded, axis=1).reshape(
                im, ic, zh, zw)
        elif self._method == 'avg':
            return lib.mean(feature_map_unfolded, axis=1).reshape(
                im, ic, zh, zw)

    def backward(self, *grads, index):
        im, ic, ih, iw, kh, kw, sh, sw, zh, zw = self._cache[:10]

        if self._method == 'max':
            tmp = lib.tile(lib.arange(kh * kw), im * ic * zh * zw
                ).reshape(im * ic * zh * zw, kh * kw)

            tmp = (tmp == self._indices.reshape(-1, 1)).astype(lib.GLOBAL_DTYPE)
            tmp *= grads[0].reshape(-1, 1)
        elif self._method == 'avg':
            tmp = lib.tile(grads[0] / kh / kw, (1, kh * kw))

        tmp = _col2im_for_pooling(tmp, (im, ic, ih, iw), (kh, kw), (sh, sw))
        gh, gw = tmp.shape[2:]
        dh, dw = ih - gh, iw - gw
        tmp = lib.pad(tmp, ((0, 0), (0, 0), (0, dh), (0, dw)))

        if self._auto_padding and self._auto_padded:
            dh, dw = self._cache[-2:]
            tmp = tmp[:, :, :-dh, :-dw]

        return tmp
