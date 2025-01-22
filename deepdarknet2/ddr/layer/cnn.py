from .context import (
    relu, plus, dot, batch_norm_2d, conv2d,
    initializer, Variable)

from .layer import Layer


class Conv2DLayer(Layer):
    def __init__(self, input, output, ksize=3, strides=(1,1), padding=(1,1),
                 kernel_initializer=initializer.Gaussian(),
                 bias_initiailizer=initializer.Constant(.1)):
        self.image = Variable()
        self.kernel = Variable(shape=(input,ksize,ksize,output), initializer=kernel_initializer,
                               name='conv2d.kernel')
        self.gain = Variable(shape=(1,output,1,1), initializer=initializer.Constant(1.0),
                             name='conv2d.gain')
        self.bias = Variable(shape=(1,output,1,1), initializer=bias_initiailizer,
                             name='conv2d.bias')
        # 进行相关的操作
        self.result = relu(plus(dot(batch_norm_2d(conv2d(self.image, self.kernel, strides=strides, 
                                                         padding=padding)), self.gain), self.bias),)
        self.result.name = 'conv2d.f_map'

        super().__init__(
            input_set=(self.image,),
            param_set=(self.kernel, self.gain, self.bias),
            output_set=(self.result,))
        

class Conv2D(Layer):
    def __init__(self, input, output, ksize=3, strides=(1,1), padding=(1,1),
                 kernel_initializer=initializer.Gaussian()):
        self.image = Variable(name='image')
        self.kernel = Variable(shape=(input,ksize,ksize,output), initializer=kernel_initializer,
                               name='conv2d.kernel')
        self.result = conv2d(self.image, self.kernel, strides=strides, padding=padding)
        self.result.name = 'conv2d.f_map'
        super().__init__(
            input_set=(self.image, ),
            param_set=(self.kernel, ),
            output_set=(self.result,))
        

class ResLayer(Layer):
    def __init__(self, input, output, ksize=3,
                 kernel_initializer=initializer.Gaussian(0,.001),
                 bias_initializer=initializer.Constant(.1)):
        assert output == input * 2 or output == input
        self.image = Variable()
        self.kernel1 = Variable(shape=(input,ksize,ksize,output), initializer=kernel_initializer,
                                name='res.kernel1')
        self.gain1 = Variable(shape=(1,output,1,1), initializer=initializer.Constant(1.0),
                              name='res.gain1')
        self.bias1 = Variable(shape=(1,output,1,1), initializer=bias_initializer,
                              name='res.bias1')
        self.kernel2 = Variable(shape=(output,ksize,ksize,output), initializer=kernel_initializer,
                                name='res.kernel2')
        self.gain2= Variable(shape=(1,output,1,1), initializer=initializer.Constant(1.0),
                             name='res.gain2')
        self.bias2 = Variable(shape=(1,output,1,1), initializer=bias_initializer,
                              name='res.bias2')
        
        param_set = (self.kernel1, self.gain1, self.bias1, self.kernel2,
                     self.gain2, self.bias2)
        
        # 计算部分
        self.tmp = relu(plus(dot(
            batch_norm_2d(conv2d(self.image, self.kernel1, 
                                 strides=(1,1) if output == input else (2,2), 
                                 padding=(ksize//2,ksize//2))), 
                                 self.gain1), self.bias1))
        self.tmp = plus(dot(
            batch_norm_2d(conv2d(self.tmp, self.kernel2, 
                                 strides=(1,1), padding=(ksize//2,ksize//2))), 
                                 self.gain2), self.bias2)
        if output != input:
            # 如果进行过降采样
            self.kernel0 = Variable(shape=(input,1,1,output), initializer=kernel_initializer,
                                    name='res.ds_kernel')
            self.tmp = plus(self.tmp, conv2d(self.image, self.kernel0, strides=(2,2), padding=(0,0)))
            param_set += (self.kernel0,)
        else:
            # 没有进行降采样
            self.tmp = plus(self.tmp, self.image)
        self.result = relu(self.tmp)
        self.result.name = 'res.f_map'

        super().__init__(
            input_set=(self.image,),
            param_set=param_set, output_set=(self.result,))
