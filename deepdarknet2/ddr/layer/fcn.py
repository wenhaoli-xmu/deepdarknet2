from .context import (
    Variable,
    relu, plus, dot, batch_norm_1d, matmul,
    initializer)

from .layer import Layer


class Linear(Layer):
    def __init__(self, input, output, bias=True,
                 bias_initializer=initializer.Constant(),
                 weight_initializer=initializer.Gaussian()):
        self.feature = Variable()
        self.weight = Variable(shape=(input,output),
                               initializer=weight_initializer,
                               name='linear.weight')
        self.result = matmul(self.feature, self.weight)

        if bias is True:
            b = Variable(shape=(1,output),
                            initializer=bias_initializer,
                            name='linear.bias')
            self.result = plus(self.result, b)

        super().__init__(
            input_set=(self.feature,),
            param_set=(self.weight,b) if bias else (self.weight,),
            output_set=(self.result,))


class DenseLayer(Layer):
    def __init__(self, input, output,
                 weight_initializer=initializer.Gaussian(0,.001),
                 gain_initializer=initializer.Constant(1.0),
                 bias_initializer=initializer.Constant(.1)):
        self.feature = Variable()
        self.weight = Variable(shape=(input,output),
                               initializer=weight_initializer)
        self.gain = Variable(shape=(1,output), 
                             initializer=gain_initializer)
        self.bias = Variable(shape=(1,output),
                             initializer=bias_initializer)
        self.tmp1 = matmul(self.feature, self.weight)
        self.tmp2 = batch_norm_1d(self.tmp1, sample_axis=0)
        self.tmp3 = dot(self.tmp2, self.gain)
        self.tmp4 = plus(self.tmp3, self.bias)
        self.result = relu(self.tmp4)

        super().__init__(
            input_set=(self.feature,),
            param_set=(self.weight, self.gain, self.bias),
            output_set=(self.result,))


