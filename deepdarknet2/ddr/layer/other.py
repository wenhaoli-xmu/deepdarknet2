from .context import (
    Variable, initializer,
    softmax, cross_entropy, mse)

from .layer import Layer


class SoftmaxOutputLayer(Layer):
    def __init__(self, feature_axis: int, prevent_overflow=True):
        self.feature = Variable(name='score')
        self.result = softmax(self.feature, feature_axis=feature_axis,
                              prevent_overflow=prevent_overflow)
        self.result.name = 'prob'
        super().__init__(input_set=(self.feature,),
                         output_set=(self.result,),
                         extra_output_set=(self.result,))


class CrossEntropyLayer(Layer):
    def __init__(self, feature_axis: int, is_reduced: bool = False):
        self.prob = Variable(name='ce.predict')
        self.ref = Variable(name='ce.label')
        self.cost = cross_entropy(self.ref, self.prob, 
            feature_axis=feature_axis, is_reduced=is_reduced)
        self.cost.name = 'ce.loss'

        super().__init__(
            input_set=(self.prob,),
            output_set=(self.cost,),
            extra_input_set=(self.ref,))


class MSELayer(Layer):
    def __init__(self, sample_axis: int):
        self.predict = Variable(name='mse.predict')
        self.refer = Variable(name='mse.label')
        self.cost = mse(self.predict, self.refer, sample_axis=sample_axis)
        self.cost.name = 'mse.loss'

        super().__init__(
            input_set=(self.predict,),
            output_set=(self.cost,),
            extra_input_set=(self.refer,))
