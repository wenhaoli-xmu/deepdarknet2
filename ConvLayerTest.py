import numpy as np
import matplotlib.pyplot as plt

from deepdarknet.graph import ComputationalGraph, Variable
from deepdarknet.optimizer import optimizer, CostFunction
from deepdarknet.meta import Sum
from deepdarknet.layer import ConvLayer

I = Variable('I', (1, 3, 10, 10), need_grad=True)
K = Variable('K', (3, 5, 5, 2), need_grad=True)
B = Variable('B', (1, 2, 1, 1), need_grad=True)
A = Variable('A', (1, 2, 2, 2), ConvLayer(method='same', 
                                           activation='relu', 
                                           maxpooling=6))
J = Variable('J', (1, 1), Sum())

g = ComputationalGraph()
g.insert_node(I, K, B, A, J)
g.insert_edge(I, A)
g.insert_edge(K, A)
g.insert_edge(B, A)
g.insert_edge(A, J)

g.initialize(store_leaves=True, debug_info=True)

i = np.random.randn(*I._placeholder).reshape(-1, 1)
k = np.random.randn(*K._placeholder).reshape(-1, 1)
b = np.random.randn(*B._placeholder).reshape(-1, 1)

class Network(CostFunction):
    def value(self, params):
        I.feed(params[: i.size])
        K.feed(params[i.size: i.size + k.size])
        B.feed(params[i.size + k.size: ])

        g.forward()

        return float(J._value)

    def gradient(self, params):
        I.feed(params[: i.size])
        K.feed(params[i.size: i.size + k.size])
        B.feed(params[i.size + k.size: ])

        g.forward()
        g.backward()

        gradI = I._grad.reshape(-1, 1)
        gradK = K._grad.reshape(-1, 1)
        gradB = B._grad.reshape(-1, 1)

        grads = (gradI, gradK, gradB)
        return np.vstack(grads)

    def hessian(self, params):
        raise NotImplementedError

    def after_gradient_check_callback(self, params, compare):
        print(compare)
        input('..')

optimizer(Network(), 
          method='adam', 
          check_grads=True, 
          #grad_check_seq=np.linspace(0, i.size + k.size + b.size, 30, endpoint=False).tolist()
          ).optimize(np.vstack((i, k, b)))