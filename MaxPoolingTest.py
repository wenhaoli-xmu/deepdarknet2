import numpy as np
from deepdarknet.conv import Pooling, Conv
from deepdarknet.meta import Sum
from deepdarknet.graph import Variable, ComputationalGraph
from deepdarknet.optimizer import optimizer, CostFunction

a = np.zeros(36).reshape(1, 1, 6, 6)
k = np.zeros(12).reshape(1, 2, 2, 3)

A = Variable('A', (1, 1, 6, 6), need_grad=True)
C = Variable('C', (1, 1, 3, 3), Pooling(window_shape=(2, 2),
                                         strides=(2, 2),
                                         auto_padding=True))
K = Variable('K', (1, 2, 2, 3), need_grad=True)
Z = Variable('Z', (1, 3, 2, 2), Conv())
J = Variable('J', (1, ), Sum())

A.feed(a)

g = ComputationalGraph()
g.insert_node(A, C, K, Z, J)
g.insert_edge(A, C)
g.insert_edge(C, Z)
g.insert_edge(K, Z)
g.insert_edge(Z, J)

g.initialize(store_leaves=False, debug_info=True)

curve_data = []

class Network(CostFunction):
    def value(self, params):
        A.feed(params[:a.size])
        K.feed(params[a.size:])
        g.forward()
        return float(J._value)

    def gradient(self, params):
        A.feed(params[:a.size])
        K.feed(params[a.size:])
        g.forward()
        g.backward()
        grads = (A._grad.reshape(-1, 1), K._grad.reshape(-1, 1))
        return np.vstack(grads)

    def hessian(self, params):
        raise NotImplementedError

    def after_gradient_check_callback(self, params, compare):
        print(compare)
        input('..')

network = Network()

init = np.random.randn(a.size + k.size).reshape(-1, 1)

adam = optimizer(network, method='adam', check_grads=True)
adam.optimize(init)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(curve_data)
plt.show()
