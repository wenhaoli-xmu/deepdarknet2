import numpy as np

from deepdarknet.conv import Conv
from deepdarknet.meta import Sum
from deepdarknet.graph import Variable, ComputationalGraph
from deepdarknet.optimizer import optimizer, CostFunction

a = np.arange(150).reshape(3, 2, 5, 5)
b = np.arange(180).reshape(2, 3, 3, 10)

A = Variable('A', (-1, 2, 5, 5), need_grad=True)
B = Variable('B', (2, 3, 3, 10), need_grad=True)
C = Variable('C', (-1, 10, 3, 3), Conv(), need_grad=False)
J = Variable('J', (1, ), Sum())

A.feed(a)
B.feed(b)

g = ComputationalGraph()
g.insert_node(A, B, C, J)
g.insert_edge(A, C)
g.insert_edge(B, C)
g.insert_edge(C, J)

g.initialize(store_leaves=False, debug_info=True)

curve_data = []

class Network(CostFunction):
    def value(self, params):
        A.feed(params[:A._value.size])
        B.feed(params[A._value.size:])

        g.forward()
        return float(J._value)

    def gradient(self, params):
        A.feed(params[:A._value.size])
        B.feed(params[A._value.size:])

        g.forward()
        g.backward()
        
        gradA = A._grad.reshape(-1, 1)
        gradB = B._grad.reshape(-1, 1)
        grads = (gradA, gradB)

        return np.vstack(grads)

    def hessian(self, params):
        raise NotImplementedError

    def after_gradient_check_callback(self, params, compare):
        mse = np.sum((compare[:, 0] - compare[:, 1]) ** 2)
        if mse > 1e-6:
            print(compare)
            input('>>')

network = Network()

adam = optimizer(network, method='adam', max_iter=200,
                 learning_rate=1.0, check_grads=True,)
adam.optimize(np.ones((330, ), dtype=np.float64).reshape(-1, 1) * 0.1)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(curve_data)
plt.show()
