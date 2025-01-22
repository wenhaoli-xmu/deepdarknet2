import numpy as np
from deepdarknet.conv import Padding, Conv
from deepdarknet.meta import Sum
from deepdarknet.graph import Variable, ComputationalGraph
from deepdarknet.optimizer import CostFunction, optimizer

a = np.random.randn(27).reshape(1, 3, 3, 3)
c = np.random.randn(270).reshape(3, 3, 3, 10)

A = Variable('A', (1, 3, 3, 3), need_grad=True)
B = Variable('B', (1, 3, 7, 7), Padding(2, 2))
C = Variable('C', (3, 3, 3, 10), need_grad=True)
D = Variable('D', (1, 10, 5, 5), Conv())
E = Variable('E', (1, 1), Sum())

g = ComputationalGraph()
g.insert_node(A, B, C, D, E)
g.insert_edge(A, B)
g.insert_edge(B, D)
g.insert_edge(C, D)
g.insert_edge(D, E)
g.initialize(store_leaves=True, debug_info=True)

class Network(CostFunction):
    def value(self, params):
        A.feed(params[:a.size])
        C.feed(params[a.size:])
        g.forward()
        return float(E._value)

    def gradient(self, params):
        A.feed(params[:a.size])
        C.feed(params[a.size:])
        g.forward()
        g.backward()
        grads = (A._grad.reshape(-1, 1), 
                 C._grad.reshape(-1, 1))
        return np.vstack(grads)

    def hessian(self, params):
        raise NotImplementedError

    def after_gradient_check_callback(self, params, compare):
        print(compare)
        input('...')

adam = optimizer(Network(), check_grads=True)
adam.optimize(np.random.randn(a.size + c.size).reshape(-1, 1))

