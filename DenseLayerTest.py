import numpy as np

from deepdarknet.layer import DenseLayer
from deepdarknet.meta import Sum
from deepdarknet.graph import Variable, ComputationalGraph
from deepdarknet.optimizer import optimizer, CostFunction

x = np.random.randn(2, 5).reshape(-1, 1)
w = np.random.randn(5, 6).reshape(-1, 1)
b = np.random.randn(1, 6).reshape(-1, 1)

X = Variable('A', (2, 5), need_grad=True)
W = Variable('W', (5, 6), need_grad=True)
B = Variable('B', (1, 6), need_grad=True)
A = Variable('A', (2, 6), DenseLayer(activation='relu'))
J = Variable('J', (1, 1), Sum())

g = ComputationalGraph()
g.insert_node(X, W, B, A, J)
g.insert_edge(X, A)
g.insert_edge(W, A)
g.insert_edge(B, A)
g.insert_edge(A, J)
g.initialize(store_leaves=True, debug_info=True)

class Network(CostFunction):
	def value(self, params):
		X.feed(params[: x.size])
		W.feed(params[x.size: x.size + w.size])
		B.feed(params[x.size + w.size: ])

		g.forward()
		return float(J._value)

	def gradient(self, params):
		X.feed(params[: x.size])
		W.feed(params[x.size: x.size + w.size])
		B.feed(params[x.size + w.size: ])

		g.forward()
		g.backward()

		x_grad = X._grad.reshape(-1, 1)
		w_grad = W._grad.reshape(-1, 1)
		b_grad = B._grad.reshape(-1, 1)

		grads = (x_grad, w_grad, b_grad)

		return np.vstack(grads)

	def hessian(self, params):
		raise NotImplementedError

	def after_gradient_check_callback(self, params, compare):
		print(compare)
		input('..')

optimizer(Network(),
		  method='adam',
		  check_grads=True).optimize(np.vstack((x, w, b)))
