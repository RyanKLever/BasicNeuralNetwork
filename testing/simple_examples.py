import matplotlib.pylab as plt
import numpy as np


# ------------------- Simple Idea 1 ------------------
# This example shows the output of the sigmoid
# activation function
# x = np.arange(-8, 8, 0.1)
# f = 1 / (1 + np.exp(-x))
# plt.plot(x, f)
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.show()


# ------------------- Simple Idea 2 ------------------
# This example shows how the weights affect the slope
# of the output of the sigmoid activation function
# x = np.arange(-8, 8, 0.1)
# w1 = 0.5
# w2 = 1.0
# w3 = 2.0
# l1 = 'w = ' + str(w1)
# l2 = 'w = ' + str(w2)
# l3 = 'w = ' + str(w3)
# for w, l in [(w1, l1), (w2, l2), (w3, l3)]:
#     f = 1 / (1 + np.exp(-x * w))
#     plt.plot(x, f, label=l)
# plt.xlabel('x')
# plt.ylabel('h_w(x)')
# plt.legend(loc=2)
# plt.show()


# ------------------- Simple Idea 3 ------------------
# This example shows the effect that bias can have
# on the sigmoid activation function
# x = np.arange(-8, 8, 0.1)
# w = 5.0
# b1 = -8.0
# b2 = 0.0
# b3 = 8.0
# l1 = 'b = ' + str(b1)
# l2 = 'b = ' + str(b2)
# l3 = 'b = ' + str(b3)
# for b, l in [(b1, l1), (b2, l2), (b3, l3)]:
#     f = 1 / (1 + np.exp(-(x * w+b)))
#     plt.plot(x, f, label=l)
# plt.xlabel('x')
# plt.ylabel('h_wb(x)')
# plt.legend(loc=2)
# plt.show()


# ------------------- Simple Idea 4 ------------------
# def f(x):
#     return 1 / (1 + np.exp(-x))
#
#
# def simple_looped_nn_calc(n_layers, x, w, b):
#     h = []
#     for layer in range(n_layers-1):
#         # Setup the input array which the weights will be multiplied by for
#         # each layer. If it's the first layer, the input array will be the x
#         # input vector. If it's not the first layer, the input to the next
#         # layer will be the output of the previous layer.
#         if layer == 0:
#             node_in = x
#         else:
#             node_in = h
#         # Setup the output array for the nodes in layer l + 1
#         h = np.zeros((w[layer].shape[0],))
#         # loop through the rows of the weight array
#         for i in range(w[layer].shape[0]):
#             # setup the sum inside the activation function
#             f_sum = 0
#             # loop through the columns of the weight array
#             for j in range(w[layer].shape[1]):
#                 f_sum += w[layer][i][j] * node_in[j]
#             # add the bias
#             f_sum += b[layer][i]
#             # finally use the activation function to calculate the
#             # i-th output i.e. h1, h2, h3
#             h[i] = f(f_sum)
#     return h
#
#
# def matrix_feed_forward_calc(n_layers, x, w, b):
#     h = []
#     for layer in range(n_layers-1):
#         if layer == 0:
#             node_in = x
#         else:
#             node_in = h
#         z = w[layer].dot(node_in) + b[layer]
#         h = f(z)
#     return h
#
#
# w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])
# w2 = np.zeros((1, 3))
# w2[0, :] = np.array([0.5, 0.5, 0.5])
# b1 = np.array([0.8, 0.8, 0.8])
# b2 = np.array([0.2])
# w = [w1, w2]
# b = [b1, b2]
# # a dummy x input vector
# x = [1.5, 2.0, 3.0]
#
# print(simple_looped_nn_calc(3, x, w, b))
# # print(%timeit simple_looped_nn_calc(3, x, w, b))
# print(matrix_feed_forward_calc(3, x, w, b))


# ------------------- Simple Idea 5 ------------------
# Consider the equation f(x) = x^4 - 3x^3 + 2
# f_prime(x) = 4x^3 - 9x^2
# We can get the gradient of the function by using
# this little script below

def df(x):
    return 4 * x**3 - 9 * x**2


x_old = 0
x_new = 6
gamma = 0.01
precision = 0.00001

while abs(x_new - x_old) > precision:
    x_old = x_new
    x_new += -gamma * df(x_old)

print("The local minimum occurrs at %f" % x_new)















