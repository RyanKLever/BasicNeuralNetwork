from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as r


def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect


def f(x):
    return 1 / (1 + np.exp(-x))


def f_deriv(x):
    return f(x) * (1 - f(x))


def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b


def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b


def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for layer in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x,
        # otherwise, it is the output from the last layer
        if layer == 1:
            node_in = x
        else:
            node_in = h[layer]
        # z^(l+1) = W^(l)*h^(l) + b^(l)
        z[layer + 1] = W[layer].dot(node_in) + b[layer]
        h[layer + 1] = f(z[layer + 1])  # h^(l) = f(z^(l))
    return h, z


def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y-h_out) * f_deriv(z_out)


def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)


def train_nn(nn_structure, X, y, iter_num=3000, alpha=0.25):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt % 1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z
            # values, to be used in the gradient descent step
            h, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for layer in range(len(nn_structure), 0, -1):
                if layer == len(nn_structure):
                    h1 = h[layer]
                    z1 = z[layer]
                    delta[layer] = calculate_out_layer_delta(y[i, :], h1, z1)
                    avg_cost += np.linalg.norm((y[i, :]-h[layer]))
                else:
                    if layer > 1:
                        a1 = delta[layer + 1]
                        a2 = W[layer]
                        a3 = z[layer]
                        delta[layer] = calculate_hidden_delta(a1, a2, a3)
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    npt = np.transpose(h[layer][:, np.newaxis])
                    npa = np.newaxis
                    tri_W[layer] += np.dot(delta[layer + 1][:, npa], npt)
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[layer] += delta[layer + 1]
        # perform the gradient descent step for the weights in each layer
        for layer in range(len(nn_structure) - 1, 0, -1):
            W[layer] += -alpha * (1.0/m * tri_W[layer])
            b[layer] += -alpha * (1.0/m * tri_b[layer])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func


digits = load_digits()
print(digits.data.shape)
plt.gray()
plt.matshow(digits.images[1])
plt.show()

digits.data[0, :]
Out[2]: array(
    [
        0., 0., 5., 13., 9., 1., 0., 0., 0., 0., 13.,
        15., 10., 15., 5., 0., 0., 3., 15., 2., 0., 11.,
        8., 0., 0., 4., 12., 0., 0., 8., 8., 0., 0.,
        5., 8., 0., 0., 9., 8., 0., 0., 4., 11., 0.,
        1., 12., 7., 0., 0., 2., 14., 5., 10., 12., 0.,
        0., 0., 0., 6., 13., 10., 0., 0., 0.
    ]
)

X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)
X[0, :]
Out[3]: array(
    [
        0., -0.33501649, -0.04308102, 0.27407152, -0.66447751,
        -0.84412939, -0.40972392, -0.12502292, -0.05907756, -0.62400926,
        0.4829745, 0.75962245, -0.05842586, 1.12772113, 0.87958306,
        -0.13043338, -0.04462507, 0.11144272, 0.89588044, -0.86066632,
        -1.14964846, 0.51547187, 1.90596347, -0.11422184, -0.03337973,
        0.48648928, 0.46988512, -1.49990136, -1.61406277, 0.07639777,
        1.54181413, -0.04723238, 0., 0.76465553, 0.05263019,
        -1.44763006, -1.73666443, 0.04361588, 1.43955804, 0.,
        -0.06134367, 0.8105536, 0.63011714, -1.12245711, -1.06623158,
        0.66096475, 0.81845076, -0.08874162, -0.03543326, 0.74211893,
        1.15065212, -0.86867056, 0.11012973, 0.53761116, -0.75743581,
        -0.20978513, -0.02359646, -0.29908135, 0.08671869, 0.20829258,
        -0.36677122, -1.14664746, -0.5056698, -0.19600752
    ]
)

y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)
y_train[0], y_v_train[0]
Out[8]: (1, array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]))

nn_structure = [64, 30, 10]

W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train)

plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()
