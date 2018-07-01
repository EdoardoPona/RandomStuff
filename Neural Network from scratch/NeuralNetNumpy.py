""" a 3 layer neural network made with numpy only on mnist dataset """
import numpy as np
import random
import matplotlib.pyplot as plt


def one_hot(i):
    out = np.zeros((10))
    out[i] = 1
    return out


def shuffle_list(*ls):
    l =list(zip(*ls))
    random.shuffle(l)
    return zip(*l)


data = open('train.csv', 'r').readlines()
targets = []
del data[0]

print('loading dataset')
for i, line in enumerate(data):
    data[i] = np.array([int(i) for i in line.split(',')][1:])
    targets.append(one_hot(int(line[0])))
print('done')


def get_batch(b_size):
    i = random.randint(0, len(data)-b_size)
    return np.array(data[i:i+b_size]), np.array(targets[i:i+b_size])


def get_accuracy(output, targets, batch_size):
    correct = 0
    for o, t in zip(output, targets):
        a = np.argmax(o, 0)
        b = np.argmax(t, 0)

        if a == b:
            correct += 1

    return correct / batch_size


def init_weight(shape):
    return np.random.randn(shape[0], shape[1]) * np.sqrt(1 / shape[0])


def init_bias(size):
    return np.zeros(size)


def sigmoid(x):
    return 1 / (1 + np.exp(-x.astype(np.float128)))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(X):             # stable version
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def cross_entropy(predictions, targets, epsilon=1e-12):

    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(np.sum(targets*np.log(predictions+1e-9)))/N
    return ce


def mse(y, t):
    return np.sum(1/2*(y - t)**2)


class NeuralNetwork:

    def __init__(self):
        self.w0 = init_weight((784, 512))
        self.b0 = init_bias(512)

        self.w1 = init_weight((512, 128))
        self.b1 = init_bias(128)

        self.w2 = init_weight((128, 10))
        self.b2 = init_bias(10)

        self.B2 = init_weight((10, 128))           # replaces self.w2.T if using feedback training
        self.B1 = init_weight((128, 512))           # replaces self.w1.T


    def forward(self, x):
        self.x = x

        self.a0 = np.dot(x, self.w0)
        self.h0 = sigmoid(self.a0)

        self.a1 = np.dot(self.h0, self.w1)
        self.h1 = sigmoid(self.a1)

        self.a2 = np.dot(self.h1, self.w2)
        self.out = self.a2

        return self.out

    def backwards(self, t):
        e = (self.out - t)    # * d_sigmoid(self.a2)    # when using mse

        dJdw2 = np.dot(self.h1.T, e)
        dJdb2 = np.sum(e, axis=0)

        delta1 = np.dot(e * d_sigmoid(self.a2), self.w2.T) * d_sigmoid(self.a1)
        dJdw1 = np.dot(self.h0.T, delta1)
        dJdb1 = np.sum(delta1, axis=0)

        delta0 = np.dot(delta1, self.w1.T) * d_sigmoid(self.a0)
        dJdw0 = np.dot(self.x.T, delta0)
        dJdb0 = np.sum(delta0, axis=0)

        return (dJdw0, dJdb0), (dJdw1, dJdb1), (dJdw2, dJdb2)


network = NeuralNetwork()

costs = []

batch_size = 50
lr = 1e-3


def train_SGD(iter_num):
    for i in range(int(iter_num)):
        b, t = get_batch(batch_size)

        out = network.forward(b)
        cost = cross_entropy(out, t)

        gradients = network.backwards(t)

        network.w0 -= lr * gradients[0][0]
        network.w1 -= lr * gradients[1][0]
        network.w2 -= lr * gradients[2][0]

        network.b0 -= lr * gradients[0][1]
        network.b1 -= lr * gradients[1][1]
        network.b2 -= lr * gradients[2][1]

        if i % 10 == 0:
            print(i, cost, get_accuracy(out, t, batch_size))
        costs.append(cost)


def train_momentum(iter_num, momentum=0.9):
    updates = [(0, 0) for i in range(3)]     # [w_update, b_update] for every layer
    for i in range(int(iter_num)):
        b, t = get_batch(batch_size)

        out = network.forward(b)
        cost = cross_entropy(out, t)

        gradients = network.backwards(t)

        updates = [(lr*gradients[i][0] + momentum*updates[i][0], lr*gradients[i][1] + momentum*updates[i][1]) for i in range(3)]

        network.w0 -= updates[0][0]
        network.w1 -= updates[1][0]
        network.w2 -= updates[2][0]

        network.b0 -= updates[0][1]
        network.b1 -= updates[1][1]
        network.b2 -= updates[2][1]

        if i % 10 == 0:
            print(i, cost, get_accuracy(out, t, batch_size))
        costs.append(cost)


def train_nesterov(iter_num, momentum=0.9):
    updates = [(0, 0) for i in range(3)]     # [w_update, b_update] for every layer
    for i in range(int(iter_num)):
        b, t = get_batch(batch_size)

        out = network.forward(b)
        cost = cross_entropy(out, t)

        # approximating future parameters
        network.w0 -= momentum * updates[0][0]
        network.w1 -= momentum * updates[1][0]
        network.w2 -= momentum * updates[2][0]

        network.b0 -= momentum * updates[0][1]
        network.b1 -= momentum * updates[1][1]
        network.b2 -= momentum * updates[2][1]

        gradients = network.backwards(t)

        network.w0 -= lr * gradients[0][0]
        network.w1 -= lr * gradients[1][0]
        network.w2 -= lr * gradients[2][0]

        network.b0 -= lr * gradients[0][1]
        network.b1 -= lr * gradients[1][1]
        network.b2 -= lr * gradients[2][1]

        updates = [(lr*gradients[i][0] + momentum*updates[i][0], lr*gradients[i][1] + momentum*updates[i][1]) for i in range(3)]

        if i % 10 == 0:
            print(i, cost, get_accuracy(out, t, batch_size))
        costs.append(cost)

train_nesterov(3e3)
plt.plot(list(range(len(costs))), costs)
plt.show()
