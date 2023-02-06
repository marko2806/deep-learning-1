import data
import numpy as np
import matplotlib.pyplot as plt

param_niter = 100000
param_delta = 0.01
param_lambda = 0.001


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def stable_softmax(x):
    max = np.tile(np.max(x, axis=1), (x.shape[1], 1))
    exp_x_shifted = np.exp(x - np.transpose(max))
    sum = np.sum(exp_x_shifted, axis=1)
    tiled = np.tile(sum.transpose(), (x.shape[1], 1))
    #print("Tiled")
    #print(tiled)
    probs = exp_x_shifted / np.transpose(tiled)
    #print("Probs")
    #print(probs)
    return probs


def fcann2_classify(X, weights, biases):
    input = X
    h = None
    for i in range(0, len(weights)):
        s = input @ weights[i] + biases[i]
        if i != (len(weights) - 1):
            h = relu(s)
        else:
            h = stable_softmax(s)
        input = h
    return h.argmax(axis=1)

# X -> dimenzija n_items x n_components
def fcann2_train(X, Y_, n_hidden_layers, n_classes):
    weights = [np.random.randn(X.shape[1], n_hidden_layers), np.random.randn(n_hidden_layers, n_classes)]
    biases = [np.random.randn(n_hidden_layers), np.random.randn(n_classes)]
    Yoh_ = data.one_hot_encode(Y_, n_classes)
    for i in range(param_niter):
        s_1 = X @ weights[0] + biases[0]
        h_1 = relu(s_1)
        s_2 = h_1 @ weights[1] + biases[1]
        output = stable_softmax(s_2)

        g_s2 = (output - Yoh_)
        g_w2 = np.transpose(g_s2) @ h_1
        g_b2 = np.sum(g_s2, axis=0)
        g_s1 = (output - Yoh_) @ np.diag(np.diagonal(s_2) > 0) @ np.transpose(weights[1])
        g_w1 = np.transpose(g_s1) @ X
        g_b1 = np.sum(g_s1, axis=0)
        loss = (-1 / X.shape[0]) * np.sum(np.log(output) * Yoh_) + param_lambda \
            *(np.linalg.norm(weights[0]) + np.linalg.norm(weights[1]))
        print(f"Iteration {i + 1}: {loss}")

        weights[1] -= (1 / X.shape[0]) * (param_delta * np.transpose(g_w2))
        biases[1] -= (1 / X.shape[0]) * param_delta * np.transpose(g_b2)
        weights[0] -= (1 / X.shape[0]) * param_delta * np.transpose(g_w1)
        biases[0] -= (1 / X.shape[0]) * param_delta * np.transpose(g_b1)

    return weights, biases


if __name__ == '__main__':
    np.random.seed(100)
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    W, b = fcann2_train(X, Y_, 15, 2)

    Y = fcann2_classify(X, W, b)

    data.graph_surface(lambda x: fcann2_classify(x, W, b), [[-10,-10], [10,10]], offset=0.0)
    data.graph_data(X, Y_, Y)
    plt.show()

    print(data.confusion_matrix(Y_, Y, 2))
