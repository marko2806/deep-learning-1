import numpy as np
import matplotlib.pyplot as plt


class Random2DGaussian:
    minx = 0
    maxx = 10
    miny = 0
    maxy = 10

    def __init__(self):
        x_range, y_range = (self.maxx - self.minx, self.maxy - self.miny)
        self.mean = np.random.sample(2) * (x_range, y_range) + (self.minx, self.miny)
        eigvals = (np.random.random_sample(2) * ((self.maxx - self.minx) / 5, (self.maxy - self.miny) / 5)) ** 2
        D = np.diag(eigvals)
        theta = 2 * np.pi * np.random.sample()
        R = [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]]
        self.Sigma = np.dot(np.dot(np.transpose(R), D), R)

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mean, self.Sigma, n)

def myDummyDecision(X):
    scores = X[:, 0] + X[:, 1] - 5
    return scores


def sample_gauss_2d(C, N):
    X = []
    Y_ = []
    for i in range(C):
        X.append(Random2DGaussian())
        Y_.append(i)
    X = np.vstack(G.get_sample(N) for G in X)
    Y_ = np.hstack([y] * N for y in Y_)
    return X, Y_

def eval_perf_binary(Y, Y_true):
    tp = np.sum(np.bitwise_and(Y, Y_true))

    tn = np.sum(np.bitwise_and(np.bitwise_not(Y), np.bitwise_not(Y_true)))
    fp = np.sum(np.bitwise_and(Y, np.bitwise_not(Y_true)))
    fn = np.sum(np.bitwise_and(np.bitwise_not(Y), Y_true))
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return accuracy, precision, recall

def eval_AP(ranked_labels):
    """Recovers AP from ranked labels"""

    n = len(ranked_labels)
    pos = sum(ranked_labels)
    neg = n - pos

    tp = pos
    tn = 0
    fn = 0
    fp = neg

    sumprec = 0
    # IPython.embed()
    for x in ranked_labels:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if x:
            sumprec += precision

        # print (x, tp,tn,fp,fn, precision, recall, sumprec)
        # IPython.embed()

        tp -= x
        fn += x
        fp -= not x
        tn += not x

    return sumprec / pos


def graph_data(X, Y_, Y, special=[]):
    # colors of the datapoint markers
    palette = ([0.5, 0.5, 0.5], [1, 1, 1], [0.2, 0.2, 0.2])
    colors = np.tile([0.0, 0.0, 0.0], (Y_.shape[0], 1))
    for i in range(len(palette)):
        colors[Y_ == i] = palette[i]

    # sizes of the datapoint markers
    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40

    # draw the correctly classified datapoints
    good = (Y_ == Y)
    plt.scatter(X[good, 0], X[good, 1], cmap=colors[good],
                s=sizes[good], marker='o', edgecolors='black')

    # draw the incorrectly classified datapoints
    bad = (Y_ != Y)
    plt.scatter(X[bad, 0], X[bad, 1], cmap=colors[bad],
                s=sizes[bad], marker='s', edgecolors='black')



def graph_surface(fun, rect, offset, width=256, height=256):
    lsw = np.linspace(rect[0][1], rect[1][1], width)
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    x, y = np.meshgrid(lsh, lsw)
    grid = np.stack((x.flatten(), y.flatten()), axis=1)
    results = fun(grid).reshape((width, height))

    # fix the range and offset
    delta = offset if offset else 0
    maxval = max(np.max(results) - delta, - (np.min(results) - delta))

    # draw the surface and the offset
    plt.pcolormesh(x, y, results,
                   vmin=delta - maxval, vmax=delta + maxval)
    if offset != None:
        plt.contour(x, y, results, colors='black', levels=[offset])
'''
if __name__ == "__main__":
    np.random.seed(100)
    G = Random2DGaussian()
    X = G.get_sample(100)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
'''

if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = sample_gauss_2d(2, 100)

    # get the class predictions
    Y = myDummyDecision(X) > 0.5

    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, bbox, offset=0.5)

    # graph the data points
    graph_data(X, Y_, Y)

    # show the results
    plt.show()