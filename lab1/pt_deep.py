import torch
import torch.nn as nn
import numpy as np
import data
import matplotlib.pyplot as plt


class PTDeep(nn.Module):
    def __init__(self, network_structure = None, activation_function=None):
        super().__init__()
        self.activation_function = activation_function
        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()
        for i in range(len(network_structure) - 1):
            layer_weights = torch.nn.Parameter(torch.tensor(np.random.randn(network_structure[i], network_structure[i + 1])))
            self.weights.append(layer_weights)
            self.biases.append(torch.nn.Parameter(torch.tensor(np.random.randn(network_structure[i + 1]))))

    def forward(self, X):
        input = X
        h = None

        for i in range(0, len(self.weights)):
            w = self.weights[i]
            y = torch.mm(input, w) + self.biases[i]
            if i == (len(self.weights) - 1):
                h = torch.softmax(y, 1)
            else:
                h = self.activation_function()(y)
            input = h
        return h

    def get_loss(self, X, Yoh_):
        output = self.forward(X)
        softmax_y = torch.log(output) * Yoh_
        return -torch.mean(softmax_y)


def count_params(model):
    number_of_parameters = 0
    for name, param in model.named_parameters():
        if len(param.shape) == 2: #weights
            number_of_parameters += param.shape[0] * param.shape[1]
        elif len(param.shape) == 1: #biases
            number_of_parameters += param.shape[0]
        print(f"{name}, dimensions: {param.shape}")
    print(number_of_parameters)

def train(model, X, Yoh_, param_niter, param_delta, param_lambda=0.0, use_adam=False):
    # inicijalizacija optimizatora
    if use_adam:
        optimizer = torch.optim.Adam(list(model.weights.parameters()) + list(model.biases.parameters()), lr=param_delta,
                                weight_decay=param_lambda)
    else:
        optimizer = torch.optim.SGD(list(model.weights.parameters()) + list(model.biases.parameters()), lr=param_delta,
                                weight_decay=param_lambda)

    # petlja učenja
    for i in range(param_niter):
        #model.forward(X)
        loss = model.get_loss(X, Yoh_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Iteration {i + 1}: {loss}")
        #print(f"Biases: {model.biases[2]}")
    return model.weights, model.biases
def eval(model, X):
    output = model.forward(torch.tensor(X))
    return output.detach().numpy()

def confusion_matrix(Y, Y_, nclasses):
    cm = np.zeros((nclasses, nclasses))
    for y, y_ in zip(Y, Y_):
        cm[y,y_] += 1
    return cm

def get_metric_for_classes(confusion_matrix):
    nclasses = confusion_matrix.shape[0]
    precisions = []
    recalls = []
    for i in range(nclasses):
        tp = confusion_matrix[i][i]
        fp = np.sum(confusion_matrix[i, :]) - tp
        fn = np.sum(confusion_matrix[:, i]) - tp
        precisions.append(tp / (tp + fp))
        recalls.append(tp / (tp + fn))
    return precisions, recalls

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    nclasses = 3
    config = [2, nclasses]
    X, Y_ = data.sample_gauss_2d(nclasses, 25)
    Yoh_ = torch.tensor(data.one_hot_encode(Y_, nclasses))

    # definiraj model:
    ptlr = PTDeep(config, torch.nn.ReLU)
    #count_params(ptlr)
    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, torch.tensor(X), Yoh_, 10000, 0.05, 0)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)

    Y_out = np.argmax(probs, axis=1)
    print(Y_)
    print(Y_out)

    # ispiši performansu (preciznost i odziv po razredima)

    cm = confusion_matrix(Y_out, Y_, nclasses)
    precisions, recalls = get_metric_for_classes(cm)

    print(f"Precisions: {precisions}\nRecalls:{recalls}\n")
    print(cm)

    count_params(ptlr)

    # iscrtaj rezultate, decizijsku plohu

    data.graph_surface(lambda x: np.max(eval(ptlr, torch.tensor(x)), axis=1), [[-15, -15], [15, 15]])
    data.graph_data(X, Y_, Y_out)
    plt.show()