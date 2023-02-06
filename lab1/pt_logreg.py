import torch
import torch.nn as nn
import numpy as np
import data
import matplotlib.pyplot as plt

nclasses = 3
nsamples = 25
param_niter = 10000
param_lr = 0.05
param_lambda = 0.01


class PTLogreg(nn.Module):
    def __init__(self, input_dimension, nclasses):
        super().__init__()
        self.W = torch.nn.Parameter(torch.tensor(np.random.randn(input_dimension, nclasses), requires_grad=True))
        self.b = torch.nn.Parameter(torch.tensor(np.random.randn(nclasses), requires_grad=True))

    def forward(self, X):
        scores = torch.mm(X, self.W) + self.b # b se zbraja po retcima
        return torch.softmax(scores, dim=1)  # dim -> softmax po retcima

    def get_loss(self, X, Yoh_):
        output = self.forward(X)
        softmax_y = torch.log(output) * Yoh_
        return -torch.mean(softmax_y)


def train(model, X, Yoh_, param_niter, param_delta, param_lambda=0.0):
    # inicijalizacija optimizatora
    optimizer = torch.optim.SGD([model.W, model.b], lr=param_delta)

    # petlja učenja
    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_) + param_lambda * torch.linalg.norm(model.W)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Iteration {i + 1}: {loss}")


def eval(model, X):
    output = model.forward(X)
    return output.detach().numpy()


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = data.sample_gauss_2d(nclasses, nsamples)
    Yoh_ = data.one_hot_encode(Y_, nclasses)
    X = torch.tensor(X)
    Yoh_ = torch.tensor(Yoh_)

    # definiraj model:
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])
    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, X, Yoh_, param_niter, param_lr, param_lambda)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)

    Y_out = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    cm = data.confusion_matrix(Y_out, Y_, nclasses)
    precisions, recalls = data.get_metric_for_classes(cm)
    print(f"Precisions: {precisions}\nRecalls:{recalls}\n")
    print("Confusion matrix:")
    print(cm)
    # iscrtaj rezultate, decizijsku plohu
    data.graph_surface(lambda x: np.max(eval(ptlr, torch.tensor(x)), axis=1), [[-15, -15], [15, 15]])
    data.graph_data(X, Y_, Y_out)
    plt.show()
