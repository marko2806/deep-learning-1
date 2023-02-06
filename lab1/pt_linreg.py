import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara

X = torch.tensor([1, 2])
Y = torch.tensor([3, 5])

a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=0.01)

for i in range(100):
    # afin regresijski model
    Y_ = a*X + b
    diff = (Y-Y_)
    # kvadratni gubitak
    # dodan faktor 1/N tako da loss ne ovisi o broju podataka
    loss = (1 / Y_.shape[0]) * torch.sum(diff**2)

    # računanje gradijenata
    loss.backward()

    # korak optimizacije
    optimizer.step()

    # -2 * X^T * (y_t - y)
    grad_a = (-2 / Y_.shape[0]) * np.transpose(X.detach().numpy()) @ diff.detach().numpy()
    # -2 * (y_t - y)
    grad_b = (-2 / Y_.shape[0]) * torch.sum(diff)

    print(f"Gradient a: {grad_a}, Gradient b: {grad_b}")
    print(f"Gradient a: {a.grad}, Gradient b: {b.grad}")

    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()

    print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')