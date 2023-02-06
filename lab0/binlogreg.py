import data
import numpy as np

param_niter = 1000
param_delta = 0.01

def binlogreg_train(X,Y_):
    '''
    Argumenti
      X:  podatci, np.array NxD
      Y_: indeksi razreda, np.array Nx1

    Povratne vrijednosti
      w, b: parametri logističke regresije
    '''
    w = np.random.randn(X.shape[1])
    b = 0

    for i in range(param_niter):
        # klasifikacijske mjere
        scores = np.dot(X, w) + b  # N x 1
        # vjerojatnosti razreda c_1
        probs = 1 / (1 + np.exp(-scores))   # N x 1

        # gubitak
        loss =  -1 / X.shape[0] * np.sum(np.log(probs))  # scalar

        # dijagnostički ispis
        if i % 10 == 0:
          print("iteration {}: loss {}".format(i, loss))

        # derivacije gubitka po klasifikacijskim mjerama
        dL_dscores = probs - Y_  # N x 1

        # gradijenti parametara
        grad_w = 1 / X.shape[0] * np.transpose(np.dot(np.transpose(dL_dscores), X))  # D x 1
        grad_b = 1 / X.shape[0] * np.sum(dL_dscores) # 1 x 1

        # poboljšani parametri
        w += -param_delta * grad_w
        b += -param_delta * grad_b
    return w, b

def binlogreg_classify(X, w, b):
    scores = np.dot(X, w) + b
    return 1 / (1 + np.exp(-scores))

if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset
    X,Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    w,b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = probs > 0.5
    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print (accuracy, recall, precision, AP)