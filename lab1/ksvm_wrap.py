from sklearn import svm
import data
import matplotlib.pyplot as plt
import numpy as np

nclasses = 2
nsamples = 10


class KSVMWrap():
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.X = X
        self.Y_ = Y_
        self.svm = svm.SVC(C=param_svm_c, gamma=param_svm_gamma)
        self.svm.fit(X, Y_)

    def predict(self, X):
        return self.svm.predict(X)

    def get_scores(self, X):
        return self.svm.decision_function(X)

    def support(self):
        return self.svm.support_


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(6, nclasses, nsamples)

    wrap = KSVMWrap(X, Y_)
    Y_out = wrap.predict(X)

    cm = data.confusion_matrix(Y_out, Y_, nclasses)
    precisions, recalls = data.get_metric_for_classes(cm)

    print(f"Precisions: {precisions}\nRecalls:{recalls}\n")
    print("Confusion matrix")
    print(cm)

    data.graph_surface(wrap.get_scores, [[-2, -2], [10, 10]], offset=0.0)
    data.graph_data(X, Y_, Y_out, special=wrap.support())
    plt.show()
