import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors

class CORAL:
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        Xs_new = np.dot(Xs, A_coral)
        return Xs_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        Xs_new = self.fit(Xs, Xt)
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred

#zero_normalization
def zero_normalize(X):
    for i in range(np.size(X, axis=0)):
        X_mean = np.mean(X, axis=1)
        X[i, :] -= X_mean[i]
        X_norm = np.linalg.norm(X, axis=1)
        X[i, :] /= X_norm[i]
    return X

if __name__ == '__main__':
    domains = ['MNIST_vs_USPS.mat', 'USPS_vs_MNIST.mat']
    for i in range(2):
        for j in range(2):
            if i != j:
                src, tar = domains[i], domains[j]
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['X_src'].T, src_domain['Y_src'], tar_domain['X_src'].T, tar_domain['Y_src']
                coral = CORAL()
                acc, ypre = coral.fit_predict(Xs, Ys, Xt, Yt)
                print(acc)

    