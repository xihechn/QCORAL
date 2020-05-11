import numpy as np
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
import matplotlib.pyplot as plt

#rotation operator
def Ry(theta):
    Ry = np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]])
    return Ry

#parameterized unitary
def Unitary(x):
    U_H = np.kron(Hadamard, Hadamard)
    U1 = np.kron(Ry(x[0]), Ry(x[1]))
    U_CNOT = np.dot(CNOT2, CNOT1)
    U2 = np.kron(Ry(x[2]), Ry(x[3]))
    U3 = np.kron(Ry(x[4]), Ry(x[5]))
    U4 = np.kron(Ry(x[6]), Ry(x[7]))
    U = np.dot(U4, np.dot(U_CNOT, np.dot(U3, np.dot(U_CNOT, np.dot(U2, np.dot(U_CNOT, np.dot(U1, U_H)))))))

    return U

#cost function
def cost_function(x):
    cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
    cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
    cov_src_align = np.dot(np.dot(Unitary(x), cov_src), Unitary(x).T)
    cost = np.linalg.norm(cov_src_align - cov_tar)

    return cost

#compute gradient
def numerical_gradient(f, x):
    t = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp_val = x[i]
        x[i] = tmp_val + t 
        fxh1 = f(x)

        x[i] = tmp_val - t
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2*t)
        x[i] = tmp_val
    return grad

#zero_normalization
def zero_normalize(X):
    for i in range(np.size(X, axis=0)):
        X_mean = np.mean(X, axis=1)
        X[i, :] -= X_mean[i]
        X_norm = np.linalg.norm(X, axis=1)
        X[i, :] /= X_norm[i]
    return X


#optimization with AdaGrad
def fit(f, init_x, lr, step_num):
    x = init_x
    cost_history = []

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        f(x)
        cost_history.append(f(x))

        h = np.zeros_like(grad)

        for index in range(len(x)):
            x[index] -= lr * grad[index]
            h[index] += grad[index] * grad[index]
            x[index] -= lr * grad[index] / (np.sqrt(h[index]) + 1e-7)

    return cost_history, x

#classification
def fit_predict(final_theta, Xs, Ys, Xt, Yt):
    Xs_new = np.dot(Unitary(final_theta), Xs.T).T
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
    clf.fit(Xs_new, Ys.ravel())
    y_pred = clf.predict(Xt)
    acc = sklearn.metrics.accuracy_score(Yt, y_pred)

    return acc, y_pred

if __name__ == '__main__':
    zero_state = np.array([[1], [0]])
    one_state = np.array([[0], [1]])
    
    #Pauli operators
    PauliX = np.array([[0, 1], [1, 0]])
    PauliY = np.array([[0, -1j], [1j, 0]])
    PauliZ = np.array([[1, 0], [0, -1]])
    PauliI = np.eye(2)

    #Hadamard operator
    Hadamard = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

    #CNOT operators
    CNOT1 = np.kron(np.dot(zero_state, zero_state.T), PauliI) + np.kron(np.dot(one_state, one_state.T), PauliX)
    CNOT2 = np.kron(PauliI, np.dot(zero_state, zero_state.T)) + np.kron(PauliX, np.dot(one_state, one_state.T))

    #theta_initialization
    init_theta = np.random.uniform(low=0, high=2*np.pi, size=16)

    #learning rate
    lr = 0.01

    #step numbers
    step_num=1000

    data = ['X1.txt', 'X2.txt']
    labels = ['Y1.txt', 'Y2.txt']
    for i in range(2):
        for j in range(2):
            if i != j:
                Xs = np.loadtxt(data[i])
                Xt = np.loadtxt(data[j])
                Ys = np.loadtxt(labels[i])
                Yt = np.loadtxt(labels[j])
                cost_history, final_theta = fit(cost_function, init_theta, lr, step_num)
                acc, yrep = fit_predict(final_theta, Xs, Ys, Xt, Yt)
                print(final_theta)
                print(acc)




   