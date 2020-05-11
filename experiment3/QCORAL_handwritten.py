import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors

#rotation operator
def Ry(theta):
    Ry = np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]])
    return Ry

#parameterized unitary
def Unitary(x):
    U_H = np.kron(Hadamard, np.kron(Hadamard, np.kron(Hadamard, np.kron(Hadamard, np.kron(Hadamard, np.kron(Hadamard, np.kron(Hadamard, Hadamard)))))))
    U1 = np.kron(Ry(x[0]), np.kron(Ry(x[1]), np.kron(Ry(x[2]), np.kron(Ry(x[3]), np.kron(Ry(x[4]), np.kron(Ry(x[5]), np.kron(Ry(x[6]), Ry(x[7]))))))))
    U_CNOT1 = np.kron(PauliX, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, PauliI)))))))
    U_CNOT2 = np.kron(PauliI, np.kron(PauliX, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, PauliI)))))))
    U_CNOT3 = np.kron(PauliI, np.kron(PauliI, np.kron(PauliX, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, PauliI)))))))
    U_CNOT4 = np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliX, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, PauliI)))))))
    U_CNOT5 = np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliX, np.kron(PauliI, np.kron(PauliI, PauliI)))))))
    U_CNOT6 = np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliX, np.kron(PauliI, PauliI)))))))
    U_CNOT7 = np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliX, PauliI)))))))
    U_CNOT8 = np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, np.kron(PauliI, PauliX)))))))
    U_CNOT = np.dot(U_CNOT8, np.dot(U_CNOT7, np.dot(U_CNOT6, np.dot(U_CNOT5, np.dot(U_CNOT4, np.dot(U_CNOT3, np.dot(U_CNOT2, U_CNOT1)))))))
    U2 = np.kron(Ry(x[8]), np.kron(Ry(x[9]), np.kron(Ry(x[10]), np.kron(Ry(x[11]), np.kron(Ry(x[12]), np.kron(Ry(x[13]), np.kron(Ry(x[14]), Ry(x[15]))))))))
    U3 = np.kron(Ry(x[16]), np.kron(Ry(x[17]), np.kron(Ry(x[18]), np.kron(Ry(x[19]), np.kron(Ry(x[20]), np.kron(Ry(x[21]), np.kron(Ry(x[22]), Ry(x[23]))))))))
    U4 = np.kron(Ry(x[24]), np.kron(Ry(x[25]), np.kron(Ry(x[26]), np.kron(Ry(x[27]), np.kron(Ry(x[28]), np.kron(Ry(x[29]), np.kron(Ry(x[30]), Ry(x[31]))))))))
    U5 = np.kron(Ry(x[32]), np.kron(Ry(x[33]), np.kron(Ry(x[34]), np.kron(Ry(x[35]), np.kron(Ry(x[36]), np.kron(Ry(x[37]), np.kron(Ry(x[38]), Ry(x[39]))))))))
    U6 = np.kron(Ry(x[40]), np.kron(Ry(x[41]), np.kron(Ry(x[42]), np.kron(Ry(x[43]), np.kron(Ry(x[44]), np.kron(Ry(x[45]), np.kron(Ry(x[46]), Ry(x[47]))))))))
    U7 = np.kron(Ry(x[48]), np.kron(Ry(x[49]), np.kron(Ry(x[50]), np.kron(Ry(x[51]), np.kron(Ry(x[52]), np.kron(Ry(x[53]), np.kron(Ry(x[54]), Ry(x[55]))))))))
    U8 = np.kron(Ry(x[56]), np.kron(Ry(x[57]), np.kron(Ry(x[58]), np.kron(Ry(x[59]), np.kron(Ry(x[60]), np.kron(Ry(x[61]), np.kron(Ry(x[62]), Ry(x[63]))))))))
    U = np.dot(U8, np.dot(U_CNOT, np.dot(U7, np.dot(U_CNOT, np.dot(U6, np.dot(U_CNOT, np.dot(U5, np.dot(U_CNOT, np.dot(U4, np.dot(U_CNOT, np.dot(U3, np.dot(U_CNOT, np.dot(U2, np.dot(U_CNOT, np.dot(U1, U_H)))))))))))))))

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
        print(f(x))

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
    init_theta = np.random.uniform(low=0, high=2*np.pi, size=80)

    #learning rate
    lr = 0.001

    #step numbers
    step_num=1000

    domains = ['MNIST_vs_USPS.mat', 'USPS_vs_MNIST.mat']
    for i in range(2):
        for j in range(2):
            if i != j:
                src, tar = domains[i], domains[j]
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['X_src'].T, src_domain['Y_src'], tar_domain['X_src'].T, tar_domain['Y_src']
                data1 = zero_normalize(Xs)
                data2 = zero_normalize(Xt)
                Xs, Xt= data1, data2
                cost_history, final_theta = fit(cost_function, init_theta, lr, step_num)
                acc, yrep = fit_predict(final_theta, Xs, Ys, Xt, Yt)
                print(final_theta)
                print(acc)