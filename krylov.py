import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import datasets
from graph_manip import Graph
import pickle as pkl


k_dim = 10 # krylov

dataset_name = 'pubmed'
dataset_path = 'examples/codes/data'


gamma = 0
bb = 2
norm_flag = False

eps = 1e-7

def arnoldi_iteration(A, b, n: int):
    """Compute a basis of the (n + 1)-Krylov subspace of the matrix A.

    This is the space spanned by the vectors {b, Ab, ..., A^n b}.

    Parameters
    ----------
    A : array_like
        An m × m array.
    b : array_like
        Initial vector (length m).
    n : int
        One less than the dimension of the Krylov subspace, or equivalently the *degree* of the Krylov space. Must be >= 1.

    Returns
    -------
    Q : numpy.array
        An m x (n + 1) array, where the columns are an orthonormal basis of the Krylov subspace.
    h : numpy.array
        An (n + 1) x n array. A on basis Q. It is upper Hessenberg.
    """
    h = np.zeros((n + 1, n))
    q = np.zeros((A.shape[0], n + 1))
    # Normalize the input vector
    q[:, 0] = b / np.linalg.norm(b, 2)  # Use it as the first Krylov vector
    for k in range(1, n + 1):
        v = A @ q[:, k - 1]  # Generate a new candidate vector
        for j in range(k):  # Subtract the projections on previous vectors
            h[j, k - 1] = np.dot(q[:, j].T, v)
            v = v - h[j, k - 1] * q[:, j]
        h[k, k - 1] = np.linalg.norm(v, 2)
        if h[k, k - 1] > eps:  # Add the produced vector to the list, unless
            q[:, k] = v / h[k, k - 1]
        else:
            print('happening')
            return h[:-1, :], q[:, :-1]

    return h[:-1, :], q[:, :-1]


def arnoldi_method(a, x0, krylov_dim):
    n_dim = a.shape[0]
    h = np.zeros((krylov_dim, krylov_dim))
    q = np.zeros((n_dim, krylov_dim))

    q0 = x0 / np.linalg.norm(x0)
    q[:, 0] = q0

    for k in range(krylov_dim):
        w = a @ q[:, k]
        for j in range(k + 1):
            q_j = q[:, j]
            h[j, k] = q_j.T @ w
            w = w - h[j, k] * q_j

        if k + 1 < krylov_dim:
            h[k + 1, k] = np.linalg.norm(w)
            if abs(h[k + 1, k]) > eps:  # Add the produced vector to the list, unless
                q[:, k + 1] = w / h[k + 1, k]
            else:  # If that happens, stop iterating.
                print('happening')
                return h, q, 0

    return h, q, 1


def plus_ones(graph, LX, X, b=1):
    G = nx.Graph(graph)
    connected_components = [np.array(list(c)) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    b = b/X.shape[0]
    for c in connected_components:
        ones_for_c_val = sum(X[c, :])
        ones_for_c_mat = np.ones((len(c), X.shape[1]))
        LX[c, :] += b * ones_for_c_val * ones_for_c_mat

    return LX


if __name__ == '__main__':
    pkl_file_name = dataset_path + '/' + dataset_name + '_krylov_' + str(k_dim) + '_b_' + str(bb) + '.LX'
    K, X, Y = datasets.load_data(dataset_name, dataset_path)
    G = Graph(K, gamma=gamma, b=bb, norm=norm_flag)
    X_dim = X.shape[1]

    L = G.L + sp.sparse.eye(K.shape[0])

    # approximated L^(-1/2)X
    LX = np.zeros([X.shape[0], X.shape[1]])
    for i in range(X_dim):

        ## progress check
        if i % 10 == 0:
            print('krylov processed:'+ str(i) + '/' + str(X_dim))

        ## if X is all zero, we do not process krylov
        if np.sum(abs(X[:, i])) < eps:
            LX[:, i] = 0
            continue

        if np.sum(abs(G.L*X[:, i])) < eps:
            LX[:, i] = 0
            continue

        # H, Q = arnoldi_method(G.L, X[:, i], k_dim)
        # H, Q = arnoldi_iteration(G.L, X[:, i], k_dim)
        H, Q, is_ok = arnoldi_method(L, X[:, i], k_dim)
        if np.isnan(H).any():
            print('nan')
        if np.isinf(H).any():
            print('inf')
        if H.shape[0] != H.shape[1]:
            print('shape')
        if is_ok != 1:
            print('not ok')

        H_inv_sq = (sp.linalg.sqrtm(sp.linalg.pinv(H))).real
        if np.isnan(H_inv_sq).any():
            print('nan')
            try:
                H_inv_sq = sp.linalg.pinv(sp.linalg.sqrtm(H).real)
            except:
                continue
            if np.isnan(H_inv_sq).any():
                print('nan')
                continue

        LX[:, i] = Q @ H_inv_sq @ Q.T @ X[:, i]

    LX = plus_ones(K, LX, X, 1)

    LX = {'LX': LX, 'k_dim': k_dim}

    with open(pkl_file_name, 'wb') as f:
        pkl.dump(LX, f)



