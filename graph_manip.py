import networkx as nx
import scipy as sp
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

'''
 The following block is in order to erase warining message for incidence matrix from networkx.
 Other message should not be ignored.
'''
import warnings

warnings.filterwarnings('ignore')


class Graph:

    def __init__(self, A, gamma=0, b=3, norm=False):
        self.norm = norm
        self.pL = None
        self.n = A.shape[0]
        self.b = b/self.n
        self.A = A + gamma * sp.sparse.eye(self.n)
        self.D = sp.sparse.spdiags(self.A.sum(axis=0), 0, A.shape[0], A.shape[1])
        self.Dsq, self.L = self.laplacian(self.norm, gamma)

    def laplacian(self, norm, gamma):
        if norm:
            Dsq = sp.sparse.spdiags(np.power(self.A.sum(axis=0), -1 / 2), 0, self.A.shape[0], self.A.shape[1])
            L = Dsq @ (self.D - self.A) @ Dsq
            return Dsq, L

        else:
            L = self.D - self.A
            return None, L

    def plaplacian(self, is_dense=False):
        if is_dense is False:
            pL = sp.linalg.pinv(self.L.toarray())
        else:
            pL = sp.linalg.pinv(self.L)
        return pL

    def plaplaciansq(self, is_dense=False):
        if is_dense is False:
            u, s, vh = sp.linalg.svd(self.L.toarray(), full_matrices=False, check_finite=False, lapack_driver='gesvd')
        else:
            u, s, vh = sp.linalg.svd(self.L, full_matrices=False, check_finite=False, lapack_driver='gesvd')

        t = u.dtype.char.lower()
        maxS = np.max(s)

        atol = 0.
        rtol = max(self.L.shape) * np.finfo(t).eps

        val = atol + maxS * rtol
        rank = np.sum(s > val)

        u = u[:, :rank]
        u /= np.power(s[:rank], 1 / 2)
        pL = (u @ vh[:rank]).conj().T
        if self.norm is False:
            #pL = pL + (self.b ** 1 / 2) * sp.ones([self.n, self.n])
            pL = pL + (self.b) * sp.ones([self.n, self.n])
        else:
            pL = pL + (self.b ** 1 / 2) * self.Dsq @ np.ones([self.n, self.n]) @ self.Dsq

        return pL

    def plaplaciansq_nonsp(self):
        u, s, vh = sp.linalg.svd(self.L, full_matrices=False, check_finite=False)
        t = u.dtype.char.lower()
        maxS = np.max(s)

        atol = 0.
        rtol = max(self.L.shape) * np.finfo(t).eps

        val = atol + maxS * rtol
        rank = np.sum(s > val)

        u = u[:, :rank]
        u /= np.power(s[:rank], 1 / 2)
        pL = (u @ vh[:rank]).conj().T
        if self.norm is False:
            #pL = pL + (self.b ** 1 / 2) * sp.ones([self.n, self.n])
            pL = pL + (self.b) * sp.ones([self.n, self.n])
        else:
            pL = pL + (self.b ** 1 / 2) * self.Dsq @ np.ones([self.n, self.n]) @ self.Dsq

        return pL

    def approx_plaplacian(self, epsilon=0.2, delta=0.2):
        G = nx.from_scipy_sparse_array(self.A)

        # incidence matrix
        C = nx.incidence_matrix(G).T

        ## weight matrix
        # weight vector
        w = np.array([edge[2]['weight'] for edge in list(G.edges.data())])
        # weight matrix
        W = sp.sparse.spdiags(w, 0, w.shape[0], w.shape[0])

        ## approx Lp
        # make JL lemma random matrix
        d = 8 * np.log(self.n ** 2 / delta) / (epsilon ** 2)
        R = np.random.normal(0, 1 / d, size=(int(np.ceil(d)), C.shape[0]))

        # make random matrix
        if self.norm:
            P = R @ W @ C @ self.Dsq
        else:
            P = W @ C
            P = R @ P

        pL = sp.sparse.linalg.spsolve(self.L, P.T)

        return pL

    def approx_plaplacian_taylor(self, k=1):
        d = self.A.sum(axis=0)
        c = min(2*d.max(), self.n)
        print("c:{:.4f}".format(c))
        H = sp.sparse.eye(self.n) - self.L / c

        pL = 1/np.sqrt(c) * (sp.sparse.eye(self.n) + 1/2 * H)

        if k >= 2:
            pL += 1/np.sqrt(c) * 3/8 * H @ H

        if k >= 3:
            pL += 1/np.sqrt(c) * 5/16 * H @ H @ H

        '''
        if k >= 3: 
            to be implemented
        '''

        pL += (self.b) * sp.ones([self.n, self.n])

        return pL

    '''
    ## JL lemma 
    def approx_plaplacian_sq_jl(self, epsilon=0.2, delta=0.2):
        G = nx.from_scipy_sparse_array(self.A)

        # incidence matrix
        C = nx.incidence_matrix(G).T

        ## weight matrix
        # weight vector
        w = np.array([edge[2]['weight'] ** 1/2 for edge in list(G.edges.data())])
        # weight matrix
        W = sp.sparse.spdiags(w, 0, w.shape[0], w.shape[0])

        ## approx Lp
        # make JL lemma random matrix
        d = 8 * np.log(self.n ** 2 / delta) / (epsilon ** 2)
        R = np.random.normal(0, 1 / d, size=(int(np.ceil(d)), C.shape[0]))

        # make random matrix
        if self.norm:
            P = R @ W @ C @ self.Dsq
        else:
            P = W @ C
            P = R @ P

        pL = sp.sparse.linalg.spsolve(self.L, P.T)

        return pL.T
        '''


if __name__ == "__main__":
    pass
