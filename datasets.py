from copy import copy

from scipy.spatial.distance import pdist, squareform
from numpy import exp
import numpy as np
from scipy.sparse import csc_matrix
import pickle as pkl
import os
import networkx as nx
import random
import utils_ as utils
import torch
from graph_manip import Graph
#from torch_geometric.data import InMemoryDataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def twomoons(n, sigma, k, noise=0.2):
    from sklearn.datasets import make_moons
    X, Y = make_moons(n_samples=n, noise=noise, shuffle=False)

    ## Make a gram matrix for Gaussian kernel with diagonal 0
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    K = exp(-sigma * (pairwise_dists ** 2))
    # fill 0s to diagonal elements
    np.fill_diagonal(K, 0)

    # Reduce from a complete graph to a knn graph
    K = make_knngraph(K, k)

    return K, X, Y


def make_knngraph_(A, k=3):
    '''
    :param A: adjacency matrix
    :param k: the number of the neighbors
    :return: an adjacency matrix which is knn graph
    '''
    from scipy.stats import rankdata
    idx = np.ceil(rankdata(-A, axis=0))
    idx[idx > k] = 0
    idx = idx + idx.T
    idx[idx > 0] = 1
    A[idx == 0] = 0

    # Sparcify a matrix
    A = csc_matrix(A)

    return A


def load_data(dname, dloc):
    """Load datasets"""
    X = None
    y = None
    graphs = None
    name = os.path.abspath(os.path.join(dloc, dname))
    with open(name + ".graph", "rb") as f:
        graph = pkl.load(f)
        #A = nx.to_scipy_sparse_matrix(graph, nodelist=None, dtype=None, weight='weight', format='csr')
        A = nx.to_scipy_sparse_array(graph, nodelist=None, dtype=None, weight='weight', format='csr')
        # A = to_scipy_sparse_array(graph, nodelist=None, dtype=None, weight='weight', format='csr')
    with open(name + ".y", "rb") as f:
        y = pkl.load(f)
    if os.path.exists(name + ".X"):
        with open(name + ".X", "rb") as f:
            X = pkl.load(f)
    return A, X, y


def load_data_krylov(dname, dloc):
    """Load datasets"""
    X = None
    y = None
    graphs = None
    name = os.path.abspath(os.path.join(dloc, dname))
    with open(name + ".LX", "rb") as f:
        graph = pkl.load(f)
        LX = graph['LX']
        # A = nx.to_scipy_sparse_array(graph, nodelist=None, dtype=None, weight='weight', format='csr')
        # A = to_scipy_sparse_array(graph, nodelist=None, dtype=None, weight='weight', format='csr')
    with open(name + ".y", "rb") as f:
        y = pkl.load(f)
    #if os.path.exists(name + ".X"):
    #    with open(name + ".X", "rb") as f:
    #        X = pkl.load(f)
    if os.path.exists(name + ".X"):
        with open(name + ".X", "rb") as f:
            X = pkl.load(f)
    return LX, X, y


def load_data_pkl(dname, dloc):
    """Load datasets"""
    X = None
    y = None
    graphs = None
    dname = dname + '_lp.pkl'
    name = os.path.abspath(os.path.join(dloc, dname))
    f =  open(name, "rb")
    pLsq, X, Y = pkl.load(f)

    return pLsq, X, Y

class GraphDataset:
    def __init__(self, data):
        """
        :param data: data must be GraphData
        :param num_classes: number of classes
        :param num_features: number of features (in this case n of vertices
        """
        self.data = data
        self.num_classes = int(max(data.y) + 1)
        self.num_features = data.x.shape[0]


class GraphData:

    def __init__(self, X, y, shuffle_ratio=0.1, k_ratio=0.1, train_mask=0.07, val_mask=0.07, test_mask=0.86,
                 is_zero_one=True, is_norm=False):

        self.is_norm = is_norm

        # constructing A
        pairwise_dists = squareform(pdist(X, 'euclidean'))
        K = np.exp(- 0.1 * pairwise_dists ** 2)
        np.fill_diagonal(K, 0)
        k = np.ceil(k_ratio * K.shape[0])
        self.A = utils.make_knngraph(K, k, is_zero_one=is_zero_one)

        G = nx.from_numpy_array(self.A)
        n_vertices = G.number_of_nodes()

        self.edge_index = self.shuffle_edge(G, shuffle_ratio)
        y = torch.Tensor(y)
        # self.y = y
        self.y = y.to(torch.long)

        # constructing masks
        indices = [i for i in range(n_vertices)]
        self.train_mask, self.test_mask, self.val_mask = self.generate_masks_v2(indices, y, train_mask=train_mask,
                                                                                val_mask=val_mask, test_mask=test_mask)

        # reconstruct a graph from shuffled edges
        G = nx.from_edgelist(self.edge_index.T.numpy()).to_undirected()
        A = nx.adjacency_matrix(G).toarray()
        self.A = torch.Tensor(A)

        if is_norm is False:
            self.x = torch.eye(n_vertices)
        else:
            self.x = torch.diag(torch.Tensor(np.power(self.A.sum(axis=0), 1 / 2)), 0)

    def shuffle_edge(self, G, shuffle_ratio=0.1):

        edge_list = []
        for line in nx.generate_edgelist(G, data=False):
            pairs = line.split(' ')
            edge_list.append([int(pairs[0]), int(pairs[1])])

        indices = [i for i in range(len(edge_list))]
        n_edges = len(indices)
        if shuffle_ratio == 0:
            n = 1
        else:
            n = int(np.ceil(shuffle_ratio * n_edges))

        np.random.shuffle(edge_list)
        shuffled_edge_list = edge_list[:-n]

        # Ensure uniform distribution of labels
        node_indices = [i for i in range(G.number_of_nodes())]
        np.random.shuffle(node_indices)
        # shuffled_edge_list = np.random.shuffle(indices)
        # shuffled_edge_list = shuffled_edge_list[:-n]

        # A relatively long list
        gen = self.pair_generator(node_indices)

        # Get n pairs:
        pairs = []
        for i in range(n):
            pair = next(gen)
            pair = list(pair)
            pairs.append(pair)

        output_edge_list = shuffled_edge_list + pairs

        from random import randint
        for i in range(G.number_of_nodes()):
            if sum(sum(np.array(output_edge_list) == i)) == 0:
                output_edge_list.append([i, randint(0, G.number_of_nodes()-1)])
                output_edge_list.append([i, randint(0, G.number_of_nodes()-1)])

        output_edge_list = torch.Tensor(output_edge_list)
        output_edge_list = torch.t(output_edge_list)
        output_edge_list = output_edge_list.to(torch.long)

        return output_edge_list

    def pair_generator(self, numbers):
        """Return an iterator of random pairs from a list of numbers."""
        # Keep track of already generated pairs
        used_pairs = set()

        while True:
            pair = random.sample(numbers, 2)
            # Avoid generating both (1, 2) and (2, 1)
            pair = tuple(sorted(pair))
            if pair not in used_pairs:
                used_pairs.add(pair)
                yield pair

    def generate_masks(self, indices, train_mask=0.8, val_mask=0.1, test_mask=0.1):
        from sklearn.model_selection import train_test_split
        indices_train, indices_test_valid = train_test_split(indices, train_size=train_mask)
        indices_test, indices_val = train_test_split(indices_test_valid, train_size=test_mask / (val_mask + test_mask))

        train_mask = torch.Tensor([False for i in range(len(indices))])
        train_mask[indices_train] = True
        train_mask = train_mask.to(torch.bool)

        test_mask = torch.Tensor([False for i in range(len(indices))])
        test_mask[indices_test] = True
        test_mask = test_mask.to(torch.bool)

        val_mask = torch.Tensor([False for i in range(len(indices))])
        val_mask[indices_val] = True
        val_mask = val_mask.to(torch.bool)

        # return indices_train, indices_test, indices_val

        return train_mask, test_mask, val_mask

    def generate_masks_v2(self, indices, y, train_mask=0.8, val_mask=0.1, test_mask=0.1):
        from sklearn.model_selection import train_test_split
        k = int(max(y) + 1)
        indices_full = copy(indices)
        indices_full = np.array(indices_full)
        indices_train = []
        indices_test = []
        indices_val = []
        for i in range(k):
            indices = [y[ii] == i for ii in range(len(y))]
            indices = indices_full[indices]

            indices_train_tmp, indices_test_valid_tmp = train_test_split(indices, train_size=train_mask)
            indices_test_tmp, indices_val_tmp = train_test_split(indices_test_valid_tmp,
                                                                 train_size=test_mask / (val_mask + test_mask))

            indices_train += list(indices_train_tmp)
            indices_test += list(indices_test_tmp)
            indices_val += list(indices_val_tmp)

        train_mask = torch.Tensor([False for i in range(len(indices_full))])
        train_mask[indices_train] = True
        train_mask = train_mask.to(torch.bool)

        test_mask = torch.Tensor([False for i in range(len(indices_full))])
        test_mask[indices_test] = True
        test_mask = test_mask.to(torch.bool)

        val_mask = torch.Tensor([False for i in range(len(indices_full))])
        val_mask[indices_val] = True
        val_mask = val_mask.to(torch.bool)

        # return indices_train, indices_test, indices_val

        return train_mask, test_mask, val_mask

    def plt_show(self):
        import matplotlib.pyplot as plt
        G = nx.from_edgelist(self.edge_index.T.numpy())
        AA = nx.adjacency_matrix(G).toarray()
        plt.imshow(AA)
        plt.colorbar()
        plt.show()

        return None

    def plaplacian(self, gamma=0.1):
        G = Graph(self.A.numpy(), gamma=gamma, norm=self.is_norm)
        self.pL = torch.Tensor(G.plaplacian(is_dense=True))

    def plaplacian_x(self, gamma=0.1):
        G = Graph(self.A.numpy(), gamma=gamma, norm=self.is_norm)
        self.x = torch.Tensor(G.plaplaciansq(is_dense=True))

def load_dataset(name='iris', path='/', k_ratio=0.1, shuffle_ratio=0.1, train_mask=0.07, val_mask=0.07, test_mask=0.86,
                 is_norm=False):
    import sklearn.datasets
    if name == 'iris':
        d = sklearn.datasets.load_iris()

    if name == 'wine':
        d = sklearn.datasets.load_wine()

    if name == 'digit':
        d = sklearn.datasets.load_digits()

    if name == 'cancer':
        d = sklearn.datasets.load_breast_cancer()

    X = d.data
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    K = (pairwise_dists ** 2)
    np.fill_diagonal(K, 0)
    k = np.ceil(k_ratio * K.shape[0])
    K = utils.make_knngraph(K, k, is_zero_one=True)
    GD = GraphData(K, d.target, k_ratio=k_ratio, shuffle_ratio=shuffle_ratio, train_mask=train_mask, val_mask=val_mask,
                   test_mask=test_mask, is_norm=is_norm)

    output_gds = GraphDataset(GD)

    return output_gds


def load_data_plain(name='iris', path='/', k_ratio=0.1, shuffle_ratio=0.1, train_mask=0.07, val_mask=0.07,
                    test_mask=0.86, is_norm=False):
    import sklearn.datasets
    if name == 'iris':
        d = sklearn.datasets.load_iris()

    if name == 'wine':
        d = sklearn.datasets.load_wine()

    if name == 'digit':
        d = sklearn.datasets.load_digits()

    if name == 'cancer':
        d = sklearn.datasets.load_breast_cancer()

    X = d.data
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    K = (pairwise_dists ** 2)
    np.fill_diagonal(K, 0)
    k = np.ceil(k_ratio * K.shape[0])
    K = utils.make_knngraph(K, k, is_zero_one=True)
    GD = GraphData(K, d.target, k_ratio=k_ratio, shuffle_ratio=shuffle_ratio, train_mask=train_mask, val_mask=val_mask,
                   test_mask=test_mask, is_norm=is_norm)

    output_gds = GraphDataset(GD)

    return output_gds


def to_scipy_sparse_array(G, nodelist=None, dtype=None, weight="weight", format="csr"):
    """
    Taken from networkx
    """
    import scipy as sp
    import scipy.sparse  # call as sp.sparse

    if len(G) == 0:
        raise nx.NetworkXError("Graph has no nodes or edges")

    if nodelist is None:
        nodelist = list(G)
        nlen = len(G)
    else:
        nlen = len(nodelist)
        if nlen == 0:
            raise nx.NetworkXError("nodelist has no nodes")
        nodeset = set(G.nbunch_iter(nodelist))
        if nlen != len(nodeset):
            for n in nodelist:
                if n not in G:
                    raise nx.NetworkXError(f"Node {n} in nodelist is not in G")
            raise nx.NetworkXError("nodelist contains duplicates.")
        if nlen < len(G):
            G = G.subgraph(nodelist)

    index = dict(zip(nodelist, range(nlen)))
    coefficients = zip(
        *((index[u], index[v], wt) for u, v, wt in G.edges(data=weight, default=1))
    )
    try:
        row, col, data = coefficients
    except ValueError:
        # there is no edge in the subgraph
        row, col, data = [], [], []

    if G.is_directed():
        A = sp.sparse.coo_array((data, (row, col)), shape=(nlen, nlen), dtype=dtype)
    else:
        # symmetrize matrix
        d = data + data
        r = row + col
        c = col + row
        # selfloop entries get double counted when symmetrizing
        # so we subtract the data on the diagonal
        selfloops = list(nx.selfloop_edges(G, data=weight, default=1))
        if selfloops:
            diag_index, diag_data = zip(*((index[u], -wt) for u, v, wt in selfloops))
            d += diag_data
            r += diag_index
            c += diag_index
        A = sp.sparse.coo_array((d, (r, c)), shape=(nlen, nlen), dtype=dtype)
    try:
        return A.asformat(format)
    except ValueError as err:
        raise nx.NetworkXError(f"Unknown sparse matrix format: {format}") from err


if __name__ == '__main__':
    # K, X, Y = twomoons(100, 0.1, 10)

    iris_gds = load_dataset('iris')
