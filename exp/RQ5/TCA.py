import numpy as np
from scipy.linalg import eig
from sklearn.metrics import pairwise

import warnings
warnings.filterwarnings('ignore')


def kernel(ker, x1, x2, gamma):
    k = None
    if not ker or ker == 'primal':
        k = x1
    elif ker == 'linear':
        if x2 is not None:
            k = pairwise.linear_kernel(
                np.asarray(x1).T, np.asarray(x2).T)
        else:
            k = pairwise.linear_kernel(np.asarray(x1).T)
    elif ker == 'rbf':
        if x2 is not None:
            k = pairwise.rbf_kernel(
                np.asarray(x1).T, np.asarray(x2).T, gamma)
        else:
            k = pairwise.rbf_kernel(
                np.asarray(x1).T, None, gamma)
    return k


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        """
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        """
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, xs, xt):
        """
        Transform Xs and Xt
        :param xs: ns * n_feature, source feature
        :param xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        """
        x = np.hstack((xs.T, xt.T))
        x /= np.linalg.norm(x, axis=0)
        m, n = x.shape
        ns, nt = len(xs), len(xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, x, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T
        w, V = eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new
